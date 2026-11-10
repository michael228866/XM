"""Demo runtime safety and append-only provenance. No MT5 import or connection."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TradingSafetyError(RuntimeError):
    pass


def require_demo_account(mt5, expected_login, expected_server):
    """Validate explicit identity every time; never learn an expected account from MT5."""
    if not str(expected_login or "").isdigit() or not expected_server:
        raise TradingSafetyError("Set XM_EXPECTED_LOGIN and XM_EXPECTED_SERVER explicitly")
    account = mt5.account_info()
    if account is None:
        raise TradingSafetyError("account_info failed")
    if (str(account.login) != str(expected_login)
            or account.server != expected_server
            or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO):
        raise TradingSafetyError("Expected Demo account identity does not match")
    return account


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return [json_value(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite provenance value")
    return value


def canonical_bytes(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RuntimeRun:
    """A new directory per process; old CSVs and run manifests are never migrated."""

    def __init__(self, root, configuration, model, source, account):
        root = Path(root)
        config = json_value(configuration)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(source).parent,
                text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path(source).parent,
                text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
        started = datetime.now(timezone.utc)
        self.run_id = started.strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex
        self.directory = root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.identity = {
            "run_id": self.run_id, "account_login": str(account.login),
            "account_server": account.server, "currency": account.currency,
            "model_sha256": sha256_file(model),
            "configuration_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
        }
        manifest = {
            **self.identity, "schema_version": 1, "started_at": started.isoformat(),
            "git_commit": commit, "git_dirty": dirty,
            "runtime_source_sha256": sha256_file(source),
            "foundation_source_sha256": sha256_file(__file__),
            "configuration": config, "account_trade_mode": account.trade_mode,
            "risk_basis": "broker_calculated_fill_to_initial_sl_when_available",
        }
        with (self.directory / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)

    def append(self, stream, record):
        if stream not in {"signals", "orders", "deals", "events"}:
            raise ValueError("Unknown runtime stream")
        # Observation identity is separate from the historical trade's origin.
        value = {**record, "capture_run_id": self.run_id,
                 "account_login": self.identity["account_login"],
                 "account_server": self.identity["account_server"],
                 "currency": self.identity["currency"]}
        with (self.directory / (stream + ".jsonl")).open("a", encoding="utf-8") as handle:
            handle.write(canonical_bytes(value).decode("utf-8") + "\n")


def intrabar_evidence(direction, entry, tp, sl, high, low):
    """OHLC evidence only, not an executable simulator or a claimed fill."""
    if direction not in {"long", "short"}:
        raise ValueError("Unknown direction")
    take = high >= entry + tp if direction == "long" else low <= entry - tp
    stop = low <= entry - sl if direction == "long" else high >= entry + sl
    if take and stop:
        return {"observed_order": "Unknown", "research_convention": "stop_first"}
    return {"observed_order": "tp" if take else "sl" if stop else "neither",
            "research_convention": "stop_first"}
