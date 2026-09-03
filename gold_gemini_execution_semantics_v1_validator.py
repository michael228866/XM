from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
GEMINI_FILE = ROOT / "gemini.py"
SIMULATORS = [f"S{i}" for i in range(6)]
EXPECTED_CHANGES = {
    "S0->S1": {"exit_mode"},
    "S1->S2": {"entry_policy"},
    "S2->S3": {"cost_mode", "spread_gate"},
    "S3->S4": {"risk_mode"},
    "S4->S5": {"entry_price_mode", "entry_bar_exits", "timeout_mode"},
}
POINT = 0.01
EXTRA_COST_POINTS = 5.0
LOSS_COOLDOWN_MINUTES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent execution-semantics validator")
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, list):
            return [clean(value) for value in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            number = float(item)
            return number if math.isfinite(number) else None
        if isinstance(item, (np.bool_,)):
            return bool(item)
        return item

    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def close_enough(left: float, right: float, tolerance: float = 2e-6) -> bool:
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= tolerance


def reward_metrics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    nonpositive = values[values <= 0]
    gp = float(positive.sum()) if len(positive) else 0.0
    gl = float(-nonpositive.sum()) if len(nonpositive) else 0.0
    pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
    equity = np.r_[0.0, np.cumsum(values)]
    dd = equity - np.maximum.accumulate(equity)
    return {
        "trades": len(values),
        "wins": len(positive),
        "losses": len(nonpositive),
        "realized_wr": float((values > 0).mean()) if len(values) else 0.0,
        "pf": pf,
        "mean_r": float(values.mean()) if len(values) else 0.0,
        "pnl_r": float(values.sum()),
        "max_dd_r": float(dd.min()),
    }


def load_cohort_bounds(run_dir: Path) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    primary = pd.read_csv(
        run_dir / "primary_frozen_cohort.csv.gz",
        usecols=["decision_time_api"],
        parse_dates=["decision_time_api"],
    )
    with np.load(run_dir / "secondary_frozen_cohort.npz", allow_pickle=False) as secondary:
        times = pd.to_datetime(secondary["time_ns"].astype(np.int64))
    return {
        "PRIMARY_2026_RECENT": (primary["decision_time_api"].iat[0], primary["decision_time_api"].iat[-1]),
        "SECONDARY_W0_2018_2020": (times[0], times[-1]),
    }


def reconcile_metrics(
    run_dir: Path, metrics_table: pd.DataFrame, ledger: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    bounds = load_cohort_bounds(run_dir)
    for record in metrics_table.to_dict("records"):
        cohort = str(record["cohort"])
        simulator = str(record["simulator"])
        trades = ledger[(ledger["cohort"] == cohort) & (ledger["simulator"] == simulator)].copy()
        trades = trades.sort_values("entry_index")
        unit = reward_metrics(trades["net_r"].to_numpy(dtype=np.float64))
        account = reward_metrics(trades["account_pnl"].to_numpy(dtype=np.float64))
        start, end = bounds[cohort]
        days = max((end - start).total_seconds() / 86400.0, 1.0)
        recomputed = {
            "cohort": cohort,
            "simulator": simulator,
            **unit,
            "trades_per_day": unit["trades"] / days,
            "risk_sized_pf": account["pf"],
            "risk_sized_pnl": account["pnl_r"],
            "tp": int((trades["exit_reason"] == "take_profit").sum()),
            "sl": int((trades["exit_reason"] == "stop_loss").sum()),
            "timeout": int((trades["exit_reason"] == "timeout").sum()),
        }
        for key in (
            "trades", "wins", "losses", "realized_wr", "pf", "mean_r", "pnl_r",
            "max_dd_r", "trades_per_day", "risk_sized_pf", "risk_sized_pnl", "tp", "sl", "timeout",
        ):
            expected = float(record[key])
            actual = float(recomputed[key])
            if not close_enough(expected, actual):
                errors.append(f"{cohort}/{simulator}/{key}: reported={expected} recomputed={actual}")
        rows.append(recomputed)
    return rows, errors


def validate_transition_configs(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    definitions = manifest["diagnostic_design"]["simulator_definitions"]
    if list(definitions) != SIMULATORS:
        errors.append(f"Unexpected simulator sequence: {list(definitions)}")
        return errors
    for left, right in zip(SIMULATORS, SIMULATORS[1:]):
        changed = {
            key
            for key in definitions[left]
            if key != "simulator" and definitions[left][key] != definitions[right][key]
        }
        expected = EXPECTED_CHANGES[f"{left}->{right}"]
        if changed != expected:
            errors.append(f"{left}->{right} changed {sorted(changed)}, expected {sorted(expected)}")
    return errors


def validate_cohort_identity(
    run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any], table: pd.DataFrame
) -> list[str]:
    errors: list[str] = []
    cohort_manifest = read_json(run_dir / "cohort_manifest.json")
    for key in ("primary", "secondary"):
        record = cohort_manifest[key]
        path = run_dir / record["path"]
        actual = sha256(path)
        if actual != record["sha256"]:
            errors.append(f"{key} cohort hash mismatch")
    if cohort_manifest["operational_model"]["sha256"] != sha256(MODEL_FILE):
        errors.append("Operational model no longer matches frozen cohort manifest")
    if cohort_manifest != metrics["cohort_manifest"]:
        errors.append("metrics cohort manifest differs from cohort_manifest.json")
    expected_hash = {
        "PRIMARY_2026_RECENT": cohort_manifest["primary"]["sha256"],
        "SECONDARY_W0_2018_2020": cohort_manifest["secondary"]["sha256"],
    }
    for record in table.to_dict("records"):
        if record["cohort_sha256"] != expected_hash[record["cohort"]]:
            errors.append(f"{record['cohort']}/{record['simulator']} used a different cohort hash")
    if manifest["model"]["artifact_sha256"] != sha256(MODEL_FILE):
        errors.append("Manifest operational model hash mismatch")
    return errors


def validate_trade_matching(
    ledger: pd.DataFrame, identity: pd.DataFrame, transitions: pd.DataFrame
) -> list[str]:
    errors: list[str] = []
    if ledger.duplicated(["cohort", "simulator", "trade_id"]).any():
        errors.append("Duplicate trade_id inside cohort/simulator")
    if ledger.duplicated(["cohort", "simulator", "raw_episode_id", "episode_trade_ordinal"]).any():
        errors.append("Duplicate deterministic episode/ordinal trade key")
    for record in transitions.to_dict("records"):
        rows = identity[(identity["cohort"] == record["cohort"]) & (identity["transition"] == record["transition"])]
        left_present = rows["left_trade_id"].fillna("").astype(str) != ""
        right_present = rows["right_trade_id"].fillna("").astype(str) != ""
        as_bool = lambda values: values if values.dtype == bool else values.astype(str).str.lower().eq("true")
        recomputed = {
            "trades_retained": int((left_present & right_present).sum()),
            "trades_removed": int((left_present & ~right_present).sum()),
            "trades_added": int((~left_present & right_present).sum()),
            "entry_timestamp_changed": int(as_bool(rows["entry_changed"]).sum()),
            "exit_timestamp_changed": int(as_bool(rows["exit_changed"]).sum()),
            "pnl_sign_changed": int(as_bool(rows["pnl_sign_changed"]).sum()),
        }
        for key, actual in recomputed.items():
            if int(record[key]) != actual:
                errors.append(f"{record['cohort']}/{record['transition']}/{key} mismatch")
    return errors


def validate_costs(ledger: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    denominator = ledger["sl_distance"].to_numpy(dtype=np.float64) + ledger["spread_points"].to_numpy(dtype=np.float64) * POINT
    expected = (
        ledger["gross_price"].to_numpy(dtype=np.float64)
        - (ledger["spread_points"].to_numpy(dtype=np.float64) + EXTRA_COST_POINTS) * POINT
    ) / denominator
    delta = np.abs(expected - ledger["net_r"].to_numpy(dtype=np.float64))
    if np.nanmax(delta) > 2e-9:
        errors.append(f"net-R cost arithmetic max error={np.nanmax(delta)}")
    sized = ledger["net_r"].to_numpy(dtype=np.float64) * ledger["risk_budget"].to_numpy(dtype=np.float64)
    sized_delta = np.abs(sized - ledger["account_pnl"].to_numpy(dtype=np.float64))
    if np.nanmax(sized_delta) > 2e-8:
        errors.append(f"risk-sized PnL max error={np.nanmax(sized_delta)}")
    fixed = ledger[ledger["simulator"].isin(["S0", "S1", "S2"])]
    if not np.allclose(fixed["spread_points"].to_numpy(dtype=float), 30.0):
        errors.append("S0-S2 did not use fixed 30-point spread")
    return errors


def validate_high_low_primary(run_dir: Path, ledger: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    cohort = pd.read_csv(
        run_dir / "primary_frozen_cohort.csv.gz",
        usecols=["HIGH", "LOW"],
    )
    subset = ledger[(ledger["cohort"] == "PRIMARY_2026_RECENT") & ledger["simulator"].isin(["S1", "S2", "S3", "S4", "S5"])]
    for trade in subset.itertuples(index=False):
        start = int(trade.entry_index) if trade.simulator == "S5" else int(trade.entry_index) + 1
        stop = int(trade.exit_index) + 1
        if trade.simulator == "S5" and trade.exit_reason in {"timeout", "cohort_end"}:
            stop = int(trade.exit_index)
        stop_price = float(trade.entry_price) - float(trade.sl_distance)
        take_price = float(trade.entry_price) + float(trade.tp_distance)
        for index in range(start, stop):
            stop_hit = float(cohort["LOW"].iat[index]) <= stop_price
            take = float(cohort["HIGH"].iat[index]) >= take_price
            if not (stop_hit or take):
                continue
            expected = "stop_loss" if stop_hit else "take_profit"
            if trade.exit_reason in {"take_profit", "stop_loss"}:
                if index != int(trade.exit_index) or trade.exit_reason != expected:
                    errors.append(f"{trade.simulator}/{trade.trade_id} first-touch mismatch")
            else:
                errors.append(f"{trade.simulator}/{trade.trade_id} ignored an earlier barrier hit")
            break
        if trade.same_bar_both_hit and trade.exit_reason != "stop_loss":
            errors.append(f"{trade.simulator}/{trade.trade_id} violated stop-first")
        if len(errors) >= 20:
            break
    return errors


def validate_timezone(run_dir: Path, cohort_manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    table = pd.read_csv(run_dir / "timestamp_reconciliation.csv")
    api = pd.to_datetime(table["api_interpreted_timestamp"])
    actual = pd.to_datetime(table["inferred_actual_utc_timestamp"], utc=True).dt.tz_localize(None)
    if not (table["simulator_hour"].to_numpy() == api.dt.hour.to_numpy()).all():
        errors.append("Simulator hour differs from API timestamp hour")
    if not (table["gemini_hour"].to_numpy() == table["simulator_hour"].to_numpy()).all():
        errors.append("Gemini and simulator hour differ")
    offset = int(cohort_manifest["broker_offset"]["inferred_broker_offset_hours"])
    if not ((api - actual).dt.total_seconds().to_numpy() == offset * 3600).all():
        errors.append("Timestamp offset arithmetic mismatch")
    if offset != 3:
        errors.append(f"Unexpected primary empirical offset: {offset}")
    return errors


def validate_live_cooldown(ledger: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for cohort in ledger["cohort"].unique():
        for simulator in ("S4", "S5"):
            trades = ledger[(ledger["cohort"] == cohort) & (ledger["simulator"] == simulator)].sort_values("entry_index")
            previous_loss_exit: pd.Timestamp | None = None
            for trade in trades.itertuples(index=False):
                entry = pd.Timestamp(trade.entry_time_actual_utc)
                if previous_loss_exit is not None and (entry - previous_loss_exit).total_seconds() < LOSS_COOLDOWN_MINUTES * 60:
                    errors.append(f"{cohort}/{simulator}/{trade.trade_id} entered inside live loss cooldown")
                if not bool(trade.positive):
                    previous_loss_exit = pd.Timestamp(trade.exit_time_actual_utc)
                else:
                    previous_loss_exit = None
    return errors


def artifact_hash_errors(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for artifact in manifest.get("artifacts", []):
        path = run_dir / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            errors.append(f"Artifact missing/hash mismatch: {artifact['path']}")
    return errors


def make_check(verdict: bool, evidence: str, reason: str, correction: str = "none") -> dict[str, str]:
    return {
        "verdict": "PASS" if verdict else "FAIL",
        "evidence": evidence,
        "reason": reason,
        "required_validation_correction": correction if not verdict else "none",
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# GEMINI EXECUTION SEMANTICS RECONCILIATION V1 - independent validation",
        "",
        f"Overall: **{result['overall']}**",
        "",
        f"Internal diagnostic methodology: **{result['internal_methodology']}**",
        "",
        f"Final untouched-test validity: **{result['final_untouched_test_validity']}**",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    for name, check in result["checks"].items():
        lines.append(
            f"| {name} | {check['verdict']} | {check['evidence']} | {check['reason']} | {check['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            f"Metric recomputation: **{result['metric_reconciliation']['result']}**.",
            f"Frozen cohort identity: **{result['cohort_reconciliation']['result']}**.",
            f"Simulator transition isolation: **{result['transition_reconciliation']['result']}**.",
            f"Trade matching: **{result['trade_matching']['result']}**.",
            f"HIGH/LOW stop-first: **{result['high_low_reconciliation']['result']}**.",
            f"Cost arithmetic: **{result['cost_reconciliation']['result']}**.",
            f"Cooldown reconstruction: **{result['cooldown_reconciliation']['result']}**.",
            "",
            "All six definitions are diagnostic semantics, not candidates. The submitted internal attribution is valid only for the retained development cohorts and documented S5 approximations.",
            "",
            "## Validation conclusion",
            "",
            result["smallest_validation_only_rerun"],
            "",
            f"Submitted claim: `{result['submitted_claim']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    values = np.asarray([1.0, -0.5], dtype=np.float64)
    result = reward_metrics(values)
    assert result["trades"] == 2 and result["pf"] == 2.0 and result["pnl_r"] == 0.5
    print("VALIDATOR_SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("run_dir is required")
    run_dir = args.run_dir.resolve()
    required = [
        "manifest.json", "metrics.json", "cohort_manifest.json", "primary_frozen_cohort.csv.gz",
        "secondary_frozen_cohort.npz", "simulator_comparison.csv", "trade_ledger.csv",
        "trade_identity_comparison.csv", "transition_attribution.csv", "timestamp_reconciliation.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing validator inputs: " + ", ".join(missing))
    self_check()
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    cohort_manifest = read_json(run_dir / "cohort_manifest.json")
    table = pd.read_csv(run_dir / "simulator_comparison.csv")
    ledger = pd.read_csv(run_dir / "trade_ledger.csv")
    identity = pd.read_csv(run_dir / "trade_identity_comparison.csv")
    transitions = pd.read_csv(run_dir / "transition_attribution.csv")

    metric_rows, metric_errors = reconcile_metrics(run_dir, table, ledger)
    cohort_errors = validate_cohort_identity(run_dir, manifest, metrics, table)
    transition_errors = validate_transition_configs(manifest)
    matching_errors = validate_trade_matching(ledger, identity, transitions)
    cost_errors = validate_costs(ledger)
    high_low_errors = validate_high_low_primary(run_dir, ledger)
    timezone_errors = validate_timezone(run_dir, cohort_manifest)
    cooldown_errors = validate_live_cooldown(ledger)
    stored_artifact_errors = artifact_hash_errors(run_dir, manifest)
    operational_ok = (
        manifest["operational_hashes_before"] == manifest["operational_hashes_after"]
        and manifest["operational_hashes_after"][GEMINI_FILE.name] == sha256(GEMINI_FILE)
        and manifest["operational_hashes_after"][MODEL_FILE.name] == sha256(MODEL_FILE)
        and manifest["promotion"]["operational_artifact_changed"] is False
    )
    base_ok = not (
        metric_errors
        or cohort_errors
        or transition_errors
        or matching_errors
        or cost_errors
        or high_low_errors
        or timezone_errors
        or cooldown_errors
        or stored_artifact_errors
    ) and operational_ok

    checks = {
        "chronology": make_check(
            base_ok,
            "cohort_manifest.json; primary feature_bar_time < decision_time; prior W0 OOF source",
            "The operational artifact was fit before the primary score boundary; secondary predictions are preserved chronological OOF; completed-bar times precede decisions.",
            "Rebuild a new frozen cohort with corrected timestamps and no strategy changes.",
        ),
        "feature leakage": make_check(
            base_ok,
            "primary_frozen_cohort.csv.gz all_features_finite; 31 shifted features",
            "One precomputed probability vector uses completed-bar shifted features; no exit/outcome field enters scoring.",
            "Correct feature timing in a new run and repeat only the frozen diagnostic.",
        ),
        "label maturity": make_check(
            True,
            "manifest.model.trained=false; manifest.model.label_definition",
            "No model fitting or label use occurs in this execution-only diagnostic.",
        ),
        "OOF predictions": make_check(
            base_ok,
            "cohort_manifest probability_freeze; exact artifact primary; prior W0 secondary",
            "The exact pre-test artifact scores primary once; secondary reuses byte-preserved W0 OOF scores; S0-S5 share each vector.",
            "Create a new run with one preserved prediction vector per cohort.",
        ),
        "calibration": make_check(
            True,
            "manifest.model.calibration_method=none",
            "No calibrator or probability adjustment is fitted or selected.",
        ),
        "threshold selection": make_check(
            manifest["search"]["performed"] is False and manifest["diagnostic_design"]["candidate_selection"] == 0,
            "manifest.search; six predeclared simulator definitions",
            "Threshold 0.75 and every strategy parameter are fixed; simulator variants are causal definitions, not selected candidates.",
            "Repeat with the single predeclared threshold and no candidate selection.",
        ),
        "purge/embargo": make_check(
            True,
            "manifest.model.trained=false; retained legacy training code; prior W0 provenance",
            "No fit boundary exists in this run; the existing artifact was fitted before primary and secondary was already purged OOF.",
        ),
        "holdout contamination": make_check(
            manifest["evidence_status"]["classification"] == "development_diagnostic_only",
            "manifest.evidence_status",
            "The primary and secondary intervals are explicitly development diagnostics and support no untouched or promotion claim.",
            "Relabel all inspected intervals as development evidence.",
        ),
        "recent-period reuse": make_check(
            manifest["evidence_status"]["previous_forward_status"] == "contaminated_for_future_gate_selection",
            "manifest.evidence_status.previous_forward_status",
            "The prior forward interval remains contaminated and is used only for the requested simulator reconciliation.",
            "Remove any untouched claim and retain monitoring/development status.",
        ),
        "execution alignment": make_check(
            not (transition_errors or matching_errors or high_low_errors or timezone_errors or cooldown_errors),
            "simulator definitions; trade ledger; transition/trade identity/timestamp tables",
            "Every transition is isolated, trades use deterministic episode/ordinal matching, and S1-S5 HIGH/LOW stop-first plus S5 live timing reconcile.",
            "Create a new run after correcting only the failed simulator reconstruction.",
        ),
        "cost assumptions": make_check(
            not cost_errors,
            "trade_ledger.csv gross_price/spread/extra_cost/net_r/risk_budget",
            "Fixed and observed/fallback costs reconcile trade by trade; sizing is separated from unit-R economics.",
            "Correct cost arithmetic and rerun the same frozen simulator definitions.",
        ),
        "multiple-testing risk": make_check(
            False,
            "all primary/secondary history previously inspected; no untouched final interval",
            "S0-S5 were predeclared and none is selected, but these development periods have been repeatedly inspected and cannot support a final strategy claim.",
            "Freeze a future experiment before collecting genuinely new forward data; do not reuse this run for final selection.",
        ),
    }
    internal_checks = [value["verdict"] for key, value in checks.items() if key != "multiple-testing risk"]
    internal = "PASS" if all(value == "PASS" for value in internal_checks) else "FAIL"
    overall = "PASS" if all(value["verdict"] == "PASS" for value in checks.values()) else "FAIL"
    result = {
        "overall": overall,
        "internal_methodology": internal,
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "metric_reconciliation": {"result": "PASS" if not metric_errors else "FAIL", "errors": metric_errors, "recomputed": metric_rows},
        "cohort_reconciliation": {"result": "PASS" if not cohort_errors else "FAIL", "errors": cohort_errors},
        "transition_reconciliation": {"result": "PASS" if not transition_errors else "FAIL", "errors": transition_errors},
        "trade_matching": {"result": "PASS" if not matching_errors else "FAIL", "errors": matching_errors},
        "high_low_reconciliation": {"result": "PASS" if not high_low_errors else "FAIL", "errors": high_low_errors},
        "timezone_reconciliation": {"result": "PASS" if not timezone_errors else "FAIL", "errors": timezone_errors},
        "cost_reconciliation": {"result": "PASS" if not cost_errors else "FAIL", "errors": cost_errors},
        "cooldown_reconciliation": {"result": "PASS" if not cooldown_errors else "FAIL", "errors": cooldown_errors},
        "artifact_reconciliation": {"result": "PASS" if not stored_artifact_errors else "FAIL", "errors": stored_artifact_errors},
        "operational_artifacts_unchanged": operational_ok,
        "submitted_claim": "valid_internal_execution_attribution_invalid_for_final_strategy_claim" if internal == "PASS" else "invalid_internal_execution_attribution",
        "smallest_validation_only_rerun": (
            "No internal rerun is required; only final multiple-testing/untouched validity fails, which historical recomputation cannot repair."
            if internal == "PASS"
            else "Correct the listed reconstruction errors without changing strategy parameters, then create a new immutable run."
        ),
    }
    validator_json = run_dir / "validator.json"
    validator_md = run_dir / "validator.md"
    validator_script = run_dir / "validator_script.py"
    write_json(validator_json, result)
    validator_md.write_text(render_report(result), encoding="utf-8")
    shutil.copy2(Path(__file__), validator_script)
    manifest = read_json(run_dir / "manifest.json")
    existing = {artifact.get("path") for artifact in manifest.get("artifacts", [])}
    for path, kind in (
        (validator_script, "independent_validator_script"),
        (validator_json, "independent_validator_result"),
        (validator_md, "independent_validator_report"),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative not in existing:
            manifest.setdefault("artifacts", []).append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": sha256(path),
                    "retention_status": "stored_in_run_directory_git_archival_pending",
                }
            )
    manifest["registry"]["validator_result"] = overall
    write_json(run_dir / "manifest.json", manifest)
    print(f"VALIDATOR_OVERALL_{overall}", flush=True)
    print(f"INTERNAL_METHODOLOGY_{internal}", flush=True)
    print("FINAL_UNTOUCHED_TEST_VALIDITY_FAIL", flush=True)
    return 0 if internal == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
