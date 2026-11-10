"""Offline Forward data-contract acceptance; never imports MT5 or executes a strategy."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from demo_runtime_foundation_v1 import canonical_bytes
from forward_edge_diagnosis_v3 import load_v2
from forward_test_report_v2 import (attach_versions, digest, link_signals, markdown_table,
                                    safe_path, snapshot_inputs, time_value, write_csv)
from forward_trading_ledger_v1 import build_ledger, identifier, number, scope

ROOT = Path(__file__).resolve().parent
MONEY = ("profit", "commission", "swap", "fee")
BROKER_FIELDS = ("order_id", "position_id", "deal_time", "symbol", "magic", "entry", "type",
                 "volume", "price", *MONEY, "currency")


def known(value):
    return value not in (None, "", "Unknown")


def aware(value):
    result = time_value(value)
    if result.tzinfo is None:
        raise ValueError("Timestamp timezone missing")
    return result.astimezone(timezone.utc)


def fingerprint(row):
    values = []
    for field in BROKER_FIELDS:
        value = row.get(field)
        if field in {"volume", "price", "magic", "entry", "type", *MONEY}:
            value = str(number(value).normalize())
        elif field == "deal_time":
            # Naive timestamps remain distinguishable; they never acquire UTC here.
            value = time_value(value).isoformat()
        elif field.endswith("_id"):
            value = identifier(value)
        values.append(value)
    return tuple(values)


def deal_index(rows):
    index, errors, duplicates = {}, [], 0
    for row in rows:
        key = (*scope(row), identifier(row.get("deal_id")))
        try:
            if key[-1] is None:
                raise ValueError("Missing deal_id")
            value = fingerprint(row)
            if key in index:
                duplicates += 1
                if fingerprint(index[key]) != value or any(index[key].get(f) != row.get(f) for f in ("reason", "comment")):
                    raise ValueError("Conflicting duplicate deal")
            else:
                index[key] = row
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"key": list(key), "error": str(exc)})
    return index, errors, duplicates


def validate_manifests(manifests):
    registry, errors = {}, []
    for manifest in manifests:
        if not isinstance(manifest, dict):
            errors.append({"run_id": None, "error": "Manifest must be a JSON object"})
            continue
        try:
            run = manifest.get("run_id")
            if not known(run) or manifest.get("schema_version") != 1:
                raise ValueError("Missing run_id or unsupported schema")
            if type(manifest.get("account_trade_mode")) is not int or manifest["account_trade_mode"] != 0:
                raise ValueError("Recorded account is not verified Demo mode=0")
            if not all(known(manifest.get(k)) for k in ("account_login", "account_server", "currency")):
                raise ValueError("Manifest account identity incomplete")
            aware(manifest.get("started_at"))
            for key in ("model_sha256", "configuration_sha256", "runtime_source_sha256", "foundation_source_sha256"):
                if not re.fullmatch(r"[0-9a-fA-F]{64}", str(manifest.get(key, ""))):
                    raise ValueError("Invalid "+key)
            if not re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest.get("git_commit", ""))):
                raise ValueError("Missing Git commit identity")
            config = manifest.get("configuration")
            if not isinstance(config, dict) or digest(canonical_bytes(config)) != manifest["configuration_sha256"]:
                raise ValueError("Configuration checksum mismatch")
            threshold = number(config.get("CONF_THRESHOLD"))
            if not 0 <= threshold <= 1:
                raise ValueError("Invalid confidence threshold")
            if run in registry and canonical_bytes(registry[run]) != canonical_bytes(manifest):
                raise ValueError("Conflicting run manifests")
            registry[run] = manifest
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"run_id": manifest.get("run_id"), "error": str(exc)})
    # Any invalid/conflicting receipt blocks the global acceptance gate.
    return registry, errors


def exit_category(rows):
    labels = []
    # MT5 DEAL_REASON enum values, recorded evidence only; EXPERT does not imply Timeout.
    names = {0: "CLIENT", 1: "MOBILE", 2: "WEB", 4: "SL", 5: "TP", 6: "SO"}
    for row in rows:
        try:
            reason = number(row.get("reason"))
            label = names.get(reason, "Unknown")
            comment = str(row.get("comment", "")).lower()
            if (label == "TP" and comment.startswith("[sl")) or (label == "SL" and comment.startswith("[tp")):
                label = "Unknown"
        except ValueError:
            label = "Unknown"
        labels.append(label)
    return "+".join(sorted(set(labels))) if labels and "Unknown" not in labels else "Unknown"


def inspect(deals, orders=(), signals=(), manifests=(), broker=(), expected=None, capture_errors=()):
    runtime_index, deal_errors, duplicates = deal_index(deals)
    broker_index, broker_errors, broker_duplicates = deal_index(broker)
    registry, manifest_errors = validate_manifests(manifests)
    order_errors = [{"order_id": o.get("order_id"), "error": "Order request must be a nested JSON object"}
                    for o in orders if not isinstance(o.get("request", {}), dict)]
    # Reject malformed receipts before entering the frozen V1 ledger; never invent a request.
    orders = [o for o in orders if isinstance(o.get("request", {}), dict)]
    ledger = build_ledger(deals, orders, signals)
    linked = link_signals(ledger["trades"], deals, signals)
    attach_versions(ledger["trades"], linked, list(registry.values()))
    issues, rows = [], []
    def issue(code, position, detail):
        issues.append({"code": code, "position_id": position, "detail": detail})
    for code, errors in (("deal_integrity", deal_errors), ("broker_integrity", broker_errors),
                         ("order_format", order_errors),
                         ("manifest_invalid", manifest_errors), ("input_capture", capture_errors)):
        for error in errors:
            issue(code, "GLOBAL", json.dumps(error, ensure_ascii=False))
    if not expected or not all(known(expected.get(k)) for k in ("account_login", "account_server", "currency")) or not str(expected.get("account_login")).isdigit():
        issue("expected_account_missing", "GLOBAL", "Supply expected login, server and currency explicitly; never infer them from observed data.")
    reconciliation = Decimal(0)
    comparable = 0
    for trade in ledger["trades"]:
        position = trade["position_id"]
        key = (trade["account_login"], trade["account_server"], position)
        local = [r for k, r in runtime_index.items() if (*scope(r), identifier(r.get("position_id"))) == key]
        entries = [r for r in local if str(r.get("entry")) == "0"]
        exits = [r for r in local if str(r.get("entry")) in {"1", "3"}]
        closed = trade["accounting_state"] == "complete"
        checks = {"lifecycle": trade["state"] in {"complete", "open"},
                  "account": bool(expected) and all(str(trade.get(k)) == str(expected.get(k)) for k in ("account_login", "account_server", "currency")),
                  "utc_time": trade["time_basis"] == "UTC",
                  "version": False, "initial_risk": False, "decision_and_fill": False,
                  "cost_fields": bool(local), "broker_match": bool(local), "reconciliation": False,
                  "exit_reason": None if trade["accounting_state"] == "open" else False}
        matched = linked.get(key, [])
        manifest = registry.get(trade["verified_run_id"])
        metadata = []
        try:
            if not manifest or any(str(manifest.get(k)) != str(trade.get(k)) for k in ("account_login", "account_server", "currency")):
                raise ValueError("No matching origin manifest")
            if aware(manifest["started_at"]) > aware(trade["accounting_entry_time"]):
                raise ValueError("Capture after trade origin is not origin evidence")
            for order_id in {identifier(r.get("order_id")) for r in entries}:
                candidates = {canonical_bytes(o): o for o in orders if scope(o) == key[:2]
                              and identifier(o.get("order_id")) == order_id and o.get("stage") == "result"}
                if len(candidates) != 1:
                    raise ValueError("Missing/conflicting entry order receipt")
                receipt = next(iter(candidates.values()))
                if any(receipt.get(f) != manifest.get(f) for f in ("run_id", "model_sha256", "configuration_sha256", "account_login", "account_server", "currency")):
                    raise ValueError("Order origin differs from manifest")
                metadata.append(receipt)
            checks["version"] = bool(entries and matched and metadata)
        except (ValueError, TypeError, KeyError) as exc:
            issue("version_detail", position, str(exc))
        try:
            checks["initial_risk"] = bool(metadata) and trade["initial_risk"] is not None and trade["initial_risk"] > 0
            checks["decision_and_fill"] = bool(metadata and matched)
            for receipt in metadata:
                request = receipt.get("request", {})
                if not isinstance(request, dict):
                    raise ValueError("Order request must be a JSON object")
                sl, fill, requested = (number(request.get("sl")), number(receipt.get("broker_price")), number(request.get("price")))
                side = number(request.get("type"))
                checks["initial_risk"] &= receipt.get("risk_source") == "order_calc_profit_fill_to_requested_initial_sl" and sl > 0 and ((side == 0 and sl < fill) or (side == 1 and sl > fill))
                checks["decision_and_fill"] &= requested > 0 and fill > 0
                fills = [r for r in entries if identifier(r.get("order_id")) == identifier(receipt.get("order_id"))]
                volume = sum((number(r["volume"]) for r in fills), Decimal(0))
                value = sum((number(r["volume"])*number(r["price"]) for r in fills), Decimal(0))
                checks["decision_and_fill"] &= volume == number(receipt.get("filled_volume")) and value == fill*volume
            for signal in matched:
                checks["decision_and_fill"] &= manifest is not None and aware(manifest["started_at"]) <= aware(signal.get("event_time")) <= aware(trade["accounting_entry_time"])
        except (ValueError, TypeError, KeyError):
            checks["initial_risk"] = checks["decision_and_fill"] = False
        for row in local:
            try:
                for field in MONEY:
                    number(row.get(field))
            except ValueError:
                checks["cost_fields"] = False
            other = broker_index.get((*scope(row), identifier(row.get("deal_id"))))
            try:
                if not other or fingerprint(row) != fingerprint(other):
                    checks["broker_match"] = False
                if other and known(row.get("reason")) and known(other.get("reason")) and number(row["reason"]) != number(other["reason"]):
                    checks["broker_match"] = False
                if other:
                    comment = str(row.get("comment", "")).lower()
                    if (comment.startswith("[sl") and str(other.get("reason")) == "5") or (comment.startswith("[tp") and str(other.get("reason")) == "4"):
                        checks["broker_match"] = False
            except (ValueError, TypeError):
                checks["broker_match"] = False
        reason_rows = [broker_index.get((*scope(r), identifier(r.get("deal_id"))), r) for r in exits]
        category = exit_category(reason_rows)
        if closed:
            checks["exit_reason"] = category != "Unknown"
        difference = None
        try:
            raw = sum((sum(number(r[f]) for f in MONEY) for r in local), Decimal(0))
            observed = number(trade["observed_net_pnl"])
            difference = raw-observed
            checks["reconciliation"] = bool(local) and difference == 0
            reconciliation += difference
            comparable += 1
        except (ValueError, TypeError):
            checks["reconciliation"] = False
        for check, passed in checks.items():
            if passed is False:
                issue(check, position, "Missing, mismatched or insufficient evidence; not inferred.")
        rows.append({"account_login": key[0], "account_server": key[1], "position_id": position,
                     "currency": trade["currency"], "state": trade["state"], "accounting_state": trade["accounting_state"],
                     "run_id": trade["verified_run_id"], "model_sha256": trade["model_sha256"],
                     "configuration_sha256": manifest["configuration_sha256"] if manifest else "Unknown",
                     "runtime_source_sha256": manifest["runtime_source_sha256"] if manifest else "Unknown",
                     "git_commit": manifest["git_commit"] if manifest else "Unknown",
                     "conf_threshold": trade["conf_threshold"], "initial_risk": trade["initial_risk"],
                     "recorded_net_pnl": trade["observed_net_pnl"], "reconciliation_difference": str(difference) if difference is not None else None,
                     "exit_category": category if closed else "Pending_open_position", "checks": checks,
                     "status": "PASS" if all(v is not False for v in checks.values()) else "NOT_READY"})
    # Supplied broker scope must match the complete supplied runtime scope; no hidden missing deals.
    extras = set(broker_index)-set(runtime_index)
    for key in sorted(extras, key=str):
        issue("broker_only_deal", "GLOBAL", str(key))
    if not rows:
        issue("no_samples", "GLOBAL", "Empty inputs cannot pass acceptance")
    versions = {(r["model_sha256"], r["configuration_sha256"], r["runtime_source_sha256"])
                for r in rows if r["checks"]["version"]}
    if len(versions) > 1:
        issue("mixed_strategy_versions", "GLOBAL", "Partition different model/configuration/runtime-source versions before accepting a frozen observation period.")
    coverage = []
    for check in (rows[0]["checks"] if rows else ()):
        applicable = [r for r in rows if r["checks"][check] is not None]
        count = sum(r["checks"][check] is True for r in applicable)
        coverage.append({"check": check, "passed_n": count, "applicable_n": len(applicable),
                         "coverage": count/len(applicable) if applicable else None})
    return {"schema_version": 4, "status": "PASS" if rows and not issues else "NOT_READY",
            "scope": "Supplied-file contract only; not authenticated broker/account completeness certification or strategy performance approval",
            "trades": rows, "coverage": coverage, "issues": issues,
            "summary": {"trades": len(rows), "complete": sum(r["accounting_state"] == "complete" for r in rows),
                        "open": sum(r["accounting_state"] == "open" for r in rows),
                        "runtime_duplicates": duplicates, "broker_duplicates": broker_duplicates,
                        "distinct_recorded_versions": len(versions),
                        "comparable_positions": comparable, "reconciliation_difference": str(reconciliation),
                        "issue_counts": dict(Counter(i["code"] for i in issues))},
            "limitations": ["Zero recorded fees do not establish why fees were zero or prove absence of separate account charges.",
                            "Model/source hashes are retained receipt identities, not independently authenticated historical artifact custody.",
                            "No missing data, run identity, risk or Timeout is synthesized. Spread is never deducted again.",
                            "Open trades have Pending exit evidence and are not completed performance samples."]}


def run(args):
    paths = {k: list(getattr(args, k)) for k in ("deals", "orders", "signals", "manifests", "broker_deals")}
    sealed = None
    if args.input_seal:
        seal_path = safe_path(args.input_seal)
        sealed = json.loads(seal_path.read_text(encoding="utf-8"))
        if sealed["schema_version"] != 4 or args.v2_dir:
            raise ValueError("Invalid V4 reproduction seal")
        actual = {(kind, str(safe_path(p))) for kind, items in paths.items() for p in items}
        retained = {(i["kind"], str(safe_path(seal_path.parent/i["snapshot"]))) for i in sealed["inputs"]}
        if actual != retained:
            raise ValueError("Reproduction input set differs from seal")
        for item in sealed["inputs"]:
            if digest(safe_path(seal_path.parent/item["snapshot"]).read_bytes()) != item["sha256"]:
                raise ValueError("Reproduction input checksum mismatch")
        for name, sha in sealed["source_sha256"].items():
            source = safe_path(ROOT/name)
            if not source.is_relative_to(ROOT) or digest(source.read_bytes()) != sha:
                raise ValueError("Reproduction source checksum mismatch")
        if sealed["expected_account"] != dict(account_login=args.expected_login, account_server=args.expected_server, currency=args.expected_currency):
            raise ValueError("Reproduction expected account differs from seal")
    parent = None
    if args.v2_dir:
        if any(paths[k] for k in ("deals", "orders", "signals", "manifests")):
            raise ValueError("Choose sealed V2 or explicit runtime inputs")
        _, _, manifest, _ = load_v2(args.v2_dir)
        parent = digest((safe_path(args.v2_dir)/"manifest.json").read_bytes())
        for item in manifest["inputs"]:
            if item["kind"] in paths:
                paths[item["kind"]].append(str(safe_path(args.v2_dir)/item["snapshot"]))
    if not paths["deals"]:
        raise ValueError("Supply --deals or --v2-dir")
    if set(map(safe_path, paths["deals"])) & set(map(safe_path, paths["broker_deals"])):
        raise ValueError("The same file cannot serve as runtime and broker evidence")
    artifacts, records, capture_errors = [], {}, []
    for kind, supplied in paths.items():
        records[kind], items = snapshot_inputs(supplied, kind)
        if sealed:
            previous = {i["snapshot"]: i for i in sealed["inputs"]}
            for item in items:
                item["source_changed_during_read"] |= previous[item["snapshot"]]["source_changed_during_read"]
        artifacts.extend(items)
        for item in items:
            if item["rejected"] or item["source_changed_during_read"]:
                capture_errors.append({"snapshot": item["snapshot"], "rejected": item["rejected"],
                                       "source_changed_during_read": item["source_changed_during_read"]})
    expected = dict(account_login=args.expected_login, account_server=args.expected_server, currency=args.expected_currency)
    result = inspect(records["deals"], records["orders"], records["signals"], records["manifests"], records["broker_deals"], expected, capture_errors)
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"inputs").mkdir(); (output/"source").mkdir()
    for item in artifacts:
        (output/item["snapshot"]).write_bytes(item["data"])
    source_hashes = {}
    for name in ("forward_data_quality_v4.py", "forward_edge_diagnosis_v3.py", "forward_test_report_v2.py",
                 "forward_trading_ledger_v1.py", "demo_runtime_foundation_v1.py", "training_holdout_guard_v1.py"):
        data = (ROOT/name).read_bytes()
        (output/"source"/name).write_bytes(data)
        source_hashes[name] = digest(data)
    (output/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    flattened = [{**{k: v for k, v in row.items() if k != "checks"}, **row["checks"]} for row in result["trades"]]
    for name, rows in (("trades", flattened), ("coverage", result["coverage"]), ("issues", result["issues"])):
        write_csv(output/(name+".csv"), rows)
    (output/"report.md").write_text("\n".join([
        "# XM V4 — Forward 資料品質驗收", "", "## "+result["status"], "",
        "只驗收指定檔案的資料契約，不代表broker認證、策略獲利或可啟動交易。", "",
        "```json", json.dumps(result["summary"], ensure_ascii=False, indent=2), "```", "",
        markdown_table(result["coverage"], ["check", "passed_n", "applicable_n", "coverage"]), "",
        "缺漏逐筆見issues.csv；None表示尚未適用，不能算成通過的完成交易。", "",
        *["- "+line for line in result["limitations"]], ""]), encoding="utf-8")
    replay = ["python", "-B", "source/forward_data_quality_v4.py", "--input-seal", "manifest.json", "--output", "reproduced"]
    for item in artifacts:
        replay.extend(["--"+item["kind"].replace("_", "-"), item["snapshot"]])
    for field in ("expected_login", "expected_server", "expected_currency"):
        if getattr(args, field):
            replay.extend(["--"+field.replace("_", "-"), getattr(args, field)])
    manifest = {"schema_version": 4, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": [{k: v for k, v in item.items() if k != "data"} for item in artifacts],
                "expected_account": expected, "parent_v2_manifest_sha256": parent, "source_sha256": source_hashes,
                "output_sha256": {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()},
                "reproduce_argv": replay}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir")
    parser.add_argument("--input-seal", help="Verify retained V4 inputs/source/account before reproduction")
    parser.add_argument("--output", required=True)
    for name in ("deals", "orders", "signals", "manifests", "broker-deals"):
        parser.add_argument("--"+name, action="append", default=[])
    for name in ("expected-login", "expected-server", "expected-currency"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: "+str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"output": args.output, "status": result["status"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
