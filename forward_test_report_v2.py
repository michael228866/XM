"""Read-only report of recorded Forward deals; never loads MT5, models or a simulator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from demo_runtime_foundation_v1 import canonical_bytes
from forward_trading_ledger_v1 import build_ledger, identifier, number, scope, utc_time
from training_holdout_guard_v1 import check_path

ROOT = Path(__file__).resolve().parent
LEGACY_CLOCK = "recorded_clock_unknown_timezone"
QUALIFICATION = "Recorded file values only; account/server/currency and unrecorded charges are not verified"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(path):
    resolved = check_path(path, ROOT)
    if any(part.casefold() == "future_holdout" for part in resolved.parts):
        raise PermissionError("Forward reports cannot access future_holdout")
    return resolved


def parse_snapshot(data, suffix):
    text = data.decode("utf-8-sig")
    rows, rejected = [], []
    if text and not text.endswith(("\n", "\r")):
        lines = text.splitlines(keepends=True)
        rejected.append({"line": len(lines), "reason": "Unterminated final physical line; incomplete capture cannot be excluded"})
        text = "".join(lines[:-1])
    if suffix.lower() == ".jsonl":
        for line, content in enumerate(text.splitlines(), 1):
            if not content.strip():
                continue
            try:
                row = json.loads(content)
                if not isinstance(row, dict):
                    raise ValueError("Expected object")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                rejected.append({"line": line, "reason": str(exc)})
    else:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("CSV header missing or duplicated")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                rejected.append({"line": reader.line_num, "reason": "CSV field count mismatch"})
            else:
                rows.append(row)
    return rows, rejected


def snapshot_inputs(paths, kind):
    rows, artifacts = [], []
    for index, supplied in enumerate(paths):
        path = safe_path(supplied)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if kind == "manifests":
            parsed, rejected = [json.loads(data.decode("utf-8-sig"))], []
        else:
            parsed, rejected = parse_snapshot(data, path.suffix)
        rows.extend(parsed)
        artifacts.append({
            "kind": kind, "source": str(path), "snapshot": f"inputs/{kind}_{index:03d}{path.suffix}",
            "sha256": digest(data), "bytes": len(data), "rows": len(parsed),
            "rejected": rejected, "source_changed_during_read":
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns),
            "data": data,
        })
    return rows, artifacts


def time_value(value):
    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return value if value.tzinfo is None else value.astimezone(timezone.utc)


def is_in_window(value, window):
    if value is None:
        return False
    value = time_value(value)
    return window is None or window[0] <= value < window[1]


def metric(value, n, unit="recorded_file_units", limitation=QUALIFICATION):
    return {"value": value, "n": n, "unit": unit, "limitation": limitation}


def money_sum(values):
    return float(sum((number(value) for value in values), Decimal(0)))


def summarize(trades, cash=(), window=None, *, chronological=True):
    values = [t["accounting_net_pnl"] for t in trades]
    wins, losses = [v for v in values if v > 0], [v for v in values if v < 0]
    gain, loss = money_sum(wins), -money_sum(losses)
    avg_win = money_sum(wins)/len(wins) if wins else None
    avg_loss = money_sum(losses)/len(losses) if losses else None
    # R is available only from the strict V1 ledger, never inferred from old risk_budget.
    risk = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    by_time = defaultdict(list)
    for event in cash:
        by_time[event["time"]].append(event["net"])
    equity = peak = drawdown = Decimal(0)
    for when in sorted(by_time):
        equity += sum((number(v) for v in by_time[when]), Decimal(0))
        peak = max(peak, equity)
        drawdown = min(drawdown, equity-peak)
    streak = maximum = 0
    exits = defaultdict(list)
    for trade in trades:
        exits[trade["accounting_exit_time"]].append(trade["accounting_net_pnl"])
    ambiguous_order = any(len(v) > 1 and any(x < 0 for x in v) and any(x >= 0 for x in v)
                          for v in exits.values())
    for when in sorted(exits):
        for pnl in exits[when]:
            streak = streak+1 if pnl < 0 else 0
            maximum = max(maximum, streak)
    days = (window[1]-window[0]).total_seconds()/86400 if window else None
    result = {
        "total_net_pnl": metric(money_sum(values), len(values)),
        "average_net_pnl": metric(money_sum(values)/len(values) if values else None, len(values)),
        "win_rate": metric(len(wins)/len(values) if values else None, len(values), "fraction"),
        "profit_factor": metric(gain/loss if loss else None, len(values), "ratio"),
        "mean_r": metric(math.fsum(risk)/len(risk) if risk else None, len(risk), "R", "Only complete captured initial-SL risk; planned budget is not substituted"),
        "average_win": metric(avg_win, len(wins)), "average_loss": metric(avg_loss, len(losses)),
        "payoff_ratio": metric(avg_win/abs(avg_loss) if avg_win is not None and avg_loss else None,
                               len(wins)+len(losses), "ratio"),
        "realized_max_drawdown": metric(float(drawdown) if cash and chronological else None, len(cash),
            limitation=QUALIFICATION if chronological else "Cannot order different/unknown clock contexts"),
        "trades_per_day": metric(len(values)/days if days else None, len(values), "trades/calendar_day",
                                  "Explicit observation window required; no inferred activity coverage"),
        "observed_cash_pnl": metric(money_sum(e["net"] for e in cash) if cash else None, len(cash)),
        "max_losing_streak": metric(maximum if chronological and not ambiguous_order and trades else None,
                                    len(trades), "trades", "Descriptive only; ties with mixed outcomes remain Unknown"),
        "gross_trade_profit": metric(money_sum(t["profit"] for t in trades), len(trades)),
        "explicit_cost": metric(money_sum(t["explicit_cost"] for t in trades), len(trades)),
    }
    gross = money_sum(t["profit"] for t in trades)
    result["cost_to_net_gross_profit"] = metric(
        result["explicit_cost"]["value"]/gross if gross > 0 else None, len(trades), "ratio",
        "Explicit commission/swap/fee divided by signed total broker profit; unavailable when denominator <=0")
    positive_gross = money_sum(max(t["profit"], 0) for t in trades)
    result["cost_to_positive_gross_profit"] = metric(
        result["explicit_cost"]["value"]/positive_gross if positive_gross > 0 else None,
        len(trades), "ratio", "Recorded explicit cost only; spread already embedded in fill profit")
    result["profit_factor_status"] = "finite" if loss else "infinite" if gain else "undefined"
    result["observation_days"] = days
    result["assessment"] = "insufficient_sample" if len(trades) < 20 else "descriptive_only_not_predictive"
    result["risk_missing_samples"] = len(trades)-len(risk)
    return result


def link_signals(trades, deals, signals):
    entry_by_position = defaultdict(list)
    indexes = {"deal_id": defaultdict(list), "order_id": defaultdict(list), "position_ticket": defaultdict(list)}
    for row in deals:
        if str(row.get("entry")) == "0" and identifier(row.get("position_id")):
            entry_by_position[(*scope(row), identifier(row["position_id"]))].append(row)
    for row in signals:
        if row.get("status") not in {"order_opened", "order_partial"}:
            continue
        for field, index in indexes.items():
            value = identifier(row.get(field))
            if value:
                index[(*scope(row), value)].append(row)
    linked = {}
    for trade in trades:
        key = (trade["account_login"], trade["account_server"], trade["position_id"])
        entries = entry_by_position[key]
        deals_for_trade = {identifier(r.get("deal_id")) for r in entries}
        orders_for_trade = {identifier(r.get("order_id")) for r in entries}
        candidates = []
        for entry in entries:
            candidates.extend(indexes["deal_id"].get((*scope(entry), identifier(entry.get("deal_id"))), []))
            candidates.extend(indexes["order_id"].get((*scope(entry), identifier(entry.get("order_id"))), []))
        candidates.extend(indexes["position_ticket"].get(key, []))
        candidates = list({canonical_bytes(r): r for r in candidates}.values())
        conflict = any(
            (identifier(r.get("deal_id")) and identifier(r.get("deal_id")) not in deals_for_trade)
            or (identifier(r.get("order_id")) and identifier(r.get("order_id")) not in orders_for_trade)
            or (identifier(r.get("position_ticket")) and identifier(r.get("position_ticket")) != key[2])
            for r in candidates)
        covered_orders = {identifier(r.get("order_id")) for r in candidates}
        complete_link = bool(candidates) and not conflict and orders_for_trade <= covered_orders
        scores = set()
        for row in candidates:
            try:
                score = float(number(row.get("buy_prob")))
                if not 0 <= score <= 1:
                    raise ValueError("Out of range")
                scores.add(score)
            except ValueError:
                complete_link = False
        score = next(iter(scores)) if complete_link and len(scores) == 1 else None
        linked[key] = candidates if complete_link else []
        trade["recorded_buy_score"] = score
        trade["signal_link"] = "explicit_ids_account_unverified" if complete_link and "Unknown" in key[:2] else "explicit_ids" if complete_link else "Unknown"
        trade["signal_link_issue"] = "conflicting_ids" if conflict else "missing_or_mixed_score" if score is None else None
    return linked


def attach_versions(trades, linked, manifests):
    registry = {}
    for manifest in manifests:
        run = manifest.get("run_id")
        if not run or digest(canonical_bytes(manifest.get("configuration"))) != manifest.get("configuration_sha256"):
            raise ValueError("Manifest run/configuration identity invalid")
        for field in ("model_sha256", "runtime_source_sha256"):
            value = manifest.get(field, "")
            if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
                raise ValueError("Manifest hash missing/invalid: "+field)
        if run in registry and registry[run] != manifest:
            raise ValueError("Conflicting run manifests")
        registry[run] = manifest
    for trade in trades:
        key = (trade["account_login"], trade["account_server"], trade["position_id"])
        matched = linked.get(key, [])
        versions = []
        for row in matched:
            manifest = registry.get(row.get("run_id"))
            if manifest and all(row.get(k) == manifest.get(k) for k in
                                ("model_sha256", "configuration_sha256", "account_login", "account_server")):
                if scope(row) == key[:2]:
                    versions.append(manifest)
        unique = {m["run_id"] for m in versions}
        manifest = versions[0] if matched and len(versions) == len(matched) and len(unique) == 1 else None
        trade["model_sha256"] = manifest["model_sha256"] if manifest else "Unknown"
        trade["conf_threshold"] = manifest["configuration"].get("CONF_THRESHOLD", "Unknown") if manifest else "Unknown"
        trade["git_commit"] = manifest.get("git_commit") or "Unknown" if manifest else "Unknown"
        trade["runtime_source_sha256"] = manifest.get("runtime_source_sha256", "Unknown") if manifest else "Unknown"
        trade["verified_run_id"] = manifest["run_id"] if manifest else "Unknown"


def metric_row(label, summary, basis=None):
    result = {"group": label}
    if basis:
        result["time_basis"] = basis
    for key, value in summary.items():
        if isinstance(value, dict):
            result[key] = value["value"]
            result[key+"_n"] = value["n"]
        else:
            result[key] = value
    return result


def build_report(deals, signals, orders, manifests, windows):
    ledger = build_ledger(deals, orders, signals)
    linked = link_signals(ledger["trades"], deals, signals)
    attach_versions(ledger["trades"], linked, manifests)
    selected = [t for t in ledger["trades"] if t["accounting_state"] == "complete"
                and is_in_window(t["accounting_exit_time"], windows.get(t["time_basis"]))]
    cash = [e for e in ledger["observed_cash_events"]
            if is_in_window(e["time"], windows.get(e["time_basis"]))]
    bases = {t["time_basis"] for t in selected} | {e["time_basis"] for e in cash}
    unit_scopes = {(t["account_login"], t["account_server"], t["currency"]) for t in [*selected, *cash]}
    if len(unit_scopes) > 1:
        raise ValueError("Multiple account/server/currency scopes; partition inputs before reporting")
    summary = summarize(selected, cash, chronological=len(bases) == 1 and len(unit_scopes) == 1)
    partitions, daily, weekly, sessions, scores, holding = [], [], [], [], [], []
    for basis in sorted(bases):
        trades = [t for t in selected if t["time_basis"] == basis]
        events = [e for e in cash if e["time_basis"] == basis]
        summary_by_basis = summarize(trades, events, windows.get(basis))
        partitions.append(metric_row(basis, summary_by_basis, basis))
        groups = defaultdict(list)
        for trade in trades:
            when = time_value(trade["accounting_exit_time"])
            groups[when.date().isoformat()].append(trade)
        dates = sorted(groups)
        if windows.get(basis):
            start, end = windows[basis]
            date = start.date()
            dates = []
            while datetime.combine(date, datetime.min.time(), tzinfo=start.tzinfo) < end:
                dates.append(date.isoformat())
                date += timedelta(days=1)
        weeks = defaultdict(list)
        week_dates = defaultdict(list)
        for date in dates:
            date_value = datetime.fromisoformat(date)
            iso = date_value.isocalendar()
            week = f"{iso.year}-W{iso.week:02d}"
            weeks[week].extend(groups[date])
            week_dates[week].append(date)
            day_cash = [e for e in events if time_value(e["time"]).date().isoformat() == date]
            daily.append(metric_row(date, summarize(groups[date], day_cash), basis))
        for week in sorted(weeks):
            week_cash = [e for e in events if time_value(e["time"]).date().isoformat() in week_dates[week]]
            weekly.append(metric_row(week, summarize(weeks[week], week_cash), basis))
    grouped_sessions, grouped_scores, grouped_holding = defaultdict(list), defaultdict(list), defaultdict(list)
    for band in ("below_0.70", "0.70_to_below_0.75", "0.75_to_1.00", "Unknown"):
        grouped_scores[band] = []
    for trade in selected:
        entry = time_value(trade["accounting_entry_time"])
        exit_time = time_value(trade["accounting_exit_time"])
        session = f"recorded_UTC_{(entry.hour//8)*8:02d}_{(entry.hour//8+1)*8:02d}" if entry.tzinfo else "Unknown_timezone"
        grouped_sessions[session].append(trade)
        score = trade["recorded_buy_score"]
        band = "Unknown" if score is None else "below_0.70" if score < .70 else "0.70_to_below_0.75" if score < .75 else "0.75_to_1.00"
        grouped_scores[band].append(trade)
        minutes = (exit_time-entry).total_seconds()/60
        trade["holding_minutes"] = minutes
        band = "0_to_15m" if minutes <= 15 else "over_15_to_60m" if minutes <= 60 else "over_60_to_90m" if minutes <= 90 else "over_90m"
        grouped_holding[trade["time_basis"]+":"+band].append(trade)
    for target, grouping in ((sessions, grouped_sessions), (scores, grouped_scores), (holding, grouped_holding)):
        for label in sorted(grouping):
            target.append(metric_row(label, summarize(grouping[label], chronological=False)))
    raw_sums = {key: Decimal(0) for key in ("profit", "commission", "swap", "fee")}
    seen, numeric_invalid, conflicts = {}, 0, 0
    for row in deals:
        key = (*scope(row), identifier(row.get("deal_id")))
        try:
            fields = {field: number(row.get(field)) for field in raw_sums}
        except ValueError:
            numeric_invalid += 1
            continue
        if key[2] is None:
            numeric_invalid += 1
            continue
        if key in seen:
            conflicts += seen[key] != fields
            continue
        seen[key] = fields
        for field, value in fields.items():
            raw_sums[field] += value
    recorded_net = sum(raw_sums.values())
    accounted_decimal = sum((number(t["observed_net_pnl"]) for t in ledger["trades"]
                             if t["observed_net_pnl"] is not None), Decimal(0))
    accounted = float(accounted_decimal)
    integrity = {
        "input_deal_rows": len(deals), "unique_deal_ids_by_scope": len(seen),
        "duplicate_deals": ledger["duplicate_deals"], "conflicting_cost_duplicates": conflicts,
        "invalid_numeric_rows": numeric_invalid,
        "positions": len(ledger["trades"]), "entry_types": dict(Counter(str(r.get("entry")) for r in deals)),
        "strict_states": dict(Counter(t["state"] for t in ledger["trades"])),
        "recorded_accounting_states": dict(Counter(t["accounting_state"] for t in ledger["trades"])),
        "issue_counts": dict(Counter(issue for t in ledger["trades"] for issue in t["issues"])),
        "raw_unique_sums": {key: str(value) for key, value in raw_sums.items()},
        "raw_unique_net": str(recorded_net), "ledger_observed_net": accounted,
        "reconciliation_difference": str(recorded_net-accounted_decimal),
        "known_model_versions": sum(t["verified_run_id"] != "Unknown" for t in ledger["trades"]),
        "unknown_model_versions": sum(t["verified_run_id"] == "Unknown" for t in ledger["trades"]),
        "score_linked_positions": sum(t["recorded_buy_score"] is not None for t in ledger["trades"]),
        "all_explicit_costs_recorded_zero": bool(deals) and numeric_invalid == 0 and all(
            number(r.get(k)) == 0 for r in deals for k in ("commission", "swap", "fee")),
        "deal_time_bounds": {}, "signal_time_bounds": {},
        "claim": "File-internal reconciliation only; not broker statement or account completeness certification",
    }
    for key, rows, field in (("deal_time_bounds", deals, "deal_time"), ("signal_time_bounds", signals, "event_time")):
        grouped = defaultdict(list)
        for row in rows:
            try:
                value = time_value(row.get(field))
                grouped[LEGACY_CLOCK if value.tzinfo is None else "UTC"].append(value.isoformat())
            except (ValueError, TypeError):
                continue
        integrity[key] = {basis: {"first": min(v), "last": max(v), "rows": len(v)} for basis, v in grouped.items()}
    limitations = [
        "No MT5 connection, broker statement, model execution or historical strategy simulation was used.",
        "Unknown account/server/currency are never populated from the current runtime or user description.",
        "Recorded zero fees are not proof that all broker costs were retained; spread is not deducted again.",
        "Naive timestamps remain an independent recorded-clock cohort; sessions are Unknown for these rows.",
        "Explicit +00:00 is the recorded clock convention, not independent certification of broker UTC accuracy.",
        "Unversioned score bands can span different models; they do not validate threshold 0.70 versus 0.75.",
        "Closed-position metrics omit open/Unknown positions; cash drawdown includes known partial exits and entry fees.",
        "Drawdown is realized recorded cash, not equity including floating PnL or a percentage drawdown.",
        "Zero-record dates within supplied windows are retained; these are not proof that no actual trades occurred.",
        "Groups below 20 trades are marked insufficient_sample; larger groups remain descriptive, not predictive.",
        "Observed history has been inspected; no untouched OOS or sustainable-alpha claim is made.",
    ]
    frequency = []
    for basis in sorted(bases):
        dates = [r for r in daily if r["time_basis"] == basis]
        counts = [r["total_net_pnl_n"] for r in dates]
        frequency.append({"time_basis": basis, "n_days": len(counts),
                          "zero_record_days": sum(n == 0 for n in counts),
                          "days_with_records": sum(n > 0 for n in counts),
                          "min_records_per_day": min(counts) if counts else None,
                          "max_records_per_day": max(counts) if counts else None,
                          "median_records_per_day": statistics.median(counts) if counts else None,
                          "mean_records_per_day": math.fsum(counts)/len(counts) if counts else None,
                          "limitation": "Recorded trade exits; continuity of capture not certified"})
    return {"schema_version": 2, "qualification": QUALIFICATION, "integrity": integrity,
            "summary": summary, "time_partitions": partitions, "daily": daily, "weekly": weekly,
            "sessions": sessions, "score_bands": scores, "holding_bands": holding,
            "frequency_distribution": frequency,
            "ledger": ledger, "limitations": limitations,
            "windows": {basis: [a.isoformat(), b.isoformat()] for basis, (a, b) in windows.items() if a is not None}}


def markdown_table(rows, columns):
    def display(value):
        if value is None:
            return "Unknown / N/A"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "|"+"---|"*len(columns)] +
                     ["| " + " | ".join(display(r.get(c)) for c in columns)+" |" for r in rows])


def render_markdown(report):
    integrity = report["integrity"]
    lines = ["# Forward Test Ledger Verification V2", "",
             "**本報告是既有檔案的唯讀對帳與描述統計，不是回測、策略通過或經認證的單一 Demo 帳戶績效。**", "",
             "## 資料與對帳", "", "```json", json.dumps(integrity, ensure_ascii=False, indent=2), "```", "",
             markdown_table([{"kind": r["kind"], "rows": r["rows"], "sha256": r["sha256"],
                              "rejected_rows": len(r["rejected"]), "changed_during_read": r["source_changed_during_read"]}
                             for r in report.get("input_integrity", [])],
                            ["kind", "rows", "sha256", "rejected_rows", "changed_during_read"]), "",
             "## 檔案記錄值績效（帳戶／幣別／外部成本未驗證）", "",
             "嚴格 V1 狀態與附條件的數值統計分開；Unknown 身分不因可以加總損益而變成已確認。", "",
             markdown_table([{"metric": k, **v} for k, v in report["summary"].items() if isinstance(v, dict)],
                            ["metric", "value", "n", "unit", "limitation"])]
    for title, key in (("時間分組與已實現回撤", "time_partitions"), ("Session（依入場時間）", "sessions"),
                       ("已記錄入場分數帶", "score_bands"), ("持倉時間", "holding_bands")):
        lines.extend(["", "## "+title, "", markdown_table(report[key],
            ["group", "total_net_pnl", "total_net_pnl_n", "win_rate", "profit_factor", "realized_max_drawdown",
             "average_net_pnl", "max_losing_streak", "trades_per_day", "assessment"])])
    lines.extend(["", "## 每日／每週與交易頻率", "",
                  "完整資料見 daily.csv、weekly.csv；保留明確視窗內零成交紀錄日，這不保證當日沒有實際交易。以下為每日記錄淨損益最低的十日，僅作描述。", "",
                  markdown_table(report["frequency_distribution"],
                                 ["time_basis", "n_days", "days_with_records", "zero_record_days", "max_records_per_day", "median_records_per_day"]), "",
                  markdown_table(sorted(report["daily"], key=lambda r: r["total_net_pnl"])[:10],
                                 ["group", "time_basis", "total_net_pnl", "total_net_pnl_n", "win_rate"]),
                  "", "## 限制", "", *["- "+v for v in report["limitations"]], "",
                  "## 重現", "", "輸入原樣快照、SHA256、報表程式與 V1 帳本快照、Git commit/dirty、參數與重現命令見 manifest.json。",
                  "report.json 與所有資料 CSV 可由保留快照重建；不得用目前模型／參數為舊交易補造版本。", ""])
    return "\n".join(lines)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run_report(args):
    deals, deal_inputs = snapshot_inputs(args.deals, "deals")
    signals, signal_inputs = snapshot_inputs(args.signals, "signals")
    orders, order_inputs = snapshot_inputs(args.orders, "orders")
    manifests, manifest_inputs = snapshot_inputs(args.manifests, "manifests")
    windows = {"UTC": (utc_time(args.start), utc_time(args.end))}
    if bool(args.recorded_start) != bool(args.recorded_end):
        raise ValueError("Both recorded-clock bounds are required")
    if args.recorded_start:
        bounds = time_value(args.recorded_start), time_value(args.recorded_end)
        if any(v.tzinfo is not None for v in bounds):
            raise ValueError("Recorded-clock bounds must remain naive, with no invented timezone")
        windows[LEGACY_CLOCK] = bounds
    if any(end <= start for start, end in windows.values()):
        raise ValueError("Observation end must follow start")
    report = build_report(deals, signals, orders, manifests, windows)
    artifacts = [*deal_inputs, *signal_inputs, *order_inputs, *manifest_inputs]
    report["input_integrity"] = [{k: v for k, v in item.items() if k != "data"} for item in artifacts]
    output = safe_path(args.output)
    # Complete all calculations before creating a new, never reused output directory.
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "source").mkdir()
    for item in artifacts:
        with (output / item["snapshot"]).open("xb") as handle:
            handle.write(item["data"])
    sources = {}
    for name in ("forward_test_report_v2.py", "forward_trading_ledger_v1.py",
                 "demo_runtime_foundation_v1.py", "training_holdout_guard_v1.py"):
        data = (Path(__file__).resolve().parent / name).read_bytes()
        with (output / "source" / name).open("xb") as handle:
            handle.write(data)
        sources[name] = digest(data)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    manifest = {"schema_version": 2, "git_commit": commit, "git_dirty": dirty,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version, "input_mode": "read_only",
                "source_sha256": sources, "inputs": report["input_integrity"], "windows": report["windows"],
                "authorization": "User requested read-only Forward Test Ledger Verification V2; no training/replay/promotion",
                "reproduce_argv": ["python", "-B", "source/forward_test_report_v2.py"]}
    for item in artifacts:
        flag = {"deals": "--deals", "signals": "--signals", "orders": "--orders", "manifests": "--manifests"}[item["kind"]]
        manifest["reproduce_argv"].extend([flag, item["snapshot"]])
    manifest["reproduce_argv"].extend(["--start", args.start, "--end", args.end, "--output", "reproduced"])
    if args.recorded_start:
        manifest["reproduce_argv"].extend(["--recorded-start", args.recorded_start, "--recorded-end", args.recorded_end])
    for name, value in (("report.json", report),):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    write_csv(output / "trades.csv", report["ledger"]["trades"])
    write_csv(output / "summary.csv", [{"metric": k, **v} for k, v in report["summary"].items() if isinstance(v, dict)])
    for key in ("daily", "weekly", "sessions", "score_bands", "holding_bands", "time_partitions", "frequency_distribution"):
        write_csv(output / (key+".csv"), report[key])
    # Seal outputs only after writing them; never reuse a previous report directory.
    manifest["output_sha256"] = {p.name: digest(p.read_bytes()) for p in output.iterdir()
                                 if p.is_file() and p.name != "manifest.json"}
    # Exclusive creation prevents replacing any pre-existing manifest.
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", action="append", required=True)
    parser.add_argument("--signals", action="append", default=[])
    parser.add_argument("--orders", action="append", default=[])
    parser.add_argument("--manifests", action="append", default=[])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--recorded-start")
    parser.add_argument("--recorded-end")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_report(args)
    print(json.dumps({"output": args.output, "integrity": report["integrity"],
                      "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
