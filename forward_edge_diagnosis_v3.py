"""Sealed V2 Forward-record diagnosis only; no MT5/model execution or strategy simulation."""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import random
import re
import statistics
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from forward_test_report_v2 import (
    LEGACY_CLOCK, digest, markdown_table, money_sum, parse_snapshot, safe_path, time_value, write_csv,
)

ROOT = Path(__file__).resolve().parent
LABELS = {"Confirmed", "Supported Hypothesis", "Insufficient Evidence"}


def conclusion(label, text, n, source, limitation):
    if label not in LABELS:
        raise ValueError("Unknown evidence label")
    return {"label": label, "text": text, "n": n, "source": source, "limitation": limitation}


def safe_child(root, name):
    path = safe_path(Path(root)/name)
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Manifest path escapes its report directory")
    return path


def load_v2(directory):
    root = safe_path(directory)
    manifest = json.loads(safe_child(root, "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if digest(safe_child(root, name).read_bytes()) != expected:
            raise ValueError("V2 output seal mismatch: "+name)
    for name, expected in manifest["source_sha256"].items():
        if digest(safe_child(root, "source/"+name).read_bytes()) != expected:
            raise ValueError("V2 source seal mismatch: "+name)
    if digest((ROOT/"forward_test_report_v2.py").read_bytes()) != manifest["source_sha256"]["forward_test_report_v2.py"]:
        raise ValueError("V2 helper differs from retained source; use its exact preserved copy")
    report = json.loads(safe_child(root, "report.json").read_text(encoding="utf-8"))
    records = defaultdict(list)
    input_inventory = []
    for item in manifest["inputs"]:
        path = safe_child(root, item["snapshot"])
        data = path.read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError("V2 input seal mismatch: "+item["snapshot"])
        if item["kind"] in {"deals", "signals", "orders"}:
            rows, rejected = parse_snapshot(data, path.suffix)
            if rejected != item["rejected"] or len(rows) != item["rows"]:
                raise ValueError("V2 parsed-input identity mismatch")
            records[item["kind"]].extend(rows)
        input_inventory.append({"kind": item["kind"], "path": item["snapshot"],
                                "sha256": item["sha256"], "rows": item["rows"]})
    return report, records, manifest, input_inventory


def financial(trades, additional_cost=0):
    cost = Decimal(str(additional_cost))
    if not cost.is_finite() or cost < 0:
        raise ValueError("Hypothetical additional cost must be finite and nonnegative")
    values = [float(Decimal(str(t["accounting_net_pnl"]))-cost) for t in trades]
    wins, losses = [v for v in values if v > 0], [v for v in values if v < 0]
    gain, loss = money_sum(wins), -money_sum(losses)
    avg_win = gain/len(wins) if wins else None
    avg_loss = -loss/len(losses) if losses else None
    zero = sum(v == 0 for v in values)
    wr = len(wins)/len(values) if values else None
    conditional_be = abs(avg_loss)/(avg_win+abs(avg_loss)) if avg_win and avg_loss else None
    # Zero-profit trades contribute neither side of expectancy. Use the same total-N denominator as WR.
    be = conditional_be*(1-zero/len(values)) if conditional_be is not None else None
    return {
        "n": len(values), "n_wins": len(wins), "n_losses": len(losses), "n_zero": zero,
        "gross_profit": gain, "gross_loss": loss,
        "net_pnl": money_sum(values), "net_pf": gain/loss if loss else None,
        "pf_status": "finite" if loss else "infinite" if gain else "undefined",
        "win_rate": wr, "average_win": avg_win, "average_loss": avg_loss,
        "payoff_ratio": avg_win/abs(avg_loss) if avg_win and avg_loss else None,
        "break_even_win_rate": be, "wr_minus_break_even": wr-be if be is not None else None,
        "net_expectancy_per_trade": money_sum(values)/len(values) if values else None,
        "gross_positive_broker_profit": money_sum(max(t["profit"], 0) for t in trades),
        "gross_negative_broker_profit": -money_sum(min(t["profit"], 0) for t in trades),
        "recorded_explicit_cost": money_sum(t["explicit_cost"] for t in trades),
        "recorded_cost_to_positive_broker_profit": money_sum(t["explicit_cost"] for t in trades)/money_sum(max(t["profit"], 0) for t in trades) if any(t["profit"] > 0 for t in trades) else None,
        "additional_cost_per_closed_trade": float(cost),
        "unit": "recorded_file_units_account_currency_unverified",
        "source": "V2 sealed ledger; gross_profit/loss are sums of winning/losing NET trades",
    }


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    value = ordered[int((len(ordered)-1)*q)]
    return value if math.isfinite(value) else None


def week_bootstrap(trades, window, repetitions=2000, seed=42):
    if not 100 <= repetitions <= 100000:
        raise ValueError("Bootstrap repetitions must be 100..100000")
    start, end = [time_value(v) for v in window]
    if end <= start:
        raise ValueError("Invalid observation window")
    blocks = {}
    day = start.date()
    while datetime.combine(day, datetime.min.time(), tzinfo=start.tzinfo) < end:
        iso = day.isocalendar()
        blocks.setdefault((iso.year, iso.week), [])
        day += timedelta(days=1)
    for trade in trades:
        when = time_value(trade["accounting_exit_time"])
        if (when.tzinfo is None) != (start.tzinfo is None):
            raise ValueError("Never bootstrap across clock contexts")
        if not start <= when < end:
            continue
        iso = when.isocalendar()
        blocks[(iso.year, iso.week)].append(trade["accounting_net_pnl"])
    weekly = list(blocks.values())
    rng = random.Random(seed)
    means, wrs, pfs = [], [], []
    empty = 0
    for _ in range(repetitions):
        sample = [p for _ in weekly for p in weekly[rng.randrange(len(weekly))]]
        if not sample:
            empty += 1
            continue
        gain = money_sum(v for v in sample if v > 0)
        loss = -money_sum(v for v in sample if v < 0)
        means.append(money_sum(sample)/len(sample))
        wrs.append(sum(v > 0 for v in sample)/len(sample))
        if loss:
            pfs.append(gain/loss)
        elif gain:
            pfs.append(float("inf"))
    active = sum(bool(b) for b in weekly)
    return {
        "n_trades": len(trades), "calendar_week_blocks": len(weekly), "active_week_blocks": active,
        "repetitions": repetitions, "seed": seed, "empty_resamples": empty,
        "mean_pnl_95_interval": [percentile(means, .025), percentile(means, .975)],
        "win_rate_95_interval": [percentile(wrs, .025), percentile(wrs, .975)],
        "pf_95_interval": [percentile(pfs, .025), percentile(pfs, .975)],
        "pf_infinite_resamples": sum(math.isinf(v) for v in pfs),
        "valid_mean_resamples": len(means), "valid_pf_resamples": len(pfs),
        "label": "Insufficient Evidence" if active < 8 else "Supported Hypothesis",
        "method": "Fixed ISO-calendar-week clusters; draw the same number of weeks with replacement, keeping all trades within each sampled week, including empty observed weeks",
        "limitations": ["Within-week dependence is retained; cross-week dependence/regime ordering is not.",
                       "Partial boundary weeks are retained; sparse/zero-record weeks are not verified inactivity.",
                       "Percentile intervals use nearest order statistics, are descriptive, and are not untouched OOS or proof of alpha.",
                       "Unknown accounts, model mixtures and missing costs remain unresolved; empty resamples are excluded and counted.",
                       "Infinite PF bounds are null with infinite-resample counts, never silently changed to a finite value."],
    }


def correlation(x, y):
    if len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    xx, yy = math.fsum((v-mx)**2 for v in x), math.fsum((v-my)**2 for v in y)
    return math.fsum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(xx*yy) if xx and yy else None


def average_ranks(values):
    pairs = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0]*len(values)
    i = 0
    while i < len(pairs):
        j = i+1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        for k in range(i, j):
            ranks[pairs[k][0]] = (i+1+j)/2
        i = j
    return ranks


def holding_relationship(trades):
    valid = [t for t in trades if t.get("holding_minutes") is not None and t["entry_volume"] > 0]
    x = [t["holding_minutes"] for t in valid]
    y = [t["accounting_net_pnl"] for t in valid]
    normalized = [t["accounting_net_pnl"]/t["entry_volume"] for t in valid]
    return {"n": len(valid), "pearson_duration_pnl": correlation(x, y),
            "spearman_duration_pnl": correlation(average_ranks(x), average_ranks(y)),
            "pearson_duration_pnl_per_lot": correlation(x, normalized),
            "label": "Supported Hypothesis" if len(valid) >= 20 else "Insufficient Evidence",
            "limitation": "Association only: outcome/TP/SL determines duration; volume, unversioned models and recorded-clock uncertainty confound it. Per-lot PnL is not R."}


def concentration(trades):
    losses = sorted((t for t in trades if t["accounting_net_pnl"] < 0), key=lambda t: t["accounting_net_pnl"])
    total = -money_sum(t["accounting_net_pnl"] for t in losses)
    return [{"top_k": k, "n_losses_in_slice": min(k, len(losses)), "n_all_losses": len(losses),
             "loss_amount": -money_sum(t["accounting_net_pnl"] for t in losses[:k]),
             "share_of_gross_loss": -money_sum(t["accounting_net_pnl"] for t in losses[:k])/total if total else None,
             "label": "Confirmed", "source": "V2 sealed ledger; observed losses retained, none removed"}
            for k in (1, 3, 5, 10)]


def classify_exit(trade, deals, enums):
    exit_rows = [r for r in deals if str(r.get("position_id")) == trade["position_id"]
                 and str(r.get("entry")) in {"1", "3"}
                 and str(r.get("account_login") or "Unknown") == trade["account_login"]
                 and str(r.get("account_server") or "Unknown") == trade["account_server"]]
    labels = []
    for row in exit_rows:
        reason = str(row.get("reason"))
        comment = str(row.get("comment", "")).lower()
        label = "TP" if reason == str(enums.get("DEAL_REASON_TP", "unverified")) else "SL" if reason == str(enums.get("DEAL_REASON_SL", "unverified")) else "Unknown"
        if (label == "TP" and comment.startswith("[sl")) or (label == "SL" and comment.startswith("[tp")):
            label = "Unknown_conflicting_reason_comment"
        labels.append(label)
    category = labels[0] if labels and len(set(labels)) == 1 else "Mixed_or_Unknown"
    return {"position_id": trade["position_id"], "time_basis": trade["time_basis"],
            "exit_category": category, "n_exit_deals": len(exit_rows),
            "recorded_reasons": sorted({str(r.get("reason")) for r in exit_rows}),
            "timeout": "Unknown_not_inferred_from_duration" if category not in {"TP", "SL"} else "not_indicated_by_recorded_exit_reason",
            "label": "Confirmed" if category in {"TP", "SL"} else "Insufficient Evidence",
            "limitation": "Classification of recorded broker reason only; expert/market close does not uniquely identify Timeout."}


def signal_layers(signals):
    result = []
    for basis in (LEGACY_CLOCK, "UTC"):
        rows = []
        invalid_time = 0
        for row in signals:
            try:
                when = time_value(row.get("event_time"))
                if (when.tzinfo is None) == (basis == LEGACY_CLOCK):
                    rows.append(row)
            except (ValueError, TypeError):
                invalid_time += 1
        statuses = Counter(r.get("status", "Unknown") for r in rows)
        reasons = Counter()
        not_valid = [r for r in rows if r.get("status") == "not_valid"]
        for row in not_valid:
            reasons.update(str(row.get("message", "")).split(";") if row.get("message") else ["Unknown_reason"])
        result.append({"time_basis": basis, "n_signal_rows": len(rows), "statuses": dict(statuses),
                       "not_valid_reasons_overlapping": dict(reasons), "invalid_timestamp_rows": invalid_time,
                       "n_not_valid_with_reason": sum(bool(r.get("message")) for r in not_valid),
                       "label": "Confirmed", "causal_attribution": "Insufficient Evidence",
                       "limitation": "Rows are not independent episodes/trades. entry_mode can overlap Session/RSI and threshold definitions are unversioned; gate ordering/counterfactual recovery is not reconstructed."})
    return result


def source_constants(data):
    tree = ast.parse(data.decode("utf-8-sig"))
    wanted = {"MODEL_FILE", "MODEL_OUTPUT_MODE", "USE_META_OVERLAY", "CONF_THRESHOLD", "TP_ATR_MULT", "SL_ATR_MULT", "MAX_HOLD_MINUTES", "RISK_PER_TRADE"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in wanted or node.targets[0].id.startswith("DEAL_REASON_"):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    fee_defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 3:
            if isinstance(node.args[1], ast.Constant) and node.args[1].value == "fee":
                try:
                    fee_defaults.append(ast.literal_eval(node.args[2]))
                except (ValueError, TypeError):
                    pass
    return constants, fee_defaults


def evidence_inventory(args, v2_deals):
    artifacts, findings, versions = [], [], []
    retained, git_sources = {}, []
    roles = {"account_context_manifest": "context_manifest", "account_context_seal": "context_seal",
             "archived_forward_deals": "archive_deals", "archived_forward_deals_seal": "archive_seal",
             "local_mt5_enum_source_read_not_imported": "enum_source", "runtime_log_context_only": "audit_log",
             "unlinked_workbook_headers_only": "workbook"}
    if args.retained_evidence:
        if args.git_revision or any(getattr(args, name) for name in roles.values()):
            raise ValueError("Retained evidence cannot be mixed with fresh evidence")
        for specification in args.retained_evidence:
            item = json.loads(specification)
            path = safe_path(item["path"])
            if digest(path.read_bytes()) != item["sha256"]:
                raise ValueError("Retained evidence seal mismatch")
            retained[str(path)] = item
            if item["role"] == "source_history_only":
                git_sources.append((item["revision"], path.read_bytes()))
            else:
                field = roles[item["role"]]
                if field in {"audit_log", "workbook"}:
                    getattr(args, field).append(str(path))
                else:
                    if getattr(args, field):
                        raise ValueError("Duplicate retained singleton role")
                    setattr(args, field, str(path))
    def retain(path, role):
        path = safe_path(path)
        data = path.read_bytes()
        prior = retained.get(str(path), {})
        name = prior.get("snapshot", f"evidence/{len(artifacts):03d}_{path.name}")
        if Path(name).parent.as_posix() != "evidence":
            raise ValueError("Invalid retained evidence path")
        artifacts.append({"path": str(path), "snapshot": name, "role": role,
                          "original_name": prior.get("original_name", path.name), "sha256": digest(data), "data": data})
        return data
    if bool(args.context_manifest) != bool(args.context_seal):
        raise ValueError("Context manifest and seal must be supplied together")
    if args.context_manifest:
        data = retain(args.context_manifest, "account_context_manifest")
        seal_data = retain(args.context_seal, "account_context_seal")
        manifest, seal = json.loads(data), json.loads(seal_data)
        expected = seal["file_sha256"].get("manifest.json")
        if digest(data) != expected:
            raise ValueError("Account-context manifest seal mismatch")
        fetch = manifest.get("data", {}).get("mt5_fetch", {})
        findings.append(conclusion("Confirmed", "封存 manifest 的資料擷取帳戶情境與 seal 相符，但不是逐筆 Forward 成交身分證明。", 1,
                                   "account_context_manifest", "不得將此擷取情境套用到126筆交易；Demo trade_mode亦未在broker_info中記錄。"))
        findings[-1]["observed_broker_info"] = fetch.get("broker_info")
        findings[-1]["retrieved_at_utc"] = fetch.get("retrieved_at_utc")
        findings[-1]["trade_identity_coverage_added"] = 0
    if bool(args.archive_deals) != bool(args.archive_seal):
        raise ValueError("Archive deals and seal must be supplied together")
    if args.archive_deals:
        data = retain(args.archive_deals, "archived_forward_deals")
        seal_data = retain(args.archive_seal, "archived_forward_deals_seal")
        seal = json.loads(seal_data)["file_sha256"]
        expected = seal.get("input_snapshots/"+artifacts[-2]["original_name"])
        if digest(data) != expected:
            raise ValueError("Archived Forward deals seal mismatch")
        rows, rejected = parse_snapshot(data, Path(args.archive_deals).suffix)
        current = {str(r.get("deal_id")): r for r in v2_deals}
        matched, equal = 0, 0
        for row in rows:
            other = current.get(str(row.get("deal_id")))
            if other:
                matched += 1
                equal += all(Decimal(str(row[k])) == Decimal(str(other[k])) for k in ("profit", "commission", "swap", "fee", "volume"))
        findings.append(conclusion("Confirmed", "歷史封存成交與V2相同Deal ID的成本／數量欄位可逐筆核對。", matched,
                                   "archived_forward_deals", "重複封存不是獨立broker statement，不能證明費用完整或交易帳戶。"))
        findings[-1].update({"matched_deals": matched, "equal_cost_volume_deals": equal, "rejected": rejected})
    for revision in args.git_revision:
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
            raise ValueError("Use an explicit hexadecimal Git revision")
        data = subprocess.check_output(["git", "show", revision+":gemini.py"], cwd=ROOT)
        git_sources.append((revision, data))
    for revision, data in git_sources:
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
            raise ValueError("Invalid retained revision")
        constants, defaults = source_constants(data)
        artifacts.append({"path": "git:"+revision+":gemini.py", "snapshot": "evidence/git_"+revision+"_gemini.py",
                          "role": "source_history_only", "revision": revision,
                          "original_name": "gemini.py", "sha256": digest(data), "data": data})
        versions.append({"revision": revision, **constants, "fee_getattr_defaults": defaults,
                         "label": "Confirmed", "trade_assignment": "Unknown",
                         "limitation": "Git content is source history, not a running-process/deal deployment receipt."})
    enums = {}
    if args.enum_source:
        data = retain(args.enum_source, "local_mt5_enum_source_read_not_imported")
        enums, _ = source_constants(data)
        enums = {k: v for k, v in enums.items() if k.startswith("DEAL_REASON_")}
    for name in args.audit_log:
        data = retain(name, "runtime_log_context_only")
        text = data.decode("utf-8-sig", errors="replace")
        findings.append(conclusion("Confirmed", "已保留runtime紀錄供核對；僅PID、路徑或模型檔名不足以證明歷史逐筆設定。", len(text.splitlines()),
                                   artifacts[-1]["snapshot"], "不由啟動日期或交易結果反推門檻。"))
    for name in args.workbook:
        data = retain(name, "unlinked_workbook_headers_only")
        headers = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for item in ("xl/sharedStrings.xml", "xl/worksheets/sheet1.xml"):
                if item in archive.namelist():
                    headers.extend(n.text for n in ET.fromstring(archive.read(item)).iter() if n.tag.endswith("}t") and n.text)
        fields = headers[:12]
        findings.append(conclusion("Insufficient Evidence", "檢查既有工作簿表頭，未形成與本批Deal/Position的可信關聯。", 0,
                                   artifacts[-1]["snapshot"], "沒有逐筆ID／帳戶綁定時，不能依Symbol或Profit($)認定來源或Currency。"))
        findings[-1]["header_excerpt"] = fields
    return artifacts, findings, versions, enums


def grouped_financial(trades, key_function, cohort):
    grouped = defaultdict(list)
    for trade in trades:
        grouped[key_function(trade)].append(trade)
    return [{"cohort": cohort, "group": key, **financial(grouped[key]), "label": "Confirmed",
             "statistical_assessment": "Insufficient Evidence" if len(grouped[key]) < 20 else "Descriptive only"}
            for key in sorted(grouped)]


def diagnose(v2, records, enums, costs, repetitions=2000, seed=42):
    complete = [t for t in v2["ledger"]["trades"] if t["accounting_state"] == "complete"]
    cohorts, period_rows, cost_rows, attribution_rows, exit_rows = {}, [], [], [], []
    for trade in v2["ledger"]["trades"]:
        attribution_rows.append({k: trade.get(k) for k in ("position_id", "time_basis", "account_login", "account_server", "currency", "model_sha256", "conf_threshold", "git_commit", "verified_run_id", "recorded_buy_score", "signal_link")})
        exit_rows.append(classify_exit(trade, records["deals"], enums))
    for cohort in (LEGACY_CLOCK, "UTC"):
        trades = [t for t in complete if t["time_basis"] == cohort]
        if cohort not in v2["windows"]:
            continue
        start, end = [time_value(x) for x in v2["windows"][cohort]]
        trades = [t for t in trades if start <= time_value(t["accounting_exit_time"]) < end]
        session = lambda t: "Unknown_market_timezone" if cohort == LEGACY_CLOCK else f"recorded_UTC_{time_value(t['accounting_entry_time']).hour//8*8:02d}_{time_value(t['accounting_entry_time']).hour//8*8+8:02d}"
        monthly = grouped_financial(trades, lambda t: t["accounting_exit_time"][:7], cohort)
        sessions = grouped_financial(trades, session, cohort)
        session_months = grouped_financial(trades, lambda t: t["accounting_exit_time"][:7]+":"+session(t), cohort)
        score_rows = []
        for band in ("below_0.70", "0.70_to_below_0.75", "0.75_to_1.00", "Unknown"):
            def band_for(t):
                score = t.get("recorded_buy_score")
                return "Unknown" if score is None else "below_0.70" if score < .70 else "0.70_to_below_0.75" if score < .75 else "0.75_to_1.00"
            selected = [t for t in trades if band_for(t) == band]
            score_rows.append({"cohort": cohort, "group": band, **financial(selected),
                               "version_verified_n": sum(t["verified_run_id"] != "Unknown" for t in selected),
                               "label": "Insufficient Evidence", "limitation": "No threshold comparison without reliable model/config attribution and sufficient fills."})
        partition = next((r for r in v2["time_partitions"] if r["time_basis"] == cohort), {})
        risks = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
        cohorts[cohort] = {
            "label": "Confirmed", "scope": "Recorded file values only, not a verified strategy version",
            "metrics": financial(trades), "window": v2["windows"][cohort],
            "realized_max_drawdown": partition.get("realized_max_drawdown"),
            "drawdown_cash_event_n": partition.get("realized_max_drawdown_n"),
            "max_losing_streak": partition.get("max_losing_streak"),
            "trades_per_day": partition.get("trades_per_day"),
            "calendar_days": (end-start).total_seconds()/86400,
            "days_with_exit_records": len({t["accounting_exit_time"][:10] for t in trades}),
            "frequency_limitation": "Zero-record days are not independently verified zero-trading days.",
            "mean_r": {"value": statistics.mean(risks) if risks else None, "n": len(risks),
                       "label": "Confirmed" if risks else "Insufficient Evidence",
                       "limitation": "Only complete initial-risk receipts from V2; per-lot PnL is not substituted for R."},
            "loss_concentration": concentration(trades), "holding_relationship": holding_relationship(trades),
            "weekly_block_bootstrap": week_bootstrap(trades, v2["windows"][cohort], repetitions, seed),
            "monthly": monthly, "sessions": sessions, "session_months": session_months, "score_bands": score_rows,
            "holding_bands": grouped_financial(trades, lambda t: "Unknown" if t.get("holding_minutes") is None else "00_up_to_15m" if t["holding_minutes"] <= 15 else "01_15_to_60m" if t["holding_minutes"] <= 60 else "02_60_to_90m" if t["holding_minutes"] <= 90 else "03_over_90m", cohort),
            "exit_categories": dict(Counter(r["exit_category"] for r in exit_rows if r["time_basis"] == cohort)),
            "volume_range": [min((t["entry_volume"] for t in trades), default=None), max((t["entry_volume"] for t in trades), default=None)],
        }
        period_rows.extend({"period_kind": "month", **row} for row in monthly)
        for kind, key in (("day", lambda t: t["accounting_exit_time"][:10]),
                          ("week", lambda t: "%04d-W%02d" % time_value(t["accounting_exit_time"]).isocalendar()[:2])):
            period_rows.extend({"period_kind": kind, **row} for row in grouped_financial(trades, key, cohort))
        for cost in costs:
            cost_rows.append({"cohort": cohort, **financial(trades, cost),
                              "label": "Confirmed", "scenario": "recorded_baseline" if cost == 0 else "hypothetical_additional_cost",
                              "limitation": "Fixed extra recorded-file units per CLOSED trade, not an observed broker fee or a spread deduction; trade identities unchanged."})
    coverage = {"n_trades": len(v2["ledger"]["trades"]),
                "account_verified_n": sum(t["account_login"] != "Unknown" and t["account_server"] != "Unknown" for t in complete),
                "currency_recorded_n": sum(bool(t["currency"]) for t in complete),
                "model_version_verified_n": sum(t["verified_run_id"] != "Unknown" for t in complete),
                "threshold_070_verified_n": sum(t["verified_run_id"] != "Unknown" and t["conf_threshold"] == .70 for t in complete),
                "threshold_075_verified_n": sum(t["verified_run_id"] != "Unknown" and t["conf_threshold"] == .75 for t in complete),
                "score_linked_n": sum(t["recorded_buy_score"] is not None for t in complete),
                "initial_risk_valid_n": sum(t["r_multiple"] is not None for t in complete),
                "explicit_cost_fields_complete_deal_n": sum(all(r.get(k) not in (None, "") for k in ("commission", "swap", "fee")) for r in records["deals"]),
                "broker_cost_completeness_verified_n": 0,
                "limitation": "Field presence and mathematically balanced positions are not external broker/account verification."}
    # Conclusions state observable problems; no slice is promoted to a filter.
    top = [
        conclusion("Confirmed", "勝率須連同實際盈虧比與損益兩平勝率解讀；各時間分組提供完整payoff與每筆期望，不能把高勝率直接當作交易優勢。", len(complete),
                   "cohorts.*.metrics", "依記錄成本；未核對外部費用與模型版本。"),
        conclusion("Confirmed", "早期與UTC分組的獲利結構必須分開，合併總數不能代表現行0.70或0.75策略。", len(complete),
                   "cohorts and coverage", "分組差異可能涉及版本、市況、風險額度或資料情境，尚無因果歸屬。"),
        conclusion("Insufficient Evidence", "帳戶、版本及完整費用證據不足，無法確認可持續成本後正期望；固定追加成本情境僅揭露脆弱度。", len(complete),
                   "coverage and cost_sensitivity", "不把假設費用當真實費用，不以診斷資料宣稱untouched OOS。"),
    ]
    if "UTC" in cohorts and cohorts["UTC"]["metrics"]["n"]:
        metrics = cohorts["UTC"]["metrics"]
        top[0] = conclusion("Confirmed", f"UTC段記錄勝率{metrics['win_rate']:.2%}、損益兩平勝率{metrics['break_even_win_rate']:.2%}；平均獲利{metrics['average_win']:.4f}、平均虧損絕對值{abs(metrics['average_loss']):.4f}，每筆記錄期望{metrics['net_expectancy_per_trade']:.4f}，PF={metrics['net_pf']:.4f}。", metrics["n"],
                            "cohorts.UTC.metrics", "這是該批記錄的代數分解，不證明現行版本的未來期望；不確定性請見分組bootstrap，非獨立績效驗證。") if metrics["average_win"] and metrics["average_loss"] else top[0]
    top[1]["observed_cohort_net_pnl"] = {k: c["metrics"]["net_pnl"] for k, c in cohorts.items()}
    top[2]["coverage"] = {k: coverage[k] for k in ("account_verified_n", "currency_recorded_n", "model_version_verified_n", "broker_cost_completeness_verified_n")}
    hypothesis = conclusion("Supported Hypothesis", "在同一個已凍結且可追溯的Demo模型版本中，檢驗成本後平均獲利／虧損比是否足以覆蓋損益兩平勝率；先確認記錄改善後是否仍呈現UTC段的負期望。", len(complete),
                            "cohorts.*.metrics and coverage", "下一階段只預先固定資料契約、模型hash、原門檻與費用核對，再收集新Forward；不改Session、分數門檻或出場條件，不在本次啟動。")
    return {"schema_version": 3, "cohorts": cohorts, "cost_sensitivity": cost_rows, "periods": period_rows,
            "v2_integrity": v2["integrity"],
            "coverage": coverage, "trade_attribution": attribution_rows, "exit_classification": exit_rows,
            "recorded_signal_layers": signal_layers(records["signals"]), "top_three_problems": top,
            "next_hypothesis": hypothesis,
            "pooled_strategy_claim": conclusion("Insufficient Evidence", "不把126筆跨時間／未知版本交易當成單一策略的獨立績效驗證。", len(complete), "V2 sealed ledger", "僅唯讀描述統計。"),
            "method": {"costs": "Fixed preregistered descriptive sensitivity, not parameter selection or a simulator",
                       "bootstrap": "ISO calendar-week cluster bootstrap, fixed seed; within-week dependence retained, cross-week ordering lost",
                       "sessions": "Recorded UTC 8-hour buckets, not a certified Asia/London/New York mapping",
                       "holding": "Outcome-conditioned association only, no recommended exit rule",
                       "hypothesis_generation": "All slices inspected; not untouched OOS; no production filter selected"}}


def render(report):
    lines = ["# XM V3 — Forward Trading Edge Diagnosis", "",
             "本報告使用已封存V2的同一批Forward交易，不增加樣本、不訓練／回測／搜尋。所有金額為未驗證帳戶幣別的檔案記錄單位。", "",
             "**Confirmed = 可由保留記錄核對；Supported Hypothesis = 有資料支持但未證實的假說；Insufficient Evidence = 不足以歸因或驗證。**", "",
             "## 身分、版本與成本證據", "", "```json", json.dumps(report["coverage"], ensure_ascii=False, indent=2), "```", ""]
    for item in report.get("evidence_findings", []):
        lines.append(f"- **{item['label']}**：{item['text']} N={item['n']}；來源={item['source']}。{item['limitation']}")
        details = {k: v for k, v in item.items() if k not in {"label", "text", "n", "source", "limitation"}}
        if details:
            lines.extend(["", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```", ""])
    if report.get("source_versions"):
        lines.extend(["", "### Git設定差異（Confirmed：僅來源歷史；成交歸屬Unknown）", "",
                      markdown_table(report["source_versions"], ["revision", "MODEL_FILE", "CONF_THRESHOLD", "USE_META_OVERLAY", "TP_ATR_MULT", "SL_ATR_MULT", "MAX_HOLD_MINUTES", "RISK_PER_TRADE", "fee_getattr_defaults", "trade_assignment"]), ""])
    lines.extend(["", "Git來源內容只列於source_versions.csv，不能視為runtime部署證明。費用欄全為0的外部原因仍Unknown；舊來源的getattr預設0只證明可能的記錄機制。", "",
                  "### V2記錄內完整性（Confirmed）", "", "```json", json.dumps(report["v2_integrity"], ensure_ascii=False, indent=2), "```", ""])
    for name, cohort in report["cohorts"].items():
        metrics = cohort["metrics"]
        lines.extend(["## "+name, "", "**Confirmed（記錄內描述；版本與幣別未驗證）**", "",
                      markdown_table([metrics], ["n", "n_wins", "n_losses", "gross_profit", "gross_loss", "net_pnl", "net_pf", "win_rate", "break_even_win_rate", "wr_minus_break_even", "average_win", "average_loss", "payoff_ratio", "net_expectancy_per_trade"]), "",
                      f"記錄內Commission／Swap／Fee合計={metrics['recorded_explicit_cost']}，占正broker profit比={metrics['recorded_cost_to_positive_broker_profit']}；完整外部成本未驗證。", "",
                      f"已實現現金回撤={cohort['realized_max_drawdown']}（cash events N={cohort['drawdown_cash_event_n']}）；最大連敗={cohort['max_losing_streak']}；記錄交易/日={cohort['trades_per_day']}。", "",
                      f"完整觀測窗={cohort['calendar_days']}日；有出場紀錄={cohort['days_with_exit_records']}日。無紀錄日不等於已確認未交易。Mean-R={cohort['mean_r']['value']}，N={cohort['mean_r']['n']}。", "",
                      "### 虧損集中度（Confirmed）", "", markdown_table(cohort["loss_concentration"], ["top_k", "n_losses_in_slice", "n_all_losses", "loss_amount", "share_of_gross_loss"]), "",
                      "### 持倉關係（非因果）", "", "```json", json.dumps(cohort["holding_relationship"], ensure_ascii=False, indent=2), "```", "",
                      markdown_table(cohort["holding_bands"], ["group", "n", "net_pnl", "net_pf", "win_rate"]), "",
                      "### 週區塊Bootstrap與限制", "", "```json", json.dumps(cohort["weekly_block_bootstrap"], ensure_ascii=False, indent=2), "```", "",
                      "### 月份／Session／分數帶", "", markdown_table(cohort["monthly"], ["group", "n", "net_pnl", "net_pf", "win_rate", "statistical_assessment"]), "",
                      markdown_table(cohort["sessions"], ["group", "n", "net_pnl", "net_pf", "payoff_ratio", "statistical_assessment"]), "",
                      "Session跨月結果見session_months.csv。月份較佳不代表穩定或可增設過濾條件。", "",
                      markdown_table(cohort["score_bands"], ["group", "n", "net_pnl", "net_pf", "version_verified_n", "label"]), "",
                      "出場分類="+json.dumps(cohort["exit_categories"], ensure_ascii=False)+"；只證明記錄reason，不以持倉分鐘反推Timeout。", ""])
    lines.extend(["## 假設成本敏感度（與實際記錄分開）", "",
                  "0為記錄基準；其他情境是每筆完整平倉額外扣固定檔案單位，並非估計broker費用，不重複扣spread，也不改交易路徑。", "",
                  markdown_table(report["cost_sensitivity"], ["cohort", "scenario", "additional_cost_per_closed_trade", "n", "net_pnl", "net_pf", "net_expectancy_per_trade"]), "",
                  "## 頻率限制層", "", "```json", json.dumps(report["recorded_signal_layers"], ensure_ascii=False, indent=2), "```", "",
                  "**Insufficient Evidence**：訊號列不是獨立事件。entry_mode與Session等條件重疊且版本未知，只能確認記錄在哪些狀態停止，不能算出放寬後可新增多少交易。", "",
                  "## 最有證據的前三項問題", ""])
    for i, item in enumerate(report["top_three_problems"], 1):
        lines.append(f"{i}. **{item['label']}** — {item['text']} N={item['n']}；來源={item['source']}。{item['limitation']}")
    hypothesis = report["next_hypothesis"]
    lines.extend(["", "## 下一階段的一個假說", "", f"**{hypothesis['label']}** — {hypothesis['text']} N={hypothesis['n']}。{hypothesis['limitation']}", "",
                  "## 重現與範圍", "", "輸入V2 seals／精確快照、補充證據、程式快照、SHA256、seed與重現命令見manifest.json。", "已查看資料不重新宣稱untouched OOS；本機封存未宣稱遠端保存。", ""])
    return "\n".join(lines)


def run(args):
    v2, records, v2_manifest, inventory = load_v2(args.v2_dir)
    costs = sorted({float(x) for x in args.costs.split(",")})
    if not costs or costs[0] != 0 or any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("Costs must include baseline 0 and only nonnegative finite scenarios")
    artifacts, findings, versions, enums = evidence_inventory(args, records["deals"])
    report = diagnose(v2, records, enums, costs, args.bootstrap, args.seed)
    report.update({"v2_identity": {"input_inventory": inventory,
                                   "manifest_sha256": digest(safe_child(args.v2_dir, "manifest.json").read_bytes()),
                                   "report_sha256": v2_manifest["output_sha256"]["report.json"]},
                   "evidence_findings": findings, "source_versions": versions})
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"source").mkdir(); (output/"evidence").mkdir()
    # Retain a compact, fully sealed V2 copy: no market series or historical model data.
    retained = output/"v2"; retained.mkdir(); (retained/"inputs").mkdir(); (retained/"source").mkdir()
    v2_root = safe_path(args.v2_dir)
    names = ["manifest.json", *v2_manifest["output_sha256"],
             *[item["snapshot"] for item in v2_manifest["inputs"]],
             *["source/"+name for name in v2_manifest["source_sha256"]]]
    for name in names:
        target = retained/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(safe_child(v2_root, name).read_bytes())
    for item in artifacts:
        with (output/item["snapshot"]).open("xb") as handle:
            handle.write(item["data"])
    source_hashes = {}
    for name in ("forward_edge_diagnosis_v3.py", "forward_test_report_v2.py", "forward_trading_ledger_v1.py", "demo_runtime_foundation_v1.py", "training_holdout_guard_v1.py"):
        data = (ROOT/name).read_bytes()
        with (output/"source"/name).open("xb") as handle:
            handle.write(data)
        source_hashes[name] = digest(data)
    with (output/"report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(render(report))
    tables = {"period_diagnosis": report["periods"], "cost_sensitivity": report["cost_sensitivity"],
              "trade_attribution": report["trade_attribution"], "exit_classification": report["exit_classification"],
              "source_versions": versions, "coverage": [report["coverage"]],
              "conclusions": report["top_three_problems"]+[report["next_hypothesis"]]}
    for key in ("session_months", "score_bands", "loss_concentration", "holding_bands"):
        tables[key] = [{"cohort": cohort, **row} for cohort, values in report["cohorts"].items() for row in values[key]]
    for name, rows in tables.items():
        write_csv(output/(name+".csv"), rows)
    replay = ["python", "-B", "source/forward_edge_diagnosis_v3.py", "--v2-dir", "v2", "--costs", args.costs,
              "--bootstrap", str(args.bootstrap), "--seed", str(args.seed), "--output", "reproduced"]
    # Git snapshots are read from retained files during reproduction, not from a changing repository.
    for item in artifacts:
        specification = {k: item[k] for k in ("snapshot", "role", "original_name", "sha256")}
        specification["path"] = item["snapshot"]
        if "revision" in item:
            specification["revision"] = item["revision"]
        replay.extend(["--retained-evidence", json.dumps(specification, ensure_ascii=False)])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    manifest = {"schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version, "git_commit": commit, "git_dirty": dirty,
                "source_sha256": source_hashes, "v2_identity": report["v2_identity"],
                "evidence": [{k: v for k, v in item.items() if k != "data"} for item in artifacts],
                "cost_scenarios": costs, "bootstrap_repetitions": args.bootstrap, "bootstrap_seed": args.seed,
                "reproduce_argv": replay,
                "output_sha256": {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()},
                "authorization": "User-authorized read-only Forward edge diagnosis; no training/simulation/promotion"}
    with (output/"manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--costs", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    for name in ("context-manifest", "context-seal", "archive-deals", "archive-seal", "enum-source"):
        parser.add_argument("--"+name)
    for name in ("git-revision", "audit-log", "workbook", "retained-evidence"):
        parser.add_argument("--"+name, action="append", default=[])
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"output": args.output, "coverage": report["coverage"],
                      "metrics": {k: v["metrics"] for k, v in report["cohorts"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
