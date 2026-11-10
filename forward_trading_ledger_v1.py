"""Pure forward-deal accounting. No MT5, model loading, replay, fitting or data defaults."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


def identifier(value):
    text = str(value or "").strip()
    return text if text and text != "0" else None


def number(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Missing/invalid numeric field") from None
    if not result.is_finite():
        raise ValueError("Non-finite numeric field")
    return result


def utc_time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone Unknown; naive historical time is not assumed UTC")
    return result.astimezone(timezone.utc)


def scope(row):
    return (str(row.get("account_login") or "Unknown"),
            str(row.get("account_server") or "Unknown"))


def order_key(row, order_id=None):
    return (*scope(row), identifier(order_id if order_id is not None else row.get("order_id")))


def build_ledger(deals, orders=(), signals=()):
    """Link only explicit identifiers. capture_run_id never supplies trade-origin identity."""
    groups = defaultdict(list)
    issues = defaultdict(set)
    order_positions = defaultdict(set)
    order_metadata = defaultdict(list)
    for row in orders:
        if row.get("stage") == "result" and identifier(row.get("order_id")):
            order_metadata[order_key(row)].append(row)
            position = identifier(row.get("request", {}).get("position"))
            if position:
                order_positions[order_key(row)].add(position)
    for row in signals:
        if identifier(row.get("order_id")) and identifier(row.get("position_ticket")):
            order_positions[order_key(row)].add(identifier(row["position_ticket"]))
    deals = [dict(row) for row in deals]
    for row in deals:
        if identifier(row.get("order_id")) and identifier(row.get("position_id")):
            order_positions[order_key(row)].add(identifier(row["position_id"]))
    seen = {}
    duplicate_count = 0
    for index, row in enumerate(deals):
        position = identifier(row.get("position_id"))
        if position is None:
            candidates = order_positions.get(order_key(row), set())
            if len(candidates) == 1:
                position = next(iter(candidates))
        key = (*scope(row), position or f"Unknown-row-{index}")
        if not identifier(row.get("deal_id")):
            issues[key].add("missing_deal_id")
        else:
            deal_key = (*scope(row), identifier(row["deal_id"]))
            # Observation timestamps/run IDs are not broker deal identity.
            broker_fields = ("order_id", "position_id", "deal_time", "symbol", "magic",
                             "type", "entry", "volume", "price", "profit", "commission", "swap", "fee")
            fingerprint = []
            for field in broker_fields:
                value = row.get(field, "")
                if field in {"volume", "price", "profit", "commission", "swap", "fee", "entry", "type"}:
                    try:
                        value = str(number(value).normalize())
                    except ValueError:
                        value = repr(value)
                elif field == "position_id":
                    value = position
                fingerprint.append(str(value))
            fingerprint = tuple(fingerprint)
            if deal_key in seen:
                previous_key, previous_fingerprint = seen[deal_key]
                duplicate_count += 1
                if fingerprint != previous_fingerprint:
                    issues[key].add("conflicting_duplicate_deal")
                    issues[previous_key].add("conflicting_duplicate_deal")
                continue
            seen[deal_key] = (key, fingerprint)
        if position is None:
            issues[key].add("missing_or_ambiguous_position_id")
        groups[key].append(row)

    trades = []
    cash_events = []
    observed_cash_events = []
    for key, rows in groups.items():
        reasons = set(issues[key])
        if "Unknown" in key[:2]:
            reasons.add("account_scope_unknown")
        symbols = {r.get("symbol") for r in rows}
        currencies = {r.get("currency") for r in rows}
        if len(symbols) != 1 or not next(iter(symbols)):
            reasons.add("symbol_missing_or_mixed")
        if len(currencies) != 1:
            reasons.add("currency_mixed")
        elif not next(iter(currencies)):
            reasons.add("currency_missing")
        parsed = []
        for row in rows:
            try:
                time = datetime.fromisoformat(str(row["deal_time"]).replace("Z", "+00:00"))
                if time.tzinfo is None:
                    reasons.add("timezone_unknown")
                else:
                    time = time.astimezone(timezone.utc)
                fields = {k: number(row.get(k)) for k in
                          ("volume", "price", "profit", "commission", "swap", "fee")}
                entry, side = number(row["entry"]), number(row["type"])
                if entry not in (0, 1, 3) or side not in (0, 1):
                    raise ValueError("Unsupported reversal or non-trading deal")
                if fields["volume"] <= 0 or fields["price"] <= 0:
                    raise ValueError("Invalid volume/price")
                stamp = row.get("time_msc")
                if stamp not in (None, ""):
                    stamp = number(stamp)
                    if stamp < 0 or stamp != stamp.to_integral_value():
                        raise ValueError("Invalid millisecond timestamp")
                    precise_time = datetime.fromtimestamp(float(stamp / 1000), timezone.utc)
                    if time.tzinfo is None:
                        raise ValueError("Cannot verify time_msc against naive deal_time")
                    if abs((precise_time - time).total_seconds()) >= 1:
                        raise ValueError("Conflicting deal_time/time_msc")
                    time = precise_time
                if not identifier(row.get("order_id")):
                    reasons.add("missing_order_id")
                parsed.append((time, int(entry), int(side), fields, row))
            except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                reasons.add("invalid_deal:" + str(exc))
        if len({r[0].tzinfo is None for r in parsed}) > 1:
            reasons.add("mixed_timestamp_context")
        parsed.sort(key=lambda r: (r[0].tzinfo is not None, r[0], int(r[4].get("time_msc") or 0), r[1]))
        for time in {r[0] for r in parsed}:
            tied = [r for r in parsed if r[0] == time]
            if any(r[1] == 0 for r in tied) and any(r[1] != 0 for r in tied):
                reasons.add("entry_exit_order_unknown_within_timestamp")
        balance = Decimal(0)
        entry_volume = exit_volume = Decimal(0)
        totals = {k: Decimal(0) for k in ("profit", "commission", "swap", "fee")}
        direction = None
        for _, entry, side, fields, _ in parsed:
            if entry == 0:
                if direction is None:
                    direction = side
                if direction != side:
                    reasons.add("mixed_entry_direction")
                balance += fields["volume"]
                entry_volume += fields["volume"]
            else:
                if direction is None or direction == side:
                    reasons.add("exit_without_matching_entry")
                balance -= fields["volume"]
                exit_volume += fields["volume"]
            if balance < 0:
                reasons.add("exit_exceeds_observed_entry")
            for field in totals:
                totals[field] += fields[field]
        # A reused position lifecycle cannot be reliably assigned one initial risk/run.
        closed_once = False
        running = Decimal(0)
        for _, entry, _, fields, _ in parsed:
            if closed_once and entry == 0:
                reasons.add("reopened_position_id")
            running += fields["volume"] if entry == 0 else -fields["volume"]
            closed_once |= running == 0
        if entry_volume == 0:
            reasons.add("missing_entry_history")
        state = "Unknown" if reasons else "complete" if balance == 0 else "open"
        context_issues = {"account_scope_unknown", "currency_missing", "timezone_unknown"}
        accounting_issues = reasons - context_issues
        accounting_state = "Unknown" if accounting_issues else "complete" if balance == 0 else "open"
        if accounting_state != "Unknown":
            for when, _, _, fields, row in parsed:
                observed_cash_events.append({
                    "time": when.isoformat(), "account_login": key[0],
                    "account_server": key[1], "currency": row.get("currency"),
                    "position_id": key[2], "net": float(sum(fields[k] for k in totals)),
                    "time_basis": "recorded_clock_unknown_timezone" if when.tzinfo is None else "UTC",
                    "context_issues": sorted(reasons & context_issues),
                })
        if state != "Unknown":
            for when, _, _, fields, row in parsed:
                cash_events.append({
                    "time": when.isoformat(), "account_login": key[0],
                    "account_server": key[1], "currency": row["currency"],
                    "net": float(sum(fields[k] for k in totals)),
                })
        cash_pnl = sum(totals.values()) if len(parsed) == len(rows) else None
        metadata = []
        entry_orders = defaultdict(Decimal)
        for _, entry, _, fields, row in parsed:
            if entry == 0:
                entry_orders[order_key(row)] += fields["volume"]
        risk = Decimal(0)
        risk_known = bool(entry_orders)
        run_ids = set()
        versions = set()
        for order, volume in entry_orders.items():
            candidates = order_metadata.get(order, [])
            unique = {json.dumps(r, sort_keys=True, default=str): r for r in candidates}
            if len(unique) != 1:
                risk_known = False
                continue
            item = next(iter(unique.values()))
            metadata.append(item)
            if item.get("run_id"):
                run_ids.add(item["run_id"])
            if item.get("model_sha256") and item.get("configuration_sha256"):
                versions.add((item["model_sha256"], item["configuration_sha256"]))
            try:
                captured_risk = number(item.get("initial_risk"))
                if captured_risk <= 0 or number(item.get("filled_volume")) != volume:
                    raise ValueError("Incomplete risk volume")
                risk += captured_risk
            except ValueError:
                risk_known = False
        origin_known = len(metadata) == len(entry_orders) and bool(entry_orders)
        costs = -(totals["commission"] + totals["swap"] + totals["fee"])
        trades.append({
            "account_login": key[0], "account_server": key[1], "position_id": key[2],
            "symbol": next(iter(symbols)) if len(symbols) == 1 else None,
            "currency": next(iter(currencies)) if len(currencies) == 1 else None,
            "state": state, "issues": sorted(reasons), "deal_count": len(rows),
            "accounting_state": accounting_state, "accounting_issues": sorted(accounting_issues),
            "deal_ids": sorted(str(r.get("deal_id", "")) for r in rows),
            "order_ids": sorted({str(r.get("order_id", "")) for r in rows}),
            "entry_time": parsed[0][0].isoformat() if parsed else None,
            "exit_time": parsed[-1][0].isoformat() if state == "complete" else None,
            "accounting_entry_time": parsed[0][0].isoformat() if parsed else None,
            "accounting_exit_time": parsed[-1][0].isoformat() if accounting_state == "complete" else None,
            "time_basis": ("recorded_clock_unknown_timezone" if parsed[0][0].tzinfo is None else "UTC") if parsed and "mixed_timestamp_context" not in reasons else "Unknown",
            "entry_volume": float(entry_volume), "exit_volume": float(exit_volume),
            "remaining_volume": float(balance),
            "observed_net_pnl": float(cash_pnl) if cash_pnl is not None else None,
            "net_pnl": float(cash_pnl) if state == "complete" and cash_pnl is not None else None,
            "accounting_net_pnl": float(cash_pnl) if accounting_state == "complete" and cash_pnl is not None else None,
            **{k: float(v) for k, v in totals.items()},
            "explicit_cost": float(costs) if cash_pnl is not None else None,
            "spread_deducted_again": False,
            "initial_risk": float(risk) if risk_known else None,
            "r_multiple": float(cash_pnl / risk) if state == "complete" and risk_known and risk > 0 else None,
            "run_ids": sorted(run_ids) if origin_known else [],
            "version_pairs": sorted(versions) if origin_known else [],
            "provenance": "known" if origin_known and len(run_ids) == 1 and len(versions) == 1 else "Unknown",
        })
    return {"trades": trades, "cash_events": cash_events,
            "observed_cash_events": observed_cash_events, "duplicate_deals": duplicate_count,
            "input_deal_rows": len(deals)}


def calculate_metrics(ledger, start, end):
    """Realized cash metrics only, using an explicit UTC observation window."""
    start, end = utc_time(start), utc_time(end)
    if end <= start:
        raise ValueError("Observation end must be after start")
    all_trades = ledger["trades"]
    complete = [t for t in all_trades if t["state"] == "complete"
                and start <= utc_time(t["exit_time"]) < end]
    cash = [e for e in ledger["cash_events"] if start <= utc_time(e["time"]) < end]
    units = {(t["account_login"], t["account_server"], t["currency"]) for t in [*complete, *cash]}
    if len(units) > 1:
        raise ValueError("Metrics require one account/server/currency; partition ledger first")
    complete.sort(key=lambda t: (t["exit_time"], t["position_id"]))
    pnl = [t["net_pnl"] for t in complete]
    wins, losses = [v for v in pnl if v > 0], [v for v in pnl if v < 0]
    gross_win, gross_loss = math.fsum(wins), -math.fsum(losses)
    # Exits with identical timestamps form one realized cash event; no invented order.
    by_time = defaultdict(list)
    for event in cash:
        by_time[event["time"]].append(event["net"])
    equity = peak = drawdown = 0.0
    for when in sorted(by_time):
        equity += math.fsum(by_time[when])
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    r_values = [t["r_multiple"] for t in complete if t["r_multiple"] is not None]
    pf = gross_win / gross_loss if gross_loss else None
    return {
        "window_start": start.isoformat(), "window_end_exclusive": end.isoformat(),
        "day_denominator": "elapsed_calendar_days", "trades": len(complete),
        "net_pnl": math.fsum(pnl), "win_rate": len(wins) / len(pnl) if pnl else None,
        "profit_factor": pf, "profit_factor_status": "finite" if gross_loss else "infinite" if gross_win else "undefined",
        "mean_r": math.fsum(r_values) / len(r_values) if r_values else None,
        "mean_r_samples": len(r_values), "missing_risk_samples": len(complete) - len(r_values),
        "realized_max_drawdown": drawdown, "drawdown_unit": "account_currency",
        "drawdown_basis": "observed_deal_cash_events_in_window_including_open_positions_costs",
        "observed_realized_cash_pnl": math.fsum(e["net"] for e in cash),
        "cash_event_samples": len(cash),
        "trades_per_day": len(complete) / ((end - start).total_seconds() / 86400),
        "average_win": math.fsum(wins) / len(wins) if wins else None,
        "average_loss": math.fsum(losses) / len(losses) if losses else None,
        "breakeven_trades": sum(v == 0 for v in pnl),
        "open_samples": sum(t["state"] == "open" for t in all_trades),
        "unknown_samples": sum(t["state"] == "Unknown" for t in all_trades),
        "complete_samples_outside_window": sum(t["state"] == "complete" for t in all_trades) - len(complete),
        "unknown_provenance_samples": sum(t["provenance"] == "Unknown" for t in complete),
        "missing_data_samples": sum(t["state"] == "Unknown" for t in all_trades),
        "version_pairs": sorted({tuple(pair) for t in complete for pair in t["version_pairs"]}),
        "duplicate_deals": ledger["duplicate_deals"],
    }


def read_records(path):
    from training_holdout_guard_v1 import check_path
    path = check_path(path, Path(__file__).resolve().parent)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", action="append", required=True)
    parser.add_argument("--orders", action="append", default=[])
    parser.add_argument("--signals", action="append", default=[])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, help="New directory; never overwrite a report")
    args = parser.parse_args()
    deals = [r for p in args.deals for r in read_records(p)]
    orders = [r for p in args.orders for r in read_records(p)]
    signals = [r for p in args.signals for r in read_records(p)]
    ledger = build_ledger(deals, orders, signals)
    metrics = calculate_metrics(ledger, args.start, args.end)
    from training_holdout_guard_v1 import check_path
    output = check_path(args.output, Path(__file__).resolve().parent)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("ledger.json", ledger), ("metrics.json", metrics)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
