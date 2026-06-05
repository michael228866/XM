import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CONF_THRESHOLD = 0.525
EDGE_THRESHOLD = 0.0
MAX_SPREAD_POINTS = 45.0
ALLOWED_ENTRY_HOURS = {0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23}
ALLOWED_ENTRY_WEEKDAYS = {0, 1, 2, 4}
EXCLUDED_RSI_RANGES = [(35.0, 45.0)]


@dataclass(frozen=True)
class SignalRow:
    event_time: datetime
    bar_time: datetime
    status: str
    buy_prob: float
    sell_prob: float
    edge: float
    hour: int
    weekday: int
    rsi: float
    in_session: bool
    rsi_ok: bool
    valid: bool
    spread_points: float | None
    lot: float | None
    retcode: str


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def parse_row(row: dict[str, str], line_no: int) -> SignalRow:
    try:
        return SignalRow(
            event_time=datetime.fromisoformat(row["event_time"]),
            bar_time=datetime.fromisoformat(row["bar_time"]),
            status=row["status"],
            buy_prob=float(row["buy_prob"]),
            sell_prob=float(row["sell_prob"]),
            edge=float(row["edge"]),
            hour=int(row["hour"]),
            weekday=int(row["weekday"]),
            rsi=float(row["rsi"]),
            in_session=parse_bool(row["in_session"]),
            rsi_ok=parse_bool(row["rsi_ok"]),
            valid=parse_bool(row["valid"]),
            spread_points=parse_float(row.get("spread_points", "")),
            lot=parse_float(row.get("lot", "")),
            retcode=row.get("retcode", "").strip(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CSV data at line {line_no}: {exc}") from exc


def load_rows(path: Path) -> list[SignalRow]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Log file is empty: {path}")
        return [parse_row(row, index + 2) for index, row in enumerate(reader)]


def expected_in_session(row: SignalRow) -> bool:
    return row.hour in ALLOWED_ENTRY_HOURS and row.weekday in ALLOWED_ENTRY_WEEKDAYS


def expected_rsi_ok(row: SignalRow) -> bool:
    return not any(low <= row.rsi <= high for low, high in EXCLUDED_RSI_RANGES)


def expected_valid_without_position(row: SignalRow) -> bool:
    return (
        expected_in_session(row)
        and expected_rsi_ok(row)
        and row.buy_prob >= CONF_THRESHOLD
        and row.edge >= EDGE_THRESHOLD
        and row.buy_prob >= row.sell_prob
    )


def primary_block_reason(row: SignalRow) -> str:
    if not expected_in_session(row):
        return "out_of_session"
    if not expected_rsi_ok(row):
        return "rsi_excluded"
    if row.buy_prob < CONF_THRESHOLD:
        return "buy_below_threshold"
    if row.edge < EDGE_THRESHOLD:
        return "negative_edge"
    if row.buy_prob < row.sell_prob:
        return "sell_dominates"
    if not row.valid:
        return "likely_position_open"
    return "valid"


def average(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def validate_rows(rows: list[SignalRow]) -> tuple[Counter, list[str]]:
    reasons = Counter(primary_block_reason(row) for row in rows)
    anomalies: list[str] = []
    seen_keys: set[tuple[datetime, datetime]] = set()

    for row in rows:
        key = (row.event_time, row.bar_time)
        if key in seen_keys:
            anomalies.append(f"duplicate row: {row.event_time.isoformat()} / {row.bar_time.isoformat()}")
        seen_keys.add(key)

        if row.in_session != expected_in_session(row):
            anomalies.append(f"in_session mismatch at {row.event_time.isoformat()}")
        if row.rsi_ok != expected_rsi_ok(row):
            anomalies.append(f"rsi_ok mismatch at {row.event_time.isoformat()}")
        if row.status == "order_opened":
            if not row.valid:
                anomalies.append(f"order opened while valid=False at {row.event_time.isoformat()}")
            if not expected_valid_without_position(row):
                anomalies.append(f"order opened despite failed signal gate at {row.event_time.isoformat()}")
            if row.spread_points is None:
                anomalies.append(f"order opened without spread at {row.event_time.isoformat()}")
            elif row.spread_points > MAX_SPREAD_POINTS:
                anomalies.append(f"order opened with wide spread at {row.event_time.isoformat()}: {row.spread_points}")
            if row.lot is None or row.lot <= 0:
                anomalies.append(f"order opened with invalid lot at {row.event_time.isoformat()}: {row.lot}")
            if row.retcode != "10009":
                anomalies.append(f"order opened with unexpected retcode at {row.event_time.isoformat()}: {row.retcode}")

    return reasons, anomalies


def print_summary(rows: list[SignalRow], reasons: Counter, anomalies: list[str]) -> None:
    statuses = Counter(row.status for row in rows)
    by_day = Counter(row.event_time.date().isoformat() for row in rows)
    orders = [row for row in rows if row.status == "order_opened"]
    order_buy_probs = [row.buy_prob for row in orders]
    order_edges = [row.edge for row in orders]
    order_lots = [row.lot for row in orders if row.lot is not None]
    order_spreads = [row.spread_points for row in orders if row.spread_points is not None]
    orders_by_hour = Counter(row.hour for row in orders)
    orders_by_weekday = Counter(row.weekday for row in orders)

    print("Gemini signal log validation")
    print("=" * 28)
    print(f"Rows: {len(rows)}")
    print(f"Range: {rows[0].event_time.isoformat()} -> {rows[-1].event_time.isoformat()}")
    print()

    print("Status counts:")
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")
    print()

    print("Rows by day:")
    for day in sorted(by_day):
        print(f"  {day}: {by_day[day]}")
    print()

    print("Primary gate classification:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    print()

    print("Opened order sanity:")
    print(f"  orders: {len(orders)}")
    print(f"  buy_prob avg/min/max: {fmt(average(order_buy_probs))} / {fmt(min(order_buy_probs) if order_buy_probs else None)} / {fmt(max(order_buy_probs) if order_buy_probs else None)}")
    print(f"  edge avg/min/max: {fmt(average(order_edges))} / {fmt(min(order_edges) if order_edges else None)} / {fmt(max(order_edges) if order_edges else None)}")
    print(f"  lot avg/min/max: {fmt(average(order_lots), 2)} / {fmt(min(order_lots) if order_lots else None, 2)} / {fmt(max(order_lots) if order_lots else None, 2)}")
    print(f"  spread avg/min/max: {fmt(average(order_spreads), 2)} / {fmt(min(order_spreads) if order_spreads else None, 2)} / {fmt(max(order_spreads) if order_spreads else None, 2)}")
    print(f"  by hour: {dict(sorted(orders_by_hour.items()))}")
    print(f"  by weekday: {dict(sorted(orders_by_weekday.items()))}")
    print()

    if anomalies:
        print("Anomalies:")
        for anomaly in anomalies[:50]:
            print(f"  {anomaly}")
        if len(anomalies) > 50:
            print(f"  ... {len(anomalies) - 50} more")
    else:
        print("Anomalies: none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gemini.py signal log behavior.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default="gemini_signal_log.csv",
        help="Path to gemini_signal_log.csv",
    )
    args = parser.parse_args()

    try:
        rows = load_rows(Path(args.log_file))
        if not rows:
            raise ValueError("Log file contains no data rows.")
        reasons, anomalies = validate_rows(rows)
        print_summary(rows, reasons, anomalies)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
