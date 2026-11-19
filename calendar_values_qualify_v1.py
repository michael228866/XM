"""Read-only data-qualification check for an ExportUsdCalendarValuesV2.mq5 CSV (STRATEGY_RESEARCH_2026_11.md 8.4).

Checks format, types, missing values, duplicate value_id / (event_id, period, revision), event coverage and time range.
It never computes strategy performance, never touches accounts, never rescales values (the exporter already divided
by 10^6) and never converts server time to UTC. A historical export can NOT prove point-in-time availability: no
column (revision=0, value_id, actual == next previous) carries the first-receipt time, so that stays Unknown.

Usage: python calendar_values_qualify_v1.py PATH [--encoding auto|utf-8|gbk|...]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

COLUMNS = ["value_id", "event_id", "importance", "scheduled_time", "period", "revision",
           "actual", "forecast", "previous", "revised_previous", "impact_type", "export_time", "name"]
INT_COLUMNS = ["value_id", "event_id", "importance", "scheduled_time", "period", "revision", "impact_type", "export_time"]
VALUE_COLUMNS = ["actual", "forecast", "previous", "revised_previous"]
# 8.3 polarity table (fixed before any performance); events outside it are reported, never used in S.
POLARITY = {e: 1 for e in [840030016, 840190001, 840030018, 840020010, 840020011, 840040001, 840040003, 840040002,
                           840040008, 840500001, 840500002, 840030021, 840020008, 840120001, 840180002, 840510001,
                           840070001, 840020013, 840010001, 840010002, 840030001, 840030005, 840030007, 840030006,
                           840010007, 840010018, 840050014]}
POLARITY.update({840030015: -1, 840140001: -1})
LONG_MIN = -9223372036854775808
UNCONFIRMABLE = [
    "forecast 是否為公布前版本（匯出檔只有目前版本）",
    "actual 是否為首發值（revision、value_id、與下一期 previous 一致只支持格式與一致性）",
    "R + 2 分鐘前是否已收到（匯出檔沒有收到時間）",
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def decode(raw, encoding):
    if encoding != "auto":
        return raw.decode(encoding)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")   # strict: a GBK file fails here instead of being silently misread


def server_time(seconds):
    # MT5 stores trade-server wall time as epoch seconds; format it without any timezone conversion.
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " (server)"


def parse_value(text):
    """'' -> None (missing, never 0). Raises ValueError on garbage or an unconverted LONG_MIN."""
    if text == "":
        return None
    v = float(text)
    if not math.isfinite(v):
        raise ValueError("non-finite")
    if v == LONG_MIN or v == LONG_MIN / 1e6:
        raise ValueError("LONG_MIN 未轉成空值")
    return v


def check(path, encoding="auto"):
    before = sha256(path)
    with open(path, "rb") as f:
        raw = f.read()
    report = {"path": path, "sha256": before, "format_ok": True, "errors": [], "notes": [],
              "unconfirmable": UNCONFIRMABLE, "point_in_time": "Unknown（歷史匯出：只能回顧探索）"}
    try:
        text = decode(raw, encoding)
    except UnicodeDecodeError as e:
        report["format_ok"] = False
        report["errors"].append(f"編碼錯誤（{e.reason}）；舊版 FILE_ANSI 檔請明確指定 --encoding gbk")
        return report

    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0] != COLUMNS:
        missing = [c for c in COLUMNS if not rows or c not in rows[0]]
        report["format_ok"] = False
        report["errors"].append(f"欄位不符契約；缺少 {missing}" if missing else "欄位順序或多餘欄位不符契約")
        return report

    good, bad_lines = [], Counter()
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != len(COLUMNS):
            bad_lines["欄位數錯誤"] += 1
            report["errors"].append(f"第 {n} 行欄位數 {len(r)}")
            continue
        rec = dict(zip(COLUMNS, r))
        try:
            for c in INT_COLUMNS:
                rec[c] = int(rec[c])
            for c in VALUE_COLUMNS:
                rec[c] = parse_value(rec[c])
        except ValueError as e:
            bad_lines["型別錯誤"] += 1
            report["errors"].append(f"第 {n} 行型別錯誤：{e}")
            continue
        rec["line"] = n
        good.append(rec)
    if bad_lines:
        report["format_ok"] = False

    full = Counter(tuple(r[c] for c in COLUMNS) for r in good)
    exact_dups = sum(k - 1 for k in full.values() if k > 1)
    ids = Counter(r["value_id"] for r in good)
    dup_ids = sorted(i for i, k in ids.items() if k > 1)
    versions = defaultdict(list)
    for r in good:
        versions[(r["event_id"], r["period"], r["revision"])].append(r)
    conflicts = sorted(k for k, v in versions.items()
                       if len({(x["actual"], x["forecast"], x["scheduled_time"]) for x in v}) > 1)
    periods = defaultdict(set)
    for r in good:
        periods[(r["event_id"], r["period"])].add(r["revision"])
    multi_rev = sum(1 for v in periods.values() if len(v) > 1)
    if exact_dups:
        report["errors"].append(f"完全重複列 {exact_dups}")
    if dup_ids:
        report["errors"].append(f"重複 value_id {len(dup_ids)}：{dup_ids[:10]}")
    if conflicts:
        report["errors"].append(f"同一 (event_id, period, revision) 數值互相矛盾 {len(conflicts)}：{conflicts[:10]}")
    if exact_dups or dup_ids or conflicts:
        report["format_ok"] = False

    # Rows usable for retrospective exploration: polarity event, actual and forecast present, no conflict,
    # and not an actual dated after the export (that would be a contradiction, not a release).
    conflict_keys = set(conflicts)
    reasons = Counter()
    usable = []
    for r in good:
        if r["event_id"] not in POLARITY:
            reasons["不在極性表"] += 1
        elif r["actual"] is None:
            reasons["缺 actual"] += 1
        elif r["forecast"] is None:
            reasons["缺 forecast"] += 1
        elif (r["event_id"], r["period"], r["revision"]) in conflict_keys:
            reasons["版本矛盾"] += 1
        elif r["scheduled_time"] > r["export_time"]:
            reasons["公布時間晚於匯出卻有 actual"] += 1
        else:
            usable.append(r)
    first = {}
    for r in usable:   # one row per (event_id, period): the lowest revision (format choice, not a first-release proof)
        k = (r["event_id"], r["period"])
        if k not in first or r["revision"] < first[k]["revision"]:
            first[k] = r
    reasons["同期較高 revision"] += len(usable) - len(first)
    usable = list(first.values())

    coverage = {}
    for e in POLARITY:
        ev = [r for r in good if r["event_id"] == e]
        ok = [r for r in usable if r["event_id"] == e]
        coverage[str(e)] = {
            "rows": len(ev), "usable": len(ok),
            "forecast_present": sum(r["forecast"] is not None for r in ev),
            "zero_actual": sum(r["actual"] == 0 for r in ev),
            "first": server_time(min(r["scheduled_time"] for r in ev)) if ev else None,
            "last": server_time(max(r["scheduled_time"] for r in ev)) if ev else None,
        }
    absent = [e for e, c in coverage.items() if c["rows"] == 0]
    if absent:
        report["notes"].append(f"極性表中 {len(absent)} 個事件沒有任何列：{absent}")
    if multi_rev:
        report["notes"].append(f"{multi_rev} 個 (event_id, period) 有多個 revision；探索只取最小 revision，仍不代表首發已證明")
    report["notes"].append("數值未再縮放（匯出端已除以 10^6）；時間為伺服器時間，未轉 UTC")

    report.update({
        "rows": len(rows) - 1, "parsed_rows": len(good), "bad_rows": dict(bad_lines),
        "exact_duplicates": exact_dups, "duplicate_value_ids": len(dup_ids), "version_conflicts": len(conflicts),
        "multi_revision_periods": multi_rev,
        "time_range": [server_time(min(r["scheduled_time"] for r in good)),
                       server_time(max(r["scheduled_time"] for r in good))] if good else None,
        "export_times": sorted({server_time(r["export_time"]) for r in good}),
        "exploratory_usable_rows": len(usable), "excluded": dict(reasons), "coverage": coverage,
    })
    after = sha256(path)
    if after != before:
        raise RuntimeError("輸入檔在檢查期間被改動")
    report["sha256_unchanged"] = True
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path")
    p.add_argument("--encoding", default="auto")
    a = p.parse_args(argv)
    report = check(a.path, a.encoding)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["format_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
