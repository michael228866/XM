from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import drl_trading_v2
import gold_gemini_execution_aligned_label_v1 as baseline
import gold_gemini_execution_semantics_v1 as semantics
from barrier_classifier_strategy import HORIZON as LEGACY_HORIZON_ROWS
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import predict_positive


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "GEMINI MACRO EVENT TIMING INFORMATION V1"
EXPERIMENT_SLUG = "gemini_macro_event_timing_v1"
GEMINI_FILE = ROOT / "gemini.py"
OPERATIONAL_MODEL = ROOT / "gold_long_recent_candidate_xgb.json"
BASELINE_RUN = ROOT / "training_runs" / "20260903T071729Z_gemini_execution_aligned_label_v1"
DATA_FOUNDATION = ROOT / "gold_data_foundation1_report.json"
MODEL_IDS = ("B0_technical_control", "B1_macro_event_timing")
EVENT_TYPES = ("CPI", "EMPLOYMENT", "PCE", "FOMC")
EVENT_FEATURES = (
    "EVENT_MINUTES_SINCE",
    "EVENT_POST_0_15",
    "EVENT_POST_15_60",
    "EVENT_POST_60_240",
    "EVENT_TYPE_CPI",
    "EVENT_TYPE_EMPLOYMENT",
    "EVENT_TYPE_PCE",
    "EVENT_TYPE_FOMC",
)
ACQUISITION_START_YEAR = 2015
ACQUISITION_END_YEAR = 2024
MIN_ANNUAL_RELEASES = {"CPI": 11, "EMPLOYMENT": 11, "PCE": 11, "FOMC": 7}
MIN_COVERAGE = 0.95
USER_AGENT = "Mozilla/5.0 (compatible; XM-GOLD-research/1.0; research provenance audit)"
EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    baseline.write_json(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = list(fields or (list(rows[0]) if rows else []))
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        raise ValueError(f"CSV schema required for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(baseline.sanitize(rows))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def normalized_text(document: str) -> str:
    parser = TextParser()
    parser.feed(document)
    return " ".join(html.unescape(" ".join(parser.parts)).replace("\xa0", " ").split())


def links(document: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(document)
    return sorted({urllib.parse.urljoin(base_url, item) for item in parser.links})


def content_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(url: str, provenance: list[dict[str, Any]], purpose: str) -> str | None:
    acquired = now_utc()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            status = int(getattr(response, "status", 200))
            final_url = response.geturl()
        provenance.append(
            {
                "purpose": purpose,
                "requested_url": url,
                "final_url": final_url,
                "official_domain": urllib.parse.urlparse(final_url).netloc.lower(),
                "http_status": status,
                "bytes": len(body),
                "content_sha256": content_sha256(body),
                "acquired_at_utc": acquired,
                "status": "ok",
                "error": "",
            }
        )
        return body.decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        provenance.append(
            {
                "purpose": purpose,
                "requested_url": url,
                "final_url": "",
                "official_domain": urllib.parse.urlparse(url).netloc.lower(),
                "http_status": "",
                "bytes": 0,
                "content_sha256": "",
                "acquired_at_utc": acquired,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None


TIME_RE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>a\.?m\.?|p\.?m\.?)"
    r"(?:\s*\((?P<zone1>E(?:S|D)?T)\)|\s+(?P<zone2>E(?:S|D)?T))?",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<day>\d{1,2}),\s+(?P<year>20\d{2})",
    re.IGNORECASE,
)


def official_timestamp(document: str, date_hint: datetime | None = None) -> tuple[datetime, str, str] | None:
    text = normalized_text(document)
    anchors = [match.start() for match in re.finditer(r"embargoed until|for release", text, re.IGNORECASE)]
    snippets = [text[index : index + 320] for index in anchors] or [text[:1000]]
    for snippet in snippets:
        time_match = TIME_RE.search(snippet)
        date_match = DATE_RE.search(snippet)
        if time_match is None:
            continue
        if date_match is not None:
            date_value = datetime.strptime(
                f"{date_match.group('month')} {date_match.group('day')} {date_match.group('year')}",
                "%B %d %Y",
            )
        elif date_hint is not None:
            date_value = date_hint
        else:
            continue
        hour = int(time_match.group("hour")) % 12
        if time_match.group("ampm").lower().startswith("p"):
            hour += 12
        local = datetime(date_value.year, date_value.month, date_value.day, hour, int(time_match.group("minute")), tzinfo=EASTERN)
        zone_text = time_match.group("zone1") or time_match.group("zone2") or "ET"
        published = snippet[: min(len(snippet), 260)]
        return local.astimezone(UTC), zone_text.upper(), published
    return None


def event_row(event_type: str, timestamp: datetime, zone: str, source: str, release_id: str, evidence: str) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "official_release_timestamp": evidence,
        "original_timezone": zone,
        "release_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "official_source": urllib.parse.urlparse(source).netloc.lower(),
        "source_document": source,
        "source_release_identifier": release_id,
        "acquisition_timestamp_utc": now_utc(),
        "verification_basis": "official document release or embargo timestamp",
        "broker_wall_timestamp": utc_to_broker_wall(timestamp).isoformat(),
    }


def url_date(url: str) -> datetime | None:
    name = Path(urllib.parse.urlparse(url).path).name
    for pattern, fmt in ((r"_(\d{8})\.(?:htm|html)$", "%m%d%Y"), (r"(20\d{6})", "%Y%m%d")):
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt)
            except ValueError:
                pass
    return None


def extract_bls(kind: str, slug: str, provenance: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_url = f"https://www.bls.gov/bls/news-release/{slug}.htm"
    document = fetch(index_url, provenance, f"{kind}_archive_index")
    discovered: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if document is None:
        return events, discovered
    candidates = []
    for url in links(document, index_url):
        if f"/news.release/archives/{slug}_" not in url.lower():
            continue
        date_hint = url_date(url)
        if date_hint and ACQUISITION_START_YEAR <= date_hint.year <= ACQUISITION_END_YEAR:
            candidates.append((date_hint, url))
    for date_hint, url in sorted(set(candidates)):
        discovered.append({"event_type": kind, "expected_date": date_hint.date().isoformat(), "source_document": url})
        release = fetch(url, provenance, f"{kind}_release_document")
        if release is None:
            continue
        parsed = official_timestamp(release, date_hint)
        if parsed is None:
            continue
        timestamp, zone, evidence = parsed
        events.append(event_row(kind, timestamp, zone, url, Path(urllib.parse.urlparse(url).path).stem, evidence))
    return events, discovered


def extract_bea(provenance: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_url = "https://www.bea.gov/sitemap.xml"
    root = fetch(root_url, provenance, "PCE_sitemap_index")
    discovered: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if root is None:
        return events, discovered
    documents = [root]
    sitemap_urls = re.findall(r"<loc>\s*(https://www\.bea\.gov/[^<]*sitemap[^<]*)</loc>", root, re.IGNORECASE)
    for url in sitemap_urls[:20]:
        child = fetch(html.unescape(url), provenance, "PCE_sitemap_child")
        if child:
            documents.append(child)
    page_urls: set[str] = set()
    for document in documents:
        for url in re.findall(r"<loc>\s*(https://www\.bea\.gov/[^<]+)</loc>", document, re.IGNORECASE):
            value = html.unescape(url)
            if re.search(r"/news/20(?:1[5-9]|2[0-4])/(?:personal-income|personal-consumption)", value, re.IGNORECASE):
                page_urls.add(value)
    for url in sorted(page_urls):
        year_match = re.search(r"/news/(20\d{2})/", url)
        expected_year = int(year_match.group(1)) if year_match else 0
        discovered.append({"event_type": "PCE", "expected_date": f"{expected_year:04d}-unknown", "source_document": url})
        release = fetch(url, provenance, "PCE_release_document")
        if release is None:
            continue
        parsed = official_timestamp(release)
        if parsed is None:
            continue
        timestamp, zone, evidence = parsed
        if ACQUISITION_START_YEAR <= timestamp.year <= ACQUISITION_END_YEAR:
            events.append(event_row("PCE", timestamp, zone, url, Path(urllib.parse.urlparse(url).path).name, evidence))
            discovered[-1]["expected_date"] = timestamp.date().isoformat()
    return events, discovered


def extract_fomc(provenance: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"]
    indexes.extend(
        f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        for year in range(ACQUISITION_START_YEAR, 2021)
    )
    candidate_urls: set[str] = set()
    for index_url in indexes:
        document = fetch(index_url, provenance, "FOMC_calendar_or_historical_index")
        if document is None:
            continue
        for url in links(document, index_url):
            lowered = url.lower()
            if ("pressreleases/monetary20" in lowered or "fomcstatement20" in lowered) and not lowered.endswith((".mp3", ".htm#")):
                date_hint = url_date(url)
                if date_hint and ACQUISITION_START_YEAR <= date_hint.year <= ACQUISITION_END_YEAR:
                    candidate_urls.add(url.split("#", 1)[0])
    discovered: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for url in sorted(candidate_urls):
        date_hint = url_date(url)
        discovered.append({"event_type": "FOMC", "expected_date": date_hint.date().isoformat() if date_hint else "unknown", "source_document": url})
        release = fetch(url, provenance, "FOMC_statement_document")
        if release is None:
            continue
        parsed = official_timestamp(release, date_hint)
        if parsed is None:
            continue
        timestamp, zone, evidence = parsed
        events.append(event_row("FOMC", timestamp, zone, url, Path(urllib.parse.urlparse(url).path).stem, evidence))
    return events, discovered


def utc_to_broker_wall(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    day = datetime(value.year, 3, 31, 1, tzinfo=UTC)
    march_last_sunday = day - timedelta(days=(day.weekday() + 1) % 7)
    day = datetime(value.year, 10, 31, 1, tzinfo=UTC)
    october_last_sunday = day - timedelta(days=(day.weekday() + 1) % 7)
    offset = 3 if march_last_sunday <= value < october_last_sunday else 2
    return (value + timedelta(hours=offset)).replace(tzinfo=None)


def acquire_events(run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    provenance: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    for kind, slug in (("CPI", "cpi"), ("EMPLOYMENT", "empsit")):
        events, expected = extract_bls(kind, slug, provenance)
        all_events.extend(events)
        discovered.extend(expected)
    events, expected = extract_bea(provenance)
    all_events.extend(events)
    discovered.extend(expected)
    events, expected = extract_fomc(provenance)
    all_events.extend(events)
    discovered.extend(expected)

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count: dict[tuple[str, int], int] = {}
    for row in all_events:
        timestamp = pd.Timestamp(row["release_timestamp_utc"])
        duplicate_count[(row["event_type"], timestamp.year)] = duplicate_count.get((row["event_type"], timestamp.year), 0) + int((row["event_type"], row["release_timestamp_utc"]) in deduped)
        deduped[(row["event_type"], row["release_timestamp_utc"])] = row
    rows = sorted(deduped.values(), key=lambda item: (item["release_timestamp_utc"], item["event_type"]))
    write_csv(
        run_dir / "event_source_provenance.csv",
        provenance,
        ["purpose", "requested_url", "final_url", "official_domain", "http_status", "bytes", "content_sha256", "acquired_at_utc", "status", "error"],
    )
    write_csv(
        run_dir / "event_timestamp_dataset.csv",
        rows,
        ["event_type", "official_release_timestamp", "original_timezone", "release_timestamp_utc", "official_source", "source_document", "source_release_identifier", "acquisition_timestamp_utc", "verification_basis", "broker_wall_timestamp"],
    )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["release_timestamp_utc"] = pd.to_datetime(frame["release_timestamp_utc"], utc=True)
        frame["broker_wall_timestamp"] = pd.to_datetime(frame["broker_wall_timestamp"])
    return frame, discovered, provenance


def fold_years(fold_name: str) -> tuple[int, int]:
    start, end = fold_name.split("_")
    return int(start), int(end)


def coverage_audit(events: pd.DataFrame, discovered: list[dict[str, Any]], baseline_times: pd.DatetimeIndex) -> tuple[list[dict[str, Any]], bool, list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for fold_name, _, _ in baseline.FOLDS:
        first_year, last_year = fold_years(fold_name)
        years = last_year - first_year + 1
        for event_type in EVENT_TYPES:
            expected_rows = [
                row for row in discovered
                if row["event_type"] == event_type
                and re.match(r"20\d{2}", str(row["expected_date"]))
                and first_year <= int(str(row["expected_date"])[:4]) <= last_year
            ]
            if len(events):
                selected = events[
                    events["event_type"].eq(event_type)
                    & events["release_timestamp_utc"].dt.year.between(first_year, last_year)
                ]
            else:
                selected = events
            expected = len(expected_rows)
            verified = len(selected)
            minimum = MIN_ANNUAL_RELEASES[event_type] * years
            coverage = verified / expected if expected else 0.0
            passed = expected >= minimum and coverage >= MIN_COVERAGE and verified <= expected
            if not passed:
                failures.append(f"{fold_name}/{event_type}: expected={expected}, verified={verified}, coverage={coverage:.1%}, minimum={minimum}")
            rows.append(
                {
                    "fold": fold_name,
                    "event_type": event_type,
                    "expected_identified_releases": expected,
                    "verified_timestamped_releases": verified,
                    "coverage_percentage": coverage,
                    "first_timestamp_utc": selected["release_timestamp_utc"].min().isoformat() if verified else "",
                    "last_timestamp_utc": selected["release_timestamp_utc"].max().isoformat() if verified else "",
                    "duplicate_timestamps": int(selected.duplicated(["event_type", "release_timestamp_utc"]).sum()) if verified else 0,
                    "ambiguous_timestamps": max(expected - verified, 0),
                    "missing_timestamps": max(expected - verified, 0),
                    "minimum_expected_releases": minimum,
                    "coverage_gate_pass": passed,
                }
            )
    constructable_rows = 0
    if len(events):
        earliest = events["broker_wall_timestamp"].min()
        constructable_rows = int((baseline_times >= earliest).sum())
    constructability = constructable_rows / len(baseline_times) if len(baseline_times) else 0.0
    rows.append(
        {
            "fold": "all_scored_rows",
            "event_type": "ALL",
            "expected_identified_releases": "",
            "verified_timestamped_releases": len(events),
            "coverage_percentage": constructability,
            "first_timestamp_utc": events["release_timestamp_utc"].min().isoformat() if len(events) else "",
            "last_timestamp_utc": events["release_timestamp_utc"].max().isoformat() if len(events) else "",
            "duplicate_timestamps": int(events.duplicated(["event_type", "release_timestamp_utc"]).sum()) if len(events) else 0,
            "ambiguous_timestamps": "",
            "missing_timestamps": "",
            "minimum_expected_releases": "",
            "coverage_gate_pass": constructability == 1.0,
            "scored_rows": len(baseline_times),
            "causally_constructable_rows": constructable_rows,
        }
    )
    if constructability != 1.0:
        failures.append(f"scored-row causal constructability={constructability:.2%}, required=100%")
    if len(events) and events.duplicated(["event_type", "release_timestamp_utc"]).any():
        failures.append("duplicate verified event timestamps remain")
    return rows, not failures, failures


def add_event_features(history: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if events.empty:
        raise RuntimeError("No verified events available")
    left = history[["TIME_DT"]].copy()
    left["row_index"] = np.arange(len(left), dtype=np.int64)
    right = events[["broker_wall_timestamp", "release_timestamp_utc", "event_type", "source_release_identifier"]].copy()
    right = right.sort_values(["broker_wall_timestamp", "event_type"]).drop_duplicates("broker_wall_timestamp", keep="last")
    merged = pd.merge_asof(
        left.sort_values("TIME_DT"),
        right.sort_values("broker_wall_timestamp"),
        left_on="TIME_DT",
        right_on="broker_wall_timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("row_index")
    minutes = (merged["TIME_DT"] - merged["broker_wall_timestamp"]).dt.total_seconds() / 60.0
    if (minutes.dropna() < 0).any():
        raise RuntimeError("Pre-release event alignment detected")
    result = history.copy()
    result["EVENT_MINUTES_SINCE"] = np.minimum(minutes.fillna(1440.0), 1440.0) / 1440.0
    result["EVENT_POST_0_15"] = ((minutes >= 0.0) & (minutes < 15.0)).astype(np.int8)
    result["EVENT_POST_15_60"] = ((minutes >= 15.0) & (minutes < 60.0)).astype(np.int8)
    result["EVENT_POST_60_240"] = ((minutes >= 60.0) & (minutes < 240.0)).astype(np.int8)
    active = (minutes >= 0.0) & (minutes < 240.0)
    for event_type in EVENT_TYPES:
        result[f"EVENT_TYPE_{event_type}"] = (active & merged["event_type"].eq(event_type)).astype(np.int8)
    result["MATCHED_EVENT_TYPE"] = merged["event_type"].fillna("NONE")
    result["MATCHED_EVENT_UTC"] = merged["release_timestamp_utc"]
    result["EVENT_MINUTES_RAW"] = minutes
    checks = []
    active_rows = merged["release_timestamp_utc"].notna()
    for row in merged.loc[active_rows].iloc[:: max(int(active_rows.sum() // 500), 1)].itertuples(index=False):
        event_utc = pd.Timestamp(row.release_timestamp_utc)
        broker_wall = pd.Timestamp(row.TIME_DT)
        checks.append(
            {
                "decision_timestamp_broker_wall": broker_wall.isoformat(),
                "event_release_timestamp_utc": event_utc.isoformat(),
                "event_broker_wall_timestamp": pd.Timestamp(row.broker_wall_timestamp).isoformat(),
                "event_type": row.event_type,
                "event_release_le_decision": bool(pd.Timestamp(row.broker_wall_timestamp) <= broker_wall),
                "timezone_conversion": "official America/New_York -> UTC -> XM EET/EEST broker wall",
            }
        )
    return result, checks


def preregister(run_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = baseline.read_json(run_dir / "manifest.json")
    if manifest.get("status") != "in_progress" or manifest.get("git_dirty") is not False:
        raise RuntimeError("Formal run must be in progress and start git-clean")
    if baseline.sha256(Path(__file__)) != manifest.get("training_script_sha256"):
        raise RuntimeError("Executed script differs from immutable snapshot")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "origin/main")
    if head != upstream or head != manifest.get("git_commit"):
        raise RuntimeError("HEAD, origin/main, and manifest commit must match")
    operational = {GEMINI_FILE.name: baseline.sha256(GEMINI_FILE), OPERATIONAL_MODEL.name: baseline.sha256(OPERATIONAL_MODEL)}
    manifest["pre_run_git"] = {
        "pre_run_git_commit": head,
        "pre_run_git_dirty": False,
        "head_sha": head,
        "origin_main_sha": upstream,
        "head_equals_origin_main": True,
    }
    manifest["paired_design"] = {
        "models": list(MODEL_IDS),
        "only_design_change": "append exact frozen eight-feature U.S. macro release timing family to B1",
        "event_types": list(EVENT_TYPES),
        "event_features": list(EVENT_FEATURES),
        "training_window_months": baseline.TRAIN_MONTHS,
        "folds": [name for name, _, _ in baseline.FOLDS],
        "xgboost_parameters": baseline.FIXED_XGB_PARAMETERS,
        "threshold": baseline.THRESHOLD,
        "parameter_search": False,
        "threshold_search": False,
        "event_subset_search": False,
    }
    manifest["operational_hashes_before"] = operational
    manifest["dependency_sha256"] = {
        "baseline_run_manifest.json": baseline.sha256(BASELINE_RUN / "manifest.json"),
        "gold_gemini_execution_aligned_label_v1.py": baseline.sha256(ROOT / "gold_gemini_execution_aligned_label_v1.py"),
        "gold_gemini_execution_semantics_v1.py": baseline.sha256(ROOT / "gold_gemini_execution_semantics_v1.py"),
        "gold_data_foundation1_report.json": baseline.sha256(DATA_FOUNDATION),
    }
    write_json(run_dir / "manifest.json", manifest)
    return operational, manifest


def event_definition() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "event_universe": {
            "CPI": "official BLS CPI release timestamp",
            "EMPLOYMENT": "official BLS Employment Situation release timestamp; NFP is not separate",
            "PCE": "official BEA Personal Income and Outlays/PCE release timestamp",
            "FOMC": "official Federal Reserve monetary-policy statement/rate-decision release timestamp",
        },
        "features": list(EVENT_FEATURES),
        "feature_rules": {
            "EVENT_MINUTES_SINCE": "min(minutes since most recent qualifying release, 1440) / 1440",
            "EVENT_POST_0_15": "0 <= minutes < 15",
            "EVENT_POST_15_60": "15 <= minutes < 60",
            "EVENT_POST_60_240": "60 <= minutes < 240",
            "event_type_flags": "matching type is one only for 0 <= minutes < 240; otherwise all zero",
            "no_prior_event": "EVENT_MINUTES_SINCE=1 and all seven flags=0",
        },
        "excluded_information": ["actual", "forecast", "consensus", "surprise", "revisions", "NLP", "future schedule changes"],
        "readiness_gate": {
            "minimum_verified_over_identified": MIN_COVERAGE,
            "minimum_annual_identified_releases": MIN_ANNUAL_RELEASES,
            "all_scored_rows_causally_constructable": True,
            "duplicate_verified_timestamps_allowed": False,
        },
        "timezone": "official America/New_York release timestamp converted to UTC; GOLD CSV uses audited XM EET/EEST broker wall conversion",
        "acquisition_years": [ACQUISITION_START_YEAR, ACQUISITION_END_YEAR],
        "no_search": True,
    }


def paired_training(history: pd.DataFrame, labels: pd.DataFrame, base_features: list[str], run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    times = history["TIME_DT"]
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    target = labels["C1_TARGET"].to_numpy(dtype=np.int8)
    mature = labels["C1_MATURE"].to_numpy(dtype=bool)
    maturity_ns = labels["C1_MATURITY_NS"].to_numpy(dtype=np.int64)
    legacy_index = np.arange(len(history), dtype=np.int64) + LEGACY_HORIZON_ROWS
    legacy_mature = legacy_index < len(history)
    legacy_maturity_ns = np.full(len(history), np.iinfo(np.int64).max, dtype=np.int64)
    legacy_maturity_ns[legacy_mature] = time_ns[legacy_index[legacy_mature]]
    features_by_model = {MODEL_IDS[0]: base_features, MODEL_IDS[1]: [*base_features, *EVENT_FEATURES]}
    parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for fold_code, (fold_name, fold_start, fold_end) in enumerate(baseline.FOLDS):
        start_ns = int(fold_start.to_datetime64().astype("datetime64[ns]").astype(np.int64))
        lower = fold_start - pd.DateOffset(months=baseline.TRAIN_MONTHS)
        train_mask = (
            (times >= lower) & (times < fold_start) & legacy_mature & mature
            & (legacy_maturity_ns < start_ns) & (maturity_ns < start_ns)
        )
        score_mask = (times >= fold_start) & (times < fold_end) & mature
        train_idx = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        score_idx = np.flatnonzero(score_mask.to_numpy(dtype=bool))
        if not len(train_idx) or not len(score_idx):
            raise RuntimeError(f"Empty train/score fold {fold_name}")
        base_train = history.loc[train_idx, base_features].to_numpy(dtype=np.float32)
        base_score = history.loc[score_idx, base_features].to_numpy(dtype=np.float32)
        scores: dict[str, np.ndarray] = {}
        hashes: dict[str, Any] = {}
        for model_id, features in features_by_model.items():
            model = baseline.train_one(history, features, train_idx, target[train_idx])
            scores[model_id] = predict_positive(model, history.loc[score_idx], features).astype(np.float32)
            path = models_dir / f"{model_id}_{fold_name}_xgb.json"
            model.save_model(path)
            models.append({
                "model_id": model_id,
                "fold": fold_name,
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": baseline.sha256(path),
                "train_rows": len(train_idx),
                "target_sha256": baseline.array_sha256(target[train_idx]),
                "features": features,
            })
            matrix_train = history.loc[train_idx, features].to_numpy(dtype=np.float32)
            matrix_score = history.loc[score_idx, features].to_numpy(dtype=np.float32)
            hashes[f"x_train_sha256_{model_id}"] = baseline.array_sha256(matrix_train)
            hashes[f"x_score_sha256_{model_id}"] = baseline.array_sha256(matrix_score)
            del model, matrix_train, matrix_score
            gc.collect()
        part = history.loc[score_idx, ["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD", "MATCHED_EVENT_TYPE", "MATCHED_EVENT_UTC", "EVENT_MINUTES_RAW", *EVENT_FEATURES]].copy()
        part["global_index"] = score_idx
        part["fold"] = fold_name
        part["fold_code"] = fold_code
        part["C1_TARGET"] = target[score_idx]
        part["C1_NET_R"] = labels["C1_NET_R"].to_numpy(dtype=np.float64)[score_idx]
        part["C1_STRESS_R"] = labels["C1_STRESS_R"].to_numpy(dtype=np.float64)[score_idx]
        for model_id in MODEL_IDS:
            part[f"score_{model_id}"] = scores[model_id]
        parts.append(part)
        latest = int(max(legacy_maturity_ns[train_idx].max(), maturity_ns[train_idx].max()))
        provenance.append({
            "fold": fold_name,
            "train_start": times.iat[int(train_idx[0])].isoformat(),
            "train_feature_end": times.iat[int(train_idx[-1])].isoformat(),
            "latest_training_label_information_time": pd.Timestamp(latest).isoformat(),
            "score_start": times.iat[int(score_idx[0])].isoformat(),
            "score_end": times.iat[int(score_idx[-1])].isoformat(),
            "train_rows": len(train_idx),
            "score_rows": len(score_idx),
            "training_window_months": baseline.TRAIN_MONTHS,
            "strict_label_maturity_before_score": latest < start_ns,
            "train_timestamp_sha256": baseline.array_sha256(time_ns[train_idx]),
            "score_timestamp_sha256": baseline.array_sha256(time_ns[score_idx]),
            "target_sha256_B0": baseline.array_sha256(target[train_idx]),
            "target_sha256_B1": baseline.array_sha256(target[train_idx]),
            "base_x_train_sha256": baseline.array_sha256(base_train),
            "base_x_score_sha256": baseline.array_sha256(base_score),
            "parameters_B0": baseline.FIXED_XGB_PARAMETERS,
            "parameters_B1": baseline.FIXED_XGB_PARAMETERS,
            "random_seed_B0": baseline.RANDOM_STATE,
            "random_seed_B1": baseline.RANDOM_STATE,
            **hashes,
        })
        del base_train, base_score
    return pd.concat(parts, ignore_index=True).sort_values("TIME_DT").reset_index(drop=True), provenance, models


def evaluate(scored: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_metrics: list[dict[str, Any]] = []
    probability: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    trades_all: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, Any]]] = {}
    pooled_days = sum(baseline.core.fold_days(start, end) for _, start, end in baseline.FOLDS)
    scopes = [(name, scored["fold"].eq(name).to_numpy()) for name, _, _ in baseline.FOLDS]
    scopes.append(("pooled", np.ones(len(scored), dtype=bool)))
    for model_id in MODEL_IDS:
        cohort = scored[["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD"]].copy()
        cohort["buy_prob"] = scored[f"score_{model_id}"].to_numpy(dtype=np.float32)
        cohort["sell_prob"] = np.float32(0.0)
        cohort = semantics.finalize_cohort(cohort, f"MACRO_{model_id}", offset_hours=0)
        trades, audit = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        for trade in trades:
            entry = pd.Timestamp(trade["entry_time_api"])
            trade["model_id"] = model_id
            trade["fold"] = next((name for name, start, end in baseline.FOLDS if start <= entry < end), "outside")
            trade["trade_id"] = f"{model_id}_{trade['trade_id']}"
            match = scored.loc[scored["TIME_DT"].eq(entry)]
            if len(match):
                trade["event_minutes_since"] = match["EVENT_MINUTES_RAW"].iat[0]
                trade["event_type"] = match["MATCHED_EVENT_TYPE"].iat[0]
        ledgers[model_id] = trades
        trades_all.extend(trades)
        for fold_name, start, end in baseline.FOLDS:
            selected = [trade for trade in trades if trade["fold"] == fold_name]
            fold_metrics.append({"model_id": model_id, "fold": fold_name, **baseline.trade_metrics(selected, baseline.core.fold_days(start, end))})
        fold_metrics.append({"model_id": model_id, "fold": "pooled", **baseline.trade_metrics(trades, pooled_days), **audit})

        score_all = scored[f"score_{model_id}"].to_numpy(dtype=np.float64)
        target_all = scored["C1_TARGET"].to_numpy(dtype=np.int8)
        reward_all = scored["C1_NET_R"].to_numpy(dtype=np.float64)
        for fold_name, mask in scopes:
            score, target, reward = score_all[mask], target_all[mask], reward_all[mask]
            probability.append({"model_id": model_id, "fold": fold_name, "observations": len(score), **baseline.safe_classification(target, score)})
            order = np.argsort(score, kind="stable")
            bucket = np.empty(len(score), dtype=np.int8)
            bucket[order] = np.minimum(np.arange(len(score)) * 10 // len(score) + 1, 10)
            for number in range(1, 11):
                selected = bucket == number
                stats = baseline.reward_metrics(reward[selected])
                deciles.append({"model_id": model_id, "fold": fold_name, "decile": number, "observations": int(selected.sum()), "mean_probability": float(score[selected].mean()), "positive_rate": float(target[selected].mean()), "mean_r": stats["mean_r"], "pf": stats["pf"]})
            top10 = baseline.reward_metrics(reward[bucket == 10])
            top20 = baseline.reward_metrics(reward[bucket >= 9])
            ranking.append({
                "model_id": model_id,
                "fold": fold_name,
                "observations": len(score),
                "spearman_score_realized_net_r": baseline.spearman(score, reward),
                "spearman_score_positive_net_r": baseline.spearman(score, target.astype(float)),
                "top_decile_mean_r": top10["mean_r"],
                "top_decile_pf": top10["pf"],
                "top_quintile_mean_r": top20["mean_r"],
                "top_quintile_pf": top20["pf"],
            })
    return fold_metrics, probability, {"ranking": ranking, "deciles": deciles, "trades": trades_all, "ledgers": ledgers}


def marginal(ledgers: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b0 = {trade["entry_time_api"]: trade for trade in ledgers[MODEL_IDS[0]]}
    b1 = {trade["entry_time_api"]: trade for trade in ledgers[MODEL_IDS[1]]}
    common = sorted(set(b0) & set(b1))
    only0 = [b0[key] for key in sorted(set(b0) - set(b1))]
    only1 = [b1[key] for key in sorted(set(b1) - set(b0))]
    rows = [{"match_type": "common", "b0_entry": key, "b1_entry": key, "b0_net_r": b0[key]["net_r"], "b1_net_r": b1[key]["net_r"]} for key in common]
    rows += [{"match_type": "B0_only", "b0_entry": trade["entry_time_api"], "b1_entry": "", "b0_net_r": trade["net_r"], "b1_net_r": ""} for trade in only0]
    rows += [{"match_type": "B1_only", "b0_entry": "", "b1_entry": trade["entry_time_api"], "b0_net_r": "", "b1_net_r": trade["net_r"]} for trade in only1]
    def economics(items: list[dict[str, Any]]) -> dict[str, Any]:
        return baseline.reward_metrics(np.asarray([item["net_r"] for item in items], dtype=float))
    return rows, {
        "common_trades": len(common),
        "B0_only_trades": len(only0),
        "B1_only_trades": len(only1),
        "changed_entry_pairs": 0,
        "matching_rule": "exact executable entry timestamp; unmatched entries remain model-only",
        "B0_only_trade_economics": economics(only0),
        "B1_only_trade_economics": economics(only1),
    }


def event_state_attribution(scored: pd.DataFrame, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    windows = (("0_15", 0, 15), ("15_60", 15, 60), ("60_240", 60, 240), ("over_240", 240, math.inf))
    minutes = scored["EVENT_MINUTES_RAW"].to_numpy(dtype=float)
    rewards = scored["C1_NET_R"].to_numpy(dtype=float)
    for name, lower, upper in windows:
        selected = np.isfinite(minutes) & (minutes >= lower) & (minutes < upper)
        rows.append({"scope": "standalone_rows", "group": name, "event_type": "ALL", "observations_or_trades": int(selected.sum()), **baseline.reward_metrics(rewards[selected])})
    for event_type in EVENT_TYPES:
        selected = scored["MATCHED_EVENT_TYPE"].eq(event_type).to_numpy() & np.isfinite(minutes) & (minutes < 240)
        rows.append({"scope": "standalone_rows", "group": "0_240", "event_type": event_type, "observations_or_trades": int(selected.sum()), **baseline.reward_metrics(rewards[selected])})
    for model_id in MODEL_IDS:
        model_trades = [trade for trade in trades if trade["model_id"] == model_id]
        for name, lower, upper in windows:
            selected = [trade for trade in model_trades if trade.get("event_minutes_since") is not None and np.isfinite(float(trade["event_minutes_since"])) and lower <= float(trade["event_minutes_since"]) < upper]
            rows.append({"scope": "executable_trades", "model_id": model_id, "group": name, "event_type": "ALL", "observations_or_trades": len(selected), **baseline.reward_metrics(np.asarray([trade["net_r"] for trade in selected], dtype=float))})
    return rows


def decision(fold_metrics: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = {row["model_id"]: row for row in fold_metrics if row["fold"] == "pooled"}
    b0, b1 = pooled[MODEL_IDS[0]], pooled[MODEL_IDS[1]]
    fold_names = [name for name, _, _ in baseline.FOLDS]
    rank_by = {(row["model_id"], row["fold"]): row for row in ranking}
    positive_spearman = sum(
        (rank_by[(MODEL_IDS[1], name)]["spearman_score_realized_net_r"] or -math.inf)
        > (rank_by[(MODEL_IDS[0], name)]["spearman_score_realized_net_r"] or -math.inf)
        for name in fold_names
    )
    positive_top_mean = sum(rank_by[(MODEL_IDS[1], name)]["top_decile_mean_r"] > rank_by[(MODEL_IDS[0], name)]["top_decile_mean_r"] for name in fold_names)
    positive_top_pf = sum(rank_by[(MODEL_IDS[1], name)]["top_decile_pf"] > rank_by[(MODEL_IDS[0], name)]["top_decile_pf"] for name in fold_names)
    discrimination = positive_spearman >= 2 and positive_top_mean >= 2 and positive_top_pf >= 2
    full_floor = bool(
        b1["realized_wr"] >= 0.60 and b1["pf"] > 1.05 and b1["mean_r"] > 0
        and b1["pnl_r"] > 0 and (b1["break_even_adjusted_edge"] or -1) > 0
        and b1["cost_stress_pf"] > 1.0
        and all(
            row["pf"] >= 0.8 and row["mean_r"] >= -0.10 and row["realized_wr"] >= 0.50
            for row in fold_metrics if row["model_id"] == MODEL_IDS[1] and row["fold"] != "pooled"
        )
    )
    b1_rank = rank_by[(MODEL_IDS[1], "pooled")]
    high_rank_positive = b1_rank["top_decile_mean_r"] > 0 and b1_rank["top_decile_pf"] > 1 and b1_rank["top_quintile_mean_r"] > 0 and b1_rank["top_quintile_pf"] > 1
    if not discrimination:
        classification = "macro_event_timing_failed_cross_regime_incremental_information_gate"
        next_hypothesis = "Acquire a different genuinely external timestamp-aligned information family; do not retune macro-event windows on inspected history."
    elif not high_rank_positive:
        classification = "incremental_discrimination_exists_but_economically_insufficient"
        next_hypothesis = "Validate a different external information family because fixed macro-event high-rank economics remain negative."
    elif not full_floor:
        classification = "positive_high_rank_economics_but_fixed_threshold_s5_not_quality_ready"
        next_hypothesis = "A separately preregistered calibration/ranking experiment using the frozen B1 model only."
    else:
        classification = "development_shadow_candidate_eligible"
        next_hypothesis = "Freeze B1 for a new untouched shadow-forward interval; do not promote production."
    delta = {key: b1[key] - b0[key] for key in ("trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r")}
    return {
        "folds_with_positive_spearman_improvement": positive_spearman,
        "folds_with_top_decile_mean_r_improvement": positive_top_mean,
        "folds_with_top_decile_pf_improvement": positive_top_pf,
        "cross_regime_incremental_discrimination": discrimination,
        "high_rank_economics_positive": high_rank_positive,
        "B1_economic_viability": b1["pf"] > 1 and b1["mean_r"] > 0 and b1["pnl_r"] > 0 and (b1["break_even_adjusted_edge"] or -1) > 0,
        "B1_full_quality_floor": full_floor,
        "shadow_candidate_frozen": False,
        "new_forward_cutoff": None,
        "B1_minus_B0": delta,
        "classification": classification,
        "single_next_research_hypothesis": next_hypothesis,
    }


def save_oof(path: Path, scored: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=scored["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        open=scored["OPEN"].to_numpy(dtype=np.float64), high=scored["HIGH"].to_numpy(dtype=np.float64), low=scored["LOW"].to_numpy(dtype=np.float64), close=scored["CLOSE"].to_numpy(dtype=np.float64),
        atr=scored["ATR"].to_numpy(dtype=np.float64), rsi=scored["M1_RSI"].to_numpy(dtype=np.float64), spread=scored["SPREAD"].to_numpy(dtype=np.float64),
        target=scored["C1_TARGET"].to_numpy(dtype=np.int8), net_r=scored["C1_NET_R"].to_numpy(dtype=np.float64), fold_code=scored["fold_code"].to_numpy(dtype=np.int8),
        score_b0=scored[f"score_{MODEL_IDS[0]}"].to_numpy(dtype=np.float32), score_b1=scored[f"score_{MODEL_IDS[1]}"].to_numpy(dtype=np.float32),
    )


def placeholder_outputs(run_dir: Path, readiness: dict[str, Any]) -> None:
    status = {"status": "not_run_data_readiness_failure", "reason": "; ".join(readiness["failures"])}
    write_json(run_dir / "fold_model_provenance.json", {"models": [], "folds": [], **status})
    schemas = {
        "model_comparison.csv": ["model_id", "fold", "status", "reason"],
        "fold_metrics.csv": ["model_id", "fold", "status", "reason"],
        "information_gain.csv": ["fold", "metric", "B0", "B1", "B1_minus_B0", "status"],
        "probability_diagnostics.csv": ["model_id", "fold", "status", "reason"],
        "ranking_diagnostics.csv": ["model_id", "fold", "status", "reason"],
        "ranking_deciles.csv": ["model_id", "fold", "decile", "status", "reason"],
        "trade_ledger.csv": ["model_id", "trade_id", "entry_time_api", "exit_time_api", "net_r", "status"],
        "trade_identity_comparison.csv": ["match_type", "b0_entry", "b1_entry", "status"],
        "event_state_attribution.csv": ["scope", "group", "event_type", "status", "reason"],
    }
    for name, fields in schemas.items():
        write_csv(run_dir / name, [{"status": status["status"], "reason": status["reason"]}], fields)


def report(metrics: dict[str, Any]) -> str:
    readiness = metrics["data_readiness"]
    lines = [
        f"# {EXPERIMENT}", "", f"Status: `{metrics['run_status']}`", "",
        "## Data-readiness gate", "",
        f"Gate: `{'PASS' if readiness['passed'] else 'FAIL'}`",
        f"Verified events: `{readiness['verified_events']}`; source fetches: `{readiness['source_fetches']}`.", "",
    ]
    if readiness["failures"]:
        lines += ["Failures:", ""] + [f"- {item}" for item in readiness["failures"]] + [""]
    if not readiness["passed"]:
        lines += ["Training was correctly stopped before fitting B0/B1. No partially fabricated or unofficial timestamps were substituted.", ""]
    else:
        decision_data = metrics["decision"]
        lines += [
            "## Result", "",
            f"Classification: `{decision_data['classification']}`",
            f"Cross-regime incremental discrimination: `{decision_data['cross_regime_incremental_discrimination']}`",
            f"B1 full quality floor: `{decision_data['B1_full_quality_floor']}`",
            "", "No production file was modified and no production promotion occurred.", "",
        ]
    lines += ["## Evidence classification", "", "All 2018-2024 folds are repeatedly inspected development evidence, not an untouched final test.", ""]
    return "\n".join(lines)


def finalize_manifest(run_dir: Path, manifest: dict[str, Any], operational_before: dict[str, str], metrics: dict[str, Any], model_inventory: list[dict[str, Any]]) -> None:
    operational_after = {GEMINI_FILE.name: baseline.sha256(GEMINI_FILE), OPERATIONAL_MODEL.name: baseline.sha256(OPERATIONAL_MODEL)}
    if operational_before != operational_after:
        raise RuntimeError("Operational artifacts changed")
    readiness = metrics["data_readiness"]
    manifest = baseline.read_json(run_dir / "manifest.json")
    manifest["data"].update({
        "symbols": ["GOLD#"],
        "data_sources": ["official BLS release documents", "official BEA release documents", "official Federal Reserve FOMC documents", "repository-local XM historical CSV exports"],
        "source_files": baseline.source_inventory(),
        "timezone": "official U.S. release timestamps -> UTC -> audited XM EET/EEST broker wall; evidence remains development-only",
        "data_start_utc": "2015-01-01T00:00:00Z",
        "data_end_utc": "2024-12-31T23:59:59Z",
        "train_start_utc": "2016-07-01T00:00:00 broker wall" if readiness["passed"] else "not_run",
        "train_end_utc": "2022-12-30T19:57:00 broker wall" if readiness["passed"] else "not_run",
        "train_rows": metrics.get("total_train_rows", 0),
        "validation_start_utc": "2018-01-01T00:00:00 broker wall",
        "validation_end_utc": "2025-01-01T00:00:00 broker wall",
        "validation_rows": metrics.get("scored_rows", 0),
        "test_start_utc": "not_applicable_no_untouched_test",
        "test_end_utc": "not_applicable_no_untouched_test",
        "test_rows": 0,
        "purge_details": "fixed C1 exact-exit maturity plus legacy 240-row parity purge, both strictly before each fold start",
        "embargo_details": "strict maturity purge; no additional embargo",
        "raw_snapshot_retained": False,
        "reproducibility_claim": "official response hashes and extracted timestamp dataset retained; upstream pages may change",
        "mt5_fetch": {"used": False, "not_applicable_reason": "fixed historical CSV experiment"},
    })
    manifest["model"].update({
        "trained": readiness["passed"],
        "model_type": "paired fixed XGBoost binary classifiers" if readiness["passed"] else "not_trained_data_readiness_failure",
        "parameters": baseline.FIXED_XGB_PARAMETERS,
        "boosted_rounds_or_estimators": baseline.N_ESTIMATORS,
        "features": ["31 frozen technical features", *EVENT_FEATURES],
        "feature_count": 39 if readiness["passed"] else 0,
        "label_definition": "C1 execution-aligned positive net-R binary target",
        "horizon": 90,
        "label_tp_sl_semantics": "S5 HIGH/LOW first-touch, stop-first same-bar, next-open, observed/fallback costs",
        "execution_tp_sl_semantics": "identical S5 semantics",
        "calibration_method": "none",
        "artifact_path": model_inventory[-1]["path"] if model_inventory else "none",
        "artifact_sha256": model_inventory[-1]["sha256"] if model_inventory else "none",
        "retention_status": "all research fold models inside run" if model_inventory else "no model created because readiness gate failed",
    })
    manifest["search"].update({"performed": False, "predefined_search_space": {"definitions": list(MODEL_IDS)}, "candidate_results_file": "candidates.csv", "selection_metric": "none paired hypothesis test"})
    pooled_b1 = next((row for row in metrics.get("fold_metrics", []) if row["model_id"] == MODEL_IDS[1] and row["fold"] == "pooled"), None)
    manifest["registry"].update({
        "parent_or_incumbent": f"{BASELINE_RUN.name}/C1_execution_aligned_label",
        "selected_configuration": "none",
        "trades_per_day": pooled_b1["trades_per_day"] if pooled_b1 else 0,
        "realized_win_rate": pooled_b1["realized_wr"] if pooled_b1 else 0,
        "pf": pooled_b1["pf"] if pooled_b1 else 0,
        "mean_r": pooled_b1["mean_r"] if pooled_b1 else 0,
        "pnl": pooled_b1["pnl_r"] if pooled_b1 else 0,
        "max_dd": pooled_b1["max_dd_r"] if pooled_b1 else 0,
        "validator_result": "PENDING",
    })
    manifest["promotion"].update({"status": "not_promoted", "gemini_py_changed": False, "operational_model_changed": False, "operational_artifact_changed": False})
    manifest["operational_hashes_after"] = operational_after
    manifest["data_readiness"] = readiness
    manifest["research_decision"] = metrics.get("decision", {"classification": "data_readiness_failure", "shadow_candidate_frozen": False, "new_forward_cutoff": None})
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "FINALIZED.json", "stdout.log"}:
            baseline.add_artifact(manifest, run_dir, path, "research_evidence")
    for item in model_inventory:
        baseline.add_artifact(manifest, run_dir, run_dir / item["path"], "research_fold_model")
    write_json(run_dir / "manifest.json", manifest)


def self_check() -> None:
    sample = "Transmission of material in this release is embargoed until 8:30 a.m. (ET), Friday, January 12, 2018"
    parsed = official_timestamp(sample)
    assert parsed and parsed[0] == datetime(2018, 1, 12, 13, 30, tzinfo=UTC)
    assert utc_to_broker_wall(parsed[0]) == datetime(2018, 1, 12, 15, 30)
    sample = "For release at 2:00 p.m. EDT, Wednesday, June 13, 2018"
    parsed = official_timestamp(sample)
    assert parsed and parsed[0] == datetime(2018, 6, 13, 18, 0, tzinfo=UTC)
    assert len(EVENT_FEATURES) == 8 and len(EVENT_TYPES) == 4
    print("SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    run_dir = args.run_dir.resolve()
    operational_before, manifest = preregister(run_dir)
    self_check()
    definition_path = run_dir / "event_family_definition.json"
    write_json(definition_path, event_definition())
    events, discovered, provenance = acquire_events(run_dir)
    prior_oof = BASELINE_RUN / "paired_oof_predictions.npz"
    with np.load(prior_oof, allow_pickle=False) as data:
        baseline_times = pd.DatetimeIndex(pd.to_datetime(data["time_ns"].astype(np.int64)))
    coverage_rows, readiness_pass, failures = coverage_audit(events, discovered, baseline_times)
    write_csv(run_dir / "event_coverage.csv", coverage_rows)
    readiness = {"passed": readiness_pass, "failures": failures, "verified_events": len(events), "identified_events": len(discovered), "source_fetches": len(provenance), "minimum_coverage": MIN_COVERAGE}
    model_inventory: list[dict[str, Any]] = []
    if not readiness_pass:
        write_csv(run_dir / "feature_alignment_audit.csv", [{"status": "not_constructed_data_readiness_failure", "causality_violation": "not_evaluated", "reason": "; ".join(failures)}])
        placeholder_outputs(run_dir, readiness)
        metrics = {"experiment": EXPERIMENT, "run_status": "fail", "data_readiness": readiness, "fold_metrics": [], "decision": {"classification": "data_readiness_failure", "shadow_candidate_frozen": False, "new_forward_cutoff": None, "single_next_research_hypothesis": "Complete reproducible official macro timestamp acquisition before any model comparison."}}
    else:
        drl_trading_v2.DATA_DIR = str(ROOT)
        history, base_features = prepare_barrier_data()
        history = history.copy().reset_index(drop=True)
        if len(base_features) != 31:
            raise RuntimeError(f"Expected 31 baseline features, got {len(base_features)}")
        history, alignment_rows = add_event_features(history, events)
        write_csv(run_dir / "feature_alignment_audit.csv", alignment_rows, ["decision_timestamp_broker_wall", "event_release_timestamp_utc", "event_broker_wall_timestamp", "event_type", "event_release_le_decision", "timezone_conversion"])
        labels = baseline.build_execution_aligned_labels(history)
        scored, fold_provenance, model_inventory = paired_training(history, labels, base_features, run_dir)
        fold_metrics, probability_rows, evaluated = evaluate(scored)
        ranking_rows = evaluated["ranking"]
        information_rows = []
        probability_by = {(row["model_id"], row["fold"]): row for row in probability_rows}
        ranking_by = {(row["model_id"], row["fold"]): row for row in ranking_rows}
        for fold in [name for name, _, _ in baseline.FOLDS] + ["pooled"]:
            for metric_name, source in (("roc_auc", probability_by), ("pr_auc", probability_by), ("brier", probability_by), ("spearman_score_realized_net_r", ranking_by), ("top_decile_mean_r", ranking_by), ("top_decile_pf", ranking_by), ("top_quintile_mean_r", ranking_by), ("top_quintile_pf", ranking_by)):
                b0 = source[(MODEL_IDS[0], fold)][metric_name]
                b1 = source[(MODEL_IDS[1], fold)][metric_name]
                information_rows.append({"fold": fold, "metric": metric_name, "B0": b0, "B1": b1, "B1_minus_B0": b1 - b0})
        identity_rows, marginal_data = marginal(evaluated["ledgers"])
        attribution = event_state_attribution(scored, evaluated["trades"])
        decision_data = decision(fold_metrics, ranking_rows)
        comparison = [*fold_metrics, {"model_id": "B1_minus_B0", "fold": "pooled_delta", **decision_data["B1_minus_B0"]}]
        write_csv(run_dir / "fold_metrics.csv", fold_metrics)
        write_csv(run_dir / "model_comparison.csv", comparison)
        write_csv(run_dir / "information_gain.csv", information_rows)
        write_csv(run_dir / "probability_diagnostics.csv", probability_rows)
        write_csv(run_dir / "ranking_diagnostics.csv", ranking_rows)
        write_csv(run_dir / "ranking_deciles.csv", evaluated["deciles"])
        write_csv(run_dir / "trade_ledger.csv", evaluated["trades"])
        write_csv(run_dir / "trade_identity_comparison.csv", identity_rows, ["match_type", "b0_entry", "b1_entry", "b0_net_r", "b1_net_r"])
        write_csv(run_dir / "event_state_attribution.csv", attribution)
        write_json(run_dir / "fold_model_provenance.json", {"base_features": base_features, "event_features": list(EVENT_FEATURES), "folds": fold_provenance, "models": model_inventory})
        save_oof(run_dir / "paired_oof_predictions.npz", scored)
        metrics = {
            "experiment": EXPERIMENT,
            "run_status": "pass" if decision_data["B1_full_quality_floor"] else ("research_only" if decision_data["cross_regime_incremental_discrimination"] else "fail"),
            "data_readiness": readiness,
            "fold_metrics": fold_metrics,
            "probability_diagnostics": probability_rows,
            "ranking_diagnostics": ranking_rows,
            "information_gain": information_rows,
            "marginal_trade_identity": marginal_data,
            "event_state_attribution": attribution,
            "decision": decision_data,
            "scored_rows": len(scored),
            "total_train_rows": sum(row["train_rows"] for row in fold_provenance),
        }
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(report(metrics), encoding="utf-8")
    with (run_dir / "candidates.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if metrics["fold_metrics"]:
            for model_id in MODEL_IDS:
                row = next(item for item in metrics["fold_metrics"] if item["model_id"] == model_id and item["fold"] == "pooled")
                writer.writerow([model_id, json.dumps({"features": 31 if model_id == MODEL_IDS[0] else 39}), "pooled", row["trades"], row["trades_per_day"], row["tp_first_wr"], row["realized_wr"], row["pf"], row["mean_r"], row["pnl_r"], row["max_dd_r"], row["break_even_wr"], row["break_even_adjusted_edge"], row["cost_stress_pf"], metrics["decision"]["classification"]])
        else:
            writer.writerow(["NO_MODEL", "{}", "none", 0, 0, 0, 0, 0, 0, 0, 0, "", "", "not_run", "data_readiness_failure"])
    finalize_manifest(run_dir, manifest, operational_before, metrics, model_inventory)
    print(f"COMPLETE status={metrics['run_status']} readiness={readiness_pass} events={len(events)} operational_artifact_changed=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
