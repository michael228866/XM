"""Build the frozen, data-only CFTC GOLD COT feature foundation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TIMESTAMP_RUN = (
    ROOT
    / "training_runs"
    / "20260905T171629Z_gemini_macro_event_integration_foundation_v1"
)
TIMESTAMP_ARCHIVE = TIMESTAMP_RUN / "exact_timestamps.npz"
TREASURY_RUN = (
    ROOT
    / "training_runs"
    / "20260909T074821Z_gemini_us_treasury_real_rate_foundation_v1"
)
PARENT_RUN = (
    ROOT
    / "training_runs"
    / "20260903T071729Z_gemini_execution_aligned_label_v1"
)
OPERATIONAL_FILES = (
    ROOT / "gemini.py",
    ROOT / "gold_long_recent_candidate_xgb.json",
)
NY = ZoneInfo("America/New_York")
YEARS = tuple(range(2015, 2025))
TARGET_YEARS = tuple(range(2016, 2025))
CONTRACT_CODE = "088691"
CONTRACT_NAME = "GOLD - COMMODITY EXCHANGE INC."
SOURCE_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_{year}.zip"
FEATURE_NAMES = (
    "COT_MM_NET_PCT_OI",
    "COT_MM_NET_CHG_1W",
    "COT_PROD_NET_PCT_OI",
    "COT_SWAP_NET_PCT_OI",
    "COT_MM_VS_PROD_SPREAD",
)
BLOCKS = {
    "fold1_train": (530218, "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6"),
    "fold1_score": (1058080, "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94"),
    "fold2_train": (532563, "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44"),
    "fold2_score": (708197, "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8"),
    "fold3_train": (533580, "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b"),
    "fold3_score": (708020, "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449"),
}
ALIASES = {
    "contract_name": {
        "marketandexchangenames",
        "marketandexchangename",
        "contractmarketname",
    },
    "report_date": {
        "reportdateasyyyymmdd",
        "asofdateinformyyyymmdd",
        "asofdateinformyymmdd",
    },
    "contract_code": {"cftccontractmarketcode"},
    "open_interest": {"openinterestall"},
    "producer_long": {
        "prodmercpositionslongall",
        "producermerchantpositionslongall",
    },
    "producer_short": {
        "prodmercpositionsshortall",
        "producermerchantpositionsshortall",
    },
    "swap_long": {"swappositionslongall", "swapdealerpositionslongall"},
    "swap_short": {"swappositionsshortall", "swapdealerpositionsshortall"},
    "managed_money_long": {
        "mmoneypositionslongall",
        "managedmoneypositionslongall",
    },
    "managed_money_short": {
        "mmoneypositionsshortall",
        "managedmoneypositionsshortall",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    payload = (
        str(values.dtype).encode("ascii")
        + np.asarray(values.shape, dtype=np.int64).tobytes()
        + values.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def logical_matrix_hash(matrices: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"CFTC_GOLD_COT_FOUNDATION_V1\0")
    digest.update(json.dumps(FEATURE_NAMES, separators=(",", ":")).encode("ascii"))
    for block in BLOCKS:
        matrix = np.ascontiguousarray(matrices[block], dtype=np.float64)
        digest.update(b"\0" + block.encode("ascii") + b"\0")
        digest.update(str(matrix.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize_field(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return text
    return text.zfill(6)


def normalize_name(value: Any) -> str:
    return " ".join(str(value).upper().split())


def map_schema(columns: list[str]) -> dict[str, str]:
    normalized: dict[str, list[str]] = {}
    for column in columns:
        normalized.setdefault(normalize_field(column), []).append(column)
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        matches = [
            original
            for alias in aliases
            for original in normalized.get(alias, [])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Schema field {canonical!r} requires exactly one alias match; "
                f"found {matches!r} in {columns!r}"
            )
        mapping[canonical] = matches[0]
    return mapping


def parse_report_date(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    if parsed.isna().any():
        bad = text[parsed.isna()].head(5).tolist()
        raise ValueError(f"Unparseable CFTC report dates: {bad}")
    return parsed.dt.normalize()


def availability_ns(report_date: pd.Timestamp) -> int:
    day = report_date.date() + timedelta(days=4)
    local = datetime(day.year, day.month, day.day, tzinfo=NY)
    return int(local.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def verify_git_gate(run_dir: Path, expected_commit: str) -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    upstream = git_output("rev-parse", "@{u}")
    if head != upstream:
        raise RuntimeError(f"Git gate failed: HEAD {head} != upstream {upstream}")
    if expected_commit != head:
        raise RuntimeError(
            f"Git gate failed: manifest commit {expected_commit} != HEAD {head}"
        )
    allowed = run_dir.resolve().relative_to(ROOT.resolve()).as_posix() + "/"
    unexpected = []
    for line in git_output("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"').replace("\\", "/")
        if not (path == allowed.rstrip("/") or path.startswith(allowed)):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "Git gate failed: unexpected dirty paths exist: " + "; ".join(unexpected)
        )
    return {"head": head, "upstream": upstream, "unexpected_dirty_paths": []}


def verify_finalized_archive(run: Path) -> None:
    final = json.loads((run / "FINALIZED.json").read_text(encoding="utf-8"))
    failures = [
        relative
        for relative, expected in final["file_sha256"].items()
        if not (run / relative).is_file()
        or sha256_file(run / relative) != expected
    ]
    if failures:
        raise RuntimeError(f"Prior finalized archive changed: {run.name}: {failures}")


def download(url: str, destination: Path, attempts: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "XM-GOLD-CFTC-Foundation/1.0 (reproducible research)",
            "Accept": "application/zip,application/octet-stream,*/*",
        },
    )
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP {status}")
                body = response.read()
            if not body.startswith(b"PK"):
                raise ValueError("response is not a ZIP archive")
            destination.write_bytes(body)
            return {
                "url": url,
                "http_status": status,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempts": attempt,
            }
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"Official CFTC acquisition failed for {url}: {errors}")


def read_annual_zip(path: Path) -> tuple[pd.DataFrame, str, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in {".txt", ".csv"}
        ]
        if len(members) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one TXT/CSV member, found {members}"
            )
        member = members[0]
        raw = archive.read(member)
    frame = pd.read_csv(
        io.BytesIO(raw), dtype="string", keep_default_na=True, low_memory=False
    )
    mapping = map_schema(frame.columns.tolist())
    return frame, member, mapping


def canonicalize_gold(
    frame: pd.DataFrame, mapping: dict[str, str], source_year: int
) -> pd.DataFrame:
    selected = frame[
        frame[mapping["contract_code"]].map(normalize_code) == CONTRACT_CODE
    ].copy()
    if selected.empty:
        raise ValueError(f"No GOLD contract code {CONTRACT_CODE} in source year {source_year}")
    names = selected[mapping["contract_name"]].map(normalize_name).unique().tolist()
    if names != [CONTRACT_NAME]:
        raise ValueError(
            f"Unexpected name(s) for contract {CONTRACT_CODE} in {source_year}: {names}"
        )
    result = pd.DataFrame(
        {
            "report_date": parse_report_date(selected[mapping["report_date"]]),
            "contract_code": selected[mapping["contract_code"]].map(normalize_code),
            "contract_name": selected[mapping["contract_name"]].map(normalize_name),
            "source_year": source_year,
        }
    )
    numeric = (
        "open_interest",
        "producer_long",
        "producer_short",
        "swap_long",
        "swap_short",
        "managed_money_long",
        "managed_money_short",
    )
    for field in numeric:
        result[field] = pd.to_numeric(
            selected[mapping[field]].str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    return result


def build_observations(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    observations = pd.concat(frames, ignore_index=True).sort_values(
        ["report_date", "source_year"], ignore_index=True
    )
    duplicate_dates = observations[observations.duplicated("report_date", keep=False)]
    numeric = [
        "open_interest",
        "producer_long",
        "producer_short",
        "swap_long",
        "swap_short",
        "managed_money_long",
        "managed_money_short",
    ]
    required = observations[observations.report_date.dt.year.isin(TARGET_YEARS)]
    missing_counts = {field: int(required[field].isna().sum()) for field in numeric}
    nonpositive_oi = int((required.open_interest <= 0).sum())
    ordering = bool(observations.report_date.is_monotonic_increasing)
    gap_days = observations.report_date.diff().dt.days
    gaps = [
        {
            "previous_report_date": observations.iloc[index - 1].report_date.date().isoformat(),
            "report_date": observations.iloc[index].report_date.date().isoformat(),
            "calendar_gap_days": int(gap),
            "classification": "official_observation_gap",
        }
        for index, gap in enumerate(gap_days)
        if index and pd.notna(gap) and gap > 10
    ]
    audit = {
        "target_start": "2016-01-01",
        "target_end": "2024-12-31",
        "target_observations": int(len(required)),
        "prehistory_observations": int((observations.report_date.dt.year < 2016).sum()),
        "duplicate_report_dates": int(len(duplicate_dates)),
        "annual_overlap_duplicates": int(len(duplicate_dates)),
        "missing_required_numeric": missing_counts,
        "nonpositive_open_interest": nonpositive_oi,
        "observation_ordering": ordering,
        "unexpected_contract_code": int((observations.contract_code != CONTRACT_CODE).sum()),
        "unexpected_contract_name": int((observations.contract_name != CONTRACT_NAME).sum()),
        "long_observation_gaps": gaps,
    }
    if len(required) == 0 or not (required.report_date.dt.year == 2016).any():
        raise ValueError("Required 2016-2024 GOLD observations are incomplete")
    if not (observations.report_date.dt.year < 2016).any():
        raise ValueError("Pre-2016 observation required for first 2016 weekly change")
    if (
        audit["duplicate_report_dates"]
        or any(missing_counts.values())
        or nonpositive_oi
        or not ordering
        or audit["unexpected_contract_code"]
        or audit["unexpected_contract_name"]
    ):
        raise ValueError(f"CFTC observation integrity failed: {audit}")
    observations["COT_MM_NET_PCT_OI"] = (
        observations.managed_money_long - observations.managed_money_short
    ) / observations.open_interest
    observations["COT_MM_NET_CHG_1W"] = observations.COT_MM_NET_PCT_OI.diff()
    observations["COT_PROD_NET_PCT_OI"] = (
        observations.producer_long - observations.producer_short
    ) / observations.open_interest
    observations["COT_SWAP_NET_PCT_OI"] = (
        observations.swap_long - observations.swap_short
    ) / observations.open_interest
    observations["COT_MM_VS_PROD_SPREAD"] = (
        observations.COT_MM_NET_PCT_OI - observations.COT_PROD_NET_PCT_OI
    )
    observations["availability_time_utc"] = observations.report_date.map(
        lambda value: pd.Timestamp(availability_ns(value), unit="ns", tz="UTC")
    )
    first_2016 = observations.index[observations.report_date.dt.year == 2016][0]
    audit["first_2016_change_has_predecessor"] = bool(
        math.isfinite(observations.loc[first_2016, "COT_MM_NET_CHG_1W"])
    )
    audit["all_derived_features_finite_2016_2024"] = bool(
        np.isfinite(required_indexed(observations)[list(FEATURE_NAMES)].to_numpy()).all()
    )
    if not audit["first_2016_change_has_predecessor"] or not audit[
        "all_derived_features_finite_2016_2024"
    ]:
        raise ValueError("Derived CFTC features are not finite for all 2016-2024 observations")
    return observations, audit


def required_indexed(observations: pd.DataFrame) -> pd.DataFrame:
    return observations[observations.report_date.dt.year.isin(TARGET_YEARS)]


def load_timestamps() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    with np.load(TIMESTAMP_ARCHIVE, allow_pickle=False) as source:
        for block, (expected_rows, expected_hash) in BLOCKS.items():
            broker = source[f"{block}_broker_ns"].astype(np.int64)
            utc = source[f"{block}_utc_ns"].astype(np.int64)
            actual_hash = array_hash(broker)
            passed = len(broker) == expected_rows and actual_hash == expected_hash
            if not passed:
                raise ValueError(
                    f"Frozen timestamp identity failed for {block}: "
                    f"rows={len(broker)}, hash={actual_hash}"
                )
            arrays[block] = utc
            audit[block] = {
                "rows": len(utc),
                "broker_timestamp_sha256": actual_hash,
                "expected_broker_timestamp_sha256": expected_hash,
                "identity_pass": passed,
                "first_utc": pd.Timestamp(int(utc[0]), unit="ns", tz="UTC").isoformat(),
                "last_utc": pd.Timestamp(int(utc[-1]), unit="ns", tz="UTC").isoformat(),
            }
    return arrays, audit


def build_matrices(
    observations: pd.DataFrame, timestamps: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
    availability = observations.availability_time_utc.array.asi8
    values = observations[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    report_dates = observations.report_date.to_numpy()
    matrices: dict[str, np.ndarray] = {}
    coverage: list[dict[str, Any]] = []
    unresolved_total = 0
    for block, utc in timestamps.items():
        indices = np.searchsorted(availability, utc, side="right") - 1
        valid = indices >= 0
        matrix = np.full((len(utc), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
        matrix[valid] = values[indices[valid]]
        future_leak = int(
            (availability[indices[valid]] > utc[valid]).sum()
        )
        unresolved = int((~np.isfinite(matrix).all(axis=1)).sum())
        unresolved_total += unresolved
        matrices[block] = matrix
        coverage.append(
            {
                "block": block,
                "total_gold_timestamps": len(utc),
                "constructable_rows": int(np.isfinite(matrix).all(axis=1).sum()),
                "unresolved_rows": unresolved,
                "future_leak_rows": future_leak,
                "first_effective_report_date": (
                    pd.Timestamp(report_dates[indices[valid][0]]).date().isoformat()
                    if valid.any()
                    else None
                ),
                "last_effective_report_date": (
                    pd.Timestamp(report_dates[indices[valid][-1]]).date().isoformat()
                    if valid.any()
                    else None
                ),
                "finite_pct_all_five": float(
                    np.isfinite(matrix).all(axis=1).mean() * 100
                ),
                "matrix_sha256": array_hash(matrix),
            }
        )
        if future_leak:
            raise RuntimeError(f"Future availability leakage detected in {block}")
    return matrices, coverage, unresolved_total


def add_artifact(manifest: dict[str, Any], run_dir: Path, path: Path, kind: str) -> None:
    relative = path.relative_to(run_dir).as_posix()
    record = {
        "kind": kind,
        "path": relative,
        "sha256": sha256_file(path),
        "retention_status": "stored_in_run_directory",
    }
    manifest.setdefault("artifacts", [])[:] = [
        value for value in manifest.get("artifacts", []) if value.get("path") != relative
    ]
    manifest["artifacts"].append(record)


def run_foundation(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError("--run-dir must be created by training_run_history.py create")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_name") != "gemini_cftc_gold_cot_foundation_v1":
        raise ValueError("Unexpected formal experiment_name")
    if manifest.get("status") != "in_progress":
        raise ValueError("Formal run must be in_progress")
    verify_finalized_archive(TREASURY_RUN)
    verify_finalized_archive(PARENT_RUN)
    git_gate = verify_git_gate(run_dir, str(manifest.get("git_commit")))
    production_before = {path.name: sha256_file(path) for path in OPERATIONAL_FILES}
    write_json(run_dir / "operational_safety_pre.json", production_before)

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=False)
    source_records: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    parse_failures: list[dict[str, Any]] = []
    for year in YEARS:
        path = raw_dir / f"fut_disagg_txt_hist_{year}.zip"
        url = SOURCE_URL.format(year=year)
        record = download(url, path)
        record.update({"year": year, "path": path.relative_to(run_dir).as_posix()})
        source_records.append(record)
        try:
            raw_frame, member, mapping = read_annual_zip(path)
            canonical = canonicalize_gold(raw_frame, mapping, year)
            annual_frames.append(canonical)
            schemas.append(
                {
                    "year": year,
                    "source_member": member,
                    "raw_columns": raw_frame.columns.tolist(),
                    "mapping": [
                        {
                            "canonical_field": canonical_field,
                            "official_raw_field": raw_field,
                            "raw_dtype": str(raw_frame[raw_field].dtype),
                            "mapping_rule": (
                                "unique normalized alias match; zero or multiple "
                                "matches fail"
                            ),
                        }
                        for canonical_field, raw_field in mapping.items()
                    ],
                }
            )
        except Exception as exc:
            parse_failures.append(
                {"year": year, "error": f"{type(exc).__name__}: {exc}"}
            )
    write_json(run_dir / "source_hashes.json", source_records)
    write_json(run_dir / "source_schema.json", schemas)
    if parse_failures:
        write_json(run_dir / "source_parse_failures.json", parse_failures)
        raise RuntimeError(f"Official CFTC source parsing failed: {parse_failures}")

    observations, integrity = build_observations(annual_frames)
    observations.to_csv(
        run_dir / "cot_observations.csv", index=False, float_format="%.17g"
    )
    feature_columns = ["report_date", "availability_time_utc", *FEATURE_NAMES]
    required_indexed(observations)[feature_columns].to_csv(
        run_dir / "cftc_gold_cot_features.csv", index=False, float_format="%.17g"
    )
    timestamps, timestamp_audit = load_timestamps()
    matrices, coverage, unresolved = build_matrices(observations, timestamps)
    matrix_path = run_dir / "cftc_gold_cot_features.npz"
    np.savez_compressed(
        matrix_path,
        feature_names=np.asarray(FEATURE_NAMES),
        **matrices,
    )
    logical_hash = logical_matrix_hash(matrices)
    physical_hash = sha256_file(matrix_path)
    source_dataset_hash = sha256_file(run_dir / "cot_observations.csv")

    identity_rows = [
        {
            "check": "official_cftc_only",
            "value": "PASS",
            "detail": SOURCE_URL,
        },
        {
            "check": "gold_contract_identity",
            "value": "PASS",
            "detail": f"{CONTRACT_CODE} {CONTRACT_NAME}",
        },
        {
            "check": "source_hash_provenance",
            "value": "PASS",
            "detail": f"{len(source_records)} retained annual ZIP files",
        },
        {"check": "schema_audit", "value": "PASS", "detail": "unique aliases"},
        {
            "check": "observation_integrity",
            "value": "PASS",
            "detail": f"{integrity['target_observations']} target observations",
        },
        {
            "check": "six_gold_timestamp_hashes",
            "value": "PASS",
            "detail": "all expected row counts and hashes matched",
        },
        {
            "check": "causal_availability",
            "value": "PASS",
            "detail": "Report_Date +4 calendar days 00:00 America/New_York",
        },
        {"check": "future_leak_rows", "value": "0", "detail": "backward as-of"},
        {
            "check": "unresolved_source_rows",
            "value": str(unresolved),
            "detail": "all six blocks",
        },
        {
            "check": "all_5_features_finite",
            "value": "PASS" if unresolved == 0 else "FAIL",
            "detail": "float64",
        },
        {
            "check": "all_6_blocks_constructable",
            "value": "PASS" if unresolved == 0 else "FAIL",
            "detail": "exact timestamp rows",
        },
    ]
    pd.DataFrame(identity_rows).to_csv(run_dir / "identity_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(run_dir / "coverage_audit.csv", index=False)

    internal_pass = unresolved == 0
    production_after = {path.name: sha256_file(path) for path in OPERATIONAL_FILES}
    if production_before != production_after:
        raise RuntimeError("Operational file changed during data-foundation execution")
    metrics = {
        "summary": {
            "internal_methodology": "PASS" if internal_pass else "FAIL",
            "data_certification": "PASS" if internal_pass else "FAIL",
            "independent_validator": "PENDING",
            "data_foundation_ready": False,
            "unresolved_source_rows": unresolved,
            "future_leak_rows": 0,
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_dtype": "float64",
        "source_audit": integrity,
        "timestamp_audit": timestamp_audit,
        "coverage": coverage,
        "source_dataset_sha256": source_dataset_hash,
        "logical_feature_matrix_sha256": logical_hash,
        "physical_npz_sha256": physical_hash,
        "git_gate": git_gate,
        "operational_safety": {
            "before": production_before,
            "after": production_after,
            "gemini_py_changed": False,
            "operational_model_changed": False,
            "operational_artifact_changed": False,
        },
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
    }
    write_json(run_dir / "metrics.json", metrics)
    report = [
        "# CFTC GOLD COT POSITIONING FOUNDATION V1",
        "",
        "Status: **PENDING INDEPENDENT VALIDATOR**",
        "",
        "DATA-ONLY. No labels, predictions, trades, outcomes, or model training were accessed.",
        "",
        f"- Internal methodology: **{metrics['summary']['internal_methodology']}**",
        f"- Data certification: **{metrics['summary']['data_certification']}**",
        "- Independent validator: **PENDING**",
        f"- Official annual ZIP files retained: **{len(source_records)}**",
        f"- 2016-2024 GOLD observations: **{integrity['target_observations']}**",
        f"- Unresolved source rows: **{unresolved}**",
        f"- Future-leak rows: **0**",
        f"- Logical matrix SHA-256: `{logical_hash}`",
        f"- Physical NPZ SHA-256: `{physical_hash}`",
        "",
        "## Frozen feature order",
        "",
        *[f"- `{name}`" for name in FEATURE_NAMES],
        "",
        "## Coverage",
        "",
        *[
            f"- {row['block']}: {row['constructable_rows']:,}/"
            f"{row['total_gold_timestamps']:,}; finite {row['finite_pct_all_five']:.6f}%"
            for row in coverage
        ],
        "",
        "**DATA FOUNDATION READY = NO (independent validator pending)**",
        "",
        "Model training performed: NO",
        "Strategy evaluation performed: NO",
    ]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest["git_commit"] = git_gate["head"]
    manifest["git_dirty"] = False
    manifest["data"].update(
        {
            "symbols": ["CFTC GOLD 088691", "GOLD# timestamps only"],
            "data_sources": [
                "Official CFTC annual Disaggregated Commitments of Traders Futures Only ZIP files"
            ],
            "source_files": [
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "retention_status": "stored_in_run_directory",
                    "source_url": record["url"],
                    "acquired_at_utc": record["acquired_at_utc"],
                }
                for record in source_records
            ]
            + [
                {
                    "path": str(TIMESTAMP_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(TIMESTAMP_ARCHIVE),
                    "retention_status": "retained_in_prior_finalized_run; not duplicated",
                }
            ],
            "timezone": (
                "CFTC Report_Date +4 calendar days 00:00 America/New_York "
                "converted to UTC; GOLD exact UTC inherited"
            ),
            "data_start_utc": observations.availability_time_utc.min().isoformat(),
            "data_end_utc": observations.availability_time_utc.max().isoformat(),
            "train_start_utc": min(
                row["first_utc"]
                for name, row in timestamp_audit.items()
                if name.endswith("_train")
            ),
            "train_end_utc": max(
                row["last_utc"] for name, row in timestamp_audit.items() if name.endswith("_train")
            ),
            "train_rows": sum(BLOCKS[name][0] for name in BLOCKS if name.endswith("_train")),
            "validation_start_utc": min(
                row["first_utc"]
                for name, row in timestamp_audit.items()
                if name.endswith("_score")
            ),
            "validation_end_utc": max(
                row["last_utc"] for name, row in timestamp_audit.items() if name.endswith("_score")
            ),
            "validation_rows": sum(BLOCKS[name][0] for name in BLOCKS if name.endswith("_score")),
            "test_start_utc": "not_applicable_data_only",
            "test_end_utc": "not_applicable_data_only",
            "test_rows": "not_applicable_data_only",
            "purge_details": "not_applicable_data_only",
            "embargo_details": "not_applicable_data_only",
            "raw_snapshot_retained": True,
            "reproducibility_claim": (
                "Full for retained official CFTC ZIP files, frozen formulas, "
                "and referenced finalized timestamp archive"
            ),
            "mt5_fetch": {
                "used": False,
                "terminal_path": None,
                "terminal_info": None,
                "broker_info": None,
                "fetch_start_utc": None,
                "fetch_end_utc": None,
                "retrieved_at_utc": None,
                "returned_rows": None,
                "not_applicable_reason": (
                    "CFTC data acquired from official CFTC sources; frozen GOLD "
                    "timestamps inherited from finalized repository evidence; "
                    "no MT5 API fetch performed."
                ),
            },
        }
    )
    manifest["model"].update(
        {
            "trained": False,
            "not_applicable_reason": "Data-foundation-only run; no model was trained.",
        }
    )
    manifest["search"].update(
        {
            "performed": False,
            "not_applicable_reason": (
                "Frozen five-feature CFTC GOLD COT data foundation; no feature "
                "or parameter search."
            ),
        }
    )
    manifest["registry"].update(
        {
            "parent_or_incumbent": PARENT_RUN.name,
            "selected_configuration": (
                "frozen five-feature CFTC GOLD Disaggregated Futures Only data "
                "foundation"
            ),
            "trades_per_day": "not_applicable_data_only",
            "realized_win_rate": "not_applicable_data_only",
            "pf": "not_applicable_data_only",
            "mean_r": "not_applicable_data_only",
            "pnl": "not_applicable_data_only",
            "max_dd": "not_applicable_data_only",
            "validator_result": "PENDING independent data validator",
        }
    )
    manifest["promotion"] = {
        "requested": False,
        "gate_result": "not_applicable_data_only",
        "replacement_authorized": False,
        "operational_artifact_changed": False,
    }
    for path, kind in (
        (run_dir / "source_hashes.json", "source_provenance"),
        (run_dir / "source_schema.json", "source_schema"),
        (run_dir / "cot_observations.csv", "canonical_observations"),
        (run_dir / "cftc_gold_cot_features.csv", "weekly_features"),
        (matrix_path, "feature_matrix"),
        (run_dir / "identity_audit.csv", "identity_audit"),
        (run_dir / "coverage_audit.csv", "coverage_audit"),
        (run_dir / "operational_safety_pre.json", "operational_safety"),
        (run_dir / "metrics.json", "metrics"),
        (run_dir / "report.md", "report"),
    ):
        add_artifact(manifest, run_dir, path, kind)
    write_json(run_dir / "manifest.json", manifest)
    print("CFTC_FOUNDATION_INTERNAL_PASS" if internal_pass else "CFTC_FOUNDATION_INTERNAL_FAIL")
    print(f"run_dir={run_dir}")
    print(f"logical_feature_matrix_sha256={logical_hash}")


def self_test() -> None:
    columns = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code",
        "Open_Interest_All",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "Swap__Positions_Long_All",
        "Swap__Positions_Short_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
    ]
    mapping = map_schema(columns)
    assert mapping["swap_short"] == "Swap__Positions_Short_All"
    winter = availability_ns(pd.Timestamp("2024-01-02"))
    summer = availability_ns(pd.Timestamp("2024-07-02"))
    assert pd.Timestamp(winter, unit="ns", tz="UTC").hour == 5
    assert pd.Timestamp(summer, unit="ns", tz="UTC").hour == 4
    a = {name: np.zeros((2, 5), dtype=np.float64) for name in BLOCKS}
    assert logical_matrix_hash(a) == logical_matrix_hash(a)
    try:
        map_schema(columns + ["Managed_Money_Positions_Long_All"])
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous schema aliases must fail")
    raw = pd.DataFrame(
        [
            [
                CONTRACT_NAME,
                "2015-12-29",
                CONTRACT_CODE,
                "100",
                "10",
                "20",
                "30",
                "20",
                "40",
                "10",
            ],
            [
                CONTRACT_NAME,
                "2016-01-05",
                CONTRACT_CODE,
                "200",
                "20",
                "30",
                "40",
                "20",
                "80",
                "20",
            ],
        ],
        columns=columns,
        dtype="string",
    )
    observations, audit = build_observations([canonicalize_gold(raw, mapping, 2015)])
    assert audit["first_2016_change_has_predecessor"]
    assert observations.iloc[-1].COT_MM_NET_PCT_OI == 0.3
    assert observations.iloc[-1].COT_MM_NET_CHG_1W == 0.0
    decision_time = np.asarray(
        [availability_ns(pd.Timestamp("2016-01-05"))], dtype=np.int64
    )
    matrices, coverage, unresolved = build_matrices(
        observations, {"synthetic": decision_time}
    )
    assert unresolved == 0 and coverage[0]["future_leak_rows"] == 0
    assert matrices["synthetic"][0, 0] == 0.3
    print("SELF_TEST_PASS")


def record_failure(run_dir: Path, error: Exception) -> None:
    """Leave a finalizable failure record without masking the original error."""
    run_dir = run_dir.resolve()
    if not (run_dir / "manifest.json").is_file():
        return
    failure = {
        "stage": "data_foundation",
        "error_type": type(error).__name__,
        "error": str(error),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
    }
    failure_path = run_dir / "source_acquisition_or_build_failure.json"
    write_json(failure_path, failure)
    write_json(
        run_dir / "metrics.json",
        {
            "summary": {
                "internal_methodology": "FAIL",
                "data_certification": "FAIL",
                "independent_validator": "NOT_RUN",
                "data_foundation_ready": False,
                "failure": failure,
            },
            "model_training_performed": False,
            "strategy_evaluation_performed": False,
        },
    )
    (run_dir / "report.md").write_text(
        "# CFTC GOLD COT POSITIONING FOUNDATION V1\n\n"
        "Status: **FAIL**\n\n"
        "The data-only foundation stopped without substituting another provider.\n\n"
        f"- Failure stage: `{failure['stage']}`\n"
        f"- Failure: `{failure['error_type']}: {failure['error']}`\n"
        "- DATA FOUNDATION READY = NO\n"
        "- Model training performed: NO\n"
        "- Strategy evaluation performed: NO\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safety_path = run_dir / "operational_safety_pre.json"
    production_before = (
        json.loads(safety_path.read_text(encoding="utf-8"))
        if safety_path.is_file()
        else None
    )
    production_after = {
        path.name: sha256_file(path) for path in OPERATIONAL_FILES if path.is_file()
    }
    failure["operational_safety"] = {
        "before": production_before,
        "after": production_after,
        "unchanged": (
            production_before == production_after
            if production_before
            else "not_established"
        ),
    }
    write_json(failure_path, failure)
    retained_raw = sorted((run_dir / "raw").glob("*.zip")) if (run_dir / "raw").is_dir() else []
    source_files = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(path),
            "retention_status": "stored_in_run_directory_partial_acquisition",
        }
        for path in retained_raw
    ]
    source_files.append(
        {
            "path": failure_path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(failure_path),
            "retention_status": "stored_in_run_directory_failure_evidence",
        }
    )
    manifest["data"].update(
        {
            "symbols": ["CFTC GOLD 088691", "GOLD# timestamps only"],
            "data_sources": ["Official CFTC Disaggregated Futures Only acquisition attempt"],
            "source_files": source_files,
            "timezone": "CFTC Report_Date +4 calendar days at 00:00 America/New_York; data-only",
            "data_start_utc": "not_available_foundation_failed",
            "data_end_utc": "not_available_foundation_failed",
            "train_start_utc": "not_applicable_data_only_failed",
            "train_end_utc": "not_applicable_data_only_failed",
            "train_rows": 0,
            "validation_start_utc": "not_applicable_data_only_failed",
            "validation_end_utc": "not_applicable_data_only_failed",
            "validation_rows": 0,
            "test_start_utc": "not_applicable_data_only",
            "test_end_utc": "not_applicable_data_only",
            "test_rows": 0,
            "purge_details": "not_applicable_data_only",
            "embargo_details": "not_applicable_data_only",
            "raw_snapshot_retained": bool(retained_raw),
            "reproducibility_claim": (
                "Failed acquisition/build evidence retained; complete data "
                "foundation not constructed"
            ),
            "mt5_fetch": {
                "used": False,
                "terminal_path": None,
                "terminal_info": None,
                "broker_info": None,
                "fetch_start_utc": None,
                "fetch_end_utc": None,
                "retrieved_at_utc": None,
                "returned_rows": None,
                "not_applicable_reason": "Official CFTC source task; no MT5 API fetch performed.",
            },
        }
    )
    manifest["model"].update(
        trained=False,
        not_applicable_reason="Data-foundation-only run; no model was trained.",
    )
    manifest["search"].update(
        performed=False,
        not_applicable_reason="Frozen five-feature foundation; no search.",
    )
    manifest["registry"].update(
        parent_or_incumbent=PARENT_RUN.name,
        selected_configuration="CFTC GOLD COT data foundation failed before certification",
        trades_per_day="not_applicable_data_only",
        realized_win_rate="not_applicable_data_only",
        pf="not_applicable_data_only",
        mean_r="not_applicable_data_only",
        pnl="not_applicable_data_only",
        max_dd="not_applicable_data_only",
        validator_result="NOT_RUN; foundation failed",
    )
    manifest["promotion"] = {
        "requested": False,
        "gate_result": "not_applicable_data_only",
        "replacement_authorized": False,
        "operational_artifact_changed": False,
    }
    for path in (failure_path, run_dir / "metrics.json", run_dir / "report.md"):
        add_artifact(manifest, run_dir, path, "failure_evidence")
    write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")
    try:
        run_foundation(arguments.run_dir)
    except Exception as error:
        record_failure(arguments.run_dir, error)
        raise


if __name__ == "__main__":
    main()
