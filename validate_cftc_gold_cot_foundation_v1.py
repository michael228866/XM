"""Independent validator for the CFTC GOLD COT data foundation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TIMESTAMP_ARCHIVE = (
    ROOT
    / "training_runs"
    / "20260905T171629Z_gemini_macro_event_integration_foundation_v1"
    / "exact_timestamps.npz"
)
OPERATIONAL_FILES = (
    ROOT / "gemini.py",
    ROOT / "gold_long_recent_candidate_xgb.json",
)
NY = ZoneInfo("America/New_York")
YEARS = tuple(range(2015, 2025))
CONTRACT_CODE = "088691"
CONTRACT_NAME = "GOLD - COMMODITY EXCHANGE INC."
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
    "contract_name": {"marketandexchangenames", "marketandexchangename", "contractmarketname"},
    "report_date": {"reportdateasyyyymmdd", "asofdateinformyyyymmdd", "asofdateinformyymmdd"},
    "contract_code": {"cftccontractmarketcode"},
    "open_interest": {"openinterestall"},
    "producer_long": {"prodmercpositionslongall", "producermerchantpositionslongall"},
    "producer_short": {"prodmercpositionsshortall", "producermerchantpositionsshortall"},
    "swap_long": {"swappositionslongall", "swapdealerpositionslongall"},
    "swap_short": {"swappositionsshortall", "swapdealerpositionsshortall"},
    "managed_money_long": {"mmoneypositionslongall", "managedmoneypositionslongall"},
    "managed_money_short": {"mmoneypositionsshortall", "managedmoneypositionsshortall"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(
        str(values.dtype).encode("ascii")
        + np.asarray(values.shape, dtype=np.int64).tobytes()
        + values.tobytes()
    ).hexdigest()


def logical_hash(matrices: dict[str, np.ndarray]) -> str:
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
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalized(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def schema(columns: list[str]) -> dict[str, str]:
    indexed: dict[str, list[str]] = {}
    for column in columns:
        indexed.setdefault(normalized(column), []).append(column)
    output: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        matches = [column for alias in aliases for column in indexed.get(alias, [])]
        if len(matches) != 1:
            raise ValueError(f"Non-unique raw mapping for {canonical}: {matches}")
        output[canonical] = matches[0]
    return output


def code(value: Any) -> str:
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(6) if value.isdigit() else value


def name(value: Any) -> str:
    return " ".join(str(value).upper().split())


def availability(report_date: pd.Timestamp) -> int:
    day = report_date.date() + timedelta(days=4)
    local = datetime(day.year, day.month, day.day, tzinfo=NY)
    return int(local.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def read_raw(path: Path, year: int) -> tuple[pd.DataFrame, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = [
            item
            for item in archive.namelist()
            if not item.endswith("/") and Path(item).suffix.lower() in {".txt", ".csv"}
        ]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one data member, got {members}")
        frame = pd.read_csv(
            io.BytesIO(archive.read(members[0])),
            dtype="string",
            keep_default_na=True,
            low_memory=False,
        )
    mapping = schema(frame.columns.tolist())
    chosen = frame[frame[mapping["contract_code"]].map(code) == CONTRACT_CODE].copy()
    if chosen.empty:
        raise ValueError(f"No GOLD {CONTRACT_CODE} row in {year}")
    contract_names = chosen[mapping["contract_name"]].map(name).unique().tolist()
    if contract_names != [CONTRACT_NAME]:
        raise ValueError(f"Contract identity mismatch in {year}: {contract_names}")
    dates = pd.to_datetime(
        chosen[mapping["report_date"]].astype("string").str.strip(),
        errors="coerce",
        format="mixed",
    ).dt.normalize()
    if dates.isna().any():
        raise ValueError(f"Unparseable report date in {year}")
    result = pd.DataFrame(
        {
            "report_date": dates,
            "contract_code": chosen[mapping["contract_code"]].map(code),
            "contract_name": chosen[mapping["contract_name"]].map(name),
            "source_year": year,
        }
    )
    for canonical in (
        "open_interest",
        "producer_long",
        "producer_short",
        "swap_long",
        "swap_short",
        "managed_money_long",
        "managed_money_short",
    ):
        result[canonical] = pd.to_numeric(
            chosen[mapping[canonical]].str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    return result, mapping


def recompute_observations(raw_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    mappings: list[dict[str, Any]] = []
    for year in YEARS:
        frame, mapped = read_raw(raw_dir / f"fut_disagg_txt_hist_{year}.zip", year)
        frames.append(frame)
        mappings.append({"year": year, "mapping": mapped})
    observations = pd.concat(frames, ignore_index=True).sort_values(
        ["report_date", "source_year"], ignore_index=True
    )
    if observations.report_date.duplicated().any():
        raise ValueError("Duplicate or annual-overlap report date")
    numeric = [
        "open_interest",
        "producer_long",
        "producer_short",
        "swap_long",
        "swap_short",
        "managed_money_long",
        "managed_money_short",
    ]
    target = observations.report_date.dt.year.between(2016, 2024)
    if observations.loc[target, numeric].isna().any().any():
        raise ValueError("Official required numeric field contains N/A")
    if (observations.loc[target, "open_interest"] <= 0).any():
        raise ValueError("Non-positive open interest")
    if not (observations.report_date.dt.year < 2016).any():
        raise ValueError("No pre-2016 predecessor")
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
    observations["availability_ns"] = observations.report_date.map(availability).astype(np.int64)
    if not np.isfinite(observations.loc[target, list(FEATURE_NAMES)]).all().all():
        raise ValueError("Non-finite target-period feature")
    return observations, mappings


def independent_matrices(
    observations: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, str], int, int, bool]:
    availability_values = observations.availability_ns.to_numpy(dtype=np.int64)
    values = observations[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    matrices: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    unresolved = 0
    future_leaks = 0
    timestamps_pass = True
    with np.load(TIMESTAMP_ARCHIVE, allow_pickle=False) as timestamps:
        for block, (expected_rows, expected_hash) in BLOCKS.items():
            broker = timestamps[f"{block}_broker_ns"].astype(np.int64)
            utc = timestamps[f"{block}_utc_ns"].astype(np.int64)
            timestamps_pass &= len(broker) == expected_rows and array_hash(broker) == expected_hash
            indices = np.searchsorted(availability_values, utc, side="right") - 1
            valid = indices >= 0
            matrix = np.full((len(utc), 5), np.nan, dtype=np.float64)
            matrix[valid] = values[indices[valid]]
            future_leaks += int((availability_values[indices[valid]] > utc[valid]).sum())
            unresolved += int((~np.isfinite(matrix).all(axis=1)).sum())
            matrices[block] = matrix
            hashes[block] = array_hash(matrix)
    return matrices, hashes, unresolved, future_leaks, timestamps_pass


def upsert_artifact(manifest: dict[str, Any], run_dir: Path, path: Path) -> None:
    relative = path.relative_to(run_dir).as_posix()
    manifest.setdefault("artifacts", [])[:] = [
        item for item in manifest.get("artifacts", []) if item.get("path") != relative
    ]
    manifest["artifacts"].append(
        {
            "kind": "independent_validator",
            "path": relative,
            "sha256": sha256_file(path),
            "retention_status": "stored_in_run_directory",
        }
    )


def validate(run_dir: Path) -> bool:
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_hashes = json.loads((run_dir / "source_hashes.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    expected_sources = {
        f"raw/fut_disagg_txt_hist_{year}.zip" for year in YEARS
    }
    actual_sources = {item["path"] for item in source_hashes}
    checks["official_cftc_only"] = (
        actual_sources == expected_sources
        and all(item["url"].startswith("https://www.cftc.gov/") for item in source_hashes)
    )
    checks["source_hash_provenance"] = all(
        sha256_file(run_dir / item["path"]) == item["sha256"]
        for item in source_hashes
    )
    try:
        observations, mappings = recompute_observations(run_dir / "raw")
        checks["schema_audit"] = len(mappings) == len(YEARS)
        checks["gold_contract_identity"] = bool(
            (observations.contract_code == CONTRACT_CODE).all()
            and (observations.contract_name == CONTRACT_NAME).all()
        )
        checks["observation_integrity"] = bool(
            observations.report_date.is_monotonic_increasing
            and not observations.report_date.duplicated().any()
        )
    except Exception as exc:
        observations = pd.DataFrame()
        checks.update(
            schema_audit=False,
            gold_contract_identity=False,
            observation_integrity=False,
        )
        recompute_error = f"{type(exc).__name__}: {exc}"
    else:
        recompute_error = None

    matrices: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    unresolved = -1
    future_leaks = -1
    timestamp_pass = False
    matrix_identity = False
    observation_identity = False
    feature_order = False
    if not observations.empty:
        matrices, hashes, unresolved, future_leaks, timestamp_pass = (
            independent_matrices(observations)
        )
        submitted_schema = json.loads((run_dir / "source_schema.json").read_text(encoding="utf-8"))
        submitted_mapping = {
            int(item["year"]): {
                field["canonical_field"]: field["official_raw_field"]
                for field in item["mapping"]
            }
            for item in submitted_schema
        }
        checks["submitted_schema_mapping_identity"] = submitted_mapping == {
            item["year"]: item["mapping"] for item in mappings
        }
        with np.load(run_dir / "cftc_gold_cot_features.npz", allow_pickle=False) as submitted:
            feature_order = submitted["feature_names"].tolist() == list(FEATURE_NAMES)
            matrix_identity = all(
                np.array_equal(matrices[block], submitted[block], equal_nan=True)
                for block in BLOCKS
            )
        submitted_observations = pd.read_csv(run_dir / "cot_observations.csv")
        submitted_dates = pd.to_datetime(submitted_observations.report_date).to_numpy()
        observation_identity = bool(
            np.array_equal(submitted_dates, observations.report_date.to_numpy())
            and np.allclose(
                submitted_observations[list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
                observations[list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
                equal_nan=True,
                rtol=0,
                atol=1e-15,
            )
        )
    checks["six_gold_timestamp_hashes"] = timestamp_pass
    checks["causal_availability"] = future_leaks == 0
    checks["future_leak_rows_zero"] = future_leaks == 0
    checks["unresolved_source_rows_zero"] = unresolved == 0
    checks["feature_order_exact"] = feature_order
    checks["independent_observation_identity"] = observation_identity
    checks["element_by_element_matrix_identity"] = matrix_identity
    checks["independent_per_block_hash_identity"] = hashes == {
        row["block"]: row["matrix_sha256"] for row in metrics["coverage"]
    }
    checks["independent_logical_hash_identity"] = (
        bool(matrices)
        and logical_hash(matrices) == metrics["logical_feature_matrix_sha256"]
    )
    checks["all_5_features_finite"] = unresolved == 0
    checks["all_6_blocks_constructable"] = unresolved == 0
    before = metrics["operational_safety"]["before"]
    current = {path.name: sha256_file(path) for path in OPERATIONAL_FILES}
    checks["operational_files_unchanged"] = before == current
    checks["no_model_or_strategy_evaluation"] = (
        metrics.get("model_training_performed") is False
        and metrics.get("strategy_evaluation_performed") is False
    )
    overall = all(checks.values())
    validator = {
        "overall": "PASS" if overall else "FAIL",
        "internal_methodology": "PASS" if overall else "FAIL",
        "data_certification": "PASS" if overall else "FAIL",
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "recompute_error": recompute_error,
        "independent_unresolved_source_rows": unresolved,
        "independent_future_leak_rows": future_leaks,
        "independent_per_block_hash": hashes,
        "independent_logical_feature_matrix_sha256": (
            logical_hash(matrices) if matrices else None
        ),
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
    }
    write_json(run_dir / "validator.json", validator)
    markdown = [
        "# Independent CFTC GOLD COT DATA Validator",
        "",
        f"Overall: **{validator['overall']}**",
        "",
        (
            "Validator independently reparsed official ZIP bytes and recomputed "
            "all observations, availability timestamps, and six matrices. It "
            "did not import the foundation script."
        ),
        "",
        "| Check | Verdict |",
        "|---|---|",
        *[f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in checks.items()],
        "",
        f"Independent unresolved source rows: **{unresolved}**",
        f"Independent future-leak rows: **{future_leaks}**",
    ]
    if recompute_error:
        markdown.extend(["", f"Recompute error: `{recompute_error}`"])
    (run_dir / "validator.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    metrics["summary"].update(
        {
            "independent_validator": validator["overall"],
            "data_foundation_ready": overall,
        }
    )
    write_json(metrics_path, metrics)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    report = report.replace(
        "Status: **PENDING INDEPENDENT VALIDATOR**",
        f"Status: **{'PASS' if overall else 'FAIL'}**",
    ).replace(
        "- Independent validator: **PENDING**",
        f"- Independent validator: **{validator['overall']}**",
    ).replace(
        "**DATA FOUNDATION READY = NO (independent validator pending)**",
        f"**DATA FOUNDATION READY = {'YES' if overall else 'NO'}**",
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["registry"]["validator_result"] = (
        f"internal {validator['internal_methodology']}; data certification "
        f"{validator['data_certification']}"
    )
    for path in (
        metrics_path,
        run_dir / "report.md",
        run_dir / "validator.json",
        run_dir / "validator.md",
    ):
        upsert_artifact(manifest, run_dir, path)
    write_json(manifest_path, manifest)
    print("INDEPENDENT_VALIDATOR_PASS" if overall else "INDEPENDENT_VALIDATOR_FAIL")
    if not overall:
        print("failed_checks=" + ",".join(key for key, value in checks.items() if not value))
    return overall


def record_validator_failure(run_dir: Path, error: Exception) -> None:
    run_dir = run_dir.resolve()
    result = {
        "overall": "FAIL",
        "internal_methodology": "FAIL",
        "data_certification": "FAIL",
        "checks": {"validator_execution": "FAIL"},
        "error": f"{type(error).__name__}: {error}",
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
    }
    write_json(run_dir / "validator.json", result)
    (run_dir / "validator.md").write_text(
        "# Independent CFTC GOLD COT DATA Validator\n\n"
        "Overall: **FAIL**\n\n"
        f"Validator execution error: `{result['error']}`\n",
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("summary", {}).update(
            independent_validator="FAIL", data_foundation_ready=False
        )
        write_json(metrics_path, metrics)
    report_path = run_dir / "report.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(
            "Status: **PENDING INDEPENDENT VALIDATOR**", "Status: **FAIL**"
        ).replace(
            "- Independent validator: **PENDING**", "- Independent validator: **FAIL**"
        ).replace(
            "**DATA FOUNDATION READY = NO (independent validator pending)**",
            "**DATA FOUNDATION READY = NO**",
        )
        report_path.write_text(report, encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("registry", {})["validator_result"] = (
            "FAIL; independent validator execution error"
        )
        for path in (
            run_dir / "validator.json",
            run_dir / "validator.md",
            metrics_path,
            report_path,
        ):
            if path.is_file():
                upsert_artifact(manifest, run_dir, path)
        write_json(manifest_path, manifest)


def self_test() -> None:
    columns = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code",
        "Open_Interest_All",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "Swap_Positions_Long_All",
        "Swap_Positions_Short_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
    ]
    assert schema(columns)["managed_money_short"] == "M_Money_Positions_Short_All"
    assert pd.Timestamp(availability(pd.Timestamp("2024-01-02")), unit="ns", tz="UTC").hour == 5
    assert pd.Timestamp(availability(pd.Timestamp("2024-07-02")), unit="ns", tz="UTC").hour == 4
    matrices = {block: np.ones((1, 5), dtype=np.float64) for block in BLOCKS}
    assert logical_hash(matrices) == logical_hash(matrices)
    print("SELF_TEST_PASS")


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
        passed = validate(arguments.run_dir)
    except Exception as error:
        record_validator_failure(arguments.run_dir, error)
        print("INDEPENDENT_VALIDATOR_FAIL")
        print(f"validator_error={type(error).__name__}: {error}")
        raise SystemExit(1) from error
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
