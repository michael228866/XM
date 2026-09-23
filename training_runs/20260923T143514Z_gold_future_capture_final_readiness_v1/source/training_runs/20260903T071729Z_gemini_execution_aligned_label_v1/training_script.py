from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import drl_trading_v2
import gold_gemini_core_gate_v1 as core
import gold_gemini_execution_semantics_v1 as semantics
from barrier_classifier_strategy import (
    HORIZON as LEGACY_HORIZON_ROWS,
    LABEL_SL_ATR,
    LABEL_TP_ATR,
    MIN_SL_PRICE as LEGACY_MIN_SL_PRICE,
    MIN_TP_PRICE as LEGACY_MIN_TP_PRICE,
)
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import predict_positive, train_binary_model


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "GEMINI EXECUTION-ALIGNED LABEL MODEL VALIDATION V1"
EXPERIMENT_SLUG = "gemini_execution_aligned_label_v1"
GEMINI_FILE = ROOT / "gemini.py"
OPERATIONAL_MODEL = ROOT / "gold_long_recent_candidate_xgb.json"
S5_SOURCE = ROOT / "gold_gemini_execution_semantics_v1.py"
S5_RUN = ROOT / "training_runs" / "20260903T044116Z_gemini_execution_semantics_v1"

FOLDS = core.FOLDS
MODEL_IDS = ("C0_legacy_label", "C1_execution_aligned_label")
TRAIN_MONTHS = 18
THRESHOLD = 0.75
MIN_ENTRY_RSI = 22.0
EXCLUDED_RSI = (35.0, 45.0)
N_ESTIMATORS = 220
RANDOM_STATE = 42
EXECUTION_HORIZON_MINUTES = 90
POINT = semantics.POINT
BASE_EXTRA_COST_POINTS = semantics.EXTRA_COST_POINTS
STRESS_EXTRA_COST_POINTS = semantics.STRESS_EXTRA_COST_POINTS
FALLBACK_SPREAD_POINTS = semantics.LEGACY_SPREAD_POINTS
PREVIOUS_FORWARD_CUTOFF = "2026-09-01T02:00:00Z"
PREVIOUS_FORWARD_STATUS = "contaminated_for_future_gate_selection"

FIXED_XGB_PARAMETERS = {
    "objective": "binary:logistic",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}
QUALITY_FLOOR = {
    "realized_wr_min": 0.60,
    "pf_strict_min": 1.05,
    "mean_r_strict_min": 0.0,
    "pnl_r_strict_min": 0.0,
    "break_even_edge_strict_min": 0.0,
    "cost_stress_pf_strict_min": 1.00,
}
CATASTROPHIC_FOLD = {
    "minimum_trades": 10,
    "realized_wr_min": 0.50,
    "pf_min": 0.80,
    "mean_r_min": -0.10,
    "max_dd_r_min": -20.0,
}
PROBABILITY_LEVELS = (0.65, 0.70, 0.75, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty evidence table: {path.name}")
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def add_artifact(manifest: dict[str, Any], run_dir: Path, path: Path, kind: str) -> None:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    existing = {item.get("path") for item in manifest.get("artifacts", [])}
    if relative not in existing:
        manifest.setdefault("artifacts", []).append(
            {
                "kind": kind,
                "path": relative,
                "sha256": sha256(path),
                "retention_status": "stored_in_run_directory_git_archival_pending",
            }
        )


def source_inventory() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ROOT.glob("GOLD#_*.csv")):
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "retention_status": (
                    "repository-local historical export excluded from Git; identity retained "
                    "but durable remote raw-snapshot preservation is not claimed"
                ),
            }
        )
    if not rows:
        raise FileNotFoundError("No GOLD# historical CSV sources found")
    return rows


def preregister(run_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "in_progress":
        raise RuntimeError("Training run is not in progress")
    if sha256(Path(__file__)) != manifest.get("training_script_sha256"):
        raise RuntimeError("Executed script differs from immutable run snapshot")
    if manifest.get("git_dirty") is not False:
        raise RuntimeError("Formal experiment requires pre_run_git_dirty=false")
    head = git("rev-parse", "HEAD")
    tracking_ref = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    upstream = git("rev-parse", tracking_ref)
    if head != upstream or head != manifest.get("git_commit"):
        raise RuntimeError("Pre-run HEAD, upstream, and manifest commit do not match")

    operational = {
        GEMINI_FILE.name: sha256(GEMINI_FILE),
        OPERATIONAL_MODEL.name: sha256(OPERATIONAL_MODEL),
    }
    manifest["pre_run_git"] = {
        "pre_run_git_commit": head,
        "pre_run_git_dirty": False,
        "head_sha": head,
        "remote_ref": tracking_ref,
        "origin_main_sha": upstream,
        "head_equals_origin_main": True,
    }
    manifest["evidence_status"] = {
        "classification": "chronological_development_hypothesis_validation",
        "all_historical_intervals_are_development": True,
        "previous_forward_cutoff": PREVIOUS_FORWARD_CUTOFF,
        "previous_forward_status": PREVIOUS_FORWARD_STATUS,
        "untouched_oos_claim": False,
        "new_forward_cutoff": None,
    }
    manifest["paired_design"] = {
        "model_definitions": list(MODEL_IDS),
        "only_design_change": "binary training target semantics",
        "training_window_months": TRAIN_MONTHS,
        "folds": [
            {"fold": name, "start": start.isoformat(), "end_exclusive": end.isoformat()}
            for name, start, end in FOLDS
        ],
        "direction": "LONG only",
        "threshold": THRESHOLD,
        "minimum_entry_rsi": MIN_ENTRY_RSI,
        "excluded_rsi": list(EXCLUDED_RSI),
        "tp_atr": semantics.TP_ATR,
        "sl_atr": semantics.SL_ATR,
        "max_hold_wall_clock_minutes": EXECUTION_HORIZON_MINUTES,
        "max_open_positions": 1,
        "xgboost_parameters": FIXED_XGB_PARAMETERS,
        "parameter_search": False,
        "threshold_search": False,
        "label_search": False,
        "material_improvement_rule": (
            "C1 PF minus C0 PF >= 0.05 and C1 Mean-R/PnL-R exceed C0"
        ),
    }
    manifest["search"].update(
        {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": None,
            "not_applicable_reason": (
                "Exactly one paired causal comparison was preregistered; no model, label, "
                "threshold, feature, or parameter selection occurs."
            ),
        }
    )
    manifest["model"].update(
        {
            "trained": True,
            "model_type": "paired fold-specific XGBoost binary logistic research models",
            "parameters": FIXED_XGB_PARAMETERS,
            "boosted_rounds_or_estimators": N_ESTIMATORS,
            "features": ["pending_history_load"],
            "feature_count": 1,
            "label_definition": (
                "C0 legacy clean-window long label versus C1 standalone S5 net realized R > 0"
            ),
            "horizon": {
                "C0_rows": LEGACY_HORIZON_ROWS,
                "C1_wall_clock_minutes": EXECUTION_HORIZON_MINUTES,
            },
            "label_tp_sl_semantics": (
                "C0 existing 240-row clean-window 1.8/1.2 ATR label; C1 next-open, "
                "entry-bar HIGH/LOW stop-first, 1.3/1.6 ATR, 90 wall-clock minutes, "
                "observed/fallback spread plus configured costs"
            ),
            "execution_tp_sl_semantics": "exact preserved S5 implementation for both C0 and C1",
            "calibration_method": "none",
            "artifact_path": "pending",
            "artifact_sha256": "0" * 64,
            "retention_status": "pending fold-model persistence",
            "fold_models_planned": 6,
            "operational_artifact_retrained": False,
        }
    )
    manifest["promotion"].update(
        {
            "requested": False,
            "gate_result": "not_requested_research_only",
            "replacement_authorized": False,
            "operational_artifact_changed": False,
        }
    )
    manifest["operational_hashes_before"] = operational
    manifest["dependency_sha256"] = {
        S5_SOURCE.name: sha256(S5_SOURCE),
        "gold_gemini_core_gate_v1.py": sha256(ROOT / "gold_gemini_core_gate_v1.py"),
        "barrier_classifier_strategy.py": sha256(ROOT / "barrier_classifier_strategy.py"),
        "barrier_research_suite.py": sha256(ROOT / "barrier_research_suite.py"),
        "barrier_final_train.py": sha256(ROOT / "barrier_final_train.py"),
        "drl_trading_v2.py": sha256(ROOT / "drl_trading_v2.py"),
        "preserved_s5_manifest.json": sha256(S5_RUN / "manifest.json"),
    }
    write_json(manifest_path, manifest)
    print(f"PREREGISTERED commit={head} upstream={tracking_ref}", flush=True)
    return operational, manifest


def effective_spread(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(frame["SPREAD"], errors="coerce").to_numpy(dtype=np.float64)
    observed = np.isfinite(raw) & (raw > 0.0)
    return np.where(observed, raw, FALLBACK_SPREAD_POINTS), observed


def build_execution_aligned_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Standalone S5 outcome for every feature-complete row; no portfolio state."""
    n = len(frame)
    times = frame["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    open_ = frame["OPEN"].to_numpy(dtype=np.float64)
    high = frame["HIGH"].to_numpy(dtype=np.float64)
    low = frame["LOW"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    spread, observed = effective_spread(frame)
    tp = np.maximum(atr * semantics.TP_ATR, semantics.MIN_TP_PRICE)
    sl = np.maximum(atr * semantics.SL_ATR, semantics.MIN_SL_PRICE)
    denominator = sl + spread * POINT
    valid = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(tp)
        & np.isfinite(sl)
        & (denominator > 0.0)
    )
    unresolved = valid.copy()
    exit_index = np.full(n, -1, dtype=np.int64)
    exit_type = np.full(n, -1, dtype=np.int8)  # 0 timeout, 1 TP, 2 SL
    gross_price = np.full(n, np.nan, dtype=np.float64)
    same_bar_both = np.zeros(n, dtype=bool)
    minute_ns = 60 * 1_000_000_000

    for offset in range(EXECUTION_HORIZON_MINUTES + 1):
        limit = n - offset
        if limit <= 0:
            break
        active = unresolved[:limit].copy()
        if not active.any():
            break
        elapsed = (times[offset : offset + limit] - times[:limit]) / minute_ns
        timed_out = active & (elapsed >= EXECUTION_HORIZON_MINUTES)
        if timed_out.any():
            selected = np.flatnonzero(timed_out)
            exits = selected + offset
            exit_index[selected] = exits
            exit_type[selected] = 0
            gross_price[selected] = open_[exits] - open_[selected]
            unresolved[selected] = False

        active &= ~timed_out
        if not active.any():
            continue
        stop_hit = active & (low[offset : offset + limit] <= open_[:limit] - sl[:limit])
        take_hit = active & (high[offset : offset + limit] >= open_[:limit] + tp[:limit])
        both = stop_hit & take_hit
        if stop_hit.any():
            selected = np.flatnonzero(stop_hit)
            exit_index[selected] = selected + offset
            exit_type[selected] = 2
            gross_price[selected] = -sl[selected]
            same_bar_both[selected] = both[selected]
            unresolved[selected] = False
        take_only = take_hit & ~stop_hit
        if take_only.any():
            selected = np.flatnonzero(take_only)
            exit_index[selected] = selected + offset
            exit_type[selected] = 1
            gross_price[selected] = tp[selected]
            unresolved[selected] = False

    mature = exit_index >= 0
    net_r = np.full(n, np.nan, dtype=np.float64)
    stress_r = np.full(n, np.nan, dtype=np.float64)
    net_r[mature] = (
        gross_price[mature] - (spread[mature] + BASE_EXTRA_COST_POINTS) * POINT
    ) / denominator[mature]
    stress_r[mature] = (
        gross_price[mature] - (spread[mature] + STRESS_EXTRA_COST_POINTS) * POINT
    ) / denominator[mature]
    target = np.zeros(n, dtype=np.int8)
    target[mature & (net_r > 0.0)] = 1
    maturity_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    maturity_ns[mature] = times[exit_index[mature]]
    return pd.DataFrame(
        {
            "C1_TARGET": target,
            "C1_MATURE": mature,
            "C1_NET_R": net_r,
            "C1_STRESS_R": stress_r,
            "C1_GROSS_PRICE": gross_price,
            "C1_EXIT_INDEX": exit_index,
            "C1_EXIT_TYPE": exit_type,
            "C1_MATURITY_NS": maturity_ns,
            "C1_SPREAD_POINTS": spread,
            "C1_SPREAD_OBSERVED": observed,
            "C1_SAME_BAR_BOTH": same_bar_both,
        }
    )


def first_touch_diagnostic(
    frame: pd.DataFrame,
    *,
    entry_column: str,
    start_offset: int,
    horizon_rows: int,
    tp_atr: float,
    sl_atr: float,
    min_tp: float,
    min_sl: float,
) -> np.ndarray:
    """Diagnostic TP-first flag only; never used as a fitted target."""
    entry = frame[entry_column].to_numpy(dtype=np.float64)
    high = frame["HIGH"].to_numpy(dtype=np.float64)
    low = frame["LOW"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    tp = np.maximum(atr * tp_atr, min_tp)
    sl = np.maximum(atr * sl_atr, min_sl)
    n = len(frame)
    result = np.zeros(n, dtype=np.int8)
    unresolved = np.isfinite(entry) & np.isfinite(tp) & np.isfinite(sl)
    for offset in range(start_offset, horizon_rows + 1):
        limit = n - offset
        if limit <= 0:
            break
        active = unresolved[:limit].copy()
        if not active.any():
            break
        stop = active & (low[offset : offset + limit] <= entry[:limit] - sl[:limit])
        take = active & (high[offset : offset + limit] >= entry[:limit] + tp[:limit])
        resolved = stop | take
        result[np.flatnonzero(take & ~stop)] = 1
        unresolved[np.flatnonzero(resolved)] = False
    return result


def label_definition() -> dict[str, Any]:
    return {
        "hypotheses": {
            "C0": {
                "name": MODEL_IDS[0],
                "definition": (
                    "BARRIER_TARGET == 1: from decision-row CLOSE, within the next 240 M1 "
                    "rows future HIGH reaches max(1.8 ATR, 1.0) and no future LOW reaches "
                    "max(1.2 ATR, 0.8); otherwise 0"
                ),
                "source": "barrier_classifier_strategy.build_barrier_target",
            },
            "C1": {
                "name": MODEL_IDS[1],
                "decision": "31 shifted features from the preceding completed M1 bar",
                "entry": "current/next executable M1 bar OPEN",
                "monitoring": "entry bar onward",
                "tp": "max(1.3 * shifted ATR, 1.5 price units)",
                "sl": "max(1.6 * shifted ATR, 0.6 price units)",
                "timeout": "first available M1 OPEN at elapsed wall-clock minutes >= 90",
                "intrabar": "HIGH/LOW; stop-first when both reachable",
                "spread": "observed positive SPREAD else 30-point fallback",
                "base_extra_cost_points": BASE_EXTRA_COST_POINTS,
                "denominator": "SL distance + entry spread price",
                "target_one": "standalone net realized R > 0",
                "target_zero": "standalone net realized R <= 0",
                "zero_return_convention": "zero is non-positive and maps to target 0",
                "portfolio_state_in_label": False,
            },
        },
        "paired_invariants": {
            "features": 31,
            "train_months": TRAIN_MONTHS,
            "folds": [name for name, _, _ in FOLDS],
            "xgboost": FIXED_XGB_PARAMETERS,
            "only_y_semantics_differs": True,
        },
        "diagnostic_counterfactuals_not_trained": [
            "legacy-geometry first-touch label",
            "C1 gross-positive before costs",
            "C1 exit-type attribution",
        ],
    }


def train_one(
    history: pd.DataFrame,
    features: list[str],
    indices: np.ndarray,
    target: np.ndarray,
) -> xgb.XGBClassifier:
    train = history.loc[indices, features].copy()
    train["BARRIER_TARGET"] = target.astype(np.int8)
    model = train_binary_model(train, features, 1, N_ESTIMATORS)
    del train
    return model


def paired_training(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    features: list[str],
    run_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    times = history["TIME_DT"]
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    c0 = (history["BARRIER_TARGET"].to_numpy(dtype=np.int8) == 1).astype(np.int8)
    c1 = labels["C1_TARGET"].to_numpy(dtype=np.int8)
    c1_mature = labels["C1_MATURE"].to_numpy(dtype=bool)
    c1_maturity = labels["C1_MATURITY_NS"].to_numpy(dtype=np.int64)
    legacy_maturity_index = np.arange(len(history), dtype=np.int64) + LEGACY_HORIZON_ROWS
    legacy_mature = legacy_maturity_index < len(history)
    legacy_maturity_ns = np.full(len(history), np.iinfo(np.int64).max, dtype=np.int64)
    legacy_maturity_ns[legacy_mature] = time_ns[legacy_maturity_index[legacy_mature]]
    parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    model_inventory: list[dict[str, Any]] = []

    legacy_first_touch = first_touch_diagnostic(
        history,
        entry_column="CLOSE",
        start_offset=1,
        horizon_rows=LEGACY_HORIZON_ROWS,
        tp_atr=LABEL_TP_ATR,
        sl_atr=LABEL_SL_ATR,
        min_tp=LEGACY_MIN_TP_PRICE,
        min_sl=LEGACY_MIN_SL_PRICE,
    )
    c1_gross_positive = (
        labels["C1_MATURE"].to_numpy(dtype=bool)
        & (labels["C1_GROSS_PRICE"].to_numpy(dtype=np.float64) > 0.0)
    ).astype(np.int8)
    c1_exit = labels["C1_EXIT_TYPE"].to_numpy(dtype=np.int8)

    for fold_code, (fold_name, fold_start, fold_end) in enumerate(FOLDS):
        score_start_ns = int(fold_start.to_datetime64().astype("datetime64[ns]").astype(np.int64))
        lower = fold_start - pd.DateOffset(months=TRAIN_MONTHS)
        train_mask = (
            (times >= lower)
            & (times < fold_start)
            & legacy_mature
            & c1_mature
            & (legacy_maturity_ns < score_start_ns)
            & (c1_maturity < score_start_ns)
        )
        score_mask = (times >= fold_start) & (times < fold_end) & c1_mature
        train_indices = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        score_indices = np.flatnonzero(score_mask.to_numpy(dtype=bool))
        if len(train_indices) == 0 or len(score_indices) == 0:
            raise RuntimeError(f"Empty paired train/score data: {fold_name}")

        latest_c0 = int(legacy_maturity_ns[train_indices].max())
        latest_c1 = int(c1_maturity[train_indices].max())
        latest_information = max(latest_c0, latest_c1)
        if latest_information >= score_start_ns:
            raise RuntimeError(f"Label maturity overlap: {fold_name}")

        x_train = history.loc[train_indices, features].to_numpy(dtype=np.float32)
        x_score = history.loc[score_indices, features].to_numpy(dtype=np.float32)
        train_time_hash = array_sha256(time_ns[train_indices])
        score_time_hash = array_sha256(time_ns[score_indices])
        x_train_hash = array_sha256(x_train)
        x_score_hash = array_sha256(x_score)
        c0_hash = array_sha256(c0[train_indices])
        c1_hash = array_sha256(c1[train_indices])
        print(
            f"PAIRED TRAIN {fold_name}: train={len(train_indices):,} "
            f"{times.iat[int(train_indices[0])]}..{times.iat[int(train_indices[-1])]} "
            f"score={len(score_indices):,}",
            flush=True,
        )

        scores: dict[str, np.ndarray] = {}
        for model_id, target in ((MODEL_IDS[0], c0), (MODEL_IDS[1], c1)):
            model = train_one(history, features, train_indices, target[train_indices])
            scores[model_id] = predict_positive(
                model, history.loc[score_indices], features
            ).astype(np.float32)
            artifact = models_dir / f"{model_id}_{fold_name}_xgb.json"
            model.save_model(artifact)
            model_inventory.append(
                {
                    "model_id": model_id,
                    "fold": fold_name,
                    "path": artifact.relative_to(run_dir).as_posix(),
                    "sha256": sha256(artifact),
                    "train_rows": len(train_indices),
                    "target_sha256": c0_hash if model_id == MODEL_IDS[0] else c1_hash,
                }
            )
            del model
            gc.collect()

        part = history.loc[
            score_indices,
            ["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD"],
        ].copy()
        part["global_index"] = score_indices
        part["fold"] = fold_name
        part["fold_code"] = fold_code
        part["feature_bar_time"] = history["TIME_DT"].shift(1).loc[score_indices].to_numpy()
        part["C1_TARGET"] = c1[score_indices]
        part["C1_NET_R"] = labels["C1_NET_R"].to_numpy(dtype=np.float64)[score_indices]
        part["C1_STRESS_R"] = labels["C1_STRESS_R"].to_numpy(dtype=np.float64)[score_indices]
        part["C1_EXIT_TYPE"] = c1_exit[score_indices]
        part["C1_EXIT_INDEX"] = labels["C1_EXIT_INDEX"].to_numpy(dtype=np.int64)[score_indices]
        for model_id in MODEL_IDS:
            part[f"score_{model_id}"] = scores[model_id]
        parts.append(part)

        train_c0 = c0[train_indices]
        train_c1 = c1[train_indices]
        agree_pos = (train_c0 == 1) & (train_c1 == 1)
        agree_neg = (train_c0 == 0) & (train_c1 == 0)
        c0_pos_c1_neg = (train_c0 == 1) & (train_c1 == 0)
        c0_neg_c1_pos = (train_c0 == 0) & (train_c1 == 1)
        disagreement = train_c0 != train_c1
        train_exit = c1_exit[train_indices]
        cause = {
            "clean_window_vs_legacy_first_touch": int(
                np.sum(train_c0 != legacy_first_touch[train_indices])
            ),
            "legacy_label_vs_runtime_geometry_or_horizon": int(
                np.sum(legacy_first_touch[train_indices] != c1_gross_positive[train_indices])
            ),
            "timeout_realized_outcome": int(np.sum(disagreement & (train_exit == 0))),
            "runtime_stop_first_loss": int(np.sum(c0_pos_c1_neg & (train_exit == 2))),
            "runtime_tp_not_legacy_positive": int(np.sum(c0_neg_c1_pos & (train_exit == 1))),
            "cost_flips_gross_positive_to_nonpositive": int(
                np.sum((c1_gross_positive[train_indices] == 1) & (train_c1 == 0))
            ),
            "next_open_entry_and_runtime_path_unisolated_component": int(
                np.sum(legacy_first_touch[train_indices] != c1_gross_positive[train_indices])
            ),
        }
        label_rows.append(
            {
                "fold": fold_name,
                "training_rows": len(train_indices),
                "c0_positive_prevalence": float(train_c0.mean()),
                "c1_positive_prevalence": float(train_c1.mean()),
                "agreeing_positive": int(agree_pos.sum()),
                "agreeing_negative": int(agree_neg.sum()),
                "c0_positive_c1_negative": int(c0_pos_c1_neg.sum()),
                "c0_negative_c1_positive": int(c0_neg_c1_pos.sum()),
                "disagreement_count": int(disagreement.sum()),
                "disagreement_rate": float(disagreement.mean()),
                **cause,
                "cause_counts_are_overlapping_diagnostics": True,
            }
        )
        provenance.append(
            {
                "fold": fold_name,
                "train_start": times.iat[int(train_indices[0])].isoformat(),
                "train_feature_end": times.iat[int(train_indices[-1])].isoformat(),
                "latest_training_entry_time": times.iat[int(train_indices[-1])].isoformat(),
                "latest_c0_label_information_time": pd.Timestamp(latest_c0).isoformat(),
                "latest_c1_label_information_time": pd.Timestamp(latest_c1).isoformat(),
                "latest_training_label_information_time": pd.Timestamp(latest_information).isoformat(),
                "score_start": times.iat[int(score_indices[0])].isoformat(),
                "score_end": times.iat[int(score_indices[-1])].isoformat(),
                "train_rows": len(train_indices),
                "score_rows": len(score_indices),
                "training_window_months": TRAIN_MONTHS,
                "strict_label_maturity_before_score": bool(latest_information < score_start_ns),
                "train_timestamp_sha256": train_time_hash,
                "score_timestamp_sha256": score_time_hash,
                "x_train_sha256_C0": x_train_hash,
                "x_train_sha256_C1": x_train_hash,
                "x_score_sha256_C0": x_score_hash,
                "x_score_sha256_C1": x_score_hash,
                "c0_label_sha256": c0_hash,
                "c1_label_sha256": c1_hash,
                "random_seed_C0": RANDOM_STATE,
                "random_seed_C1": RANDOM_STATE,
                "parameters_C0": FIXED_XGB_PARAMETERS,
                "parameters_C1": FIXED_XGB_PARAMETERS,
            }
        )
        del x_train, x_score
        gc.collect()

    scored = pd.concat(parts, ignore_index=True).sort_values("TIME_DT").reset_index(drop=True)
    if not scored["TIME_DT"].is_monotonic_increasing:
        raise RuntimeError("Paired scored rows are not chronological")
    return scored, provenance, label_rows, model_inventory


def reward_metrics(values: np.ndarray) -> dict[str, Any]:
    return semantics.reward_metrics(np.asarray(values, dtype=np.float64))


def trade_metrics(trades: list[dict[str, Any]], days: int) -> dict[str, Any]:
    values = np.asarray([item["net_r"] for item in trades], dtype=np.float64)
    stress = np.asarray([item["stress_r"] for item in trades], dtype=np.float64)
    result = reward_metrics(values)
    stress_result = reward_metrics(stress)
    holds = np.asarray(
        [
            (pd.Timestamp(item["exit_time_api"]) - pd.Timestamp(item["entry_time_api"]))
            .total_seconds()
            / 60.0
            for item in trades
        ],
        dtype=np.float64,
    )
    account = np.asarray([item["account_pnl"] for item in trades], dtype=np.float64)
    result.update(
        {
            "trades_per_day": len(trades) / max(days, 1),
            "tp_first_wr": (
                sum(item["exit_reason"] == "take_profit" for item in trades) / len(trades)
                if trades
                else 0.0
            ),
            "tp_count": int(sum(item["exit_reason"] == "take_profit" for item in trades)),
            "sl_count": int(sum(item["exit_reason"] == "stop_loss" for item in trades)),
            "profitable_timeout_count": int(
                sum(item["exit_reason"] == "timeout" and item["net_r"] > 0 for item in trades)
            ),
            "losing_timeout_count": int(
                sum(item["exit_reason"] in {"timeout", "cohort_end"} and item["net_r"] <= 0 for item in trades)
            ),
            "cohort_end_count": int(sum(item["exit_reason"] == "cohort_end" for item in trades)),
            "average_hold_minutes": float(holds.mean()) if len(holds) else None,
            "cost_stress_pf": stress_result["pf"],
            "cost_stress_mean_r": stress_result["mean_r"],
            "cost_stress_pnl_r": stress_result["pnl_r"],
            "risk_sized_pf": (
                float(account[account > 0].sum() / -account[account <= 0].sum())
                if len(account) and account[account <= 0].sum() < 0
                else (math.inf if len(account) and account[account > 0].sum() > 0 else 0.0)
            ),
        }
    )
    return result


def evaluation_cohort(scored: pd.DataFrame, model_id: str) -> pd.DataFrame:
    columns = ["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD"]
    cohort = scored[columns].copy()
    cohort["buy_prob"] = scored[f"score_{model_id}"].to_numpy(dtype=np.float32)
    cohort["sell_prob"] = np.float32(0.0)
    cohort = semantics.finalize_cohort(cohort, f"PAIRED_{model_id}", offset_hours=0)
    return cohort


def execute_models(
    scored: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ledgers: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    pooled_days = sum(core.fold_days(start, end) for _, start, end in FOLDS)
    for model_id in MODEL_IDS:
        cohort = evaluation_cohort(scored, model_id)
        trades, audit = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        for item in trades:
            entry = pd.Timestamp(item["entry_time_api"])
            fold = next(
                (name for name, start, end in FOLDS if start <= entry < end),
                "outside_fold",
            )
            item["model_id"] = model_id
            item["fold"] = fold
            item["trade_id"] = f"{model_id}_{item['trade_id']}"
        ledgers[model_id] = trades
        audits[model_id] = audit
        all_trades.extend(trades)
        for fold_name, start, end in FOLDS:
            selected = [item for item in trades if item["fold"] == fold_name]
            metric_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold_name,
                    **trade_metrics(selected, core.fold_days(start, end)),
                }
            )
        pooled = trade_metrics(trades, pooled_days)
        metric_rows.append({"model_id": model_id, "fold": "pooled", **pooled, **audit})
        print(
            f"S5 {model_id}: trades={pooled['trades']} tpd={pooled['trades_per_day']:.4f} "
            f"WR={pooled['realized_wr']:.2%} PF={pooled['pf']:.4f} "
            f"Mean-R={pooled['mean_r']:.4f}",
            flush=True,
        )
    return metric_rows, all_trades, {"ledgers": ledgers, "audits": audits}


def safe_classification(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.int8)
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if len(np.unique(target)) < 2:
        return {"roc_auc": None, "pr_auc": None, "log_loss": None, "brier": None}
    return {
        "roc_auc": float(roc_auc_score(target, score)),
        "pr_auc": float(average_precision_score(target, score)),
        "log_loss": float(log_loss(target, score, labels=[0, 1])),
        "brier": float(brier_score_loss(target, score)),
    }


def spearman(score: np.ndarray, outcome: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(outcome)
    if valid.sum() < 3:
        return None
    left = pd.Series(score[valid]).rank(method="average").to_numpy(dtype=np.float64)
    right = pd.Series(outcome[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def probability_and_ranking(
    scored: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    probability_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    scopes = [(name, scored["fold"].eq(name).to_numpy()) for name, _, _ in FOLDS]
    scopes.append(("pooled", np.ones(len(scored), dtype=bool)))
    target = scored["C1_TARGET"].to_numpy(dtype=np.int8)
    realized = scored["C1_NET_R"].to_numpy(dtype=np.float64)
    for model_id in MODEL_IDS:
        values = scored[f"score_{model_id}"].to_numpy(dtype=np.float64)
        for fold_name, mask in scopes:
            score = values[mask]
            y = target[mask]
            reward = realized[mask]
            quantiles = np.quantile(score, [0.50, 0.75, 0.90, 0.95, 0.99])
            probability_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold_name,
                    "observations": len(score),
                    "mean": float(score.mean()),
                    "median": float(quantiles[0]),
                    "p75": float(quantiles[1]),
                    "p90": float(quantiles[2]),
                    "p95": float(quantiles[3]),
                    "p99": float(quantiles[4]),
                    "max": float(score.max()),
                    **{f"fraction_ge_{int(level * 100):02d}": float((score >= level).mean()) for level in PROBABILITY_LEVELS},
                    **safe_classification(y, score),
                }
            )
            order = np.argsort(score, kind="stable")
            decile = np.empty(len(score), dtype=np.int8)
            decile[order] = np.minimum((np.arange(len(score)) * 10 // max(len(score), 1)) + 1, 10)
            for number in range(1, 11):
                selected = decile == number
                stats = reward_metrics(reward[selected])
                decile_rows.append(
                    {
                        "model_id": model_id,
                        "fold": fold_name,
                        "decile": number,
                        "observations": int(selected.sum()),
                        "mean_probability": float(score[selected].mean()),
                        "positive_rate": float(y[selected].mean()),
                        "mean_r": stats["mean_r"],
                        "pf": stats["pf"],
                    }
                )
            top10 = decile == 10
            top20 = decile >= 9
            top10_stats = reward_metrics(reward[top10])
            top20_stats = reward_metrics(reward[top20])
            ranking_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold_name,
                    "observations": len(score),
                    "spearman_score_realized_net_r": spearman(score, reward),
                    "spearman_score_positive_net_r": spearman(score, y.astype(np.float64)),
                    "top_decile_mean_r": top10_stats["mean_r"],
                    "top_decile_pf": top10_stats["pf"],
                    "top_quintile_mean_r": top20_stats["mean_r"],
                    "top_quintile_pf": top20_stats["pf"],
                }
            )
    return probability_rows, ranking_rows, decile_rows


def economic_subset(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([item["net_r"] for item in trades], dtype=np.float64)
    return reward_metrics(values)


def marginal_identity(
    execution: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c0 = execution["ledgers"][MODEL_IDS[0]]
    c1 = execution["ledgers"][MODEL_IDS[1]]
    by0 = {item["entry_time_api"]: item for item in c0}
    by1 = {item["entry_time_api"]: item for item in c1}
    common = sorted(set(by0) & set(by1))
    only0_keys = sorted(set(by0) - set(by1))
    only1_keys = sorted(set(by1) - set(by0))
    only0 = [by0[key] for key in only0_keys]
    only1 = [by1[key] for key in only1_keys]

    unmatched1 = set(range(len(only1)))
    changed_pairs: list[tuple[int, int]] = []
    for left_index, left in enumerate(only0):
        left_time = pd.Timestamp(left["entry_time_api"])
        choices = []
        for right_index in unmatched1:
            right = only1[right_index]
            if left["fold"] != right["fold"]:
                continue
            delta = abs((pd.Timestamp(right["entry_time_api"]) - left_time).total_seconds())
            if delta <= EXECUTION_HORIZON_MINUTES * 60:
                choices.append((delta, pd.Timestamp(right["entry_time_api"]), right_index))
        if choices:
            _, _, right_index = min(choices)
            unmatched1.remove(right_index)
            changed_pairs.append((left_index, right_index))

    rows: list[dict[str, Any]] = []
    for timestamp in common:
        left, right = by0[timestamp], by1[timestamp]
        rows.append(
            {
                "match_type": "exact_common_entry",
                "c0_trade_id": left["trade_id"],
                "c1_trade_id": right["trade_id"],
                "c0_entry_time": timestamp,
                "c1_entry_time": timestamp,
                "entry_delta_minutes": 0.0,
                "c0_net_r": left["net_r"],
                "c1_net_r": right["net_r"],
                "outcome_sign_changed": bool((left["net_r"] > 0) != (right["net_r"] > 0)),
            }
        )
    for left_index, right_index in changed_pairs:
        left, right = only0[left_index], only1[right_index]
        rows.append(
            {
                "match_type": "changed_entry_within_90m_nearest_greedy",
                "c0_trade_id": left["trade_id"],
                "c1_trade_id": right["trade_id"],
                "c0_entry_time": left["entry_time_api"],
                "c1_entry_time": right["entry_time_api"],
                "entry_delta_minutes": (
                    pd.Timestamp(right["entry_time_api"]) - pd.Timestamp(left["entry_time_api"])
                ).total_seconds()
                / 60.0,
                "c0_net_r": left["net_r"],
                "c1_net_r": right["net_r"],
                "outcome_sign_changed": bool((left["net_r"] > 0) != (right["net_r"] > 0)),
            }
        )
    paired0 = {left for left, _ in changed_pairs}
    paired1 = {right for _, right in changed_pairs}
    for index, trade in enumerate(only0):
        if index not in paired0:
            rows.append(
                {
                    "match_type": "C0_only_unmatched",
                    "c0_trade_id": trade["trade_id"],
                    "c1_trade_id": "",
                    "c0_entry_time": trade["entry_time_api"],
                    "c1_entry_time": "",
                    "entry_delta_minutes": "",
                    "c0_net_r": trade["net_r"],
                    "c1_net_r": "",
                    "outcome_sign_changed": "",
                }
            )
    for index, trade in enumerate(only1):
        if index not in paired1:
            rows.append(
                {
                    "match_type": "C1_only_unmatched",
                    "c0_trade_id": "",
                    "c1_trade_id": trade["trade_id"],
                    "c0_entry_time": "",
                    "c1_entry_time": trade["entry_time_api"],
                    "entry_delta_minutes": "",
                    "c0_net_r": "",
                    "c1_net_r": trade["net_r"],
                    "outcome_sign_changed": "",
                }
            )
    summary = {
        "matching_rule": (
            "Exact entry timestamp first; remaining trades greedily matched by smallest absolute "
            "entry-time difference within the same fold and 90 minutes, deterministic timestamp tie-break."
        ),
        "common_trades": len(common),
        "c0_only_exact_identity": len(only0),
        "c1_only_exact_identity": len(only1),
        "changed_entry_timestamp_pairs": len(changed_pairs),
        "changed_pair_outcome_sign_changes": int(
            sum(bool(row["outcome_sign_changed"]) for row in rows if row["match_type"].startswith("changed_entry"))
        ),
        "c0_only_trade_economics": economic_subset(only0),
        "c1_only_trade_economics": economic_subset(only1),
    }
    return rows, summary


def catastrophic(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["trades"] < CATASTROPHIC_FOLD["minimum_trades"]
        or metrics["realized_wr"] < CATASTROPHIC_FOLD["realized_wr_min"]
        or metrics["pf"] < CATASTROPHIC_FOLD["pf_min"]
        or metrics["mean_r"] < CATASTROPHIC_FOLD["mean_r_min"]
        or metrics["max_dd_r"] < CATASTROPHIC_FOLD["max_dd_r_min"]
    )


def decisions(
    metric_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    pooled = {
        row["model_id"]: row for row in metric_rows if row["fold"] == "pooled"
    }
    c0, c1 = pooled[MODEL_IDS[0]], pooled[MODEL_IDS[1]]
    fold_c1 = [row for row in metric_rows if row["model_id"] == MODEL_IDS[1] and row["fold"] != "pooled"]
    viability_reasons = []
    for condition, reason in (
        (c1["pf"] <= 1.0, "PF<=1"),
        (c1["mean_r"] <= 0.0, "Mean-R<=0"),
        (c1["pnl_r"] <= 0.0, "PnL-R<=0"),
        ((c1["break_even_adjusted_edge"] or -1.0) <= 0.0, "BE-edge<=0"),
    ):
        if condition:
            viability_reasons.append(reason)
    floor_reasons = list(viability_reasons)
    for condition, reason in (
        (c1["realized_wr"] < QUALITY_FLOOR["realized_wr_min"], "WR<60%"),
        (c1["pf"] <= QUALITY_FLOOR["pf_strict_min"], "PF<=1.05"),
        (c1["cost_stress_pf"] <= QUALITY_FLOOR["cost_stress_pf_strict_min"], "stress-PF<=1"),
    ):
        if condition:
            floor_reasons.append(reason)
    catastrophic_folds = [row["fold"] for row in fold_c1 if catastrophic(row)]
    if catastrophic_folds:
        floor_reasons.append("catastrophic-fold=" + ",".join(catastrophic_folds))
    delta = {
        "trades": c1["trades"] - c0["trades"],
        "trades_per_day": c1["trades_per_day"] - c0["trades_per_day"],
        "realized_wr": c1["realized_wr"] - c0["realized_wr"],
        "pf": c1["pf"] - c0["pf"],
        "mean_r": c1["mean_r"] - c0["mean_r"],
        "pnl_r": c1["pnl_r"] - c0["pnl_r"],
        "max_dd_r": c1["max_dd_r"] - c0["max_dd_r"],
    }
    materially_improves = bool(
        delta["pf"] >= 0.05 and delta["mean_r"] > 0.0 and delta["pnl_r"] > 0.0
    )
    c1_fold_rank = [
        row for row in ranking_rows if row["model_id"] == MODEL_IDS[1] and row["fold"] != "pooled"
    ]
    c0_fold_rank = [
        row for row in ranking_rows if row["model_id"] == MODEL_IDS[0] and row["fold"] != "pooled"
    ]
    c1_rank_values = [row["spearman_score_realized_net_r"] for row in c1_fold_rank]
    c0_rank_values = [row["spearman_score_realized_net_r"] for row in c0_fold_rank]
    ranking_consistent = bool(
        sum(value is not None and value > 0.0 for value in c1_rank_values) >= 2
        and all(value is None or value >= -0.02 for value in c1_rank_values)
        and np.nanmean([value if value is not None else np.nan for value in c1_rank_values])
        > np.nanmean([value if value is not None else np.nan for value in c0_rank_values])
    )
    viability = not viability_reasons
    full_floor = not floor_reasons
    if not viability:
        classification = "execution_alignment_alone_does_not_recover_robust_alpha"
        next_hypothesis = (
            "NEW TIMESTAMP-ALIGNED ALPHA INFORMATION: preregister one external entry-time "
            "information family and test incremental cross-regime discrimination under the frozen "
            "execution-aligned target before any further model or threshold tuning."
        )
    elif not full_floor:
        classification = "promising_execution_aligned_signal_not_strategy_ready" if materially_improves else "economically_positive_but_not_materially_better_or_quality_ready"
        next_hypothesis = (
            "PREREGISTERED CALIBRATION-RANKING VALIDATION under the frozen C1 model architecture."
            if ranking_consistent
            else "NEW TIMESTAMP-ALIGNED ALPHA INFORMATION under the frozen execution-aligned target."
        )
    else:
        classification = "shadow_candidate_worthy_development_result"
        next_hypothesis = "UNTOUCHED SHADOW FORWARD VALIDATION of the completely frozen C1 specification."
    return {
        "c1_minus_c0": delta,
        "c1_materially_improves_c0": materially_improves,
        "c1_economic_viability": viability,
        "economic_viability_failure_reasons": viability_reasons,
        "c1_full_quality_floor": full_floor,
        "full_quality_floor_failure_reasons": floor_reasons,
        "catastrophic_folds": catastrophic_folds,
        "ranking_consistently_positive": ranking_consistent,
        "classification": classification,
        "single_next_research_hypothesis": next_hypothesis,
        "shadow_candidate_frozen": full_floor,
    }


def freeze_shadow_if_qualified(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    features: list[str],
    run_dir: Path,
    decision: dict[str, Any],
    definition_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not decision["c1_full_quality_floor"]:
        return {
            "frozen": False,
            "reason": "C1 did not pass the complete preregistered development quality floor.",
            "new_forward_cutoff": None,
        }, None
    mature = labels["C1_MATURE"].to_numpy(dtype=bool)
    indices = np.flatnonzero(mature)
    if not len(indices):
        raise RuntimeError("No mature C1 rows available for the conditional shadow fit")
    training_end = history["TIME_DT"].iat[int(indices[-1])]
    lower = training_end - pd.DateOffset(months=TRAIN_MONTHS)
    indices = np.flatnonzero(
        mature
        & (history["TIME_DT"].to_numpy(dtype="datetime64[ns]") >= lower.to_datetime64())
        & (history["TIME_DT"].to_numpy(dtype="datetime64[ns]") <= training_end.to_datetime64())
    )
    target = labels["C1_TARGET"].to_numpy(dtype=np.int8)[indices]
    model = train_one(history, features, indices, target)
    artifact = run_dir / "models" / "C1_shadow_final_xgb.json"
    model.save_model(artifact)
    del model
    freeze_timestamp = now_utc()
    feature_hash = hashlib.sha256(
        json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    specification = {
        "run_id": run_dir.name,
        "candidate_id": "C1_EXECUTION_ALIGNED_SHADOW_V1",
        "model_path": artifact.relative_to(run_dir).as_posix(),
        "model_sha256": sha256(artifact),
        "training_script_sha256": sha256(Path(__file__)),
        "feature_list": features,
        "feature_list_sha256": feature_hash,
        "label_definition_path": definition_path.relative_to(run_dir).as_posix(),
        "label_definition_sha256": sha256(definition_path),
        "training_window_months": TRAIN_MONTHS,
        "training_start": history["TIME_DT"].iat[int(indices[0])].isoformat(),
        "training_end": training_end.isoformat(),
        "training_rows": len(indices),
        "xgboost_parameters": FIXED_XGB_PARAMETERS,
        "threshold": THRESHOLD,
        "minimum_entry_rsi": MIN_ENTRY_RSI,
        "excluded_rsi": list(EXCLUDED_RSI),
        "tp_atr": semantics.TP_ATR,
        "sl_atr": semantics.SL_ATR,
        "hold_wall_clock_minutes": EXECUTION_HORIZON_MINUTES,
        "session_hours": sorted(semantics.ALLOWED_HOURS),
        "session_weekdays": sorted(semantics.ALLOWED_WEEKDAYS),
        "s5_source_sha256": sha256(S5_SOURCE),
        "frozen_at_utc": freeze_timestamp,
        "new_untouched_forward_cutoff_utc": freeze_timestamp,
        "production_promotion": False,
    }
    path = run_dir / "shadow_candidate.json"
    write_json(path, specification)
    return {
        "frozen": True,
        "candidate_id": specification["candidate_id"],
        "model_sha256": specification["model_sha256"],
        "specification_path": path.relative_to(run_dir).as_posix(),
        "new_forward_cutoff": freeze_timestamp,
        "production_promotion": False,
    }, {
        "model_id": MODEL_IDS[1],
        "fold": "shadow_final",
        "path": artifact.relative_to(run_dir).as_posix(),
        "sha256": specification["model_sha256"],
        "train_rows": len(indices),
        "target_sha256": array_sha256(target),
    }


def save_oof(path: Path, scored: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        global_index=scored["global_index"].to_numpy(dtype=np.int64),
        time_ns=scored["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        feature_time_ns=scored["feature_bar_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        fold_code=scored["fold_code"].to_numpy(dtype=np.int8),
        c1_target=scored["C1_TARGET"].to_numpy(dtype=np.int8),
        c1_net_r=scored["C1_NET_R"].to_numpy(dtype=np.float32),
        c1_stress_r=scored["C1_STRESS_R"].to_numpy(dtype=np.float32),
        c1_exit_type=scored["C1_EXIT_TYPE"].to_numpy(dtype=np.int8),
        c1_exit_index=scored["C1_EXIT_INDEX"].to_numpy(dtype=np.int64),
        open=scored["OPEN"].to_numpy(dtype=np.float64),
        high=scored["HIGH"].to_numpy(dtype=np.float64),
        low=scored["LOW"].to_numpy(dtype=np.float64),
        close=scored["CLOSE"].to_numpy(dtype=np.float64),
        atr=scored["ATR"].to_numpy(dtype=np.float64),
        rsi=scored["M1_RSI"].to_numpy(dtype=np.float64),
        spread=scored["SPREAD"].to_numpy(dtype=np.float64),
        score_c0=scored[f"score_{MODEL_IDS[0]}"].to_numpy(dtype=np.float32),
        score_c1=scored[f"score_{MODEL_IDS[1]}"].to_numpy(dtype=np.float32),
    )


def markdown_metric(row: dict[str, Any]) -> str:
    return (
        f"| {row['model_id']} | {row['fold']} | {row['trades']} | {row['trades_per_day']:.4f} | "
        f"{row['realized_wr']:.2%} | {row['tp_first_wr']:.2%} | {row['pf']:.4f} | "
        f"{row['mean_r']:.4f} | {row['pnl_r']:.2f} | {row['max_dd_r']:.2f} | "
        f"{row['cost_stress_pf']:.4f} |"
    )


def report_text(metrics: dict[str, Any]) -> str:
    rows = metrics["fold_metrics"]
    decision = metrics["decision"]
    labels = metrics["label_comparison"]
    pooled_labels = labels[-1]
    marginal = metrics["marginal_trade_identity"]
    probability = metrics["probability_diagnostics"]
    ranking = metrics["ranking_diagnostics"]
    pooled_probability = {row["model_id"]: row for row in probability if row["fold"] == "pooled"}
    pooled_ranking = {row["model_id"]: row for row in ranking if row["fold"] == "pooled"}
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Status: **development paired-label research**. No operational artifact was changed.",
        "",
        "## Paired executable economics under the same S5 simulator",
        "",
        "| Model | Fold | Trades | Trades/day | Realized WR | TP-first WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[markdown_metric(row) for row in rows],
        "",
        "## C1 minus C0 pooled delta",
        "",
        "```json",
        json.dumps(sanitize(decision["c1_minus_c0"]), indent=2),
        "```",
        "",
        "## Label attribution",
        "",
        f"C0 positive prevalence: {pooled_labels['c0_positive_prevalence']:.2%}.",
        f"C1 positive prevalence: {pooled_labels['c1_positive_prevalence']:.2%}.",
        f"Disagreement: {pooled_labels['disagreement_rate']:.2%} ({pooled_labels['disagreement_count']:,} rows).",
        "Cause counts in label_comparison.csv are diagnostics and may overlap; no counterfactual label was trained.",
        "",
        "## Probability and ranking diagnostics",
        "",
    ]
    for model_id in MODEL_IDS:
        p = pooled_probability[model_id]
        r = pooled_ranking[model_id]
        lines.append(
            f"- {model_id}: p>=0.75={p['fraction_ge_75']:.2%}; "
            f"ROC-AUC={p['roc_auc']:.4f}; Brier={p['brier']:.4f}; "
            f"Spearman(score, net-R)={r['spearman_score_realized_net_r']:.4f}; "
            f"top-decile Mean-R={r['top_decile_mean_r']:.4f}, PF={r['top_decile_pf']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Marginal executable trades",
            "",
            f"Common exact entries: {marginal['common_trades']}; C0-only: {marginal['c0_only_exact_identity']}; "
            f"C1-only: {marginal['c1_only_exact_identity']}; changed-entry pairs: {marginal['changed_entry_timestamp_pairs']}.",
            f"C0-only economics: {json.dumps(sanitize(marginal['c0_only_trade_economics']))}.",
            f"C1-only economics: {json.dumps(sanitize(marginal['c1_only_trade_economics']))}.",
            "",
            "## Decision",
            "",
            f"C1 economic viability: **{decision['c1_economic_viability']}**.",
            f"C1 full quality floor: **{decision['c1_full_quality_floor']}**.",
            f"C1 materially improves C0: **{decision['c1_materially_improves_c0']}**.",
            f"Shadow candidate frozen: **{decision['shadow_candidate_frozen']}**.",
            f"Classification: **{decision['classification']}**.",
            "",
            "## Single next research hypothesis (not implemented)",
            "",
            decision["single_next_research_hypothesis"],
            "",
            "## Evidence limits",
            "",
            "All folds are previously inspected development data. The paired causal attribution may be internally valid, but it is not untouched final OOS proof. Historical CSV identities are hashed; their large raw snapshots are excluded from Git and durable remote raw-data preservation is not claimed.",
            "",
            "## Operational safety",
            "",
            "`gemini.py` and `gold_long_recent_candidate_xgb.json` remained byte-identical.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    assert TRAIN_MONTHS == 18 and THRESHOLD == 0.75
    assert N_ESTIMATORS == 220 and RANDOM_STATE == 42
    assert semantics.SIMULATORS[-1].simulator == "S5"
    times = pd.date_range("2026-01-01", periods=100, freq="min")
    frame = pd.DataFrame(
        {
            "TIME_DT": times,
            "OPEN": np.full(100, 100.0),
            "HIGH": np.full(100, 100.1),
            "LOW": np.full(100, 99.9),
            "CLOSE": np.full(100, 100.0),
            "ATR": np.full(100, 1.0),
            "SPREAD": np.full(100, 30.0),
        }
    )
    frame.loc[0, ["HIGH", "LOW"]] = [102.0, 98.0]
    labels = build_execution_aligned_labels(frame)
    assert labels["C1_EXIT_TYPE"].iat[0] == 2  # same-bar stop-first
    assert labels["C1_TARGET"].iat[0] == 0
    assert labels["C1_EXIT_INDEX"].iat[1] == 91  # wall-clock timeout at 90 minutes
    assert labels["C1_TARGET"].iat[1] == 0
    tp_frame = frame.copy()
    tp_frame["M1_RSI"] = 50.0
    tp_frame.loc[2, "HIGH"] = 102.0
    tp_labels = build_execution_aligned_labels(tp_frame)
    tp_frame["buy_prob"] = 0.0
    tp_frame.loc[2, "buy_prob"] = 1.0
    tp_frame["sell_prob"] = 0.0
    cohort = semantics.finalize_cohort(tp_frame, "SELF_CHECK", offset_hours=0)
    cohort["session_api"] = True
    cohort["session_actual_utc"] = True
    ledger, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
    assert len(ledger) == 1 and ledger[0]["exit_index"] == 2
    assert ledger[0]["exit_reason"] == "take_profit"
    assert math.isclose(ledger[0]["net_r"], tp_labels["C1_NET_R"].iat[2])
    assert array_sha256(np.arange(4, dtype=np.int64)) == array_sha256(np.arange(4, dtype=np.int64))
    print("SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    for path in (GEMINI_FILE, OPERATIONAL_MODEL, S5_SOURCE, S5_RUN / "FINALIZED.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    operational_before, manifest = preregister(run_dir)
    self_check()
    definition_path = run_dir / "label_definition.json"
    write_json(definition_path, label_definition())

    drl_trading_v2.DATA_DIR = str(ROOT)
    history, features = prepare_barrier_data()
    history = history.copy().reset_index(drop=True)
    if len(features) != 31:
        raise RuntimeError(f"Expected 31 frozen features, got {len(features)}")
    incumbent = xgb.XGBClassifier()
    incumbent.load_model(OPERATIONAL_MODEL)
    if incumbent.get_booster().feature_names != features:
        raise RuntimeError("Current artifact feature order differs from historical pipeline")
    del incumbent
    print(
        f"HISTORY rows={len(history):,} {history['TIME_DT'].iat[0]}..{history['TIME_DT'].iat[-1]} "
        f"features={len(features)}",
        flush=True,
    )

    labels = build_execution_aligned_labels(history)
    scored, provenance, label_rows, models = paired_training(
        history, labels, features, run_dir
    )
    total_train_rows = sum(row["training_rows"] for row in label_rows)
    pooled_label = {
        "fold": "pooled_training_rows_fold_weighted",
        "training_rows": total_train_rows,
        "c0_positive_prevalence": sum(row["c0_positive_prevalence"] * row["training_rows"] for row in label_rows) / total_train_rows,
        "c1_positive_prevalence": sum(row["c1_positive_prevalence"] * row["training_rows"] for row in label_rows) / total_train_rows,
        "agreeing_positive": sum(row["agreeing_positive"] for row in label_rows),
        "agreeing_negative": sum(row["agreeing_negative"] for row in label_rows),
        "c0_positive_c1_negative": sum(row["c0_positive_c1_negative"] for row in label_rows),
        "c0_negative_c1_positive": sum(row["c0_negative_c1_positive"] for row in label_rows),
        "disagreement_count": sum(row["disagreement_count"] for row in label_rows),
        "disagreement_rate": sum(row["disagreement_count"] for row in label_rows) / total_train_rows,
        "clean_window_vs_legacy_first_touch": sum(row["clean_window_vs_legacy_first_touch"] for row in label_rows),
        "legacy_label_vs_runtime_geometry_or_horizon": sum(row["legacy_label_vs_runtime_geometry_or_horizon"] for row in label_rows),
        "timeout_realized_outcome": sum(row["timeout_realized_outcome"] for row in label_rows),
        "runtime_stop_first_loss": sum(row["runtime_stop_first_loss"] for row in label_rows),
        "runtime_tp_not_legacy_positive": sum(row["runtime_tp_not_legacy_positive"] for row in label_rows),
        "cost_flips_gross_positive_to_nonpositive": sum(row["cost_flips_gross_positive_to_nonpositive"] for row in label_rows),
        "next_open_entry_and_runtime_path_unisolated_component": sum(row["next_open_entry_and_runtime_path_unisolated_component"] for row in label_rows),
        "cause_counts_are_overlapping_diagnostics": True,
    }
    label_rows.append(pooled_label)

    fold_metrics, trades, execution = execute_models(scored)
    probability_rows, ranking_rows, decile_rows = probability_and_ranking(scored)
    identity_rows, marginal = marginal_identity(execution)
    decision = decisions(fold_metrics, ranking_rows)
    shadow, shadow_model = freeze_shadow_if_qualified(
        history, labels, features, run_dir, decision, definition_path
    )
    if shadow_model is not None:
        models.append(shadow_model)

    write_csv(run_dir / "fold_metrics.csv", fold_metrics)
    delta_row = {
        "model_id": "C1_minus_C0",
        "fold": "pooled_delta",
        **decision["c1_minus_c0"],
    }
    write_csv(run_dir / "model_comparison.csv", [*fold_metrics, delta_row])
    write_csv(run_dir / "label_comparison.csv", label_rows)
    write_csv(run_dir / "probability_diagnostics.csv", probability_rows)
    write_csv(run_dir / "ranking_diagnostics.csv", ranking_rows)
    write_csv(run_dir / "ranking_deciles.csv", decile_rows)
    write_csv(run_dir / "trade_ledger.csv", trades)
    write_csv(run_dir / "trade_identity_comparison.csv", identity_rows or [{"match_type": "none"}])
    write_json(
        run_dir / "fold_model_provenance.json",
        {"features": features, "folds": provenance, "models": models},
    )
    oof_path = run_dir / "paired_oof_predictions.npz"
    save_oof(oof_path, scored)

    metrics = {
        "experiment": EXPERIMENT,
        "fold_metrics": fold_metrics,
        "label_comparison": label_rows,
        "probability_diagnostics": probability_rows,
        "ranking_diagnostics": ranking_rows,
        "marginal_trade_identity": marginal,
        "execution_audits": execution["audits"],
        "decision": decision,
        "quality_floor": QUALITY_FLOOR,
        "catastrophic_fold_definition": CATASTROPHIC_FOLD,
    }
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(report_text(metrics), encoding="utf-8")

    model_sha_path = run_dir / "models.sha256"
    model_sha_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in models),
        encoding="utf-8",
    )
    primary_model = shadow_model or next(
        item for item in models if item["model_id"] == MODEL_IDS[1] and item["fold"] == "2023_2024"
    )
    (run_dir / "model.sha256").write_text(
        f"{primary_model['sha256']}  {Path(primary_model['path']).name}\n",
        encoding="utf-8",
    )

    operational_after = {
        GEMINI_FILE.name: sha256(GEMINI_FILE),
        OPERATIONAL_MODEL.name: sha256(OPERATIONAL_MODEL),
    }
    if operational_before != operational_after:
        raise RuntimeError("Operational artifact changed during paired training")

    manifest = read_json(run_dir / "manifest.json")
    source_files = source_inventory()
    manifest["data"].update(
        {
            "symbols": ["GOLD#"],
            "data_sources": ["repository-local XM historical CSV exports"],
            "source_files": source_files,
            "timezone": (
                "naive broker/export timestamps retained; exact historical UTC mapping unresolved; "
                "both models use identical timestamps and S5 session semantics"
            ),
            "data_start_utc": history["TIME_DT"].iat[0].isoformat(),
            "data_end_utc": history["TIME_DT"].iat[-1].isoformat(),
            "train_start_utc": min(item["train_start"] for item in provenance),
            "train_end_utc": max(item["train_feature_end"] for item in provenance),
            "train_rows": total_train_rows,
            "validation_start_utc": FOLDS[0][1].isoformat(),
            "validation_end_utc": FOLDS[-1][2].isoformat(),
            "validation_rows": len(scored),
            "test_start_utc": "not_applicable_no_untouched_test",
            "test_end_utc": "not_applicable_no_untouched_test",
            "test_rows": 0,
            "purge_details": (
                "Per fold use the intersection of C0 240-row label maturity and exact C1 S5 "
                "exit maturity; latest information timestamp is strictly before score start."
            ),
            "embargo_details": "strict maturity purge; no additional embargo",
            "raw_snapshot_retained": False,
            "reproducibility_claim": "code/model/OOF evidence retained; full raw-data reproduction not claimed",
            "mt5_fetch": {
                "used": False,
                "not_applicable_reason": "This paired historical run uses fixed local CSV exports, not MT5 retrieval.",
            },
            "folds": provenance,
            "scored_oof_path": oof_path.relative_to(run_dir).as_posix(),
            "scored_oof_sha256": sha256(oof_path),
        }
    )
    manifest["model"].update(
        {
            "features": features,
            "feature_count": len(features),
            "artifact_path": primary_model["path"],
            "artifact_sha256": primary_model["sha256"],
            "retention_status": "all six fold models stored in run directory; primary manifest identity is C1 2023_2024 fold model",
            "fold_models_trained": len(models),
            "fold_model_inventory": "fold_model_provenance.json",
            "label_definition_sha256": sha256(definition_path),
        }
    )
    pooled_c1 = next(row for row in fold_metrics if row["model_id"] == MODEL_IDS[1] and row["fold"] == "pooled")
    manifest["registry"].update(
        {
            "parent_or_incumbent": f"{OPERATIONAL_MODEL.name}@{operational_before[OPERATIONAL_MODEL.name]}",
            "selected_configuration": "none_paired_C0_vs_C1_fixed_label_hypothesis",
            "trades_per_day": pooled_c1["trades_per_day"],
            "realized_win_rate": pooled_c1["realized_wr"],
            "pf": pooled_c1["pf"],
            "mean_r": pooled_c1["mean_r"],
            "pnl": pooled_c1["pnl_r"],
            "max_dd": pooled_c1["max_dd_r"],
            "validator_result": "PENDING",
        }
    )
    manifest["operational_hashes_after"] = operational_after
    manifest["research_decision"] = decision
    manifest["shadow_candidate"] = shadow
    if shadow["frozen"]:
        manifest["evidence_status"]["new_forward_cutoff"] = shadow["new_forward_cutoff"]
    for path, kind in (
        (definition_path, "frozen_label_definition"),
        (run_dir / "fold_model_provenance.json", "fold_model_provenance"),
        (run_dir / "fold_metrics.csv", "fold_metrics"),
        (run_dir / "model_comparison.csv", "paired_model_comparison"),
        (run_dir / "label_comparison.csv", "label_comparison"),
        (run_dir / "probability_diagnostics.csv", "probability_diagnostics"),
        (run_dir / "ranking_diagnostics.csv", "ranking_diagnostics"),
        (run_dir / "ranking_deciles.csv", "ranking_deciles"),
        (run_dir / "trade_ledger.csv", "executable_trade_ledger"),
        (run_dir / "trade_identity_comparison.csv", "marginal_trade_identity"),
        (oof_path, "paired_oof_predictions"),
        (run_dir / "models.sha256", "fold_model_hash_inventory"),
        (run_dir / "model.sha256", "primary_model_hash"),
        (run_dir / "metrics.json", "metrics"),
        (run_dir / "report.md", "report"),
    ):
        add_artifact(manifest, run_dir, path, kind)
    for item in models:
        add_artifact(manifest, run_dir, run_dir / item["path"], "trained_fold_model")
    if shadow["frozen"]:
        add_artifact(manifest, run_dir, run_dir / shadow["specification_path"], "frozen_shadow_candidate_specification")
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"COMPLETE viability={decision['c1_economic_viability']} "
        f"full_floor={decision['c1_full_quality_floor']} "
        "operational_artifact_changed=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
