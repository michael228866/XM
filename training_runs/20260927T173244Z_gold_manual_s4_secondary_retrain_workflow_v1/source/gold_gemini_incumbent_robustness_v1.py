from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import drl_trading_v2
import gold_gemini_core_gate_v1 as core
from barrier_classifier_strategy import evaluate as legacy_evaluate
from barrier_classifier_strategy import HORIZON as LABEL_HORIZON
from barrier_classifier_strategy import LABEL_SL_ATR, LABEL_TP_ATR
from barrier_classifier_strategy import MIN_SL_PRICE as LABEL_MIN_SL_PRICE
from barrier_classifier_strategy import MIN_TP_PRICE as LABEL_MIN_TP_PRICE
from barrier_final_train import FINAL_PARAMS, prepare_barrier_data
from barrier_research_suite import make_direction_probs, predict_positive, train_binary_model
from gold_generation11_execution_aligned import add_targets


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "GEMINI INCUMBENT ROBUSTNESS ATTRIBUTION V1"
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
GEMINI_FILE = ROOT / "gemini.py"
PRIOR_REPORT_JSON = ROOT / "gold_long_recent_walk_forward.json"
PRIOR_REPORT_MD = ROOT / "gold_long_recent_walk_forward.md"
PRIOR_RUN = ROOT / "training_runs" / "20260902T075259Z_gemini_core_gate_v1"
PRIOR_PREDICTIONS = PRIOR_RUN / "oof_predictions.npz"
PRIOR_MANIFEST = PRIOR_RUN / "manifest.json"
SIGNAL_LOG = ROOT / "gemini_signal_log.csv"
TRADE_LOG = ROOT / "gemini_trade_history.csv"

PREVIOUS_FORWARD_CUTOFF = "2026-09-01T02:00:00Z"
PREVIOUS_FORWARD_STATUS = "contaminated_for_future_gate_selection"
MODEL_GENERATED_AT = pd.Timestamp("2026-08-25T06:49:39.192110Z")
LABEL_BUFFER_END = pd.Timestamp("2025-01-15 00:00:00")

FOLDS = core.FOLDS
WINDOWS: dict[str, int | None] = {
    "W0_expanding": None,
    "W1_trailing_24m": 24,
    "W2_trailing_18m": 18,
    "W3_trailing_12m": 12,
}
THRESHOLD = 0.75
RSI_POLICY = "R0"
MIN_ENTRY_RSI = 22.0
EXCLUDED_RSI = (35.0, 45.0)
PROBABILITY_BUCKETS = (
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 1.0000001),
)
PROBABILITY_LEVELS = (0.60, 0.65, 0.70, 0.75, 0.80)
N_ESTIMATORS = 220
RANDOM_STATE = 42


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


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def add_artifact(manifest: dict[str, Any], run_dir: Path, path: Path, kind: str) -> None:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    entry = {
        "kind": kind,
        "path": relative,
        "sha256": sha256(path),
        "retention_status": "stored_in_run_directory_git_archival_pending",
    }
    existing = {item.get("path") for item in manifest.get("artifacts", [])}
    if relative not in existing:
        manifest.setdefault("artifacts", []).append(entry)


def snapshot_input(source: Path, destination: Path, compress: bool = False) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with source.open("rb") as reader, gzip.open(destination, "wb", compresslevel=6) as writer:
            shutil.copyfileobj(reader, writer)
    else:
        shutil.copy2(source, destination)
    return destination


def preregister(run_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "in_progress":
        raise RuntimeError("Training run is not in progress")
    if sha256(Path(__file__)) != manifest.get("training_script_sha256"):
        raise RuntimeError("Executed diagnostic script differs from immutable snapshot")
    if manifest.get("git_dirty") is not False:
        raise RuntimeError("Formal diagnostic requires pre_run_git_dirty=false")

    head = git("rev-parse", "HEAD")
    tracking_ref = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    tracking_sha = git("rev-parse", tracking_ref)
    if head != tracking_sha or head != manifest.get("git_commit"):
        raise RuntimeError("Pre-run HEAD, upstream, and manifest commit do not match")

    operational = {
        GEMINI_FILE.name: sha256(GEMINI_FILE),
        MODEL_FILE.name: sha256(MODEL_FILE),
    }
    manifest["pre_run_git"] = {
        "pre_run_git_commit": manifest["git_commit"],
        "pre_run_git_dirty": False,
        "remote_branch": tracking_ref,
        "head_sha": head,
        "upstream_sha": tracking_sha,
        "head_equals_upstream": True,
    }
    manifest["evidence_status"] = {
        "classification": "development_diagnostic_only",
        "previous_forward_cutoff": PREVIOUS_FORWARD_CUTOFF,
        "previous_forward_status": PREVIOUS_FORWARD_STATUS,
        "new_forward_cutoff": None,
        "candidate_selected": False,
        "production_claim": False,
    }
    manifest["diagnostic_design"] = {
        "strategy_configurations": 1,
        "direction": "LONG only",
        "threshold": THRESHOLD,
        "minimum_entry_rsi": MIN_ENTRY_RSI,
        "excluded_rsi_range": list(EXCLUDED_RSI),
        "tp_atr": core.TP_ATR,
        "sl_atr": core.SL_ATR,
        "max_hold_minutes": core.MAX_HOLD_MINUTES,
        "max_open_positions": 1,
        "training_windows": WINDOWS,
        "probability_buckets": [
            {"low_inclusive": low, "high_exclusive": min(high, 1.0)}
            for low, high in PROBABILITY_BUCKETS
        ],
        "selection_or_promotion": "none",
    }
    manifest["search"].update(
        {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": None,
            "not_applicable_reason": (
                "No candidate search or strategy selection; W0-W3 are four pre-specified "
                "training-window diagnostics under one frozen strategy configuration."
            ),
        }
    )
    manifest["model"].update(
        {
            "trained": False,
            "model_type": "XGBoost binary logistic diagnostic fold replicas",
            "parameters": {
                "n_estimators": N_ESTIMATORS,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 80,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": RANDOM_STATE,
                "tree_method": "hist",
            },
            "boosted_rounds_or_estimators": N_ESTIMATORS,
            "label_definition": "BARRIER_TARGET == 1 versus all other classes",
            "horizon": LABEL_HORIZON,
            "label_tp_sl_semantics": (
                f"clean long barrier over {LABEL_HORIZON} future M1 rows; "
                f"TP=max({LABEL_TP_ATR} ATR,{LABEL_MIN_TP_PRICE}); "
                f"SL=max({LABEL_SL_ATR} ATR,{LABEL_MIN_SL_PRICE})"
            ),
            "execution_tp_sl_semantics": (
                "Frozen control: long HIGH/LOW first touch beginning after entry, "
                "TP=1.3 ATR, SL=1.6 ATR, stop-first same-bar, timeout=90 M1 rows"
            ),
            "calibration_method": "none; diagnosis only",
            "artifact_path": None,
            "artifact_sha256": None,
            "retention_status": "diagnostic_fold_models_not_saved",
            "not_applicable_reason": (
                "W1-W3 fold replicas are trained only to produce frozen diagnostic OOF "
                "predictions; they are not candidate artifacts. W0 is reused byte-for-byte "
                "from the preserved parent run."
            ),
            "diagnostic_fold_models_planned": 9,
            "operational_candidate_trained": False,
        }
    )
    manifest["promotion"].update(
        {
            "requested": False,
            "gate_result": "not_requested_diagnostic_only",
            "replacement_authorized": False,
            "operational_artifact_changed": False,
        }
    )
    manifest["operational_hashes_before"] = operational
    manifest["dependency_sha256"] = {
        "gold_gemini_core_gate_v1.py": sha256(ROOT / "gold_gemini_core_gate_v1.py"),
        "gold_long_recent_walk_forward.py": sha256(ROOT / "gold_long_recent_walk_forward.py"),
        "barrier_classifier_strategy.py": sha256(ROOT / "barrier_classifier_strategy.py"),
        "drl_trading_v2.py": sha256(ROOT / "drl_trading_v2.py"),
    }
    write_json(manifest_path, manifest)
    print(f"PREREGISTERED commit={head} upstream={tracking_ref}", flush=True)
    return operational, manifest


def source_files() -> list[dict[str, Any]]:
    values = []
    for path in sorted(ROOT.glob("GOLD#_*.csv")):
        values.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "retention_status": "local_historical_export_not_durable_git_artifact",
            }
        )
    return values


def build_base_oof(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    with np.load(PRIOR_PREDICTIONS) as prior:
        global_index = prior["global_index"].astype(np.int64)
        time_ns = prior["time_ns"].astype(np.int64)
        fold_code = prior["fold_code"].astype(np.int8)
        score_w0 = prior["score"].astype(np.float32)
    expected_time = history["TIME_DT"].to_numpy(dtype="datetime64[ns]")[global_index].astype(np.int64)
    if not np.array_equal(time_ns, expected_time):
        raise RuntimeError("Preserved W0 predictions no longer align to source history")
    fold_names = np.asarray([name for name, _, _ in FOLDS], dtype=object)[fold_code]
    for code, (name, start, end) in enumerate(FOLDS):
        mask = fold_code == code
        times = pd.to_datetime(time_ns[mask])
        if len(times) == 0 or times.min() < start or times.max() >= end:
            raise RuntimeError(f"W0 fold-code boundary mismatch: {name}")

    close = history["CLOSE"].to_numpy(dtype=np.float64)
    high = history["HIGH"].to_numpy(dtype=np.float64)
    low = history["LOW"].to_numpy(dtype=np.float64)
    atr = history["ATR"].to_numpy(dtype=np.float64)
    outcome = history["LONG_OUTCOME"].to_numpy(dtype=np.int8)[global_index]
    offset = history["LONG_EXIT_OFFSET"].to_numpy(dtype=np.int16)[global_index]
    exit_index = global_index + offset.astype(np.int64)
    if np.any(outcome < 0) or np.any(offset <= 0) or np.any(exit_index >= len(history)):
        raise RuntimeError("Immature first-touch outcome entered the diagnostic cohort")
    take = np.maximum(atr[global_index] * core.TP_ATR, core.MIN_TP_PRICE)
    stop = np.maximum(atr[global_index] * core.SL_ATR, core.MIN_SL_PRICE)
    gross = np.where(
        outcome == 1,
        take,
        np.where(outcome == 2, -stop, close[exit_index] - close[global_index]),
    )
    spread, observed = core.effective_spread(history)
    denominator = stop + spread[global_index] * core.POINT
    reward = (
        gross - (spread[global_index] + core.BASE_EXTRA_COST_POINTS) * core.POINT
    ) / denominator
    stress_reward = (
        gross - (spread[global_index] + core.STRESS_EXTRA_COST_POINTS) * core.POINT
    ) / denominator
    times = pd.to_datetime(time_ns)
    base = pd.DataFrame(
        {
            "global_index": global_index,
            "fold": fold_names,
            "time": times,
            "score": score_w0,
            "rsi": history["M1_RSI"].to_numpy(dtype=np.float64)[global_index],
            "session_ok": (
                pd.Series(times).dt.hour.isin(core.ALLOWED_HOURS).to_numpy()
                & pd.Series(times).dt.dayofweek.isin(core.ALLOWED_WEEKDAYS).to_numpy()
            ),
            "spread_points": spread[global_index],
            "spread_observed": observed[global_index],
            "spread_ok": spread[global_index] <= core.spread_limit(atr[global_index]),
            "outcome": outcome,
            "target": history["BARRIER_TARGET"].to_numpy(dtype=np.int8)[global_index] == 1,
            "exit_offset": offset,
            "exit_time": pd.to_datetime(
                history["TIME_DT"].to_numpy(dtype="datetime64[ns]")[exit_index]
            ),
            "gross_pnl_price": gross,
            "denominator": denominator,
            "reward": reward,
            "stress_reward": stress_reward,
            "close": close[global_index],
            "high": high[global_index],
            "low": low[global_index],
            "atr": atr[global_index],
        }
    )
    if not base["time"].is_monotonic_increasing:
        raise RuntimeError("Diagnostic OOF base is not chronological")
    identity = {
        "source_run": PRIOR_RUN.name,
        "source_path": PRIOR_PREDICTIONS.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(PRIOR_PREDICTIONS),
        "rows": int(len(base)),
        "alignment_verified": True,
    }
    return base, score_w0, identity


def train_window_scores(
    history: pd.DataFrame,
    features: list[str],
    base: pd.DataFrame,
    score_w0: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    scores = {"W0_expanding": score_w0.copy()}
    prior = read_json(PRIOR_MANIFEST)["data"]["folds"]
    provenance = [dict(item, scheme="W0_expanding", source="preserved_parent_oof") for item in prior]
    times = history["TIME_DT"]

    for scheme, months in WINDOWS.items():
        if months is None:
            continue
        values = np.full(len(base), np.nan, dtype=np.float32)
        for fold_name, fold_start, fold_end in FOLDS:
            lower = fold_start - pd.DateOffset(months=months)
            pre = history.index[(times >= lower) & (times < fold_start)].to_numpy(dtype=np.int64)
            if len(pre) <= LABEL_HORIZON:
                raise RuntimeError(f"Insufficient {scheme} training rows for {fold_name}")
            train_index = pre[:-LABEL_HORIZON]
            score_positions = np.flatnonzero(base["fold"].to_numpy() == fold_name)
            score_index = base["global_index"].to_numpy(dtype=np.int64)[score_positions]
            maturity_index = int(train_index[-1]) + LABEL_HORIZON
            if maturity_index >= int(score_index[0]):
                raise RuntimeError(f"Label maturity overlap: {scheme}/{fold_name}")
            train = history.loc[train_index]
            scored = history.loc[score_index]
            print(
                f"TRAIN {scheme}/{fold_name}: rows={len(train):,} "
                f"{train['TIME_DT'].iat[0]}..{train['TIME_DT'].iat[-1]} "
                f"score_rows={len(scored):,}",
                flush=True,
            )
            model = train_binary_model(train, features, 1, N_ESTIMATORS)
            values[score_positions] = predict_positive(model, scored, features)
            del model, train, scored
            gc.collect()
            provenance.append(
                {
                    "scheme": scheme,
                    "fold": fold_name,
                    "train_start": history["TIME_DT"].iat[int(train_index[0])].isoformat(),
                    "train_end": history["TIME_DT"].iat[int(train_index[-1])].isoformat(),
                    "train_rows": int(len(train_index)),
                    "window_months": months,
                    "label_horizon_rows": LABEL_HORIZON,
                    "latest_training_label_bar": history["TIME_DT"].iat[maturity_index].isoformat(),
                    "score_start": history["TIME_DT"].iat[int(score_index[0])].isoformat(),
                    "score_end": history["TIME_DT"].iat[int(score_index[-1])].isoformat(),
                    "score_rows": int(len(score_index)),
                    "chronology_assertion": bool(maturity_index < int(score_index[0])),
                    "model_retention": "not_saved_diagnostic_replica",
                }
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Missing OOF scores for {scheme}")
        scores[scheme] = values
    return scores, provenance


def metric_rows(
    base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for scheme, values in scores.items():
        base["score"] = values
        trades, audit = core.execute(base, THRESHOLD, RSI_POLICY)
        nested[scheme] = {"folds": {}}
        for fold_name, start, end in FOLDS:
            selected = [trade for trade in trades if trade["fold"] == fold_name]
            metrics = core.trade_metrics(selected, core.fold_days(start, end))
            metrics.update(audit[fold_name])
            nested[scheme]["folds"][fold_name] = metrics
            rows.append({"scheme": scheme, "fold": fold_name, **metrics})
        days = sum(core.fold_days(start, end) for _, start, end in FOLDS)
        pooled = core.trade_metrics(trades, days)
        for key in next(iter(audit.values())):
            pooled[key] = int(sum(item[key] for item in audit.values()))
        nested[scheme]["pooled"] = pooled
        rows.append({"scheme": scheme, "fold": "pooled", **pooled})
        for trade in trades:
            ledger.append({"scheme": scheme, **trade})
        print(
            f"RESULT {scheme}: trades={pooled['trades']} tpd={pooled['trades_per_day']:.4f} "
            f"WR={pooled['realized_wr']:.2%} PF={pooled['pf']:.4f} "
            f"Mean-R={pooled['mean_r']:.4f}",
            flush=True,
        )
    return rows, ledger, nested


def probability_distribution_rows(
    base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for scheme, values in scores.items():
        for fold in [name for name, _, _ in FOLDS] + ["pooled"]:
            mask = np.ones(len(base), dtype=bool) if fold == "pooled" else base["fold"].eq(fold).to_numpy()
            part = values[mask]
            row = {
                "scheme": scheme,
                "fold": fold,
                "observations": int(len(part)),
                "mean": float(np.mean(part)),
                "median": float(np.median(part)),
                "p75": float(np.quantile(part, 0.75)),
                "p90": float(np.quantile(part, 0.90)),
                "p95": float(np.quantile(part, 0.95)),
                "p99": float(np.quantile(part, 0.99)),
                "maximum": float(np.max(part)),
            }
            row.update({f"fraction_ge_{level:.2f}": float(np.mean(part >= level)) for level in PROBABILITY_LEVELS})
            rows.append(row)
    return rows


def ece(score: np.ndarray, target: np.ndarray) -> float:
    total = len(score)
    error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        high = edges[index + 1]
        mask = (score >= edges[index]) & ((score < high) if index < 9 else (score <= high))
        if mask.any():
            error += mask.sum() / total * abs(float(score[mask].mean()) - float(target[mask].mean()))
    return float(error)


def calibration_rows(
    base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    buckets = []
    target_all = base["target"].to_numpy(dtype=np.float64)
    for scheme, values in scores.items():
        for fold in [name for name, _, _ in FOLDS] + ["pooled"]:
            mask = np.ones(len(base), dtype=bool) if fold == "pooled" else base["fold"].eq(fold).to_numpy()
            score = values[mask].astype(np.float64)
            target = target_all[mask]
            ge = score >= THRESHOLD
            summaries.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "observations": int(len(score)),
                    "predicted_mean": float(score.mean()),
                    "observed_positive_label_rate": float(target.mean()),
                    "brier": float(np.mean((score - target) ** 2)),
                    "ece_10_equal_width": ece(score, target),
                    "ge_075_observations": int(ge.sum()),
                    "ge_075_predicted_mean": float(score[ge].mean()) if ge.any() else None,
                    "ge_075_observed_positive_label_rate": float(target[ge].mean()) if ge.any() else None,
                }
            )
            edges = np.linspace(0.0, 1.0, 11)
            for index in range(10):
                high = edges[index + 1]
                selected = (score >= edges[index]) & ((score < high) if index < 9 else (score <= high))
                buckets.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "bucket_low": edges[index],
                        "bucket_high": high,
                        "observations": int(selected.sum()),
                        "predicted_mean": float(score[selected].mean()) if selected.any() else None,
                        "observed_positive_label_rate": float(target[selected].mean()) if selected.any() else None,
                        "absolute_calibration_error": (
                            abs(float(score[selected].mean()) - float(target[selected].mean()))
                            if selected.any()
                            else None
                        ),
                    }
                )
    return summaries, buckets


def probability_bucket_rows(base: pd.DataFrame, scores: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for scheme, values in scores.items():
        for low, high in PROBABILITY_BUCKETS:
            bucket = (values >= low) & (values < high)
            diagnostic = base.copy(deep=False)
            diagnostic = diagnostic.assign(score=np.where(bucket, values, -1.0))
            trades, _ = core.execute(diagnostic, low, RSI_POLICY)
            episode = core.episode_mask(diagnostic, low)
            for fold in [name for name, _, _ in FOLDS] + ["pooled"]:
                fold_mask = np.ones(len(base), dtype=bool) if fold == "pooled" else base["fold"].eq(fold).to_numpy()
                selected_trades = trades if fold == "pooled" else [item for item in trades if item["fold"] == fold]
                days = (
                    sum(core.fold_days(start, end) for _, start, end in FOLDS)
                    if fold == "pooled"
                    else next(core.fold_days(start, end) for name, start, end in FOLDS if name == fold)
                )
                metrics = core.trade_metrics(selected_trades, days)
                rows.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "bucket_low": low,
                        "bucket_high": min(high, 1.0),
                        "observations": int((bucket & fold_mask).sum()),
                        "independent_episodes": int((episode & fold_mask).sum()),
                        "executable_trades": metrics["trades"],
                        "realized_wr": metrics["realized_wr"],
                        "mean_r": metrics["mean_r"],
                        "pf": metrics["pf"],
                        "average_realized_r": metrics["mean_r"],
                    }
                )
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def raw_reward_metrics(values: np.ndarray) -> dict[str, Any]:
    result = core.reward_metrics(values.astype(np.float64))
    return {
        "observations": result["trades"],
        "realized_wr": result["realized_wr"],
        "mean_r": result["mean_r"],
        "pf": result["pf"],
    }


def ranking_rows(
    base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    deciles = []
    rewards_all = base["reward"].to_numpy(dtype=np.float64)
    for scheme, values in scores.items():
        for fold in [name for name, _, _ in FOLDS] + ["pooled"]:
            mask = np.ones(len(base), dtype=bool) if fold == "pooled" else base["fold"].eq(fold).to_numpy()
            score = values[mask].astype(np.float64)
            reward = rewards_all[mask]
            ranks = pd.Series(score).rank(method="first", pct=True).to_numpy()
            decile = np.minimum((ranks * 10).astype(int), 9) + 1
            top10 = decile == 10
            top20 = decile >= 9
            bottom10 = decile == 1
            summary = {
                "scheme": scheme,
                "fold": fold,
                "observations": int(len(score)),
                "spearman_probability_vs_realized_r": spearman(score, reward),
            }
            for label, selected in (("top_decile", top10), ("top_quintile", top20), ("bottom_decile", bottom10)):
                summary.update({f"{label}_{key}": value for key, value in raw_reward_metrics(reward[selected]).items()})
            summaries.append(summary)
            for number in range(1, 11):
                selected = decile == number
                deciles.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "decile": number,
                        "probability_min": float(score[selected].min()),
                        "probability_max": float(score[selected].max()),
                        **raw_reward_metrics(reward[selected]),
                    }
                )
    return summaries, deciles


def regime_rows(history: pd.DataFrame, base: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    close = history["CLOSE"].astype(float)
    direction = np.sign(close.diff())
    history = history.copy()
    history["PAST_240_RETURN"] = close.shift(1) / close.shift(241) - 1.0
    history["PAST_60_DIRECTIONAL_PERSISTENCE"] = direction.shift(1).rolling(60).mean().abs()
    mtf = [name for name in features if name.endswith("_TREND")]
    rows = []
    for fold_name, _, _ in FOLDS:
        indexes = base.loc[base["fold"].eq(fold_name), "global_index"].to_numpy(dtype=np.int64)
        frame = history.loc[indexes]
        rows.append(
            {
                "fold": fold_name,
                "observations": int(len(frame)),
                "median_atr": float(frame["ATR"].median()),
                "median_atr_to_price": float((frame["ATR"] / frame["CLOSE"]).median()),
                "median_vola_ratio": float(frame["VOLA_RATIO"].median()),
                "mean_entry_known_roc5": float(frame["ROC_5"].mean()),
                "positive_entry_known_roc5_fraction": float((frame["ROC_5"] > 0).mean()),
                "mean_past_240_return": float(frame["PAST_240_RETURN"].mean()),
                "median_past_60_directional_persistence": float(frame["PAST_60_DIRECTIONAL_PERSISTENCE"].median()),
                "mean_mtf_bullish_fraction": float((frame[mtf] > 0).mean(axis=1).mean()),
                "context_use": "descriptive_only_not_a_filter",
            }
        )
    return rows


def legacy_params() -> dict[str, Any]:
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": THRESHOLD,
            "edge_threshold": 0.0,
            "tp_atr": core.TP_ATR,
            "sl_atr": core.SL_ATR,
            "max_hold": core.MAX_HOLD_MINUTES,
            "direction_mode": "long",
            "risk_per_trade": core.RISK_PER_TRADE,
            "allowed_entry_hours": sorted(core.ALLOWED_HOURS),
            "allowed_entry_weekdays": sorted(core.ALLOWED_WEEKDAYS),
            "excluded_rsi_ranges": [(0.0, MIN_ENTRY_RSI), EXCLUDED_RSI],
            "extra_cost_points": core.BASE_EXTRA_COST_POINTS,
        }
    )
    return params


def execution_reconciliation_rows(base: pd.DataFrame, score_w0: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    params = legacy_params()
    for fold_name, start, end in FOLDS:
        mask = base["fold"].eq(fold_name).to_numpy()
        frame = base.loc[mask]
        probs = make_direction_probs(score_w0[mask], "long")
        common = {
            "hours": frame["time"].dt.hour.to_numpy(dtype=np.int16),
            "weekdays": frame["time"].dt.dayofweek.to_numpy(dtype=np.int8),
            "dates": frame["time"].dt.date.to_numpy(),
            "rsi_values": frame["rsi"].to_numpy(dtype=np.float64),
        }
        for method, intrabar in (("legacy_close_only", False), ("legacy_simulator_high_low", True)):
            stats = legacy_evaluate(
                params,
                frame["close"].to_numpy(dtype=np.float64),
                frame["atr"].to_numpy(dtype=np.float64),
                probs,
                highs=frame["high"].to_numpy(dtype=np.float64) if intrabar else None,
                lows=frame["low"].to_numpy(dtype=np.float64) if intrabar else None,
                **common,
            )
            rows.append(
                {
                    "method": method,
                    "fold": fold_name,
                    "trades": int(stats["trades"]),
                    "trades_per_day": int(stats["trades"]) / core.fold_days(start, end),
                    "realized_wr": float(stats["win_rate"]),
                    "pf": float(stats["profit_factor"]),
                    "pnl_account_currency": float(stats["pnl"]),
                    "max_dd_pct": float(stats["max_drawdown_pct"]),
                    "tp_count": int(stats["take_profit_exits"]),
                    "sl_count": int(stats["stop_loss_exits"]),
                    "timeout_count": int(stats["timeout_exits"]),
                    "note": "fixed 30-point spread plus 5-point extra cost; legacy filtered-rising signal semantics",
                }
            )
    return rows


def copy_live_inputs(run_dir: Path) -> tuple[Path, Path]:
    inputs = run_dir / "input_snapshots"
    signal = snapshot_input(SIGNAL_LOG, inputs / "gemini_signal_log.csv.gz", compress=True)
    trades = snapshot_input(TRADE_LOG, inputs / "gemini_trade_history.csv", compress=False)
    return signal, trades


def live_summary(signal_snapshot: Path, trade_snapshot: Path) -> dict[str, Any]:
    signals = pd.read_csv(signal_snapshot, compression="gzip", low_memory=False)
    signals["event_dt"] = pd.to_datetime(signals["event_time"], utc=True, errors="coerce", format="mixed")
    sell = pd.to_numeric(signals["sell_prob"], errors="coerce")
    current = signals[(signals["event_dt"] >= MODEL_GENERATED_AT) & sell.eq(0.0)].copy()
    current["buy_prob"] = pd.to_numeric(current["buy_prob"], errors="coerce")
    if current.empty:
        return {"available": False, "reason": "No current long-binary signal rows"}

    history = pd.read_csv(trade_snapshot, low_memory=False)
    history["deal_dt"] = pd.to_datetime(history["deal_time"], utc=True, errors="coerce", format="mixed")
    numeric = ["profit", "commission", "swap", "fee"]
    for name in numeric:
        history[name] = pd.to_numeric(history[name], errors="coerce").fillna(0.0)
    history["net"] = history[numeric].sum(axis=1)
    exits = history[
        (history["deal_dt"] >= MODEL_GENERATED_AT)
        & history["symbol"].eq("GOLD#")
        & history["magic"].eq(20260514)
        & history["entry"].eq(1)
    ].copy()
    opened = current[current["status"].eq("order_opened")].copy()
    risk_by_position = {
        int(row.position_ticket): float(row.risk_budget)
        for row in opened.itertuples()
        if pd.notna(row.position_ticket) and pd.notna(row.risk_budget) and float(row.risk_budget) > 0
    }
    exits["risk_budget"] = exits["position_id"].map(risk_by_position)
    exits["net_r"] = exits["net"] / exits["risk_budget"]
    values = exits["net"].to_numpy(dtype=np.float64)
    gains = float(values[values > 0].sum())
    losses = float(-values[values <= 0].sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.maximum(equity, 0.0)) if len(equity) else np.array([0.0])
    span_days = max((current["event_dt"].max() - current["event_dt"].min()).total_seconds() / 86400.0, 1.0)
    reasons = exits["reason"].value_counts().to_dict()
    return {
        "available": True,
        "classification": "monitoring_only_contaminated_for_future_gate_selection",
        "signal_start": current["event_dt"].min().isoformat(),
        "signal_end": current["event_dt"].max().isoformat(),
        "scored_rows": int(len(current)),
        "p_ge_075_rows": int((current["buy_prob"] >= THRESHOLD).sum()),
        "independent_ge_075_episodes": int(
            ((current["buy_prob"] >= THRESHOLD) & ~(current["buy_prob"].shift(fill_value=0) >= THRESHOLD)).sum()
        ),
        "orders_opened": int(len(opened)),
        "closed_trades": int(len(exits)),
        "trades_per_day": float(len(exits) / span_days),
        "wins": int((values > 0).sum()),
        "losses": int((values <= 0).sum()),
        "realized_wr": float((values > 0).mean()) if len(values) else 0.0,
        "tp_first_wr": float(reasons.get(5, 0) / len(exits)) if len(exits) else 0.0,
        "pf": math.inf if losses == 0 and gains > 0 else (gains / losses if losses else 0.0),
        "mean_r": float(exits["net_r"].mean()) if exits["net_r"].notna().all() and len(exits) else None,
        "pnl": float(values.sum()),
        "max_dd": float(np.min(equity - peaks)) if len(equity) else 0.0,
        "tp_count": int(reasons.get(5, 0)),
        "sl_count": int(reasons.get(4, 0)),
        "timeout_count": int(len(exits) - reasons.get(5, 0) - reasons.get(4, 0)),
        "model_sha256": sha256(MODEL_FILE),
        "signal_snapshot_sha256": sha256(signal_snapshot),
        "trade_snapshot_sha256": sha256(trade_snapshot),
    }


def evidence_rows(
    prior: dict[str, Any],
    window_metrics: dict[str, Any],
    provenance: list[dict[str, Any]],
    live: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = prior["selected"]
    params = selected["params"]
    rows: list[dict[str, Any]] = []
    originals = (
        ("A_original_recent_validation", selected["validation"], "2026-04-01", "2026-06-01", "ephemeral validation model; not retained", None),
        ("B_original_recent_test", selected["test"], "2026-06-01", prior["data"]["end"], MODEL_FILE.name, sha256(MODEL_FILE)),
    )
    for evidence_id, stats, start, end, artifact, model_hash in originals:
        days = max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0, 1.0)
        rows.append(
            {
                "evidence_id": evidence_id,
                "train_interval": "2025-01-01 to score-start minus 240 rows",
                "scored_interval": f"{start} to {end}",
                "training_rows": None,
                "scored_rows": prior["data"]["validation_rows"] if "validation" in evidence_id else prior["data"]["test_rows"],
                "model_trained_strictly_before_scored_interval": "yes",
                "model_artifact_identity": artifact,
                "model_sha256": model_hash,
                "executable_trades": stats["trades"],
                "trades_per_day": stats["trades"] / days,
                "realized_wr": stats["win_rate"],
                "tp_first_wr": stats["take_profit_exits"] / stats["trades"],
                "pf": stats["profit_factor"],
                "mean_r": None,
                "pnl": stats["pnl"],
                "max_dd": stats["max_drawdown_pct"],
                "tp_count": stats["take_profit_exits"],
                "sl_count": stats["stop_loss_exits"],
                "timeout_count": stats["timeout_exits"],
                "threshold": params["threshold"],
                "rsi_policy": "minimum 22; exclude 35-45",
                "tp_sl": f"{params['tp_atr']}/{params['sl_atr']} ATR",
                "hold": params["max_hold"],
                "session": params["session_profile"],
                "costs": "fixed 30-point spread + 5-point extra; test_cost_10 separately retained",
                "spread_assumptions": "fixed 30 points; no observed-entry spread",
                "label_semantics": "240-row clean-window 1.8/1.2 ATR barrier",
                "execution_semantics": "legacy close-only threshold exits; intrabar HIGH/LOW not supplied",
                "evidence_classification": "selected development evidence; formerly called validation/test; not untouched",
            }
        )

    prov = {(item["scheme"], item["fold"]): item for item in provenance}
    for fold_name, _, _ in FOLDS:
        metrics = window_metrics["W0_expanding"]["folds"][fold_name]
        info = prov[("W0_expanding", fold_name)]
        rows.append(
            {
                "evidence_id": f"C_W0_OOF_{fold_name}",
                "train_interval": f"{info['train_start']} to {info['train_end']}",
                "scored_interval": f"{info['score_start']} to {info['score_end']}",
                "training_rows": info["train_rows"],
                "scored_rows": info["score_rows"],
                "model_trained_strictly_before_scored_interval": "yes",
                "model_artifact_identity": f"fold-specific replica; predictions in {PRIOR_PREDICTIONS.relative_to(ROOT).as_posix()}",
                "model_sha256": None,
                "executable_trades": metrics["trades"],
                "trades_per_day": metrics["trades_per_day"],
                "realized_wr": metrics["realized_wr"],
                "tp_first_wr": metrics["tp_first_wr"],
                "pf": metrics["pf"],
                "mean_r": metrics["mean_r"],
                "pnl": metrics["pnl"],
                "max_dd": metrics["max_dd"],
                "tp_count": metrics["tp_count"],
                "sl_count": metrics["sl_count"],
                "timeout_count": metrics["timeout_count"],
                "threshold": THRESHOLD,
                "rsi_policy": "minimum 22; exclude 35-45",
                "tp_sl": "1.3/1.6 ATR",
                "hold": 90,
                "session": "expanded hours; weekdays 0-4",
                "costs": "observed spread or 30-point fallback + 5-point extra",
                "spread_assumptions": "observed when >0; otherwise 30-point fallback; spread gate",
                "label_semantics": "240-row clean-window 1.8/1.2 ATR barrier",
                "execution_semantics": "HIGH/LOW first-touch; stop-first; threshold episode before entry filters",
                "evidence_classification": "genuine chronological OOF development evidence",
            }
        )
    pooled = window_metrics["W0_expanding"]["pooled"]
    rows.append(
        {
            "evidence_id": "D_W0_OOF_pooled",
            "train_interval": "fold-specific expanding history",
            "scored_interval": "2018-01-02 to 2024-12-31",
            "training_rows": None,
            "scored_rows": sum(item["score_rows"] for item in provenance if item["scheme"] == "W0_expanding"),
            "model_trained_strictly_before_scored_interval": "yes",
            "model_artifact_identity": "three fold-specific replicas; predictions retained",
            "model_sha256": None,
            "executable_trades": pooled["trades"],
            "trades_per_day": pooled["trades_per_day"],
            "realized_wr": pooled["realized_wr"],
            "tp_first_wr": pooled["tp_first_wr"],
            "pf": pooled["pf"],
            "mean_r": pooled["mean_r"],
            "pnl": pooled["pnl"],
            "max_dd": pooled["max_dd"],
            "tp_count": pooled["tp_count"],
            "sl_count": pooled["sl_count"],
            "timeout_count": pooled["timeout_count"],
            "threshold": THRESHOLD,
            "rsi_policy": "minimum 22; exclude 35-45",
            "tp_sl": "1.3/1.6 ATR",
            "hold": 90,
            "session": "expanded hours; weekdays 0-4",
            "costs": "observed spread or 30-point fallback + 5-point extra",
            "spread_assumptions": "observed/fallback with spread gate",
            "label_semantics": "240-row clean-window 1.8/1.2 ATR barrier",
            "execution_semantics": "HIGH/LOW first-touch; stop-first; threshold episode before filters",
            "evidence_classification": "pooled chronological OOF development evidence",
        }
    )
    if live.get("available"):
        rows.append(
            {
                "evidence_id": "E_current_live_monitoring",
                "train_interval": "2025-01-01 to 2026-06-01 minus 240 rows",
                "scored_interval": f"{live['signal_start']} to {live['signal_end']}",
                "training_rows": None,
                "scored_rows": live["scored_rows"],
                "model_trained_strictly_before_scored_interval": "yes",
                "model_artifact_identity": MODEL_FILE.name,
                "model_sha256": live["model_sha256"],
                "executable_trades": live["closed_trades"],
                "trades_per_day": live["trades_per_day"],
                "realized_wr": live["realized_wr"],
                "tp_first_wr": live["tp_first_wr"],
                "pf": live["pf"],
                "mean_r": live["mean_r"],
                "pnl": live["pnl"],
                "max_dd": live["max_dd"],
                "tp_count": live["tp_count"],
                "sl_count": live["sl_count"],
                "timeout_count": live["timeout_count"],
                "threshold": THRESHOLD,
                "rsi_policy": "minimum 22; exclude 35-45",
                "tp_sl": "1.3/1.6 ATR",
                "hold": 90,
                "session": "current operational session",
                "costs": "broker-realized net deal profit/commission/swap/fee",
                "spread_assumptions": "actual broker execution",
                "label_semantics": "not applicable to realized monitoring trades",
                "execution_semantics": "actual MT5 execution",
                "evidence_classification": live["classification"],
            }
        )
    return rows


def methodology_rows() -> list[dict[str, str]]:
    return [
        {"area": "training history", "original_recent": "2025-01-01 to each 2026 score boundary; recent rolling fit", "genuine_oof_control": "expanding 2014 history before each 2018-2024 fold", "materiality": "high", "can_explain_gap": "yes; different fitted models and regimes"},
        {"area": "training-window length", "original_recent": "about 15-17 months", "genuine_oof_control": "about 4, 7, and 9 years", "materiality": "high", "can_explain_gap": "tested directly by frozen W0-W3 diagnostic"},
        {"area": "model and class weighting", "original_recent": "binary logistic, 220 trees, lr .05, depth 4, min-child 80, balanced binary weights, seed 42", "genuine_oof_control": "same model family, parameters, weights, and seed", "materiality": "low", "can_explain_gap": "no verified architecture-parameter difference"},
        {"area": "selection", "original_recent": "288 threshold/TP/SL/hold/session combinations selected on Apr-May 2026", "genuine_oof_control": "reported fixed T0_R0, though parent run also evaluated 20 threshold/RSI combinations", "materiality": "high", "can_explain_gap": "yes; old 70% validation is selection-conditioned and optimistic"},
        {"area": "artifact identity", "original_recent": "exact operational artifact scored only Jun-Aug 2026 test", "genuine_oof_control": "three historical fold replicas; exact operational artifact not scored", "materiality": "critical", "can_explain_gap": "yes; 25.40% is architecture replication evidence, not exact-artifact evidence"},
        {"area": "feature formulas/order", "original_recent": "build_feature_frame uses add_indicators, shifted 31 features, backward MTF as-of", "genuine_oof_control": "prepare_barrier_data uses same add_indicators and shifted 31-feature order", "materiality": "low", "can_explain_gap": "no verified formula difference"},
        {"area": "feature data source", "original_recent": "dynamic MT5 copy_rates; raw snapshot not retained", "genuine_oof_control": "local XM CSV exports with hashes", "materiality": "unknown", "can_explain_gap": "possible but not quantifiable because original raw snapshot is missing"},
        {"area": "timezone/session semantics", "original_recent": "MT5 epoch converted to naive timestamp", "genuine_oof_control": "naive CSV export timestamps; exact UTC mapping unproven", "materiality": "potentially high", "can_explain_gap": "unresolved; session-hour equivalence cannot be proven"},
        {"area": "label", "original_recent": "clean-window long label, horizon 240, TP 1.8 ATR/min1.0, SL 1.2 ATR/min0.8", "genuine_oof_control": "same BARRIER_TARGET implementation", "materiality": "low", "can_explain_gap": "no verified label-formula difference"},
        {"area": "label maturity", "original_recent": "last 240 rows removed before validation/final fit", "genuine_oof_control": "last 240 pre-fold rows purged and latest label bar verified", "materiality": "low", "can_explain_gap": "no verified maturity difference"},
        {"area": "execution path", "original_recent": "legacy simulator called without HIGH/LOW; close-only threshold exits", "genuine_oof_control": "precomputed M1 HIGH/LOW first-touch, stop-first same-bar", "materiality": "critical", "can_explain_gap": "yes; directly changes TP/SL ordering and realized WR/PF"},
        {"area": "signal episode order", "original_recent": "filters applied before rising-edge state", "genuine_oof_control": "threshold episode formed before session/RSI/spread filters", "materiality": "high", "can_explain_gap": "yes; signals becoming eligible later in a persistent probability run are treated differently"},
        {"area": "live entry semantics", "original_recent": "legacy filtered rising edge", "genuine_oof_control": "pre-filter threshold episode", "materiality": "high", "can_explain_gap": "neither is exact live behavior; live checks every new completed eligible bar when flat"},
        {"area": "risk/cooldown state", "original_recent": "legacy drawdown/loss-streak guards including a 120-tick pause after three losses", "genuine_oof_control": "15-minute cooldown after every loss plus independently reconstructed daily/rolling guards", "materiality": "high", "can_explain_gap": "yes; executable identities and risk-scaled PF are not definitionally identical"},
        {"area": "spread/cost", "original_recent": "fixed 30-point spread plus 5-point extra; no spread gate", "genuine_oof_control": "observed spread when positive, 30-point fallback, 5-point extra, spread gate", "materiality": "medium", "can_explain_gap": "yes for PF/trade count; direction is measurable in retained ledgers"},
        {"area": "metric units", "original_recent": "risk-scaled account-currency PnL and percent DD", "genuine_oof_control": "unscaled net R, PF, Mean-R, and DD-R", "materiality": "high for PnL/DD", "can_explain_gap": "yes for numeric PnL/DD; WR remains comparable only after execution semantics match"},
        {"area": "software environment", "original_recent": "exact package lock not retained", "genuine_oof_control": "Python/package versions retained", "materiality": "unknown", "can_explain_gap": "cannot be excluded, but no evidence it is primary"},
    ]


def stability_summary(window_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scheme in WINDOWS:
        folds = list(window_metrics[scheme]["folds"].values())
        rows.append(
            {
                "scheme": scheme,
                "positive_pf_folds": int(sum(item["pf"] > 1.0 and item["mean_r"] > 0 for item in folds)),
                "worst_fold_pf": float(min(item["pf"] for item in folds)),
                "median_fold_pf": float(np.median([item["pf"] for item in folds])),
                "fold_pf_std": float(np.std([item["pf"] for item in folds])),
                "fold_wr_std": float(np.std([item["realized_wr"] for item in folds])),
                "pooled_pf": window_metrics[scheme]["pooled"]["pf"],
                "pooled_wr": window_metrics[scheme]["pooled"]["realized_wr"],
                "pooled_mean_r": window_metrics[scheme]["pooled"]["mean_r"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda item: (
            item["positive_pf_folds"],
            item["worst_fold_pf"],
            item["median_fold_pf"],
            -item["fold_pf_std"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, 1):
        item["descriptive_stability_rank"] = rank
    return rows


def time_proximity_analysis(
    window_metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for ordinal, (fold_name, start, end) in enumerate(FOLDS, 1):
        metrics = window_metrics["W0_expanding"]["folds"][fold_name]
        rows.append(
            {
                "fold": fold_name,
                "recency_ordinal": ordinal,
                "test_midpoint": (start + (end - start) / 2).isoformat(),
                "trades": metrics["trades"],
                "trades_per_day": metrics["trades_per_day"],
                "realized_wr": metrics["realized_wr"],
                "pf": metrics["pf"],
                "mean_r": metrics["mean_r"],
                "pnl": metrics["pnl"],
                "max_dd": metrics["max_dd"],
            }
        )
    ordinal = np.asarray([row["recency_ordinal"] for row in rows], dtype=np.float64)
    summary = {
        "sample_size_folds": len(rows),
        "warning": "three-fold descriptive correlation only; not inferential evidence",
        "spearman_recency_vs_realized_wr": spearman(
            ordinal, np.asarray([row["realized_wr"] for row in rows])
        ),
        "spearman_recency_vs_pf": spearman(
            ordinal, np.asarray([row["pf"] for row in rows])
        ),
        "spearman_recency_vs_mean_r": spearman(
            ordinal, np.asarray([row["mean_r"] for row in rows])
        ),
        "spearman_recency_vs_trades_per_day": spearman(
            ordinal, np.asarray([row["trades_per_day"] for row in rows])
        ),
    }
    return rows, summary


def calibration_and_ranking_verdicts(
    calibration: list[dict[str, Any]], ranking: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cal = [item for item in calibration if item["scheme"] == "W0_expanding" and item["fold"] != "pooled"]
    enough = all(item["ge_075_observations"] >= 30 for item in cal)
    rates = [item["ge_075_observed_positive_label_rate"] for item in cal if item["ge_075_observed_positive_label_rate"] is not None]
    errors = [
        abs(item["ge_075_predicted_mean"] - item["ge_075_observed_positive_label_rate"])
        for item in cal
        if item["ge_075_predicted_mean"] is not None and item["ge_075_observed_positive_label_rate"] is not None
    ]
    calibration_stable = enough and len(rates) == 3 and max(rates) - min(rates) <= 0.10 and max(errors) - min(errors) <= 0.10
    rank = [item for item in ranking if item["scheme"] == "W0_expanding" and item["fold"] != "pooled"]
    correlations = [item["spearman_probability_vs_realized_r"] for item in rank]
    positive = sum(item is not None and item > 0.0 for item in correlations)
    destructive = any(item is not None and item < -0.02 for item in correlations)
    top_better = sum(item["top_decile_mean_r"] > item["bottom_decile_mean_r"] for item in rank)
    ranking_stable = positive >= 2 and not destructive and top_better >= 2
    return (
        {
            "stable": calibration_stable,
            "criterion": "all folds >=30 P>=0.75 rows; observed-rate and calibration-gap ranges <=10pp",
            "all_folds_enough_rows": enough,
            "observed_rate_range": max(rates) - min(rates) if len(rates) == 3 else None,
            "calibration_gap_range": max(errors) - min(errors) if len(errors) == 3 else None,
        },
        {
            "stable": ranking_stable,
            "criterion": "positive Spearman and top-decile improvement in >=2/3 folds; no Spearman below -0.02",
            "positive_spearman_folds": positive,
            "top_decile_better_folds": top_better,
            "destructive_fold": destructive,
            "fold_spearman": correlations,
        },
    )


def attribution_and_conclusions(
    window_metrics: dict[str, Any],
    stability: list[dict[str, Any]],
    calibration_verdict: dict[str, Any],
    ranking_verdict: dict[str, Any],
    execution_rows: list[dict[str, Any]],
    live: dict[str, Any],
) -> dict[str, Any]:
    w0 = window_metrics["W0_expanding"]
    latest = w0["folds"]["2023_2024"]
    earlier = [w0["folds"]["2018_2020"], w0["folds"]["2021_2022"]]
    later_improvement = latest["pf"] > max(item["pf"] for item in earlier) and latest["realized_wr"] > max(item["realized_wr"] for item in earlier)
    best_stability = min(stability, key=lambda item: item["descriptive_stability_rank"])
    execution = pd.DataFrame(execution_rows)
    close_group = execution[execution["method"].eq("legacy_close_only")]
    highlow_group = execution[execution["method"].eq("legacy_simulator_high_low")]
    execution_wr_delta = float(close_group["realized_wr"].mean() - highlow_group["realized_wr"].mean())
    methodology_material = abs(execution_wr_delta) >= 0.03 or not calibration_verdict["stable"]
    return {
        "why_25_40": (
            "The pooled W0 result is generated by fold-specific expanding-history replicas, not the exact "
            "operational artifact. Its >=0.75 executable cohort is concentrated according to the fold table, "
            "and that cohort loses under observed/fallback costs and HIGH/LOW first-touch. The old 66-70% "
            "figures came from a recent-window, selection-conditioned pipeline with close-only exit detection."
        ),
        "direct_comparability": "partial",
        "exact_operational_artifact_demonstrably_bad": False,
        "exact_artifact_reason": (
            "The exact artifact has a retained 15-trade selected-period result and only a very small live "
            "monitoring sample; W0 historical scores came from different fold replicas."
        ),
        "later_fold_stronger": later_improvement,
        "older_history_degrades_architecture": (
            sum(window_metrics[key]["pooled"]["pf"] > w0["pooled"]["pf"] for key in WINDOWS if key != "W0_expanding") >= 2
        ),
        "most_stable_window_diagnostic": best_stability["scheme"],
        "no_window_is_production_selected": True,
        "calibration_stable": calibration_verdict["stable"],
        "ranking_stable": ranking_verdict["stable"],
        "execution_mean_wr_close_minus_highlow": execution_wr_delta,
        "methodology_difference_material": methodology_material,
        "recent_regime_specialization_evidence": (
            "suggestive but unproven: later historical performance and the selected 2026 evidence are stronger, "
            "but the comparison is confounded by training window, selection, raw data, timezone, and execution semantics"
        ),
        "classification": "methodology mismatch not yet resolved",
        "single_next_research_hypothesis": (
            "EXECUTION-SEMANTICS RECONCILIATION: on one frozen prediction cohort, reproduce the exact live "
            "barwise eligibility order, position re-entry behavior, observed spread, and HIGH/LOW exits, then "
            "compare it with both retained simulators without selecting parameters."
        ),
        "live_monitoring_trades": live.get("closed_trades"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = [sanitize(row) for row in rows]
    if not values:
        raise ValueError(f"No rows for {path.name}")
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Status: **research_only diagnostic**. No candidate was selected and no operational artifact changed.",
        "",
        "## Evidence classification",
        "",
        "All historical intervals and live rows are development/monitoring evidence. The old forward cutoff remains contaminated; no new cutoff was created.",
        "",
        "## Fold-by-fold fixed-control results",
        "",
        "| Scheme | Fold | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["window_metric_rows"]:
        lines.append(
            f"| {row['scheme']} | {row['fold']} | {row['trades']} | {row['trades_per_day']:.4f} | "
            f"{row['realized_wr']:.2%} | {format_metric(row['pf'])} | {row['mean_r']:.4f} | "
            f"{row['pnl']:.2f} | {row['max_dd']:.2f} |"
        )
    lines.extend(
        [
            "",
            "W0 is the byte-preserved Gen core-gate control. W1-W3 change only the historical training-window length.",
            "",
            "## Original recent evidence versus W0 OOF and live monitoring",
            "",
            "| Evidence | Scored interval | Trades | Trades/day | WR | PF | Mean-R | PnL | Max DD | Classification |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in metrics["evidence_rows"]:
        lines.append(
            f"| {row['evidence_id']} | {row['scored_interval']} | {row['executable_trades']} | "
            f"{format_metric(row['trades_per_day'])} | {float(row['realized_wr']):.2%} | "
            f"{format_metric(row['pf'])} | {format_metric(row['mean_r'])} | "
            f"{format_metric(row['pnl'], 2)} | {format_metric(row['max_dd'], 2)} | {row['evidence_classification']} |"
        )
    lines.extend(
        [
            "",
            "Original PnL is risk-scaled account currency and DD is percent; OOF PnL/DD are net R. They are intentionally not treated as the same unit.",
            "",
            "## Verified methodology differences",
            "",
            "| Area | Original recent pipeline | Genuine OOF control | Materiality | Can explain gap |",
            "|---|---|---|---|---|",
        ]
    )
    for row in metrics["methodology_rows"]:
        lines.append(f"| {row['area']} | {row['original_recent']} | {row['genuine_oof_control']} | {row['materiality']} | {row['can_explain_gap']} |")
    lines.extend(
        [
            "",
            "## Probability diagnosis",
            "",
            f"Absolute P(long)=0.75 calibration stable: **{metrics['calibration_verdict']['stable']}**.",
            f"Probability ranking stable: **{metrics['ranking_verdict']['stable']}**.",
            "See probability_distribution.csv, calibration_summary.csv, calibration_buckets.csv, probability_buckets.csv, ranking_summary.csv, and ranking_deciles.csv.",
            "",
            "## Time-proximity diagnosis",
            "",
            f"Three-fold Spearman recency vs WR: {format_metric(metrics['time_proximity_summary']['spearman_recency_vs_realized_wr'])}; "
            f"vs PF: {format_metric(metrics['time_proximity_summary']['spearman_recency_vs_pf'])}; "
            f"vs Mean-R: {format_metric(metrics['time_proximity_summary']['spearman_recency_vs_mean_r'])}; "
            f"vs trades/day: {format_metric(metrics['time_proximity_summary']['spearman_recency_vs_trades_per_day'])}.",
            "These are descriptive correlations across only three folds and are not inferential evidence.",
            "",
            "## Attribution conclusion",
            "",
            metrics["conclusions"]["why_25_40"],
            "",
            f"Direct comparability with the old result: **{metrics['conclusions']['direct_comparability']}**.",
            f"Current core classification: **{metrics['conclusions']['classification']}**.",
            f"Most stable predefined window diagnostically: **{metrics['conclusions']['most_stable_window_diagnostic']}**; this is not a production selection.",
            f"Exact operational artifact demonstrably bad: **{metrics['conclusions']['exact_operational_artifact_demonstrably_bad']}**.",
            f"Later W0 fold stronger than both earlier folds: **{metrics['conclusions']['later_fold_stronger']}**.",
            f"Older expanding history diagnostically degrades the architecture: **{metrics['conclusions']['older_history_degrades_architecture']}**.",
            "",
            "## Required conclusions",
            "",
            f"1. 25.40% cause: {metrics['conclusions']['why_25_40']}",
            f"2. Directly comparable to the old 66-70% result: {metrics['conclusions']['direct_comparability']}.",
            "3. Exact differences are enumerated in methodology_reconciliation.csv and the table above.",
            "4. Material differences are training history, selection conditioning, artifact identity, exit path, signal-episode order, risk/cooldown state, spread/cost handling, metric units, and unresolved timestamp/data identity.",
            f"5. Exact current artifact demonstrably bad: {metrics['conclusions']['exact_operational_artifact_demonstrably_bad']}; {metrics['conclusions']['exact_artifact_reason']}",
            "6. Historical fold replicas are weak outside the recent selected evidence, but architecture robustness is not equivalent to exact-artifact quality.",
            f"7. Performance improves nearer 2025-2026: {metrics['conclusions']['later_fold_stronger']} within comparable W0 folds; the 2026 comparison remains confounded.",
            f"8. Older training history degrades the architecture: {metrics['conclusions']['older_history_degrades_architecture']} under the predefined diagnostic rule.",
            f"9. Diagnostically most stable window: {metrics['conclusions']['most_stable_window_diagnostic']}; no production selection is made.",
            f"10. P(long)=0.75 calibration stable: {metrics['conclusions']['calibration_stable']}.",
            f"11. Probability ranking stable: {metrics['conclusions']['ranking_stable']}.",
            f"12. Classification: {metrics['conclusions']['classification']}.",
            f"13. Single next hypothesis: {metrics['conclusions']['single_next_research_hypothesis']}",
            "",
            "## Single next hypothesis (not implemented)",
            "",
            metrics["conclusions"]["single_next_research_hypothesis"],
            "",
            "## Operational safety",
            "",
            "`gemini.py` and `gold_long_recent_candidate_xgb.json` hashes were checked before and after and remained unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_prediction_artifact(
    path: Path, base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> None:
    payload = {
        "global_index": base["global_index"].to_numpy(dtype=np.int64),
        "time_ns": base["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "fold_code": pd.Categorical(base["fold"], categories=[name for name, _, _ in FOLDS]).codes.astype(np.int8),
        "rsi": base["rsi"].to_numpy(dtype=np.float32),
        "session_ok": base["session_ok"].to_numpy(dtype=np.bool_),
        "spread_points": base["spread_points"].to_numpy(dtype=np.float32),
        "spread_observed": base["spread_observed"].to_numpy(dtype=np.bool_),
        "spread_ok": base["spread_ok"].to_numpy(dtype=np.bool_),
        "outcome": base["outcome"].to_numpy(dtype=np.int8),
        "target": base["target"].to_numpy(dtype=np.bool_),
        "exit_time_ns": base["exit_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "gross_pnl_price": base["gross_pnl_price"].to_numpy(dtype=np.float32),
        "denominator": base["denominator"].to_numpy(dtype=np.float32),
        "reward": base["reward"].to_numpy(dtype=np.float32),
        "stress_reward": base["stress_reward"].to_numpy(dtype=np.float32),
    }
    payload.update({f"score_{scheme}": value.astype(np.float32) for scheme, value in scores.items()})
    np.savez_compressed(path, **payload)


def self_check() -> None:
    assert list(WINDOWS.values()) == [None, 24, 18, 12]
    assert THRESHOLD == 0.75 and RSI_POLICY == "R0"
    assert core.TP_ATR == 1.3 and core.SL_ATR == 1.6
    assert core.MAX_HOLD_MINUTES == 90
    assert core.ALLOWED_HOURS == frozenset((0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23))
    sample = np.array([0.1, 0.2, 0.3])
    assert spearman(sample, sample) == 1.0
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
    required = [MODEL_FILE, GEMINI_FILE, PRIOR_REPORT_JSON, PRIOR_REPORT_MD, PRIOR_PREDICTIONS, PRIOR_MANIFEST, SIGNAL_LOG, TRADE_LOG]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs: " + ", ".join(missing))

    operational_before, manifest = preregister(run_dir)
    self_check()
    prior_dir = run_dir / "prior_evidence"
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_json = snapshot_input(PRIOR_REPORT_JSON, prior_dir / PRIOR_REPORT_JSON.name)
    prior_md = snapshot_input(PRIOR_REPORT_MD, prior_dir / PRIOR_REPORT_MD.name)
    signal_snapshot, trade_snapshot = copy_live_inputs(run_dir)
    live = live_summary(signal_snapshot, trade_snapshot)

    drl_trading_v2.DATA_DIR = str(ROOT)
    history, features = prepare_barrier_data()
    history = history.loc[history["TIME_DT"] < LABEL_BUFFER_END].copy().reset_index(drop=True)
    history = add_targets(history)
    print(
        f"HISTORY rows={len(history):,} {history['TIME_DT'].iat[0]}..{history['TIME_DT'].iat[-1]} features={len(features)}",
        flush=True,
    )
    if len(features) != 31:
        raise RuntimeError("Expected frozen 31-feature architecture")

    base, score_w0, w0_identity = build_base_oof(history)
    scores, provenance = train_window_scores(history, features, base, score_w0)
    window_rows, trade_ledger, window_metrics = metric_rows(base, scores)
    distribution = probability_distribution_rows(base, scores)
    calibration, calibration_buckets = calibration_rows(base, scores)
    buckets = probability_bucket_rows(base, scores)
    ranking, ranking_deciles = ranking_rows(base, scores)
    regime = regime_rows(history, base, features)
    execution = execution_reconciliation_rows(base, score_w0)
    stability = stability_summary(window_metrics)
    proximity_rows, proximity_summary = time_proximity_analysis(window_metrics)
    calibration_verdict, ranking_verdict = calibration_and_ranking_verdicts(calibration, ranking)
    prior = read_json(PRIOR_REPORT_JSON)
    evidence = evidence_rows(prior, window_metrics, provenance, live)
    methods = methodology_rows()
    conclusions = attribution_and_conclusions(
        window_metrics, stability, calibration_verdict, ranking_verdict, execution, live
    )

    table_map = {
        "window_metrics.csv": window_rows,
        "trade_ledger.csv": trade_ledger,
        "evidence_comparison.csv": evidence,
        "methodology_reconciliation.csv": methods,
        "probability_distribution.csv": distribution,
        "probability_buckets.csv": buckets,
        "calibration_summary.csv": calibration,
        "calibration_buckets.csv": calibration_buckets,
        "ranking_summary.csv": ranking,
        "ranking_deciles.csv": ranking_deciles,
        "regime_context.csv": regime,
        "execution_reconciliation.csv": execution,
        "window_stability.csv": stability,
        "time_proximity.csv": proximity_rows,
    }
    for filename, rows in table_map.items():
        write_csv(run_dir / filename, rows)

    prediction_path = run_dir / "diagnostic_predictions.npz"
    save_prediction_artifact(prediction_path, base, scores)
    write_json(
        run_dir / "oof_model_provenance.json",
        {"features": features, "w0_identity": w0_identity, "fold_models": provenance},
    )
    metrics = {
        "experiment": EXPERIMENT,
        "frozen_strategy": manifest["diagnostic_design"],
        "window_metrics": window_metrics,
        "window_metric_rows": window_rows,
        "window_stability": stability,
        "time_proximity_rows": proximity_rows,
        "time_proximity_summary": proximity_summary,
        "evidence_rows": evidence,
        "methodology_rows": methods,
        "probability_distribution": distribution,
        "calibration_summary": calibration,
        "calibration_verdict": calibration_verdict,
        "ranking_summary": ranking,
        "ranking_verdict": ranking_verdict,
        "regime_context": regime,
        "execution_reconciliation": execution,
        "live_monitoring": live,
        "conclusions": conclusions,
    }
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(report_markdown(metrics), encoding="utf-8")

    operational_after = {
        GEMINI_FILE.name: sha256(GEMINI_FILE),
        MODEL_FILE.name: sha256(MODEL_FILE),
    }
    if operational_after != operational_before:
        raise RuntimeError("Operational artifact changed during diagnostic run")

    manifest = read_json(run_dir / "manifest.json")
    manifest["data"].update(
        {
            "symbols": ["GOLD#"],
            "data_sources": [
                "XM MT5 historical CSV exports for 2014-2024 diagnostic folds",
                "preserved Gen core-gate W0 OOF predictions",
                "preserved original recent-window report",
                "immutable snapshots of current live signal/deal logs",
            ],
            "source_files": source_files()
            + [
                {"path": PRIOR_PREDICTIONS.relative_to(ROOT).as_posix(), "sha256": sha256(PRIOR_PREDICTIONS), "retention_status": "git_preserved_parent_run"},
                {"path": prior_json.relative_to(run_dir).as_posix(), "sha256": sha256(prior_json), "retention_status": "stored_in_run_directory_git_archival_pending"},
                {"path": prior_md.relative_to(run_dir).as_posix(), "sha256": sha256(prior_md), "retention_status": "stored_in_run_directory_git_archival_pending"},
                {"path": signal_snapshot.relative_to(run_dir).as_posix(), "sha256": sha256(signal_snapshot), "retention_status": "stored_in_run_directory_git_archival_pending"},
                {"path": trade_snapshot.relative_to(run_dir).as_posix(), "sha256": sha256(trade_snapshot), "retention_status": "stored_in_run_directory_git_archival_pending"},
            ],
            "timezone": "XM export/server timestamps; exact UTC mapping unproven and explicitly audited",
            "data_start_utc": history["TIME_DT"].iat[0].isoformat(),
            "data_end_utc": base["time"].iat[-1].isoformat(),
            "train_start_utc": min(item["train_start"] for item in provenance),
            "train_end_utc": max(item["train_end"] for item in provenance),
            "train_rows": max(item["train_rows"] for item in provenance),
            "validation_start_utc": base["time"].iat[0].isoformat(),
            "validation_end_utc": base["time"].iat[-1].isoformat(),
            "validation_rows": int(len(base)),
            "test_start_utc": "not_applicable_development_diagnostic",
            "test_end_utc": "not_applicable_development_diagnostic",
            "test_rows": 0,
            "purge_details": "Full 240 source rows removed before every fold under every W0-W3 scheme",
            "embargo_details": "latest_training_label_bar < first_scored_bar asserted for all 12 scheme/fold records",
            "raw_snapshot_retained": False,
            "reproducibility_claim": "code_predictions_and_compact_live_inputs_git_preserved_historical_market_csv_hashes_only",
            "mt5_fetch": {
                "used": False,
                "terminal_path": None,
                "terminal_info": None,
                "broker_info": None,
                "fetch_start_utc": None,
                "fetch_end_utc": None,
                "retrieved_at_utc": None,
                "returned_rows": None,
                "not_applicable_reason": "No dynamic MT5 retrieval; local historical exports and log snapshots only",
            },
            "folds": provenance,
        }
    )
    manifest["model"].update(
        {
            "features": features,
            "feature_count": len(features),
            "diagnostic_fold_models_trained": 9,
            "w0_fold_predictions_reused": 3,
        }
    )
    w0 = window_metrics["W0_expanding"]["pooled"]
    manifest["registry"].update(
        {
            "parent_or_incumbent": f"{MODEL_FILE.name}@{operational_before[MODEL_FILE.name]}",
            "selected_configuration": "none_diagnostic_only_fixed_T0_R0_W0_W3",
            "trades_per_day": w0["trades_per_day"],
            "realized_win_rate": w0["realized_wr"],
            "pf": w0["pf"],
            "mean_r": w0["mean_r"],
            "pnl": w0["pnl"],
            "max_dd": w0["max_dd"],
            "validator_result": "PENDING",
        }
    )
    manifest["operational_hashes_after"] = operational_after
    manifest["promotion"]["operational_artifact_changed"] = False
    manifest["result"] = {
        "classification": conclusions["classification"],
        "direct_comparability": conclusions["direct_comparability"],
        "single_next_research_hypothesis": conclusions["single_next_research_hypothesis"],
        "no_candidate_created": True,
    }
    for filename in [*table_map, "diagnostic_predictions.npz", "oof_model_provenance.json", "metrics.json", "report.md"]:
        add_artifact(manifest, run_dir, run_dir / filename, "diagnostic_evidence")
    for path, kind in ((prior_json, "prior_report_snapshot"), (prior_md, "prior_report_snapshot"), (signal_snapshot, "live_monitoring_input_snapshot"), (trade_snapshot, "live_monitoring_input_snapshot")):
        add_artifact(manifest, run_dir, path, kind)
    write_json(run_dir / "manifest.json", manifest)
    print("DIAGNOSTIC_COMPLETE research_only no_candidate operational_artifact_changed=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
