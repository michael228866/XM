from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb

import drl_trading_v2
import gemini as live
import gold_gemini_core_gate_v1 as core
import gold_gemini_incumbent_robustness_v1 as prior_diagnostic
from barrier_classifier_strategy import evaluate as legacy_evaluate
from barrier_final_train import FINAL_PARAMS, prepare_barrier_data
from barrier_research_suite import make_direction_probs, predict_positive
from drl_trading_v2 import BASE_FEATURES, add_indicators
from gold_recent_walk_forward import (
    DEFAULT_START,
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    copy_rates,
    timeframe_warmup_start,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "GEMINI EXECUTION SEMANTICS RECONCILIATION V1"
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
GEMINI_FILE = ROOT / "gemini.py"
OLD_REPORT_FILE = ROOT / "gold_long_recent_walk_forward.json"
SIGNAL_LOG_FILE = ROOT / "gemini_signal_log.csv"
PRIOR_RUN = ROOT / "training_runs" / "20260903T005405Z_gemini_incumbent_robustness_v1"
PRIOR_PREDICTIONS = PRIOR_RUN / "diagnostic_predictions.npz"
PRIOR_REPORT = PRIOR_RUN / "report.md"

PRIMARY_START_API = pd.Timestamp("2026-06-01 00:00:00")
PRIMARY_END_API = pd.Timestamp("2026-08-25 05:49:00")
PRIMARY_EXPECTED_ROWS = 83_911
SECONDARY_START_API = pd.Timestamp("2018-01-02 00:00:00")
SECONDARY_END_API = pd.Timestamp("2020-12-31 18:50:00")
PREVIOUS_FORWARD_CUTOFF = "2026-09-01T02:00:00Z"
PREVIOUS_FORWARD_STATUS = "contaminated_for_future_gate_selection"

THRESHOLD = 0.75
MIN_ENTRY_RSI = 22.0
EXCLUDED_RSI = (35.0, 45.0)
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
MAX_HOLD_MINUTES = 90
LEGACY_SPREAD_POINTS = 30.0
EXTRA_COST_POINTS = 5.0
STRESS_EXTRA_COST_POINTS = 10.0
POINT = 0.01
RISK_PER_TRADE = 0.014
INITIAL_BALANCE = 1000.0
ALLOWED_HOURS = frozenset(live.ALLOWED_ENTRY_HOURS)
ALLOWED_WEEKDAYS = frozenset(live.ALLOWED_ENTRY_WEEKDAYS)


@dataclass(frozen=True)
class SimulatorDefinition:
    simulator: str
    exit_mode: str
    entry_policy: str
    cost_mode: str
    spread_gate: bool
    risk_mode: str
    entry_price_mode: str
    entry_bar_exits: bool
    timeout_mode: str


SIMULATORS = (
    SimulatorDefinition("S0", "close_only", "legacy_filtered_rising", "fixed", False, "legacy", "close", False, "bar_count"),
    SimulatorDefinition("S1", "high_low_stop_first", "legacy_filtered_rising", "fixed", False, "legacy", "close", False, "bar_count"),
    SimulatorDefinition("S2", "high_low_stop_first", "live_barwise", "fixed", False, "legacy", "close", False, "bar_count"),
    SimulatorDefinition("S3", "high_low_stop_first", "live_barwise", "observed_fallback", True, "legacy", "close", False, "bar_count"),
    SimulatorDefinition("S4", "high_low_stop_first", "live_barwise", "observed_fallback", True, "live", "close", False, "bar_count"),
    SimulatorDefinition("S5", "high_low_stop_first", "live_barwise", "observed_fallback", True, "live", "open", True, "wall_clock"),
)

TRANSITION_ALLOWED_CHANGES = {
    "S0->S1": ["exit_mode"],
    "S1->S2": ["entry_policy"],
    "S2->S3": ["cost_mode", "spread_gate"],
    "S3->S4": ["risk_mode"],
    "S4->S5": ["entry_price_mode", "entry_bar_exits", "timeout_mode"],
}


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
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: sanitize(value) for key, value in row.items()} for row in rows])


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def add_artifact(manifest: dict[str, Any], run_dir: Path, path: Path, kind: str) -> None:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    existing = {item.get("path") for item in manifest.get("artifacts", [])}
    if relative in existing:
        return
    manifest.setdefault("artifacts", []).append(
        {
            "kind": kind,
            "path": relative,
            "sha256": sha256(path),
            "retention_status": "stored_in_run_directory_git_archival_pending",
        }
    )


def snapshot_gzip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, compresslevel=6, mtime=0
        ) as writer:
            shutil.copyfileobj(reader, writer)


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
    upstream = git("rev-parse", tracking_ref)
    if head != upstream or head != manifest.get("git_commit"):
        raise RuntimeError("Pre-run HEAD, upstream, and manifest commit do not match")

    operational = {
        GEMINI_FILE.name: sha256(GEMINI_FILE),
        MODEL_FILE.name: sha256(MODEL_FILE),
    }
    definitions = {definition.simulator: definition.__dict__ for definition in SIMULATORS}
    manifest["pre_run_git"] = {
        "pre_run_git_commit": manifest["git_commit"],
        "pre_run_git_dirty": False,
        "remote_branch": tracking_ref,
        "head_sha": head,
        "origin_main_sha": upstream,
        "head_equals_origin_main": True,
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
        "candidate_selection": 0,
        "direction": "LONG only",
        "threshold": THRESHOLD,
        "minimum_entry_rsi": MIN_ENTRY_RSI,
        "excluded_rsi_range": list(EXCLUDED_RSI),
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold_minutes": MAX_HOLD_MINUTES,
        "max_open_positions": 1,
        "allowed_hours": sorted(ALLOWED_HOURS),
        "allowed_weekdays": sorted(ALLOWED_WEEKDAYS),
        "simulator_definitions": definitions,
        "transition_allowed_changes": TRANSITION_ALLOWED_CHANGES,
        "primary_cohort": f"{PRIMARY_START_API.isoformat()}..{PRIMARY_END_API.isoformat()} broker/API labels",
        "secondary_cohort": f"{SECONDARY_START_API.isoformat()}..{SECONDARY_END_API.isoformat()} prior W0",
    }
    manifest["search"].update(
        {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": None,
            "not_applicable_reason": "Six simulator definitions are fixed semantic transitions, not strategy candidates.",
        }
    )
    manifest["model"].update(
        {
            "trained": False,
            "model_type": "existing XGBoost long-binary operational artifact; no fitting",
            "parameters": {},
            "boosted_rounds_or_estimators": None,
            "features": list(live.FEATURE_COLUMNS),
            "feature_count": len(live.FEATURE_COLUMNS),
            "label_definition": "not applicable; no labels or fitting in this execution diagnostic",
            "horizon": None,
            "label_tp_sl_semantics": "not applicable; no model fitting",
            "execution_tp_sl_semantics": "S0 close-only; S1-S5 HIGH/LOW stop-first; exact definitions retained in manifest",
            "calibration_method": "none",
            "artifact_path": MODEL_FILE.name,
            "artifact_sha256": operational[MODEL_FILE.name],
            "retention_status": "operational artifact snapshotted by training_run_history.py create",
            "not_applicable_reason": "Operational artifact is loaded read-only and never retrained or saved.",
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
        name: sha256(ROOT / name)
        for name in (
            "gemini.py",
            "gold_long_recent_walk_forward.py",
            "gold_recent_walk_forward.py",
            "barrier_classifier_strategy.py",
            "barrier_final_train.py",
            "gold_gemini_incumbent_robustness_v1.py",
        )
    }
    write_json(manifest_path, manifest)
    print(f"PREREGISTERED commit={head} upstream={tracking_ref}", flush=True)
    return operational, manifest


def infer_broker_offset_hours(signal_log: Path) -> dict[str, Any]:
    deltas: list[float] = []
    by_date: dict[str, list[float]] = {}
    if signal_log.is_file():
        with signal_log.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    event = pd.Timestamp(row.get("event_time"))
                    bar = pd.Timestamp(row.get("bar_time"))
                except (TypeError, ValueError):
                    continue
                if event is pd.NaT or bar is pd.NaT or event.tzinfo is None or bar.tzinfo is None:
                    continue
                event = event.tz_convert("UTC")
                bar = bar.tz_convert("UTC")
                if not (pd.Timestamp("2026-06-01T00:00:00Z") <= event <= pd.Timestamp("2026-08-25T23:59:59Z")):
                    continue
                delta = (bar - event).total_seconds() / 60.0
                if 120.0 <= delta <= 240.0:
                    deltas.append(delta)
                    by_date.setdefault(event.date().isoformat(), []).append(delta)
    if not deltas:
        raise RuntimeError("No contemporaneous live-log rows available to infer broker timestamp offset")
    median_delta = float(np.median(deltas))
    offset = int(round((median_delta + 1.0) / 60.0))
    return {
        "rows": len(deltas),
        "first_date": min(by_date),
        "last_date": max(by_date),
        "median_bar_minus_event_minutes": median_delta,
        "p01_minutes": float(np.quantile(deltas, 0.01)),
        "p99_minutes": float(np.quantile(deltas, 0.99)),
        "completed_bar_adjustment_minutes": 1,
        "inferred_broker_offset_hours": offset,
        "basis": "bar_time minus UTC event_time plus one completed-bar minute; contemporaneous gemini_signal_log rows",
    }


def build_primary_features() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    m1 = copy_rates("M1", DEFAULT_START - timedelta(days=7), PRIMARY_END_API.to_pydatetime().replace(tzinfo=timezone.utc))
    raw_m1_rows = len(m1)
    m1 = add_indicators(m1)
    diff = m1["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    m1["M1_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
    m1["VOLA_MA"] = m1["ATR"].rolling(240).mean()
    m1["VOLA_RATIO"] = m1["ATR"] / (m1["VOLA_MA"] + 1e-6)
    frame = m1.sort_values("TIME_DT").copy()
    frame["FEATURE_BAR_TIME"] = frame["TIME_DT"].shift(1)
    mtf_rows: dict[str, int] = {}
    features: list[str] = list(BASE_FEATURES)
    for timeframe in live.MTF_ORDER:
        higher = copy_rates(
            timeframe,
            timeframe_warmup_start(DEFAULT_START, timeframe),
            PRIMARY_END_API.to_pydatetime().replace(tzinfo=timezone.utc),
        )
        mtf_rows[timeframe] = len(higher)
        higher = add_indicators(higher)
        trend_col = f"{timeframe}_TREND"
        higher[trend_col] = np.where(
            higher["CLOSE"] > higher["CLOSE"].rolling(20).mean(), 1, -1
        )
        higher[trend_col] = higher[trend_col].shift(1)
        frame = pd.merge_asof(
            frame.sort_values("TIME_DT"),
            higher[["TIME_DT", trend_col]].sort_values("TIME_DT"),
            on="TIME_DT",
            direction="backward",
        )
        features.append(trend_col)
    frame[features] = frame[features].shift(1)
    frame = frame[frame["TIME_DT"] >= DEFAULT_START.replace(tzinfo=None)]
    frame = frame.dropna(subset=features + ["ATR"])
    primary = frame[
        (frame["TIME_DT"] >= PRIMARY_START_API) & (frame["TIME_DT"] <= PRIMARY_END_API)
    ].copy().reset_index(drop=True)
    if primary.empty:
        raise RuntimeError("Primary cohort is empty")
    if len(features) != 31:
        raise RuntimeError(f"Expected 31 features, got {len(features)}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    model.set_params(device="cpu")
    primary["buy_prob"] = predict_positive(model, primary, features).astype(np.float32)
    primary["sell_prob"] = np.float32(0.0)
    primary["all_features_finite"] = np.isfinite(primary[features].to_numpy(dtype=np.float64)).all(axis=1)
    if not primary["all_features_finite"].all():
        raise RuntimeError("Non-finite primary feature entered frozen cohort")
    identity = {
        "raw_m1_rows_with_warmup": raw_m1_rows,
        "mtf_rows": mtf_rows,
        "scored_rows": len(primary),
        "expected_scored_rows": PRIMARY_EXPECTED_ROWS,
        "row_count_matches_old_report": len(primary) == PRIMARY_EXPECTED_ROWS,
        "first_api_timestamp": primary["TIME_DT"].iat[0],
        "last_api_timestamp": primary["TIME_DT"].iat[-1],
        "feature_count": len(features),
        "probability_source": "one read-only predict_proba call with exact operational artifact",
    }
    return primary, features, identity


def effective_spread(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    observed = np.isfinite(raw) & (raw > 0.0)
    return np.where(observed, raw, LEGACY_SPREAD_POINTS), observed


def finalize_cohort(frame: pd.DataFrame, cohort: str, offset_hours: int) -> pd.DataFrame:
    result = frame.copy()
    if "decision_time_api" not in result:
        result["decision_time_api"] = pd.to_datetime(result["TIME_DT"])
    result["decision_time_api"] = pd.to_datetime(result["decision_time_api"])
    result["decision_time_actual_utc"] = result["decision_time_api"] - pd.Timedelta(hours=offset_hours)
    result["cohort"] = cohort
    spread, observed = effective_spread(result["SPREAD"])
    result["effective_spread_points"] = spread
    result["spread_observed"] = observed
    result["session_api"] = (
        result["decision_time_api"].dt.hour.isin(ALLOWED_HOURS)
        & result["decision_time_api"].dt.dayofweek.isin(ALLOWED_WEEKDAYS)
    )
    result["session_actual_utc"] = (
        result["decision_time_actual_utc"].dt.hour.isin(ALLOWED_HOURS)
        & result["decision_time_actual_utc"].dt.dayofweek.isin(ALLOWED_WEEKDAYS)
    )
    raw = result["buy_prob"].to_numpy(dtype=np.float64) >= THRESHOLD
    times = result["decision_time_api"].to_numpy(dtype="datetime64[ns]")
    gap = np.ones(len(result), dtype=bool)
    if len(result) > 1:
        gap[1:] = np.diff(times).astype("timedelta64[s]").astype(np.int64) > 120
    new_episode = raw & (np.r_[False, ~raw[:-1]] | gap)
    episode_id = np.cumsum(new_episode).astype(np.int64) - 1
    episode_id[~raw] = -1
    result["raw_signal"] = raw
    result["raw_episode_id"] = episode_id
    return result.reset_index(drop=True)


def save_primary_cohort(path: Path, cohort: pd.DataFrame, features: list[str]) -> None:
    columns = list(dict.fromkeys([
        "time", "TIME_DT", "FEATURE_BAR_TIME", "decision_time_api", "decision_time_actual_utc",
        "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD",
        "effective_spread_points", "spread_observed", "session_api", "session_actual_utc",
        "buy_prob", "sell_prob", "raw_signal", "raw_episode_id", "all_features_finite",
        *features,
    ]))
    cohort[columns].to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def load_secondary_cohort(offset_hours: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    drl_trading_v2.DATA_DIR = str(ROOT)
    history, features = prepare_barrier_data()
    history = history.loc[history["TIME_DT"] < prior_diagnostic.LABEL_BUFFER_END].copy().reset_index(drop=True)
    with np.load(PRIOR_PREDICTIONS, allow_pickle=False) as prior:
        indices = prior["global_index"].astype(np.int64)
        times_ns = prior["time_ns"].astype(np.int64)
        fold_code = prior["fold_code"].astype(np.int8)
        scores = prior["score_W0_expanding"].astype(np.float32)
    mask = fold_code == 0
    indices = indices[mask]
    expected = history["TIME_DT"].to_numpy(dtype="datetime64[ns]")[indices].astype(np.int64)
    if not np.array_equal(expected, times_ns[mask]):
        raise RuntimeError("Secondary W0 rows no longer align with local history")
    source = history.iloc[indices]
    secondary = pd.DataFrame(
        {
            "time": (times_ns[mask] // 1_000_000_000).astype(np.int64),
            "TIME_DT": pd.to_datetime(times_ns[mask]),
            "FEATURE_BAR_TIME": pd.to_datetime(times_ns[mask]) - pd.Timedelta(minutes=1),
            "OPEN": source["OPEN"].to_numpy(dtype=np.float64),
            "HIGH": source["HIGH"].to_numpy(dtype=np.float64),
            "LOW": source["LOW"].to_numpy(dtype=np.float64),
            "CLOSE": source["CLOSE"].to_numpy(dtype=np.float64),
            "ATR": source["ATR"].to_numpy(dtype=np.float64),
            "M1_RSI": source["M1_RSI"].to_numpy(dtype=np.float64),
            "SPREAD": source["SPREAD"].to_numpy(dtype=np.float64),
            "buy_prob": scores[mask],
            "sell_prob": np.float32(0.0),
            "all_features_finite": True,
        }
    )
    secondary = finalize_cohort(secondary, "SECONDARY_W0_2018_2020", offset_hours=0)
    identity = {
        "source_run": PRIOR_RUN.name,
        "source_predictions": PRIOR_PREDICTIONS.relative_to(ROOT).as_posix(),
        "source_predictions_sha256": sha256(PRIOR_PREDICTIONS),
        "source_report_sha256": sha256(PRIOR_REPORT),
        "rows": len(secondary),
        "first_timestamp": secondary["decision_time_api"].iat[0],
        "last_timestamp": secondary["decision_time_api"].iat[-1],
        "timezone_status": "historical CSV timestamp timezone not independently recoverable; recorded hour retained",
        "features_in_source_architecture": len(features),
    }
    return secondary, identity


def save_secondary_cohort(path: Path, cohort: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=cohort["decision_time_api"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        open=cohort["OPEN"].to_numpy(dtype=np.float64),
        high=cohort["HIGH"].to_numpy(dtype=np.float64),
        low=cohort["LOW"].to_numpy(dtype=np.float64),
        close=cohort["CLOSE"].to_numpy(dtype=np.float64),
        atr=cohort["ATR"].to_numpy(dtype=np.float64),
        rsi=cohort["M1_RSI"].to_numpy(dtype=np.float64),
        spread=cohort["SPREAD"].to_numpy(dtype=np.float64),
        buy_prob=cohort["buy_prob"].to_numpy(dtype=np.float32),
        raw_episode_id=cohort["raw_episode_id"].to_numpy(dtype=np.int64),
    )


def spread_limit_points(tp_distance: float) -> float:
    if tp_distance <= 0:
        return float(live.MAX_SPREAD_POINTS)
    ratio_limit = (tp_distance * live.MAX_SPREAD_TO_TP_RATIO) / live.PRICE_PER_POINT
    return float(min(live.HARD_MAX_SPREAD_POINTS, max(live.MAX_SPREAD_POINTS, ratio_limit)))


def reward_metrics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    wins = values[values > 0]
    losses = values[values <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    payoff = None if avg_win is None or avg_loss is None or avg_loss == 0 else avg_win / abs(avg_loss)
    break_even = None if payoff is None else 1.0 / (1.0 + payoff)
    wr = float((values > 0).mean()) if len(values) else 0.0
    equity = np.r_[0.0, np.cumsum(values)]
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "trades": len(values),
        "wins": len(wins),
        "losses": len(losses),
        "realized_wr": wr,
        "average_winner_r": avg_win,
        "average_loser_r": avg_loss,
        "payoff_ratio": payoff,
        "break_even_wr": break_even,
        "break_even_adjusted_edge": None if break_even is None else wr - break_even,
        "pf": pf,
        "mean_r": float(values.mean()) if len(values) else 0.0,
        "pnl_r": float(values.sum()),
        "max_dd_r": float(drawdown.min()),
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
    }


def calculate_risk_multiplier(state: dict[str, Any], risk_mode: str, now_actual: pd.Timestamp) -> tuple[float, list[str]]:
    reasons: list[str] = []
    balance = state["balance"]
    if risk_mode == "legacy":
        state["peak_balance"] = max(state["peak_balance"], balance)
        risk_mult = 1.0
        drawdown = max(0.0, 1.0 - balance / max(state["peak_balance"], 1e-9))
        if drawdown >= FINAL_PARAMS["drawdown_guard_start_pct"]:
            span = FINAL_PARAMS["drawdown_guard_full_pct"] - FINAL_PARAMS["drawdown_guard_start_pct"]
            progress = min((drawdown - FINAL_PARAMS["drawdown_guard_start_pct"]) / span, 1.0)
            risk_mult *= 1.0 - ((1.0 - FINAL_PARAMS["drawdown_guard_min_risk_mult"]) * progress)
            reasons.append("legacy_drawdown")
        if state["loss_streak"] >= FINAL_PARAMS["loss_streak_threshold"]:
            risk_mult *= FINAL_PARAMS["loss_streak_risk_mult"]
            reasons.append("legacy_loss_streak")
    else:
        risk_mult = 1.0
        last_loss = state.get("last_loss_actual")
        if last_loss is not None and (now_actual - last_loss).total_seconds() < live.POST_LOSS_RISK_CAP_MINUTES * 60:
            risk_mult = min(risk_mult, live.POST_LOSS_MAX_RISK_MULT)
            reasons.append("live_post_loss_cap")
    recent = np.asarray(state["account_rewards"][-live.ROLLING_GUARD_WINDOW :], dtype=np.float64)
    if len(recent) >= live.ROLLING_GUARD_MIN_TRADES:
        gp = float(recent[recent > 0].sum())
        gl = float(-recent[recent < 0].sum())
        pf = gp / gl if gl > 0 else math.inf
        if pf < live.ROLLING_GUARD_MIN_PROFIT_FACTOR:
            risk_mult *= live.ROLLING_GUARD_RISK_MULT if risk_mode == "live" else FINAL_PARAMS["rolling_guard_risk_mult"]
            reasons.append(f"{risk_mode}_rolling_pf")
    return risk_mult, reasons


def base_row_eligibility(row: pd.Series, definition: SimulatorDefinition, use_actual_utc_session: bool = False) -> tuple[bool, str]:
    if not bool(row["raw_signal"]):
        return False, "below_threshold"
    session = bool(row["session_actual_utc"] if use_actual_utc_session else row["session_api"])
    if not session:
        return False, "session"
    rsi = float(row["M1_RSI"])
    if definition.entry_policy == "legacy_filtered_rising":
        if 0.0 <= rsi <= MIN_ENTRY_RSI:
            return False, "rsi_floor"
    elif rsi < MIN_ENTRY_RSI:
        return False, "rsi_floor"
    if EXCLUDED_RSI[0] <= rsi <= EXCLUDED_RSI[1]:
        return False, "rsi_range"
    if definition.spread_gate:
        tp_distance = max(float(row["ATR"]) * TP_ATR, MIN_TP_PRICE)
        if float(row["effective_spread_points"]) > spread_limit_points(tp_distance):
            return False, "spread"
    return True, "eligible"


def start_state() -> dict[str, Any]:
    return {
        "balance": INITIAL_BALANCE,
        "peak_balance": INITIAL_BALANCE,
        "loss_streak": 0,
        "trade_pause": 0,
        "daily_locked": False,
        "current_api_date": None,
        "day_start_balance": INITIAL_BALANCE,
        "account_rewards": [],
        "last_loss_actual": None,
        "position": None,
        "prev_signal": 0,
    }


def close_trade(
    state: dict[str, Any],
    definition: SimulatorDefinition,
    cohort: pd.DataFrame,
    exit_index: int,
    exit_price: float,
    exit_reason: str,
    both_hit: bool,
) -> dict[str, Any]:
    position = state["position"]
    gross_price = exit_price - position["entry_price"]
    spread_points = position["spread_points"]
    denominator = position["sl_distance"] + spread_points * POINT
    net_r = (gross_price - (spread_points + EXTRA_COST_POINTS) * POINT) / denominator
    stress_r = (gross_price - (spread_points + STRESS_EXTRA_COST_POINTS) * POINT) / denominator
    account_pnl = net_r * position["risk_budget"]
    before = state["balance"]
    state["balance"] += account_pnl
    state["peak_balance"] = max(state["peak_balance"], state["balance"])
    state["account_rewards"].append(account_pnl)
    is_win = net_r > 0
    state["loss_streak"] = 0 if is_win else state["loss_streak"] + 1
    exit_actual = pd.Timestamp(cohort["decision_time_actual_utc"].iat[exit_index])
    if not is_win:
        state["last_loss_actual"] = exit_actual
    if definition.risk_mode == "legacy" and not is_win and state["loss_streak"] >= FINAL_PARAMS["loss_streak_pause_threshold"]:
        state["trade_pause"] = max(state["trade_pause"], FINAL_PARAMS["loss_streak_pause_ticks"])
    trade = {
        "cohort": str(cohort["cohort"].iat[0]),
        "simulator": definition.simulator,
        "entry_index": position["entry_index"],
        "exit_index": exit_index,
        "entry_time_api": pd.Timestamp(cohort["decision_time_api"].iat[position["entry_index"]]).isoformat(),
        "exit_time_api": pd.Timestamp(cohort["decision_time_api"].iat[exit_index]).isoformat(),
        "entry_time_actual_utc": pd.Timestamp(cohort["decision_time_actual_utc"].iat[position["entry_index"]]).isoformat(),
        "exit_time_actual_utc": exit_actual.isoformat(),
        "raw_episode_id": int(position["raw_episode_id"]),
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "tp_distance": position["tp_distance"],
        "sl_distance": position["sl_distance"],
        "exit_reason": exit_reason,
        "same_bar_both_hit": both_hit,
        "gross_price": gross_price,
        "spread_points": spread_points,
        "spread_observed": position["spread_observed"],
        "extra_cost_points": EXTRA_COST_POINTS,
        "gross_r": gross_price / denominator,
        "net_r": net_r,
        "stress_r": stress_r,
        "risk_mult": position["risk_mult"],
        "risk_guard_reasons": ";".join(position["risk_guard_reasons"]),
        "risk_budget": position["risk_budget"],
        "account_pnl": account_pnl,
        "balance_before": before,
        "balance_after": state["balance"],
        "positive": is_win,
    }
    state["position"] = None
    return trade


def open_trade(
    state: dict[str, Any], definition: SimulatorDefinition, cohort: pd.DataFrame, index: int
) -> None:
    row = cohort.iloc[index]
    spread_points = LEGACY_SPREAD_POINTS if definition.cost_mode == "fixed" else float(row["effective_spread_points"])
    risk_mult, reasons = calculate_risk_multiplier(
        state, definition.risk_mode, pd.Timestamp(row["decision_time_actual_utc"])
    )
    sl_distance = max(float(row["ATR"]) * SL_ATR, MIN_SL_PRICE)
    tp_distance = max(float(row["ATR"]) * TP_ATR, MIN_TP_PRICE)
    risk_budget = max(state["balance"], 0.0) * RISK_PER_TRADE * risk_mult
    state["position"] = {
        "entry_index": index,
        "entry_price": float(row["OPEN"] if definition.entry_price_mode == "open" else row["CLOSE"]),
        "entry_actual": pd.Timestamp(row["decision_time_actual_utc"]),
        "tp_distance": tp_distance,
        "sl_distance": sl_distance,
        "spread_points": spread_points,
        "spread_observed": bool(row["spread_observed"]) if definition.cost_mode != "fixed" else False,
        "risk_mult": risk_mult,
        "risk_guard_reasons": reasons,
        "risk_budget": risk_budget,
        "raw_episode_id": int(row["raw_episode_id"]),
        "hold_bars": 0,
    }


def live_daily_guard_active(state: dict[str, Any], now_actual: pd.Timestamp, ledger: list[dict[str, Any]]) -> bool:
    local_date = (now_actual.tz_localize("UTC") + pd.Timedelta(hours=8)).date() if now_actual.tzinfo is None else now_actual.tz_convert("Asia/Taipei").date()
    realized = sum(
        trade["account_pnl"]
        for trade in ledger
        if (pd.Timestamp(trade["exit_time_actual_utc"]).tz_localize("UTC") + pd.Timedelta(hours=8)).date() == local_date
    )
    return realized <= -state["balance"] * live.MAX_DAILY_LOSS_PCT


def simulate(
    cohort: pd.DataFrame,
    definition: SimulatorDefinition,
    use_actual_utc_session: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = start_state()
    ledger: list[dict[str, Any]] = []
    audit = {
        "raw_qualifying_rows": int(cohort["raw_signal"].sum()),
        "independent_episodes": int((cohort["raw_episode_id"].diff().fillna(1) > 0).sum()),
        "session_blocked_rows": 0,
        "rsi_floor_blocked_rows": 0,
        "rsi_range_blocked_rows": 0,
        "spread_blocked_rows": 0,
        "position_blocked_rows": 0,
        "risk_cooldown_blocked_rows": 0,
        "daily_loss_blocked_rows": 0,
        "legacy_pause_blocked_rows": 0,
        "eligible_flat_rows": 0,
        "entries": 0,
        "later_entries_in_episode": 0,
        "entries_after_prior_close_same_episode": 0,
    }
    entries_by_episode: dict[int, int] = {}

    for index in range(len(cohort)):
        row = cohort.iloc[index]
        api_time = pd.Timestamp(row["decision_time_api"])
        actual_time = pd.Timestamp(row["decision_time_actual_utc"])
        api_date = api_time.date()
        if state["current_api_date"] is None or api_date != state["current_api_date"]:
            state["current_api_date"] = api_date
            state["day_start_balance"] = state["balance"]
            state["daily_locked"] = False

        if definition.simulator == "S5" and state["position"] is not None:
            elapsed = (actual_time - state["position"]["entry_actual"]).total_seconds() / 60.0
            if elapsed >= MAX_HOLD_MINUTES:
                ledger.append(close_trade(state, definition, cohort, index, float(row["OPEN"]), "timeout", False))

        eligible, reason = base_row_eligibility(row, definition, use_actual_utc_session)
        if bool(row["raw_signal"]) and not eligible:
            key = {
                "session": "session_blocked_rows",
                "rsi_floor": "rsi_floor_blocked_rows",
                "rsi_range": "rsi_range_blocked_rows",
                "spread": "spread_blocked_rows",
            }.get(reason)
            if key:
                audit[key] += 1

        if eligible and definition.risk_mode == "legacy":
            if state["daily_locked"]:
                eligible = False
                audit["daily_loss_blocked_rows"] += 1
            elif state["trade_pause"] > 0:
                eligible = False
                audit["legacy_pause_blocked_rows"] += 1
        elif eligible and definition.risk_mode == "live":
            last_loss = state["last_loss_actual"]
            if last_loss is not None and (actual_time - last_loss).total_seconds() < live.LOSS_COOLDOWN_MINUTES * 60:
                eligible = False
                audit["risk_cooldown_blocked_rows"] += 1
            elif live_daily_guard_active(state, actual_time, ledger):
                eligible = False
                audit["daily_loss_blocked_rows"] += 1

        if definition.simulator == "S5":
            if state["position"] is None:
                if eligible:
                    audit["eligible_flat_rows"] += 1
                    open_trade(state, definition, cohort, index)
                    episode = int(row["raw_episode_id"])
                    previous = entries_by_episode.get(episode, 0)
                    if previous:
                        audit["entries_after_prior_close_same_episode"] += 1
                    if index > 0 and bool(cohort["raw_signal"].iat[index - 1]):
                        audit["later_entries_in_episode"] += 1
                    entries_by_episode[episode] = previous + 1
                    audit["entries"] += 1
            elif eligible:
                audit["position_blocked_rows"] += 1
            if state["position"] is not None:
                position = state["position"]
                take_hit = float(row["HIGH"]) >= position["entry_price"] + position["tp_distance"]
                stop_hit = float(row["LOW"]) <= position["entry_price"] - position["sl_distance"]
                if stop_hit or take_hit:
                    reason = "stop_loss" if stop_hit else "take_profit"
                    price = position["entry_price"] - position["sl_distance"] if stop_hit else position["entry_price"] + position["tp_distance"]
                    ledger.append(close_trade(state, definition, cohort, index, price, reason, bool(stop_hit and take_hit)))
            continue

        has_signal = eligible
        if state["position"] is None:
            can_enter = has_signal
            if definition.entry_policy == "legacy_filtered_rising":
                can_enter = can_enter and state["prev_signal"] != 1
            if can_enter:
                audit["eligible_flat_rows"] += 1
                open_trade(state, definition, cohort, index)
                episode = int(row["raw_episode_id"])
                previous = entries_by_episode.get(episode, 0)
                if previous:
                    audit["entries_after_prior_close_same_episode"] += 1
                if index > 0 and bool(cohort["raw_signal"].iat[index - 1]):
                    audit["later_entries_in_episode"] += 1
                entries_by_episode[episode] = previous + 1
                audit["entries"] += 1
        else:
            if has_signal:
                audit["position_blocked_rows"] += 1
            position = state["position"]
            position["hold_bars"] += 1
            exit_reason = None
            exit_price = float(row["CLOSE"])
            both_hit = False
            if definition.exit_mode == "high_low_stop_first":
                take_hit = float(row["HIGH"]) >= position["entry_price"] + position["tp_distance"]
                stop_hit = float(row["LOW"]) <= position["entry_price"] - position["sl_distance"]
                both_hit = bool(take_hit and stop_hit)
                if stop_hit:
                    exit_reason = "stop_loss"
                    exit_price = position["entry_price"] - position["sl_distance"]
                elif take_hit:
                    exit_reason = "take_profit"
                    exit_price = position["entry_price"] + position["tp_distance"]
            else:
                pnl = float(row["CLOSE"]) - position["entry_price"]
                if pnl >= position["tp_distance"]:
                    exit_reason = "take_profit"
                elif pnl <= -position["sl_distance"]:
                    exit_reason = "stop_loss"
            if exit_reason is None and position["hold_bars"] >= MAX_HOLD_MINUTES:
                exit_reason = "timeout"
            if exit_reason is not None:
                ledger.append(close_trade(state, definition, cohort, index, exit_price, exit_reason, both_hit))
                if definition.risk_mode == "legacy" and state["balance"] <= state["day_start_balance"] * (1.0 - FINAL_PARAMS["max_daily_loss_pct"]):
                    state["daily_locked"] = True

        if state["position"] is None and state["trade_pause"] > 0:
            state["trade_pause"] -= 1
        state["prev_signal"] = 1 if has_signal else 0

    if state["position"] is not None:
        ledger.append(
            close_trade(
                state,
                definition,
                cohort,
                len(cohort) - 1,
                float(cohort["CLOSE"].iat[-1]),
                "cohort_end",
                False,
            )
        )
    ordinal: dict[int, int] = {}
    for trade_id, trade in enumerate(ledger):
        episode = trade["raw_episode_id"]
        trade["trade_id"] = f"{definition.simulator}_{trade_id:06d}"
        trade["episode_trade_ordinal"] = ordinal.get(episode, 0)
        ordinal[episode] = trade["episode_trade_ordinal"] + 1
    return ledger, audit


def simulator_metrics(
    cohort: pd.DataFrame,
    definition: SimulatorDefinition,
    ledger: list[dict[str, Any]],
    audit: dict[str, Any],
    cohort_hash: str,
) -> dict[str, Any]:
    unit = reward_metrics(np.asarray([trade["net_r"] for trade in ledger], dtype=np.float64))
    account = reward_metrics(np.asarray([trade["account_pnl"] for trade in ledger], dtype=np.float64))
    days = max((cohort["decision_time_api"].iat[-1] - cohort["decision_time_api"].iat[0]).total_seconds() / 86400.0, 1.0)
    return {
        "cohort": str(cohort["cohort"].iat[0]),
        "cohort_sha256": cohort_hash,
        "simulator": definition.simulator,
        **definition.__dict__,
        **audit,
        **unit,
        "trades_per_day": unit["trades"] / days,
        "tp_first_wr": sum(trade["exit_reason"] == "take_profit" for trade in ledger) / max(len(ledger), 1),
        "tp": sum(trade["exit_reason"] == "take_profit" for trade in ledger),
        "sl": sum(trade["exit_reason"] == "stop_loss" for trade in ledger),
        "timeout": sum(trade["exit_reason"] == "timeout" for trade in ledger),
        "cohort_end": sum(trade["exit_reason"] == "cohort_end" for trade in ledger),
        "same_bar_both": sum(bool(trade["same_bar_both_hit"]) for trade in ledger),
        "risk_sized_pf": account["pf"],
        "risk_sized_pnl": account["pnl_r"],
        "risk_sized_max_dd": account["max_dd_r"],
        "final_balance": INITIAL_BALANCE + account["pnl_r"],
    }


def reproduce_legacy_stats(cohort: pd.DataFrame) -> dict[str, Any]:
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": THRESHOLD,
            "edge_threshold": 0.0,
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "max_hold": MAX_HOLD_MINUTES,
            "direction_mode": "long",
            "risk_per_trade": RISK_PER_TRADE,
            "allowed_entry_hours": sorted(ALLOWED_HOURS),
            "allowed_entry_weekdays": sorted(ALLOWED_WEEKDAYS),
            "excluded_rsi_ranges": [(0.0, MIN_ENTRY_RSI), EXCLUDED_RSI],
        }
    )
    probs = make_direction_probs(cohort["buy_prob"].to_numpy(dtype=np.float32), "long")
    return legacy_evaluate(
        params,
        cohort["CLOSE"].to_numpy(dtype=np.float64),
        cohort["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=cohort["decision_time_api"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=cohort["decision_time_api"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=cohort["decision_time_api"].dt.date.to_numpy(),
        rsi_values=cohort["M1_RSI"].to_numpy(dtype=np.float64),
    )


def match_ledgers(
    cohort_name: str,
    left_name: str,
    right_name: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_map = {(trade["raw_episode_id"], trade["episode_trade_ordinal"]): trade for trade in left}
    right_map = {(trade["raw_episode_id"], trade["episode_trade_ordinal"]): trade for trade in right}
    keys = sorted(set(left_map) | set(right_map))
    rows: list[dict[str, Any]] = []
    counts = {
        "trades_retained": 0,
        "trades_removed": 0,
        "trades_added": 0,
        "entry_timestamp_changed": 0,
        "exit_timestamp_changed": 0,
        "tp_to_sl": 0,
        "sl_to_tp": 0,
        "timeout_to_tp_or_sl": 0,
        "pnl_sign_changed": 0,
    }
    for episode, ordinal in keys:
        a = left_map.get((episode, ordinal))
        b = right_map.get((episode, ordinal))
        if a is None:
            counts["trades_added"] += 1
        elif b is None:
            counts["trades_removed"] += 1
        else:
            counts["trades_retained"] += 1
            counts["entry_timestamp_changed"] += int(a["entry_time_api"] != b["entry_time_api"])
            counts["exit_timestamp_changed"] += int(a["exit_time_api"] != b["exit_time_api"])
            counts["tp_to_sl"] += int(a["exit_reason"] == "take_profit" and b["exit_reason"] == "stop_loss")
            counts["sl_to_tp"] += int(a["exit_reason"] == "stop_loss" and b["exit_reason"] == "take_profit")
            counts["timeout_to_tp_or_sl"] += int(a["exit_reason"] == "timeout" and b["exit_reason"] in {"take_profit", "stop_loss"})
            counts["pnl_sign_changed"] += int(bool(a["positive"]) != bool(b["positive"]))
        rows.append(
            {
                "cohort": cohort_name,
                "transition": f"{left_name}->{right_name}",
                "matching_rule": "raw_episode_id plus within-episode trade ordinal",
                "raw_episode_id": episode,
                "episode_trade_ordinal": ordinal,
                "left_trade_id": "" if a is None else a["trade_id"],
                "right_trade_id": "" if b is None else b["trade_id"],
                "left_entry_time": "" if a is None else a["entry_time_api"],
                "right_entry_time": "" if b is None else b["entry_time_api"],
                "left_exit_time": "" if a is None else a["exit_time_api"],
                "right_exit_time": "" if b is None else b["exit_time_api"],
                "left_exit_reason": "" if a is None else a["exit_reason"],
                "right_exit_reason": "" if b is None else b["exit_reason"],
                "left_net_r": "" if a is None else a["net_r"],
                "right_net_r": "" if b is None else b["net_r"],
                "entry_changed": bool(a is not None and b is not None and a["entry_time_api"] != b["entry_time_api"]),
                "exit_changed": bool(a is not None and b is not None and a["exit_time_api"] != b["exit_time_api"]),
                "pnl_sign_changed": bool(a is not None and b is not None and bool(a["positive"]) != bool(b["positive"])),
            }
        )
    return rows, counts


def transition_row(
    cohort: str,
    left: dict[str, Any],
    right: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "transition": f"{left['simulator']}->{right['simulator']}",
        **identity,
        "delta_trades": right["trades"] - left["trades"],
        "delta_trades_per_day": right["trades_per_day"] - left["trades_per_day"],
        "delta_wr": right["realized_wr"] - left["realized_wr"],
        "delta_pf": right["risk_sized_pf"] - left["risk_sized_pf"],
        "delta_mean_r": right["mean_r"] - left["mean_r"],
        "delta_pnl_r": right["pnl_r"] - left["pnl_r"],
        "delta_max_dd_r": right["max_dd_r"] - left["max_dd_r"],
    }


def intrabar_counterfactual(cohort: pd.DataFrame, s0: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "s0_trades": len(s0),
        "sl_before_close_only_tp": 0,
        "tp_intrabar_close_only_missed": 0,
        "same_bar_both_reachable": 0,
        "timeout_classification_changed": 0,
        "outcome_changed": 0,
    }
    for trade in s0:
        entry = int(trade["entry_index"])
        max_exit = min(entry + MAX_HOLD_MINUTES, len(cohort) - 1)
        outcome = "timeout"
        exit_index = max_exit
        both = False
        for index in range(entry + 1, max_exit + 1):
            take = float(cohort["HIGH"].iat[index]) >= trade["entry_price"] + trade["tp_distance"]
            stop = float(cohort["LOW"].iat[index]) <= trade["entry_price"] - trade["sl_distance"]
            if stop or take:
                outcome = "stop_loss" if stop else "take_profit"
                exit_index = index
                both = bool(stop and take)
                break
        counts["sl_before_close_only_tp"] += int(trade["exit_reason"] == "take_profit" and outcome == "stop_loss")
        counts["tp_intrabar_close_only_missed"] += int(trade["exit_reason"] != "take_profit" and outcome == "take_profit")
        counts["same_bar_both_reachable"] += int(both)
        counts["timeout_classification_changed"] += int((trade["exit_reason"] == "timeout") != (outcome == "timeout"))
        counts["outcome_changed"] += int(trade["exit_reason"] != outcome)
        rows.append(
            {
                "cohort": trade["cohort"],
                "s0_trade_id": trade["trade_id"],
                "entry_time": trade["entry_time_api"],
                "s0_exit_time": trade["exit_time_api"],
                "s0_exit_reason": trade["exit_reason"],
                "intrabar_exit_time": pd.Timestamp(cohort["decision_time_api"].iat[exit_index]).isoformat(),
                "intrabar_exit_reason": outcome,
                "same_bar_both_reachable": both,
                "outcome_changed": trade["exit_reason"] != outcome,
            }
        )
    return rows, counts


def entry_state_attribution(
    cohort: pd.DataFrame,
    ledgers: dict[str, list[dict[str, Any]]],
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw = cohort["raw_signal"].to_numpy(dtype=bool)
    eligible = (
        raw
        & cohort["session_api"].to_numpy(dtype=bool)
        & (cohort["M1_RSI"].to_numpy(dtype=np.float64) >= MIN_ENTRY_RSI)
        & ~(
            (cohort["M1_RSI"].to_numpy(dtype=np.float64) >= EXCLUDED_RSI[0])
            & (cohort["M1_RSI"].to_numpy(dtype=np.float64) <= EXCLUDED_RSI[1])
        )
    )
    previous_eligible = np.r_[False, eligible[:-1]]
    legacy = eligible & ~previous_eligible
    previous_raw = np.r_[False, raw[:-1]]
    gaps = np.r_[True, np.diff(cohort["decision_time_api"].to_numpy(dtype="datetime64[ns]")).astype("timedelta64[s]").astype(np.int64) > 120]
    core_episode = raw & (~previous_raw | gaps)
    spread_ok = cohort["effective_spread_points"].to_numpy(dtype=np.float64) <= np.asarray(
        [spread_limit_points(max(float(value) * TP_ATR, MIN_TP_PRICE)) for value in cohort["ATR"]],
        dtype=np.float64,
    )
    core_opportunity = core_episode & eligible & spread_ok
    s5_entries = {trade["entry_index"] for trade in ledgers["S5"]}
    live_entry = np.asarray([index in s5_entries for index in range(len(cohort))], dtype=bool)
    return {
        "legacy_filtered_rising_opportunities": int(legacy.sum()),
        "core_gate_episode_opportunities": int(core_opportunity.sum()),
        "live_equivalent_entries": int(live_entry.sum()),
        "seen_by_all_three": int((legacy & core_opportunity & live_entry).sum()),
        "legacy_only": int((legacy & ~core_opportunity & ~live_entry).sum()),
        "core_gate_only": int((core_opportunity & ~legacy & ~live_entry).sum()),
        "live_equivalent_only": int((live_entry & ~legacy & ~core_opportunity).sum()),
        "legacy_and_live_not_core": int((legacy & live_entry & ~core_opportunity).sum()),
        "core_and_live_not_legacy": int((core_opportunity & live_entry & ~legacy).sum()),
        "entries_delayed_within_raw_episode": audits["S5"]["later_entries_in_episode"],
        "entries_enabled_after_prior_close_same_episode": audits["S5"]["entries_after_prior_close_same_episode"],
        "eligible_rows_prevented_by_occupancy": audits["S5"]["position_blocked_rows"],
        "rsi_floor_blocked_rows": audits["S5"]["rsi_floor_blocked_rows"],
        "rsi_range_blocked_rows": audits["S5"]["rsi_range_blocked_rows"],
        "session_blocked_rows": audits["S5"]["session_blocked_rows"],
        "spread_blocked_rows": audits["S5"]["spread_blocked_rows"],
        "risk_cooldown_blocked_rows": audits["S5"]["risk_cooldown_blocked_rows"],
        "matching_note": "opportunity intersections use exact decision-row identity; S5 counts are executable entries after occupancy/risk",
    }


def timestamp_rows(primary: pd.DataFrame, offset: dict[str, Any]) -> list[dict[str, Any]]:
    zone = f"UTC+{int(offset['inferred_broker_offset_hours']):02d}:00 empirical broker wall-time"
    return [
        {
            "raw_source_timestamp": int(row.time),
            "api_interpreted_timestamp": pd.Timestamp(row.decision_time_api).isoformat(),
            "interpreted_timezone": zone,
            "inferred_actual_utc_timestamp": pd.Timestamp(row.decision_time_actual_utc).isoformat() + "Z",
            "simulator_hour": pd.Timestamp(row.decision_time_api).hour,
            "gemini_hour": pd.Timestamp(row.decision_time_api).hour,
            "actual_utc_hour": pd.Timestamp(row.decision_time_actual_utc).hour,
            "simulator_session_eligible": bool(row.session_api),
            "gemini_session_eligible": bool(row.session_api),
            "utc_corrected_session_eligible": bool(row.session_actual_utc),
            "session_diff_if_utc_corrected": bool(row.session_api) != bool(row.session_actual_utc),
            "buy_prob": float(row.buy_prob),
        }
        for row in primary.itertuples(index=False)
    ]


def enrich_cost_attribution(
    cohort: pd.DataFrame, s2: list[dict[str, Any]], s3: list[dict[str, Any]]
) -> dict[str, Any]:
    fixed = np.asarray([trade["net_r"] for trade in s2], dtype=np.float64)
    observed: list[float] = []
    observed_count = 0
    fallback_count = 0
    for trade in s2:
        index = int(trade["entry_index"])
        spread = float(cohort["effective_spread_points"].iat[index])
        observed_count += int(bool(cohort["spread_observed"].iat[index]))
        fallback_count += int(not bool(cohort["spread_observed"].iat[index]))
        denominator = trade["sl_distance"] + spread * POINT
        observed.append((trade["gross_price"] - (spread + EXTRA_COST_POINTS) * POINT) / denominator)
    s3_values = np.asarray([trade["net_r"] for trade in s3], dtype=np.float64)
    fixed_metrics = reward_metrics(fixed)
    observed_metrics = reward_metrics(np.asarray(observed, dtype=np.float64))
    actual_metrics = reward_metrics(s3_values)
    return {
        "s2_fixed_cost_same_identities": fixed_metrics,
        "s2_observed_fallback_cost_same_identities": observed_metrics,
        "s3_observed_cost_and_spread_gate": actual_metrics,
        "pure_cost_delta_pf": observed_metrics["pf"] - fixed_metrics["pf"],
        "pure_cost_delta_mean_r": observed_metrics["mean_r"] - fixed_metrics["mean_r"],
        "pure_cost_delta_pnl_r": observed_metrics["pnl_r"] - fixed_metrics["pnl_r"],
        "spread_gate_trade_delta": actual_metrics["trades"] - observed_metrics["trades"],
        "observed_spread_entries": observed_count,
        "fallback_spread_entries": fallback_count,
    }


def legacy_reproduction(old: dict[str, Any], s0: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    expected = old["selected"]["test"]
    tolerances = {"trades": 0, "win_rate": 0.005, "profit_factor": 0.02, "pnl": 0.50}
    residual = {
        "trades": s0["trades"] - expected["trades"],
        "win_rate": s0["realized_wr"] - expected["win_rate"],
        "profit_factor": s0["risk_sized_pf"] - expected["profit_factor"],
        "pnl": s0["risk_sized_pnl"] - expected["pnl"],
    }
    reproduced = (
        abs(residual["trades"]) <= tolerances["trades"]
        and abs(residual["win_rate"]) <= tolerances["win_rate"]
        and abs(residual["profit_factor"]) <= tolerances["profit_factor"]
        and abs(residual["pnl"]) <= tolerances["pnl"]
    )
    return {
        "reproduced": reproduced,
        "expected": expected,
        "s0": {key: s0[key] for key in ("trades", "realized_wr", "risk_sized_pf", "risk_sized_pnl")},
        "residual": residual,
        "tolerances": tolerances,
        "repository_evaluator_cross_check": {
            "trades": int(reference["trades"]),
            "win_rate": float(reference["win_rate"]),
            "profit_factor": float(reference["profit_factor"]),
            "pnl": float(reference["pnl"]),
        },
    }


def metric_table_markdown(rows: list[dict[str, Any]], cohort: str) -> list[str]:
    lines = [
        "| Simulator | Trades | Trades/day | WR | PF sized | Mean-R | PnL-R | Max DD-R | TP | SL | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["cohort"] != cohort:
            continue
        lines.append(
            f"| {row['simulator']} | {row['trades']} | {row['trades_per_day']:.4f} | "
            f"{row['realized_wr']:.2%} | {row['risk_sized_pf']:.4f} | {row['mean_r']:.4f} | "
            f"{row['pnl_r']:.2f} | {row['max_dd_r']:.2f} | {row['tp']} | {row['sl']} | {row['timeout']} |"
        )
    return lines


def make_report(metrics: dict[str, Any]) -> str:
    primary = metrics["simulator_metrics"]
    transitions = {row["transition"]: row for row in metrics["transition_attribution"] if row["cohort"] == "PRIMARY_2026_RECENT"}
    reproduction = metrics["legacy_reproduction"]
    intrabar = metrics["intrabar_attribution"]["counts"]
    entry = metrics["entry_state_attribution"]
    timezone_result = metrics["timezone_attribution"]
    s5 = next(row for row in primary if row["cohort"] == "PRIMARY_2026_RECENT" and row["simulator"] == "S5")
    conclusions = metrics["conclusions"]
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Status: **research_only diagnostic**. No model was trained, no strategy candidate was selected, and no operational artifact changed.",
        "",
        "## Primary exact-artifact frozen cohort",
        "",
        f"Rows: {metrics['cohort_manifest']['primary']['rows']:,}; SHA-256: `{metrics['cohort_manifest']['primary']['sha256']}`.",
        "",
        *metric_table_markdown(primary, "PRIMARY_2026_RECENT"),
        "",
        "## Secondary W0 historical-replica cohort",
        "",
        "This cohort is reported separately and is never pooled with the exact-artifact primary cohort.",
        "",
        *metric_table_markdown(primary, "SECONDARY_W0_2018_2020"),
        "",
        "## S0 reproduction",
        "",
        f"Old result reproduced within frozen tolerances: **{reproduction['reproduced']}**.",
        f"Residuals: trades {reproduction['residual']['trades']:+d}, WR {reproduction['residual']['win_rate']:+.4%}, "
        f"PF {reproduction['residual']['profit_factor']:+.4f}, account PnL {reproduction['residual']['pnl']:+.4f}.",
        "",
        "## Primary transition attribution",
        "",
        "| Transition | Δ trades | Δ WR | Δ PF | Δ Mean-R | Δ PnL-R | Retained | Removed | Added | PnL sign changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for transition in ("S0->S1", "S1->S2", "S2->S3", "S3->S4", "S4->S5"):
        row = transitions[transition]
        lines.append(
            f"| {transition} | {row['delta_trades']:+d} | {row['delta_wr']:+.2%} | {row['delta_pf']:+.4f} | "
            f"{row['delta_mean_r']:+.4f} | {row['delta_pnl_r']:+.2f} | {row['trades_retained']} | "
            f"{row['trades_removed']} | {row['trades_added']} | {row['pnl_sign_changed']} |"
        )
    lines.extend(
        [
            "",
            "## Intrabar causal attribution",
            "",
            f"S0 trades={intrabar['s0_trades']}; outcome changed={intrabar['outcome_changed']}; "
            f"SL before close-only TP={intrabar['sl_before_close_only_tp']}; intrabar TP missed by close-only={intrabar['tp_intrabar_close_only_missed']}; "
            f"same-bar both reachable={intrabar['same_bar_both_reachable']}; timeout classification changed={intrabar['timeout_classification_changed']}.",
            "",
            "## Entry state machine",
            "",
            f"Legacy opportunities={entry['legacy_filtered_rising_opportunities']}; core-gate opportunities={entry['core_gate_episode_opportunities']}; "
            f"S5 entries={entry['live_equivalent_entries']}; S5-only={entry['live_equivalent_only']}; "
            f"delayed within persistent signal={entry['entries_delayed_within_raw_episode']}; re-entry after prior close in same episode={entry['entries_enabled_after_prior_close_same_episode']}; "
            f"occupancy-blocked eligible rows={entry['eligible_rows_prevented_by_occupancy']}.",
            "",
            "## Timezone/session reconciliation",
            "",
            f"Contemporaneous live log rows imply broker/API wall-time offset UTC+{metrics['broker_offset']['inferred_broker_offset_hours']}. "
            f"S0-S5 use the exact hour that gemini.py uses. Correcting to inferred actual UTC would change "
            f"{timezone_result['candidate_rows_with_session_change']} qualifying rows and "
            f"{timezone_result['s5_trade_identity_difference']} S5 trade identities; this alternative is diagnostic only.",
            "",
            "## Required conclusions",
            "",
        ]
    )
    for index, answer in enumerate(conclusions["answers"], start=1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## Reproducibility limits",
            "",
            "The primary M1 row count and S0 metrics are directly reconciled against the old report. The original raw MT5 snapshot was not retained, so the newly fetched broker history cannot be proven byte-identical. S5 is live-equivalent conditional on the one frozen probability cohort; historical tick path, broker fill/slippage, lot rounding, and exact account state are unavailable and explicitly excluded.",
            "",
            "## Operational safety",
            "",
            "`gemini.py` and `gold_long_recent_candidate_xgb.json` remained byte-identical before and after the run.",
        ]
    )
    return "\n".join(lines) + "\n"


def conclusions(
    reproduction: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    intrabar: dict[str, int],
    entry: dict[str, Any],
    timezone_result: dict[str, Any],
    cost_result: dict[str, Any],
) -> dict[str, Any]:
    primary = {row["simulator"]: row for row in metrics_rows if row["cohort"] == "PRIMARY_2026_RECENT"}
    trans = {row["transition"]: row for row in transitions if row["cohort"] == "PRIMARY_2026_RECENT"}
    s5 = primary["S5"]
    s5_positive = bool(s5["risk_sized_pf"] > 1 and s5["mean_r"] > 0 and s5["break_even_adjusted_edge"] is not None and s5["break_even_adjusted_edge"] > 0)
    if not reproduction["reproduced"]:
        next_hypothesis = "SOURCE-SNAPSHOT RECONCILIATION: recover or prove the exact 2026 MT5 raw snapshot/timezone and original probability vector before any further alpha research."
    elif trans["S0->S1"]["delta_wr"] <= -0.10 or trans["S0->S1"]["delta_pf"] <= -0.20:
        next_hypothesis = "EXECUTION-ALIGNED LABEL/MODEL VALIDATION: test whether a model trained on HIGH/LOW stop-first executable outcomes contains alpha, without threshold tuning."
    elif s5_positive:
        next_hypothesis = "RECENT-WINDOW EXECUTION-ALIGNED MODELING: freeze a rolling recent training design and validate it chronologically under S5 semantics, without frequency expansion."
    else:
        next_hypothesis = "ALPHA VALIDITY UNDER EXACT EXECUTION: require a genuinely new entry-time information source to discriminate S5-positive trades before any frequency work."
    if reproduction["reproduced"] and not s5_positive:
        old_interpretation = "a combination of recent-sample signal quality, selection conditioning, and simulator optimism/execution mismatch; it is not execution-robust alpha"
    elif reproduction["reproduced"]:
        old_interpretation = "a combination of recent-regime signal quality, selection conditioning, and simulator semantics; S5 remains positive but sample size is small"
    else:
        old_interpretation = "not fully attributable because the old result was not reproduced; residual source snapshot/timezone/probability mismatch remains"
    answers = [
        f"Original ~15 trade / 66.67% / PF ~1.34 reproduced: {reproduction['reproduced']}.",
        f"Residual mismatch: trades {reproduction['residual']['trades']:+d}, WR {reproduction['residual']['win_rate']:+.4%}, PF {reproduction['residual']['profit_factor']:+.4f}, account PnL {reproduction['residual']['pnl']:+.4f}.",
        f"Close-only to HIGH/LOW first-touch: WR {trans['S0->S1']['delta_wr']:+.2%}, PF {trans['S0->S1']['delta_pf']:+.4f}.",
        f"Intrabar ordering changed {intrabar['outcome_changed']} S0 outcomes; {intrabar['same_bar_both_reachable']} first-touch bars could reach both barriers.",
        f"Entry-state-machine transition S1->S2 changed trades by {trans['S1->S2']['delta_trades']:+d}, WR by {trans['S1->S2']['delta_wr']:+.2%}, PF by {trans['S1->S2']['delta_pf']:+.4f}.",
        f"RSI/session/filter ordering: S5 blocked session={entry['session_blocked_rows']}, RSI-floor={entry['rsi_floor_blocked_rows']}, RSI-range={entry['rsi_range_blocked_rows']}, spread={entry['spread_blocked_rows']} raw rows; exact trade deltas are retained in transition_attribution.csv.",
        f"Spread/cost semantics S2->S3: WR {trans['S2->S3']['delta_wr']:+.2%}, PF {trans['S2->S3']['delta_pf']:+.4f}; pure identical-identity cost ΔPF={cost_result['pure_cost_delta_pf']:+.4f}, ΔMean-R={cost_result['pure_cost_delta_mean_r']:+.4f}.",
        f"Risk/cooldown S3->S4: trades {trans['S3->S4']['delta_trades']:+d}, WR {trans['S3->S4']['delta_wr']:+.2%}, PF {trans['S3->S4']['delta_pf']:+.4f}; cooldown-blocked rows={entry['risk_cooldown_blocked_rows']}.",
        f"Timezone/session interpretation changes {timezone_result['candidate_rows_with_session_change']} qualifying rows and {timezone_result['s5_trade_identity_difference']} S5 trade identities under the diagnostic UTC-corrected alternative.",
        f"S5 primary: trades={s5['trades']}, trades/day={s5['trades_per_day']:.4f}, WR={s5['realized_wr']:.2%}, PF={s5['risk_sized_pf']:.4f}, Mean-R={s5['mean_r']:.4f}, PnL-R={s5['pnl_r']:.2f}, Max DD-R={s5['max_dd_r']:.2f}.",
        f"S5 economically positive on the recent cohort: {s5_positive}.",
        f"Old 66-70% interpretation: {old_interpretation}.",
        "The historical 25.40% OOF weakness remains relevant as cross-regime architecture evidence; simulator reconciliation does not turn fold-specific historical replicas into the exact current artifact or erase their poor ranking/economics.",
        f"Single next research hypothesis: {next_hypothesis}",
    ]
    return {
        "answers": answers,
        "s5_economically_positive": s5_positive,
        "old_result_interpretation": old_interpretation,
        "historical_25_40_relevant": True,
        "single_next_research_hypothesis": next_hypothesis,
    }


def self_check() -> None:
    assert [definition.simulator for definition in SIMULATORS] == [f"S{i}" for i in range(6)]
    for left, right in zip(SIMULATORS, SIMULATORS[1:]):
        changed = {
            key
            for key in left.__dict__
            if key != "simulator" and left.__dict__[key] != right.__dict__[key]
        }
        assert changed == set(TRANSITION_ALLOWED_CHANGES[f"{left.simulator}->{right.simulator}"])
    sample = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-06-01", periods=5, freq="min"),
            "OPEN": [100.0] * 5,
            "HIGH": [100.0, 102.0, 100.1, 100.1, 100.1],
            "LOW": [100.0, 97.0, 99.9, 99.9, 99.9],
            "CLOSE": [100.0, 101.5, 100.0, 100.0, 100.0],
            "ATR": [1.0] * 5,
            "M1_RSI": [50.0] * 5,
            "SPREAD": [30.0] * 5,
            "buy_prob": [0.8, 0.8, 0.1, 0.1, 0.1],
            "sell_prob": [0.0] * 5,
            "all_features_finite": [True] * 5,
            "time": np.arange(5),
            "FEATURE_BAR_TIME": pd.date_range("2026-05-31 23:59", periods=5, freq="min"),
        }
    )
    sample = finalize_cohort(sample, "SELF_CHECK", 0)
    s0, s0_audit = simulate(sample, SIMULATORS[0])
    s1, _ = simulate(sample, SIMULATORS[1])
    assert s0[0]["exit_reason"] == "take_profit"
    assert s1[0]["exit_reason"] == "stop_loss" and s1[0]["same_bar_both_hit"]
    s0_metrics = simulator_metrics(sample, SIMULATORS[0], s0, s0_audit, "self-check")
    reference = reproduce_legacy_stats(sample)
    assert reference["trades"] == s0_metrics["trades"]
    assert abs(reference["win_rate"] - s0_metrics["realized_wr"]) < 1e-12
    assert (
        math.isinf(reference["profit_factor"])
        and math.isinf(s0_metrics["risk_sized_pf"])
    ) or abs(reference["profit_factor"] - s0_metrics["risk_sized_pf"]) < 1e-12
    assert abs(reference["pnl"] - s0_metrics["risk_sized_pnl"]) < 1e-12
    print("SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    run_dir = args.run_dir.resolve()
    required = [MODEL_FILE, GEMINI_FILE, OLD_REPORT_FILE, SIGNAL_LOG_FILE, PRIOR_PREDICTIONS, PRIOR_REPORT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs: " + ", ".join(missing))
    operational_before, manifest = preregister(run_dir)
    self_check()

    signal_snapshot = run_dir / "input_snapshots" / "gemini_signal_log.csv.gz"
    snapshot_gzip(SIGNAL_LOG_FILE, signal_snapshot)
    broker_offset = infer_broker_offset_hours(SIGNAL_LOG_FILE)
    if broker_offset["inferred_broker_offset_hours"] != 3:
        raise RuntimeError(f"Expected empirically supported primary broker offset +3, got {broker_offset}")

    if not DEFAULT_TERMINAL.is_file():
        raise FileNotFoundError(DEFAULT_TERMINAL)
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        primary_raw, features, primary_fetch = build_primary_features()
    finally:
        mt5.shutdown()
    primary = finalize_cohort(primary_raw, "PRIMARY_2026_RECENT", broker_offset["inferred_broker_offset_hours"])
    primary_path = run_dir / "primary_frozen_cohort.csv.gz"
    save_primary_cohort(primary_path, primary, features)
    primary_hash = sha256(primary_path)

    secondary, secondary_identity = load_secondary_cohort(0)
    secondary_path = run_dir / "secondary_frozen_cohort.npz"
    save_secondary_cohort(secondary_path, secondary)
    secondary_hash = sha256(secondary_path)
    cohorts = [(primary, primary_hash), (secondary, secondary_hash)]
    print(f"PRIMARY rows={len(primary):,} hash={primary_hash}", flush=True)
    print(f"SECONDARY rows={len(secondary):,} hash={secondary_hash}", flush=True)

    all_metrics: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_identity: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    per_cohort_ledgers: dict[str, dict[str, list[dict[str, Any]]]] = {}
    per_cohort_audits: dict[str, dict[str, dict[str, Any]]] = {}
    for cohort, cohort_hash in cohorts:
        cohort_name = str(cohort["cohort"].iat[0])
        ledgers: dict[str, list[dict[str, Any]]] = {}
        audits: dict[str, dict[str, Any]] = {}
        metrics_by_sim: dict[str, dict[str, Any]] = {}
        for definition in SIMULATORS:
            ledger, audit = simulate(cohort, definition)
            ledgers[definition.simulator] = ledger
            audits[definition.simulator] = audit
            metric = simulator_metrics(cohort, definition, ledger, audit, cohort_hash)
            metrics_by_sim[definition.simulator] = metric
            all_metrics.append(metric)
            all_trades.extend(ledger)
            print(
                f"{cohort_name} {definition.simulator}: trades={metric['trades']} WR={metric['realized_wr']:.2%} "
                f"PF={metric['risk_sized_pf']:.4f} Mean-R={metric['mean_r']:.4f}",
                flush=True,
            )
        for left, right in zip(SIMULATORS, SIMULATORS[1:]):
            identity_rows, identity_counts = match_ledgers(
                cohort_name, left.simulator, right.simulator, ledgers[left.simulator], ledgers[right.simulator]
            )
            all_identity.extend(identity_rows)
            transitions.append(
                transition_row(cohort_name, metrics_by_sim[left.simulator], metrics_by_sim[right.simulator], identity_counts)
            )
        per_cohort_ledgers[cohort_name] = ledgers
        per_cohort_audits[cohort_name] = audits

    reference_stats = reproduce_legacy_stats(primary)
    old_report = read_json(OLD_REPORT_FILE)
    primary_s0 = next(row for row in all_metrics if row["cohort"] == "PRIMARY_2026_RECENT" and row["simulator"] == "S0")
    reproduction = legacy_reproduction(old_report, primary_s0, reference_stats)
    if (
        reference_stats["trades"] != primary_s0["trades"]
        or abs(reference_stats["win_rate"] - primary_s0["realized_wr"]) > 1e-12
        or abs(reference_stats["profit_factor"] - primary_s0["risk_sized_pf"]) > 1e-8
        or abs(reference_stats["pnl"] - primary_s0["risk_sized_pnl"]) > 1e-8
    ):
        raise RuntimeError("Independent S0 ledger does not reconcile with repository legacy evaluator")

    intrabar_rows, intrabar_counts = intrabar_counterfactual(
        primary, per_cohort_ledgers["PRIMARY_2026_RECENT"]["S0"]
    )
    entry_result = entry_state_attribution(
        primary,
        per_cohort_ledgers["PRIMARY_2026_RECENT"],
        per_cohort_audits["PRIMARY_2026_RECENT"],
    )
    tz_ledger, tz_audit = simulate(primary, SIMULATORS[-1], use_actual_utc_session=True)
    _, tz_identity = match_ledgers(
        "PRIMARY_2026_RECENT", "S5", "S5_UTC_CORRECTED", per_cohort_ledgers["PRIMARY_2026_RECENT"]["S5"], tz_ledger
    )
    timezone_result = {
        "candidate_rows_with_session_change": int(
            (primary["raw_signal"] & (primary["session_api"] != primary["session_actual_utc"])).sum()
        ),
        "all_rows_with_session_change": int((primary["session_api"] != primary["session_actual_utc"]).sum()),
        "s5_api_hour_trades": len(per_cohort_ledgers["PRIMARY_2026_RECENT"]["S5"]),
        "s5_utc_corrected_trades": len(tz_ledger),
        "s5_trade_identity_difference": tz_identity["trades_removed"] + tz_identity["trades_added"] + tz_identity["entry_timestamp_changed"],
        "identity_counts": tz_identity,
        "utc_corrected_audit": tz_audit,
        "policy": "S0-S5 use broker/API-labeled hour because gemini.py does; UTC-corrected alternative is diagnostic only",
    }
    cost_result = enrich_cost_attribution(
        primary,
        per_cohort_ledgers["PRIMARY_2026_RECENT"]["S2"],
        per_cohort_ledgers["PRIMARY_2026_RECENT"]["S3"],
    )
    conclusion_result = conclusions(
        reproduction, all_metrics, transitions, intrabar_counts, entry_result, timezone_result, cost_result
    )

    timestamp_path = run_dir / "timestamp_reconciliation.csv"
    write_csv(timestamp_path, timestamp_rows(primary, broker_offset))
    write_csv(run_dir / "simulator_comparison.csv", all_metrics)
    write_csv(run_dir / "trade_ledger.csv", all_trades)
    write_csv(run_dir / "trade_identity_comparison.csv", all_identity)
    write_csv(run_dir / "transition_attribution.csv", transitions)
    write_csv(run_dir / "intrabar_trade_attribution.csv", intrabar_rows)
    write_json(run_dir / "entry_state_attribution.json", entry_result)
    write_json(run_dir / "cost_attribution.json", cost_result)
    write_json(run_dir / "timezone_attribution.json", timezone_result)

    cohort_manifest = {
        "created_at_utc": now_utc(),
        "operational_model": {"path": MODEL_FILE.name, "sha256": operational_before[MODEL_FILE.name]},
        "probability_freeze": "one probability vector per cohort; all S0-S5 consume identical in-memory and retained values",
        "primary": {
            **primary_fetch,
            "rows": len(primary),
            "path": primary_path.name,
            "sha256": primary_hash,
            "api_timestamp_start": primary["decision_time_api"].iat[0],
            "api_timestamp_end": primary["decision_time_api"].iat[-1],
            "inferred_actual_utc_start": primary["decision_time_actual_utc"].iat[0],
            "inferred_actual_utc_end": primary["decision_time_actual_utc"].iat[-1],
            "data_source": "XMGlobal-MT5 6 via D:/XM2/terminal64.exe; re-fetched historical bars",
            "raw_snapshot_limit": "old 2026 fetch was not retained; current row-count/S0 reconciliation can test but not prove byte identity",
        },
        "secondary": {**secondary_identity, "path": secondary_path.name, "sha256": secondary_hash},
        "broker_offset": broker_offset,
    }
    write_json(run_dir / "cohort_manifest.json", cohort_manifest)

    metrics = {
        "experiment": EXPERIMENT,
        "status": "research_only",
        "candidate_selected": False,
        "cohort_manifest": cohort_manifest,
        "broker_offset": broker_offset,
        "simulator_definitions": {definition.simulator: definition.__dict__ for definition in SIMULATORS},
        "simulator_metrics": all_metrics,
        "transition_attribution": transitions,
        "legacy_reproduction": reproduction,
        "intrabar_attribution": {"counts": intrabar_counts, "table": "intrabar_trade_attribution.csv"},
        "entry_state_attribution": entry_result,
        "timezone_attribution": timezone_result,
        "cost_attribution": cost_result,
        "conclusions": conclusion_result,
    }
    report_path = run_dir / "report.md"
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    report_path.write_text(make_report(metrics), encoding="utf-8")

    operational_after = {GEMINI_FILE.name: sha256(GEMINI_FILE), MODEL_FILE.name: sha256(MODEL_FILE)}
    if operational_after != operational_before:
        raise RuntimeError("Operational artifact changed during diagnostic")
    manifest = read_json(run_dir / "manifest.json")
    terminal_info = {} if terminal is None else terminal._asdict()
    account_info = {} if account is None else account._asdict()
    manifest["data"].update(
        {
            "symbols": [live.SYMBOL],
            "data_sources": ["XMGlobal-MT5 6 historical rates", "prior preserved W0 OOF cohort"],
            "source_files": [
                {"path": primary_path.name, "sha256": primary_hash, "retention_status": "stored_in_run_directory"},
                {"path": secondary_path.name, "sha256": secondary_hash, "retention_status": "stored_in_run_directory"},
            ],
            "timezone": "MT5 API timestamp empirically maps to broker UTC+3 during primary; exact recorded API hour retained",
            "data_start_utc": primary["decision_time_actual_utc"].iat[0],
            "data_end_utc": primary["decision_time_actual_utc"].iat[-1],
            "train_start_utc": None,
            "train_end_utc": "before 2026-06-01 broker/API test boundary per retained legacy training code/report",
            "train_rows": None,
            "validation_start_utc": None,
            "validation_end_utc": None,
            "validation_rows": None,
            "test_start_utc": primary["decision_time_actual_utc"].iat[0],
            "test_end_utc": primary["decision_time_actual_utc"].iat[-1],
            "test_rows": len(primary),
            "purge_details": "not applicable to this no-fitting run; operational artifact was fit before primary",
            "embargo_details": "not applicable to this no-fitting run",
            "raw_snapshot_retained": True,
            "reproducibility_claim": "simulator-exact from frozen cohort; original MT5 fetch byte identity unavailable",
            "mt5_fetch": {
                "used": True,
                "terminal_path": str(DEFAULT_TERMINAL),
                "terminal_info": {key: terminal_info.get(key) for key in ("name", "path", "build", "connected")},
                "broker_info": {key: account_info.get(key) for key in ("login", "server", "currency")},
                "fetch_start_utc": str(DEFAULT_START),
                "fetch_end_utc": PRIMARY_END_API,
                "retrieved_at_utc": now_utc(),
                "returned_rows": len(primary),
                "not_applicable_reason": None,
            },
        }
    )
    manifest["registry"].update(
        {
            "parent_or_incumbent": f"{MODEL_FILE.name}@{operational_before[MODEL_FILE.name]}",
            "selected_configuration": "none_diagnostic_S0_through_S5",
            "trades_per_day": primary_s0["trades_per_day"],
            "realized_win_rate": primary_s0["realized_wr"],
            "pf": primary_s0["risk_sized_pf"],
            "mean_r": primary_s0["mean_r"],
            "pnl": primary_s0["pnl_r"],
            "max_dd": primary_s0["max_dd_r"],
            "validator_result": None,
        }
    )
    manifest["operational_hashes_after"] = operational_after
    manifest["promotion"]["operational_artifact_changed"] = False
    manifest["diagnostic_result"] = {
        "s0_reproduced": reproduction["reproduced"],
        "s5_economically_positive": conclusion_result["s5_economically_positive"],
        "classification": conclusion_result["old_result_interpretation"],
        "single_next_hypothesis": conclusion_result["single_next_research_hypothesis"],
    }
    artifact_files = [
        primary_path,
        secondary_path,
        signal_snapshot,
        run_dir / "cohort_manifest.json",
        run_dir / "simulator_comparison.csv",
        run_dir / "trade_ledger.csv",
        run_dir / "trade_identity_comparison.csv",
        run_dir / "transition_attribution.csv",
        run_dir / "intrabar_trade_attribution.csv",
        timestamp_path,
        run_dir / "entry_state_attribution.json",
        run_dir / "cost_attribution.json",
        run_dir / "timezone_attribution.json",
        metrics_path,
        report_path,
    ]
    for path in artifact_files:
        add_artifact(manifest, run_dir, path, "execution_semantics_diagnostic_evidence")
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"DIAGNOSTIC_COMPLETE S0_reproduced={reproduction['reproduced']} "
        f"S5_positive={conclusion_result['s5_economically_positive']} operational_artifact_changed=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
