"""Fixed paired B0 technical-control versus B1 macro-timing information test."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import drl_trading_v2
import gold_gemini_execution_aligned_label_v1 as base
import gold_gemini_execution_semantics_v1 as semantics
import training_run_history as archive

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "GEMINI MACRO EVENT TIMING B0 VS B1 V1"
FOUNDATION = ROOT / "training_runs/20260905T171629Z_gemini_macro_event_integration_foundation_v1"
REFERENCE = ROOT / "training_runs/20260903T071729Z_gemini_execution_aligned_label_v1"
MODEL_IDS = ("B0_technical_control", "B1_technical_plus_macro")
MACRO_FEATURES = [
    "EVENT_MINUTES_SINCE", "EVENT_POST_0_15", "EVENT_POST_15_60",
    "EVENT_POST_60_240", "EVENT_TYPE_CPI", "EVENT_TYPE_EMPLOYMENT",
    "EVENT_TYPE_PCE", "EVENT_TYPE_FOMC",
]
EXPECTED_MACRO_DATASET_SHA = "cd212f31ca0f9b8294c485e20aa863216eb859a6825540a82d40a46f81162592"
EXPECTED_EVENT_MATRIX_SHA = "8f0bf303448bccb8ae418cd9b980447a835704731875336dddd4e0a64f74aac1"
REFERENCE_METRICS = {
    "scored_rows": 2474297, "prediction_count": 2474297, "trades": 689,
    "realized_wr": 0.5660377358490566, "pf": 0.8247098331219882,
    "mean_r": -0.07690539315994348, "pnl_r": -52.98781588720106,
    "max_dd_r": -59.57958015890608, "cost_stress_pf": 0.7838186120438296,
    "spearman_score_realized_net_r": 0.29006366478201323,
    "top_decile_mean_r": -0.13362516057222326, "top_decile_pf": 0.7214634681339611,
    "top_quintile_mean_r": -0.15795575250743074, "top_quintile_pf": 0.6956121020778562,
}
REFERENCE_PROBABILITY = {
    "mean": 0.5047674962576704, "median": 0.494555801153183,
    "p75": 0.570514440536499, "p90": 0.6531402349472046,
    "p95": 0.685541009902954, "p99": 0.722646541595459,
    "max": 0.861997663974762,
}
REPRODUCTION_TOLERANCES = {
    "scored_rows": 0, "prediction_count": 0, "trades": 2,
    "realized_wr": 0.005, "pf": 0.02, "mean_r": 0.01,
    "spearman_score_realized_net_r": 0.005, "probability_stat": 0.005,
}
INCREMENTAL_GATE = {
    "minimum_improved_spearman_folds": 2,
    "minimum_pooled_spearman_delta": 0.005,
    "maximum_remaining_fold_spearman_deterioration": 0.01,
}

# The reused evaluator resolves these globals inside its own module.
base.MODEL_IDS = MODEL_IDS


def inventory(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): base.sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def schema_hash(features: list[str]) -> str:
    return hashlib.sha256(json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def preregister(run: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = base.read_json(run / "manifest.json")
    if manifest["git_dirty"] is not False or base.sha256(Path(__file__)) != manifest["training_script_sha256"]:
        raise RuntimeError("Formal run requires the clean committed immutable script")
    head = base.git("rev-parse", "HEAD")
    upstream_ref = base.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    upstream = base.git("rev-parse", upstream_ref)
    if head != upstream or head != manifest["git_commit"]:
        raise RuntimeError("HEAD, upstream, and run manifest commit differ")
    if archive.validate_run(FOUNDATION):
        raise RuntimeError("Certified foundation provenance no longer validates")
    foundation_metrics = base.read_json(FOUNDATION / "metrics.json")
    if (foundation_metrics["macro_event_dataset_sha256"] != EXPECTED_MACRO_DATASET_SHA
            or foundation_metrics["event_feature_matrix_sha256"] != EXPECTED_EVENT_MATRIX_SHA
            or foundation_metrics["incomplete_event_history_rows_exact"] != 0):
        raise RuntimeError("Frozen macro foundation identity/readiness mismatch")
    operational = {p.name: base.sha256(p) for p in (base.GEMINI_FILE, base.OPERATIONAL_MODEL)}
    manifest["pre_run_git"] = {"pre_run_git_commit": head, "pre_run_git_dirty": False,
        "head_sha": head, "remote_ref": upstream_ref, "origin_main_sha": upstream, "head_equals_origin_main": True}
    manifest["operational_hashes_before"] = operational
    manifest["protected_finalized_runs_before"] = {
        p.parent.name: inventory(p.parent) for p in sorted((ROOT / "training_runs").glob("*/FINALIZED.json"))
    }
    manifest["foundation_identity"] = {
        "run_id": FOUNDATION.name, "finalized_sha256": base.sha256(FOUNDATION / "FINALIZED.json"),
        "macro_event_dataset_sha256": EXPECTED_MACRO_DATASET_SHA,
        "event_feature_matrix_sha256": EXPECTED_EVENT_MATRIX_SHA,
        "incomplete_event_history_rows_exact": 0,
    }
    manifest["paired_design"] = {
        "model_definitions": list(MODEL_IDS), "only_design_change": "add frozen eight-feature macro timing family",
        "technical_feature_count": 31, "b1_feature_count": 39, "training_window_months": base.TRAIN_MONTHS,
        "folds": [{"fold": n, "start": str(s), "end_exclusive": str(e)} for n, s, e in base.FOLDS],
        "target": "standalone S5 net realized R > 0", "class_weighting": "balanced binary weights from frozen train rows",
        "xgboost_parameters": base.FIXED_XGB_PARAMETERS, "seed": base.RANDOM_STATE,
        "threshold": base.THRESHOLD, "minimum_entry_rsi": base.MIN_ENTRY_RSI,
        "excluded_rsi": list(base.EXCLUDED_RSI), "tp_atr": semantics.TP_ATR, "sl_atr": semantics.SL_ATR,
        "hold_wall_clock_minutes": base.EXECUTION_HORIZON_MINUTES, "simulator": semantics.SIMULATORS[-1].simulator,
        "parameter_search": False, "threshold_search": False, "event_subset_search": False,
        "reproduction_tolerances": REPRODUCTION_TOLERANCES,
        "incremental_discrimination_gate": INCREMENTAL_GATE,
    }
    manifest["evidence_status"] = {"classification": "chronological_development_paired_information_test",
        "all_historical_intervals_are_development": True, "untouched_oos_claim": False,
        "previous_forward_status": base.PREVIOUS_FORWARD_STATUS, "new_forward_cutoff": None}
    manifest["search"].update({"performed": False, "predefined_search_space": {"information_sets": list(MODEL_IDS)},
        "candidate_results_file": "candidates.csv",
        "not_applicable_reason": "Exactly two fixed paired information sets; no parameters, thresholds, windows, subsets or interactions searched."})
    manifest["model"].update({"trained": True, "model_type": "paired fold-specific XGBoost binary logistic models",
        "parameters": base.FIXED_XGB_PARAMETERS, "boosted_rounds_or_estimators": base.N_ESTIMATORS,
        "features": ["pending_verified_39_feature_schema"], "feature_count": 39,
        "label_definition": "standalone next-open S5 nominal net R > 0",
        "horizon": {"wall_clock_minutes": base.EXECUTION_HORIZON_MINUTES},
        "label_tp_sl_semantics": "next-open; entry-bar HIGH/LOW; same-bar stop-first; 1.3/1.6 ATR; observed/fallback spread; nominal cost; 90 wall-clock minutes",
        "execution_tp_sl_semantics": "same frozen S5 for B0 and B1", "calibration_method": "none",
        "artifact_path": "pending", "artifact_sha256": "0" * 64,
        "retention_status": "all fold models will remain in this research run", "not_applicable_reason": None})
    manifest["promotion"].update({"requested": False, "gate_result": "not_requested_research_only",
        "replacement_authorized": False, "operational_artifact_changed": False})
    manifest["dependency_sha256"] = {p.name: base.sha256(p) for p in (
        ROOT / "gold_gemini_execution_aligned_label_v1.py", ROOT / "gold_gemini_execution_semantics_v1.py",
        ROOT / "gold_gemini_core_gate_v1.py", ROOT / "barrier_research_suite.py",
        ROOT / "barrier_final_train.py", ROOT / "drl_trading_v2.py")}
    base.write_json(run / "manifest.json", manifest)
    print(f"PREREGISTERED {head} reproduction={REPRODUCTION_TOLERANCES} incremental={INCREMENTAL_GATE}", flush=True)
    return operational, manifest


def load_macro_mapping() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metrics = base.read_json(FOUNDATION / "metrics.json")
    if base.sha256(FOUNDATION / "canonical_macro_events.csv") != EXPECTED_MACRO_DATASET_SHA:
        raise RuntimeError("Macro event CSV bytes changed")
    with np.load(FOUNDATION / "event_features.npz", allow_pickle=False) as matrix:
        utc = matrix["utc_ns"].astype(np.int64)
        raw_values = matrix["features"].astype(np.float64)
        values = raw_values.astype(np.float32)
        names = matrix["feature_names"].astype(str).tolist()
        incomplete = matrix["incomplete_mask"].astype(bool)
    if names != MACRO_FEATURES or base.array_sha256(raw_values) != EXPECTED_EVENT_MATRIX_SHA or incomplete.any():
        raise RuntimeError("Certified event matrix logical identity/schema changed")
    broker_parts, utc_parts, blocks = [], [], []
    with np.load(FOUNDATION / "exact_timestamps.npz", allow_pickle=False) as exact:
        for i, fold in enumerate(metrics["blocks"]):
            key = fold["block"]
            broker = exact[key + "_broker_ns"].astype(np.int64)
            block_utc = exact[key + "_utc_ns"].astype(np.int64)
            if len(broker) != fold["rows"] or base.array_sha256(broker) != fold["timestamp_sha256"]:
                raise RuntimeError(f"Frozen timestamp mismatch: {key}")
            broker_parts.append(broker); utc_parts.append(block_utc)
            blocks.append({"block": key, "rows": len(broker), "timestamp_sha256": base.array_sha256(broker), "matched": True})
    broker = np.concatenate(broker_parts); mapped_utc = np.concatenate(utc_parts)
    order = np.argsort(broker, kind="stable"); broker, mapped_utc = broker[order], mapped_utc[order]
    unique, first, counts = np.unique(broker, return_index=True, return_counts=True)
    for position, count in zip(first[counts > 1], counts[counts > 1]):
        if not np.all(mapped_utc[position:position + count] == mapped_utc[position]):
            raise RuntimeError("Overlapping exact blocks disagree on UTC mapping")
    mapped_utc = mapped_utc[first]
    index = np.searchsorted(utc, mapped_utc)
    if np.any(index == len(utc)) or not np.array_equal(utc[index], mapped_utc):
        raise RuntimeError("Exact timestamp has no certified macro feature row")
    return unique, values[index], {"blocks": blocks, "all_six_timestamp_hashes_matched": True,
        "macro_event_dataset_sha256": EXPECTED_MACRO_DATASET_SHA,
        "event_feature_matrix_sha256": EXPECTED_EVENT_MATRIX_SHA, "unique_exact_timestamps": len(unique)}


def macro_for(time_ns: np.ndarray, broker_ns: np.ndarray, macro: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(broker_ns, time_ns)
    if np.any(positions == len(broker_ns)) or not np.array_equal(broker_ns[positions], time_ns):
        raise RuntimeError("Training/scoring timestamp is outside certified exact timestamp universe")
    return macro[positions]


def train_model(history: pd.DataFrame, technical: list[str], indices: np.ndarray,
                target: np.ndarray, macro: np.ndarray | None) -> xgb.XGBClassifier:
    frame = history.loc[indices, technical].copy()
    features = list(technical)
    if macro is not None:
        for column, values in zip(MACRO_FEATURES, macro.T):
            frame[column] = values
        features += MACRO_FEATURES
    frame["BARRIER_TARGET"] = target.astype(np.int8)
    model = base.train_binary_model(frame, features, 1, base.N_ESTIMATORS)
    del frame
    return model


def score_model(model: xgb.XGBClassifier, history: pd.DataFrame, technical: list[str],
                indices: np.ndarray, macro: np.ndarray | None) -> np.ndarray:
    frame = history.loc[indices, technical].copy()
    features = list(technical)
    if macro is not None:
        for column, values in zip(MACRO_FEATURES, macro.T):
            frame[column] = values
        features += MACRO_FEATURES
    score = base.predict_positive(model, frame, features).astype(np.float32)
    del frame
    return score


def fold_plan(history: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, Any]]:
    times = history["TIME_DT"]
    ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    maturity = labels["C1_MATURITY_NS"].to_numpy(dtype=np.int64)
    mature = labels["C1_MATURE"].to_numpy(dtype=bool)
    legacy_index = np.arange(len(history), dtype=np.int64) + base.LEGACY_HORIZON_ROWS
    legacy_ok = legacy_index < len(history)
    legacy_maturity = np.full(len(history), np.iinfo(np.int64).max, dtype=np.int64)
    legacy_maturity[legacy_ok] = ns[legacy_index[legacy_ok]]
    result = []
    for code, (name, start, end) in enumerate(base.FOLDS):
        start_ns = start.to_datetime64().astype("datetime64[ns]").astype(np.int64)
        train = ((times >= start - pd.DateOffset(months=base.TRAIN_MONTHS)) & (times < start)
                 & mature & legacy_ok & (maturity < start_ns) & (legacy_maturity < start_ns))
        score = (times >= start) & (times < end) & mature
        ti, si = np.flatnonzero(train.to_numpy()), np.flatnonzero(score.to_numpy())
        if not len(ti) or not len(si) or max(maturity[ti].max(), legacy_maturity[ti].max()) >= start_ns:
            raise RuntimeError(f"Invalid empty/immature fold {name}")
        result.append({"fold": name, "fold_code": code, "train": ti, "score": si,
            "latest_label_information_ns": int(max(maturity[ti].max(), legacy_maturity[ti].max()))})
    return result


def distribution(score: np.ndarray) -> dict[str, float]:
    q = np.quantile(score, [.5, .75, .9, .95, .99])
    return {"mean": float(score.mean()), "median": float(q[0]), "p75": float(q[1]),
            "p90": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(score.max())}


def diagnostics(scored: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    probability, ranking, deciles = [], [], []
    scopes = [(name, scored["fold"].eq(name).to_numpy()) for name, _, _ in base.FOLDS]
    scopes.append(("pooled", np.ones(len(scored), dtype=bool)))
    y_all = scored["C1_TARGET"].to_numpy(dtype=np.int8)
    reward_all = scored["C1_NET_R"].to_numpy(dtype=np.float64)
    for model_id in MODEL_IDS:
        all_score = scored[f"score_{model_id}"].to_numpy(dtype=np.float64)
        for fold, mask in scopes:
            score, y, reward = all_score[mask], y_all[mask], reward_all[mask]
            order = np.argsort(score, kind="stable")
            decile = np.empty(len(score), dtype=np.int8)
            decile[order] = np.minimum(np.arange(len(score)) * 10 // len(score) + 1, 10)
            for number in range(1, 11):
                chosen = decile == number; stats = base.reward_metrics(reward[chosen])
                deciles.append({"model_id": model_id, "fold": fold, "decile": number,
                    "observations": int(chosen.sum()), "mean_probability": float(score[chosen].mean()),
                    "realized_wr": stats["realized_wr"], "pf": stats["pf"], "mean_r": stats["mean_r"], "pnl_r": stats["pnl_r"]})
            top10, top20 = decile == 10, decile >= 9
            ten, twenty = base.reward_metrics(reward[top10]), base.reward_metrics(reward[top20])
            classification = base.safe_classification(y, score)
            probability.append({"model_id": model_id, "fold": fold, "observations": len(score),
                **distribution(score), **classification})
            ranking.append({"model_id": model_id, "fold": fold, "observations": len(score),
                "spearman_score_realized_net_r": base.spearman(score, reward),
                "spearman_score_positive_net_r": base.spearman(score, y.astype(float)),
                "top_decile_count": int(top10.sum()), "top_decile_wr": ten["realized_wr"],
                "top_decile_pf": ten["pf"], "top_decile_mean_r": ten["mean_r"], "top_decile_pnl_r": ten["pnl_r"],
                "top_quintile_count": int(top20.sum()), "top_quintile_wr": twenty["realized_wr"],
                "top_quintile_pf": twenty["pf"], "top_quintile_mean_r": twenty["mean_r"], "top_quintile_pnl_r": twenty["pnl_r"]})
    return probability, ranking, deciles


def evaluate_one(scored: pd.DataFrame, model_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cohort = base.evaluation_cohort(scored, model_id)
    trades, audit = semantics.simulate(cohort, semantics.SIMULATORS[-1])
    for trade in trades:
        entry = pd.Timestamp(trade["entry_time_api"])
        trade["model_id"] = model_id
        trade["fold"] = next(n for n, s, e in base.FOLDS if s <= entry < e)
        trade["trade_id"] = f"{model_id}_{trade['trade_id']}"
    rows = []
    for fold, start, end in base.FOLDS:
        rows.append({"model_id": model_id, "fold": fold,
            **base.trade_metrics([t for t in trades if t["fold"] == fold], base.core.fold_days(start, end))})
    days = sum(base.core.fold_days(s, e) for _, s, e in base.FOLDS)
    rows.append({"model_id": model_id, "fold": "pooled", **base.trade_metrics(trades, days), **audit})
    return rows, trades, audit


def reproduction_gate(scored: pd.DataFrame, metric: dict[str, Any], probability: dict[str, Any],
                      ranking: dict[str, Any]) -> dict[str, Any]:
    actual = {"scored_rows": len(scored), "prediction_count": int(scored[f"score_{MODEL_IDS[0]}"].notna().sum()),
        **{k: metric[k] for k in ("trades", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf")},
        **{k: ranking[k] for k in ("spearman_score_realized_net_r", "top_decile_mean_r", "top_decile_pf",
                                   "top_quintile_mean_r", "top_quintile_pf")}}
    differences, failures = {}, []
    for key, reference in REFERENCE_METRICS.items():
        differences[key] = actual[key] - reference
        tolerance = REPRODUCTION_TOLERANCES.get(key)
        if tolerance is not None and abs(differences[key]) > tolerance:
            failures.append(f"{key}: |{differences[key]}| > {tolerance}")
    probability_differences = {k: probability[k] - v for k, v in REFERENCE_PROBABILITY.items()}
    failures += [f"probability {k}: |{v}| > {REPRODUCTION_TOLERANCES['probability_stat']}"
                 for k, v in probability_differences.items() if abs(v) > REPRODUCTION_TOLERANCES["probability_stat"]]
    return {"pass": not failures, "reference": REFERENCE_METRICS, "reproduced": actual,
        "differences": differences, "reference_probability_distribution": REFERENCE_PROBABILITY,
        "reproduced_probability_distribution": {k: probability[k] for k in REFERENCE_PROBABILITY},
        "probability_differences": probability_differences, "tolerances": REPRODUCTION_TOLERANCES,
        "material_failures": failures}


def event_attribution(scored: pd.DataFrame, macro_values: np.ndarray) -> list[dict[str, Any]]:
    reward = scored["C1_NET_R"].to_numpy(dtype=float)
    target = scored["C1_TARGET"].to_numpy(dtype=np.int8)
    age = macro_values[:, 0] * 1440
    groups = [("window_0_15", age < 15), ("window_15_60", (age >= 15) & (age < 60)),
              ("window_60_240", (age >= 60) & (age < 240)), ("window_gt_or_equal_240", age >= 240)]
    groups += [(f"event_{family}", macro_values[:, 4 + i].astype(bool)) for i, family in enumerate(("CPI", "EMPLOYMENT", "PCE", "FOMC"))]
    rows = []
    for group, mask in groups:
        stats = base.reward_metrics(reward[mask])
        rows.append({"group": group, "observations": int(mask.sum()), "positive_net_r_rate": float(target[mask].mean()) if mask.any() else None,
                     **{k: stats[k] for k in ("realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r")}})
    return rows


def decision(metric_rows: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = {r["model_id"]: r for r in metric_rows if r["fold"] == "pooled"}
    b0, b1 = pooled.values()
    rank = {(r["model_id"], r["fold"]): r for r in ranking}
    comparisons = []
    for fold, _, _ in base.FOLDS:
        left, right = rank[(MODEL_IDS[0], fold)], rank[(MODEL_IDS[1], fold)]
        comparisons.append({"fold": fold,
            "spearman_improved": right["spearman_score_realized_net_r"] > left["spearman_score_realized_net_r"],
            "spearman_delta": right["spearman_score_realized_net_r"] - left["spearman_score_realized_net_r"],
            "top_decile_pf_improved": right["top_decile_pf"] > left["top_decile_pf"],
            "top_decile_mean_r_improved": right["top_decile_mean_r"] > left["top_decile_mean_r"],
            "top_quintile_pf_improved": right["top_quintile_pf"] > left["top_quintile_pf"],
            "top_quintile_mean_r_improved": right["top_quintile_mean_r"] > left["top_quintile_mean_r"]})
    pooled_delta = rank[(MODEL_IDS[1], "pooled")]["spearman_score_realized_net_r"] - rank[(MODEL_IDS[0], "pooled")]["spearman_score_realized_net_r"]
    spearman_deltas = [c["spearman_delta"] for c in comparisons]
    incremental = (sum(c["spearman_improved"] for c in comparisons) >= INCREMENTAL_GATE["minimum_improved_spearman_folds"]
                   and pooled_delta >= INCREMENTAL_GATE["minimum_pooled_spearman_delta"]
                   and min(spearman_deltas) >= -INCREMENTAL_GATE["maximum_remaining_fold_spearman_deterioration"])
    top_b1 = rank[(MODEL_IDS[1], "pooled")]
    high_positive = top_b1["top_decile_pf"] > 1 and top_b1["top_decile_mean_r"] > 0 and top_b1["top_quintile_pf"] > 1 and top_b1["top_quintile_mean_r"] > 0
    viability = b1["pf"] > 1 and b1["mean_r"] > 0 and b1["pnl_r"] > 0 and b1["break_even_adjusted_edge"] > 0
    catastrophic = [r["fold"] for r in metric_rows if r["model_id"] == MODEL_IDS[1] and r["fold"] != "pooled" and base.catastrophic(r)]
    full = viability and b1["realized_wr"] >= .60 and b1["pf"] > 1.05 and b1["cost_stress_pf"] > 1 and not catastrophic
    if not incremental:
        classification, next_hypothesis = "FAIL macro timing family", "Test one genuinely new timestamp-aligned external information family under the same execution-aligned target."
    elif not high_positive:
        classification, next_hypothesis = "incremental information but economically insufficient", "Test one other genuinely new external information family; do not tune macro windows, subsets, or threshold."
    elif not viability:
        classification, next_hypothesis = "ranking-valid / calibration-misaligned", "Preregister a separate past-only calibration/ranking experiment using this frozen information family."
    elif not full:
        classification, next_hypothesis = "promising external information family, not strategy-ready", "Collect a genuinely untouched prospective interval for the frozen information hypothesis before any strategy promotion."
    else:
        classification, next_hypothesis = "full development quality gate PASS; shadow research only", "Collect untouched shadow-forward evidence after the exact candidate freeze cutoff."
    delta = {k: b1[k] - b0[k] for k in ("trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf")}
    return {"b1_minus_b0": delta, "fold_comparisons": comparisons,
        "folds_with_improved_spearman": sum(c["spearman_improved"] for c in comparisons),
        "folds_with_improved_top_decile_pf": sum(c["top_decile_pf_improved"] for c in comparisons),
        "folds_with_improved_top_decile_mean_r": sum(c["top_decile_mean_r_improved"] for c in comparisons),
        "folds_with_improved_top_quintile_pf": sum(c["top_quintile_pf_improved"] for c in comparisons),
        "folds_with_improved_top_quintile_mean_r": sum(c["top_quintile_mean_r_improved"] for c in comparisons),
        "pooled_spearman_delta": pooled_delta, "incremental_discrimination": incremental,
        "high_score_cohort_positive_expectancy": high_positive, "b1_economic_viability": viability,
        "b1_full_quality_gate": full, "catastrophic_folds": catastrophic, "classification": classification,
        "single_next_research_hypothesis": next_hypothesis, "shadow_candidate_frozen": full}


def report(metrics: dict[str, Any]) -> str:
    lines = [f"# {EXPERIMENT}", "", f"Status: **{metrics['run_status']}**. Historical development evidence only.", "",
        "## B0 reproduction gate", "", "```json", json.dumps(base.sanitize(metrics["b0_reproduction"]), indent=2), "```", "",
        "## Fixed S5 executable results", "", "| Model | Fold | Trades | Trades/day | WR | TP-first WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics["fold_metrics"]:
        lines.append(f"| {r['model_id']} | {r['fold']} | {r['trades']} | {r['trades_per_day']:.4f} | {r['realized_wr']:.2%} | {r['tp_first_wr']:.2%} | {r['pf']:.4f} | {r['mean_r']:.4f} | {r['pnl_r']:.2f} | {r['max_dd_r']:.2f} | {r['cost_stress_pf']:.4f} |")
    lines += ["", "## Decision", "", "```json", json.dumps(base.sanitize(metrics["decision"]), indent=2), "```", "",
              "Marginal trades, probability/ranking diagnostics, event-state attribution and complete fold provenance are retained as CSV/JSON artifacts.", "",
              "No threshold, event subset, interaction, window, parameter or calibration was searched. No production artifact was changed."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert len(MACRO_FEATURES) == 8 and len(MODEL_IDS) == 2 and base.THRESHOLD == .75
        assert semantics.SIMULATORS[-1].simulator == "S5" and REPRODUCTION_TOLERANCES["pf"] == .02
        print("SELF_CHECK_PASS"); return 0
    if args.run_dir is None: raise ValueError("--run-dir required")
    run = args.run_dir.resolve()
    operational_before, manifest = preregister(run)
    shutil.copyfile(REFERENCE / "label_definition.json", run / "label_definition.json")
    history, technical = None, None
    drl_trading_v2.DATA_DIR = str(ROOT)
    history, technical = base.prepare_barrier_data()
    history = history.copy().reset_index(drop=True)
    if len(technical) != 31: raise RuntimeError("B0 feature count is not frozen 31")
    incumbent = xgb.XGBClassifier(); incumbent.load_model(base.OPERATIONAL_MODEL)
    if incumbent.get_booster().feature_names != technical: raise RuntimeError("Technical schema differs from operational reference")
    del incumbent
    labels = base.build_execution_aligned_labels(history)
    plans = fold_plan(history, labels)
    history_ns = history["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    broker_ns, macro_map, foundation = load_macro_mapping()
    target = labels["C1_TARGET"].to_numpy(dtype=np.int8)
    scored_parts, provenance, models = [], [], []
    reference_provenance = {r["fold"]: r for r in base.read_json(REFERENCE / "fold_model_provenance.json")["folds"]}
    payloads = []
    for plan in plans:
        name, ti, si = plan["fold"], plan["train"], plan["score"]
        train_macro, score_macro = macro_for(history_ns[ti], broker_ns, macro_map), macro_for(history_ns[si], broker_ns, macro_map)
        x0_train = history.loc[ti, technical].to_numpy(dtype=np.float32)
        x0_score = history.loc[si, technical].to_numpy(dtype=np.float32)
        x1_train = np.column_stack((x0_train, train_macro)).astype(np.float32)
        x1_score = np.column_stack((x0_score, score_macro)).astype(np.float32)
        ref = reference_provenance[name]
        hashes = {"train_timestamp_sha256": base.array_sha256(history_ns[ti]), "score_timestamp_sha256": base.array_sha256(history_ns[si]),
            "x_train_sha256_B0": base.array_sha256(x0_train), "x_score_sha256_B0": base.array_sha256(x0_score),
            "x_train_sha256_B1": base.array_sha256(x1_train), "x_score_sha256_B1": base.array_sha256(x1_score),
            "macro_train_sha256": base.array_sha256(train_macro), "macro_score_sha256": base.array_sha256(score_macro),
            "y_train_sha256_B0": base.array_sha256(target[ti]), "y_train_sha256_B1": base.array_sha256(target[ti])}
        if (hashes["train_timestamp_sha256"] != ref["train_timestamp_sha256"] or hashes["score_timestamp_sha256"] != ref["score_timestamp_sha256"]
                or hashes["x_train_sha256_B0"] != ref["x_train_sha256_C1"] or hashes["x_score_sha256_B0"] != ref["x_score_sha256_C1"]
                or hashes["y_train_sha256_B0"] != ref["c1_label_sha256"]):
            raise RuntimeError(f"B0 frozen cohort/matrix/target mismatch {name}")
        print(f"B0 TRAIN {name} train={len(ti):,} score={len(si):,}", flush=True)
        model = train_model(history, technical, ti, target[ti], None)
        score = score_model(model, history, technical, si, None)
        artifact = run / "models" / f"{MODEL_IDS[0]}_{name}_xgb.json"; artifact.parent.mkdir(exist_ok=True); model.save_model(artifact)
        models.append({"model_id": MODEL_IDS[0], "fold": name, "path": artifact.relative_to(run).as_posix(), "sha256": base.sha256(artifact), "train_rows": len(ti), "target_sha256": hashes["y_train_sha256_B0"]})
        part = history.loc[si, ["TIME_DT","OPEN","HIGH","LOW","CLOSE","ATR","M1_RSI","SPREAD"]].copy()
        part["global_index"], part["fold"], part["fold_code"] = si, name, plan["fold_code"]
        part["feature_bar_time"] = history["TIME_DT"].shift(1).loc[si].to_numpy()
        part["C1_TARGET"] = target[si]; part["C1_NET_R"] = labels["C1_NET_R"].to_numpy()[si]
        part["C1_STRESS_R"] = labels["C1_STRESS_R"].to_numpy()[si]
        part["C1_EXIT_TYPE"] = labels["C1_EXIT_TYPE"].to_numpy()[si]; part["C1_EXIT_INDEX"] = labels["C1_EXIT_INDEX"].to_numpy()[si]
        part[f"score_{MODEL_IDS[0]}"] = score
        scored_parts.append(part)
        provenance.append({"fold": name, "train_start": history["TIME_DT"].iat[int(ti[0])].isoformat(),
            "train_feature_end": history["TIME_DT"].iat[int(ti[-1])].isoformat(), "score_start": history["TIME_DT"].iat[int(si[0])].isoformat(),
            "score_end": history["TIME_DT"].iat[int(si[-1])].isoformat(), "latest_training_label_information_time": pd.Timestamp(plan["latest_label_information_ns"]).isoformat(),
            "train_rows": len(ti), "score_rows": len(si), "training_window_months": base.TRAIN_MONTHS,
            "strict_label_maturity_before_score": plan["latest_label_information_ns"] < history_ns[si[0]],
            **hashes, "parameters_B0": base.FIXED_XGB_PARAMETERS, "parameters_B1": base.FIXED_XGB_PARAMETERS,
            "random_seed_B0": base.RANDOM_STATE, "random_seed_B1": base.RANDOM_STATE})
        payloads.append((plan, train_macro, score_macro))
        del model, x0_train, x0_score, x1_train, x1_score; gc.collect()
    scored = pd.concat(scored_parts, ignore_index=True).sort_values("TIME_DT").reset_index(drop=True)
    b0_metrics, b0_trades, b0_audit = evaluate_one(scored, MODEL_IDS[0])
    p0, r0, d0 = diagnostics(scored.assign(**{f"score_{MODEL_IDS[1]}": scored[f"score_{MODEL_IDS[0]}"]}))
    p0 = [r for r in p0 if r["model_id"] == MODEL_IDS[0]]; r0 = [r for r in r0 if r["model_id"] == MODEL_IDS[0]]; d0 = [r for r in d0 if r["model_id"] == MODEL_IDS[0]]
    reproduction = reproduction_gate(scored, next(r for r in b0_metrics if r["fold"]=="pooled"),
        next(r for r in p0 if r["fold"]=="pooled"), next(r for r in r0 if r["fold"]=="pooled"))
    base.write_json(run / "b0_reproduction.json", reproduction)
    if not reproduction["pass"]:
        base.write_json(run / "metrics.json", {"experiment": EXPERIMENT, "run_status": "FAIL - baseline reproduction failure",
            "foundation_identity": foundation, "b0_reproduction": reproduction, "fold_metrics": b0_metrics,
            "model_training_performed": True, "b1_interpreted": False, "decision": {"classification":"FAIL - baseline reproduction failure"}})
        base.write_csv(run / "fold_metrics.csv", b0_metrics); base.write_csv(run / "trade_ledger.csv", b0_trades)
        raise RuntimeError("B0 material reproduction failure; B1 not trained or interpreted")
    print("B0_REPRODUCTION_PASS; B1 is now permitted", flush=True)
    score_b1 = np.full(len(scored), np.nan, dtype=np.float32)
    for plan, train_macro, score_macro in payloads:
        name, ti, si = plan["fold"], plan["train"], plan["score"]
        print(f"B1 TRAIN {name} train={len(ti):,} score={len(si):,}", flush=True)
        model = train_model(history, technical, ti, target[ti], train_macro)
        values = score_model(model, history, technical, si, score_macro)
        mask = scored["fold"].eq(name).to_numpy(); score_b1[mask] = values
        artifact = run / "models" / f"{MODEL_IDS[1]}_{name}_xgb.json"; model.save_model(artifact)
        models.append({"model_id": MODEL_IDS[1], "fold": name, "path": artifact.relative_to(run).as_posix(), "sha256": base.sha256(artifact), "train_rows": len(ti), "target_sha256": base.array_sha256(target[ti])})
        del model; gc.collect()
    if not np.isfinite(score_b1).all(): raise RuntimeError("Incomplete B1 OOF scores")
    scored[f"score_{MODEL_IDS[1]}"] = score_b1
    b1_metrics, b1_trades, b1_audit = evaluate_one(scored, MODEL_IDS[1])
    probability, ranking, deciles = diagnostics(scored)
    all_metrics = b0_metrics + b1_metrics
    execution = {"ledgers": {MODEL_IDS[0]: b0_trades, MODEL_IDS[1]: b1_trades}, "audits": {MODEL_IDS[0]: b0_audit, MODEL_IDS[1]: b1_audit}}
    identity_rows, marginal = base.marginal_identity(execution)
    scored_macro = macro_for(scored["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64), broker_ns, macro_map)
    attribution = event_attribution(scored, scored_macro)
    final_decision = decision(all_metrics, ranking)
    shadow = {"frozen": False, "new_forward_cutoff": None}
    if final_decision["b1_full_quality_gate"]:
        latest = next(m for m in models if m["model_id"]==MODEL_IDS[1] and m["fold"]==base.FOLDS[-1][0])
        cutoff = base.now_utc()
        shadow = {"frozen": True, "candidate_id": "B1_MACRO_TIMING_SHADOW_V1", "run_id": run.name,
            "model_path": latest["path"], "model_sha256": latest["sha256"], "training_script_sha256": base.sha256(Path(__file__)),
            "baseline_31_feature_schema_sha256": schema_hash(technical), "macro_event_dataset_sha256": EXPECTED_MACRO_DATASET_SHA,
            "event_feature_matrix_sha256": EXPECTED_EVENT_MATRIX_SHA, "feature_schema_39": technical+MACRO_FEATURES,
            "feature_schema_39_sha256": schema_hash(technical+MACRO_FEATURES), "execution_aligned_target_sha256": base.sha256(run/"label_definition.json"),
            "training_window_months": base.TRAIN_MONTHS, "model_parameters": base.FIXED_XGB_PARAMETERS,
            "threshold": base.THRESHOLD, "minimum_entry_rsi": base.MIN_ENTRY_RSI, "excluded_rsi": list(base.EXCLUDED_RSI),
            "tp_atr": semantics.TP_ATR, "sl_atr": semantics.SL_ATR, "hold_minutes": base.EXECUTION_HORIZON_MINUTES,
            "allowed_hours": sorted(semantics.ALLOWED_HOURS), "s5_sha256": base.sha256(base.S5_SOURCE),
            "frozen_at_utc": cutoff, "new_forward_cutoff": cutoff, "production_promotion": False}
        base.write_json(run / "shadow_candidate.json", shadow)
    final_decision["shadow_candidate_frozen"] = shadow["frozen"]
    base.write_csv(run / "fold_metrics.csv", all_metrics)
    base.write_csv(run / "probability_diagnostics.csv", probability); base.write_csv(run / "ranking_diagnostics.csv", ranking)
    base.write_csv(run / "ranking_deciles.csv", deciles); base.write_csv(run / "trade_ledger.csv", b0_trades+b1_trades)
    base.write_csv(run / "trade_identity_comparison.csv", identity_rows or [{"match_type":"none"}])
    base.write_csv(run / "event_state_attribution.csv", attribution)
    base.write_csv(run / "candidates.csv", [{"candidate_id": model, **next(r for r in all_metrics if r["model_id"]==model and r["fold"]=="pooled")} for model in MODEL_IDS])
    base.write_json(run / "fold_model_provenance.json", {"technical_features": technical, "macro_features": MACRO_FEATURES,
        "b0_features": technical, "b1_features": technical+MACRO_FEATURES, "technical_schema_sha256": schema_hash(technical),
        "b1_schema_sha256": schema_hash(technical+MACRO_FEATURES), "folds": provenance, "models": models,
        "foundation": foundation, "reference_run": REFERENCE.name})
    base.save_oof(run / "paired_oof_predictions.npz", scored)
    metrics = {"experiment": EXPERIMENT, "run_status": "PASS internal execution; research_only",
        "foundation_identity": foundation, "b0_reproduction": reproduction, "fold_metrics": all_metrics,
        "probability_diagnostics": probability, "ranking_diagnostics": ranking,
        "marginal_trade_identity": marginal, "event_state_attribution": attribution,
        "execution_audits": execution["audits"], "decision": final_decision,
        "shadow_candidate": shadow, "model_training_performed": True, "strategy_evaluation_performed": True,
        "production_changed": False}
    base.write_json(run / "metrics.json", metrics); (run / "report.md").write_text(report(metrics), encoding="utf-8")
    (run / "models.sha256").write_text("".join(f"{m['sha256']}  {m['path']}\n" for m in models),encoding="utf-8")
    primary = next(m for m in models if m["model_id"]==MODEL_IDS[1] and m["fold"]==base.FOLDS[-1][0])
    (run / "model.sha256").write_text(f"{primary['sha256']}  {primary['path']}\n",encoding="utf-8")
    operational_after = {p.name: base.sha256(p) for p in (base.GEMINI_FILE, base.OPERATIONAL_MODEL)}
    if operational_after != operational_before: raise RuntimeError("Operational artifact changed")
    if inventory(FOUNDATION) != manifest["protected_finalized_runs_before"][FOUNDATION.name]: raise RuntimeError("Foundation changed")
    manifest = base.read_json(run / "manifest.json")
    source_files = base.source_inventory()
    manifest["data"].update({"symbols":["GOLD#"], "data_sources":["repository-local XM CSV","immutable certified macro foundation"],
        "source_files":source_files + [{"path":str((FOUNDATION/"event_features.npz").resolve()),"sha256":base.sha256(FOUNDATION/"event_features.npz"),"retention_status":"immutable finalized Git foundation"}],
        "timezone":"naive broker timestamps for paired technical pipeline; certified EET/EEST-to-UTC mapping for macro features",
        "data_start_utc":history["TIME_DT"].iat[0].isoformat(),"data_end_utc":history["TIME_DT"].iat[-1].isoformat(),
        "train_start_utc":min(r["train_start"] for r in provenance),"train_end_utc":max(r["train_feature_end"] for r in provenance),
        "train_rows":sum(r["train_rows"] for r in provenance),"validation_start_utc":base.FOLDS[0][1].isoformat(),
        "validation_end_utc":base.FOLDS[-1][2].isoformat(),"validation_rows":len(scored),
        "test_start_utc":"not_applicable_no_untouched_test","test_end_utc":"not_applicable_no_untouched_test","test_rows":0,
        "purge_details":"Same prior C1 intersection maturity: legacy 240-row and exact standalone S5 label maturity strictly before fold score start.",
        "embargo_details":"strict label maturity; no additional embargo","raw_snapshot_retained":False,
        "reproducibility_claim":"scripts, fold models, OOF cohort, hashes and immutable macro matrix retained; large original GOLD CSV remains repository-local and hash-identified",
        "mt5_fetch":{"used":False,"not_applicable_reason":"fixed repository-local historical export"},"folds":provenance,
        "scored_oof_path":"paired_oof_predictions.npz","scored_oof_sha256":base.sha256(run/"paired_oof_predictions.npz")})
    manifest["model"].update({"features":technical+MACRO_FEATURES,"feature_count":39,"artifact_path":primary["path"],
        "artifact_sha256":primary["sha256"],"retention_status":"all six paired fold models retained in run","fold_models_trained":len(models),
        "fold_model_inventory":"fold_model_provenance.json","label_definition_sha256":base.sha256(run/"label_definition.json")})
    b1_pooled = next(r for r in all_metrics if r["model_id"]==MODEL_IDS[1] and r["fold"]=="pooled")
    manifest["registry"].update({"parent_or_incumbent":REFERENCE.name,"selected_configuration":final_decision["classification"],
        "trades_per_day":b1_pooled["trades_per_day"],"realized_win_rate":b1_pooled["realized_wr"],"pf":b1_pooled["pf"],
        "mean_r":b1_pooled["mean_r"],"pnl":b1_pooled["pnl_r"],"max_dd":b1_pooled["max_dd_r"],"validator_result":"PENDING"})
    manifest["operational_hashes_after"] = operational_after; manifest["research_decision"] = final_decision; manifest["shadow_candidate"] = shadow
    if shadow["frozen"]: manifest["evidence_status"]["new_forward_cutoff"] = shadow["new_forward_cutoff"]
    for path in [run/"label_definition.json",run/"b0_reproduction.json",run/"fold_metrics.csv",run/"probability_diagnostics.csv",run/"ranking_diagnostics.csv",
                 run/"ranking_deciles.csv",run/"trade_ledger.csv",run/"trade_identity_comparison.csv",run/"event_state_attribution.csv",
                 run/"candidates.csv",run/"fold_model_provenance.json",run/"paired_oof_predictions.npz",run/"models.sha256",run/"model.sha256",run/"metrics.json",run/"report.md"]:
        base.add_artifact(manifest,run,path,path.name)
    for model in models: base.add_artifact(manifest,run,run/model["path"],"trained_fold_model")
    if shadow["frozen"]: base.add_artifact(manifest,run,run/"shadow_candidate.json","frozen_shadow_candidate")
    base.write_json(run/"manifest.json",manifest)
    print(f"COMPLETE classification={final_decision['classification']} shadow={shadow['frozen']} production_changed=false",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
