"""Frozen score-gating research; archived B0 scores only, no training."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "gold_score_ranking_v1"
REFERENCE = "20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1"
REFERENCE_HASH = "592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe"
DAY = 86_400_000_000_000
MIN_HISTORY = 60
CANDIDATES = (
    ("A0_BASELINE_075", None, .75, None),
    ("A1_ROLLING_1D_P95", "1D", None, .95),
    ("A1_ROLLING_1D_P97", "1D", None, .97),
    ("A1_ROLLING_3D_P95", "3D", None, .95),
    ("A1_ROLLING_3D_P97", "3D", None, .97),
    ("A1_ROLLING_5D_P95", "5D", None, .95),
    ("A1_ROLLING_5D_P97", "5D", None, .97),
    ("A2_SESSION_P95", "SESSION", None, .95),
    ("A2_SESSION_P97", "SESSION", None, .97),
    ("A3_3D_F065_P95", "3D", .65, .95),
    ("A3_3D_F065_P97", "3D", .65, .97),
    ("A3_3D_F070_P95", "3D", .70, .95),
    ("A3_3D_F070_P97", "3D", .70, .97),
)
BASELINE = {
    "trades": 689, "wins": 390, "losses": 299,
    "realized_wr": .5660377358490566, "pf": .8247098331219882,
    "mean_r": -.07690539315994348, "pnl_r": -52.98781588720106,
    "max_dd_r": -59.57958015890608, "cost_stress_pf": .7838186120438296,
    "trades_per_day": .26945639421196715,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hash(path, expected):
    require(sha(path) == expected, "Artifact mutation: " + str(path))


def check_safety(before, after):
    require(bool(before) and before == after, "Operational mutation")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def definitions():
    return [dict(candidate_id=name, reference=window, absolute_floor=floor,
                 percentile_threshold=threshold)
            for name, window, floor, threshold in CANDIDATES]


def check_definitions(value):
    require(value == definitions(), "Non-frozen candidate IDs or thresholds")


def execution_spec():
    return {
        "experiment": EXPERIMENT, "candidates": definitions(),
        "score_source": REFERENCE, "score_source_finalized_sha256": REFERENCE_HASH,
        "candidate_count": 13, "candidate_set_frozen": True,
        "candidate_add_delete_or_retune_allowed": False,
        "score_arrays": "paired_oof_predictions.npz:B0_31_technical_fold{number}",
        "score_indices": "paired_oof_predictions.npz:fold{number}_indices",
        "a0_uses_same_archived_scores": True,
        "minimum_prior_scores": 60, "time_basis": "UTC",
        "rolling_reference": "t-window <= prior_timestamp < t",
        "rolling_days": [1, 3, 5], "fold_reset": True,
        "percentile": "count(prior_score <= current_score) / N_prior",
        "insufficient_history": "ineligible; percentile is NaN, not zero or imputed",
        "sessions": {"UTC_S0": [0, 8], "UTC_S1": [8, 16], "UTC_S2": [16, 24]},
        "session_reset": "UTC calendar day x bucket x fold",
        "session_intervals": "start inclusive; end exclusive",
        "session_reference": "same UTC calendar day and bucket; prior_timestamp < t",
        "threshold_comparison": ">=",
        "ties": "<= count; no jitter, random tie break, average rank or interpolation",
        "a3_gate": "score >= absolute_floor AND percentile >= percentile_threshold",
        "a3_reference": "identical to A1 causal rolling 3 calendar days",
        "frozen_execution": ["S5", "spread/cost", "trade eligibility", "TP/SL", "timeout", "risk state"],
        "current_timestamp_group_excluded": True, "model_training": False,
        "entry_gating_only": True, "simulator": "frozen S5",
        "s5_state": "one continuous replay across score folds per candidate, as baseline",
        "adaptive_search": False, "candidate_selection": "none",
        "guardrails": {"pf_min": .80, "mean_r_min": BASELINE["mean_r"] - .03,
                       "stress_pf_min": .75, "fold_trades_min": 10,
                       "fold_wr_min": .45, "fold_pf_min": .70},
        "interesting": {"wr_strict_min": BASELINE["realized_wr"],
                        "trades_per_day_strict_min": BASELINE["trades_per_day"]},
        "strong": {"wr_min": .58, "trades_per_day_min": .40},
        "target": {"wr_min": .60, "trades_per_day_min": .50},
        "production_promotion_requested": False,
    }


def causal_ranks(times, scores, reference):
    """One fold only. Sorted prior scores; equal timestamps enter as a batch."""
    times, scores = np.asarray(times, dtype=np.int64), np.asarray(scores)
    require(times.ndim == scores.ndim == 1 and len(times) == len(scores), "Rank shape")
    require(np.all(times[1:] >= times[:-1]) and np.isfinite(scores).all(), "Rank input")
    require(reference in {"1D", "3D", "5D", "SESSION"}, "Non-frozen window")
    counts = np.zeros(len(times), dtype=np.int64)
    less_equal = np.zeros(len(times), dtype=np.int64)
    ordered, left, start = [], 0, 0
    while start < len(times):
        now = int(times[start])
        lower = ((now // (DAY // 3)) * (DAY // 3) if reference == "SESSION"
                 else now - int(reference[0]) * DAY)
        while left < start and int(times[left]) < lower:
            ordered.pop(bisect.bisect_left(ordered, float(scores[left])))
            left += 1
        end = start + 1
        while end < len(times) and times[end] == now:
            end += 1
        for index in range(start, end):
            counts[index] = len(ordered)
            less_equal[index] = bisect.bisect_right(ordered, float(scores[index]))
        for value in scores[start:end]:
            bisect.insort_right(ordered, float(value))
        start = end
    percentile = np.full(len(times), np.nan, dtype=np.float64)
    valid = counts >= MIN_HISTORY
    percentile[valid] = less_equal[valid] / counts[valid]
    return counts, less_equal, percentile


def eligibility(candidate, scores, ranks):
    require(candidate in CANDIDATES, "Non-frozen candidate")
    scores = np.asarray(scores, dtype=np.float64)
    _, reference, floor, threshold = candidate
    if reference is None:
        return scores >= .75
    counts, _, percentiles = ranks[reference]
    eligible = (counts >= MIN_HISTORY) & (percentiles >= threshold)
    if floor is not None:
        eligible &= scores >= floor
    return eligible


def apply_gate(cohort, gate):
    result = cohort.copy()
    require(len(result) == len(gate), "Gate length")
    raw = np.asarray(gate, dtype=bool)
    times = result["decision_time_api"].to_numpy(dtype="datetime64[ns]")
    gap = np.ones(len(raw), dtype=bool)
    gap[1:] = np.diff(times).astype("timedelta64[s]").astype(np.int64) > 120
    episodes = np.cumsum(raw & (np.r_[False, ~raw[:-1]] | gap)).astype(np.int64) - 1
    episodes[~raw] = -1
    result["raw_signal"], result["raw_episode_id"] = raw, episodes
    return result


def check_baseline(row):
    for name, expected in BASELINE.items():
        require(abs(float(row[name]) - expected) <= 1e-12, "Baseline reproduction: " + name)


def assess(rows):
    pooled = rows[-1]
    reasons = []
    for name, floor in (("pf", .80), ("mean_r", BASELINE["mean_r"] - .03),
                        ("cost_stress_pf", .75)):
        if pooled[name] < floor:
            reasons.append(name)
    for row in rows[:-1]:
        for name, floor in (("trades", 10), ("realized_wr", .45), ("pf", .70)):
            if row[name] < floor:
                reasons.append(row["fold"] + ":" + name)
    wr, frequency = pooled["realized_wr"], pooled["trades_per_day"]
    return {
        "guardrail_failures": reasons, "guardrails_pass": not reasons,
        "interesting": not reasons and wr > BASELINE["realized_wr"] and frequency > BASELINE["trades_per_day"],
        "strong": not reasons and wr >= .58 and frequency >= .40,
        "target": not reasons and wr >= .60 and frequency >= .50,
        "delta_wr": wr - BASELINE["realized_wr"],
        "delta_trades_per_day": frequency - BASELINE["trades_per_day"],
        "frequency_uplift_pct": 100 * (frequency / BASELINE["trades_per_day"] - 1),
    }


def frozen_inputs():
    archive = ROOT / "training_runs" / REFERENCE
    check_hash(archive / "FINALIZED.json", REFERENCE_HASH)
    final = read(archive / "FINALIZED.json")
    for relative, expected in final["file_sha256"].items():
        path = (archive / relative).resolve()
        require(path.is_relative_to(archive), "Archive path")
        check_hash(path, expected)
    helper = load_module(archive / "training_script.py", "ranking_frozen_runner_helpers")
    for path, expected in helper.SOURCE_HASHES.items():
        check_hash(ROOT / path, expected)
    for name, files in ((helper.PARENT, ["training_script.py", "manifest.json"]),
                        (helper.TIMESTAMPS, ["exact_timestamps.npz"])):
        helper.verify_archive(ROOT / "training_runs" / name, files)
    old = load_module(ROOT / "training_runs" / helper.PARENT / "training_script.py", "ranking_parent")
    require(old.xgb.__version__ == "3.2.0", "XGBoost version drift")
    old.drl_trading_v2.DATA_DIR = str(ROOT)
    return archive, helper, old


def formal_run(run):
    run = run.resolve()
    require(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(), "New run required")
    manifest = read(run / "manifest.json")
    require(manifest["experiment_name"] == EXPERIMENT and manifest["status"] == "in_progress"
            and manifest["run_id"] == run.name, "Formal manifest")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").rstrip()
    require(git("rev-parse", "HEAD") == git("rev-parse", "@{u}") == manifest["git_commit"], "Git gate")
    require(sha(run / "training_script.py") == sha(Path(__file__)) == manifest["training_script_sha256"], "Snapshot")
    prefix = run.relative_to(ROOT).as_posix() + "/"
    require(all(line[3:].strip('"').startswith(prefix) for line in git("status", "--porcelain", "--untracked-files=all").splitlines()), "Dirty paths")
    require(not (run / "execution_spec.json").exists(), "Do not retry a run")
    require(read(ROOT / "execution_spec_gold_score_ranking_v1.json") == execution_spec(), "Source execution spec drift")
    archive, helper, old = frozen_inputs()
    before = helper.operational_hashes()
    model_files = [item for item in read(archive / "model_inventory.json") if item["candidate_id"] == "B0_31_technical"]
    require(len(model_files) == 3, "Exactly three archived B0 models")
    helper.write_json(run / "execution_spec.json", execution_spec())
    helper.write_json(run / "operational_safety_pre.json", before)
    (run / "validator_script.py").write_bytes((ROOT / "validate_gold_score_ranking_v1.py").read_bytes())
    try:
        history, _, folds, identities = helper.reconstruct(old)
        helper.write_json(run / "identity_audit.json", identities)
        evidence, frames, all_gates = {}, [], {item[0]: [] for item in CANDIDATES}
        with np.load(archive / "paired_oof_predictions.npz", allow_pickle=False) as predictions, np.load(
            ROOT / "training_runs" / helper.TIMESTAMPS / "exact_timestamps.npz", allow_pickle=False
        ) as timestamps:
            for number, name, _, indices in folds:
                require(np.array_equal(indices, predictions[f"fold{number}_indices"]), "Frozen score indices")
                scores = predictions[f"B0_31_technical_fold{number}"]
                require(scores.dtype == np.float32 and len(scores) == len(indices)
                        and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), "Archived scores")
                times = timestamps[f"fold{number}_score_utc_ns"]
                frame = history.loc[indices, helper.PRICE_COLUMNS].copy()
                frame["buy_prob"], frame["sell_prob"] = scores, np.float32(0)
                frames.append(frame)
                key = f"fold{number}"
                evidence[key + "_scores"], evidence[key + "_indices"], evidence[key + "_utc_ns"] = scores, indices, times
                ranks = {reference: causal_ranks(times, scores, reference) for reference in ("1D", "3D", "5D", "SESSION")}
                for reference, arrays in ranks.items():
                    for suffix, values in zip(("count", "le", "percentile"), arrays):
                        evidence[f"{key}_{reference}_{suffix}"] = values
                for candidate in CANDIDATES:
                    gate = eligibility(candidate, scores, ranks)
                    all_gates[candidate[0]].append(gate)
                    evidence[key + "_" + candidate[0]] = gate
        np.savez_compressed(run / "ranking_evidence.npz", **evidence)
        combined = pd.concat(frames, ignore_index=True)
        results, ledgers, candidates = {}, [], []
        for candidate in CANDIDATES:
            name = candidate[0]
            cohort = old.semantics.finalize_cohort(combined, name, offset_hours=0)
            cohort = apply_gate(cohort, np.concatenate(all_gates[name]))
            trades, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
            rows = helper.metric_rows(old, trades, name)
            if name == CANDIDATES[0][0]:
                check_baseline(rows[-1])
            results[name] = {"metrics": rows, "assessment": assess(rows)}
            pooled = rows[-1]
            evaluation = assess(rows)
            verdict = ("rejected_guardrails" if not evaluation["guardrails_pass"] else
                       "interesting_research_only" if evaluation["interesting"] else "not_interesting")
            candidates.append({**pooled, "pnl": pooled["pnl_r"], "max_dd": pooled["max_dd_r"],
                               **evaluation, "qualification_verdict": verdict})
            ledgers.extend({**trade, "candidate_id": name} for trade in trades)
        check_definitions(definitions())
        metrics = {"results": results, "baseline_reproduction": True, "model_training": False,
                   "adaptive_search": False, "selected_candidate": None,
                   "production_promotion_requested": False, "classification": "historical_development_only"}
        helper.write_json(run / "metrics.json", metrics)
        helper.write_csv(run / "candidates.csv", candidates)
        helper.write_csv(run / "fold_metrics.csv", [row for value in results.values() for row in value["metrics"]])
        helper.write_csv(run / "trade_ledger.csv", ledgers)
        report = "# GOLD score-ranking research\n\nResearch only; no candidate selected or promoted.\n\n"
        report += "Primary objectives: realized WR and trades/day must BOTH improve; economic guardrails reject unsafe candidates.\n\n"
        report += "```json\n" + json.dumps(read(run / "metrics.json"), indent=2) + "\n```\n"
        (run / "report.md").write_text(report, encoding="utf-8")
        provenance = {"reference_run": REFERENCE, "reference_finalized_sha256": REFERENCE_HASH,
                      "predictions_sha256": sha(archive / "paired_oof_predictions.npz"),
                      "models": model_files, "source_hashes": helper.SOURCE_HASHES}
        helper.write_json(run / "source_provenance.json", provenance)
        source_manifest = read(archive / "manifest.json")
        manifest["data"] = source_manifest["data"]
        manifest["data"].update(symbols=["GOLD#"], data_sources=["Archived B0 scores and frozen GOLD reconstruction"],
            source_files=[{"path": (archive / "FINALIZED.json").relative_to(ROOT).as_posix(),
                           "sha256": REFERENCE_HASH, "retention_status": "retained_in_prior_finalized_run"}])
        manifest["model"] = {"trained": False, "not_applicable_reason": "Reuse exact archived B0 scores/models; no fitting",
                             "features": helper.BASE_FEATURES, "feature_count": 31}
        manifest["random_seed_note"] = "No randomness or retraining; inherited B0 model seed 42"
        manifest["search"] = {"performed": False, "not_applicable_reason": "Exactly 13 predefined gates; no adaptive search",
                              "candidate_results_file": "candidates.csv", "predefined_candidates": definitions()}
        manifest["promotion"] = {"requested": False, "replacement_authorized": False,
                                 "operational_artifact_changed": False, "gate_result": "research_only"}
        manifest["registry"] = {"parent_or_incumbent": REFERENCE, "selected_configuration": "13 frozen gates; none selected",
                                "validator_result": "PENDING", "trades_per_day": "see candidates.csv",
                                "realized_win_rate": "see candidates.csv", "pf": "see candidates.csv",
                                "mean_r": "see candidates.csv", "pnl": "see candidates.csv", "max_dd": "see candidates.csv"}
        after = helper.operational_hashes()
        check_safety(before, after)
        for item in model_files:
            check_hash(archive / item["path"], item["sha256"])
        helper.write_json(run / "operational_safety_post.json", after)
        manifest["artifacts"] = [{"path": path.relative_to(run).as_posix(), "sha256": sha(path),
                                  "kind": "research_evidence", "retention_status": "stored_in_run_directory"}
                                 for path in sorted(run.rglob("*")) if path.is_file()
                                 and path.name not in {"manifest.json", "stdout.log"}]
        helper.write_json(run / "manifest.json", manifest)
    except Exception as error:
        helper.write_json(run / "failure.json", {"status": "FAIL", "error": str(error)})
        raise
    finally:
        after = helper.operational_hashes()
        helper.write_json(run / "operational_safety_post.json", after)
        check_safety(before, after)
    print("FROZEN_SCORE_RANKING_COMPLETED")


def self_test():
    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Invalid input accepted")
    scores = np.full(65, .5, dtype=np.float32)
    times = np.arange(65, dtype=np.int64) * 60_000_000_000
    ranks = causal_ranks(times, scores, "1D")
    assert ranks[0][0] == 0 and ranks[0][60] == 60
    assert np.isnan(ranks[2][:60]).all() and ranks[2][60] == 1
    mutated = scores.copy()
    mutated[63:] = 0
    altered = causal_ranks(times, mutated, "1D")
    assert np.array_equal(ranks[2][:63], altered[2][:63], equal_nan=True)
    duplicate = causal_ranks(np.array([0, 0, 1]), np.array([.1, .9, .5]), "1D")
    assert duplicate[0].tolist() == [0, 0, 2] and duplicate[1].tolist() == [0, 0, 1]
    sparse = np.array([0, 1, 3, 5, 6], dtype=np.int64) * DAY
    for window, expected in (("1D", [0, 1, 0, 0, 1]), ("3D", [0, 1, 2, 1, 2]), ("5D", [0, 1, 2, 3, 3])):
        assert causal_ranks(sparse, np.ones(5), window)[0].tolist() == expected
    sessions = np.array([0, 1, DAY // 3, DAY // 3 + 1, DAY, DAY + 1])
    assert causal_ranks(sessions, np.ones(6), "SESSION")[0].tolist() == [0, 1, 0, 1, 0, 1]
    assert causal_ranks(times, scores, "1D")[0][0] == 0  # Fresh fold call.
    assert eligibility(CANDIDATES[0], np.array([.749, .75, .751]), {}).tolist() == [False, True, True]
    assert not eligibility(CANDIDATES[1], scores, {"1D": ranks})[:60].any()
    check_baseline(BASELINE)
    rejects(lambda: check_baseline({**BASELINE, "trades": 690}))
    rejects(lambda: check_definitions(definitions() + [definitions()[0]]))
    bad = definitions()
    bad[0]["absolute_floor"] = .74
    rejects(lambda: check_definitions(bad))
    rejects(lambda: causal_ranks(times, scores, "2D"))
    rejects(lambda: check_safety({"model": "a"}, {"model": "b"}))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model"
        path.write_bytes(b"frozen")
        expected = sha(path)
        path.write_bytes(b"changed")
        rejects(lambda: check_hash(path, expected))
    # Test the frozen, pure metric aggregation helper with synthetic trades only.
    helper = load_module(ROOT / "training_runs" / REFERENCE / "training_script.py", "ranking_test_helper")
    from types import SimpleNamespace
    old = SimpleNamespace(FOLDS=[("a", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))],
                          core=SimpleNamespace(fold_days=lambda s, e: (e - s).days),
                          trade_metrics=lambda t, d: {"trades": len(t), "pnl_r": sum(x["net_r"] for x in t), "trades_per_day": len(t) / d})
    pooled = helper.metric_rows(old, [{"entry_time_api": "2020-01-01", "net_r": 1},
                                     {"entry_time_api": "2020-01-02", "net_r": -.5}], "test")[-1]
    assert pooled["pnl_r"] == .5 and pooled["trades_per_day"] == 1
    assert read(ROOT / "execution_spec_gold_score_ranking_v1.json") == execution_spec()
    # Exact P97, both A3 floors, and the AND requirement.
    for level, le_count in ((.95, 95), (.97, 97)):
        values = np.r_[np.full(le_count, .4), np.full(100 - le_count, .9), .7]
        ranked = causal_ranks(np.arange(101), values, "3D")
        assert ranked[2][-1] == level
        for candidate in CANDIDATES[9:]:
            assert bool(eligibility(candidate, values, {"3D": ranked})[-1]) == (level >= candidate[3])
    for candidate in CANDIDATES[9:]:
        for value, expected in ((candidate[2] - .001, False), (candidate[2], True)):
            values = np.r_[np.full(60, .4), value]
            ranked = causal_ranks(np.arange(61), values, "3D")
            assert bool(eligibility(candidate, values, {"3D": ranked})[-1]) == expected
        values = np.r_[np.full(60, .9), .8]
        assert not eligibility(candidate, values, {"3D": causal_ranks(np.arange(61), values, "3D")})[-1]
    # Every UTC bucket and midnight resets even after sufficient prior scores.
    for boundary in (DAY // 3, 2 * DAY // 3, DAY):
        utc = np.r_[boundary - 60 + np.arange(60), boundary, boundary + 1]
        ranked = causal_ranks(utc, np.ones(62), "SESSION")
        assert ranked[0][-2:].tolist() == [0, 1]
        assert np.isnan(ranked[2][-2:]).all()
    assert len(CANDIDATES) == 13
    boundary_scores = np.r_[np.full(57, .4), np.full(3, .6), .5]
    boundary_ranks = causal_ranks(np.arange(61), boundary_scores, "1D")
    assert boundary_ranks[1][-1] == 57 and boundary_ranks[2][-1] == .95
    assert eligibility(CANDIDATES[1], boundary_scores, {"1D": boundary_ranks})[-1]
    assert not eligibility(CANDIDATES[2], boundary_scores, {"1D": boundary_ranks})[-1]
    for score, expected in ((.649, False), (.65, True), (.70, True)):
        values = np.r_[np.full(60, .4), score]
        reference = causal_ranks(np.arange(61), values, "3D")
        assert bool(eligibility(CANDIDATES[9], values, {"3D": reference})[-1]) == expected
    # Both folds start cold even if a prior fold already had sufficient history.
    assert np.isnan(causal_ranks(np.arange(5), np.ones(5), "3D")[2]).all()
    print("SELF_TEST_PASS: causal future_mutation windows sessions baseline candidates thresholds mutation pooled ties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run_dir is None:
        parser.error("--run-dir required unless --self-test")
    else:
        formal_run(args.run_dir)


if __name__ == "__main__":
    main()
