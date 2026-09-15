"""Frozen anchored episode expansion; archived B0 scores only, no training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "gold_anchored_episode_expansion_v1"
REFERENCE = "20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1"
REFERENCE_HASH = "592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe"
CANDIDATES = (
    ("A0_BASELINE_075", 0, .75, None),
    ("E1_5M_F070", 5, .70, None),
    ("E2_5M_F072", 5, .72, None),
    ("E3_10M_F070", 10, .70, None),
    ("E4_10M_F072", 10, .72, None),
    ("C1_5M_F070_DECAY002", 5, .70, .02),
    ("C2_5M_F072_DECAY002", 5, .72, .02),
    ("C3_10M_F070_DECAY002", 10, .70, .02),
    ("C4_10M_F072_DECAY002", 10, .72, .02),
)
MINUTE_NS = 60_000_000_000

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
    return [dict(candidate_id=name, window_minutes=window, absolute_floor=floor,
                 max_score_decay=decay) for name, window, floor, decay in CANDIDATES]


def check_definitions(value):
    require(value == definitions(), "Non-frozen candidate IDs or thresholds")


def execution_spec():
    return {
        "experiment": "gold_anchored_episode_expansion_v1", "candidates": definitions(),
        "candidate_count": 9, "candidate_set_frozen": True,
        "score_source": REFERENCE, "score_source_finalized_sha256": REFERENCE_HASH,
        "score_arrays": "paired_oof_predictions.npz:B0_31_technical_fold{number}",
        "score_indices": "paired_oof_predictions.npz:fold{number}_indices",
        "same_archived_scores_all_candidates": True, "model_training": False,
        "entry_gating_only": True,
        "anchor_threshold": .75, "anchor_comparison": "score >= 0.75",
        "secondary_window": "0 < current_timestamp - anchor_timestamp <= window_minutes",
        "timestamp_basis": "archived UTC nanoseconds; elapsed calendar minutes",
        "fold_reset": True, "row_order": "archived index order; chronological timestamps",
        "anchor_update": "last anchor in timestamp batch becomes active after batch completes",
        "same_timestamp_authorization": False,
        "anchor_score": "exact score of opening anchor; no rounding",
        "secondary_score_range": "absolute_floor <= score < 0.75",
        "decay_gate": "score >= anchor_score - max_score_decay when specified",
        "secondary_quota": 1, "quota_consumption": "only successful S5 open_trade",
        "blocked_entry_consumes_quota": False,
        "episode_end": "secondary executes OR window expires OR later anchor replaces episode",
        "secondary_opens_or_extends_episode": False,
        "anchor_entry_eligibility": "unchanged baseline score >= 0.75; independent of secondary quota",
        "execution": "same frozen S5 simulate code; private entry-gate/open notification bindings only",
        "execution_state": "continuous across combined chronological score folds per candidate",
        "frozen_execution": ["spread/cost", "position occupancy", "risk", "TP/SL", "timeout"],
        "adaptive_search": False, "candidate_selection": "none", "cli_candidate_overrides": False,
        "guardrails": {"pf_min": .80, "mean_r_min": BASELINE["mean_r"] - .03,
                       "stress_pf_min": .75, "fold_trades_min": 10,
                       "fold_wr_min": .45, "fold_pf_min": .70},
        "interesting": {"wr_strict_min": BASELINE["realized_wr"],
                        "trades_per_day_strict_min": BASELINE["trades_per_day"]},
        "strong": {"wr_min": .58, "trades_per_day_min": .40},
        "target": {"wr_min": .60, "trades_per_day_min": .50},
        "strong_target_require_guardrails": True, "production_promotion_requested": False,
        "secondary_diagnostics": "actual executed secondary trades; net_r > 0 wins; others losses; full fold/calendar-day denominators; zero trades WR null",
    }


def episode_eligibility(times, scores, candidate):
    """One fold; anchors are published only after each timestamp batch."""
    require(candidate in CANDIDATES, "Non-frozen candidate")
    times, values = np.asarray(times), np.asarray(scores, dtype=np.float64)
    require(times.ndim == values.ndim == 1 and len(times) == len(values), "Episode shape")
    require(np.issubdtype(times.dtype, np.integer) and np.all(times[1:] >= times[:-1])
            and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Episode input")
    anchors = values >= .75
    prior = np.full(len(times), -1, dtype=np.int64)
    active, start = -1, 0
    while start < len(times):
        end = start + 1
        while end < len(times) and times[end] == times[start]:
            end += 1
        prior[start:end] = active
        opened = np.flatnonzero(anchors[start:end])
        if len(opened):
            active = start + int(opened[-1])
        start = end
    secondary = np.zeros(len(times), dtype=bool)
    _, minutes, floor, decay = candidate
    for i in np.flatnonzero((prior >= 0) & ~anchors):
        j = prior[i]
        secondary[i] = (0 < int(times[i]) - int(times[j]) <= minutes * MINUTE_NS
                        and values[i] >= floor
                        and (decay is None or values[i] >= values[j] - decay))
    return anchors, prior, secondary


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


def replay_episodes(semantics, cohort, anchors, prior, secondary, fold_ids):
    """Run untouched S5 code with private entry hooks; never patch shared modules."""
    require(len(cohort) == len(anchors) == len(prior) == len(secondary) == len(fold_ids), "Replay shape")
    cohort = apply_gate(cohort, anchors | secondary)
    allowed = np.zeros(len(cohort), dtype=bool)
    executed = np.zeros(len(cohort), dtype=bool)
    spent = set()

    def eligible(row, definition, use_actual_utc_session=False):
        index = int(row.name)
        key = (int(fold_ids[index]), int(prior[index]))
        allowed[index] = bool(anchors[index] or (secondary[index] and key not in spent))
        if not allowed[index]:
            return False, "below_threshold"
        return semantics.base_row_eligibility(row, definition, use_actual_utc_session)

    def opened(state, definition, frame, index):
        semantics.open_trade(state, definition, frame, index)
        require(state["position"] is not None and state["position"]["entry_index"] == index, "S5 open notification")
        executed[index] = True
        if not anchors[index]:
            key = (int(fold_ids[index]), int(prior[index]))
            require(secondary[index] and key not in spent, "Secondary quota")
            spent.add(key)

    bindings = dict(semantics.simulate.__globals__)
    bindings.update(base_row_eligibility=eligible, open_trade=opened)
    simulate = FunctionType(semantics.simulate.__code__, bindings,
                            semantics.simulate.__name__, semantics.simulate.__defaults__,
                            semantics.simulate.__closure__)
    trades, _ = simulate(cohort, semantics.SIMULATORS[-1])
    for trade in trades:
        i = int(trade["entry_index"])
        trade["entry_kind"] = "anchor" if anchors[i] else "secondary"
        trade["score_fold"] = int(fold_ids[i])
        trade["anchor_row_index"] = i if anchors[i] else int(prior[i])
    require(int(executed.sum()) == len(trades), "Executed trade count")
    return trades, allowed, executed


def secondary_metrics(trades, windows):
    result = []
    total_days = sum((pd.Timestamp(end) - pd.Timestamp(start)).days for _, start, end in windows)
    for name, start, end in [*windows, ("pooled", None, None)]:
        selected = trades if name == "pooled" else [t for t in trades
            if pd.Timestamp(start) <= pd.Timestamp(t["entry_time_api"]) < pd.Timestamp(end)]
        seconds = [t for t in selected if t["entry_kind"] == "secondary"]
        days = total_days if name == "pooled" else (pd.Timestamp(end) - pd.Timestamp(start)).days
        wins = sum(float(t["net_r"]) > 0 for t in seconds)
        result.append({"fold": name, "anchor_trade_count": len(selected) - len(seconds),
                       "secondary_trade_count": len(seconds), "secondary_wins": wins,
                       "secondary_losses": len(seconds) - wins,
                       "secondary_realized_wr": wins / len(seconds) if seconds else None,
                       "secondary_trades_per_day": len(seconds) / days})
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
    helper = load_module(archive / "training_script.py", "episode_frozen_runner_helpers")
    for path, expected in helper.SOURCE_HASHES.items():
        check_hash(ROOT / path, expected)
    for name, files in ((helper.PARENT, ["training_script.py", "manifest.json"]),
                        (helper.TIMESTAMPS, ["exact_timestamps.npz"])):
        helper.verify_archive(ROOT / "training_runs" / name, files)
    old = load_module(ROOT / "training_runs" / helper.PARENT / "training_script.py", "episode_parent")
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
    require(read(ROOT / "execution_spec_gold_anchored_episode_expansion_v1.json") == execution_spec(), "Source execution spec drift")
    archive, helper, old = frozen_inputs()
    before = helper.operational_hashes()
    model_files = [item for item in read(archive / "model_inventory.json") if item["candidate_id"] == "B0_31_technical"]
    require(len(model_files) == 3, "Exactly three archived B0 models")
    helper.write_json(run / "execution_spec.json", execution_spec())
    helper.write_json(run / "operational_safety_pre.json", before)
    (run / "validator_script.py").write_bytes((ROOT / "validate_gold_anchored_episode_expansion_v1.py").read_bytes())
    try:
        history, _, folds, identities = helper.reconstruct(old)
        helper.write_json(run / "identity_audit.json", identities)
        evidence, frames, all_gates, fold_ids = {}, [], {item[0]: [] for item in CANDIDATES}, []
        offset = 0
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
                fold_ids.extend([number] * len(scores))
                for candidate in CANDIDATES:
                    arrays = episode_eligibility(times, scores, candidate)
                    for suffix, values in zip(("anchor", "prior", "secondary"), arrays):
                        evidence[f"{key}_{candidate[0]}_{suffix}"] = values
                    anchors, prior, secondary = arrays
                    all_gates[candidate[0]].append((anchors, np.where(prior >= 0, prior + offset, -1), secondary))
                offset += len(scores)
        combined = pd.concat(frames, ignore_index=True)
        results, ledgers, candidates = {}, [], []
        for candidate in CANDIDATES:
            name = candidate[0]
            cohort = old.semantics.finalize_cohort(combined, name, offset_hours=0)
            anchors, prior, secondary = [np.concatenate([part[i] for part in all_gates[name]]) for i in range(3)]
            trades, allowed, executed = replay_episodes(old.semantics, cohort, anchors, prior, secondary, fold_ids)
            evidence[name + "_allowed"] = allowed
            evidence[name + "_executed"] = executed
            rows = helper.metric_rows(old, trades, name)
            if name == CANDIDATES[0][0]:
                check_baseline(rows[-1])
            for row, diagnostic in zip(rows, secondary_metrics(trades, helper.FOLD_WINDOWS)):
                row.update({key: value for key, value in diagnostic.items() if key != "fold"})
            results[name] = {"metrics": rows, "assessment": assess(rows)}
            pooled = rows[-1]
            evaluation = assess(rows)
            verdict = ("rejected_guardrails" if not evaluation["guardrails_pass"] else
                       "interesting_research_only" if evaluation["interesting"] else "not_interesting")
            candidates.append({**pooled, "pnl": pooled["pnl_r"], "max_dd": pooled["max_dd_r"],
                               **evaluation, "qualification_verdict": verdict})
            ledgers.extend({**trade, "candidate_id": name} for trade in trades)
        np.savez_compressed(run / "episode_evidence.npz", **evidence)
        check_definitions(definitions())
        metrics = {"results": results, "baseline_reproduction": True, "model_training": False,
                   "adaptive_search": False, "selected_candidate": None,
                   "production_promotion_requested": False, "classification": "historical_development_only"}
        helper.write_json(run / "metrics.json", metrics)
        helper.write_csv(run / "candidates.csv", candidates)
        helper.write_csv(run / "fold_metrics.csv", [row for value in results.values() for row in value["metrics"]])
        helper.write_csv(run / "trade_ledger.csv", ledgers)
        report = "# GOLD anchored episode expansion research\n\nResearch only; no candidate selected or promoted.\n\n"
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
        manifest["search"] = {"performed": False, "not_applicable_reason": "Exactly 9 predefined gates; no adaptive search",
                              "candidate_results_file": "candidates.csv", "predefined_candidates": definitions()}
        manifest["promotion"] = {"requested": False, "replacement_authorized": False,
                                 "operational_artifact_changed": False, "gate_result": "research_only"}
        manifest["registry"] = {"parent_or_incumbent": REFERENCE, "selected_configuration": "9 frozen gates; none selected",
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
    print("FROZEN_ANCHORED_EPISODE_EXPANSION_COMPLETED")


def self_test():
    assert read(ROOT / "execution_spec_gold_anchored_episode_expansion_v1.json") == execution_spec()
    assert len(CANDIDATES) == 9
    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Invalid input accepted")
    # Duplicate anchors: last archived row, strictly earlier timestamps only.
    times = np.array([0, 0, 0, 1, 2, 3], dtype=np.int64) * MINUTE_NS
    scores = np.array([.80, .75, .74, .74, .71, .74])
    a, p, e = episode_eligibility(times, scores, CANDIDATES[5])
    assert p.tolist() == [-1, -1, -1, 1, 1, 1]
    assert e.tolist() == [False, False, False, True, False, True]
    changed = scores.copy()
    changed[-1] = .99
    for left, right in zip((a, p, e), episode_eligibility(times, changed, CANDIDATES[5])):
        assert np.array_equal(left[:-1], right[:-1])
    # Every definition: inclusive window/floor, strict expiry and optional decay.
    for candidate in CANDIDATES[1:]:
        floor = candidate[2] if candidate[3] is None else max(candidate[2], .75 - candidate[3])
        t = np.array([0, candidate[1] * MINUTE_NS, candidate[1] * MINUTE_NS + 1])
        assert episode_eligibility(t, [.75, floor, floor], candidate)[2].tolist() == [False, True, False]
        assert not episode_eligibility(np.array([0, 1]), [.75, floor - .0001], candidate)[2][1]
    # Secondary does not extend time or create an episode; later anchors replace it.
    assert not episode_eligibility(np.array([0, 1, 2]), [.74, .74, .74], CANDIDATES[1])[2].any()
    assert episode_eligibility(np.array([0, 4, 6]) * MINUTE_NS, [.75, .74, .74], CANDIDATES[1])[2].tolist() == [False, True, False]
    assert episode_eligibility(np.array([0, 1, 2]), [.75, .90, .74], CANDIDATES[5])[2].tolist() == [False, False, False]
    assert not episode_eligibility(np.array([4, 5]) * MINUTE_NS, [.74, .74], CANDIDATES[1])[2].any()
    assert not episode_eligibility(times, scores, CANDIDATES[0])[2].any()
    rejects(lambda: episode_eligibility(np.array([1, 0]), [.75, .74], CANDIDATES[1]))
    rejects(lambda: episode_eligibility(np.array([0]), [np.nan], CANDIDATES[1]))
    rejects(lambda: episode_eligibility(np.array([0]), [.75], ("extra", 5, .70, None)))
    # Synthetic S5 replay only: no market loading, reconstruction, or model fitting.
    import gold_gemini_execution_semantics_v1 as s5
    day = pd.Timestamp("2020-01-06")
    while day.dayofweek not in s5.ALLOWED_WEEKDAYS:
        day += pd.Timedelta(days=1)
    day += pd.Timedelta(hours=min(s5.ALLOWED_HOURS))
    values = np.array([.75, .74, .74, .74, .74])
    frame = pd.DataFrame({"TIME_DT": pd.date_range(day, periods=5, freq="min"),
        "OPEN": 100., "HIGH": [100., 100., 110., 110., 110.], "LOW": 100.,
        "CLOSE": 100., "ATR": 1., "M1_RSI": 50., "SPREAD": 1.,
        "buy_prob": values, "sell_prob": 0.})
    cohort = s5.finalize_cohort(frame, "SYNTHETIC", 0)
    arrays = episode_eligibility(np.arange(5) * MINUTE_NS, values, CANDIDATES[1])
    original = (s5.simulate, s5.open_trade, s5.base_row_eligibility)
    trades, allowed, executed = replay_episodes(s5, cohort, *arrays, np.ones(5, dtype=int))
    assert [t["entry_index"] for t in trades] == [0, 3]
    assert [t["entry_kind"] for t in trades] == ["anchor", "secondary"]
    assert allowed.tolist() == [True, True, True, True, False]
    assert executed.tolist() == [True, False, False, True, False]
    # RSI/spread blocking must not consume the quota.
    blocked = cohort.copy()
    blocked.loc[0, "M1_RSI"] = 0
    blocked.loc[1, "effective_spread_points"] = 1e6
    blocked.loc[2, "M1_RSI"] = 0
    trades, allowed, executed = replay_episodes(s5, blocked, *arrays, np.ones(5, dtype=int))
    assert [t["entry_index"] for t in trades] == [3]
    assert allowed[:4].all() and not allowed[4]
    # Risk rejection preserves the unconsumed secondary quota.
    losing = cohort.copy()
    losing.loc[0, "LOW"] = 90.
    losing.loc[1:, "HIGH"] = 110.
    trades, allowed, executed = replay_episodes(s5, losing, *arrays, np.ones(5, dtype=int))
    assert trades[0]["net_r"] < 0 and not executed[1]
    assert allowed[1]
    # A cold next fold has no inherited episode, but its prices still close the
    # existing position from the preceding fold (S5 state remains continuous).
    first = episode_eligibility(np.arange(2) * MINUTE_NS, values[:2], CANDIDATES[1])
    second = episode_eligibility(np.arange(2, 5) * MINUTE_NS, values[2:], CANDIDATES[1])
    cold_arrays = tuple(np.concatenate((first[i], second[i])) for i in range(3))
    trades, _, executed = replay_episodes(s5, cohort, *cold_arrays, np.array([1, 1, 2, 2, 2]))
    assert len(trades) == 1 and trades[0]["entry_index"] == 0 and trades[0]["exit_index"] == 2
    assert not executed[2:].any()
    # An executed secondary ends only its own episode; a new anchor renews it.
    renewed = cohort.copy()
    renewed.loc[:, "HIGH"] = 110.
    renewed.loc[[0, 2], "M1_RSI"] = 0
    renewed_values = np.array([.75, .74, .75, .74, .74])
    renewed_arrays = episode_eligibility(np.arange(5) * MINUTE_NS, renewed_values, CANDIDATES[1])
    trades, _, executed = replay_episodes(s5, renewed, *renewed_arrays, np.ones(5, dtype=int))
    assert [t["entry_index"] for t in trades] == [1, 3]
    assert not executed[4]
    # A0 executes exactly the original synthetic S5 ledger (excluding diagnostics).
    baseline = episode_eligibility(np.arange(5) * MINUTE_NS, values, CANDIDATES[0])
    actual, _, _ = replay_episodes(s5, cohort, *baseline, np.ones(5, dtype=int))
    expected, _ = s5.simulate(cohort, s5.SIMULATORS[-1])
    for trade in actual:
        for key in ("entry_kind", "score_fold", "anchor_row_index"):
            trade.pop(key)
    assert actual == expected
    assert original == (s5.simulate, s5.open_trade, s5.base_row_eligibility)
    synthetic = [{"entry_kind": "anchor", "entry_time_api": "2020-01-01", "net_r": 1.},
                 {"entry_kind": "secondary", "entry_time_api": "2020-01-02", "net_r": -.5}]
    rows = secondary_metrics(synthetic, [("a", "2020-01-01", "2020-01-03"), ("b", "2020-01-03", "2020-01-05")])
    assert rows[-1]["anchor_trade_count"] == rows[-1]["secondary_trade_count"] == 1
    assert rows[-1]["secondary_losses"] == 1 and rows[-1]["secondary_trades_per_day"] == .25
    assert rows[1]["secondary_realized_wr"] is None
    print("SELF_TEST_PASS: frozen definitions, causal episodes, timestamp batches, windows, decay, fold reset, executed-only quota, S5 occupancy/spread/RSI, baseline, secondary diagnostics")


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
