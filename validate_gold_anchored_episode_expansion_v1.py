"""Independent episode reconstruction and S5 replay; never imports the runner."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
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

BASELINE = {'trades': 689,
 'wins': 390,
 'losses': 299,
 'realized_wr': 0.5660377358490566,
 'pf': 0.8247098331219882,
 'mean_r': -0.07690539315994348,
 'pnl_r': -52.98781588720106,
 'max_dd_r': -59.57958015890608,
 'cost_stress_pf': 0.7838186120438296,
 'trades_per_day': 0.26945639421196715}


def ensure(condition, label):
    if not condition:
        raise ValueError(label)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def intact(path, expected):
    ensure(digest(path) == expected, "artifact_mutation:" + str(path))


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, content):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(content), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def import_frozen(path, name):
    descriptor = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def expected_candidates():
    return [dict(candidate_id=name, window_minutes=window, absolute_floor=floor,
                 max_score_decay=decay) for name, window, floor, decay in CANDIDATES]


def candidates_valid(entries):
    ensure(entries == expected_candidates(), "candidate_definitions")


def expected_spec():
    return {
        "experiment": "gold_anchored_episode_expansion_v1", "candidates": expected_candidates(),
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
    """Independent strict-time anchor lookup using searchsorted, not runner batches."""
    ensure(candidate in CANDIDATES, "candidate_definition")
    times, values = np.asarray(times), np.asarray(scores, dtype=np.float64)
    ensure(times.ndim == values.ndim == 1 and len(times) == len(values), "episode_shape")
    ensure(np.issubdtype(times.dtype, np.integer) and np.all(times[1:] >= times[:-1])
           and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "episode_input")
    anchors = values >= .75
    locations = np.flatnonzero(anchors)
    slots = np.searchsorted(times[locations], times, side="left") - 1
    prior = np.full(len(times), -1, dtype=np.int64)
    valid = slots >= 0
    prior[valid] = locations[slots[valid]]
    secondary = np.zeros(len(times), dtype=bool)
    for i in range(len(times)):
        p = int(prior[i])
        if p < 0 or anchors[i]:
            continue
        elapsed = int(times[i]) - int(times[p])
        secondary[i] = (elapsed > 0 and elapsed <= candidate[1] * MINUTE_NS
                        and values[i] >= candidate[2])
        if candidate[3] is not None:
            secondary[i] &= values[i] >= values[p] - candidate[3]
    return anchors, prior, secondary


def gate_cohort(cohort, flags):
    result = cohort.copy()
    ensure(len(result) == len(flags), "gate_shape")
    flags = np.asarray(flags, dtype=bool)
    times = result.decision_time_api.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    new_episode = flags.copy()
    if len(flags) > 1:
        new_episode[1:] &= (~flags[:-1]) | (np.diff(times) // 1_000_000_000 > 120)
    numbers = np.cumsum(new_episode, dtype=np.int64) - 1
    numbers[~flags] = -1
    result["raw_signal"] = flags
    result["raw_episode_id"] = numbers
    return result


def replay_episodes(semantics, cohort, anchors, prior, secondary, fold_ids):
    """Run untouched S5 code with private entry hooks; never patch shared modules."""
    ensure(len(cohort) == len(anchors) == len(prior) == len(secondary) == len(fold_ids), "Replay shape")
    cohort = gate_cohort(cohort, anchors | secondary)
    allowed = np.zeros(len(cohort), dtype=bool)
    executed = np.zeros(len(cohort), dtype=bool)
    executions_by_episode = {}

    def eligible(row, definition, use_actual_utc_session=False):
        index = int(row.name)
        key = (int(fold_ids[index]), int(prior[index]))
        allowed[index] = bool(anchors[index] or (secondary[index] and executions_by_episode.get(key) is None))
        if not allowed[index]:
            return False, "below_threshold"
        return semantics.base_row_eligibility(row, definition, use_actual_utc_session)

    def opened(state, definition, frame, index):
        semantics.open_trade(state, definition, frame, index)
        ensure(state["position"] is not None and state["position"]["entry_index"] == index, "S5 open notification")
        executed[index] = True
        if not anchors[index]:
            key = (int(fold_ids[index]), int(prior[index]))
            ensure(secondary[index] and executions_by_episode.get(key) is None, "Secondary quota")
            executions_by_episode[key] = index

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
    ensure(int(executed.sum()) == len(trades), "Executed trade count")
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


def baseline_valid(row):
    for name, expected in BASELINE.items():
        ensure(math.isclose(float(row[name]), expected, rel_tol=0, abs_tol=1e-12), "baseline:" + name)


def operationals_valid(before, after):
    ensure(bool(before) and before == after, "operational_mutation")


def assessment(rows):
    pooled = rows[-1]
    failures = []
    for key, limit in {"pf": .80, "mean_r": BASELINE["mean_r"] - .03, "cost_stress_pf": .75}.items():
        if pooled[key] < limit:
            failures.append(key)
    for fold in rows[:-1]:
        for key, limit in {"trades": 10, "realized_wr": .45, "pf": .70}.items():
            if fold[key] < limit:
                failures.append(fold["fold"] + ":" + key)
    wr, tpd = pooled["realized_wr"], pooled["trades_per_day"]
    return {"guardrail_failures": failures, "guardrails_pass": not failures,
            "interesting": not failures and wr > BASELINE["realized_wr"] and tpd > BASELINE["trades_per_day"],
            "strong": not failures and wr >= .58 and tpd >= .40,
            "target": not failures and wr >= .60 and tpd >= .50,
            "delta_wr": wr - BASELINE["realized_wr"],
            "delta_trades_per_day": tpd - BASELINE["trades_per_day"],
            "frequency_uplift_pct": 100 * (tpd / BASELINE["trades_per_day"] - 1)}


def baseline_inputs():
    archive = ROOT / "training_runs" / REFERENCE
    intact(archive / "FINALIZED.json", REFERENCE_HASH)
    final = json_read(archive / "FINALIZED.json")
    ensure(final["run_id"] == REFERENCE, "reference_run_identity")
    for name, expected in final["file_sha256"].items():
        file = (archive / name).resolve()
        ensure(file.is_relative_to(archive), "reference_path")
        intact(file, expected)
    helper = import_frozen(archive / "validator_script.py", "episode_independent_frozen_helpers")
    for path, expected in helper.SOURCE_HASHES.items():
        intact(ROOT / path, expected)
    helper.finalized(ROOT / "training_runs" / helper.PARENT, ["training_script.py", "manifest.json"])
    helper.finalized(ROOT / "training_runs" / helper.TIMESTAMPS, ["exact_timestamps.npz"])
    old = import_frozen(ROOT / "training_runs" / helper.PARENT / "training_script.py", "episode_validation_parent")
    ensure(old.xgb.__version__ == "3.2.0", "xgboost_version")
    old.drl_trading_v2.DATA_DIR = str(ROOT)
    return archive, helper, old


def table(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def verify(run, checks):
    def check(name, value):
        checks[name] = bool(value)
        ensure(value, name)
    manifest = json_read(run / "manifest.json")
    check("formal_manifest", manifest["status"] == "in_progress" and manifest["run_id"] == run.name
          and manifest["experiment_name"] == "gold_anchored_episode_expansion_v1")
    for source, snapshot in (("gold_anchored_episode_expansion_v1.py", "training_script.py"),
                             ("validate_gold_anchored_episode_expansion_v1.py", "validator_script.py"),
                             ("execution_spec_gold_anchored_episode_expansion_v1.json", "execution_spec.json")):
        committed = subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + source], cwd=ROOT)
        check("git_source:" + source, committed.replace(b"\r\n", b"\n")
              == (run / snapshot).read_bytes().replace(b"\r\n", b"\n")
              == (ROOT / source).read_bytes().replace(b"\r\n", b"\n"))
    check("script_hash", digest(run / "training_script.py") == manifest["training_script_sha256"])
    ensure(json_read(ROOT / "execution_spec_gold_anchored_episode_expansion_v1.json") == expected_spec(), "Source execution spec drift")
    check("frozen_spec", json_read(run / "execution_spec.json") == expected_spec())
    check("no_retraining", manifest["model"]["trained"] is False)
    check("no_adaptive_search", manifest["search"]["performed"] is False
          and manifest["search"]["predefined_candidates"] == expected_candidates())
    check("no_promotion", manifest["promotion"]["requested"] is False
          and manifest["promotion"]["replacement_authorized"] is False
          and manifest["promotion"]["operational_artifact_changed"] is False)
    required = {"training_script.py", "validator_script.py", "execution_spec.json", "identity_audit.json",
                "episode_evidence.npz", "metrics.json", "candidates.csv", "fold_metrics.csv", "trade_ledger.csv",
                "report.md", "source_provenance.json", "operational_safety_pre.json", "operational_safety_post.json"}
    names = [item["path"] for item in manifest["artifacts"]]
    check("artifact_inventory", len(names) == len(set(names)) and required <= set(names))
    for item in manifest["artifacts"]:
        path = (run / item["path"]).resolve()
        check("artifact:" + item["path"], path.is_relative_to(run) and digest(path) == item["sha256"])
    before = json_read(run / "operational_safety_pre.json")
    operationals_valid(before, json_read(run / "operational_safety_post.json"))
    check("operational_pre_post", before == {name: digest(ROOT / name) for name in before}
          and set(before) == {"gemini.py", "gold_long_recent_candidate_xgb.json"})
    archive, helper, old = baseline_inputs()
    history, folds, identities = helper.rebuild(old)
    check("frozen_gold_c1_timestamps", identities == json_read(run / "identity_audit.json"))
    models = [item for item in json_read(archive / "model_inventory.json") if item["candidate_id"] == "B0_31_technical"]
    check("three_frozen_models", len(models) == 3)
    check("base_features", manifest["model"]["features"] == helper.BASE_FEATURES
          and manifest["model"]["feature_count"] == 31)
    check("source_provenance", json_read(run / "source_provenance.json") == {
        "reference_run": REFERENCE, "reference_finalized_sha256": REFERENCE_HASH,
        "predictions_sha256": digest(archive / "paired_oof_predictions.npz"),
        "models": models, "source_hashes": helper.SOURCE_HASHES})
    frames, masks, expected_keys, fold_ids = [], {item[0]: [] for item in CANDIDATES}, set(), []
    offset = 0
    execution_evidence = {}
    with np.load(run / "episode_evidence.npz", allow_pickle=False) as evidence, np.load(
        archive / "paired_oof_predictions.npz", allow_pickle=False
    ) as predictions, np.load(ROOT / "training_runs" / helper.TIMESTAMPS / "exact_timestamps.npz", allow_pickle=False) as timestamps:
        for number, name, _, indices in folds:
            prefix = f"fold{number}"
            scores = predictions[f"B0_31_technical_fold{number}"]
            utc = timestamps[f"fold{number}_score_utc_ns"]
            check("archived_indices:" + name, np.array_equal(indices, predictions[prefix + "_indices"]))
            for suffix, expected in (("scores", scores), ("indices", indices), ("utc_ns", utc)):
                key = prefix + "_" + suffix
                expected_keys.add(key)
                check("archived_evidence:" + key, evidence[key].dtype == expected.dtype and np.array_equal(evidence[key], expected))
            model_record = next(item for item in models if item["fold"] == name)
            intact(archive / model_record["path"], model_record["sha256"])
            helper.training_configuration(model_record["training_config"])
            model = old.xgb.XGBClassifier()
            model.load_model(archive / model_record["path"])
            model.set_params(device="cpu")
            check("frozen_model:" + name, model.get_booster().num_boosted_rounds() == 220
                  and model.get_booster().feature_names == helper.BASE_FEATURES)
            recomputed = model.predict_proba(history.loc[indices, helper.BASE_FEATURES])[:, 1].astype(np.float32)
            check("archived_predictions:" + name, scores.dtype == np.float32 and np.array_equal(recomputed, scores))
            fold_ids.extend([number] * len(scores))
            for candidate in CANDIDATES:
                arrays = episode_eligibility(utc, scores, candidate)
                for suffix, expected in zip(("anchor", "prior", "secondary"), arrays):
                    key = f"{prefix}_{candidate[0]}_{suffix}"
                    expected_keys.add(key)
                    check("episode_eligibility:" + key, evidence[key].dtype == expected.dtype
                          and np.array_equal(evidence[key], expected))
                anchor, prior, secondary = arrays
                masks[candidate[0]].append((anchor, np.where(prior >= 0, prior + offset, -1), secondary))
            offset += len(scores)
            frame = history.loc[indices, helper.PRICE_COLUMNS].copy()
            frame["buy_prob"], frame["sell_prob"] = scores, np.float32(0)
            frames.append(frame)
        for candidate in CANDIDATES:
            for suffix in ("allowed", "executed"):
                key = candidate[0] + "_" + suffix
                expected_keys.add(key)
                execution_evidence[key] = evidence[key].copy()
        check("exact_episode_evidence_keys", set(evidence.files) == expected_keys)
    metrics = json_read(run / "metrics.json")
    ids = [item[0] for item in CANDIDATES]
    check("metrics_candidates", list(metrics["results"]) == ids)
    check("research_only", metrics["model_training"] is False and metrics["adaptive_search"] is False
          and metrics["selected_candidate"] is None and metrics["production_promotion_requested"] is False)
    candidates = table(run / "candidates.csv")
    check("candidates_csv", [item["candidate_id"] for item in candidates] == ids)
    reported_folds = table(run / "fold_metrics.csv")
    scopes = [fold[0] for fold in helper.FOLD_WINDOWS] + ["pooled"]
    check("fold_scopes", len(reported_folds) == 36 and {(r["candidate_id"], r["fold"]) for r in reported_folds}
          == {(candidate, scope) for candidate in ids for scope in scopes})
    ledger = table(run / "trade_ledger.csv")
    check("ledger_candidates", {row["candidate_id"] for row in ledger} <= set(ids))
    combined, output = pd.concat(frames, ignore_index=True), {}
    for candidate in ids:
        cohort = old.semantics.finalize_cohort(combined, candidate, offset_hours=0)
        anchors, prior, secondary = [np.concatenate([part[i] for part in masks[candidate]]) for i in range(3)]
        replay, allowed, executed = replay_episodes(old.semantics, cohort, anchors, prior, secondary, fold_ids)
        for suffix, values in (("allowed", allowed), ("executed", executed)):
            recorded = execution_evidence[candidate + "_" + suffix]
            check("quota_execution:" + candidate + suffix, recorded.dtype == bool and np.array_equal(recorded, values))
        secondary_keys = [(t["score_fold"], t["anchor_row_index"]) for t in replay if t["entry_kind"] == "secondary"]
        check("max_one_secondary:" + candidate, len(secondary_keys) == len(set(secondary_keys)))
        stored = [row for row in ledger if row["candidate_id"] == candidate]
        check("ledger_count:" + candidate, len(stored) == len(replay))
        for actual, submitted in zip(replay, stored):
            check("ledger_fields:" + candidate, set(submitted) == set(actual) | {"candidate_id"})
            for key, value in actual.items():
                match = str(value) == submitted[key] if isinstance(value, (str, bool, np.bool_)) else helper.equal_number(submitted[key], value)
                check("trade:" + candidate + ":" + key, match)
        rows = helper.aggregate(stored)
        for row, diagnostic in zip(rows, secondary_metrics(stored, helper.FOLD_WINDOWS)):
            row.update({key: value for key, value in diagnostic.items() if key != "fold"})
        output[candidate] = rows
        if candidate == ids[0]:
            baseline_valid(rows[-1])
            checks["baseline_exact_reproduction"] = True
        report = metrics["results"][candidate]
        check("four_scopes:" + candidate, len(report["metrics"]) == 4)
        for row, reported in zip(rows, report["metrics"]):
            check("metric_scope:" + candidate, row["fold"] == reported["fold"])
            csv_row = next(r for r in reported_folds if r["candidate_id"] == candidate and r["fold"] == row["fold"])
            for key, value in row.items():
                if key != "fold":
                    check("metric:" + candidate + row["fold"] + key, helper.equal_number(reported[key], value)
                          and helper.equal_number(None if csv_row[key] == "" else csv_row[key], value))
        expected_assessment = assessment(rows)
        check("assessment:" + candidate, report["assessment"] == expected_assessment)
        candidate_row = next(row for row in candidates if row["candidate_id"] == candidate)
        for key, value in rows[-1].items():
            if key != "fold":
                check("pooled_csv:" + candidate + key, helper.equal_number(None if candidate_row[key] == "" else candidate_row[key], value))
        for key in ("delta_wr", "delta_trades_per_day", "frequency_uplift_pct"):
            check("delta_csv:" + candidate + key, helper.equal_number(candidate_row[key], expected_assessment[key]))
        for key in ("guardrails_pass", "interesting", "strong", "target"):
            check("assessment_csv:" + candidate + key, candidate_row[key] == str(expected_assessment[key]))
        check("guardrail_csv:" + candidate, ast.literal_eval(candidate_row["guardrail_failures"])
              == expected_assessment["guardrail_failures"])
        expected_verdict = ("rejected_guardrails" if not expected_assessment["guardrails_pass"] else
                            "interesting_research_only" if expected_assessment["interesting"] else "not_interesting")
        check("candidate_verdict:" + candidate, candidate_row["qualification_verdict"] == expected_verdict)
    check("final_operational_hashes", before == {name: digest(ROOT / name) for name in before})
    for item in models:
        intact(archive / item["path"], item["sha256"])
    return helper, output


def validate(run):
    run = run.resolve()
    ensure(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(), "Unfinalized run required")
    ensure(not (run / "validator.json").exists(), "Do not retry validation")
    checks, output = {}, {}
    try:
        _, output = verify(run, checks)
    except Exception as error:
        checks[f"{type(error).__name__}: {error}"] = False
    failed = [name for name, value in checks.items() if not value]
    verdict = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed,
               "checks": checks, "independent_metrics": output}
    write_json(run / "validator.json", verdict)
    (run / "validator.md").write_text("# Independent anchored episode expansion validator\n\nOverall: " + verdict["overall"]
                                      + "\n\nFailed checks: " + repr(failed) + "\n", encoding="utf-8")
    manifest = json_read(run / "manifest.json")
    manifest["registry"]["validator_result"] = "independent " + verdict["overall"]
    for name in ("validator.json", "validator.md"):
        manifest["artifacts"].append({"kind": "independent_validator", "path": name,
                                      "sha256": digest(run / name), "retention_status": "stored_in_run_directory"})
    write_json(run / "manifest.json", manifest)
    print("INDEPENDENT_VALIDATOR_" + verdict["overall"])
    print("failed_check_names=" + repr(failed))
    return not failed


def self_test():
    assert json_read(ROOT / "execution_spec_gold_anchored_episode_expansion_v1.json") == expected_spec()
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
        return 0
    if args.run_dir is None:
        parser.error("--run-dir required unless --self-test")
    return 0 if validate(args.run_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
