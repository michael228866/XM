"""Independent replay and causal rank audit; never imports the ranking runner."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE = "20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1"
REFERENCE_HASH = "592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe"
DAY_NS = 86_400_000_000_000
CANDIDATES = (('A0_BASELINE_075', None, 0.75, None),
 ('A1_ROLLING_1D_P95', '1D', None, 0.95),
 ('A1_ROLLING_1D_P97', '1D', None, 0.97),
 ('A1_ROLLING_3D_P95', '3D', None, 0.95),
 ('A1_ROLLING_3D_P97', '3D', None, 0.97),
 ('A1_ROLLING_5D_P95', '5D', None, 0.95),
 ('A1_ROLLING_5D_P97', '5D', None, 0.97),
 ('A2_SESSION_P95', 'SESSION', None, 0.95),
 ('A2_SESSION_P97', 'SESSION', None, 0.97),
 ('A3_3D_F065_P95', '3D', 0.65, 0.95),
 ('A3_3D_F065_P97', '3D', 0.65, 0.97),
 ('A3_3D_F070_P95', '3D', 0.7, 0.95),
 ('A3_3D_F070_P97', '3D', 0.7, 0.97))
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
    entries = []
    for name, reference, floor, level in CANDIDATES:
        entries.append({"candidate_id": name, "reference": reference,
                        "absolute_floor": floor, "percentile_threshold": level})
    return entries


def candidates_valid(entries):
    ensure(entries == expected_candidates(), "candidate_definitions")


def expected_spec():
    return {
        "experiment": "gold_score_ranking_v1", "candidates": expected_candidates(),
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


def recompute_ranks(times, scores, reference):
    """Direct counts in prior slices: independent of the runner's sorted multiset."""
    times = np.asarray(times, dtype=np.int64)
    scores = np.asarray(scores)
    ensure(times.ndim == scores.ndim == 1 and len(times) == len(scores), "ranking_shape")
    ensure(np.all(np.diff(times) >= 0) and np.isfinite(scores).all(), "ranking_input")
    widths = {"1D": DAY_NS, "3D": 3 * DAY_NS, "5D": 5 * DAY_NS}
    ensure(reference in {*widths, "SESSION"}, "non_frozen_window")
    counts = np.zeros(len(times), dtype=np.int64)
    le = np.zeros(len(times), dtype=np.int64)
    percentiles = np.full(len(times), np.nan, dtype=np.float64)
    left, group_start = 0, 0
    for index, timestamp in enumerate(times):
        if index == 0 or timestamp != times[index - 1]:
            group_start = index
        if reference == "SESSION":
            day_start = (int(timestamp) // DAY_NS) * DAY_NS
            session = (int(timestamp) - day_start) // (DAY_NS // 3)
            lower = day_start + session * (DAY_NS // 3)
        else:
            lower = int(timestamp) - widths[reference]
        while left < group_start and times[left] < lower:
            left += 1
        prior = scores[left:group_start]
        counts[index] = len(prior)
        le[index] = np.count_nonzero(prior <= scores[index])
        if len(prior) >= 60:
            percentiles[index] = le[index] / len(prior)
    return counts, le, percentiles


def recompute_gate(definition, scores, ranks):
    ensure(definition in CANDIDATES, "non_frozen_threshold")
    name, reference, floor, level = definition
    values = np.asarray(scores, dtype=np.float64)
    if name == "A0_BASELINE_075":
        return values >= .75
    count, _, percentile = ranks[reference]
    result = (count >= 60) & np.isfinite(percentile) & (percentile >= level)
    return result if floor is None else result & (values >= floor)


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
    helper = import_frozen(archive / "validator_script.py", "ranking_independent_frozen_helpers")
    for path, expected in helper.SOURCE_HASHES.items():
        intact(ROOT / path, expected)
    helper.finalized(ROOT / "training_runs" / helper.PARENT, ["training_script.py", "manifest.json"])
    helper.finalized(ROOT / "training_runs" / helper.TIMESTAMPS, ["exact_timestamps.npz"])
    old = import_frozen(ROOT / "training_runs" / helper.PARENT / "training_script.py", "ranking_validation_parent")
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
          and manifest["experiment_name"] == "gold_score_ranking_v1")
    for source, snapshot in (("gold_score_ranking_v1.py", "training_script.py"),
                             ("validate_gold_score_ranking_v1.py", "validator_script.py"),
                             ("execution_spec_gold_score_ranking_v1.json", "execution_spec.json")):
        committed = subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + source], cwd=ROOT)
        check("git_source:" + source, committed.replace(b"\r\n", b"\n")
              == (run / snapshot).read_bytes().replace(b"\r\n", b"\n")
              == (ROOT / source).read_bytes().replace(b"\r\n", b"\n"))
    check("script_hash", digest(run / "training_script.py") == manifest["training_script_sha256"])
    ensure(json_read(ROOT / "execution_spec_gold_score_ranking_v1.json") == expected_spec(), "Source execution spec drift")
    check("frozen_spec", json_read(run / "execution_spec.json") == expected_spec())
    check("no_retraining", manifest["model"]["trained"] is False)
    check("no_adaptive_search", manifest["search"]["performed"] is False
          and manifest["search"]["predefined_candidates"] == expected_candidates())
    check("no_promotion", manifest["promotion"]["requested"] is False
          and manifest["promotion"]["replacement_authorized"] is False
          and manifest["promotion"]["operational_artifact_changed"] is False)
    required = {"training_script.py", "validator_script.py", "execution_spec.json", "identity_audit.json",
                "ranking_evidence.npz", "metrics.json", "candidates.csv", "fold_metrics.csv", "trade_ledger.csv",
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
    frames, masks, expected_keys = [], {item[0]: [] for item in CANDIDATES}, set()
    with np.load(run / "ranking_evidence.npz", allow_pickle=False) as evidence, np.load(
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
            ranks = {}
            for reference in ("1D", "3D", "5D", "SESSION"):
                ranks[reference] = recompute_ranks(utc, scores, reference)
                for suffix, expected in zip(("count", "le", "percentile"), ranks[reference]):
                    key = f"{prefix}_{reference}_{suffix}"
                    expected_keys.add(key)
                    check("causal_rank:" + key, evidence[key].dtype == expected.dtype
                          and np.array_equal(evidence[key], expected, equal_nan=True))
            for candidate in CANDIDATES:
                key = prefix + "_" + candidate[0]
                expected_keys.add(key)
                mask = recompute_gate(candidate, scores, ranks)
                check("eligibility:" + key, evidence[key].dtype == bool and np.array_equal(mask, evidence[key]))
                masks[candidate[0]].append(mask)
            frame = history.loc[indices, helper.PRICE_COLUMNS].copy()
            frame["buy_prob"], frame["sell_prob"] = scores, np.float32(0)
            frames.append(frame)
        check("exact_ranking_evidence_keys", set(evidence.files) == expected_keys)
    metrics = json_read(run / "metrics.json")
    ids = [item[0] for item in CANDIDATES]
    check("metrics_candidates", list(metrics["results"]) == ids)
    check("research_only", metrics["model_training"] is False and metrics["adaptive_search"] is False
          and metrics["selected_candidate"] is None and metrics["production_promotion_requested"] is False)
    candidates = table(run / "candidates.csv")
    check("candidates_csv", [item["candidate_id"] for item in candidates] == ids)
    reported_folds = table(run / "fold_metrics.csv")
    scopes = [fold[0] for fold in helper.FOLD_WINDOWS] + ["pooled"]
    check("fold_scopes", len(reported_folds) == 52 and {(r["candidate_id"], r["fold"]) for r in reported_folds}
          == {(candidate, scope) for candidate in ids for scope in scopes})
    ledger = table(run / "trade_ledger.csv")
    check("ledger_candidates", {row["candidate_id"] for row in ledger} <= set(ids))
    combined, output = pd.concat(frames, ignore_index=True), {}
    for candidate in ids:
        cohort = old.semantics.finalize_cohort(combined, candidate, offset_hours=0)
        cohort = gate_cohort(cohort, np.concatenate(masks[candidate]))
        replay, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
        stored = [row for row in ledger if row["candidate_id"] == candidate]
        check("ledger_count:" + candidate, len(stored) == len(replay))
        for actual, submitted in zip(replay, stored):
            check("ledger_fields:" + candidate, set(submitted) == set(actual) | {"candidate_id"})
            for key, value in actual.items():
                match = str(value) == submitted[key] if isinstance(value, (str, bool, np.bool_)) else helper.equal_number(submitted[key], value)
                check("trade:" + candidate + ":" + key, match)
        rows = helper.aggregate(stored)
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
    (run / "validator.md").write_text("# Independent score-ranking validator\n\nOverall: " + verdict["overall"]
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
    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Bad input accepted")
    times = np.arange(65, dtype=np.int64) * 60_000_000_000
    scores = np.full(65, .5, dtype=np.float32)
    count, le, rank = recompute_ranks(times, scores, "1D")
    assert count[0] == 0 and count[60] == le[60] == 60 and rank[60] == 1
    assert np.isnan(rank[:60]).all()
    changed = scores.copy()
    changed[62:] = 0
    assert np.array_equal(recompute_ranks(times, changed, "1D")[2][:62], rank[:62], equal_nan=True)
    assert recompute_ranks(np.array([0, 0, 1]), np.array([.1, .9, .5]), "1D")[0].tolist() == [0, 0, 2]
    sparse = np.array([0, 1, 3, 5, 6], dtype=np.int64) * DAY_NS
    for ref, expected in (("1D", [0, 1, 0, 0, 1]), ("3D", [0, 1, 2, 1, 2]), ("5D", [0, 1, 2, 3, 3])):
        assert recompute_ranks(sparse, np.ones(5), ref)[0].tolist() == expected
    session_times = np.array([0, 1, DAY_NS // 3, DAY_NS // 3 + 1, DAY_NS, DAY_NS + 1])
    assert recompute_ranks(session_times, np.ones(6), "SESSION")[0].tolist() == [0, 1, 0, 1, 0, 1]
    assert recompute_ranks(times, scores, "1D")[0][0] == 0
    assert recompute_gate(CANDIDATES[0], np.array([.74, .75, .76]), {}).tolist() == [False, True, True]
    assert not recompute_gate(CANDIDATES[1], scores, {"1D": (count, le, rank)})[:60].any()
    baseline_valid(BASELINE)
    rejects(lambda: baseline_valid({**BASELINE, "wins": 391}))
    rejects(lambda: candidates_valid(expected_candidates() + [expected_candidates()[0]]))
    bad = expected_candidates()
    bad[1]["percentile_threshold"] = .96
    rejects(lambda: candidates_valid(bad))
    rejects(lambda: recompute_ranks(times, scores, "2D"))
    rejects(lambda: operationals_valid({"production": "a"}, {"production": "b"}))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model"
        path.write_bytes(b"a")
        expected = digest(path)
        path.write_bytes(b"b")
        rejects(lambda: intact(path, expected))
    helper = import_frozen(ROOT / "training_runs" / REFERENCE / "validator_script.py", "ranking_test_metrics")
    trades = [{"entry_time_api": "2018-01-02", "net_r": 1., "stress_r": .9},
              {"entry_time_api": "2021-01-02", "net_r": -.5, "stress_r": -.6}]
    for trade in trades:
        trade.update(exit_time_api=trade["entry_time_api"], exit_reason="timeout", account_pnl=trade["net_r"])
    pooled = helper.aggregate(trades)[-1]
    assert pooled["trades"] == 2 and pooled["pnl_r"] == .5 and pooled["max_dd_r"] == -.5
    assert pooled["pf"] == 2 and pooled["trades_per_day"] == 2 / 2557
    assert json_read(ROOT / "execution_spec_gold_score_ranking_v1.json") == expected_spec()
    # Exact P97, both A3 floors, and the AND requirement.
    for level, le_count in ((.95, 95), (.97, 97)):
        values = np.r_[np.full(le_count, .4), np.full(100 - le_count, .9), .7]
        ranked = recompute_ranks(np.arange(101), values, "3D")
        assert ranked[2][-1] == level
        for candidate in CANDIDATES[9:]:
            assert bool(recompute_gate(candidate, values, {"3D": ranked})[-1]) == (level >= candidate[3])
    for candidate in CANDIDATES[9:]:
        for value, expected in ((candidate[2] - .001, False), (candidate[2], True)):
            values = np.r_[np.full(60, .4), value]
            ranked = recompute_ranks(np.arange(61), values, "3D")
            assert bool(recompute_gate(candidate, values, {"3D": ranked})[-1]) == expected
        values = np.r_[np.full(60, .9), .8]
        assert not recompute_gate(candidate, values, {"3D": recompute_ranks(np.arange(61), values, "3D")})[-1]
    # Every UTC bucket and midnight resets even after sufficient prior scores.
    for boundary in (DAY_NS // 3, 2 * DAY_NS // 3, DAY_NS):
        utc = np.r_[boundary - 60 + np.arange(60), boundary, boundary + 1]
        ranked = recompute_ranks(utc, np.ones(62), "SESSION")
        assert ranked[0][-2:].tolist() == [0, 1]
        assert np.isnan(ranked[2][-2:]).all()
    assert len(CANDIDATES) == 13
    boundary_scores = np.r_[np.full(57, .4), np.full(3, .6), .5]
    boundary_ranks = recompute_ranks(np.arange(61), boundary_scores, "1D")
    assert boundary_ranks[1][-1] == 57 and boundary_ranks[2][-1] == .95
    assert recompute_gate(CANDIDATES[1], boundary_scores, {"1D": boundary_ranks})[-1]
    assert not recompute_gate(CANDIDATES[2], boundary_scores, {"1D": boundary_ranks})[-1]
    for score, expected in ((.649, False), (.65, True), (.70, True)):
        values = np.r_[np.full(60, .4), score]
        reference = recompute_ranks(np.arange(61), values, "3D")
        assert bool(recompute_gate(CANDIDATES[9], values, {"3D": reference})[-1]) == expected
    # Both folds start cold even if a prior fold already had sufficient history.
    assert np.isnan(recompute_ranks(np.arange(5), np.ones(5), "3D")[2]).all()
    print("SELF_TEST_PASS: causal future_mutation windows sessions baseline candidates thresholds mutation pooled ties")


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
