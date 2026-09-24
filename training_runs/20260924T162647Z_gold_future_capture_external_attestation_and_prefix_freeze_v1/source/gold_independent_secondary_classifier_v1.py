"""Frozen independent secondary classifier research; manual formal execution only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "gold_independent_secondary_classifier_v1"
SPEC_FILE = "execution_spec_" + EXPERIMENT + ".json"
SPEC_SHA256 = 'd6bda09ee19ed3b87e41e3e121f9e2438a600c808bc9e36c141ecc9dadd8bfa4'
CANDIDATES = (("A0_BASELINE_075", None), ("S1_SECONDARY_P060", .60),
              ("S2_SECONDARY_P065", .65), ("S3_SECONDARY_P070", .70),
              ("S4_SECONDARY_P075", .75))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str(values.dtype).encode("ascii")
                          + np.asarray(values.shape, dtype=np.int64).tobytes()
                          + values.tobytes()).hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), f"Empty evidence: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def operational_hashes():
    return {name: sha(ROOT / name) for name in ("gemini.py", "gold_long_recent_candidate_xgb.json")}


def specification():
    # Source text may be checked out as CRLF; hash the canonical LF specification.
    data = (ROOT / SPEC_FILE).read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(data).hexdigest() == SPEC_SHA256, "Frozen spec drift")
    return json.loads(data)


def load_module(path, name):
    descriptor = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(descriptor)
    # Import immutable snapshots without creating __pycache__ in finalized runs.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.ROOT = ROOT
    return module


def frozen_inputs(spec):
    archive = ROOT / "training_runs" / spec["reference_run"]
    require(sha(archive / "FINALIZED.json") == spec["reference_finalized_sha256"],
            "B0 finalized identity")
    for name, expected in read_json(archive / "FINALIZED.json")["file_sha256"].items():
        path = (archive / name).resolve()
        require(path.is_relative_to(archive) and sha(path) == expected, "B0 archive: " + name)
    helper = load_module(archive / "training_script.py", "secondary_frozen_data")
    require(helper.SOURCE_HASHES == spec["source_hashes"], "Source inventory drift")
    for name, expected in spec["source_hashes"].items():
        require(sha(ROOT / name) == expected, "Frozen source: " + name)
    for name in (spec["c1_run"], spec["timestamps_run"]):
        helper.verify_archive(ROOT / "training_runs" / name)
    old = load_module(ROOT / "training_runs" / spec["c1_run"] / "training_script.py",
                      "secondary_frozen_c1")
    old.drl_trading_v2.DATA_DIR = str(ROOT)
    require(xgb.__version__ == spec["xgboost_version"], "XGBoost must be 3.2.0")
    require(helper.BASE_FEATURES == spec["features"] and helper.EXPECTED_X == spec["feature_matrix_sha256"]
            and helper.EXPECTED_Y == spec["c1_target_sha256"], "Frozen features/target")
    require({key: list(value) for key, value in helper.BLOCKS.items()} == spec["blocks"], "Timestamp constants")
    require([list(row) for row in helper.FOLD_WINDOWS] == spec["folds"]
            and old.semantics.SIMULATORS[-1].simulator == "S5", "Frozen folds/S5")
    return archive, helper, old


def probabilities(values):
    values = np.asarray(values)
    require(values.ndim == 1 and np.isfinite(values).all()
            and ((values >= 0) & (values <= 1)).all(), "Invalid probabilities")
    return values


def candidate_gate(b0, secondary, candidate):
    require(candidate in CANDIDATES, "Non-frozen candidate")
    b0, secondary = probabilities(b0), probabilities(secondary)
    require(b0.shape == secondary.shape, "Probability alignment")
    primary = b0 >= .75
    extra = np.zeros(len(b0), dtype=bool)
    if candidate[1] is not None:
        extra = (b0 < .75) & (secondary >= candidate[1])
    require(not (primary & extra).any(), "Signal overlap")
    return primary, extra, primary | extra


def training_subset(train, score, train_times, score_times, b0_train):
    train, score = np.asarray(train), np.asarray(score)
    require(train.ndim == score.ndim == 1 and len(train) and len(score), "Empty fold")
    require(np.issubdtype(train.dtype, np.integer) and np.issubdtype(score.dtype, np.integer)
            and np.all(np.diff(train) > 0) and np.all(np.diff(score) > 0), "Fold index order")
    require(len(train_times) == len(train) and len(score_times) == len(score)
            and np.max(train_times) < np.min(score_times)
            and not np.intersect1d(train, score).size, "Fold isolation")
    values = probabilities(b0_train)
    require(len(values) == len(train), "Train probability alignment")
    selected = train[values < .75]
    require(len(selected) > 0, "Empty secondary universe")
    return selected


def fit_secondary(x, y, spec):
    require(xgb.__version__ == spec["xgboost_version"], "XGBoost version")
    require(list(x.columns) == spec["features"] and x.shape[1] == 31
            and all(dtype == np.float32 for dtype in x.dtypes)
            and np.isfinite(x.to_numpy()).all(), "Secondary features")
    y = np.asarray(y)
    require(y.dtype == np.int8 and y.shape == (len(x),)
            and np.array_equal(np.unique(y), [0, 1]), "Both C1 classes required")
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2.0 * counts[y])
    model = xgb.XGBClassifier(**spec["parameters"])
    model.fit(x, y, sample_weight=weights)
    require(model.get_booster().num_boosted_rounds() == 220, "Frozen rounds")
    return model, weights


def gated_cohort(semantics, frame, name, gate):
    cohort = semantics.finalize_cohort(frame, name, offset_hours=0)
    require(len(cohort) == len(gate), "Cohort alignment")
    raw = np.asarray(gate, dtype=bool)
    times = cohort["decision_time_api"].to_numpy(dtype="datetime64[ns]")
    gap = np.ones(len(raw), dtype=bool)
    gap[1:] = np.diff(times).astype("timedelta64[s]").astype(np.int64) > 120
    episodes = np.cumsum(raw & (np.r_[False, ~raw[:-1]] | gap)).astype(np.int64) - 1
    episodes[~raw] = -1
    cohort["raw_signal"], cohort["raw_episode_id"] = raw, episodes
    return cohort


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "Score distribution input")
    if not len(values):
        return {"count": 0, **dict.fromkeys(("mean", "std", "p50", "p75", "p90", "p95", "p99", "max"))}
    result = {"count": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    result.update(zip(("p50", "p75", "p90", "p95", "p99"),
                      np.percentile(values, [50, 75, 90, 95, 99], method="linear").tolist()))
    return {**result, "max": float(values.max())}


def assessment(rows, spec):
    failures = []
    for row in rows:
        limits = spec["guardrails"]["pooled" if row["fold"] == "pooled" else "fold"]
        local = [key for key, limit in limits.items()
                 if row[key] is None or math.isnan(row[key]) or row[key] < limit]
        row["guardrail_failures"], row["guardrails_pass"] = local, not local
        failures.extend(row["fold"] + ":" + key for key in local)
    pooled = rows[-1]
    wr, tpd = pooled["realized_wr"], pooled["trades_per_day"]
    baseline = spec["baseline"]
    return {"guardrail_failures": failures, "guardrails_pass": not failures,
            "interesting": not failures and wr > baseline["realized_wr"] and tpd > baseline["trades_per_day"],
            "strong": not failures and wr >= .58 and tpd >= .40,
            "target": not failures and wr >= .60 and tpd >= .50,
            "frequency_uplift_pct": 100 * (tpd / baseline["trades_per_day"] - 1),
            "delta_wr": wr - baseline["realized_wr"],
            "delta_trades_per_day": tpd - baseline["trades_per_day"]}


def evaluate(old, helper, frame, b0, secondary, fold_ids, candidate, spec):
    primary, extra, union = candidate_gate(b0, secondary, candidate)
    cohort = gated_cohort(old.semantics, frame, candidate[0], union)
    trades, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
    for trade in trades:
        index = int(trade["entry_index"])
        require(union[index], "Trade outside candidate gate")
        trade["entry_kind"] = "primary" if primary[index] else "secondary"
        trade["score_fold"] = int(fold_ids[index])
    rows = helper.metric_rows(old, trades, candidate[0])
    if candidate == CANDIDATES[0]:
        for key, expected in spec["baseline"].items():
            require(math.isclose(rows[-1][key], expected, rel_tol=0, abs_tol=1e-12),
                    "Baseline reproduction: " + key)
    for number, row in enumerate(rows, 1):
        pooled = row["fold"] == "pooled"
        mask = np.ones(len(b0), dtype=bool) if pooled else fold_ids == number
        selected = trades if pooled else [t for t in trades if t["score_fold"] == number]
        seconds = [t for t in selected if t["entry_kind"] == "secondary"]
        windows = spec["folds"] if pooled else [spec["folds"][number - 1]]
        days = sum((pd.Timestamp(end) - pd.Timestamp(start)).days for _, start, end in windows)
        wins = sum(t["net_r"] > 0 for t in seconds)
        row.update(primary_trade_count=len(selected) - len(seconds), secondary_trade_count=len(seconds),
                   secondary_wins=wins, secondary_losses=len(seconds) - wins,
                   secondary_realized_wr=wins / len(seconds) if seconds else None,
                   secondary_trades_per_day=len(seconds) / days,
                   primary_signal_count=int(primary[mask].sum()), secondary_signal_count=int(extra[mask].sum()),
                   overlap_count=int((primary[mask] & extra[mask]).sum()))
    evaluation = assessment(rows, spec)
    return {"metrics": rows, "assessment": evaluation}, trades, (primary, extra, union)


def formal_run(run):
    run = run.resolve()
    spec = specification()
    require(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(), "New run required")
    manifest = read_json(run / "manifest.json")
    require(manifest["experiment_name"] == EXPERIMENT and manifest["status"] == "in_progress"
            and manifest["run_id"] == run.name and manifest["git_dirty"] is False, "Formal manifest")
    require(git("rev-parse", "HEAD") == git("rev-parse", "@{u}") == manifest["git_commit"], "Git provenance")
    require(sha(run / "training_script.py") == sha(Path(__file__)) == manifest["training_script_sha256"], "Snapshot")
    prefix = run.relative_to(ROOT).as_posix() + "/"
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    require(all(line[3:].strip('"').startswith(prefix) for line in dirty.splitlines()), "Unexpected dirty paths")
    require(not any((run / name).exists() for name in ("execution_spec.json", "models", "failure.json")), "Do not retry a run")
    before = operational_hashes()
    # Exclusive creation claims the attempt before reconstruction or fitting.
    with (run / "execution_spec.json").open("x", encoding="utf-8") as stream:
        stream.write((ROOT / SPEC_FILE).read_text(encoding="utf-8"))
    write_json(run / "operational_safety_pre.json", before)
    (run / "validator_script.py").write_bytes((ROOT / ("validate_" + EXPERIMENT + ".py")).read_bytes())
    try:
        archive, helper, old = frozen_inputs(spec)
        history, target, folds, identities = helper.reconstruct(old)
        write_json(run / "identity_audit.json", identities)
        # All three full C1 target hashes are checked before the first fit.
        for number, _, train, _ in folds:
            require(array_hash(target[train]) == spec["c1_target_sha256"][f"fold{number}_train"], "C1 target identity")
        evidence, frames, b0_parts, fold_parts = {}, [], [], []
        with np.load(archive / "paired_oof_predictions.npz", allow_pickle=False) as saved:
            for number, name, train, score in folds:
                key = f"fold{number}"
                require(np.array_equal(score, saved[key + "_indices"]), "B0 score indices")
                item = next(item for item in spec["b0_models"] if item["fold"] == name)
                require(sha(archive / item["path"]) == item["sha256"], "B0 model hash")
                model = xgb.XGBClassifier()
                model.load_model(archive / item["path"])
                require(model.get_booster().feature_names == spec["features"], "B0 feature order")
                b0 = model.predict_proba(history.loc[score, spec["features"]].astype(np.float32))[:, 1]
                require(np.array_equal(b0, saved["B0_31_technical_" + key]), "B0 exact score reconstruction")
                b0_train = model.predict_proba(history.loc[train, spec["features"]].astype(np.float32))[:, 1]
                ns = history.TIME_DT.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                subset = training_subset(train, score, ns[train], ns[score], b0_train)
                evidence.update({key + "_train_indices": train, key + "_score_indices": score,
                                 key + "_subset_indices": subset, key + "_b0_train": b0_train,
                                 key + "_b0_score": b0, key + "_subset_target": target[subset],
                                 key + "_train_ns": ns[train], key + "_score_ns": ns[score]})
                frame = history.loc[score, helper.PRICE_COLUMNS].copy()
                frame["buy_prob"], frame["sell_prob"] = b0, np.float32(0)
                frames.append(frame)
                b0_parts.append(b0)
                fold_parts.append(np.full(len(score), number, dtype=np.int8))
        combined, b0, fold_ids = pd.concat(frames, ignore_index=True), np.concatenate(b0_parts), np.concatenate(fold_parts)
        # Reject baseline drift before any secondary model training.
        baseline_result, baseline_trades, baseline_gates = evaluate(
            old, helper, combined, b0, np.zeros_like(b0), fold_ids, CANDIDATES[0], spec)
        write_json(run / "baseline_reproduction.json", baseline_result)
        (run / "models").mkdir()
        inventory, secondary_parts = [], []
        for number, name, _, score in folds:
            key = f"fold{number}"
            subset = evidence[key + "_subset_indices"]
            x = history.loc[subset, spec["features"]].astype(np.float32)
            model, weights = fit_secondary(x, target[subset], spec)
            path = run / "models" / ("secondary_" + name + ".json")
            model.save_model(path)
            values = probabilities(model.predict_proba(history.loc[score, spec["features"]].astype(np.float32))[:, 1])
            evidence[key + "_secondary_score"] = values
            evidence[key + "_sample_weight"] = weights
            secondary_parts.append(values)
            inventory.append({"fold": name, "path": path.relative_to(run).as_posix(), "sha256": sha(path),
                              "parameters": spec["parameters"], "features": spec["features"], "seed": 42,
                              "rounds": model.get_booster().num_boosted_rounds(),
                              "training_config": json.loads(model.get_booster().save_config()),
                              "train_indices_sha256": array_hash(subset), "train_rows": len(subset),
                              "x_sha256": array_hash(x.to_numpy()), "target_sha256": array_hash(target[subset]),
                              "weights_sha256": array_hash(weights)})
            write_json(run / "model_inventory.json", inventory)
        secondary = np.concatenate(secondary_parts)
        results, ledgers, candidates = {}, [], []
        for candidate in CANDIDATES:
            if candidate == CANDIDATES[0]:
                result, trades, gates = baseline_result, baseline_trades, baseline_gates
            else:
                result, trades, gates = evaluate(old, helper, combined, b0, secondary, fold_ids, candidate, spec)
            results[candidate[0]] = result
            for suffix, values in zip(("primary", "secondary", "union"), gates):
                evidence[candidate[0] + "_" + suffix] = values
            candidates.append({**result["metrics"][-1], **result["assessment"],
                               "pnl": result["metrics"][-1]["pnl_r"], "max_dd": result["metrics"][-1]["max_dd_r"],
                               "qualification_verdict": "interesting_research_only" if result["assessment"]["interesting"] else "not_interesting"})
            ledgers.extend({**trade, "candidate_id": candidate[0]} for trade in trades)
        distributions = {}
        for number, name in [(i + 1, fold[0]) for i, fold in enumerate(spec["folds"])] + [(0, "pooled")]:
            mask = fold_ids == number if number else np.ones(len(b0), dtype=bool)
            distributions[name] = {"all_score_rows": distribution(secondary[mask]),
                                   "secondary_eligible_rows": distribution(secondary[mask & (b0 < .75)])}
        metrics = {"results": results, "score_distribution": distributions, "baseline_reproduction": True,
                   "selected_candidate": None, "production_promotion_requested": False,
                   "model_training": True, "adaptive_search": False, "classification": "historical_development_only"}
        np.savez_compressed(run / "secondary_evidence.npz", **evidence)
        write_json(run / "metrics.json", metrics)
        write_csv(run / "candidates.csv", candidates)
        write_csv(run / "fold_metrics.csv", [row for result in results.values() for row in result["metrics"]])
        write_csv(run / "trade_ledger.csv", ledgers)
        (run / "report.md").write_text(
            "# GOLD independent secondary classifier v1\n\nHistorical development only; no selection or promotion.\n"
            "Both realized WR and trades/day must improve and all frozen guardrails must pass.\n"
            "B0 train scores are in-sample routing values, not OOF performance predictions.\n\n```json\n"
            + json.dumps(read_json(run / "metrics.json"), indent=2) + "\n```\n", encoding="utf-8")
        write_json(run / "source_provenance.json", {"spec_sha256": SPEC_SHA256,
                   "reference_run": spec["reference_run"], "reference_finalized_sha256": spec["reference_finalized_sha256"],
                   "b0_models": spec["b0_models"], "source_hashes": spec["source_hashes"]})
        manifest["data"] = read_json(archive / "manifest.json")["data"]
        manifest["data"].update(
            data_sources=["Frozen GOLD reconstruction and exact archived C1 target; B0 residual training subset"],
            full_frozen_train_rows=manifest["data"]["train_rows"],
            train_rows=sum(item["train_rows"] for item in inventory),
            secondary_training_rows_by_fold={item["fold"]: item["train_rows"] for item in inventory},
        )
        parent_model = read_json(ROOT / "training_runs" / spec["c1_run"] / "manifest.json")["model"]
        manifest["model"] = {**parent_model, "trained": True,
            "model_type": "three fold-isolated XGBoost residual specialists; last fold is archival identity only",
            "parameters": spec["parameters"], "boosted_rounds_or_estimators": 220,
            "features": spec["features"], "feature_count": 31, "calibration_method": "none",
            "label_definition": "Exact archived C1 standalone S5 net realized R > 0; zero/nonpositive maps to 0",
            "artifact_path": inventory[-1]["path"], "artifact_sha256": inventory[-1]["sha256"],
            "retention_status": "stored_in_run_directory", "fold_model_inventory": "model_inventory.json",
            "fold_models_trained": 3}
        # Do not inherit parent-specific model inventories or training counts.
        manifest["model"] = {key: manifest["model"][key] for key in (
            "trained", "model_type", "parameters", "boosted_rounds_or_estimators", "features", "feature_count",
            "label_definition", "horizon", "label_tp_sl_semantics", "execution_tp_sl_semantics", "calibration_method",
            "artifact_path", "artifact_sha256", "retention_status", "fold_model_inventory", "fold_models_trained")}
        (run / "model.sha256").write_text(inventory[-1]["sha256"] + "  " + inventory[-1]["path"] + "\n", encoding="utf-8")
        manifest["random_seeds"] = {"secondary": 42}
        manifest["random_seed_note"] = "Independent CPU single-thread fold fits; subset-only balanced weights"
        manifest["search"] = {"performed": False, "not_applicable_reason": "Five frozen gates; no tuning",
                              "candidate_results_file": "candidates.csv", "predefined_candidates": spec["candidates"]}
        manifest["promotion"] = {"requested": False, "replacement_authorized": False,
                                 "operational_artifact_changed": False, "gate_result": "research_only"}
        manifest["registry"] = {"parent_or_incumbent": spec["reference_run"], "selected_configuration": "none selected; five frozen gates",
                                "validator_result": "PENDING", **{key: "see candidates.csv" for key in
                                ("trades_per_day", "realized_win_rate", "pf", "mean_r", "pnl", "max_dd")}}
    except Exception as error:
        write_json(run / "failure.json", {"status": "FAIL", "error": str(error)})
        raise
    finally:
        after = operational_hashes()
        write_json(run / "operational_safety_post.json", after)
        require(before == after, "Operational mutation")
    manifest["artifacts"] = [{"path": path.relative_to(run).as_posix(), "sha256": sha(path),
                              "kind": "research_evidence", "retention_status": "stored_in_run_directory"}
                             for path in sorted(run.rglob("*")) if path.is_file()
                             and path.name not in {"manifest.json", "stdout.log"}]
    write_json(run / "manifest.json", manifest)
    print("FROZEN_INDEPENDENT_SECONDARY_CLASSIFIER_COMPLETED")


def self_test():
    spec = specification()
    assert spec["candidates"] == [dict(candidate_id=n, secondary_threshold=t) for n, t in CANDIDATES]
    assert len(CANDIDATES) == 5 and spec["parameters"]["random_state"] == 42

    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Invalid input accepted")

    b0 = np.array([.749, .75, .9, .1], dtype=np.float64)
    for candidate in CANDIDATES:
        threshold = candidate[1] or .60
        scores = np.array([threshold, 1., 1., np.nextafter(threshold, 0.)])
        primary, extra, union = candidate_gate(b0, scores, candidate)
        assert primary.tolist() == [False, True, True, False]
        assert extra.tolist() == [candidate[1] is not None, False, False, False]
        assert np.array_equal(union, primary | extra) and not (primary & extra).any()
    for threshold in (.60, .65, .70, .75):
        candidate = next(item for item in CANDIDATES if item[1] == threshold)
        _, secondary, _ = candidate_gate(np.full(3, .5, dtype=np.float32),
                                         np.array([threshold] * 3, dtype=np.float32), candidate)
        assert secondary.all()
    train, score = np.array([0, 1, 2, 3]), np.array([4, 5])
    subset = training_subset(train, score, train, score, b0)
    assert subset.tolist() == [0, 3]
    rejects(lambda: training_subset(train, np.array([3, 5]), train, score, b0))
    rejects(lambda: training_subset(train, score, train + 10, score, b0))
    rejects(lambda: training_subset(train, score, train, score, np.ones(4)))
    rejects(lambda: candidate_gate([np.nan], [.8], CANDIDATES[1]))
    rejects(lambda: candidate_gate([.1], [1.1], CANDIDATES[1]))
    rejects(lambda: candidate_gate([.1], [.8], ("extra", .6)))
    # Small, generated matrices only. No historical reconstruction or run creation.
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(1200, 31)).astype(np.float32), columns=spec["features"])
    y = (x.iloc[:, 0].to_numpy() > 0).astype(np.int8)
    first, weights = fit_secondary(x, y, spec)
    second, again = fit_secondary(x.copy(), y.copy(), spec)
    assert np.array_equal(weights, again)
    assert first.get_booster().save_raw(raw_format="json") == second.get_booster().save_raw(raw_format="json")
    assert np.array_equal(first.predict_proba(x), second.predict_proba(x))
    assert first.get_booster().num_boosted_rounds() == 220
    # Two isolated toy folds: changing a later score row cannot enter fitting.
    later = x.iloc[1000:].copy()
    later.iloc[:, :] = 999
    a = training_subset(np.arange(1000), np.arange(1000, 1200), np.arange(1000),
                        np.arange(1000, 1200), np.full(1000, .5))
    assert a.max() == 999 and not np.intersect1d(a, later.index).size
    rejects(lambda: fit_secondary(x, np.zeros(len(x), dtype=np.int8), spec))
    rejects(lambda: fit_secondary(x.assign(B0_probability=.1), y, spec))
    assert distribution([])["mean"] is None
    assert distribution([.1, .3])["count"] == 2
    assert math.isclose(distribution([.1, .3])["p50"], .2, abs_tol=1e-12)
    # Untouched S5 entry gates, including occupancy and cross-fold state.
    import gold_gemini_execution_semantics_v1 as s5
    day = pd.Timestamp("2020-01-06")
    while day.dayofweek not in s5.ALLOWED_WEEKDAYS:
        day += pd.Timedelta(days=1)
    day += pd.Timedelta(hours=min(s5.ALLOWED_HOURS))
    scores = np.array([.75, .2, .2, .2, .2], dtype=np.float32)
    frame = pd.DataFrame({"TIME_DT": pd.date_range(day, periods=5, freq="min"), "OPEN": 100.,
        "HIGH": [100., 100., 110., 110., 110.], "LOW": 100., "CLOSE": 100.,
        "ATR": 1., "M1_RSI": 50., "SPREAD": 1., "buy_prob": scores, "sell_prob": 0.})
    _, _, gate = candidate_gate(scores, np.ones(5), CANDIDATES[0])
    actual, _ = s5.simulate(gated_cohort(s5, frame, "fixture", gate), s5.SIMULATORS[-1])
    expected, _ = s5.simulate(s5.finalize_cohort(frame, "fixture", 0), s5.SIMULATORS[-1])
    assert actual == expected and [t["entry_index"] for t in actual] == [0]
    _, _, union = candidate_gate(scores, np.full(5, .65), CANDIDATES[2])
    extra, _ = s5.simulate(gated_cohort(s5, frame, "fixture", union), s5.SIMULATORS[-1])
    assert [t["entry_index"] for t in extra] == [0, 3, 4]
    assert extra[0]["exit_index"] == 2  # Position crosses the toy fold boundary at index 2.
    print("SELF_TEST_PASS: frozen gates, boundaries, no overlap, fold isolation, deterministic fitting, A0 and S5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        formal_run(args.run_dir)


if __name__ == "__main__":
    main()
