"""Independent secondary classifier validator; never imports the new runner."""
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
    data = (ROOT / SPEC_FILE).read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(data).hexdigest() == SPEC_SHA256, "Frozen spec drift")
    return json.loads(data)


def import_frozen(path, name):
    descriptor = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(descriptor)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.ROOT = ROOT
    return module


def baseline_inputs(spec):
    archive = ROOT / "training_runs" / spec["reference_run"]
    require(sha(archive / "FINALIZED.json") == spec["reference_finalized_sha256"], "B0 finalized identity")
    for name, expected in read_json(archive / "FINALIZED.json")["file_sha256"].items():
        path = (archive / name).resolve()
        require(path.is_relative_to(archive) and sha(path) == expected, "B0 archive: " + name)
    # Independent frozen reconstruction and metric code, not the new runner.
    helper = import_frozen(archive / "validator_script.py", "secondary_independent_data")
    require(helper.SOURCE_HASHES == spec["source_hashes"], "Source inventory")
    for name, expected in spec["source_hashes"].items():
        require(sha(ROOT / name) == expected, "Frozen source: " + name)
    helper.finalized(ROOT / "training_runs" / spec["c1_run"], ["training_script.py", "manifest.json"])
    helper.finalized(ROOT / "training_runs" / spec["timestamps_run"], ["exact_timestamps.npz"])
    parent = import_frozen(ROOT / "training_runs" / spec["c1_run"] / "training_script.py", "secondary_audit_c1")
    parent.drl_trading_v2.DATA_DIR = str(ROOT)
    require(xgb.__version__ == "3.2.0", "XGBoost version")
    require(helper.BASE_FEATURES == spec["features"] and helper.EXPECTED_X == spec["feature_matrix_sha256"]
            and helper.EXPECTED_Y == spec["c1_target_sha256"], "Feature/target constants")
    require({key: list(value) for key, value in helper.BLOCKS.items()} == spec["blocks"], "Timestamp constants")
    require([list(row) for row in helper.FOLD_WINDOWS] == spec["folds"]
            and parent.semantics.SIMULATORS[-1].simulator == "S5", "Frozen fold windows/S5")
    return archive, helper, parent


def independent_gates(primary_scores, specialist_scores, candidate):
    require(candidate in CANDIDATES, "Candidate definition")
    p, s = np.asarray(primary_scores), np.asarray(specialist_scores)
    require(p.ndim == s.ndim == 1 and p.shape == s.shape, "Gate shape")
    require(np.isfinite(p).all() and np.isfinite(s).all()
            and ((p >= 0) & (p <= 1) & (s >= 0) & (s <= 1)).all(), "Probability range")
    primary = p >= .75
    secondary = np.zeros(p.size, dtype=bool)
    if candidate[1] is not None:
        locations = np.flatnonzero(p < .75)
        secondary[locations] = s[locations] >= candidate[1]
    require(np.count_nonzero(primary & secondary) == 0, "Primary/secondary overlap")
    return primary, secondary, np.logical_or(primary, secondary)


def independent_subset(train, score, train_ns, score_ns, b0_train):
    train, score, values = np.asarray(train), np.asarray(score), np.asarray(b0_train)
    require(train.ndim == score.ndim == values.ndim == 1 and len(train) and len(score), "Fold shape")
    require(len(train_ns) == len(train) == len(values) and len(score_ns) == len(score), "Fold alignment")
    require(np.issubdtype(train.dtype, np.integer) and np.issubdtype(score.dtype, np.integer)
            and (np.diff(train) > 0).all() and (np.diff(score) > 0).all(), "Fold ordering")
    require(max(train_ns) < min(score_ns) and not np.isin(train, score).any(), "Fold isolation")
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "B0 train scores")
    selected = train[np.flatnonzero(values < .75)]
    require(len(selected) > 0, "Empty secondary universe")
    return selected


def independent_fit(x, y, spec):
    require(xgb.__version__ == "3.2.0" and list(x.columns) == spec["features"]
            and x.shape[1] == 31 and all(t == np.float32 for t in x.dtypes)
            and np.isfinite(x.to_numpy()).all(), "Secondary fitting features")
    y = np.asarray(y)
    require(y.dtype == np.int8 and y.shape == (len(x),)
            and set(np.unique(y)) == {0, 1}, "Both target classes required")
    positive, negative = int(np.count_nonzero(y)), int(np.count_nonzero(y == 0))
    weights = np.where(y == 1, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
    # One reconstruction per fold; this is verification, never a retry or search.
    model = xgb.XGBClassifier(**spec["parameters"])
    model.fit(x, y, sample_weight=weights)
    return model, weights


def score_summary(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "Score summary input")
    keys = ("mean", "std", "p50", "p75", "p90", "p95", "p99", "max")
    if values.size == 0:
        return {"count": 0, **dict.fromkeys(keys)}
    quantiles = np.quantile(values, [.50, .75, .90, .95, .99], method="linear")
    return dict(zip(("count", *keys), (values.size, float(np.mean(values)),
                    float(np.std(values, ddof=0)), *quantiles.tolist(), float(np.max(values)))))


def replay(parent, frame, candidate, primary, secondary, fold_ids):
    cohort = parent.semantics.finalize_cohort(frame, candidate, offset_hours=0)
    flags = np.logical_or(primary, secondary)
    times = cohort.decision_time_api.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    starts = flags.copy()
    starts[1:] &= (~flags[:-1]) | (np.diff(times) // 1_000_000_000 > 120)
    ids = np.cumsum(starts, dtype=np.int64) - 1
    ids[~flags] = -1
    cohort["raw_signal"], cohort["raw_episode_id"] = flags, ids
    trades, _ = parent.semantics.simulate(cohort, parent.semantics.SIMULATORS[-1])
    for trade in trades:
        index = int(trade["entry_index"])
        require(flags[index], "Executed gate")
        trade["entry_kind"] = "secondary" if secondary[index] else "primary"
        trade["score_fold"] = int(fold_ids[index])
    return trades


def audited_metrics(helper, trades, candidate, primary, secondary, fold_ids, spec):
    rows = helper.aggregate(trades)
    failures = []
    for i, row in enumerate(rows):
        row["candidate_id"] = candidate
        pooled = row["fold"] == "pooled"
        mask = np.ones(len(primary), dtype=bool) if pooled else fold_ids == i + 1
        selected = trades if pooled else [t for t in trades if t["score_fold"] == i + 1]
        extras = [t for t in selected if t["entry_kind"] == "secondary"]
        windows = spec["folds"] if pooled else [spec["folds"][i]]
        days = sum((pd.Timestamp(end) - pd.Timestamp(start)).days for _, start, end in windows)
        wins = len([t for t in extras if t["net_r"] > 0])
        row.update(primary_trade_count=len(selected) - len(extras), secondary_trade_count=len(extras),
                   secondary_wins=wins, secondary_losses=len(extras) - wins,
                   secondary_realized_wr=wins / len(extras) if extras else None,
                   secondary_trades_per_day=len(extras) / days,
                   primary_signal_count=int(np.count_nonzero(primary[mask])),
                   secondary_signal_count=int(np.count_nonzero(secondary[mask])),
                   overlap_count=int(np.count_nonzero(primary[mask] & secondary[mask])))
        limits = spec["guardrails"]["pooled" if pooled else "fold"]
        local = [key for key, minimum in limits.items()
                 if row[key] is None or math.isnan(row[key]) or row[key] < minimum]
        row["guardrail_failures"], row["guardrails_pass"] = local, not local
        failures.extend(row["fold"] + ":" + key for key in local)
    wr, tpd = rows[-1]["realized_wr"], rows[-1]["trades_per_day"]
    assessment = {"guardrail_failures": failures, "guardrails_pass": not failures,
                  "interesting": not failures and wr > .5660377358490566 and tpd > .26945639421196715,
                  "strong": not failures and wr >= .58 and tpd >= .40,
                  "target": not failures and wr >= .60 and tpd >= .50,
                  "frequency_uplift_pct": 100 * (tpd / .26945639421196715 - 1),
                  "delta_wr": wr - .5660377358490566, "delta_trades_per_day": tpd - .26945639421196715}
    return {"metrics": rows, "assessment": assessment}


def equal(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            equal(a, b) for a, b in zip(actual, expected))
    if expected is None or (isinstance(expected, float) and not math.isfinite(expected)):
        return actual is None
    if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
        except (ValueError, TypeError):
            return False
    return type(actual) is type(expected) and actual == expected


def verify_csv(path, rows):
    import ast
    with path.open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(rows), "CSV row count: " + path.name)
    for saved, row in zip(actual, rows):
        require(saved.keys() == row.keys(), "CSV columns: " + path.name)
        for key, value in row.items():
            text = saved[key]
            if isinstance(value, (float, np.floating)) and not math.isfinite(value):
                require(text == str(value), "CSV nonfinite value: " + path.name + ":" + key)
                continue
            if isinstance(value, (list, bool)):
                text = ast.literal_eval(text)
            elif text == "" and (value is None or isinstance(value, float)):
                text = None
            require(equal(text, value), "CSV value: " + path.name + ":" + key)


def verify(run, checks):
    def check(name, condition):
        checks[name] = bool(condition)
        require(condition, name)

    spec = specification()
    manifest = read_json(run / "manifest.json")
    check("formal_manifest", manifest["experiment_name"] == EXPERIMENT and manifest["status"] == "in_progress"
          and manifest["run_id"] == run.name and manifest["git_dirty"] is False and not (run / "failure.json").exists())
    for source, snapshot in ((EXPERIMENT + ".py", "training_script.py"),
                             ("validate_" + EXPERIMENT + ".py", "validator_script.py"), (SPEC_FILE, "execution_spec.json")):
        committed = subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + source], cwd=ROOT)
        check("committed_source:" + source, committed.replace(b"\r\n", b"\n")
              == (run / snapshot).read_bytes().replace(b"\r\n", b"\n")
              == (ROOT / source).read_bytes().replace(b"\r\n", b"\n"))
    check("executed_validator", Path(__file__).read_bytes() == (ROOT / ("validate_" + EXPERIMENT + ".py")).read_bytes())
    check("training_snapshot", sha(run / "training_script.py") == manifest["training_script_sha256"])
    check("frozen_spec", read_json(run / "execution_spec.json") == spec)
    required = {"training_script.py", "validator_script.py", "execution_spec.json", "identity_audit.json",
                "secondary_evidence.npz", "baseline_reproduction.json", "model_inventory.json", "model.sha256",
                "metrics.json", "candidates.csv", "fold_metrics.csv", "trade_ledger.csv", "report.md",
                "source_provenance.json", "operational_safety_pre.json", "operational_safety_post.json", "environment.txt"}
    artifacts = [item["path"] for item in manifest["artifacts"]]
    check("artifact_inventory", len(artifacts) == len(set(artifacts)) and required <= set(artifacts))
    for item in manifest["artifacts"]:
        path = (run / item["path"]).resolve()
        check("artifact:" + item["path"], path.is_relative_to(run) and sha(path) == item["sha256"])
    check("no_selection_or_promotion", manifest["promotion"] == {"requested": False, "replacement_authorized": False,
          "operational_artifact_changed": False, "gate_result": "research_only"})
    check("no_tuning", manifest["search"]["performed"] is False
          and manifest["search"]["predefined_candidates"] == spec["candidates"])
    check("parameters_seed", manifest["model"]["parameters"] == spec["parameters"]
          and manifest["random_seeds"] == {"secondary": 42})
    before = operational_hashes()
    check("operational_pre_post", before == read_json(run / "operational_safety_pre.json")
          == read_json(run / "operational_safety_post.json"))
    archive, helper, parent = baseline_inputs(spec)
    history, folds, identities = helper.rebuild(parent)
    check("independent_feature_timestamp_target_identities", identities == read_json(run / "identity_audit.json"))
    target = parent.build_execution_aligned_labels(history).C1_TARGET.to_numpy(dtype=np.int8)
    for number, _, train, _ in folds:
        check(f"c1_target_before_fitting:{number}", array_hash(target[train]) == spec["c1_target_sha256"][f"fold{number}_train"])
    check("source_provenance", read_json(run / "source_provenance.json") == {
        "spec_sha256": SPEC_SHA256, "reference_run": spec["reference_run"],
        "reference_finalized_sha256": spec["reference_finalized_sha256"],
        "b0_models": spec["b0_models"], "source_hashes": spec["source_hashes"]})
    inventory = read_json(run / "model_inventory.json")
    check("three_isolated_models", len(inventory) == 3 and [item["fold"] for item in inventory] == [row[0] for row in spec["folds"]])
    check("model_file_set", {p.relative_to(run).as_posix() for p in (run / "models").rglob("*") if p.is_file()}
          == {item["path"] for item in inventory} and {item["path"] for item in inventory} <= set(artifacts))
    check("model_manifest", manifest["model"]["trained"] is True and manifest["model"]["features"] == spec["features"]
          and manifest["model"]["feature_count"] == 31 and manifest["model"]["fold_models_trained"] == 3
          and manifest["model"]["artifact_path"] == inventory[-1]["path"]
          and manifest["model"]["artifact_sha256"] == inventory[-1]["sha256"]
          and (run / "model.sha256").read_text().split()[0] == inventory[-1]["sha256"])
    check("secondary_training_counts", manifest["data"]["train_rows"] == sum(item["train_rows"] for item in inventory)
          and manifest["data"]["secondary_training_rows_by_fold"] == {item["fold"]: item["train_rows"] for item in inventory})
    parent_model = read_json(ROOT / "training_runs" / spec["c1_run"] / "manifest.json")["model"]
    check("target_execution_semantics", all(manifest["model"][key] == parent_model[key] for key in
          ("horizon", "label_tp_sl_semantics", "execution_tp_sl_semantics"))
          and manifest["model"]["calibration_method"] == "none")
    frames, primary_parts, secondary_parts, fold_parts, keys = [], [], [], [], set()
    with np.load(run / "secondary_evidence.npz", allow_pickle=False) as evidence, np.load(
        archive / "paired_oof_predictions.npz", allow_pickle=False
    ) as archived:
        for number, name, train, score in folds:
            key = f"fold{number}"
            def array_check(suffix, expected):
                full = key + "_" + suffix
                keys.add(full)
                actual = evidence[full]
                check("array:" + full, actual.dtype == expected.dtype and np.array_equal(actual, expected))
            ns = history.TIME_DT.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            b0_item = next(item for item in spec["b0_models"] if item["fold"] == name)
            check("b0_model_hash:" + key, sha(archive / b0_item["path"]) == b0_item["sha256"])
            b0_model = xgb.XGBClassifier()
            b0_model.load_model(archive / b0_item["path"])
            check("b0_features:" + key, b0_model.get_booster().feature_names == spec["features"])
            train_x = history.loc[train, spec["features"]].astype(np.float32)
            score_x = history.loc[score, spec["features"]].astype(np.float32)
            b0_train, b0_score = (b0_model.predict_proba(x)[:, 1] for x in (train_x, score_x))
            check("archived_b0_scores:" + key, np.array_equal(score, archived[key + "_indices"])
                  and np.array_equal(b0_score, archived["B0_31_technical_" + key]))
            subset = independent_subset(train, score, ns[train], ns[score], b0_train)
            for suffix, expected in (("train_indices", train), ("score_indices", score), ("subset_indices", subset),
                                     ("b0_train", b0_train), ("b0_score", b0_score), ("subset_target", target[subset]),
                                     ("train_ns", ns[train]), ("score_ns", ns[score])):
                array_check(suffix, expected)
            x = history.loc[subset, spec["features"]].astype(np.float32)
            item = inventory[number - 1]
            path = (run / item["path"]).resolve()
            check("secondary_artifact:" + key, path.is_relative_to(run / "models") and sha(path) == item["sha256"])
            check("fitting_identity:" + key, item["parameters"] == spec["parameters"] and item["features"] == spec["features"]
                  and item["seed"] == 42 and item["rounds"] == 220 and item["train_rows"] == len(subset)
                  and item["train_indices_sha256"] == array_hash(subset) and item["x_sha256"] == array_hash(x.to_numpy())
                  and item["target_sha256"] == array_hash(target[subset]))
            independently_fitted, weights = independent_fit(x, target[subset], spec)
            array_check("sample_weight", weights)
            check("weights_identity:" + key, item["weights_sha256"] == array_hash(weights))
            check("exact_frozen_training_config:" + key,
                  item["training_config"] == json.loads(independently_fitted.get_booster().save_config()))
            model = xgb.XGBClassifier()
            model.load_model(path)
            check("secondary_structure:" + key, model.get_booster().num_boosted_rounds() == 220
                  and model.get_booster().feature_names == spec["features"] and model.get_booster().num_features() == 31)
            # Saved trees and learned intercept must match a fit on exactly the audited subset.
            actual_raw = json.loads(model.get_booster().save_raw(raw_format="json"))
            expected_raw = json.loads(independently_fitted.get_booster().save_raw(raw_format="json"))
            for field in ("gradient_booster", "learner_model_param", "objective", "feature_names", "feature_types"):
                check("exact_model:" + key + ":" + field, actual_raw["learner"][field] == expected_raw["learner"][field])
            values = model.predict_proba(score_x)[:, 1]
            array_check("secondary_score", values)
            check("independently_fitted_predictions:" + key,
                  np.array_equal(values, independently_fitted.predict_proba(score_x)[:, 1]))
            frame = history.loc[score, helper.PRICE_COLUMNS].copy()
            frame["buy_prob"], frame["sell_prob"] = b0_score, np.float32(0)
            frames.append(frame)
            primary_parts.append(b0_score)
            secondary_parts.append(values)
            fold_parts.append(np.full(len(score), number, dtype=np.int8))
        frame = pd.concat(frames, ignore_index=True)
        b0, specialist, fold_ids = np.concatenate(primary_parts), np.concatenate(secondary_parts), np.concatenate(fold_parts)
        output, ledgers, candidate_rows = {}, [], []
        for candidate in CANDIDATES:
            primary, secondary, union = independent_gates(b0, specialist, candidate)
            for suffix, expected in (("primary", primary), ("secondary", secondary), ("union", union)):
                full = candidate[0] + "_" + suffix
                keys.add(full)
                check("gate:" + full, evidence[full].dtype == np.bool_ and np.array_equal(evidence[full], expected))
            trades = replay(parent, frame, candidate[0], primary, secondary, fold_ids)
            result = audited_metrics(helper, trades, candidate[0], primary, secondary, fold_ids, spec)
            if candidate == CANDIDATES[0]:
                for metric, expected in spec["baseline"].items():
                    check("baseline:" + metric, equal(result["metrics"][-1][metric], expected))
                check("baseline_pretraining_evidence", equal(read_json(run / "baseline_reproduction.json"), result))
            output[candidate[0]] = result
            candidate_rows.append({**result["metrics"][-1], **result["assessment"],
                                   "pnl": result["metrics"][-1]["pnl_r"], "max_dd": result["metrics"][-1]["max_dd_r"],
                                   "qualification_verdict": "interesting_research_only" if result["assessment"]["interesting"] else "not_interesting"})
            ledgers.extend({**trade, "candidate_id": candidate[0]} for trade in trades)
        check("exact_evidence_keys", set(evidence.files) == keys)
    distributions = {}
    for i, name in [(n + 1, row[0]) for n, row in enumerate(spec["folds"])] + [(0, "pooled")]:
        mask = fold_ids == i if i else np.ones(len(b0), dtype=bool)
        distributions[name] = {"all_score_rows": score_summary(specialist[mask]),
                               "secondary_eligible_rows": score_summary(specialist[mask & (b0 < .75)])}
    expected_metrics = {"results": output, "score_distribution": distributions, "baseline_reproduction": True,
                        "selected_candidate": None, "production_promotion_requested": False, "model_training": True,
                        "adaptive_search": False, "classification": "historical_development_only"}
    check("all_metrics_diagnostics_assessments", equal(read_json(run / "metrics.json"), expected_metrics))
    verify_csv(run / "candidates.csv", candidate_rows)
    verify_csv(run / "fold_metrics.csv", [row for result in output.values() for row in result["metrics"]])
    verify_csv(run / "trade_ledger.csv", ledgers)
    check("all_csv_and_exact_s5_ledger", True)
    check("report_metrics", equal(json.loads((run / "report.md").read_text(encoding="utf-8").split("```json\n")[1].split("\n```")[0]), expected_metrics))
    check("final_operational_files", operational_hashes() == before)
    return expected_metrics


def validate(run):
    run = run.resolve()
    require(run.parent == ROOT / "training_runs" and (run / "manifest.json").is_file()
            and not (run / "FINALIZED.json").exists(), "Unfinalized run required")
    require(not any((run / name).exists() for name in ("validator.json", "validator.md", "validator_attempt.json")), "Do not retry validation")
    # Remains after interrupts as well as failures; no automatic second attempt.
    with (run / "validator_attempt.json").open("x", encoding="utf-8") as stream:
        json.dump({"validator_sha256": sha(Path(__file__)), "one_shot": True}, stream)
    checks, output = {}, {}
    before = operational_hashes()
    try:
        output = verify(run, checks)
    except Exception as error:
        checks[f"{type(error).__name__}: {error}"] = False
    finally:
        checks["validator_operational_safety"] = before == operational_hashes()
    failed = [name for name, passed in checks.items() if not passed]
    verdict = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed,
               "checks": checks, "independent_metrics": output,
               "scope": "methodology/provenance only; historical development, not untouched promotion evidence"}
    write_json(run / "validator.json", verdict)
    (run / "validator.md").write_text("# Independent secondary classifier validator\n\nOverall: " + verdict["overall"]
        + "\n\nFailed checks: " + repr(failed)
        + "\n\nHistorical development only; no candidate selection or production promotion. No retries.\n", encoding="utf-8")
    manifest = read_json(run / "manifest.json")
    manifest["registry"]["validator_result"] = "independent " + verdict["overall"]
    for name in ("validator_attempt.json", "validator.json", "validator.md"):
        manifest["artifacts"].append({"kind": "independent_validator", "path": name,
                                      "sha256": sha(run / name), "retention_status": "stored_in_run_directory"})
    write_json(run / "manifest.json", manifest)
    print("INDEPENDENT_VALIDATOR_" + verdict["overall"])
    print("failed_check_names=" + repr(failed))
    return not failed


def self_test():
    spec = specification()
    assert spec["candidates"] == [dict(candidate_id=n, secondary_threshold=t) for n, t in CANDIDATES]
    assert len(CANDIDATES) == 5
    for candidate in CANDIDATES:
        threshold = candidate[1] or .6
        primary, secondary, union = independent_gates(
            [.749, .75, .9, .1], [threshold, 1., 1., np.nextafter(threshold, 0.)], candidate)
        assert primary.tolist() == [False, True, True, False]
        assert secondary.tolist() == [candidate[1] is not None, False, False, False]
        assert np.array_equal(union, primary | secondary) and not np.any(primary & secondary)
    train, score = np.arange(4), np.arange(4, 6)
    assert independent_subset(train, score, train, score, [.74, .75, .9, .1]).tolist() == [0, 3]

    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Invalid evidence accepted")

    rejects(lambda: independent_subset(train, score, train + 10, score, [.1] * 4))
    rejects(lambda: independent_subset(train, np.array([3, 4]), train, score, [.1] * 4))
    rejects(lambda: independent_gates([.5], [np.nan], CANDIDATES[1]))
    rejects(lambda: independent_gates([.5], [.9], ("extra", .55)))
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(1200, 31)).astype(np.float32), columns=spec["features"])
    y = (x.iloc[:, 0].to_numpy() > 0).astype(np.int8)
    a, weights = independent_fit(x, y, spec)
    b, repeat_weights = independent_fit(x.copy(), y.copy(), spec)
    assert np.array_equal(weights, repeat_weights)
    assert a.get_booster().save_raw(raw_format="json") == b.get_booster().save_raw(raw_format="json")
    assert np.array_equal(a.predict_proba(x), b.predict_proba(x))
    rejects(lambda: independent_fit(x, np.zeros(len(x), dtype=np.int8), spec))
    assert score_summary([])["count"] == 0 and score_summary([])["max"] is None
    assert math.isclose(score_summary([.1, .3])["mean"], .2, abs_tol=1e-12)
    assert equal({"a": [1., None]}, {"a": [1. + 1e-13, None]})
    assert not equal({"a": [1.]}, {"a": [1. + 1e-8]})
    # Synthetic serialization and prediction persistence; never a training run.
    import tempfile
    from types import SimpleNamespace
    import gold_gemini_execution_semantics_v1 as s5
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic.json"
        a.save_model(path)
        loaded = xgb.XGBClassifier()
        loaded.load_model(path)
        assert np.array_equal(loaded.predict_proba(x), a.predict_proba(x))
        assert json.loads(a.get_booster().save_config()) == json.loads(b.get_booster().save_config())
        actual_raw = json.loads(loaded.get_booster().save_raw(raw_format="json"))
        expected_raw = json.loads(b.get_booster().save_raw(raw_format="json"))
        for field in ("gradient_booster", "learner_model_param", "objective", "feature_names", "feature_types"):
            assert actual_raw["learner"][field] == expected_raw["learner"][field]
        rows = [{"candidate_id": "fixture", "value": .5, "empty": None, "flags": ["x"], "passed": True,
                 "infinite": math.inf}]
        csv_path = Path(folder) / "fixture.csv"
        write_csv(csv_path, rows)
        verify_csv(csv_path, rows)
        rejects(lambda: verify_csv(csv_path, [{**rows[0], "value": .6}]))
    day = pd.Timestamp("2020-01-06")
    while day.dayofweek not in s5.ALLOWED_WEEKDAYS:
        day += pd.Timedelta(days=1)
    day += pd.Timedelta(hours=min(s5.ALLOWED_HOURS))
    b0 = np.array([.75, .2, .2, .2, .2])
    frame = pd.DataFrame({"TIME_DT": pd.date_range(day, periods=5, freq="min"), "OPEN": 100.,
        "HIGH": [100., 100., 110., 110., 110.], "LOW": 100., "CLOSE": 100., "ATR": 1.,
        "M1_RSI": 50., "SPREAD": 1., "buy_prob": b0, "sell_prob": 0.})
    primary, secondary, _ = independent_gates(b0, np.ones(5), CANDIDATES[0])
    actual = replay(SimpleNamespace(semantics=s5), frame, "fixture", primary, secondary, np.array([1, 1, 2, 2, 2]))
    expected, _ = s5.simulate(s5.finalize_cohort(frame, "fixture", 0), s5.SIMULATORS[-1])
    assert [{k: v for k, v in t.items() if k not in ("entry_kind", "score_fold")} for t in actual] == expected
    primary, secondary, _ = independent_gates(b0, np.full(5, .65), CANDIDATES[2])
    actual = replay(SimpleNamespace(semantics=s5), frame, "fixture", primary, secondary, np.array([1, 1, 2, 2, 2]))
    assert [t["entry_kind"] for t in actual] == ["primary", "secondary", "secondary"]
    assert actual[0]["exit_index"] == 2 and actual[0]["score_fold"] == 1
    print("SELF_TEST_PASS: independent gates, exclusion, equality, isolation, deterministic fit, persistence, A0, S5, CSV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    return 0 if validate(args.run_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
