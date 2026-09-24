"""Frozen S4 confirmation research; formal work requires a manually created run."""
from __future__ import annotations

import argparse
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
EXPERIMENT = "gold_secondary_s4_confirmation_v1"
SPEC_FILE = "execution_spec_" + EXPERIMENT + ".json"
CANDIDATES = ("A0_BASELINE_075", "S4_CONFIRMATION")
SPEC_SHA256 = 'e7e261e095aa16fd7c0c189c6e3bfd28873b0fcb6f8e4a0296e12ddc932648af'


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


def discovery_identity(spec):
    archive = ROOT / "training_runs" / spec["discovery_run"]
    require(sha(archive / "FINALIZED.json") == spec["discovery_finalized_sha256"], "Discovery finalized identity")
    final = read_json(archive / "FINALIZED.json")
    require(final["run_id"] == archive.name and final["finalized_at_utc"], "Discovery run identity")
    for name, expected in final["file_sha256"].items():
        path = (archive / name).resolve()
        require(path.is_relative_to(archive) and sha(path) == expected, "Discovery artifact: " + name)
    require(sha(archive / "execution_spec.json") == spec["discovery_execution_spec_sha256"], "Discovery spec identity")
    ds = read_json(archive / "execution_spec.json")
    for key in ("parameters", "features", "xgboost_version", "sample_weight", "fit_input", "seed", "model_count",
                "folds", "train_window_months", "feature_matrix_sha256", "c1_target_sha256", "blocks", "b0_models", "baseline", "guardrails"):
        require(ds[key] == spec[key], "Frozen discovery definition: " + key)
    require(next(c for c in ds["candidates"] if c["candidate_id"] == "S4_SECONDARY_P075")["secondary_threshold"] == .75,
            "S4 predeclared threshold")
    manifest = read_json(archive / "manifest.json")
    require(manifest["git_commit"] == spec["discovery_source_commit"], "Discovery source commit")
    committed = subprocess.check_output(["git", "show", manifest["git_commit"] + ":execution_spec_gold_independent_secondary_classifier_v1.json"], cwd=ROOT)
    require(json.loads(committed) == ds, "Pre-run committed S4 spec")
    validator = read_json(archive / "validator.json")
    require(validator["overall"] == "PASS" and validator["failed_check_names"] == [], "Original discovery validator PASS")
    promotion = manifest["promotion"]
    require(not promotion["requested"] and not promotion["replacement_authorized"]
            and not promotion["operational_artifact_changed"], "Discovery no promotion")
    metrics = read_json(archive / "metrics.json")
    require(metrics["selected_candidate"] is None and not metrics["production_promotion_requested"], "Discovery selection")
    require([name for name, row in metrics["results"].items() if row["assessment"]["interesting"]]
            == ["S4_SECONDARY_P075"], "Only discovery S4 interesting")
    result = metrics["results"]["S4_SECONDARY_P075"]
    for key, expected in spec["discovery_reference"].items():
        actual = result["assessment"].get(key, result["metrics"][-1].get(key))
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), "Discovery reference: " + key)
    return archive, ds


def fixed_gate(b0, specialist, candidate, threshold=.75):
    require(candidate in CANDIDATES and threshold == .75, "Only fixed S4 threshold 0.75")
    b0, specialist = np.asarray(b0), np.asarray(specialist)
    require(b0.ndim == specialist.ndim == 1 and b0.shape == specialist.shape, "Gate alignment")
    require(np.isfinite(b0).all() and np.isfinite(specialist).all()
            and ((b0 >= 0) & (b0 <= 1) & (specialist >= 0) & (specialist <= 1)).all(), "Probability range")
    primary = b0 >= .75
    secondary = (b0 < .75) & (specialist >= .75) if candidate == CANDIDATES[1] else np.zeros(len(b0), dtype=bool)
    require(not (primary & secondary).any(), "Signal overlap")
    return primary, secondary, primary | secondary


def slice_windows(spec):
    windows, early, late = {}, [], []
    for name, start, end in spec["folds"]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        midpoint = start + (end - start) / 2
        windows[name] = [(start, end)]
        early.append((start, midpoint))
        late.append((midpoint, end))
    for (name, _, _), first, second in zip(spec["folds"], early, late):
        windows[name + "_H1"], windows[name + "_H2"] = [first], [second]
    windows["ALL_EARLY"], windows["ALL_LATE"] = early, late
    windows["pooled"] = [interval for name, _, _ in spec["folds"] for interval in windows[name]]
    return windows


def select_trades(trades, intervals):
    return [trade for trade in trades if any(start <= pd.Timestamp(trade["entry_time_api"]) < end for start, end in intervals)]


def duration_days(intervals):
    return sum((end - start).total_seconds() / 86400 for start, end in intervals)


def trade_identity(trade):
    return int(trade["score_fold"]), int(trade["source_entry_index"])


def wins_losses(trades):
    wins = sum(float(t["net_r"]) > 0 for t in trades)
    return {"trades": len(trades), "wins": wins, "losses": len(trades) - wins,
            "realized_wr": wins / len(trades) if trades else None}


def decomposition(trades, days):
    result = {}
    for kind in ("primary", "secondary"):
        stats = wins_losses([t for t in trades if t["entry_kind"] == kind])
        result[kind + "_trade_count"] = stats.pop("trades")
        result.update({kind + "_" + key: value for key, value in stats.items()})
        result[kind + "_trades_per_day"] = result[kind + "_trade_count"] / days
    return result


def displacement(baseline, s4, intervals):
    all_primary = {trade_identity(t): t for t in s4 if t["entry_kind"] == "primary"}
    a, b = select_trades(baseline, intervals), select_trades(s4, intervals)
    require(len({trade_identity(t) for t in baseline}) == len(baseline)
            and len({trade_identity(t) for t in s4}) == len(s4), "Duplicate trade identity")
    missing = [t for t in a if trade_identity(t) not in all_primary]
    retained = len(a) - len(missing)
    occupancy = []
    for trade in missing:
        row = int(trade["entry_index"])
        # Timeout closes before the entry phase; TP/SL closes after it.
        if any(int(t["entry_index"]) < row <= int(t["exit_index"])
               and not (t["exit_reason"] == "timeout" and row == int(t["exit_index"])) for t in s4):
            occupancy.append(trade)
    baseline_ids = {trade_identity(t) for t in baseline}
    new_primary = [t for t in b if t["entry_kind"] == "primary" and trade_identity(t) not in baseline_ids]
    secondary = [t for t in b if t["entry_kind"] == "secondary"]
    require(len(b) - len(a) == len(secondary) + len(new_primary) - len(missing), "Displacement accounting")
    def displaced_stats(trades):
        stats = wins_losses(trades)
        return {"baseline_primary_entries_displaced": stats["trades"],
                "displaced_primary_wins": stats["wins"], "displaced_primary_losses": stats["losses"],
                "displaced_primary_realized_wr": stats["realized_wr"],
                "identities": [list(trade_identity(t)) for t in trades]}
    occupied_ids = {trade_identity(t) for t in occupancy}
    stats = wins_losses(secondary)
    return {"baseline_trade_count": len(a), "s4_trade_count": len(b),
            "baseline_primary_entries_retained": retained, **displaced_stats(missing),
            "new_secondary_entries_executed": len(secondary), "new_primary_entries_executed": len(new_primary),
            "secondary_wins": stats["wins"], "secondary_losses": stats["losses"], "secondary_realized_wr": stats["realized_wr"],
            "occupancy_displacement": displaced_stats(occupancy),
            "nonoccupancy_displacement": displaced_stats([t for t in missing if trade_identity(t) not in occupied_ids])}


def stress_summary(trades, extra_cost):
    require(extra_cost in (0., .01, .02), "Frozen post-trade costs only")
    values = np.array([t["net_r"] for t in trades], dtype=np.float64) - extra_cost
    positive, negative = values[values > 0], values[values <= 0]
    profit, loss = float(positive.sum()), float(-negative.sum())
    curve = np.r_[0., np.cumsum(values)]
    return {"trades": len(values), "wins": len(positive), "losses": len(negative),
            "realized_wr": len(positive) / len(values) if len(values) else 0.,
            "pf": profit / loss if loss > 0 else (math.inf if profit > 0 else 0.),
            "mean_r": float(values.mean()) if len(values) else 0., "pnl_r": float(values.sum()),
            "max_dd_r": float((curve - np.maximum.accumulate(curve)).min())}


def confirmation_gates(rows, deltas, stresses, spec):
    s4 = {r["scope"]: r for r in rows if r["candidate_id"] == CANDIDATES[1]}
    delta = {r["scope"]: r for r in deltas}
    pooled = s4["pooled"]
    failures = []
    for scope, limits in [("pooled", spec["guardrails"]["pooled"])] + [
            (name, spec["guardrails"]["fold"]) for name, _, _ in spec["folds"]]:
        for key, minimum in limits.items():
            value = s4[scope][key]
            if value is None or math.isnan(value) or value < minimum:
                failures.append(scope + ":" + key)
    halves = [name + suffix for name, _, _ in spec["folds"] for suffix in ("_H1", "_H2")]
    good = sum(delta[name]["delta_wr"] >= 0 and delta[name]["delta_trades_per_day"] >= 0 for name in halves)
    checks = {
        "pooled_wr_improves": delta["pooled"]["delta_wr"] > 0,
        "pooled_frequency_improves": delta["pooled"]["delta_trades_per_day"] > 0,
        "original_guardrails": not failures,
        "four_of_six_halves": good >= 4,
        "early_not_both_worse": not (delta["ALL_EARLY"]["delta_wr"] < 0 and delta["ALL_EARLY"]["delta_trades_per_day"] < 0),
        "late_not_both_worse": not (delta["ALL_LATE"]["delta_wr"] < 0 and delta["ALL_LATE"]["delta_trades_per_day"] < 0),
    }
    for name, tolerance in (("R1_EXTRA_COST", .03), ("R2_EXTRA_COST", .05)):
        pair = {r["candidate_id"]: r["pf"] for r in stresses if r["scope"] == "pooled" and r["scenario"] == name}
        checks[name] = pair[CANDIDATES[1]] >= pair[CANDIDATES[0]] - tolerance
    passed = all(checks.values())
    return {"confirmation_primary_pass": passed, "confirmation_supportive": passed,
            "confirmation_strong": passed and pooled["realized_wr"] >= .58 and pooled["trades_per_day"] >= .40,
            "checks": checks, "guardrail_failures": failures, "nonnegative_half_count": good}


def robustness(ledgers, old, spec):
    windows = slice_windows(spec)
    rows, stresses = [], []
    for candidate in CANDIDATES:
        for scope, intervals in windows.items():
            trades = select_trades(ledgers[candidate], intervals)
            days = duration_days(intervals)
            rows.append({"candidate_id": candidate, "scope": scope, "days": days,
                         **old.trade_metrics(trades, days), **decomposition(trades, days)})
            for scenario, cost in spec["stress_scenarios"].items():
                stresses.append({"candidate_id": candidate, "scope": scope, "scenario": scenario,
                                 "extra_cost_r": cost, **stress_summary(trades, cost)})
    by_key = {(r["candidate_id"], r["scope"]): r for r in rows}
    deltas = []
    for scope in windows:
        a, b = (by_key[(candidate, scope)] for candidate in CANDIDATES)
        deltas.append({"scope": scope, **{"delta_" + suffix: b[key] - a[key] for suffix, key in (
            ("wr", "realized_wr"), ("trades_per_day", "trades_per_day"), ("pf", "pf"),
            ("mean_r", "mean_r"), ("cost_stress_pf", "cost_stress_pf"))}})
    displaced = {scope: displacement(ledgers[CANDIDATES[0]], ledgers[CANDIDATES[1]], intervals)
                 for scope, intervals in windows.items()}
    return {"metrics": rows, "deltas": deltas, "stress_metrics": stresses, "displacement": displaced,
            "slice_boundaries": {name: [[a.isoformat(), b.isoformat()] for a, b in spans] for name, spans in windows.items()},
            "confirmation": confirmation_gates(rows, deltas, stresses, spec),
            "selected_candidate": None, "production_promotion_requested": False, "classification": spec["classification"]}


def execute(old, discovery, frame, b0, secondary, fold_ids, source_indices, candidate):
    primary, extra, union = fixed_gate(b0, secondary, candidate)
    cohort = discovery.gated_cohort(old.semantics, frame, candidate, union)
    trades, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
    for trade in trades:
        i = int(trade["entry_index"])
        require(union[i], "Executed gate mismatch")
        trade.update(entry_kind="primary" if primary[i] else "secondary", score_fold=int(fold_ids[i]),
                     source_entry_index=int(source_indices[i]))
    return trades, (primary, extra, union)


def formal_run(run):
    run = run.resolve()
    spec = specification()
    require(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(), "New unfinalized run required")
    manifest = read_json(run / "manifest.json")
    require(manifest["experiment_name"] == EXPERIMENT and manifest["run_id"] == run.name
            and manifest["status"] == "in_progress" and manifest["git_dirty"] is False, "Formal manifest")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).rstrip()
    require(git("rev-parse", "HEAD") == git("rev-parse", "@{u}") == manifest["git_commit"], "Git provenance")
    require(sha(run / "training_script.py") == sha(Path(__file__)) == manifest["training_script_sha256"], "Runner snapshot")
    prefix = run.relative_to(ROOT).as_posix() + "/"
    require(all(line[3:].strip('"').startswith(prefix) for line in git("status", "--porcelain", "--untracked-files=all").splitlines()), "Dirty paths")
    require(not any((run / name).exists() for name in ("execution_spec.json", "models", "failure.json")), "Do not retry formal execution")
    before = operational_hashes()
    with (run / "execution_spec.json").open("x", encoding="utf-8") as stream:
        stream.write((ROOT / SPEC_FILE).read_text(encoding="utf-8"))
    write_json(run / "operational_safety_pre.json", before)
    (run / "validator_script.py").write_bytes((ROOT / ("validate_" + EXPERIMENT + ".py")).read_bytes())
    try:
        discovery_archive, ds = discovery_identity(spec)
        discovery = load_module(discovery_archive / "training_script.py", "s4_frozen_discovery")
        b0_archive, helper, old = discovery.frozen_inputs(ds)
        history, target, folds, identities = helper.reconstruct(old)
        write_json(run / "identity_audit.json", identities)
        require(xgb.__version__ == "3.2.0", "XGBoost version")
        ns = history.TIME_DT.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        evidence, frames, b0_parts, fold_parts, index_parts = {}, [], [], [], []
        for number, name, train, score in folds:
            key = f"fold{number}"
            require(array_hash(target[train]) == spec["c1_target_sha256"][key + "_train"], "Exact C1 target")
            item = next(item for item in spec["b0_models"] if item["fold"] == name)
            require(sha(b0_archive / item["path"]) == item["sha256"], "B0 model identity")
            model = xgb.XGBClassifier()
            model.load_model(b0_archive / item["path"])
            require(model.get_booster().feature_names == spec["features"], "B0 feature order")
            b0_train = model.predict_proba(history.loc[train, spec["features"]].astype(np.float32))[:, 1]
            b0_score = model.predict_proba(history.loc[score, spec["features"]].astype(np.float32))[:, 1]
            subset = discovery.training_subset(train, score, ns[train], ns[score], b0_train)
            evidence.update({key + "_train_indices": train, key + "_score_indices": score, key + "_subset_indices": subset,
                             key + "_b0_train": b0_train, key + "_b0_score": b0_score, key + "_subset_target": target[subset],
                             key + "_train_ns": ns[train], key + "_score_ns": ns[score]})
            frame = history.loc[score, helper.PRICE_COLUMNS].copy()
            frame["buy_prob"], frame["sell_prob"] = b0_score, np.float32(0)
            frames.append(frame)
            b0_parts.append(b0_score)
            fold_parts.append(np.full(len(score), number, dtype=np.int8))
            index_parts.append(score)
        frame, b0 = pd.concat(frames, ignore_index=True), np.concatenate(b0_parts)
        fold_ids, indices = np.concatenate(fold_parts), np.concatenate(index_parts)
        baseline, baseline_gates = execute(old, discovery, frame, b0, np.zeros_like(b0), fold_ids, indices, CANDIDATES[0])
        baseline_rows = helper.metric_rows(old, baseline, CANDIDATES[0])
        for key, expected in spec["baseline"].items():
            require(math.isclose(baseline_rows[-1][key], expected, rel_tol=0, abs_tol=1e-12), "Baseline reproduction: " + key)
        write_json(run / "baseline_reproduction.json", {"pass": True, "metrics": baseline_rows})
        (run / "models").mkdir()
        inventory, secondary_parts = [], []
        for number, name, _, score in folds:
            key = f"fold{number}"
            subset = evidence[key + "_subset_indices"]
            x = history.loc[subset, spec["features"]].astype(np.float32)
            model, weights = discovery.fit_secondary(x, target[subset], ds)
            path = run / "models" / ("secondary_" + name + ".json")
            model.save_model(path)
            values = model.predict_proba(history.loc[score, spec["features"]].astype(np.float32))[:, 1]
            evidence[key + "_secondary_score"], evidence[key + "_sample_weight"] = values, weights
            secondary_parts.append(values)
            inventory.append({"fold": name, "path": path.relative_to(run).as_posix(), "sha256": sha(path),
                              "parameters": spec["parameters"], "features": spec["features"], "seed": 42,
                              "rounds": model.get_booster().num_boosted_rounds(), "training_config": json.loads(model.get_booster().save_config()),
                              "train_rows": len(subset), "train_indices_sha256": array_hash(subset), "x_sha256": array_hash(x.to_numpy()),
                              "target_sha256": array_hash(target[subset]), "weights_sha256": array_hash(weights)})
            write_json(run / "model_inventory.json", inventory)
        # Only fresh arrays enter confirmation execution. Discovery arrays are reference checks.
        secondary = np.concatenate(secondary_parts)
        s4, s4_gates = execute(old, discovery, frame, b0, secondary, fold_ids, indices, CANDIDATES[1])
        ledgers = dict(zip(CANDIDATES, (baseline, s4)))
        output = robustness(ledgers, old, spec)
        equivalence = {}
        discovery_inventory = read_json(discovery_archive / "model_inventory.json")
        with np.load(discovery_archive / "secondary_evidence.npz", allow_pickle=False) as previous:
            for number, _, _, _ in folds:
                key = f"fold{number}"
                comparisons = {suffix: np.array_equal(evidence[key + "_" + suffix], previous[key + "_" + suffix])
                               for suffix in ("train_indices", "score_indices", "subset_indices", "subset_target", "sample_weight", "b0_train", "b0_score", "secondary_score")}
                comparisons["training_config"] = inventory[number - 1]["training_config"] == discovery_inventory[number - 1]["training_config"]
                require(all(comparisons.values()), "Deterministic discovery equivalence: " + key)
                equivalence[key] = comparisons
        for name, gates in zip(CANDIDATES, (baseline_gates, s4_gates)):
            for suffix, values in zip(("primary", "secondary", "union"), gates):
                evidence[name + "_" + suffix] = values
        np.savez_compressed(run / "secondary_evidence.npz", **evidence)
        output["deterministic_discovery_equivalence"] = equivalence
        write_json(run / "metrics.json", output)
        write_json(run / "displacement_metrics.json", output["displacement"])
        deltas = {row["scope"]: row for row in output["deltas"]}
        rows = [{**row, **{key: value for key, value in deltas[row["scope"]].items() if key != "scope"}}
                for row in output["metrics"]]
        helper.write_csv(run / "candidates.csv", [{**row, "pnl": row["pnl_r"], "max_dd": row["max_dd_r"],
            "qualification_verdict": "baseline" if row["candidate_id"] == CANDIDATES[0] else
            ("confirmation_supportive" if output["confirmation"]["confirmation_primary_pass"] else "confirmation_not_supportive")}
            for row in rows if row["scope"] == "pooled"])
        helper.write_csv(run / "fold_metrics.csv", [row for row in rows if row["scope"] in [f[0] for f in spec["folds"]] + ["pooled"]])
        helper.write_csv(run / "robustness_slices.csv", [row for row in rows if row["scope"] not in [f[0] for f in spec["folds"]] + ["pooled"]])
        helper.write_csv(run / "stress_metrics.csv", output["stress_metrics"])
        helper.write_csv(run / "trade_ledger.csv", [{**trade, "candidate_id": name} for name in CANDIDATES for trade in ledgers[name]])
        (run / "report.md").write_text("# GOLD secondary S4 confirmation v1\n\nHistorical development robustness only; no selection or promotion.\n"
            "Fresh fits on the same frozen historical windows do not create untouched evidence.\n"
            "Post-trade costs recompute wins as net R minus extra cost > 0.\n\n```json\n"
            + json.dumps(read_json(run / "metrics.json"), indent=2) + "\n```\n", encoding="utf-8")
        write_json(run / "source_provenance.json", {"spec_sha256": SPEC_SHA256, "discovery_run": spec["discovery_run"],
            "discovery_finalized_sha256": spec["discovery_finalized_sha256"], "reference_run": ds["reference_run"],
            "reference_finalized_sha256": ds["reference_finalized_sha256"], "b0_models": spec["b0_models"],
            "source_hashes": ds["source_hashes"], "discovery_predictions_used_for_execution": False})
        previous_manifest = read_json(discovery_archive / "manifest.json")
        manifest["data"] = previous_manifest["data"]
        manifest["model"] = {**previous_manifest["model"], "artifact_path": inventory[-1]["path"],
                             "artifact_sha256": inventory[-1]["sha256"]}
        (run / "model.sha256").write_text(inventory[-1]["sha256"] + "  " + inventory[-1]["path"] + "\n", encoding="utf-8")
        manifest["random_seeds"], manifest["random_seed_note"] = {"secondary": 42}, "Fresh isolated CPU single-thread confirmation fits"
        manifest["search"] = {"performed": False, "not_applicable_reason": "Only baseline and fixed predeclared S4; no tuning",
                              "candidate_results_file": "candidates.csv", "predefined_candidates": spec["candidates"]}
        manifest["promotion"] = {"requested": False, "replacement_authorized": False, "operational_artifact_changed": False, "gate_result": "research_only"}
        manifest["registry"] = {"parent_or_incumbent": spec["discovery_run"], "selected_configuration": "none selected; fixed S4 confirmation",
            "validator_result": "PENDING", **{key: "see candidates.csv" for key in ("trades_per_day", "realized_win_rate", "pf", "mean_r", "pnl", "max_dd")}}
    except Exception as error:
        write_json(run / "failure.json", {"status": "FAIL", "error": str(error)})
        raise
    finally:
        after = operational_hashes()
        write_json(run / "operational_safety_post.json", after)
        require(after == before, "Operational mutation")
    manifest["artifacts"] = [{"path": p.relative_to(run).as_posix(), "sha256": sha(p), "kind": "research_evidence",
                              "retention_status": "stored_in_run_directory"} for p in sorted(run.rglob("*"))
                             if p.is_file() and p.name not in {"manifest.json", "stdout.log"}]
    write_json(run / "manifest.json", manifest)
    print("FROZEN_S4_CONFIRMATION_COMPLETED")


def self_test():
    import copy
    import tempfile
    spec = specification()
    gate = fixed_gate
    windows_fn = slice_windows
    select_fn = select_trades
    displacement_fn = displacement
    stress_fn = stress_summary
    gates_fn = confirmation_gates
    assert spec['candidates'] == [{'candidate_id': CANDIDATES[0], 'secondary_threshold': None},
                                   {'candidate_id': CANDIDATES[1], 'secondary_threshold': .75}]

    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError('Invalid input accepted')

    p = np.array([.75, .9, .749, .1, .3], dtype=np.float32)
    s = np.array([1., 1., .75, .749, .8], dtype=np.float32)
    primary, extra, union = gate(p, s, CANDIDATES[1])
    assert primary.tolist() == [True, True, False, False, False]
    assert extra.tolist() == [False, False, True, False, True]
    assert np.array_equal(union, primary | extra) and not (primary & extra).any()
    assert np.array_equal(gate(p, s, CANDIDATES[0])[2], p >= .75)
    for threshold in (.60, .76, .77, .80):
        rejects(lambda: gate(p, s, CANDIDATES[1], threshold))
    rejects(lambda: gate(p, s, 'S4_SECONDARY_P075'))
    rejects(lambda: gate([np.nan], [.75], CANDIDATES[1]))
    windows = windows_fn(spec)
    assert len(windows) == 12
    assert windows['2018_2020_H1'][0][1] == pd.Timestamp('2019-07-03')
    assert windows['2021_2022_H1'][0][1] == pd.Timestamp('2022-01-01')
    assert windows['2023_2024_H1'][0][1] == pd.Timestamp('2024-01-01T12:00:00')
    assert sum((b-a).total_seconds()/86400 for a,b in windows['2023_2024_H1']) == 365.5
    probe = []
    for number, (name, _, _) in enumerate(spec['folds'], 1):
        start, end = windows[name][0]
        middle = windows[name+'_H1'][0][1]
        times = [start, middle - pd.Timedelta(nanoseconds=1), middle, end - pd.Timedelta(nanoseconds=1)]
        probe.extend({'entry_time_api': t.isoformat(), 'score_fold': number, 'source_entry_index': i} for i,t in enumerate(times))
        assert len(select_fn(probe, windows[name+'_H1'])) == 2
        assert len(select_fn(probe, windows[name+'_H2'])) == 2
    early = select_fn(probe, windows['ALL_EARLY'])
    late = select_fn(probe, windows['ALL_LATE'])
    assert len(early) == len(late) == 6 and len(select_fn(probe, windows['pooled'])) == 12
    assert {(t['score_fold'],t['source_entry_index']) for t in early}.isdisjoint(
        {(t['score_fold'],t['source_entry_index']) for t in late})

    def trade(row, kind='primary', net=1., end=None, reason='take_profit'):
        return {'entry_index':row, 'exit_index':row if end is None else end,
                'source_entry_index':row+100, 'score_fold':1, 'entry_kind':kind,
                'entry_time_api':(pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=row)).isoformat(),
                'net_r':net, 'exit_reason':reason}
    baseline = [trade(0),trade(2),trade(4,net=-1.),trade(8)]
    changed = [trade(0),trade(1,'secondary',-.5,2,'stop_loss'),trade(5),trade(7,'secondary',.1,8,'timeout')]
    d = displacement_fn(baseline, changed, windows['pooled'])
    assert d['baseline_primary_entries_retained'] == 1 and d['baseline_primary_entries_displaced'] == 3
    assert d['new_secondary_entries_executed'] == 2 and d['new_primary_entries_executed'] == 1
    assert d['displaced_primary_wins'] == 2 and d['displaced_primary_losses'] == 1
    assert d['occupancy_displacement']['baseline_primary_entries_displaced'] == 1
    assert d['occupancy_displacement']['displaced_primary_wins'] == 1
    assert d['nonoccupancy_displacement']['baseline_primary_entries_displaced'] == 2
    assert d['secondary_wins'] == d['secondary_losses'] == 1
    # Identical timestamps are not identities; fold and source row are authoritative.
    changed_identity = [dict(baseline[0], source_entry_index=999)]
    assert displacement_fn([baseline[0]], changed_identity, windows['pooled'])['baseline_primary_entries_retained'] == 0
    rejects(lambda: displacement_fn(baseline+baseline[:1],changed,windows['pooled']))
    cost_trades = [{'net_r':v} for v in [.005,.01,0.,-.1]]
    original = copy.deepcopy(cost_trades)
    assert stress_fn(cost_trades,0.)['wins'] == 2
    stressed = stress_fn(cost_trades,.01)
    assert stressed['wins'] == 0 and stressed['losses'] == 4
    assert math.isclose(stressed['pnl_r'], -.125, rel_tol=0, abs_tol=1e-12)
    assert cost_trades == original
    rejects(lambda: stress_fn(cost_trades,.03))

    rows = [{'candidate_id':CANDIDATES[1],'scope':scope,'trades':20,'realized_wr':.58,
             'trades_per_day':.40,'pf':.80,'mean_r':-.10690539315994348,'cost_stress_pf':.75}
            for scope in windows]
    deltas = [{'scope':scope,'delta_wr':.001,'delta_trades_per_day':.01} for scope in windows]
    stress = [{'candidate_id':c,'scope':'pooled','scenario':scenario,'pf':.9 if c==CANDIDATES[0] else .9-limit}
              for scenario,limit in [('R1_EXTRA_COST',.03),('R2_EXTRA_COST',.05)] for c in CANDIDATES]
    success = gates_fn(rows,deltas,stress,spec)
    assert success['confirmation_primary_pass'] and success['confirmation_supportive'] and success['confirmation_strong']
    halves=[name+half for name,_,_ in spec['folds'] for half in ('_H1','_H2')]
    fewer=copy.deepcopy(deltas)
    for d in fewer:
        if d['scope'] in halves[:2]: d['delta_wr']=-.01
    assert gates_fn(rows,fewer,stress,spec)['confirmation_primary_pass']
    for d in fewer:
        if d['scope']==halves[2]: d['delta_wr']=-.01
    assert not gates_fn(rows,fewer,stress,spec)['confirmation_primary_pass']
    for scope in ('pooled','ALL_EARLY','ALL_LATE'):
        bad=copy.deepcopy(deltas)
        for d in bad:
            if d['scope']==scope: d.update(delta_wr=-.01,delta_trades_per_day=-.01)
        assert not gates_fn(rows,bad,stress,spec)['confirmation_primary_pass']
    for key in ('delta_wr','delta_trades_per_day'):
        bad=copy.deepcopy(deltas)
        next(r for r in bad if r['scope']=='pooled')[key]=0.
        assert not gates_fn(rows,bad,stress,spec)['confirmation_primary_pass']
    for scope,key in [('pooled','pf'),('pooled','mean_r'),('pooled','cost_stress_pf'),
                      ('2018_2020','trades'),('2021_2022','realized_wr'),('2023_2024','pf')]:
        bad=copy.deepcopy(rows)
        next(r for r in bad if r['scope']==scope)[key]=-1.
        assert not gates_fn(bad,deltas,stress,spec)['confirmation_primary_pass']
    for scenario in ('R1_EXTRA_COST','R2_EXTRA_COST'):
        bad=copy.deepcopy(stress)
        next(r for r in bad if r['scenario']==scenario and r['candidate_id']==CANDIDATES[1])['pf']-=1e-6
        assert not gates_fn(rows,deltas,bad,spec)['confirmation_primary_pass']
    weak=copy.deepcopy(rows)
    next(r for r in weak if r['scope']=='pooled')['realized_wr']=.579
    assert gates_fn(weak,deltas,stress,spec)['confirmation_supportive']
    assert not gates_fn(weak,deltas,stress,spec)['confirmation_strong']

    # Read source snapshots only; never reconstruct historical data in a self-test.
    archive=ROOT/'training_runs'/spec['discovery_run']
    require(sha(archive/'FINALIZED.json')==spec['discovery_finalized_sha256'],'Test source archive identity')
    final=read_json(archive/'FINALIZED.json')
    source=archive/'training_script.py'
    require(sha(source)==final['file_sha256']['training_script.py'],'Test source snapshot')
    library=load_module(source,'s4_synthetic_frozen_helpers')
    ds=read_json(archive/'execution_spec.json')
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(1200,31)).astype(np.float32),columns=spec['features'])
    y=(x.iloc[:,0].to_numpy()>0).astype(np.int8)
    fit=library.fit_secondary
    first, weights=fit(x,y,ds)
    second, repeated=fit(x.copy(),y.copy(),ds)
    assert np.array_equal(weights,repeated)
    assert first.get_booster().save_raw(raw_format='json')==second.get_booster().save_raw(raw_format='json')
    assert np.array_equal(first.predict_proba(x),second.predict_proba(x))
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'synthetic.json'
        first.save_model(p)
        loaded=xgb.XGBClassifier()
        loaded.load_model(p)
        assert np.array_equal(loaded.predict_proba(x),first.predict_proba(x))
    print('SELF_TEST_PASS: fixed gates, midpoint slices, early/late, identities, displacement, cost crossings, deterministic fitting, A0, all confirmation gates')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    formal_run(args.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
