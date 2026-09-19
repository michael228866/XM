"""Independent one-shot S4 confirmation validator; no confirmation runner import."""
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


def independent_gate(b0, specialist, candidate, threshold=.75):
    require(candidate in CANDIDATES and threshold == .75, "Only frozen 0.75 threshold")
    p, s = np.asarray(b0), np.asarray(specialist)
    require(p.ndim == s.ndim == 1 and p.shape == s.shape, "Probability alignment")
    require(np.isfinite(p).all() and np.isfinite(s).all()
            and ((p >= 0) & (p <= 1) & (s >= 0) & (s <= 1)).all(), "Probability range")
    primary, extra = p >= .75, np.zeros(p.size, dtype=bool)
    if candidate == "S4_CONFIRMATION":
        locations = np.flatnonzero(p < .75)
        extra[locations] = s[locations] >= .75
    require(np.count_nonzero(primary & extra) == 0, "Overlap must be zero")
    return primary, extra, np.logical_or(primary, extra)


def independent_windows(spec):
    bounds = {}
    halves = []
    for name, start, end in spec["folds"]:
        lo, hi = pd.Timestamp(start).value, pd.Timestamp(end).value
        require((hi - lo) % 2 == 0 and hi > lo, "Exact calendar midpoint")
        middle = (lo + hi) // 2
        bounds[name] = [(pd.Timestamp(lo), pd.Timestamp(hi))]
        halves.append((name, (pd.Timestamp(lo), pd.Timestamp(middle)), (pd.Timestamp(middle), pd.Timestamp(hi))))
    for name, early, late in halves:
        bounds[name + "_H1"], bounds[name + "_H2"] = [early], [late]
    bounds["ALL_EARLY"] = [row[1] for row in halves]
    bounds["ALL_LATE"] = [row[2] for row in halves]
    bounds["pooled"] = [bounds[name][0] for name, _, _ in spec["folds"]]
    return bounds


def selected(trades, bounds):
    chosen = []
    for trade in trades:
        time = pd.Timestamp(trade["entry_time_api"]).value
        if any(a.value <= time < b.value for a, b in bounds):
            chosen.append(trade)
    return chosen


def counts(trades):
    wins = len([t for t in trades if float(t["net_r"]) > 0])
    return len(trades), wins, len(trades) - wins, wins / len(trades) if trades else None


def identity(trade):
    return int(trade["score_fold"]), int(trade["source_entry_index"])


def independent_displacement(a0, s4, bounds):
    a, b = selected(a0, bounds), selected(s4, bounds)
    a_ids, b_ids = {identity(t) for t in a0}, {identity(t) for t in s4}
    require(len(a_ids) == len(a0) and len(b_ids) == len(s4), "Unique entry row/fold identities")
    primary_ids = {identity(t) for t in s4 if t["entry_kind"] == "primary"}
    removed = [t for t in a if identity(t) not in primary_ids]
    occupied, other = [], []
    for t in removed:
        at = int(t["entry_index"])
        blockers = [s for s in s4 if int(s["entry_index"]) < at
                    and (int(s["exit_index"]) > at or
                         (int(s["exit_index"]) == at and s["exit_reason"] != "timeout"))]
        require(len(blockers) <= 1, "Frozen single position occupancy")
        (occupied if blockers else other).append(t)
    added_primary = [t for t in b if t["entry_kind"] == "primary" and identity(t) not in a_ids]
    added_secondary = [t for t in b if t["entry_kind"] == "secondary"]
    require(len(b) == len(a) - len(removed) + len(added_primary) + len(added_secondary), "Displacement conservation")
    def detail(trades):
        n, w, loss, wr = counts(trades)
        return {"baseline_primary_entries_displaced": n, "displaced_primary_wins": w,
                "displaced_primary_losses": loss, "displaced_primary_realized_wr": wr,
                "identities": [list(identity(t)) for t in trades]}
    _, wins, losses, wr = counts(added_secondary)
    return {"baseline_trade_count": len(a), "s4_trade_count": len(b),
            "baseline_primary_entries_retained": len(a) - len(removed), **detail(removed),
            "new_secondary_entries_executed": len(added_secondary), "new_primary_entries_executed": len(added_primary),
            "secondary_wins": wins, "secondary_losses": losses, "secondary_realized_wr": wr,
            "occupancy_displacement": detail(occupied), "nonoccupancy_displacement": detail(other)}


def independent_stress(trades, cost):
    require(cost in {0., .01, .02}, "Frozen diagnostic stress")
    returns = np.asarray([float(t["net_r"]) - cost for t in trades], dtype=np.float64)
    wins = returns > 0
    gain = float(returns[wins].sum())
    loss = -float(returns[~wins].sum())
    equity = np.concatenate((np.zeros(1), returns.cumsum()))
    return {"trades": len(returns), "wins": int(wins.sum()), "losses": int((~wins).sum()),
            "realized_wr": float(wins.mean()) if len(returns) else 0.,
            "pf": gain / loss if loss else (float("inf") if gain else 0.),
            "mean_r": float(returns.mean()) if len(returns) else 0., "pnl_r": float(returns.sum()),
            "max_dd_r": float(np.min(equity - np.maximum.accumulate(equity)))}


def independent_confirmation(rows, deltas, stresses, spec):
    s4 = {r["scope"]: r for r in rows if r["candidate_id"] == "S4_CONFIRMATION"}
    differences = {r["scope"]: r for r in deltas}
    rejected = []
    for scope in ["pooled"] + [f[0] for f in spec["folds"]]:
        for metric, floor in spec["guardrails"]["pooled" if scope == "pooled" else "fold"].items():
            value = s4[scope][metric]
            if value is None or math.isnan(value) or not value >= floor:
                rejected.append(scope + ":" + metric)
    half_count = sum(differences[name + half]["delta_wr"] >= 0
                     and differences[name + half]["delta_trades_per_day"] >= 0
                     for name, _, _ in spec["folds"] for half in ("_H1", "_H2"))
    early, late, pooled = (differences[key] for key in ("ALL_EARLY", "ALL_LATE", "pooled"))
    checks = {"pooled_wr_improves": pooled["delta_wr"] > 0,
              "pooled_frequency_improves": pooled["delta_trades_per_day"] > 0,
              "original_guardrails": len(rejected) == 0, "four_of_six_halves": half_count >= 4,
              "early_not_both_worse": early["delta_wr"] >= 0 or early["delta_trades_per_day"] >= 0,
              "late_not_both_worse": late["delta_wr"] >= 0 or late["delta_trades_per_day"] >= 0}
    for name, tolerance in (("R1_EXTRA_COST", .03), ("R2_EXTRA_COST", .05)):
        get = lambda candidate: next(r["pf"] for r in stresses if r["scenario"] == name and r["scope"] == "pooled" and r["candidate_id"] == candidate)
        checks[name] = get("S4_CONFIRMATION") >= get("A0_BASELINE_075") - tolerance
    success = all(checks.values())
    return {"confirmation_primary_pass": success, "confirmation_supportive": success,
            "confirmation_strong": success and s4["pooled"]["realized_wr"] >= .58 and s4["pooled"]["trades_per_day"] >= .40,
            "checks": checks, "guardrail_failures": rejected, "nonnegative_half_count": half_count}


def independent_robustness(ledgers, helper, spec):
    windows = independent_windows(spec)
    rows, stresses, differences = [], [], []
    for candidate in CANDIDATES:
        for scope, spans in windows.items():
            trades = selected(ledgers[candidate], spans)
            days = sum((end.value - start.value) / 86_400_000_000_000 for start, end in spans)
            diagnostics = {}
            for kind in ("primary", "secondary"):
                n, w, loss, wr = counts([t for t in trades if t["entry_kind"] == kind])
                diagnostics.update({kind + "_trade_count": n, kind + "_wins": w, kind + "_losses": loss,
                                    kind + "_realized_wr": wr, kind + "_trades_per_day": n / days})
            rows.append({"candidate_id": candidate, "scope": scope, "days": days,
                         **helper.returns_summary(trades, days), **diagnostics})
            for scenario, cost in spec["stress_scenarios"].items():
                stresses.append({"candidate_id": candidate, "scope": scope, "scenario": scenario,
                                 "extra_cost_r": cost, **independent_stress(trades, cost)})
    for scope in windows:
        a, b = [next(r for r in rows if r["scope"] == scope and r["candidate_id"] == c) for c in CANDIDATES]
        differences.append({"scope": scope, "delta_wr": b["realized_wr"] - a["realized_wr"],
                            "delta_trades_per_day": b["trades_per_day"] - a["trades_per_day"],
                            "delta_pf": b["pf"] - a["pf"], "delta_mean_r": b["mean_r"] - a["mean_r"],
                            "delta_cost_stress_pf": b["cost_stress_pf"] - a["cost_stress_pf"]})
    return {"metrics": rows, "deltas": differences, "stress_metrics": stresses,
            "displacement": {scope: independent_displacement(ledgers[CANDIDATES[0]], ledgers[CANDIDATES[1]], spans) for scope, spans in windows.items()},
            "slice_boundaries": {name: [[a.isoformat(), b.isoformat()] for a, b in spans] for name, spans in windows.items()},
            "confirmation": independent_confirmation(rows, differences, stresses, spec),
            "selected_candidate": None, "production_promotion_requested": False, "classification": spec["classification"]}


def verify(run, checks):
    def check(name, condition):
        checks[name] = bool(condition)
        require(condition, name)
    spec = specification()
    manifest = read_json(run / "manifest.json")
    check("formal_manifest", manifest["experiment_name"] == EXPERIMENT and manifest["run_id"] == run.name
          and manifest["status"] == "in_progress" and manifest["git_dirty"] is False and not (run / "failure.json").exists())
    for source, snapshot in ((EXPERIMENT + ".py", "training_script.py"),
                             ("validate_" + EXPERIMENT + ".py", "validator_script.py"), (SPEC_FILE, "execution_spec.json")):
        committed = subprocess.check_output(["git", "show", manifest["git_commit"] + ":" + source], cwd=ROOT)
        check("committed_source:" + source, committed.replace(b"\r\n", b"\n")
              == (run / snapshot).read_bytes().replace(b"\r\n", b"\n") == (ROOT / source).read_bytes().replace(b"\r\n", b"\n"))
    check("executed_validator", Path(__file__).read_bytes() == (ROOT / ("validate_" + EXPERIMENT + ".py")).read_bytes())
    check("training_snapshot", sha(run / "training_script.py") == manifest["training_script_sha256"])
    check("execution_spec", read_json(run / "execution_spec.json") == spec)
    required = {"metrics.json", "candidates.csv", "fold_metrics.csv", "robustness_slices.csv", "displacement_metrics.json",
                "stress_metrics.csv", "trade_ledger.csv", "secondary_evidence.npz", "model_inventory.json", "baseline_reproduction.json",
                "source_provenance.json", "identity_audit.json", "training_script.py", "validator_script.py", "execution_spec.json",
                "report.md", "model.sha256", "environment.txt", "operational_safety_pre.json", "operational_safety_post.json"}
    names = [item["path"] for item in manifest["artifacts"]]
    check("artifact_inventory", len(names) == len(set(names)) and required <= set(names))
    for item in manifest["artifacts"]:
        path = (run / item["path"]).resolve()
        check("artifact:" + item["path"], path.is_relative_to(run) and sha(path) == item["sha256"])
    check("no_tuning_or_promotion", manifest["search"]["performed"] is False
          and manifest["search"]["predefined_candidates"] == spec["candidates"]
          and manifest["promotion"] == {"requested": False, "replacement_authorized": False,
                                       "operational_artifact_changed": False, "gate_result": "research_only"})
    before = operational_hashes()
    check("operational_pre_post", before == read_json(run / "operational_safety_pre.json") == read_json(run / "operational_safety_post.json"))
    archive, ds = discovery_identity(spec)
    check("discovery_identity_pass_predeclared_s4", True)
    discovery = load_module(archive / "validator_script.py", "s4_independent_discovery_audit")
    b0_archive, helper, parent = discovery.baseline_inputs(ds)
    history, folds, identities = helper.rebuild(parent)
    check("exact_timestamps_features_targets", identities == read_json(run / "identity_audit.json"))
    target = parent.build_execution_aligned_labels(history).C1_TARGET.to_numpy(dtype=np.int8)
    for number, _, train, _ in folds:
        check(f"target_before_fitting:{number}", array_hash(target[train]) == spec["c1_target_sha256"][f"fold{number}_train"])
    inventory = read_json(run / "model_inventory.json")
    previous_inventory = read_json(archive / "model_inventory.json")
    check("three_models", len(inventory) == 3 and [m["fold"] for m in inventory] == [f[0] for f in spec["folds"]])
    check("model_file_inventory", {p.relative_to(run).as_posix() for p in (run / "models").rglob("*") if p.is_file()}
          == {m["path"] for m in inventory} and {m["path"] for m in inventory} <= set(names))
    check("model_manifest", manifest["model"]["trained"] is True and manifest["model"]["parameters"] == spec["parameters"]
          and manifest["model"]["features"] == spec["features"] and manifest["model"]["feature_count"] == 31
          and manifest["random_seeds"] == {"secondary": 42} and manifest["model"]["fold_models_trained"] == 3
          and manifest["model"]["artifact_path"] == inventory[-1]["path"] and manifest["model"]["artifact_sha256"] == inventory[-1]["sha256"]
          and (run / "model.sha256").read_text().split()[0] == inventory[-1]["sha256"])
    keys, equivalence = set(), {}
    frames, b0_parts, secondary_parts, fold_parts, index_parts = [], [], [], [], []
    ns = history.TIME_DT.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    with np.load(run / "secondary_evidence.npz", allow_pickle=False) as evidence, np.load(
        archive / "secondary_evidence.npz", allow_pickle=False
    ) as previous:
        for number, fold, train, score in folds:
            key = f"fold{number}"
            def array_check(suffix, expected):
                field = key + "_" + suffix
                keys.add(field)
                check("array:" + field, evidence[field].dtype == expected.dtype and np.array_equal(evidence[field], expected))
            b0_item = next(m for m in spec["b0_models"] if m["fold"] == fold)
            check("B0_model:" + key, sha(b0_archive / b0_item["path"]) == b0_item["sha256"])
            b0_model = xgb.XGBClassifier()
            b0_model.load_model(b0_archive / b0_item["path"])
            check("B0_features:" + key, b0_model.get_booster().feature_names == spec["features"])
            x_train = history.loc[train, spec["features"]].astype(np.float32)
            x_score = history.loc[score, spec["features"]].astype(np.float32)
            p_train, p_score = b0_model.predict_proba(x_train)[:, 1], b0_model.predict_proba(x_score)[:, 1]
            subset = discovery.independent_subset(train, score, ns[train], ns[score], p_train)
            for suffix, value in (("train_indices", train), ("score_indices", score), ("subset_indices", subset),
                                  ("b0_train", p_train), ("b0_score", p_score), ("subset_target", target[subset]),
                                  ("train_ns", ns[train]), ("score_ns", ns[score])):
                array_check(suffix, value)
            x = history.loc[subset, spec["features"]].astype(np.float32)
            fitted, weights = discovery.independent_fit(x, target[subset], ds)
            item = inventory[number - 1]
            path = (run / item["path"]).resolve()
            check("secondary_artifact:" + key, path.is_relative_to(run / "models") and sha(path) == item["sha256"])
            check("subset_config:" + key, item["features"] == spec["features"] and item["parameters"] == spec["parameters"]
                  and item["seed"] == 42 and item["rounds"] == 220 and item["train_rows"] == len(subset)
                  and item["train_indices_sha256"] == array_hash(subset) and item["x_sha256"] == array_hash(x.to_numpy())
                  and item["target_sha256"] == array_hash(target[subset]) and item["weights_sha256"] == array_hash(weights)
                  and item["training_config"] == json.loads(fitted.get_booster().save_config()))
            model = xgb.XGBClassifier()
            model.load_model(path)
            actual = json.loads(model.get_booster().save_raw(raw_format="json"))["learner"]
            expected = json.loads(fitted.get_booster().save_raw(raw_format="json"))["learner"]
            for field in ("gradient_booster", "learner_model_param", "objective", "feature_names", "feature_types"):
                check("independent_refit_model:" + key + ":" + field, actual[field] == expected[field])
            p_secondary = fitted.predict_proba(x_score)[:, 1]
            check("saved_model_predictions:" + key, np.array_equal(model.predict_proba(x_score)[:, 1], p_secondary))
            array_check("secondary_score", p_secondary)
            array_check("sample_weight", weights)
            reference = {suffix: np.array_equal(evidence[key + "_" + suffix], previous[key + "_" + suffix]) for suffix in
                         ("train_indices", "score_indices", "subset_indices", "subset_target", "sample_weight", "b0_train", "b0_score", "secondary_score")}
            reference["training_config"] = item["training_config"] == previous_inventory[number - 1]["training_config"]
            check("deterministic_discovery_reference:" + key, all(reference.values()))
            equivalence[key] = reference
            part = history.loc[score, helper.PRICE_COLUMNS].copy()
            part["buy_prob"], part["sell_prob"] = p_score, np.float32(0)
            frames.append(part)
            b0_parts.append(p_score)
            secondary_parts.append(p_secondary)
            fold_parts.append(np.full(len(score), number, dtype=np.int8))
            index_parts.append(score)
        frame = pd.concat(frames, ignore_index=True)
        b0, secondary, fold_ids, indices = map(np.concatenate, (b0_parts, secondary_parts, fold_parts, index_parts))
        ledgers = {}
        for candidate in CANDIDATES:
            primary, extra, union = independent_gate(b0, secondary, candidate)
            for suffix, values in (("primary", primary), ("secondary", extra), ("union", union)):
                key = candidate + "_" + suffix
                keys.add(key)
                check("gate:" + key, evidence[key].dtype == np.bool_ and np.array_equal(evidence[key], values))
            trades = discovery.replay(parent, frame, candidate, primary, extra, fold_ids)
            for trade in trades:
                trade["source_entry_index"] = int(indices[int(trade["entry_index"])])
            ledgers[candidate] = trades
        check("evidence_keys", set(evidence.files) == keys)
    output = independent_robustness(ledgers, helper, spec)
    output["deterministic_discovery_equivalence"] = equivalence
    baseline = next(r for r in output["metrics"] if r["scope"] == "pooled" and r["candidate_id"] == CANDIDATES[0])
    for key, expected in spec["baseline"].items():
        check("baseline:" + key, discovery.equal(baseline[key], expected))
    baseline_rows = [{"candidate_id": CANDIDATES[0], **r} for r in helper.aggregate(ledgers[CANDIDATES[0]])]
    check("baseline_evidence", discovery.equal(read_json(run / "baseline_reproduction.json"), {"pass": True, "metrics": baseline_rows}))
    check("complete_metrics_slices_displacement_stresses_gates", discovery.equal(read_json(run / "metrics.json"), output))
    check("displacement_artifact", discovery.equal(read_json(run / "displacement_metrics.json"), output["displacement"]))
    delta = {r["scope"]: r for r in output["deltas"]}
    rows = [{**r, **{k: v for k, v in delta[r["scope"]].items() if k != "scope"}} for r in output["metrics"]]
    originals = [f[0] for f in spec["folds"]] + ["pooled"]
    discovery.verify_csv(run / "fold_metrics.csv", [r for r in rows if r["scope"] in originals])
    discovery.verify_csv(run / "robustness_slices.csv", [r for r in rows if r["scope"] not in originals])
    discovery.verify_csv(run / "stress_metrics.csv", output["stress_metrics"])
    discovery.verify_csv(run / "trade_ledger.csv", [{**t, "candidate_id": c} for c in CANDIDATES for t in ledgers[c]])
    discovery.verify_csv(run / "candidates.csv", [{**r, "pnl": r["pnl_r"], "max_dd": r["max_dd_r"],
        "qualification_verdict": "baseline" if r["candidate_id"] == CANDIDATES[0] else
        ("confirmation_supportive" if output["confirmation"]["confirmation_primary_pass"] else "confirmation_not_supportive")}
        for r in rows if r["scope"] == "pooled"])
    check("all_csv_tables_and_exact_S5_ledger", True)
    check("report", discovery.equal(json.loads((run / "report.md").read_text(encoding="utf-8").split("```json\n")[1].split("\n```")[0]), output))
    check("source_provenance", read_json(run / "source_provenance.json") == {
        "spec_sha256": SPEC_SHA256, "discovery_run": spec["discovery_run"], "discovery_finalized_sha256": spec["discovery_finalized_sha256"],
        "reference_run": ds["reference_run"], "reference_finalized_sha256": ds["reference_finalized_sha256"],
        "b0_models": spec["b0_models"], "source_hashes": ds["source_hashes"], "discovery_predictions_used_for_execution": False})
    check("final_operational_files", operational_hashes() == before)
    return output


def validate(run):
    run = run.resolve()
    require(run.parent == ROOT / "training_runs" and (run / "manifest.json").is_file()
            and not (run / "FINALIZED.json").exists(), "Unfinalized run required")
    require(not any((run / name).exists() for name in ("validator_attempt.json", "validator.json", "validator.md")), "Do not retry validation")
    with (run / "validator_attempt.json").open("x", encoding="utf-8") as stream:
        json.dump({"validator_sha256": sha(Path(__file__)), "one_shot": True}, stream)
    checks, output = {}, {}
    before = operational_hashes()
    try:
        output = verify(run, checks)
    except Exception as error:
        checks[f"{type(error).__name__}: {error}"] = False
    finally:
        checks["validator_operational_safety"] = operational_hashes() == before
    failed = [name for name, passed in checks.items() if not passed]
    verdict = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed,
               "checks": checks, "independent_metrics": output, "scope": "historical development methodology/provenance; no untouched or promotion claim"}
    write_json(run / "validator.json", verdict)
    (run / "validator.md").write_text("# Independent S4 confirmation validator\n\nOverall: " + verdict["overall"]
        + "\n\nFailed checks: " + repr(failed) + "\n\nNo retries. No production promotion.\n", encoding="utf-8")
    manifest = read_json(run / "manifest.json")
    manifest["registry"]["validator_result"] = "independent " + verdict["overall"]
    for name in ("validator_attempt.json", "validator.json", "validator.md"):
        manifest["artifacts"].append({"path": name, "sha256": sha(run / name), "kind": "independent_validator", "retention_status": "stored_in_run_directory"})
    write_json(run / "manifest.json", manifest)
    print("INDEPENDENT_VALIDATOR_" + verdict["overall"])
    print("failed_check_names=" + repr(failed))
    return not failed


def self_test():
    import copy
    import tempfile
    spec = specification()
    gate = independent_gate
    windows_fn = independent_windows
    select_fn = selected
    displacement_fn = independent_displacement
    stress_fn = independent_stress
    gates_fn = independent_confirmation
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
    source=archive/'validator_script.py'
    require(sha(source)==final['file_sha256']['validator_script.py'],'Test source snapshot')
    library=load_module(source,'s4_synthetic_frozen_helpers')
    ds=read_json(archive/'execution_spec.json')
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(1200,31)).astype(np.float32),columns=spec['features'])
    y=(x.iloc[:,0].to_numpy()>0).astype(np.int8)
    fit=library.independent_fit
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
    return 0 if validate(args.run_dir) else 1


if __name__ == '__main__':
    raise SystemExit(main())
