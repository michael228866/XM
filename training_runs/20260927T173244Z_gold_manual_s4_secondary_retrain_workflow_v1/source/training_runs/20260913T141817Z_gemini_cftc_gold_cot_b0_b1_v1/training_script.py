"""Frozen research-only CFTC comparison. Formal execution requires --run-dir."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "gemini_cftc_gold_cot_b0_b1_v1"
CANDIDATES = ("B0_31_technical", "B1_31_plus_5_cftc_cot")
PARENT = "20260903T071729Z_gemini_execution_aligned_label_v1"
REFERENCE = "20260909T140906Z_gemini_us_treasury_real_rate_b0_b1_v1"
FOUNDATION = "20260912T171048Z_gemini_cftc_gold_cot_foundation_v1"
TIMESTAMPS = "20260905T171629Z_gemini_macro_event_integration_foundation_v1"
COT_FEATURES = ["COT_MM_NET_PCT_OI", "COT_MM_NET_CHG_1W", "COT_PROD_NET_PCT_OI",
                "COT_SWAP_NET_PCT_OI", "COT_MM_VS_PROD_SPREAD"]
LOGICAL_HASH = "be6245e7b79695b338ec0ec1c493f045123254a0fccd2952670ae553fa633a09"
PHYSICAL_HASH = "24ebca3f1a1dec99cb2b6979f922425237623ff9f1809b64e53b9c00e69467a1"
DATASET_HASH = "2713dc12ff22c8661ea3ce1c92f34c74019b137592e45edc75d78e9ec794b9ab"
BASE_FEATURES = ['M1_RSI',
 'ATR',
 'MACD_HIST',
 'BB_WIDTH',
 'BIAS_20',
 'BODY_PCT',
 'ROC_5',
 'VOLA_RATIO',
 'HOUR_SIN',
 'HOUR_COS',
 'DAY_OF_WEEK',
 'Daily_TREND',
 'H12_TREND',
 'H1_TREND',
 'H2_TREND',
 'H3_TREND',
 'H4_TREND',
 'H6_TREND',
 'H8_TREND',
 'M10_TREND',
 'M12_TREND',
 'M15_TREND',
 'M20_TREND',
 'M2_TREND',
 'M30_TREND',
 'M3_TREND',
 'M4_TREND',
 'M5_TREND',
 'M6_TREND',
 'Monthly_TREND',
 'Weekly_TREND']
BLOCKS = {'fold1_train': (530218, '6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6'),
 'fold1_score': (1058080, '47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94'),
 'fold2_train': (532563, '691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44'),
 'fold2_score': (708197, '1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8'),
 'fold3_train': (533580, '3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b'),
 'fold3_score': (708020, '1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449')}
EXPECTED_X = {'fold1_train': '14999ac58c67bff25d7977d63a80f6069dde0256752a20a9dee936c022db69cb',
 'fold1_score': '3c6eb5d77182428306f059e9a7d7607d1a4b5de2e5cc5217d9099e74c85e0316',
 'fold2_train': 'c921f11aaa44799dc798a1b54d6a745bb636b678be1212d17ba33e046acc0ac4',
 'fold2_score': 'fd77cdad26894c813e9fba0deb77ed1a5fdcaaffceaad45488da9a2015f11c36',
 'fold3_train': '64124692d42cdb077fdfff1d5cf9df26582c025ed738608168b02cfa1cb42470',
 'fold3_score': '2b55883b3410e9613983b8cebfbe1e2ea7a47daa33f44a7df535c0a5b083e717'}
EXPECTED_Y = {'fold1_train': 'd9bed87a073fd93c048b5a1ab8f0f7049f1a1bb8f3ff46f48a102c0b6763bbd3',
 'fold2_train': 'ff55356e8bdf1cb2cd5add50690156178ff32bc17f666f2103c8f51397fb3a12',
 'fold3_train': 'cd05d3647cd88a2fbd2b89fa3362ed5e2f9f13cbd611dee436f0e03e4a054259'}
EXPECTED_B0 = {'trades': 689,
 'realized_wr': 0.5660377358490566,
 'pf': 0.8247098331219882,
 'mean_r': -0.07690539315994348,
 'pnl_r': -52.98781588720106,
 'cost_stress_pf': 0.7838186120438296,
 'wins': 390,
 'losses': 299,
 'max_dd_r': -59.57958015890608,
 'trades_per_day': 0.26945639421196715}
SOURCE_HASHES = {'training_runs/20260903T071729Z_gemini_execution_aligned_label_v1/training_script.py': '9e436e4fe723eb8ad354f5f27a342b60d051c5c71d533f523871b381fc7012c6',
 'barrier_research_suite.py': 'e9b0f84e0aedff532b0d24c5845191d724a7dc44ade15d928fe0a24b7c1341af',
 'drl_trading_v2.py': 'b129d90d4a4a02cbf6d77903351ed7c3273af3cb3d4464b6a146aa25153ef767',
 'drl_train_candidate.py': '8452fe54921dad32d9af0f77b203b78d67bed765eb4e832f39f2c8ed171fd762',
 'barrier_final_train.py': 'a221bda6c61fa89bde0592453de7a8ac84efb80f06b74e7686b872484e2edee5',
 'barrier_classifier_strategy.py': 'a743d15b90321e4d933e4657abd185922bb280b36ec48612b6f1688afb909174',
 'strategy_grid_search.py': 'ad0c49c9a605718b90df49d0afb565b66af7bd3f3e9e68153dbd386be6663403',
 'gold_gemini_execution_semantics_v1.py': 'fc540b959a9fdec6fafed9acce58d70b0e57261b53d6f57c81b2cb7c3e976757',
 'gold_recent_walk_forward.py': 'cf14bbd97e88632ae53cadaabf5e3341f844f5096e8dd0493f8f96a4bf810334',
 'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07',
 'gold_gemini_incumbent_robustness_v1.py': 'adb83d97bc4bf7a44140da30f14b469425de3266b93544c6dd43e92f4ed0ae5f',
 'gold_generation11_execution_aligned.py': 'adabda7a5ed4fdd4a7c12a5f95e83835df191da102639295510cf93ebf0344fe',
 'gold_short_rule_research.py': 'fdf9262784a60a3a284b06a4fdc00b90ffe0a01d926fc552c562d7936015f7bb',
 'gold_regime_experts_walk_forward.py': '905c5a2397e8cabc2b1d538c91ed60d97f73cb4b55578af95b996d065fc51694',
 'gold_long_model_optimization.py': '3c91e18776f38822dd0b5ebc8e031718019aeb318888e2dcee09c83e0d89ef75',
 'gold_long_recent_walk_forward.py': '575b56a8a6d526a0102904dc873bac3a70d7569ac09c656720bfd76def768280',
 'gold_regime_experts_iterative.py': '914a1404a99a7a52def92d0fe3bca35ba9da6a1864d156368aec2b9a5b124a21',
 'gold_rolling_champion.py': '1bfe529c44ca59019cc864a998ec128931c121d577c43bc3befa7e743ba560da',
 'gold_generation8_residual_walk_forward.py': '470b0085b820d6ef53f80cd149fdc3bc5afcf2a3996207c68af5363df3a42f83',
 'gold_expected_r_walk_forward.py': '7cd87b364644ab88441055662e18bc1dbe19ef4fb5ee9ebfdef121aa2fe37e2e',
 'gold_expected_r_champion.py': 'ae91fed8968b0d29b963d51ccf9b5f04eff63e4847f4bfbd50989b4b10c74ff9',
 'gold_gemini_core_gate_v1.py': 'ae31c09af94466bf5deeafae4800b2e3a580dfeb31564b1123be69887f57932c'}
OPERATIONAL = ('gemini.py', 'gold_long_recent_candidate_xgb.json')
PRICE_COLUMNS = ["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "ATR", "M1_RSI", "SPREAD"]
PARAMETERS = {"objective": "binary:logistic", "n_estimators": 220,
              "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 80,
              "subsample": 0.85, "colsample_bytree": 0.85, "random_state": 42,
              "tree_method": "hist"}
FOLD_WINDOWS = [("2018_2020", "2018-01-01", "2021-01-01"),
                ("2021_2022", "2021-01-01", "2023-01-01"),
                ("2023_2024", "2023-01-01", "2025-01-01")]


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


def verify_archive(run, required=()):
    final = read_json(run / "FINALIZED.json")
    require(final.get("run_id") == run.name and bool(final.get("finalized_at_utc")),
            f"Invalid finalized identity: {run.name}")
    hashes = final.get("file_sha256")
    require(isinstance(hashes, dict) and bool(hashes), "Missing finalized hashes")
    require(set(required).issubset(hashes), "Required artifact absent from finalized hashes")
    for relative, expected in hashes.items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run.resolve()) and path.is_file(), "Invalid archive path")
        require(sha(path) == expected, f"Finalized archive mismatch: {relative}")


def logical_hash(matrices):
    digest = hashlib.sha256(b"CFTC_GOLD_COT_FOUNDATION_V1\0")
    digest.update(json.dumps(COT_FEATURES, separators=(",", ":")).encode("ascii"))
    for block in BLOCKS:
        matrix = np.ascontiguousarray(matrices[block], dtype=np.float64)
        digest.update(b"\0" + block.encode("ascii") + b"\0")
        digest.update(str(matrix.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.tobytes())
    return digest.hexdigest()


def check_identity(values, expected, label):
    require(array_hash(values) == expected, f"Frozen identity mismatch: {label}")


def check_candidates(ids):
    require(list(ids) == list(CANDIDATES), "Extra, missing or reordered candidates")


def check_b0(metrics):
    for key, expected in EXPECTED_B0.items():
        require(math.isclose(float(metrics[key]), expected, rel_tol=0, abs_tol=1e-12),
                f"B0 reproduction failed: {key}")


def operational_hashes():
    return {name: sha(ROOT / name) for name in OPERATIONAL}


def check_safety(before, after):
    require(bool(before) and before == after, "Operational artifact mutation")


def feature_list(candidate):
    require(candidate in CANDIDATES, "Unknown candidate")
    return BASE_FEATURES + (COT_FEATURES if candidate == CANDIDATES[1] else [])


def load_sources():
    for name, expected in SOURCE_HASHES.items():
        require(sha(ROOT / name) == expected, f"Frozen implementation drift: {name}")
    requirements = {PARENT: ("training_script.py", "manifest.json"),
                    REFERENCE: ("training_script.py", "model_inventory.json", "fold_metrics.csv"),
                    TIMESTAMPS: ("exact_timestamps.npz",),
                    FOUNDATION: ("manifest.json", "metrics.json", "validator.json",
                                 "cftc_gold_cot_features.npz", "cot_observations.csv")}
    for name, files in requirements.items():
        verify_archive(ROOT / "training_runs" / name, files)
    source = ROOT / "training_runs" / FOUNDATION
    require(read_json(source / "manifest.json")["status"] == "pass", "Uncertified foundation")
    require(read_json(source / "validator.json")["overall"] == "PASS", "Foundation validator FAIL")
    require(sha(source / "cftc_gold_cot_features.npz") == PHYSICAL_HASH, "CFTC physical hash mismatch")
    require(sha(source / "cot_observations.csv") == DATASET_HASH, "CFTC dataset hash mismatch")
    with np.load(source / "cftc_gold_cot_features.npz", allow_pickle=False) as archive:
        require(list(archive["feature_names"]) == COT_FEATURES, "CFTC feature order mismatch")
        matrices = {block: archive[block] for block in BLOCKS}
    for block, matrix in matrices.items():
        require(matrix.dtype == np.float64 and matrix.shape == (BLOCKS[block][0], 5)
                and np.isfinite(matrix).all(), f"Invalid CFTC block: {block}")
    require(logical_hash(matrices) == LOGICAL_HASH, "CFTC logical hash mismatch")
    spec = importlib.util.spec_from_file_location(
        "frozen_cftc_parent", ROOT / "training_runs" / PARENT / "training_script.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.drl_trading_v2.DATA_DIR = str(ROOT)
    require(old.xgb.__version__ == "3.2.0", "XGBoost must be 3.2.0")
    require(old.FIXED_XGB_PARAMETERS == PARAMETERS and old.TRAIN_MONTHS == 18
            and old.THRESHOLD == 0.75, "Parent training semantics drift")
    require([(name, str(start.date()), str(end.date())) for name, start, end in old.FOLDS]
            == FOLD_WINDOWS, "Fold windows drift")
    require(old.semantics.SIMULATORS[-1].simulator == "S5", "S5 required")
    return old, matrices


def reconstruct(old):
    history, features = old.prepare_barrier_data()
    history = history.copy().reset_index(drop=True)
    require(features == BASE_FEATURES, "Frozen 31-feature order mismatch")
    labels = old.build_execution_aligned_labels(history)
    times = history["TIME_DT"]
    ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    target = labels["C1_TARGET"].to_numpy(dtype=np.int8)
    mature = labels["C1_MATURE"].to_numpy(dtype=bool)
    maturity = labels["C1_MATURITY_NS"].to_numpy(dtype=np.int64)
    legacy_index = np.arange(len(history), dtype=np.int64) + old.LEGACY_HORIZON_ROWS
    legacy_valid = legacy_index < len(history)
    legacy_ns = np.full(len(history), np.iinfo(np.int64).max, dtype=np.int64)
    legacy_ns[legacy_valid] = ns[legacy_index[legacy_valid]]
    folds, identities = [], []
    with np.load(ROOT / "training_runs" / TIMESTAMPS / "exact_timestamps.npz",
                 allow_pickle=False) as timestamps:
        for number, (name, start, end) in enumerate(old.FOLDS, 1):
            cutoff = start.value
            train = np.flatnonzero(((times >= start - pd.DateOffset(months=18))
                                   & (times < start) & legacy_valid & mature
                                   & (legacy_ns < cutoff) & (maturity < cutoff)).to_numpy())
            score = np.flatnonzero(((times >= start) & (times < end) & mature).to_numpy())
            for suffix, indices in (("train", train), ("score", score)):
                block = f"fold{number}_{suffix}"
                require(len(indices) == BLOCKS[block][0], f"Rows mismatch: {block}")
                check_identity(ns[indices], BLOCKS[block][1], block + " timestamps")
                require(np.array_equal(ns[indices], timestamps[block + "_broker_ns"]),
                        "Timestamp archive alignment mismatch")
                x = history.loc[indices, features].to_numpy(dtype=np.float32)
                check_identity(x, EXPECTED_X[block], block + " GOLD")
                identities.append({"block": block, "rows": len(indices),
                                   "timestamp_sha256": array_hash(ns[indices]),
                                   "gold_sha256": array_hash(x)})
            check_identity(target[train], EXPECTED_Y[f"fold{number}_train"], name + " C1")
            identities[-2]["c1_sha256"] = array_hash(target[train])
            folds.append((number, name, train, score))
    return history, target, folds, identities


def metric_rows(old, trades, candidate):
    rows = []
    total_days = 0
    for name, start, end in old.FOLDS:
        days = old.core.fold_days(start, end)
        total_days += days
        selected = [item for item in trades if start <= pd.Timestamp(item["entry_time_api"]) < end]
        rows.append({"candidate_id": candidate, "fold": name, **old.trade_metrics(selected, days)})
    rows.append({"candidate_id": candidate, "fold": "pooled", **old.trade_metrics(trades, total_days)})
    return rows


def quality(rows):
    pooled = rows[-1]
    reasons = []
    for key, limit, inclusive in (("realized_wr", .60, True), ("pf", 1.05, False),
                                   ("mean_r", 0, False), ("pnl_r", 0, False),
                                   ("break_even_adjusted_edge", 0, False),
                                   ("cost_stress_pf", 1, False)):
        value = pooled[key]
        if value is None or (value < limit if inclusive else value <= limit):
            reasons.append(key)
    for row in rows[:-1]:
        if (row["trades"] < 10 or row["realized_wr"] < .5 or row["pf"] < .8
                or row["mean_r"] < -.1 or row["max_dd_r"] < -20):
            reasons.append("catastrophic_fold:" + row["fold"])
    return reasons


def formal_run(run):
    run = run.resolve()
    require(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(),
            "Use a new, unfinalized training_run_history run")
    manifest = read_json(run / "manifest.json")
    require(manifest.get("experiment_name") == EXPERIMENT
            and manifest.get("status") == "in_progress"
            and manifest.get("run_id") == run.name, "Wrong formal manifest")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "@{u}") == manifest.get("git_commit"), "Git provenance mismatch")
    require(sha(run / "training_script.py") == sha(Path(__file__))
            == manifest.get("training_script_sha256"), "Executed snapshot mismatch")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    prefix = run.relative_to(ROOT).as_posix() + "/"
    require(all(line[3:].strip('"').startswith(prefix) for line in dirty.splitlines()),
            "Unexpected dirty paths")
    require(not (run / "models").exists(), "Refusing to overwrite models")
    before = operational_hashes()
    write_json(run / "operational_safety_pre.json", before)
    try:
        (run / "validator_script.py").write_bytes(
            (ROOT / "validate_cftc_gold_cot_b0_b1_v1.py").read_bytes())
        old, cot = load_sources()
        history, target, folds, identities = reconstruct(old)
        write_json(run / "identity_audit.json", identities)
        provenance = {"foundation_run": FOUNDATION, "logical_hash": LOGICAL_HASH,
                      "physical_hash": PHYSICAL_HASH, "dataset_hash": DATASET_HASH,
                      "source_hashes": SOURCE_HASHES,
                      "finalized_hashes": {name: sha(ROOT / "training_runs" / name / "FINALIZED.json")
                                           for name in (PARENT, REFERENCE, TIMESTAMPS, FOUNDATION)}}
        write_json(run / "source_provenance.json", provenance)
        spec = {"candidates": list(CANDIDATES), "feature_lists": {mid: feature_list(mid) for mid in CANDIDATES},
                "parameters": PARAMETERS, "xgboost_version": "3.2.0", "threshold": .75,
                "training_months": 18, "folds": FOLD_WINDOWS, "simulator": "S5",
                "trailing": False, "search_performed": False, "production_replacement": False,
                "source_hashes": SOURCE_HASHES}
        write_json(run / "execution_spec.json", spec)
        models = run / "models"
        models.mkdir()
        inventory, results, ledgers, saved = [], {}, [], {}
        for candidate in CANDIDATES:
            features = feature_list(candidate)
            parts = []
            for number, name, train_indices, score_indices in folds:
                train = history.loc[train_indices, BASE_FEATURES].copy()
                score = history.loc[score_indices, BASE_FEATURES].copy()
                if candidate == CANDIDATES[1]:
                    for column, feature in enumerate(COT_FEATURES):
                        train[feature] = cot[f"fold{number}_train"][:, column]
                        score[feature] = cot[f"fold{number}_score"][:, column]
                train["BARRIER_TARGET"] = target[train_indices]
                model = old.train_binary_model(train, features, 1, 220)
                path = models / f"{candidate}_{name}_xgb.json"
                model.save_model(path)
                prediction = old.predict_positive(model, score, features).astype(np.float32)
                require(np.isfinite(prediction).all(), "Non-finite prediction")
                inventory.append({"candidate_id": candidate, "fold": name,
                                  "path": path.relative_to(run).as_posix(), "sha256": sha(path),
                                  "parameters": {key: model.get_params()[key] for key in PARAMETERS},
                                  "rounds": model.get_booster().num_boosted_rounds(),
                                  "training_config": json.loads(model.get_booster().save_config()),
                                  "features": features})
                part = history.loc[score_indices, PRICE_COLUMNS].copy()
                part["buy_prob"] = prediction
                part["sell_prob"] = np.float32(0)
                parts.append(part)
                saved[f"{candidate}_fold{number}"] = prediction
                saved[f"fold{number}_indices"] = score_indices
            cohort = old.semantics.finalize_cohort(pd.concat(parts, ignore_index=True), candidate, offset_hours=0)
            trades, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
            results[candidate] = metric_rows(old, trades, candidate)
            if candidate == CANDIDATES[0]:
                check_b0(results[candidate][-1])
            ledgers.extend({**trade, "candidate_id": candidate} for trade in trades)
        check_candidates(results)
        b0, b1 = (results[mid][-1] for mid in CANDIDATES)
        delta = {key: b1[key] - b0[key] for key in
                 ("trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf")}
        reasons = quality(results[CANDIDATES[1]])
        metrics = {"experiment": EXPERIMENT, "classification": "research_only_development_evidence",
                   "results": results, "delta_b1_minus_b0": delta, "b0_exact_reproduction": True,
                   "cftc_family_status": "FAIL" if reasons else "PASS",
                   "quality_failure_reasons": reasons, "search_performed": False,
                   "preferred_quality": b1["pf"] >= 1.15 and b1["cost_stress_pf"] >= 1.05,
                   "frequency_uplift_pct": 100 * (b1["trades_per_day"] / b0["trades_per_day"] - 1),
                   "production_promotion_requested": False}
        write_json(run / "metrics.json", metrics)
        write_json(run / "model_inventory.json", inventory)
        final_model = inventory[-1]
        (run / "model.sha256").write_text(final_model["sha256"] + "  " + final_model["path"] + "\n", encoding="utf-8")
        environment = ["python==" + sys.version.split()[0]]
        for package in ("numpy", "pandas", "scikit-learn", "xgboost", "MetaTrader5", "torch"):
            environment.append(package + "==" + importlib.metadata.version(package))
        (run / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
        write_csv(run / "fold_metrics.csv", results[CANDIDATES[0]] + results[CANDIDATES[1]])
        write_csv(run / "trade_ledger.csv", ledgers)
        write_csv(run / "candidates.csv", [{**row, "pnl": row["pnl_r"], "max_dd": row["max_dd_r"],
                                           "qualification_verdict": "baseline_reproduced" if mid == CANDIDATES[0]
                                           else metrics["cftc_family_status"]}
                                          for mid, row in zip(CANDIDATES, (b0, b1))])
        np.savez_compressed(run / "paired_oof_predictions.npz", **saved)
        report = ["# Frozen CFTC GOLD COT B0/B1", "", "Research only; historical development evidence.",
                  "No tuning or production promotion. Frequency is considered only after quality passes.",
                  "", "Family status: " + metrics["cftc_family_status"],
                  "Quality failures: " + repr(reasons), "", "## Fold and pooled results", "",
                  "```json", json.dumps({"results": results, "deltas": delta}, default=str, indent=2), "```"]
        (run / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        manifest.setdefault("model", {}).update({"trained": True, "model_type": "XGBoost",
            "parameters": PARAMETERS, "features": BASE_FEATURES, "feature_count": 31,
            "candidate_features": spec["feature_lists"], "boosted_rounds_or_estimators": 220,
            "label_definition": "Frozen C1 standalone S5 net realized R > 0",
            "horizon": "90 wall-clock minutes", "label_tp_sl_semantics": "frozen C1",
            "execution_tp_sl_semantics": "frozen S5", "calibration_method": "none",
            "artifact_path": final_model["path"], "artifact_sha256": final_model["sha256"],
            "retention_status": "all six fold models retained; artifact_path references last B1 fold"})
        reference_data = read_json(ROOT / "training_runs" / REFERENCE / "manifest.json")["data"]
        manifest["data"] = dict(reference_data)
        manifest.setdefault("data", {}).update({"symbols": ["GOLD#", "CFTC GOLD 088691"],
            "data_sources": [PARENT, REFERENCE, FOUNDATION, TIMESTAMPS],
            "raw_snapshot_retained": False,
            "reproducibility_claim": "Frozen reconstructed X/y/timestamps and finalized CFTC source; local GOLD CSVs not duplicated",
            "train_rows": sum(len(row[2]) for row in folds),
            "validation_rows": sum(len(row[3]) for row in folds),
            "timezone": "frozen C1 broker/API semantics; CFTC certified causal availability",
            "test_start_utc": "not_applicable_historical_development_comparison",
            "test_end_utc": "not_applicable_historical_development_comparison",
            "test_rows": 0,
            "source_files": [{"path": "training_runs/" + name + "/FINALIZED.json", "sha256": value,
                              "retention_status": "retained_in_prior_finalized_run"}
                             for name, value in provenance["finalized_hashes"].items()]})
        manifest["random_seeds"] = {mid: 42 for mid in CANDIDATES}
        manifest["search"] = {"performed": False, "predefined_search_space": {},
                              "candidate_results_file": "candidates.csv", "not_applicable_reason": "Frozen pair only"}
        manifest["promotion"] = {"requested": False, "replacement_authorized": False,
                                 "operational_artifact_changed": False, "gate_result": "research_only"}
        manifest.setdefault("registry", {}).update({"parent_or_incumbent": PARENT,
            "selected_configuration": CANDIDATES[1] + "_research_only",
            "trades_per_day": b1["trades_per_day"], "realized_win_rate": b1["realized_wr"],
            "pf": b1["pf"], "mean_r": b1["mean_r"], "pnl": b1["pnl_r"],
            "max_dd": b1["max_dd_r"], "validator_result": "PENDING"})
        manifest["operational_hashes_before"] = before
        manifest["operational_hashes_after"] = operational_hashes()
        check_safety(before, manifest["operational_hashes_after"])
        write_json(run / "operational_safety_post.json", manifest["operational_hashes_after"])
        manifest["artifacts"] = [{"kind": "comparison_evidence", "path": path.relative_to(run).as_posix(),
                                  "sha256": sha(path), "retention_status": "stored_in_run_directory"}
                                 for path in sorted(run.rglob("*")) if path.is_file()
                                 and path.name not in {"manifest.json", "stdout.log"}]
        write_json(run / "manifest.json", manifest)
    except Exception as error:
        failure = {"experiment": EXPERIMENT, "comparison_status": "FAIL",
                   "error_type": type(error).__name__, "error": str(error),
                   "production_promotion_requested": False, "search_performed": False}
        write_json(run / "comparison_failure.json", failure)
        write_json(run / "metrics.json", failure)
        (run / "report.md").write_text(
            "# Frozen CFTC GOLD COT B0/B1\n\nStatus: **FAIL**\n\n"
            + str(error) + "\n\nRetain this failed attempt; do not reuse the run directory.\n",
            encoding="utf-8")
        print("CFTC_B0_B1_FAIL")
        raise
    finally:
        after = operational_hashes()
        write_json(run / "operational_safety_post.json", after)
        check_safety(before, after)
    print("CFTC_DIRECT_FEATURE_FAMILY_" + metrics["cftc_family_status"])


def self_test():
    # Synthetic only: never import the parent, reconstruct GOLD, or fit a model.
    assert len(feature_list(CANDIDATES[0])) == 31
    assert len(feature_list(CANDIDATES[1])) == 36
    assert feature_list(CANDIDATES[1]) == BASE_FEATURES + COT_FEATURES
    fixture = {block: np.ones((2, 5), dtype=np.float64) for block in BLOCKS}
    expected = logical_hash(fixture)
    fixture["fold1_train"][0, 0] += 1
    def fails(action):
        try:
            action()
        except (ValueError, FileNotFoundError):
            return
        raise AssertionError("Mutation was accepted")
    fails(lambda: require(logical_hash(fixture) == expected, "CFTC logical hash mismatch"))
    fails(lambda: check_identity(np.zeros((2, 31), np.float32), EXPECTED_X["fold1_train"], "GOLD"))
    fails(lambda: check_identity(np.zeros(2, np.int8), EXPECTED_Y["fold1_train"], "C1"))
    fails(lambda: check_candidates([*CANDIDATES, "extra"]))
    check_b0(EXPECTED_B0)
    fails(lambda: check_b0({**EXPECTED_B0, "wins": 391}))
    fails(lambda: check_safety({"model": "a"}, {"model": "b"}))
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        (run / "sample").write_bytes(b"original")
        write_json(run / "FINALIZED.json", {"run_id": run.name, "finalized_at_utc": "synthetic",
                                           "file_sha256": {"sample": sha(run / "sample")}})
        verify_archive(run, ("sample",))
        (run / "sample").write_bytes(b"changed")
        fails(lambda: verify_archive(run))
    # Exercise pooled collection, including a fold boundary, without real dependencies.
    from types import SimpleNamespace
    old = SimpleNamespace(FOLDS=[("a", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
                                 ("b", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))],
                          core=SimpleNamespace(fold_days=lambda s, e: (e - s).days),
                          trade_metrics=lambda t, d: {"trades": len(t), "pnl_r": sum(x["net_r"] for x in t),
                                                       "trades_per_day": len(t) / d})
    trades = [{"entry_time_api": "2020-01-01", "net_r": 1},
              {"entry_time_api": "2020-01-02", "net_r": -.5}]
    rows = metric_rows(old, trades, CANDIDATES[0])
    assert rows[-1]["trades"] == 2 and rows[-1]["pnl_r"] == .5 and rows[-1]["trades_per_day"] == 1
    tree = ast.parse((ROOT / "validate_cftc_gold_cot_b0_b1_v1.py").read_text(encoding="utf-8"))
    peer = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CANDIDATES" for t in node.targets))
    assert tuple(peer) == CANDIDATES
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "cftc_gold_cot_b0_b1_v1"
                   for node in ast.walk(tree))
    print("SELF_TEST_PASS: feature_order logical_hash archive GOLD C1 candidates B0 safety pooled independence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")
    else:
        formal_run(args.run_dir)


if __name__ == "__main__":
    main()
