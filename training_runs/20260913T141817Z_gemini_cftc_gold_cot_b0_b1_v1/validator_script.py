"""Independent evidence validation; no runner import, model fitting or tuning."""
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
CANDIDATES = ("B0_31_technical", "B1_31_plus_5_cftc_cot")
EXPERIMENT = 'gemini_cftc_gold_cot_b0_b1_v1'
PARENT = '20260903T071729Z_gemini_execution_aligned_label_v1'
REFERENCE = '20260909T140906Z_gemini_us_treasury_real_rate_b0_b1_v1'
FOUNDATION = '20260912T171048Z_gemini_cftc_gold_cot_foundation_v1'
TIMESTAMPS = '20260905T171629Z_gemini_macro_event_integration_foundation_v1'
COT_FEATURES = ['COT_MM_NET_PCT_OI',
 'COT_MM_NET_CHG_1W',
 'COT_PROD_NET_PCT_OI',
 'COT_SWAP_NET_PCT_OI',
 'COT_MM_VS_PROD_SPREAD']
LOGICAL_HASH = 'be6245e7b79695b338ec0ec1c493f045123254a0fccd2952670ae553fa633a09'
PHYSICAL_HASH = '24ebca3f1a1dec99cb2b6979f922425237623ff9f1809b64e53b9c00e69467a1'
DATASET_HASH = '2713dc12ff22c8661ea3ce1c92f34c74019b137592e45edc75d78e9ec794b9ab'
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
PRICE_COLUMNS = ['TIME_DT', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'ATR', 'M1_RSI', 'SPREAD']
PARAMETERS = {'objective': 'binary:logistic',
 'n_estimators': 220,
 'learning_rate': 0.05,
 'max_depth': 4,
 'min_child_weight': 80,
 'subsample': 0.85,
 'colsample_bytree': 0.85,
 'random_state': 42,
 'tree_method': 'hist'}
FOLD_WINDOWS = [('2018_2020', '2018-01-01', '2021-01-01'),
 ('2021_2022', '2021-01-01', '2023-01-01'),
 ('2023_2024', '2023-01-01', '2025-01-01')]


def ensure(value, name):
    if not value:
        raise ValueError(name)


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            result.update(data)
    return result.hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path, content):
    def serial(value):
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serial(item) for item in value]
        if isinstance(value, np.generic):
            return serial(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(serial(content), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest_array(array):
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(data.dtype).encode("ascii"))
    digest.update(np.array(data.shape, dtype=np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def finalized(run, required):
    document = json_read(run / "FINALIZED.json")
    ensure(document.get("run_id") == run.name and document.get("finalized_at_utc"), "finalized_identity")
    inventory = document.get("file_sha256")
    ensure(isinstance(inventory, dict) and inventory and set(required) <= inventory.keys(), "finalized_inventory")
    for name in inventory:
        file = (run / name).resolve()
        ensure(file.is_relative_to(run.resolve()) and file.is_file(), "finalized_path")
        ensure(file_digest(file) == inventory[name], "finalized_hash:" + name)


def cot_digest(blocks):
    digest = hashlib.sha256()
    digest.update(b"CFTC_GOLD_COT_FOUNDATION_V1\0")
    digest.update(json.dumps(COT_FEATURES, separators=(",", ":")).encode("ascii"))
    for key in BLOCKS:
        value = np.ascontiguousarray(blocks[key], dtype=np.float64)
        digest.update(b"\0" + key.encode("ascii") + b"\0float64\0")
        digest.update(np.array(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def identity(array, expected, name):
    ensure(digest_array(array) == expected, name)


def candidate_ids(ids):
    ensure(tuple(ids) == CANDIDATES, "candidate_ids")


def equal_number(actual, expected):
    if expected is None or actual is None:
        return actual is None and (expected is None or not math.isfinite(expected))
    return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)


def reproduce_b0(row):
    for name, expected in EXPECTED_B0.items():
        ensure(equal_number(row[name], expected), "b0_reproduction:" + name)


def safety(pre, post):
    ensure(bool(pre) and pre == post, "operational_mutation")


def training_configuration(config):
    ensure(config["version"] == [3, 2, 0], "model_xgboost_version")
    learner = config["learner"]
    tree = learner["gradient_booster"]["tree_train_param"]
    for key in ("learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree"):
        # XGBoost stores some configuration values as float32 decimal strings.
        ensure(math.isclose(float(tree[key]), PARAMETERS[key], rel_tol=0, abs_tol=1e-7), "model_config:" + key)
    ensure(int(learner["generic_param"]["seed"]) == 42, "model_seed")
    ensure(learner["objective"]["name"] == "binary:logistic", "model_objective")
    ensure(learner["gradient_booster"]["gbtree_train_param"]["tree_method"] == "hist", "model_tree_method")


def returns_summary(trades, days):
    net = np.array([float(trade["net_r"]) for trade in trades], dtype=np.float64)
    stress = np.array([float(trade["stress_r"]) for trade in trades], dtype=np.float64)
    positive = net[net > 0]
    negative = net[net <= 0]
    def profit_factor(values):
        profit = float(values[values > 0].sum())
        loss = -float(values[values <= 0].sum())
        return profit / loss if loss > 0 else (math.inf if profit > 0 else 0.)
    average_win = float(positive.mean()) if len(positive) else None
    average_loss = float(negative.mean()) if len(negative) else None
    payoff = average_win / abs(average_loss) if average_win is not None and average_loss else None
    break_even = 1 / (1 + payoff) if payoff is not None else None
    wr = len(positive) / len(net) if len(net) else 0.
    curve = np.concatenate(([0.], np.cumsum(net)))
    result = {"trades": len(net), "wins": len(positive), "losses": len(negative),
            "realized_wr": wr, "pf": profit_factor(net),
            "mean_r": float(net.mean()) if len(net) else 0., "pnl_r": float(net.sum()),
            "max_dd_r": float(np.min(curve - np.maximum.accumulate(curve))),
            "trades_per_day": len(net) / max(days, 1), "cost_stress_pf": profit_factor(stress),
            "cost_stress_mean_r": float(stress.mean()) if len(stress) else 0.,
            "cost_stress_pnl_r": float(stress.sum()),
            "average_winner_r": average_win, "average_loser_r": average_loss,
            "payoff_ratio": payoff, "break_even_wr": break_even,
            "break_even_adjusted_edge": wr - break_even if break_even is not None else None}
    holds = [(pd.Timestamp(item["exit_time_api"]) - pd.Timestamp(item["entry_time_api"]))
             .total_seconds() / 60 for item in trades]
    accounts = np.array([float(item["account_pnl"]) for item in trades], dtype=np.float64)
    tp = sum(item["exit_reason"] == "take_profit" for item in trades)
    result.update({"gross_profit_r": float(positive.sum()), "gross_loss_r": float(-negative.sum()),
                   "tp_first_wr": tp / len(trades) if trades else 0., "tp_count": tp,
                   "sl_count": sum(item["exit_reason"] == "stop_loss" for item in trades),
                   "profitable_timeout_count": sum(item["exit_reason"] == "timeout" and float(item["net_r"]) > 0 for item in trades),
                   "losing_timeout_count": sum(item["exit_reason"] in {"timeout", "cohort_end"} and float(item["net_r"]) <= 0 for item in trades),
                   "cohort_end_count": sum(item["exit_reason"] == "cohort_end" for item in trades),
                   "average_hold_minutes": float(np.mean(holds)) if holds else None,
                   "risk_sized_pf": profit_factor(accounts)})
    return result


def aggregate(trades):
    rows = []
    days_total = 0
    for name, start, stop in FOLD_WINDOWS:
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        days = (stop - start).days
        days_total += days
        selected = [trade for trade in trades if start <= pd.Timestamp(trade["entry_time_api"]) < stop]
        rows.append({"fold": name, **returns_summary(selected, days)})
    rows.append({"fold": "pooled", **returns_summary(trades, days_total)})
    return rows


def verify_sources():
    required = {PARENT: ["training_script.py", "manifest.json"],
                REFERENCE: ["training_script.py", "model_inventory.json", "fold_metrics.csv"],
                TIMESTAMPS: ["exact_timestamps.npz"],
                FOUNDATION: ["manifest.json", "validator.json", "metrics.json",
                             "cftc_gold_cot_features.npz", "cot_observations.csv"]}
    for name, artifacts in required.items():
        finalized(ROOT / "training_runs" / name, artifacts)
    for path, value in SOURCE_HASHES.items():
        ensure(file_digest(ROOT / path) == value, "frozen_source:" + path)
    base = ROOT / "training_runs" / FOUNDATION
    ensure(json_read(base / "manifest.json")["status"] == "pass", "foundation_status")
    ensure(json_read(base / "validator.json")["overall"] == "PASS", "foundation_validator")
    ensure(file_digest(base / "cftc_gold_cot_features.npz") == PHYSICAL_HASH, "foundation_physical_hash")
    ensure(file_digest(base / "cot_observations.csv") == DATASET_HASH, "foundation_dataset_hash")
    with np.load(base / "cftc_gold_cot_features.npz", allow_pickle=False) as data:
        ensure(data["feature_names"].tolist() == COT_FEATURES, "cot_feature_order")
        cot = {name: data[name] for name in BLOCKS}
    for name, array in cot.items():
        ensure(array.dtype == np.dtype("float64") and array.shape == (BLOCKS[name][0], 5)
               and np.isfinite(array).all(), "cot_shape_dtype:" + name)
    ensure(cot_digest(cot) == LOGICAL_HASH, "cot_logical_hash")
    spec = importlib.util.spec_from_file_location(
        "independent_frozen_parent", ROOT / "training_runs" / PARENT / "training_script.py")
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    parent.drl_trading_v2.DATA_DIR = str(ROOT)
    ensure(parent.xgb.__version__ == "3.2.0", "xgboost_version")
    ensure(parent.FIXED_XGB_PARAMETERS == PARAMETERS and parent.THRESHOLD == .75
           and parent.TRAIN_MONTHS == 18, "parent_parameters")
    ensure([(n, str(s.date()), str(e.date())) for n, s, e in parent.FOLDS] == FOLD_WINDOWS,
           "parent_folds")
    return parent, cot


def rebuild(parent):
    frame, names = parent.prepare_barrier_data()
    frame = frame.reset_index(drop=True).copy()
    ensure(names == BASE_FEATURES and len(names) == 31, "base_features")
    labels = parent.build_execution_aligned_labels(frame)
    times = frame.TIME_DT
    ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    y = labels.C1_TARGET.to_numpy(dtype=np.int8)
    mature = labels.C1_MATURE.to_numpy(dtype=bool)
    matured_at = labels.C1_MATURITY_NS.to_numpy(dtype=np.int64)
    positions = np.arange(len(frame), dtype=np.int64)
    legacy_end = positions + parent.LEGACY_HORIZON_ROWS
    valid_legacy = legacy_end < len(frame)
    information_ns = np.full(len(frame), np.iinfo(np.int64).max, dtype=np.int64)
    information_ns[valid_legacy] = ns[legacy_end[valid_legacy]]
    folds, evidence = [], []
    with np.load(ROOT / "training_runs" / TIMESTAMPS / "exact_timestamps.npz", allow_pickle=False) as original:
        for number, (name, begin, finish) in enumerate(FOLD_WINDOWS, 1):
            begin, finish = pd.Timestamp(begin), pd.Timestamp(finish)
            train_mask = ((times >= begin - pd.DateOffset(months=18)) & (times < begin)).to_numpy()
            train = positions[train_mask & valid_legacy & mature & (information_ns < begin.value)
                              & (matured_at < begin.value)]
            score = positions[((times >= begin) & (times < finish)).to_numpy() & mature]
            for suffix, indices in [("train", train), ("score", score)]:
                block = "fold" + str(number) + "_" + suffix
                ensure(len(indices) == BLOCKS[block][0], "block_rows:" + block)
                identity(ns[indices], BLOCKS[block][1], "timestamp:" + block)
                ensure(np.array_equal(original[block + "_broker_ns"], ns[indices]), "timestamp_alignment")
                x = frame.loc[indices, names].to_numpy(dtype=np.float32)
                identity(x, EXPECTED_X[block], "gold_matrix:" + block)
                evidence.append({"block": block, "rows": len(indices),
                                 "timestamp_sha256": digest_array(ns[indices]), "gold_sha256": digest_array(x)})
            identity(y[train], EXPECTED_Y[f"fold{number}_train"], "c1_target:" + name)
            evidence[-2]["c1_sha256"] = digest_array(y[train])
            folds.append((number, name, train, score))
    return frame, folds, evidence


def verify_run(run, checks):
    def checked(name, value):
        checks[name] = bool(value)
        ensure(value, name)
    manifest = json_read(run / "manifest.json")
    checked("formal_manifest", manifest.get("experiment_name") == EXPERIMENT
            and manifest.get("run_id") == run.name and manifest.get("status") == "in_progress")
    commit = manifest["git_commit"]
    executed = run / "training_script.py"
    checked("executed_snapshot", file_digest(executed) == manifest["training_script_sha256"])
    committed = subprocess.check_output(["git", "show", commit + ":cftc_gold_cot_b0_b1_v1.py"], cwd=ROOT)
    checked("executed_git_identity", executed.read_bytes().replace(b"\r\n", b"\n")
            == committed.replace(b"\r\n", b"\n"))
    checked("reviewed_runner_identity", executed.read_bytes().replace(b"\r\n", b"\n")
            == (ROOT / "cftc_gold_cot_b0_b1_v1.py").read_bytes().replace(b"\r\n", b"\n"))
    validator_source = subprocess.check_output(
        ["git", "show", commit + ":validate_cftc_gold_cot_b0_b1_v1.py"], cwd=ROOT)
    checked("validator_git_identity", validator_source.replace(b"\r\n", b"\n")
            == Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            == (run / "validator_script.py").read_bytes().replace(b"\r\n", b"\n"))
    artifact_items = manifest.get("artifacts", [])
    artifact_names = [item["path"] for item in artifact_items]
    checked("required_artifact_inventory", len(artifact_names) == len(set(artifact_names))
            and {"validator_script.py", "execution_spec.json", "source_provenance.json",
                 "identity_audit.json", "model_inventory.json", "paired_oof_predictions.npz",
                 "fold_metrics.csv", "trade_ledger.csv", "candidates.csv", "metrics.json", "report.md",
                 "operational_safety_pre.json", "operational_safety_post.json", "model.sha256",
                 "environment.txt", "training_script.py"} <= set(artifact_names))
    for item in artifact_items:
        artifact = (run / item["path"]).resolve()
        checked("artifact:" + item["path"], artifact.is_relative_to(run)
                and artifact.is_file() and file_digest(artifact) == item["sha256"])
    checked("no_search", manifest["search"]["performed"] is False
            and manifest["search"]["predefined_search_space"] == {})
    checked("no_production_replacement", manifest["promotion"]["requested"] is False
            and manifest["promotion"]["replacement_authorized"] is False
            and manifest["promotion"]["operational_artifact_changed"] is False)
    checked("training_parameters", manifest["model"]["parameters"] == PARAMETERS
            and manifest["random_seeds"] == {mid: 42 for mid in CANDIDATES})
    pre, post = (json_read(run / name) for name in ("operational_safety_pre.json", "operational_safety_post.json"))
    safety(pre, post)
    checked("operational_files_unchanged", pre == {name: file_digest(ROOT / name) for name in OPERATIONAL}
            == manifest["operational_hashes_before"] == manifest["operational_hashes_after"])
    parent, cot = verify_sources()
    checks["finalized_sources_and_cftc_certification"] = True
    provenance = json_read(run / "source_provenance.json")
    checked("recorded_source_provenance", provenance == {
        "foundation_run": FOUNDATION, "logical_hash": LOGICAL_HASH, "physical_hash": PHYSICAL_HASH,
        "dataset_hash": DATASET_HASH, "source_hashes": SOURCE_HASHES,
        "finalized_hashes": {name: file_digest(ROOT / "training_runs" / name / "FINALIZED.json")
                             for name in (PARENT, REFERENCE, TIMESTAMPS, FOUNDATION)}})
    spec = json_read(run / "execution_spec.json")
    expected_lists = {CANDIDATES[0]: BASE_FEATURES, CANDIDATES[1]: BASE_FEATURES + COT_FEATURES}
    checked("frozen_execution_spec", spec == {
        "candidates": list(CANDIDATES), "feature_lists": expected_lists, "parameters": PARAMETERS,
        "xgboost_version": "3.2.0", "threshold": .75, "training_months": 18,
        "folds": [list(row) for row in FOLD_WINDOWS], "simulator": "S5", "trailing": False,
        "search_performed": False, "production_replacement": False, "source_hashes": SOURCE_HASHES})
    checked("feature_counts", len(expected_lists[CANDIDATES[0]]) == 31
            and len(expected_lists[CANDIDATES[1]]) == 36
            and manifest["model"]["candidate_features"] == expected_lists)
    frame, folds, evidence = rebuild(parent)
    checked("reconstructed_timestamp_gold_c1_identities", json_read(run / "identity_audit.json") == evidence)
    models = json_read(run / "model_inventory.json")
    expected_pairs = {(mid, name) for mid in CANDIDATES for name, _, _ in FOLD_WINDOWS}
    checked("exact_six_models", len(models) == 6
            and {(item["candidate_id"], item["fold"]) for item in models} == expected_pairs)
    checked("no_extra_models", {p.relative_to(run).as_posix() for p in (run / "models").iterdir()}
            == {item["path"] for item in models})
    checked("model_artifact_inventory", {item["path"] for item in models} <= set(artifact_names))
    checked("manifest_model_artifact", manifest["model"]["artifact_path"] == models[-1]["path"]
            and manifest["model"]["artifact_sha256"] == models[-1]["sha256"]
            and (run / "model.sha256").read_text().split()[0] == models[-1]["sha256"])
    with (run / "candidates.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    candidate_ids([row["candidate_id"] for row in candidates])
    checks["exact_candidates"] = True
    metrics = json_read(run / "metrics.json")
    candidate_ids(metrics["results"].keys())
    checked("metrics_no_search_or_promotion", metrics["search_performed"] is False
            and metrics["production_promotion_requested"] is False)
    with (run / "trade_ledger.csv").open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    checked("ledger_candidates", {row["candidate_id"] for row in ledger} <= set(CANDIDATES))
    with (run / "fold_metrics.csv").open(encoding="utf-8", newline="") as handle:
        fold_table = list(csv.DictReader(handle))
    checked("fold_table_shape", len(fold_table) == 8 and
            {(row["candidate_id"], row["fold"]) for row in fold_table}
            == {(mid, fold) for mid in CANDIDATES for fold in [r[0] for r in FOLD_WINDOWS] + ["pooled"]})
    independent = {}
    with np.load(run / "paired_oof_predictions.npz", allow_pickle=False) as predictions:
        expected_keys = {f"{mid}_fold{no}" for mid in CANDIDATES for no in (1, 2, 3)}
        expected_keys |= {f"fold{no}_indices" for no in (1, 2, 3)}
        checked("prediction_keys", set(predictions.files) == expected_keys)
        for mid in CANDIDATES:
            parts = []
            for number, fold, _, indices in folds:
                item = next(item for item in models if item["candidate_id"] == mid and item["fold"] == fold)
                path = (run / item["path"]).resolve()
                checked("model_hash:" + mid + fold, path.is_relative_to(run / "models")
                        and file_digest(path) == item["sha256"])
                checked("model_parameters:" + mid + fold, item["parameters"] == PARAMETERS
                        and item["rounds"] == 220 and item["features"] == expected_lists[mid])
                training_configuration(item["training_config"])
                checks["model_training_config:" + mid + fold] = True
                model = parent.xgb.XGBClassifier()
                model.load_model(path)
                checked("model_structure:" + mid + fold, model.get_booster().num_boosted_rounds() == 220
                        and model.get_booster().feature_names == expected_lists[mid]
                        and model.get_booster().num_features() == len(expected_lists[mid]))
                score = frame.loc[indices, BASE_FEATURES].copy()
                if mid == CANDIDATES[1]:
                    for column, name in enumerate(COT_FEATURES):
                        score[name] = cot[f"fold{number}_score"][:, column]
                model.set_params(device="cpu")
                probabilities = model.predict_proba(score[expected_lists[mid]])[:, 1].astype(np.float32)
                checked("predictions:" + mid + fold,
                        np.array_equal(indices, predictions[f"fold{number}_indices"])
                        and np.array_equal(probabilities, predictions[f"{mid}_fold{number}"]))
                cohort_part = frame.loc[indices, PRICE_COLUMNS].copy()
                cohort_part["buy_prob"] = probabilities
                cohort_part["sell_prob"] = np.float32(0)
                parts.append(cohort_part)
            cohort = parent.semantics.finalize_cohort(pd.concat(parts, ignore_index=True), mid, offset_hours=0)
            replay, _ = parent.semantics.simulate(cohort, parent.semantics.SIMULATORS[-1])
            submitted = [row for row in ledger if row["candidate_id"] == mid]
            checked("replay_trade_count:" + mid, len(replay) == len(submitted))
            for actual, stored in zip(replay, submitted):
                checked("ledger_fields:" + mid, set(stored) == set(actual) | {"candidate_id"})
                for key, value in actual.items():
                    if isinstance(value, (bool, np.bool_, str)):
                        matches = str(value) == stored[key]
                    else:
                        matches = equal_number(stored[key], value)
                    checked("replay:" + mid + ":" + key, matches)
                denominator = float(stored["sl_distance"]) + float(stored["spread_points"]) * .01
                for key, cost in (("net_r", 5.), ("stress_r", 10.)):
                    calculated = (float(stored["gross_price"]) - (float(stored["spread_points"]) + cost) * .01) / denominator
                    checked("cost_recompute:" + mid + key, equal_number(stored[key], calculated))
            independent[mid] = aggregate(submitted)
            for actual, reported in zip(independent[mid], metrics["results"][mid]):
                checked("fold_name:" + mid + actual["fold"], actual["fold"] == reported["fold"])
                table_row = next(row for row in fold_table if row["candidate_id"] == mid and row["fold"] == actual["fold"])
                for key, value in actual.items():
                    if key != "fold":
                        checked("metric:" + mid + actual["fold"] + key, equal_number(reported[key], value)
                                and equal_number(None if table_row[key] == "" else table_row[key], value))
            checked("four_metric_scopes:" + mid, len(metrics["results"][mid]) == 4)
            candidate_row = next(row for row in candidates if row["candidate_id"] == mid)
            for key, value in independent[mid][-1].items():
                if key != "fold":
                    checked("candidate_metric:" + mid + key,
                            equal_number(None if candidate_row[key] == "" else candidate_row[key], value))
    reproduce_b0(independent[CANDIDATES[0]][-1])
    checks["b0_exact_reproduction"] = True
    b0, b1 = (independent[mid][-1] for mid in CANDIDATES)
    checked("delta_fields", set(metrics["delta_b1_minus_b0"]) == {
        "trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf"})
    for key, value in metrics["delta_b1_minus_b0"].items():
        checked("delta:" + key, equal_number(value, b1[key] - b0[key]))
    reasons = []
    for key, floor, allow_equal in [("realized_wr", .60, True), ("pf", 1.05, False),
                                    ("mean_r", 0., False), ("pnl_r", 0., False),
                                    ("break_even_adjusted_edge", 0., False), ("cost_stress_pf", 1., False)]:
        if b1[key] is None or (b1[key] < floor if allow_equal else b1[key] <= floor):
            reasons.append(key)
    for row in independent[CANDIDATES[1]][:-1]:
        if row["trades"] < 10 or row["realized_wr"] < .50 or row["pf"] < .80 or row["mean_r"] < -.10 or row["max_dd_r"] < -20:
            reasons.append("catastrophic_fold:" + row["fold"])
    checked("quality_classification", metrics["quality_failure_reasons"] == reasons
            and metrics["cftc_family_status"] == ("FAIL" if reasons else "PASS"))
    checked("frequency_and_preferred_quality", equal_number(metrics["frequency_uplift_pct"],
            100 * (b1["trades_per_day"] / b0["trades_per_day"] - 1))
            and metrics["preferred_quality"] == (b1["pf"] >= 1.15 and b1["cost_stress_pf"] >= 1.05))
    checked("operational_files_unchanged_after_validation", pre == {
        name: file_digest(ROOT / name) for name in OPERATIONAL})
    return independent


def validate(run):
    run = run.resolve()
    ensure(run.parent == ROOT / "training_runs" and not (run / "FINALIZED.json").exists(),
           "Do not modify prior finalized archives")
    checks, independent, error = {}, {}, None
    try:
        independent = verify_run(run, checks)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        checks[str(exception)] = False
    failed = [name for name, result in checks.items() if not result]
    output = {"overall": "FAIL" if failed else "PASS", "checks": checks,
              "failed_check_names": failed, "error": error, "independent_metrics": independent}
    json_write(run / "validator.json", output)
    (run / "validator.md").write_text("# Independent CFTC B0/B1 validator\n\nOverall: " + output["overall"]
                                      + "\n\nFailed checks: " + repr(failed) + "\n", encoding="utf-8")
    manifest = json_read(run / "manifest.json")
    manifest.setdefault("registry", {})["validator_result"] = "independent " + output["overall"]
    artifacts = manifest.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item["path"] not in {"validator.json", "validator.md"}]
    for name in ("validator.json", "validator.md"):
        artifacts.append({"kind": "independent_validator", "path": name,
                          "sha256": file_digest(run / name), "retention_status": "stored_in_run_directory"})
    json_write(run / "manifest.json", manifest)
    print("INDEPENDENT_VALIDATOR_" + output["overall"])
    print("failed_check_names=" + repr(failed))
    return not failed


def self_test():
    ensure(len(BASE_FEATURES) == 31 and len(BASE_FEATURES + COT_FEATURES) == 36, "counts")
    ensure((BASE_FEATURES + COT_FEATURES)[31:] == COT_FEATURES, "ordering")
    def rejected(action):
        try:
            action()
        except (ValueError, FileNotFoundError):
            return
        raise AssertionError("Bad evidence accepted")
    matrices = {key: np.zeros((1, 5), np.float64) for key in BLOCKS}
    expected = cot_digest(matrices)
    matrices["fold1_train"][0, 0] = 1.
    rejected(lambda: ensure(cot_digest(matrices) == expected, "logical_hash"))
    rejected(lambda: identity(np.zeros((1, 31), np.float32), EXPECTED_X["fold1_train"], "GOLD"))
    rejected(lambda: identity(np.zeros(1, np.int8), EXPECTED_Y["fold1_train"], "C1"))
    rejected(lambda: candidate_ids([*CANDIDATES, "extra"]))
    reproduce_b0(EXPECTED_B0)
    rejected(lambda: reproduce_b0({**EXPECTED_B0, "max_dd_r": 0.}))
    rejected(lambda: safety({"gemini": "a"}, {"gemini": "b"}))
    # Eight synthetic rows, two trees; no historical data or formal run directory.
    import xgboost as xgb
    tiny = xgb.XGBClassifier(**{**PARAMETERS, "n_estimators": 2}, device="cpu")
    tiny.fit(np.arange(16, dtype=np.float32).reshape(8, 2), np.array([0, 1] * 4))
    config = json.loads(tiny.get_booster().save_config())
    training_configuration(config)
    config["learner"]["generic_param"]["seed"] = "43"
    rejected(lambda: training_configuration(config))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "matrix").write_bytes(b"original")
        json_write(path / "FINALIZED.json", {"run_id": path.name, "finalized_at_utc": "synthetic",
                                             "file_sha256": {"matrix": file_digest(path / "matrix")}})
        finalized(path, ["matrix"])
        (path / "matrix").write_bytes(b"mutated")
        rejected(lambda: finalized(path, ["matrix"]))
    ledger = [{"entry_time_api": "2018-01-02", "net_r": 1., "stress_r": .9},
              {"entry_time_api": "2021-01-01", "net_r": -.5, "stress_r": -.6},
              {"entry_time_api": "2023-01-01", "net_r": 2., "stress_r": 1.9}]
    for trade in ledger:
        trade.update(exit_time_api=trade["entry_time_api"], account_pnl=trade["net_r"],
                     exit_reason="take_profit" if trade["net_r"] > 0 else "stop_loss")
    result = aggregate(ledger)[-1]
    assert result["trades"] == 3 and result["wins"] == 2 and result["losses"] == 1
    assert result["pnl_r"] == 2.5 and result["pf"] == 6 and result["max_dd_r"] == -.5
    assert equal_number(result["cost_stress_pf"], 2.8 / .6)
    assert equal_number(result["trades_per_day"], 3 / 2557)
    tree = ast.parse((ROOT / "cftc_gold_cot_b0_b1_v1.py").read_text(encoding="utf-8"))
    peer = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CANDIDATES" for t in node.targets))
    assert tuple(peer) == CANDIDATES
    own = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "cftc_gold_cot_b0_b1_v1"
                   for n in ast.walk(own))
    print("SELF_TEST_PASS: feature_order logical_hash archive GOLD C1 candidates B0 safety pooled independence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")
    return 0 if validate(args.run_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
