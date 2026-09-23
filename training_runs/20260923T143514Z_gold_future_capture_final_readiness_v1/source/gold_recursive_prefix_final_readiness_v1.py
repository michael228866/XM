"""Source review and synthetic EMA state comparisons; never import model code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_definition():
    tree = ast.parse((ROOT / "drl_trading_v2.py").read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add_indicators")
    return hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()


def equivalence_tests():
    import numpy as np
    import pandas as pd
    # Fixed non-market fixture. No historical prices, labels or models.
    close = pd.Series([100.0 + (i % 31) * .125 + i / 8192 for i in range(4200)])
    def state(values):
        e12 = values.ewm(span=12, adjust=False).mean()
        e26 = values.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        signal = macd.ewm(span=9, adjust=False).mean()
        return {"EMA12_STATE": e12, "EMA26_STATE": e26,
                "MACD_SIGNAL_STATE": signal, "MACD_HIST": macd - signal}
    full = state(close)
    rows = []
    for warmup in (32, 254, 4096):
        partial = state(close.iloc[4096 - warmup:])
        for name, a in full.items():
            a, b = a.iloc[4096:4200].to_numpy(), partial[name].loc[4096:4199].to_numpy()
            difference = np.abs(a - b)
            rows.append({"feature_name": name, "timeframe": "M1", "full_prefix_start": "synthetic_index_0",
                "test_target_start": "synthetic_index_4096", "candidate_warmup": warmup,
                "max_abs_difference": float(difference.max()),
                "max_rel_difference": float((difference / np.maximum(np.abs(a), 1e-300)).max()),
                "exact_match": bool(np.array_equal(a, b)),
                "reason": "Synthetic implementation comparison only; rounded convergence is not certified historical identity"})
    assert any(not row["exact_match"] for row in rows if row["candidate_warmup"] == 32)
    assert all(row["exact_match"] for row in rows if row["candidate_warmup"] == 4096)
    # Confirm pandas defaults including missing observations, without redefining them.
    nan = pd.Series([1.0, float("nan"), 3.0])
    assert nan.ewm(span=12, adjust=False).mean().equals(
        nan.ewm(span=12, adjust=False, min_periods=0, ignore_na=False).mean())
    return {"synthetic_only": True, "pandas_version": pd.__version__, "tests": rows,
            "historical_equivalence_tested": False, "historical_equivalence_proven": False,
            "reason": "No complete certified native history / original initialization state supplied"}


def investigate():
    spec = json.loads((ROOT / "execution_spec_gold_independent_secondary_classifier_v1.json").read_text(encoding="utf-8"))
    features = spec["features"]
    definitions = {"M1_RSI": "diff; gains/losses where(...,0); rolling14 mean; epsilon1e-6",
        "ATR": "true range max skipna; rolling14 mean",
        "MACD_HIST": "EMA12 - EMA26 - EMA9(EMA12-EMA26); recursive",
        "BB_WIDTH": "rolling20 sample std ddof1 *4 / (rolling20 mean+1e-6)",
        "BIAS_20": "(close-rolling20 mean)/(rolling20 mean+1e-6)",
        "BODY_PCT": "abs(close-open)/(high-low+1e-6)", "ROC_5": "pct_change(5); installed pandas3 default no fill",
        "VOLA_RATIO": "ATR/(ATR.rolling240.mean()+1e-6)",
        "HOUR_SIN": "sin(2*pi*bar hour/24)", "HOUR_COS": "cos(2*pi*bar hour/24)",
        "DAY_OF_WEEK": "bar dayofweek/7"}
    return {"prefix_policy": "UNRESOLVED", "status": "PARTIAL", "approved": False,
        "historically_equivalent": False, "outcome_tuning": False,
        "pipeline_sha256": digest(ROOT / "drl_trading_v2.py"), "feature_hash": feature_definition(),
        "feature_inventory": [{"feature_name": name, "timeframe": name[:-6] if name.endswith('_TREND') else 'M1',
            "recursive": name == "MACD_HIST", "definition": definitions.get(name,
                "native close > rolling20 mean =>1 else -1; native shift1; backward merge_asof; final M1 shift1"),
            "earliest_certified_source_utc": None} for name in features],
        "ema_semantics": {"spans": [12, 26, 9], "adjust": False, "min_periods": 0, "ignore_na": False,
            "initialization": "first non-NaN close for price EMAs; first MACD=0 for signal with valid first close",
            "nan": "ignore_na=False weights absolute positions; leading NaNs remain NaN; no input imputation",
            "native_recursive_state_used": False},
        "preprocessing": "sorted filename order; normalize headings; naive DATE/TIME; sort_values TIME_DT; last file per timeframe wins; no deduplication; all31 shift1 then dropna; wrapper tail label trimming does not recompute state",
        "training_start_reference": "2016-07-01T00:00:00 broker-naive; first B0 training window, not recursive source origin",
        "data_start_reference": "2016-06-30T21:00:00Z; archived training metadata, not certified source origin",
        "historical_recursive_source_start": None,
        "source_chain": ["S4 confirmation -> discovery frozen_inputs -> B0 reconstruct -> C1 prepare_barrier_data",
            "barrier_final_train.prepare_barrier_data -> drl_trading_v2.load_and_prepare_data -> add_indicators"],
        "blockers": ["Original complete certified prefix and initialization origin unavailable",
            "Current raw collector does not compute or certify recursive state",
            "Proposal is not approved; native Monthly warmup can require 21 completed months",
            "Source timezone/calendar and exact source range remain unresolved"],
        "strategy_outcome_inspected": False, "models_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.run_dir / "FINALIZED.json").exists():
        raise FileExistsError("Finalized archive")
    for name, result in (("prefix_final_readiness.json", investigate()),
                         ("prefix_equivalence_tests.json", equivalence_tests())):
        with (args.run_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print("PREFIX_POLICY=UNRESOLVED")
