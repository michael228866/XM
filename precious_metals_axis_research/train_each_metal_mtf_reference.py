from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from barrier_classifier_strategy import HORIZON, build_barrier_target  # noqa: E402
from precious_metals_axis_research.axis_symbol_smoke import (  # noqa: E402
    DATA_DIRS,
    read_price_csv,
)
from precious_metals_axis_research.axis_timeframe_smoke import add_indicators  # noqa: E402
from precious_metals_axis_research.train_each_metal_custom import (  # noqa: E402
    OUTPUT_BEST,
    OUTPUT_CSV,
    OUTPUT_JSON,
    OUTPUT_MD,
    SYMBOL_CONFIG,
    evaluate_symbol_timeframe,
)


OUTPUT_CSV_MTF = RESEARCH_DIR / "each_metal_mtf_reference_results.csv"
OUTPUT_JSON_MTF = RESEARCH_DIR / "each_metal_mtf_reference_results.json"
OUTPUT_MD_MTF = RESEARCH_DIR / "each_metal_mtf_reference_report.md"
OUTPUT_BEST_MTF = RESEARCH_DIR / "each_metal_mtf_reference_best_by_symbol.json"

REFERENCE_TIMEFRAMES = ["M15", "M30", "H1", "H4", "H12", "Daily"]
MAX_ROWS_FOR_MTF = 12_000


def find_files(symbol: str) -> dict[str, Path]:
    files = {}
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in data_dir.glob(f"{symbol}_*.csv"):
            parts = path.stem.rsplit("_", 3)
            if len(parts) == 4:
                files[parts[1]] = path
    return files


def add_rsi_and_context(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = add_indicators(df)
    diff = df["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df[f"{prefix}_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df[f"{prefix}_TREND"] = np.where(df["CLOSE"] > df["CLOSE"].rolling(20).mean(), 1, -1)
    df[f"{prefix}_ATR_PCT"] = df["ATR"] / (df["CLOSE"].abs() + 1e-9)
    df[f"{prefix}_MACD_ATR"] = df["MACD_HIST"] / (df["ATR"] + 1e-9)
    df[f"{prefix}_VOLA_RATIO"] = df["ATR"] / (df["ATR"].rolling(120).mean() + 1e-9)
    return df


def load_rich_mtf_case(symbol: str, base_tf: str):
    files = find_files(symbol)
    if base_tf not in files:
        raise FileNotFoundError(f"{symbol} {base_tf} file was not found.")

    base = read_price_csv(files[base_tf])
    if base is None:
        raise ValueError(f"Unable to read {files[base_tf]}")
    base = add_rsi_and_context(base, "BASE")
    base["HOUR_SIN"] = np.sin(2 * np.pi * base["TIME_DT"].dt.hour / 24)
    base["HOUR_COS"] = np.cos(2 * np.pi * base["TIME_DT"].dt.hour / 24)
    base["DAY_OF_WEEK"] = base["TIME_DT"].dt.dayofweek / 7.0
    frame = base.sort_values("TIME_DT").reset_index(drop=True)

    features = [
        "BASE_RSI",
        "BASE_ATR_PCT",
        "BASE_MACD_ATR",
        "BB_WIDTH",
        "BIAS_20",
        "BODY_PCT",
        "ROC_5",
        "BASE_VOLA_RATIO",
        "HOUR_SIN",
        "HOUR_COS",
        "DAY_OF_WEEK",
    ]

    for timeframe in REFERENCE_TIMEFRAMES:
        if timeframe == base_tf or timeframe not in files:
            continue
        ref = read_price_csv(files[timeframe])
        if ref is None:
            continue
        prefix = f"REF_{timeframe}"
        ref = add_rsi_and_context(ref, prefix)
        cols = [
            "TIME_DT",
            f"{prefix}_RSI",
            f"{prefix}_TREND",
            f"{prefix}_ATR_PCT",
            f"{prefix}_MACD_ATR",
            f"{prefix}_VOLA_RATIO",
        ]
        ref[cols[1:]] = ref[cols[1:]].shift(1)
        frame = pd.merge_asof(frame, ref[cols], on="TIME_DT")
        features.extend(cols[1:])

    frame[features] = frame[features].shift(1)
    frame["BARRIER_TARGET"] = build_barrier_target(frame)
    frame = frame.iloc[:-HORIZON].dropna(
        subset=features + ["BARRIER_TARGET", "ATR", "CLOSE", "BASE_RSI"]
    )
    if len(frame) > MAX_ROWS_FOR_MTF:
        frame = frame.tail(MAX_ROWS_FOR_MTF)
    return frame.reset_index(drop=True), features


def patch_custom_outputs():
    import precious_metals_axis_research.train_each_metal_custom as custom

    custom.load_case = load_rich_mtf_case
    custom.OUTPUT_CSV = OUTPUT_CSV_MTF
    custom.OUTPUT_JSON = OUTPUT_JSON_MTF
    custom.OUTPUT_MD = OUTPUT_MD_MTF
    custom.OUTPUT_BEST = OUTPUT_BEST_MTF
    custom.MAX_ROWS_BY_TIMEFRAME.update({"M15": 12_000, "M30": 12_000, "H1": 12_000})
    custom.SYMBOL_CONFIG["GOLD#"]["timeframes"] = ["H1"]
    custom.SYMBOL_CONFIG["SILVER#"]["timeframes"] = ["H1"]
    custom.SYMBOL_CONFIG["XAUEUR#"]["timeframes"] = ["H1"]
    custom.SYMBOL_CONFIG["XPTUSD#"]["timeframes"] = ["H1"]
    custom.SYMBOL_CONFIG["XPDUSD#"]["timeframes"] = ["H1"]
    custom.SYMBOL_CONFIG["GAUCNH#"]["timeframes"] = ["H1", "H4"]
    return custom


def main() -> int:
    custom = patch_custom_outputs()
    result = custom.main()
    summary_path = OUTPUT_JSON_MTF
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["feature_note"] = (
            "Main timeframe trades are enriched with reference RSI/trend/ATR/MACD/VOLA "
            "from M15/M30/H1/H4/H12/Daily where available."
        )
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("MTF reference features were used.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
