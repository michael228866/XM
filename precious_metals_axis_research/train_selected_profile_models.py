from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.optimize_training_profiles_silver_xaueur import (  # noqa: E402
    MODEL_PROFILES,
)
from precious_metals_axis_research.walk_forward_all_metals_shared import load_frames  # noqa: E402


SILVER_MODEL_FILE = RESEARCH_DIR / "silver_h1_regime_selected_xgb.json"
SILVER_METADATA_FILE = RESEARCH_DIR / "silver_h1_regime_selected_xgb.metadata.json"
XAUEUR_SHARED_MODEL_FILE = RESEARCH_DIR / "all_metals_h1_smooth_more_trees_xgb.json"
XAUEUR_SHARED_METADATA_FILE = RESEARCH_DIR / "all_metals_h1_smooth_more_trees_xgb.metadata.json"

SILVER_PARAMS = {
    "threshold": 0.52,
    "edge_threshold": 0.0,
    "tp_atr": 6.0,
    "sl_atr": 6.0,
    "max_hold": 336,
    "direction_mode": "long",
    "regime_filter": {
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.0,
        "stress_spread_atr_max": 0.75,
    },
}

XAUEUR_PARAMS = {
    "threshold": 0.54,
    "edge_threshold": 0.0,
    "tp_atr": 2.2,
    "sl_atr": 4.2,
    "max_hold": 288,
    "direction_mode": "both",
}


def train_model(train_df: pd.DataFrame, features: list[str], profile: dict) -> xgb.XGBClassifier:
    sample_weight = build_profit_sample_weight(
        train_df, train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    )
    sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
    sample_weight = np.maximum(sample_weight, 1e-6)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        random_state=42,
        verbosity=0,
        **profile,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def train_silver() -> None:
    frame, features = load_case("SILVER#", "H1")
    profile_name = "current_symbol"
    print(f"Training SILVER# selected model rows={len(frame):,} profile={profile_name}")
    model = train_model(frame, features, MODEL_PROFILES[profile_name])
    model.save_model(str(SILVER_MODEL_FILE))
    metadata = {
        "symbol": "SILVER#",
        "timeframe": "H1",
        "profile": profile_name,
        "profile_params": MODEL_PROFILES[profile_name],
        "strategy_params": SILVER_PARAMS,
        "rows": len(frame),
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "train_selected_profile_models.py",
    }
    SILVER_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {SILVER_MODEL_FILE}")
    print(f"Wrote {SILVER_METADATA_FILE}")


def train_xaueur_shared() -> None:
    frames, features = load_frames()
    profile_name = "smooth_more_trees"
    train_df = pd.concat(frames.values(), ignore_index=True)
    print(
        f"Training all-metals shared H1 selected model rows={len(train_df):,} "
        f"profile={profile_name}"
    )
    model = train_model(train_df, features, MODEL_PROFILES[profile_name])
    model.save_model(str(XAUEUR_SHARED_MODEL_FILE))
    metadata = {
        "primary_symbol": "XAUEUR#",
        "training_symbols": list(frames),
        "timeframe": "H1",
        "profile": profile_name,
        "profile_params": MODEL_PROFILES[profile_name],
        "strategy_params": XAUEUR_PARAMS,
        "rows": len(train_df),
        "features": features,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "train_selected_profile_models.py",
    }
    XAUEUR_SHARED_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {XAUEUR_SHARED_MODEL_FILE}")
    print(f"Wrote {XAUEUR_SHARED_METADATA_FILE}")


def main() -> int:
    train_silver()
    train_xaueur_shared()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
