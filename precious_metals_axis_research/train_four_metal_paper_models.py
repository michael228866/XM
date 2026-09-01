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

from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.optimize_training_profiles_silver_xaueur import (  # noqa: E402
    MODEL_PROFILES,
    train_model as train_barrier_model,
)
from precious_metals_axis_research.xpd_alternate_target_exit import (  # noqa: E402
    load_target_case,
    train_model as train_alternate_target_model,
)


XPT_BEST_FILE = RESEARCH_DIR / "xpt_h4_fold_coverage_best.json"
XPD_BEST_FILE = RESEARCH_DIR / "xpd_alternate_target_exit_best.json"

XPT_MODEL_FILE = RESEARCH_DIR / "xpt_h4_fold_coverage_xgb.json"
XPT_METADATA_FILE = RESEARCH_DIR / "xpt_h4_fold_coverage_xgb.metadata.json"
XPD_MODEL_FILE = RESEARCH_DIR / "xpd_h12_alternate_target_xgb.json"
XPD_METADATA_FILE = RESEARCH_DIR / "xpd_h12_alternate_target_xgb.metadata.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_xpt() -> None:
    best = load_json(XPT_BEST_FILE)
    params = best["best_3x"]
    profile_name = best["profile"]
    timeframe = params["base_timeframe"]

    frame, features = load_case(params["symbol"], timeframe)
    profile = MODEL_PROFILES[profile_name]
    print(
        f"Training {params['symbol']} {timeframe} paper model "
        f"rows={len(frame):,} profile={profile_name}",
        flush=True,
    )
    model = train_barrier_model(frame, features, profile)
    model.save_model(str(XPT_MODEL_FILE))
    write_metadata(
        XPT_METADATA_FILE,
        {
            "symbol": params["symbol"],
            "timeframe": timeframe,
            "profile": profile_name,
            "profile_params": profile,
            "strategy_params": params,
            "features": features,
            "rows": len(frame),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "train_four_metal_paper_models.py",
            "selection_source": XPT_BEST_FILE.name,
        },
    )
    print(f"Wrote {XPT_MODEL_FILE}")
    print(f"Wrote {XPT_METADATA_FILE}")


def train_xpd() -> None:
    best = load_json(XPD_BEST_FILE)
    params = best["best_3x"]
    target_config = best["target_config"]
    timeframe = best["base_timeframe"]

    frame, features = load_target_case(timeframe, target_config)
    print(
        f"Training {params['symbol']} {timeframe} paper model "
        f"rows={len(frame):,} target={target_config['target_mode']}",
        flush=True,
    )
    model = train_alternate_target_model(frame, features)
    model.save_model(str(XPD_MODEL_FILE))
    write_metadata(
        XPD_METADATA_FILE,
        {
            "symbol": params["symbol"],
            "timeframe": timeframe,
            "target_config": target_config,
            "strategy_params": params,
            "features": features,
            "rows": len(frame),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "train_four_metal_paper_models.py",
            "selection_source": XPD_BEST_FILE.name,
        },
    )
    print(f"Wrote {XPD_MODEL_FILE}")
    print(f"Wrote {XPD_METADATA_FILE}")


def main() -> int:
    train_xpt()
    train_xpd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
