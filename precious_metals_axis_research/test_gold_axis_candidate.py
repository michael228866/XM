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

from barrier_final_train import FINAL_PARAMS  # noqa: E402
from barrier_meta_overlay import (  # noqa: E402
    evaluate_df,
    load_final_model,
    load_meta_overlay_model,
    predict_overlay_risk_mult,
    split_overlay_data,
    trades_per_year,
)


OUTPUT_JSON = RESEARCH_DIR / "gold_axis_smoke_test.json"
OUTPUT_MD = RESEARCH_DIR / "gold_axis_smoke_test.md"

GOLD_AXIS_CANDIDATE = {
    "threshold": 0.525,
    "edge_threshold": 0.0,
    "tp_atr": 1.3,
    "sl_atr": 2.0,
    "max_hold": 180,
    "risk_per_trade": 0.028,
    "max_daily_loss_pct": 0.05,
}

ACCEPTANCE_GATE = {
    "min_test_win_rate": 0.70,
    "min_test_profit_factor": 1.75,
    "min_test_pnl_improvement": 800.0,
    "max_test_drawdown_pct": 0.36,
    "min_validation_pnl_improvement": 100.0,
}


def compact_stats(stats, df):
    return {
        "pnl": round(float(stats["pnl"]), 2),
        "trades": int(stats["trades"]),
        "trades_per_year": round(float(trades_per_year(stats, df)), 2),
        "win_rate": round(float(stats["win_rate"]), 4),
        "profit_factor": round(float(stats["profit_factor"]), 4),
        "max_drawdown_pct": round(float(stats["max_drawdown_pct"]), 4),
        "max_consecutive_losses": int(stats["max_consecutive_losses"]),
        "stopped_out": bool(stats["stopped_out"]),
    }


def passes_gate(validation_baseline, validation_candidate, test_baseline, test_candidate):
    test_drawdown = abs(min(float(test_candidate["max_drawdown_pct"]), 0.0))
    return (
        not validation_candidate["stopped_out"]
        and not test_candidate["stopped_out"]
        and validation_candidate["pnl"]
        >= validation_baseline["pnl"] + ACCEPTANCE_GATE["min_validation_pnl_improvement"]
        and test_candidate["pnl"]
        >= test_baseline["pnl"] + ACCEPTANCE_GATE["min_test_pnl_improvement"]
        and test_candidate["win_rate"] >= ACCEPTANCE_GATE["min_test_win_rate"]
        and test_candidate["profit_factor"] >= ACCEPTANCE_GATE["min_test_profit_factor"]
        and test_drawdown <= ACCEPTANCE_GATE["max_test_drawdown_pct"]
    )


def write_report(result):
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    baseline = result["test"]["baseline"]
    candidate = result["test"]["candidate"]
    validation_baseline = result["validation"]["baseline"]
    validation_candidate = result["validation"]["candidate"]
    lines = [
        "# GOLD Axis Smoke Test",
        "",
        "This is a research-only check. It does not modify `gemini.py`.",
        "",
        f"Pass: `{result['passes_gate']}`",
        "",
        "## Test Window",
        "",
        "| Version | PnL | Win | PF | Trades | DD | Max Loss Streak |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Current meta overlay | {baseline['pnl']:.2f} | "
            f"{baseline['win_rate']:.2%} | {baseline['profit_factor']:.2f} | "
            f"{baseline['trades']} | {baseline['max_drawdown_pct']:.2%} | "
            f"{baseline['max_consecutive_losses']} |"
        ),
        (
            f"| GOLD axis candidate | {candidate['pnl']:.2f} | "
            f"{candidate['win_rate']:.2%} | {candidate['profit_factor']:.2f} | "
            f"{candidate['trades']} | {candidate['max_drawdown_pct']:.2%} | "
            f"{candidate['max_consecutive_losses']} |"
        ),
        "",
        "## Validation Window",
        "",
        "| Version | PnL | Win | PF | Trades | DD | Max Loss Streak |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Current meta overlay | {validation_baseline['pnl']:.2f} | "
            f"{validation_baseline['win_rate']:.2%} | "
            f"{validation_baseline['profit_factor']:.2f} | "
            f"{validation_baseline['trades']} | "
            f"{validation_baseline['max_drawdown_pct']:.2%} | "
            f"{validation_baseline['max_consecutive_losses']} |"
        ),
        (
            f"| GOLD axis candidate | {validation_candidate['pnl']:.2f} | "
            f"{validation_candidate['win_rate']:.2%} | "
            f"{validation_candidate['profit_factor']:.2f} | "
            f"{validation_candidate['trades']} | "
            f"{validation_candidate['max_drawdown_pct']:.2%} | "
            f"{validation_candidate['max_consecutive_losses']} |"
        ),
        "",
        "## Candidate Params",
        "",
        "```json",
        json.dumps(result["candidate_params"], indent=2),
        "```",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("Loading GOLD data and models...")
    _, features, _, _, validation_df, test_df = split_overlay_data()
    model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    regime_features = config["regime_features"]
    rule = tuple(float(value) for value in config["risk_rule"])

    validation_probs = model.predict_proba(validation_df[features]).astype("float32")
    test_probs = model.predict_proba(test_df[features]).astype("float32")
    validation_mult, _ = predict_overlay_risk_mult(
        meta_model, validation_df, validation_probs, regime_features, rule
    )
    test_mult, _ = predict_overlay_risk_mult(
        meta_model, test_df, test_probs, regime_features, rule
    )

    baseline_params = dict(FINAL_PARAMS)
    baseline_params["risk_per_trade"] = float(config["risk_per_trade"])
    candidate_params = dict(FINAL_PARAMS)
    candidate_params.update(GOLD_AXIS_CANDIDATE)

    validation_baseline_stats = evaluate_df(
        baseline_params, validation_df, validation_probs, validation_mult
    )
    validation_candidate_stats = evaluate_df(
        candidate_params, validation_df, validation_probs, validation_mult
    )
    test_baseline_stats = evaluate_df(
        baseline_params, test_df, test_probs, test_mult
    )
    test_candidate_stats = evaluate_df(
        candidate_params, test_df, test_probs, test_mult
    )

    result = {
        "symbol": "GOLD#",
        "status": "research_only",
        "acceptance_gate": ACCEPTANCE_GATE,
        "candidate_params": GOLD_AXIS_CANDIDATE,
        "validation": {
            "baseline": compact_stats(validation_baseline_stats, validation_df),
            "candidate": compact_stats(validation_candidate_stats, validation_df),
        },
        "test": {
            "baseline": compact_stats(test_baseline_stats, test_df),
            "candidate": compact_stats(test_candidate_stats, test_df),
        },
    }
    result["passes_gate"] = passes_gate(
        result["validation"]["baseline"],
        result["validation"]["candidate"],
        result["test"]["baseline"],
        result["test"]["candidate"],
    )
    write_report(result)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        "GOLD axis smoke test "
        f"{'PASSED' if result['passes_gate'] else 'FAILED'} | "
        f"test pnl={result['test']['candidate']['pnl']:.2f}, "
        f"win={result['test']['candidate']['win_rate']:.2%}, "
        f"pf={result['test']['candidate']['profit_factor']:.2f}"
    )
    return 0 if result["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
