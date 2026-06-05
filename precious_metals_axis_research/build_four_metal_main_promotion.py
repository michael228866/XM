from __future__ import annotations

import json
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent

OUTPUT_JSON = RESEARCH_DIR / "four_metal_main_promotion_plan.json"
OUTPUT_MD = RESEARCH_DIR / "four_metal_main_promotion_plan.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_profit_pass(fold: dict, min_trades: int) -> bool:
    return (
        fold["pnl_r"] > 0
        and fold["profit_factor"] >= 1.10
        and fold["trades"] >= min_trades
    )


def cost_passes(cost_rows: list[dict], through: float) -> bool:
    rows = [row for row in cost_rows if float(row["cost_multiplier"]) <= through]
    return bool(rows) and all(row["gate"] for row in rows)


def slow_main_gate(best: dict, cost_rows: list[dict], min_fold_trades: int) -> dict:
    folds = best["folds"]
    fold_passes = [fold_profit_pass(fold, min_fold_trades) for fold in folds]
    gate = (
        all(fold["pnl_r"] > 0 for fold in folds)
        and all(fold_passes)
        and best["total_r"] >= 8.0
        and best["max_drawdown_r"] >= -8.0
        and cost_passes(cost_rows, 3.0)
    )
    return {
        "gate": gate,
        "fold_profit_passes": f"{sum(fold_passes)}/{len(fold_passes)}",
        "min_fold_trades_required": min_fold_trades,
        "min_fold_trades_observed": min(fold["trades"] for fold in folds),
        "cost_passes_through_3x": cost_passes(cost_rows, 3.0),
    }


def summarize_cost(cost_rows: list[dict], cost_multiplier: float) -> dict:
    for row in cost_rows:
        if float(row["cost_multiplier"]) == cost_multiplier:
            return {
                "total_r": row["total_r"],
                "trades": row["trades"],
                "win_rate": row["weighted_win_rate"],
                "profit_factor": row["mean_profit_factor"],
                "max_drawdown_r": row["max_drawdown_r"],
                "positive_folds": row["positive_folds"],
                "passed_folds": row["passed_folds"],
                "gate": row["gate"],
            }
    raise ValueError(f"Missing {cost_multiplier}x cost row.")


def main() -> int:
    gold = load_json(PROJECT_ROOT / "high_win_profit_research" / "best_candidate.json")
    stable = load_json(RESEARCH_DIR / "training_profile_stable_selection.json")
    xpt = load_json(RESEARCH_DIR / "xpt_h4_fold_coverage_best.json")
    xpd = load_json(RESEARCH_DIR / "xpd_alternate_target_exit_best.json")

    silver_3x = next(
        row for row in stable["SILVER#"]["cost_stress"]
        if float(row["cost_multiplier"]) == 3.0
    )
    xpt_gate = slow_main_gate(xpt["best_3x"], xpt["cost_stress"], min_fold_trades=2)
    xpd_gate = slow_main_gate(xpd["best_3x"], xpd["cost_stress"], min_fold_trades=2)

    plan = {
        "status": "research_main_promotion",
        "live_files_modified": False,
        "promotion_gate": {
            "core_rule": "4/4 positive folds, profit-factor fold pass, 3x cost pass, and controlled drawdown.",
            "slow_main_min_fold_trades": {
                "XPTUSD#_H4": 2,
                "XPDUSD#_H12": 2,
            },
            "note": "Slow-main symbols can trade less frequently than GOLD# or SILVER#, but each fold must still be profitable under 3x cost stress.",
        },
        "symbols": {
            "GOLD#": {
                "role": "main_anchor",
                "source": "high_win_profit_research/best_candidate.json",
                "status": "main_research_line",
                "allocation_weight": 0.35,
                "model": "gold_barrier_final_xgb.json",
                "timeframe": "M1",
                "direction_mode": "long",
                "test_pnl": gold["test_result"]["pnl"],
                "test_trades": gold["test_result"]["trades"],
                "test_win_rate": gold["test_result"]["win_rate"],
                "test_profit_factor": gold["test_result"]["profit_factor"],
                "test_drawdown_pct": gold["test_result"]["max_drawdown_pct"],
            },
            "SILVER#": {
                "role": "main_core",
                "source": "training_profile_stable_selection.json",
                "status": "main_research_line",
                "allocation_weight": 0.25,
                "model": "silver_h1_regime_selected_xgb.json",
                "timeframe": "H1",
                "direction_mode": stable["SILVER#"]["best_3x"]["direction_mode"],
                "cost_3x": summarize_cost(stable["SILVER#"]["cost_stress"], 3.0),
                "cost_gate_scope": "passes 1x-3x; fails 4x-5x",
            },
            "XPTUSD#": {
                "role": "slow_main",
                "source": "xpt_h4_fold_coverage_best.json",
                "status": "main_research_line",
                "allocation_weight": 0.20,
                "timeframe": "H4",
                "direction_mode": xpt["best_3x"]["direction_mode"],
                "target_exit": "original barrier target with refined H4 exit",
                "cost_3x": summarize_cost(xpt["cost_stress"], 3.0),
                "cost_gate_scope": "passes 1x-5x",
                "promotion_gate": xpt_gate,
            },
            "XPDUSD#": {
                "role": "slow_main",
                "source": "xpd_alternate_target_exit_best.json",
                "status": "main_research_line",
                "allocation_weight": 0.20,
                "timeframe": "H12",
                "direction_mode": xpd["best_3x"]["direction_mode"],
                "target_exit": "future_close_h8 target with time_stop exit",
                "cost_3x": summarize_cost(xpd["cost_stress"], 3.0),
                "cost_gate_scope": "passes 1x-3x; fails 4x-5x",
                "promotion_gate": xpd_gate,
            },
        },
    }
    plan["all_four_research_main"] = all(
        item["status"] == "main_research_line"
        for item in plan["symbols"].values()
    )
    plan["all_four_gate_note"] = (
        "All four are promoted to research-main lines. This does not place live orders "
        "or change gemini.py; forward paper logging is still required before live deployment."
    )

    OUTPUT_JSON.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    lines = [
        "# Four-Metal Main Promotion Plan",
        "",
        "Research-only promotion plan. This does not modify `gemini.py` or place live orders.",
        "",
        "| Symbol | Role | Weight | TF | 3x R / PnL | Trades | Win | PF | DD | Gate Scope |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for symbol, item in plan["symbols"].items():
        if symbol == "GOLD#":
            lines.append(
                "| {symbol} | {role} | {allocation_weight:.2%} | {timeframe} | "
                "{test_pnl:.2f} | {test_trades} | {test_win_rate:.2%} | "
                "{test_profit_factor:.2f} | {test_drawdown_pct:.2%} | "
                "legacy GOLD gate |".format(symbol=symbol, **item)
            )
            continue
        cost = item["cost_3x"]
        lines.append(
            "| {symbol} | {role} | {allocation_weight:.2%} | {timeframe} | "
            "{total_r:.2f}R | {trades} | {win_rate:.2%} | {profit_factor:.2f} | "
            "{max_drawdown_r:.2f}R | {cost_gate_scope} |".format(
                symbol=symbol,
                total_r=cost["total_r"],
                trades=cost["trades"],
                win_rate=cost["win_rate"],
                profit_factor=cost["profit_factor"],
                max_drawdown_r=cost["max_drawdown_r"],
                **item,
            )
        )

    lines.extend(
        [
            "",
            "## Promotion Notes",
            "",
            "- `GOLD#` and `SILVER#` are the faster/core main lines.",
            "- `XPTUSD#` and `XPDUSD#` are slow-main lines, so the minimum fold-trade gate is lower by timeframe.",
            "- Forward paper logging is still required before live deployment.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print("All four symbols were promoted to research-main lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
