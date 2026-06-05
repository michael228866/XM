from __future__ import annotations

import json
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parent
HIGH_PROFIT_DIR = RESEARCH_DIR.parent / "high_win_profit_research"

OUTPUT_JSON = RESEARCH_DIR / "precious_metals_composite_plan.json"
OUTPUT_MD = RESEARCH_DIR / "precious_metals_composite_plan.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def score_component(component: dict) -> float:
    status = component["status"]
    if status == "enabled_research":
        if component["symbol"] == "GOLD#":
            return (
                component.get("win_rate", 0.0) * 120.0
                + component.get("profit_factor", 0.0) * 35.0
                + min(component.get("trades", 0), 250) * 0.2
                - component.get("max_drawdown_pct", 0.0) * 100.0
            )
        total_r = component.get("stress_total_r")
        if total_r is None:
            total_r = component.get("total_r", 0.0)
        return (
            total_r * 2.0
            + component.get("win_rate", 0.0) * 80.0
            + component.get("profit_factor", 0.0) * 20.0
            - abs(component.get("max_drawdown_r", component.get("max_drawdown_pct", 0.0) * 100.0))
        )
    if status == "watchlist":
        return component.get("watch_score", 0.0)
    return -1000.0


def build_components() -> list[dict]:
    gold = safe_load_json(HIGH_PROFIT_DIR / "best_candidate.json")
    silver_strict = safe_load_json(RESEARCH_DIR / "silver_h1_stress_fold2_refine_best.json")
    xpt_summary = safe_load_json(RESEARCH_DIR / "long_tf_cost_walk_forward.json")
    status = safe_load_json(RESEARCH_DIR / "composite_training_status.json") or {}

    components: list[dict] = []
    if gold:
        test = gold["test_result"]
        components.append(
            {
                "symbol": "GOLD#",
                "role": "anchor",
                "status": "enabled_research",
                "strategy": "existing_gold_high_profit_candidate",
                "timeframe": "M1 core with multi-timeframe features",
                "params": gold["params"],
                "test_pnl_currency": test["pnl"],
                "win_rate": test["win_rate"],
                "profit_factor": test["profit_factor"],
                "trades": test["trades"],
                "max_drawdown_pct": abs(test["max_drawdown_pct"]),
                "stress_total_r": None,
                "notes": [
                    "Keep as anchor because it already passed the GOLD-specific research gate.",
                    "Needs R-normalized fold export before it can be fairly combined with non-GOLD R results.",
                ],
            }
        )

    if silver_strict:
        components.append(
            {
                "symbol": "SILVER#",
                "role": "core_satellite",
                "status": "enabled_research",
                "strategy": "strict_stress_h1_candidate",
                "timeframe": "H1",
                "params": {
                    "threshold": silver_strict["threshold"],
                    "edge_threshold": silver_strict["edge_threshold"],
                    "tp_atr": silver_strict["tp_atr"],
                    "sl_atr": silver_strict["sl_atr"],
                    "max_hold": silver_strict["max_hold"],
                    "direction_mode": silver_strict["direction_mode"],
                },
                "normal_total_r": silver_strict["base_total_pnl_r"],
                "double_spread_total_r": silver_strict["stress2_total_pnl_r"],
                "stress_total_r": silver_strict["stress3_total_pnl_r"],
                "win_rate": silver_strict["stress3_weighted_win_rate"],
                "profit_factor": silver_strict["stress3_mean_profit_factor"],
                "trades": silver_strict["stress3_total_trades"],
                "positive_folds": silver_strict["stress3_positive_folds"],
                "passed_folds": silver_strict["stress3_passed_folds"],
                "max_drawdown_r": abs(silver_strict["stress3_max_drawdown_r"]),
                "notes": [
                    "Best current non-GOLD component.",
                    "Survives 3x spread with 4/4 passed folds.",
                ],
            }
        )

    current_findings = status.get("current_findings", {})
    watchlist_specs = [
        ("XAUEUR#", "gold_cross_validation", "failed_cost_aware_walk_forward"),
        ("XPTUSD#", "platinum_watchlist", "failed_walk_forward_needs_regime_filter"),
        ("XPDUSD#", "palladium_watchlist", "sample_too_thin"),
        ("GAUCNH#", "supplemental_gold_cross", "history_too_short"),
    ]
    for symbol, role, reason in watchlist_specs:
        finding = current_findings.get(symbol, {})
        components.append(
            {
                "symbol": symbol,
                "role": role,
                "status": "watchlist",
                "strategy": "disabled_until_revalidated",
                "timeframe": "TBD",
                "weight": 0.0,
                "watch_score": 0.0,
                "reason": reason,
                "verdict": finding.get("verdict", "unknown"),
                "note": finding.get("note", ""),
            }
        )

    return components


def assign_weights(components: list[dict]) -> dict:
    enabled = [item for item in components if item["status"] == "enabled_research"]
    # Research-only portfolio weights. The live system should not use these until a
    # combined forward test passes.
    weights = {
        "conservative": {
            "GOLD#": 0.55,
            "SILVER#": 0.45,
        },
        "silver_tilt": {
            "GOLD#": 0.40,
            "SILVER#": 0.60,
        },
    }
    available_symbols = {item["symbol"] for item in enabled}
    for book in weights.values():
        for symbol in list(book):
            if symbol not in available_symbols:
                book[symbol] = 0.0
        total = sum(book.values())
        if total > 0:
            for symbol in list(book):
                book[symbol] = round(book[symbol] / total, 4)
    return weights


def build_plan() -> dict:
    components = build_components()
    return {
        "status": "research_only",
        "live_files_modified": False,
        "objective": "Precious-metals composite trading plan with GOLD as anchor and robust non-GOLD metals added only after walk-forward/stress validation.",
        "selection_rules": {
            "core_enable": [
                "4/4 positive walk-forward folds",
                "at least 3/4 passed folds under stressed spread",
                "profit factor above 1.25 under stressed spread",
                "enough trades to avoid thin-sample promotion",
            ],
            "watchlist_to_enabled": [
                "cost-aware walk-forward positive on 4/4 folds",
                "no single fold dominates total profit",
                "stress spread test remains positive",
            ],
        },
        "risk_framework": {
            "portfolio_max_new_positions": 2,
            "max_enabled_symbols_initial": 2,
            "same_symbol_max_positions": 1,
            "satellite_symbols_start_at_weight": 0.0,
            "dynamic_sizing": "research-only until paper tested; do not apply to gemini.py",
        },
        "components": sorted(
            components,
            key=lambda item: (item["status"] != "enabled_research", -score_component(item)),
        ),
        "research_weights": assign_weights(components),
        "current_decision": {
            "tradable_research_composite": ["GOLD#", "SILVER#"],
            "disabled_until_revalidated": ["XAUEUR#", "XPTUSD#", "XPDUSD#", "GAUCNH#"],
            "preferred_next_test": "Build a paper portfolio runner for GOLD# + SILVER# first, then add XPTUSD# only after regime-filtered walk-forward passes.",
        },
    }


def write_markdown(plan: dict) -> None:
    lines = [
        "# Precious Metals Composite Plan",
        "",
        "Research-only plan. It does not modify `gemini.py`.",
        "",
        "## Current Composite",
        "",
        "| Symbol | Role | Status | Strategy | Key Result | Weight Notes |",
        "|---|---|---|---|---|---|",
    ]
    for item in plan["components"]:
        if item["status"] == "enabled_research":
            if item["symbol"] == "GOLD#":
                key_result = (
                    f"PnL {item['test_pnl_currency']:.2f}, "
                    f"win {item['win_rate']:.2%}, PF {item['profit_factor']:.2f}"
                )
            else:
                key_result = (
                    f"3x spread {item['stress_total_r']:.2f}R, "
                    f"win {item['win_rate']:.2%}, PF {item['profit_factor']:.2f}"
                )
            weight_notes = "enabled in research weights"
        else:
            key_result = item["reason"]
            weight_notes = "0% until revalidated"
        lines.append(
            "| {symbol} | {role} | {status} | {strategy} | {key_result} | {weight_notes} |".format(
                **item,
                key_result=key_result,
                weight_notes=weight_notes,
            )
        )

    lines.extend(
        [
            "",
            "## Research Weights",
            "",
            "| Book | GOLD# | SILVER# | Notes |",
            "|---|---:|---:|---|",
        ]
    )
    weights = plan["research_weights"]
    lines.append(
        "| conservative | {GOLD:.0%} | {SILVER:.0%} | Use until combined paper test is stable |".format(
            GOLD=weights["conservative"].get("GOLD#", 0.0),
            SILVER=weights["conservative"].get("SILVER#", 0.0),
        )
    )
    lines.append(
        "| silver_tilt | {GOLD:.0%} | {SILVER:.0%} | Higher non-GOLD exposure, research only |".format(
            GOLD=weights["silver_tilt"].get("GOLD#", 0.0),
            SILVER=weights["silver_tilt"].get("SILVER#", 0.0),
        )
    )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Current composite candidate is GOLD# + SILVER# only.",
            "- XAUEUR#, XPTUSD#, XPDUSD#, and GAUCNH# stay disabled until they pass cost-aware walk-forward and stress checks.",
            "- Next step is a combined paper portfolio runner, not live deployment.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    plan = build_plan()
    OUTPUT_JSON.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_markdown(plan)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
