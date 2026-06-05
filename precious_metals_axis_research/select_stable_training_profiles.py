from __future__ import annotations

import json
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parent
SOURCE_JSON = RESEARCH_DIR / "training_profile_optimization_results.json"
OUTPUT_JSON = RESEARCH_DIR / "training_profile_stable_selection.json"
OUTPUT_MD = RESEARCH_DIR / "training_profile_stable_selection.md"


def pick_stable(rows: list[dict], symbol: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["group"] != f"{symbol.lower().replace('#', '')}_selected_cost":
            continue
        result = row["result"]
        if result["symbol"] != symbol:
            continue
        grouped.setdefault(row["profile"], []).append(result)

    candidates = []
    for profile, cost_rows in grouped.items():
        by_cost = {float(row["cost_multiplier"]): row for row in cost_rows}
        row_3x = by_cost[3.0]
        candidates.append(
            {
                "profile": profile,
                "cost_stress": [by_cost[cost] for cost in sorted(by_cost)],
                "best_3x": row_3x,
                "cost_gate_count": sum(row["gate"] for row in cost_rows),
                "cost_total_r": round(sum(row["total_r"] for row in cost_rows), 4),
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["cost_gate_count"],
            item["best_3x"]["gate"],
            item["best_3x"]["total_r"],
            item["cost_total_r"],
        ),
        reverse=True,
    )[0]


def write_report(best: dict) -> None:
    OUTPUT_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    lines = [
        "# Stable Training Profile Selection",
        "",
        "Selects profiles by cost-stress stability first, then 3x R.",
        "",
        "## Selected",
        "",
        "| Symbol | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for symbol in ["SILVER#", "XAUEUR#"]:
        item = best[symbol]
        row = item["best_3x"]
        lines.append(
            "| {symbol} | {profile} | {cost_gate_count}/5 | {gate} | {total_r:.2f} | "
            "{trades} | {weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(
                profile=item["profile"],
                cost_gate_count=item["cost_gate_count"],
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Cost Stress",
            "",
            "| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |",
            "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in ["SILVER#", "XAUEUR#"]:
        item = best[symbol]
        fold_count = 5 if symbol == "SILVER#" else 3
        for row in item["cost_stress"]:
            lines.append(
                "| {symbol} | {profile} | {cost_multiplier:.1f}x | {gate} | "
                "{total_r:.2f} | {positive_folds}/{fold_count} | "
                "{passed_folds}/{fold_count} | {trades} | {weighted_win_rate:.2%} | "
                "{worst_fold_r:.2f} | {max_drawdown_r:.2f} |".format(
                    profile=item["profile"],
                    fold_count=fold_count,
                    **row,
                )
            )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    rows = data["rows"]
    best = {
        "SILVER#": pick_stable(rows, "SILVER#"),
        "XAUEUR#": pick_stable(rows, "XAUEUR#"),
    }
    write_report(best)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    for symbol, item in best.items():
        row = item["best_3x"]
        print(
            f"{symbol}: profile={item['profile']} cost_gates={item['cost_gate_count']}/5 "
            f"r3x={row['total_r']:.2f} gate3x={row['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
