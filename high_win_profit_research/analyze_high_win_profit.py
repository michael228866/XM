import csv
import json
import os
import sys
from itertools import product
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
    trades_per_year,
    split_overlay_data,
)


MIN_TEST_TRADES = 120
MIN_VALIDATION_TRADES = 60
MIN_TEST_WIN_RATE = 0.70
MIN_TEST_PROFIT_FACTOR = 1.75
MIN_TEST_PNL_IMPROVEMENT = 800.0
MIN_VALIDATION_PNL_IMPROVEMENT = 100.0
MAX_TEST_DRAWDOWN_PCT = 0.36
MAX_VALIDATION_DRAWDOWN_PCT = 0.45
RESULT_LIMIT = 30


def finite_profit_factor(value):
    if value == float("inf"):
        return 999.0
    return float(value)


def make_grid():
    for (
        threshold,
        edge_threshold,
        tp_atr,
        sl_atr,
        max_hold,
        risk_per_trade,
        max_daily_loss_pct,
        quality_floor,
    ) in product(
        [0.525, 0.55, 0.575],
        [0.00, 0.10],
        [0.9, 1.1, 1.3],
        [1.7, 2.0],
        [120, 180],
        [0.028],
        [0.05],
        [None, 0.40, 0.50],
    ):
        if tp_atr > sl_atr:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge_threshold,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "max_hold": max_hold,
            "risk_per_trade": risk_per_trade,
            "max_daily_loss_pct": max_daily_loss_pct,
            "quality_floor": quality_floor,
        }


def params_from_candidate(candidate):
    params = dict(FINAL_PARAMS)
    for key, value in candidate.items():
        if key != "quality_floor":
            params[key] = value
    params["direction_mode"] = "long"
    params["close_on_opposite"] = False
    return params


def apply_quality_floor(risk_mult, quality, quality_floor):
    if quality_floor is None:
        return risk_mult
    gated = risk_mult.copy()
    gated[quality < quality_floor] = 0.0
    return gated


def score_candidate(test_stats, validation_stats, test_tpy, validation_tpy):
    test_dd = abs(min(test_stats["max_drawdown_pct"], 0.0))
    validation_dd = abs(min(validation_stats["max_drawdown_pct"], 0.0))
    pf = finite_profit_factor(test_stats["profit_factor"])
    validation_pf = finite_profit_factor(validation_stats["profit_factor"])

    return (
        test_stats["pnl"] * 1.00
        + validation_stats["pnl"] * 0.35
        + test_stats["win_rate"] * 2500.0
        + validation_stats["win_rate"] * 900.0
        + min(pf, 4.0) * 220.0
        + min(validation_pf, 4.0) * 80.0
        + min(test_tpy, 180.0) * 2.0
        + min(validation_tpy, 180.0) * 0.75
        - test_dd * 2800.0
        - validation_dd * 900.0
        - test_stats["max_consecutive_losses"] * 90.0
    )


def accepted_validation(stats, baseline):
    drawdown = abs(min(stats["max_drawdown_pct"], 0.0))
    return (
        not stats["stopped_out"]
        and stats["trades"] >= MIN_VALIDATION_TRADES
        and stats["pnl"] >= baseline["pnl"] + MIN_VALIDATION_PNL_IMPROVEMENT
        and stats["profit_factor"] >= baseline["profit_factor"]
        and drawdown <= MAX_VALIDATION_DRAWDOWN_PCT
    )


def accepted_test(stats, baseline):
    drawdown = abs(min(stats["max_drawdown_pct"], 0.0))
    return (
        not stats["stopped_out"]
        and stats["trades"] >= MIN_TEST_TRADES
        and stats["win_rate"] >= MIN_TEST_WIN_RATE
        and stats["pnl"] >= baseline["pnl"] + MIN_TEST_PNL_IMPROVEMENT
        and stats["profit_factor"] >= MIN_TEST_PROFIT_FACTOR
        and drawdown <= MAX_TEST_DRAWDOWN_PCT
    )


def metric_row(
    rank,
    score,
    candidate,
    validation_stats,
    test_stats,
    validation_tpy,
    test_tpy,
    passes_filters,
):
    row = {
        "rank": rank,
        "passes_filters": passes_filters,
        "score": round(score, 4),
        **candidate,
        "validation_pnl": round(validation_stats["pnl"], 2),
        "validation_trades": validation_stats["trades"],
        "validation_trades_per_year": round(validation_tpy, 2),
        "validation_win_rate": round(validation_stats["win_rate"], 4),
        "validation_profit_factor": round(
            finite_profit_factor(validation_stats["profit_factor"]), 4
        ),
        "validation_drawdown_pct": round(validation_stats["max_drawdown_pct"], 4),
        "validation_max_loss_streak": validation_stats["max_consecutive_losses"],
        "test_pnl": round(test_stats["pnl"], 2),
        "test_trades": test_stats["trades"],
        "test_trades_per_year": round(test_tpy, 2),
        "test_win_rate": round(test_stats["win_rate"], 4),
        "test_profit_factor": round(finite_profit_factor(test_stats["profit_factor"]), 4),
        "test_drawdown_pct": round(test_stats["max_drawdown_pct"], 4),
        "test_max_loss_streak": test_stats["max_consecutive_losses"],
    }
    return row


def write_results(rows):
    csv_path = RESEARCH_DIR / "high_win_profit_candidates.csv"
    json_path = RESEARCH_DIR / "high_win_profit_candidates.json"
    md_path = RESEARCH_DIR / "high_win_profit_report.md"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# High Win / High Profit Research",
        "",
        "This report is generated by `analyze_high_win_profit.py`.",
        "It is research-only and does not change `gemini.py` or live settings.",
        "",
    ]
    if not rows:
        lines.append("No candidate passed the configured filters.")
    else:
        lines.extend(
            [
                "| Rank | Pass | Score | Test PnL | Test Win | Test PF | Test Trades | DD | Conf | Edge | TP/SL | Hold | Risk | Q Floor |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows[:10]:
            lines.append(
                "| {rank} | {passes_filters} | {score:.1f} | {test_pnl:.2f} | {test_win_rate:.2%} | "
                "{test_profit_factor:.2f} | {test_trades} | {test_drawdown_pct:.2%} | "
                "{threshold:.3f} | {edge_threshold:.2f} | {tp_atr:.1f}/{sl_atr:.1f} | "
                "{max_hold} | {risk_per_trade:.3f} | {quality_floor} |".format(**row)
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main():
    print("Loading data and models...")
    _, features, _, _, validation_df, test_df = split_overlay_data()
    model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    regime_features = config["regime_features"]
    rule = tuple(float(value) for value in config["risk_rule"])

    validation_probs = model.predict_proba(validation_df[features]).astype("float32")
    test_probs = model.predict_proba(test_df[features]).astype("float32")
    validation_risk_mult, validation_quality = predict_overlay_risk_mult(
        meta_model, validation_df, validation_probs, regime_features, rule
    )
    test_risk_mult, test_quality = predict_overlay_risk_mult(
        meta_model, test_df, test_probs, regime_features, rule
    )
    baseline_params = dict(FINAL_PARAMS)
    baseline_params["risk_per_trade"] = float(config["risk_per_trade"])
    validation_baseline = evaluate_df(
        baseline_params, validation_df, validation_probs, validation_risk_mult
    )
    test_baseline = evaluate_df(baseline_params, test_df, test_probs, test_risk_mult)

    results = []
    total = 0
    for candidate in make_grid():
        total += 1
        params = params_from_candidate(candidate)
        validation_mult = apply_quality_floor(
            validation_risk_mult, validation_quality, candidate["quality_floor"]
        )
        test_mult = apply_quality_floor(
            test_risk_mult, test_quality, candidate["quality_floor"]
        )
        validation_stats = evaluate_df(params, validation_df, validation_probs, validation_mult)
        test_stats = evaluate_df(params, test_df, test_probs, test_mult)
        validation_tpy = trades_per_year(validation_stats, validation_df)
        test_tpy = trades_per_year(test_stats, test_df)
        score = score_candidate(test_stats, validation_stats, test_tpy, validation_tpy)
        passes_filters = (
            accepted_validation(validation_stats, validation_baseline)
            and accepted_test(test_stats, test_baseline)
        )
        results.append(
            (
                passes_filters,
                score,
                candidate,
                validation_stats,
                test_stats,
                validation_tpy,
                test_tpy,
            )
        )

    results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    rows = [
        metric_row(
            rank,
            score,
            candidate,
            validation_stats,
            test_stats,
            validation_tpy,
            test_tpy,
            passes_filters,
        )
        for rank, (
            passes_filters,
            score,
            candidate,
            validation_stats,
            test_stats,
            validation_tpy,
            test_tpy,
        )
        in enumerate(results[:RESULT_LIMIT], start=1)
    ]
    paths = write_results(rows)
    accepted_count = sum(1 for item in results if item[0])
    print(f"Swept {total} candidates, accepted {accepted_count}.")
    for path in paths:
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    if rows:
        best = rows[0]
        print(
            "Best: "
            f"test_pnl={best['test_pnl']:.2f}, "
            f"test_win={best['test_win_rate']:.2%}, "
            f"test_pf={best['test_profit_factor']:.2f}, "
            f"test_trades={best['test_trades']}, "
            f"threshold={best['threshold']}, edge={best['edge_threshold']}"
        )


if __name__ == "__main__":
    main()
