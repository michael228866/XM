---
name: win-frequency-optimizer
description: Optimize GOLD# research candidates for maximum out-of-sample executable trade frequency under a win-rate target. Use for threshold, regime, expert, or model selection research; do not use for MT5 runner recovery or live-order operations.
---

# Win-Frequency Optimizer

Optimize the executable strategy, not classifier accuracy.

## Repository contract

- Read `README.md` and the candidate's code and report before changing research logic.
- Reuse Gen12 conventions from `gold_generation12_executable_events.py`: separate long/short trend/pullback experts, non-overlapping executable events, natural P(win) plus Mean-R, past-only calibration, and rolling selection.
- Reuse `add_targets` and `execution_realized_metrics` from `gold_generation11_execution_aligned.py`, `fold_pass` and execution parameters from `gold_expected_r_walk_forward.py`, and the selection folds/current benchmark from `gold_regime_experts_walk_forward.py`.
- Keep labels and evaluation aligned to TP/SL first touch using intrabar `HIGH`/`LOW`. The current comparable profile is horizon 90, TP 1.3 ATR, and SL 1.6 ATR; do not silently change it.

## Objective

Use the user's win-rate target, defaulting to 60%.

1. Filter to candidates that meet the target on every out-of-sample selection fold.
2. Require positive expectancy/PnL, PF greater than 1, an explicit drawdown limit, and nonzero trades.
3. Among feasible candidates, maximize executable trade count. Use minimum-fold frequency before total frequency when robustness differs.
4. Treat classification accuracy, raw bar count, and in-sample scores as diagnostics only.

Construct a Win Rate x Executable Trade Frequency Pareto frontier. Candidate A is dominated when another candidate has at least its OOS win rate and executable trades, with one strict improvement, while satisfying the same guardrails.

## Workflow

1. Establish the incumbent and nearest research baseline from their saved reports. State the exact data interval and execution costs.
2. Generate candidates without lookahead. Long and short may use different experts and past-only thresholds; thresholds may vary by session or volatility regime only when each group has adequate mature executable events.
3. Evaluate chronologically on the repository selection folds, then the 2025+ holdout, the 2026 recent interval, and the extra-cost stress case. Never select a threshold on the interval used to claim its result.
4. Deduplicate signals and simulate the actual single-position execution path before counting trades.
5. Keep a candidate `research_only` unless all historical folds, holdout, recent comparison, and cost guard pass. Do not edit or replace `gemini.py` merely because recent data passes.

## Required report

For every fold and benchmark report:

- executable trades and trades/day, including the denominator used;
- win rate, TP wins, SL losses, and timeouts;
- PF, PnL or mean R, and maximum drawdown;
- long/short and expert/session/regime contributions when relevant;
- Pareto membership or the exact constraint that rejected the candidate.

Lead with the maximum feasible frequency at the target win rate. If no candidate is feasible, say so and retain the incumbent; do not hide the failure behind a composite score.
