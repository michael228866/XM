---
name: missed-winner-miner
description: Mine GOLD# non-traded mature events that would have been TP-first winners and test new signal families for incremental OOS frequency at the target win rate. Use for opportunity discovery and expert-family expansion, not for loosening an existing threshold without validation.
---

# Missed-Winner Miner

Turn stable clusters of non-traded TP-first events into independent signal families, then test their incremental portfolio value.

## Repository contract

- Read the active candidate, its saved signals/report, `gold_generation12_executable_events.py`, and the first-touch/execution helpers in `gold_generation11_execution_aligned.py`.
- Use mature TP-first labels built from intrabar `HIGH`/`LOW` with the same horizon, TP, SL, and costs as the baseline. Future outcomes are labels for offline research only and must never become live features.
- Define existing signals and missed opportunities on the same timestamps and eligibility rules. Preserve direction and the long/short trend/pullback route.

## Discovery

1. Find eligible mature events not selected by the existing strategy whose net first-touch reward is positive.
2. Remove overlap with existing signals and report both timestamp overlap and execution overlap after the single-position constraint.
3. Cluster only with entry-time information. Candidate families may include trend continuation, pullback, breakout, mean reversion, volatility expansion, or session-specific behavior.
4. Prefer an interpretable family rule first. Build a separate expert only when the cluster remains stable across chronological discovery windows and cannot be represented cleanly by an existing expert.

Do not force every missed winner into one model, lower the global threshold to capture them, or select a family using the final holdout/recent interval.

## Validation

Evaluate each new family alone and unioned with the incumbent. Deduplicate chronologically, enforce actual position occupancy, and use first-touch `HIGH`/`LOW` execution.

Accept a family for further research only when it adds unique executable trades while the combined strategy maintains the user's OOS win-rate target, default 60%, on every selection fold. PF greater than 1, positive expectancy/PnL, cost robustness, and drawdown are hard guardrails.

## Required report

Report for each family:

- discovered missed winners and the discovery interval;
- OOS executable trades, trades/day, win rate, TP/SL/timeouts, PF, PnL/mean R, and max drawdown;
- raw overlap, execution overlap, and unique trades added versus the incumbent;
- combined-strategy metrics for every selection fold, holdout, recent interval, and cost stress;
- rejected families and the exact failed constraint.

Keep all families `research_only`; do not replace `gemini.py` until the complete promotion gate passes.
