# XM GOLD Research

Primary objectives:

1. Maximize out-of-sample win rate.
2. Maximize executable trade frequency.

Default OOS win-rate target: >= 60%.

Among candidates that satisfy the target,
prefer the candidate with the highest robust executable trade count.

PF > 1 and positive expectancy/PnL are hard guardrails.

Do not optimize classification accuracy as the primary objective.

All training, calibration, threshold selection,
filter discovery and validation must be chronological.

Never use future information.

Always compare against the parent baseline.

Count executable non-overlapping trades,
not raw qualifying signal rows.

Do not modify gemini.py or promote a model
unless the full validation gate passes.

Use repository-local skills when relevant:
- win-frequency-optimizer
- false-positive-miner
- missed-winner-miner
- walk-forward-validator
