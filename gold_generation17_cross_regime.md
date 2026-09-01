# GOLD Generation 17 - Cross-Regime Generalization

Only long breakout, short breakout, and short trend-continuation are studied.
No confidence-threshold sweep and no new signal family were used.

## Discovery Pareto

| Candidate | Trades | Trades/day | Pooled win | Worst fold | PF | Mean-R | Max DD | Stress PF | Discovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gen17_short_trend_continuation | 206 | 0.114 | 60.19% | 48.94% | 0.89 | -0.0452 | -22.76% | 0.84 | FAIL |
| gen17_long_breakout | 373 | 0.206 | 52.01% | 51.00% | 0.70 | -0.1439 | -54.73% | 0.67 | FAIL |
| gen17_target_portfolio | 919 | 0.509 | 50.60% | 46.54% | 0.63 | -0.1845 | -91.57% | 0.60 | FAIL |
| gen17_breakout_pair | 721 | 0.399 | 47.85% | 46.48% | 0.57 | -0.2243 | -90.41% | 0.55 | FAIL |
| gen17_short_breakout | 359 | 0.199 | 44.29% | 0.00% | 0.48 | -0.2948 | -78.53% | 0.46 | FAIL |

Discovery-qualified: `0`
Pareto frontier: `[]`
Frozen status: `diagnostic_fallback`
Production promotion: `False`

The 2025 holdout and 2026 recent data are development/diagnostic only.
