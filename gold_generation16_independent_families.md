# GOLD generation 16 independent signal families

Family-specific long/short Stage-2 models; fixed 60% calibrated probability floor.

## Selected candidate

Status: `diagnostic_fallback`
Parameters: `{"candidate_id": "both_trend_continuation__none__top1", "label": "both_trend_continuation", "experts": ["long_trend_continuation", "short_trend_continuation"], "context_mode": "none", "top_k_per_expert_day": 1, "minimum_p_win": 0.6, "minimum_expected_r": 0.0}`

| Period | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | 93 | 0.120 | 46 | 47 | 0 | 49.46% | 0.57 | -253.73 | -0.2198 | -25.37% |
| 2021_2022 | 232 | 0.450 | 115 | 117 | 0 | 49.57% | 0.59 | -505.02 | -0.2114 | -50.86% |
| 2023_2024 | 23 | 0.045 | 16 | 7 | 0 | 69.57% | 1.32 | 31.44 | 0.1001 | -6.06% |
| 2025_2026_05_holdout | 228 | 0.655 | 110 | 118 | 0 | 48.25% | 0.59 | -502.74 | -0.2137 | -50.60% |
| 2026_recent | 1 | 0.013 | 1 | 0 | 0 | 100.00% | inf | 10.22 | 0.7297 | 0.00% |

Qualified candidates: `0`
Pareto frontier: `[]`
Research success: `False`
Promotion pass: `False`

This generation is research_only and does not modify gemini.py.
