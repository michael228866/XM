# GOLD generation 15 chronological signal mining

Gen12 parent plus OOF loser-cluster filtering and missed-winner families.
All results use non-overlapping first-touch events and fixed 1.4% fractional R execution.

## Parent baseline versus diagnostic fallback

| Period | Strategy | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | parent | 127 | 0.164 | 70 | 56 | 1 | 55.12% | 0.78 | -166.97 | -0.0979 | -25.05% |
| 2018_2020 | candidate | 127 | 0.164 | 70 | 56 | 1 | 55.12% | 0.78 | -166.97 | -0.0979 | -25.05% |
| 2021_2022 | parent | 98 | 0.190 | 53 | 44 | 1 | 54.08% | 0.71 | -175.94 | -0.1363 | -19.69% |
| 2021_2022 | candidate | 98 | 0.190 | 53 | 44 | 1 | 54.08% | 0.71 | -175.94 | -0.1363 | -19.69% |
| 2023_2024 | parent | 40 | 0.078 | 21 | 19 | 0 | 52.50% | 0.69 | -83.15 | -0.1501 | -11.64% |
| 2023_2024 | candidate | 40 | 0.078 | 21 | 19 | 0 | 52.50% | 0.69 | -83.15 | -0.1501 | -11.64% |
| 2025_2026_05_holdout | parent | 106 | 0.305 | 59 | 45 | 2 | 55.66% | 0.89 | -71.74 | -0.0453 | -10.86% |
| 2025_2026_05_holdout | candidate | 106 | 0.305 | 59 | 45 | 2 | 55.66% | 0.89 | -71.74 | -0.0453 | -10.86% |
| 2026_recent | parent | 60 | 0.800 | 35 | 25 | 0 | 58.33% | 0.94 | -23.74 | -0.0237 | -11.30% |
| 2026_recent | candidate | 60 | 0.800 | 35 | 25 | 0 | 58.33% | 0.94 | -23.74 | -0.0237 | -11.30% |
| 2026_recent_cost_10 | parent | 60 | 0.800 | 35 | 25 | 0 | 58.33% | 0.92 | -33.10 | -0.0352 | -11.54% |
| 2026_recent_cost_10 | candidate | 60 | 0.800 | 35 | 25 | 0 | 58.33% | 0.92 | -33.10 | -0.0352 | -11.54% |

Selection-qualified candidates: `0`
Pareto candidates: `[]`
Qualified candidate: `none`; the displayed fallback is not deployable
Simultaneous win/frequency improvement: `False`
Research status: `research_only`

## False-positive and missed-winner diagnostics

| Period | Unique added | Added winners | Added losers | Losers removed | Winners accidentally removed |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 0 | 0 | 0 | 0 | 0 |
| 2021_2022 | 0 | 0 | 0 | 0 | 0 |
| 2023_2024 | 0 | 0 | 0 | 0 | 0 |
| 2025_2026_05_holdout | 0 | 0 | 0 | 0 | 0 |
| 2026_recent | 0 | 0 | 0 | 0 | 0 |

The candidate remains research_only and does not modify gemini.py.
