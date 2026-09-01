# GOLD generation 8 M5 residual walk-forward

Current long model stays the anchor; four calibrated M5 Expected-R experts may only add non-conflicting trades.

| Fold | Combined trades | Win | PF | PnL | DD | Anchor trades | Anchor win | Anchor PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | 582 | 30.93% | 0.48 | -535.66 | -57.31% | 507 | 28.80% | 0.41 |
| 2021_2022 | 329 | 41.95% | 0.64 | -313.03 | -37.76% | 231 | 35.06% | 0.60 |
| 2023_2024 | 171 | 49.71% | 0.68 | -171.94 | -18.72% | 44 | 47.73% | 0.70 |
| 2025_2026_05_holdout | 232 | 48.71% | 0.67 | -227.37 | -24.45% | 14 | 42.86% | 0.64 |
| 2026_recent | 43 | 62.79% | 1.17 | 37.58 | -6.35% | 19 | 63.16% | 1.16 |
| 2026_recent_cost_10 | 43 | 62.79% | 1.16 | 33.77 | -6.40% | 19 | 63.16% | 1.16 |

Qualified selection candidates: `0`
Selected: `min_R=0.1, session=controlled_expanded, quality=quality_115`
Promotion gate: `FAIL`
