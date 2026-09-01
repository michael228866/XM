# GOLD full-history long first-touch and reward-weighted walk-forward

Local history: 2014-02 through 2026-05. Recent MT5 data is the final gate.
First-touch and payoff-weighted labels are compared across all folds.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 62 | 38.71% | 0.53 | -125.71 | -13.55% |
| 2021_2022 | 41 | 36.59% | 0.41 | -130.90 | -14.49% |
| 2023_2024 | 77 | 64.94% | 1.44 | 157.65 | -9.00% |
| 2025_2026_05_holdout | 177 | 53.11% | 0.82 | -111.79 | -20.52% |
| 2026_06_recent | 69 | 43.48% | 0.53 | -137.99 | -14.42% |
| 2026_06_recent_cost_10 | 69 | 43.48% | 0.53 | -140.92 | -14.69% |

Promotion gate: `FAIL`

Selected: `{"family": "rolling_2y_reward", "profile": "regularized", "training_years": 2, "label_mode": "reward_weighted", "threshold": 0.5, "tp_atr": 1.3, "sl_atr": 1.6, "max_hold": 90, "session_profile": "expanded"}`
