# GOLD long model optimization

Research-only unless the promotion gate passes.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| validation | 112 | 63.39% | 1.31 | 216.76 | -9.93% |
| test | 124 | 54.84% | 0.87 | -83.43 | -16.46% |
| june_july | 91 | 53.85% | 0.86 | -73.35 | -16.12% |
| august | 29 | 62.07% | 1.09 | 19.09 | -7.72% |
| test_cost_10 | 124 | 54.84% | 0.86 | -89.75 | -16.00% |

Promotion gate: `FAIL`

Selected model: `{"train_start": "2025-07-01T00:00:00", "profile": "shallow", "positive_multiplier": 1.15, "threshold": 0.7, "tp_atr": 1.3, "sl_atr": 1.6, "max_hold": 90, "session_profile": "expanded"}`
