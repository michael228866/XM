# GOLD execution-aligned long model optimization

Research-only unless the promotion gate passes.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| validation | 132 | 62.12% | 1.76 | 504.25 | -10.65% |
| test | 71 | 39.44% | 0.58 | -161.07 | -17.68% |
| june_july | 55 | 34.55% | 0.55 | -151.78 | -16.75% |
| august | 16 | 56.25% | 0.84 | -20.97 | -6.93% |
| test_cost_10 | 71 | 39.44% | 0.57 | -163.45 | -17.91% |

Promotion gate: `FAIL`

Selected model: `{"train_start": "2026-01-01T00:00:00", "profile": "baseline", "positive_multiplier": 0.85, "threshold": 0.65, "tp_atr": 1.3, "sl_atr": 1.6, "max_hold": 90, "session_profile": "expanded"}`
