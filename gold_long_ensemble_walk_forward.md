# GOLD long ensemble walk-forward

Research-only unless the promotion gate passes.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| validation | 69 | 69.57% | 1.68 | 255.51 | -5.77% |
| test | 85 | 52.94% | 0.66 | -135.70 | -14.94% |
| june_july | 59 | 49.15% | 0.60 | -129.20 | -14.36% |
| august | 26 | 61.54% | 0.99 | -1.56 | -3.79% |
| test_cost_10 | 85 | 52.94% | 0.65 | -139.74 | -15.21% |

Promotion gate: `FAIL`

Selected parameters: `{"threshold": 0.6, "tp_atr": 1.1, "sl_atr": 1.6, "max_hold": 90, "risk_per_trade": 0.014, "allowed_entry_hours": [0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23], "allowed_entry_weekdays": [0, 1, 2, 3, 4], "session_profile": "expanded", "incumbent_buy_threshold": 0.4, "meta_quality_floor": 0.6}`
