# GOLD May-18 recovery sweep

The saved May-18 model is evaluated only on data after its training cutoff.
All exits use intrabar HIGH/LOW with conservative same-bar stop priority.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| historical_selection_1 | 63 | 49.21% | 0.65 | -95.53 | -13.76% |
| historical_selection_2 | 53 | 75.47% | 1.88 | 185.37 | -4.28% |
| historical_holdout | 70 | 77.14% | 1.87 | 223.94 | -5.76% |
| 2026_recent | 69 | 63.77% | 0.92 | -19.02 | -3.86% |
| 2026_recent_cost_10 | 69 | 63.77% | 0.89 | -25.18 | -4.36% |

Current benchmark: `{"pnl": 16.701348, "trades": 19, "win_rate": 0.631579, "profit_factor": 1.162366, "max_drawdown_pct": -0.047168, "max_consecutive_losses": 3, "take_profit_exits": 11, "stop_loss_exits": 7, "timeout_exits": 1, "stopped_out": false}`

Promotion gate: `FAIL`

Selected: `{"model_file": "gold_barrier_final_xgb.json", "model_output_mode": "three_class", "use_meta_overlay": true, "risk_per_trade": 0.014, "threshold": 0.5, "tp_atr": 1.3, "sl_atr": 2.0, "max_hold": 180, "session_profile": "may_baseline", "meta_quality_floor": 0.6, "allowed_entry_hours": [0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23], "allowed_entry_weekdays": [0, 1, 2, 4]}`
