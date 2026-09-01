# GOLD iterative regime experts

Separate long/short trend/pullback models using first-touch labels.
Generation 1 uses balanced full history; generation 2 adds time decay; generation 3 blends both; generation 4 uses independent direction thresholds; generation 5 adds no-lookahead rolling calibration and dynamic champions.

| Generation | Qualified | Minimum trades | Worst win | Worst PF | Total PnL |
|---|---:|---:|---:|---:|---:|
| 1_balanced | False | 857 | 52.98% | 0.62 | -1452.78 |
| 2_time_decay | False | 768 | 52.67% | 0.62 | -1261.99 |
| 3_probability_blend | False | 148 | 58.01% | 0.79 | -517.78 |
| 4_independent_direction_thresholds | False | 140 | 58.98% | 0.80 | -374.17 |
| 5_rolling_champion_shared | False | 141 | 54.05% | 0.73 | -364.24 |
| 5_rolling_champion_directional | False | 104 | 50.96% | 0.74 | -356.67 |

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 490 | 58.98% | 0.80 | -250.75 | -29.64% |
| 2021_2022 | 255 | 60.00% | 0.92 | -67.16 | -23.29% |
| 2023_2024 | 140 | 61.43% | 0.89 | -56.26 | -10.97% |
| 2025_2026_05_holdout | 0 | 0.00% | inf | 0.00 | 0.00% |
| 2026_recent | 2 | 100.00% | inf | 16.91 | 0.00% |
| 2026_recent_cost_10 | 2 | 100.00% | inf | 16.41 | 0.00% |

Current recent benchmark: `{"pnl": 16.701348, "trades": 19, "win_rate": 0.631579, "profit_factor": 1.162366, "max_drawdown_pct": -0.047168, "max_consecutive_losses": 3, "take_profit_exits": 11, "stop_loss_exits": 7, "timeout_exits": 1, "stopped_out": false}`

May-18 recent benchmark: `{"pnl": 58.829954, "trades": 137, "win_rate": 0.729927, "profit_factor": 1.130596, "max_drawdown_pct": -0.055022, "max_consecutive_losses": 3, "take_profit_exits": 100, "stop_loss_exits": 37, "timeout_exits": 0, "stopped_out": false}`

Promotion gate: `FAIL`

Selected: `{"generation": "4_independent_direction_thresholds", "balanced_weight": 0.25, "time_decay_weight": 0.75, "threshold": 0.775, "long_threshold": 0.775, "short_threshold": 0.8, "min_trend_strength": 0.1, "session_profile": "may_baseline", "tp_atr": 1.3, "sl_atr": 1.6, "max_hold": 90, "direction_mode": "both", "risk_per_trade": 0.014, "allowed_entry_hours": [0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23], "allowed_entry_weekdays": [0, 1, 2, 4], "model_files": {"balanced": {"long_trend": "gold_iterative_balanced_long_trend_xgb.json", "long_pullback": "gold_iterative_balanced_long_pullback_xgb.json", "short_trend": "gold_iterative_balanced_short_trend_xgb.json", "short_pullback": "gold_iterative_balanced_short_pullback_xgb.json"}, "time_decay": {"long_trend": "gold_iterative_time_decay_long_trend_xgb.json", "long_pullback": "gold_iterative_time_decay_long_pullback_xgb.json", "short_trend": "gold_iterative_time_decay_short_trend_xgb.json", "short_pullback": "gold_iterative_time_decay_short_pullback_xgb.json"}}}`
