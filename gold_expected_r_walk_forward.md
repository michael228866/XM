# GOLD generation 6 Expected-R walk-forward

Four independent Expected-R experts with no-lookahead rolling top-k and realized-R champions.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 296 | 45.95% | 0.64 | -267.79 | -30.32% |
| 2021_2022 | 121 | 39.67% | 0.55 | -191.22 | -19.54% |
| 2023_2024 | 134 | 55.97% | 0.70 | -133.50 | -13.82% |
| 2025_2026_05_holdout | 278 | 57.19% | 0.90 | -90.36 | -20.18% |
| 2026_recent | 66 | 48.48% | 0.69 | -81.23 | -13.00% |
| 2026_recent_cost_10 | 66 | 48.48% | 0.68 | -85.47 | -13.15% |

Current recent benchmark: `{"pnl": 16.701348, "trades": 19, "win_rate": 0.631579, "profit_factor": 1.162366, "max_drawdown_pct": -0.047168, "max_consecutive_losses": 3, "take_profit_exits": 11, "stop_loss_exits": 7, "timeout_exits": 1, "stopped_out": false}`

Promotion gate: `FAIL`

Selected: `{"generation": "6_expected_r", "top_k_per_day": 3, "minimum_expected_r": -0.3, "session_profile": "may_baseline", "qualified": false, "score": -999999942423.4452, "folds": {"2018_2020": {"pnl": -267.793819, "trades": 296, "win_rate": 0.459459, "profit_factor": 0.635839, "max_drawdown_pct": -0.303229, "max_consecutive_losses": 13, "take_profit_exits": 115, "stop_loss_exits": 148, "timeout_exits": 33, "stopped_out": false}, "2021_2022": {"pnl": -191.221326, "trades": 121, "win_rate": 0.396694, "profit_factor": 0.545054, "max_drawdown_pct": -0.195419, "max_consecutive_losses": 18, "take_profit_exits": 47, "stop_loss_exits": 73, "timeout_exits": 1, "stopped_out": false}, "2023_2024": {"pnl": -133.504252, "trades": 134, "win_rate": 0.559701, "profit_factor": 0.702249, "max_drawdown_pct": -0.138194, "max_consecutive_losses": 5, "take_profit_exits": 75, "stop_loss_exits": 57, "timeout_exits": 2, "stopped_out": false}}, "tp_atr": 1.3, "sl_atr": 1.6, "max_hold": 90, "direction_mode": "both", "risk_per_trade": 0.014, "model_files": {"long_trend": "gold_expected_r_long_trend_xgb.json", "long_pullback": "gold_expected_r_long_pullback_xgb.json", "short_trend": "gold_expected_r_short_trend_xgb.json", "short_pullback": "gold_expected_r_short_pullback_xgb.json"}, "model_profile": {"objective": "reg:pseudohubererror", "n_estimators": 320, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 120, "recency_half_life_days": 730.0, "target_clip": 2.0}, "champion_config": {"maturity_rows": 90, "window_rows": 60000, "min_rows": 20000, "block_rows": 10080, "champion_min_trades": 10, "switch_margin": 0.05, "confirm_blocks": 2}}`
