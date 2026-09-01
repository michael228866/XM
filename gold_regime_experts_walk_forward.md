# GOLD regime experts walk-forward

Four separate long/short trend/pullback experts. Research-only unless promotion passes.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 10201 | 43.49% | 0.51 | -999.80 | -100.54% |
| 2021_2022 | 8462 | 43.71% | 0.61 | -999.08 | -107.13% |
| 2023_2024 | 9164 | 46.64% | 0.61 | -999.07 | -103.62% |
| 2025_2026_05_holdout | 11622 | 59.61% | 0.66 | -994.35 | -102.80% |
| 2026_recent | 2722 | 62.60% | 0.78 | -670.90 | -73.68% |
| 2026_recent_cost_10 | 2716 | 62.59% | 0.75 | -702.19 | -76.65% |

Current recent benchmark: `{"pnl": 16.701348, "trades": 19, "win_rate": 0.631579, "profit_factor": 1.162366, "max_drawdown_pct": -0.047168, "max_consecutive_losses": 3, "take_profit_exits": 11, "stop_loss_exits": 7, "timeout_exits": 1, "stopped_out": false}`

May-18 recent benchmark: `{"pnl": 58.829954, "trades": 137, "win_rate": 0.729927, "profit_factor": 1.130596, "max_drawdown_pct": -0.055022, "max_consecutive_losses": 3, "take_profit_exits": 100, "stop_loss_exits": 37, "timeout_exits": 0, "stopped_out": false}`

Promotion gate: `FAIL`

Selected: `{"threshold": 0.5, "min_trend_strength": 0.0, "session_profile": "controlled_expanded", "tp_atr": 1.1, "sl_atr": 2.0, "max_hold": 180, "risk_per_trade": 0.014, "model_files": {"long_trend": "gold_regime_long_trend_xgb.json", "long_pullback": "gold_regime_long_pullback_xgb.json", "short_trend": "gold_regime_short_trend_xgb.json", "short_pullback": "gold_regime_short_pullback_xgb.json"}}`
