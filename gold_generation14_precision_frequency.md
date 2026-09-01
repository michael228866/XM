# GOLD generation 14 precision-constrained frequency

Gen12 executable events with a leak-free loser meta-filter and past-only per-regime thresholds.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 72 | 50.00% | 0.54 | -129.41 | -13.27% |
| 2021_2022 | 116 | 49.14% | 0.68 | -128.30 | -15.55% |
| 2023_2024 | 232 | 58.62% | 0.81 | -134.58 | -16.11% |
| 2025_2026_05_holdout | 375 | 56.27% | 0.81 | -192.00 | -22.78% |
| 2026_recent | 82 | 63.41% | 1.18 | 55.18 | -9.48% |
| 2026_recent_cost_10 | 82 | 63.41% | 1.14 | 44.10 | -9.76% |

Stage1 executable opportunities: `4748`
Stage2 retained opportunities: `105`
Estimated loser removal: `98.00%`
Estimated winner removal: `97.61%`
Qualified selection candidates: `0`
Selected: `{"generation": "14_precision_frequency", "minimum_expected_r": -0.05, "session_profile": "controlled_expanded", "target_win_rate": 0.6, "minimum_profit_factor": 1.05}`
Gen12 recent: `{"pnl": -15.337937, "trades": 51, "win_rate": 0.607843, "profit_factor": 0.92928, "max_drawdown_pct": -0.066609, "max_consecutive_losses": 6, "take_profit_exits": 31, "stop_loss_exits": 20, "timeout_exits": 0, "stopped_out": false}`
Current recent: `{"pnl": 8.837334, "trades": 20, "win_rate": 0.6, "profit_factor": 1.079812, "max_drawdown_pct": -0.055032, "max_consecutive_losses": 4, "take_profit_exits": 11, "stop_loss_exits": 8, "timeout_exits": 1, "stopped_out": false}`
Promotion gate: `FAIL`
