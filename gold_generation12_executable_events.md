# GOLD generation 12 executable-event Expected-R

Natural P(win) and Mean-R models trained only on non-overlapping executable events.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| 2018_2020 | 112 | 55.36% | 0.70 | -114.58 | -14.93% |
| 2021_2022 | 92 | 53.26% | 0.66 | -109.05 | -13.52% |
| 2023_2024 | 39 | 53.85% | 0.64 | -65.83 | -9.06% |
| 2025_2026_05_holdout | 99 | 54.55% | 0.87 | -55.60 | -8.73% |
| 2026_recent | 51 | 60.78% | 0.93 | -15.34 | -6.66% |
| 2026_recent_cost_10 | 51 | 60.78% | 0.90 | -21.06 | -6.73% |

Qualified selection candidates: `0`
Selected: `{"generation": "12_executable_events", "top_k_per_day": 3, "minimum_expected_r": -0.05, "session_profile": "may_baseline", "quality_profile": "quality_105"}`
Current recent: `{"pnl": 8.837334, "trades": 20, "win_rate": 0.6, "profit_factor": 1.079812, "max_drawdown_pct": -0.055032, "max_consecutive_losses": 4, "take_profit_exits": 11, "stop_loss_exits": 8, "timeout_exits": 1, "stopped_out": false}`
Promotion gate: `FAIL`
