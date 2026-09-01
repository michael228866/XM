# GOLD recent short walk-forward

Research-only: the live runner and production models are unchanged.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| validation | 32 | 84.38% | 3.54 | 279.06 | -3.00% |
| test | 36 | 55.56% | 0.82 | -36.21 | -10.24% |
| test_cost_10 | 36 | 55.56% | 0.81 | -39.22 | -10.32% |

Promotion gate: `FAIL`

Promotion requires validation and untouched recent test to have at least 10 trades, positive PnL, win rate >= 60%, PF >= 1.15, DD <= 15%, and a profitable 10-point cost stress test with PF >= 1.05.
