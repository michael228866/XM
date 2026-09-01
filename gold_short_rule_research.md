# GOLD short-direction research

Research-only: the live runner and production model are unchanged.

| Fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| development | 13 | 53.85% | 1.12 | 11.56 | -5.12% |
| validation | 9 | 44.44% | 0.54 | -36.58 | -6.33% |
| test | 8 | 50.00% | 0.81 | -12.06 | -3.55% |
| test_cost_10 | 8 | 50.00% | 0.80 | -12.58 | -3.58% |

Promotion gate: `FAIL`

Promotion requires validation and untouched test folds to have at least 10 trades, positive PnL, win rate >= 60%, PF >= 1.15, DD <= 15%, and the 10-point cost stress test to remain profitable with PF >= 1.05.
