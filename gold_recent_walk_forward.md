# GOLD recent walk-forward

This is research-only. It does not replace the live model.

| Model / fold | Trades | Win | PF | PnL | DD |
|---|---:|---:|---:|---:|---:|
| Incumbent validation | 64 | 70.31% | 1.38 | 283.25 | -20.30% |
| Recent validation | 131 | 58.02% | 0.80 | -209.29 | -41.15% |
| Incumbent test | 87 | 67.82% | 1.24 | 203.64 | -19.65% |
| Recent test | 68 | 57.35% | 0.66 | -182.89 | -21.55% |

Selected threshold: `0.55`
Selected TP/SL: `1.1/2.0` ATR
Promotion gate: `FAIL`

Promotion requires both recent folds to remain profitable, test win rate at least 65%, PF at least 1.15, at least 20 test trades, and more test trades than the incumbent.
