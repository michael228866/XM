# XPT / XPD Extended Timeframe Optimization

Research-only walk-forward search across H2/H4/H12/Daily with 1x-5x cost stress.

## Selected

| Symbol | TF | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |
|---|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| XPTUSD# | H4 | smooth_more_trees | 4/5 | True | 10.61 | 38 | 81.58% | 502.37 | 0.67 | -3.09 | none: conf=0.58, edge=0.0, tp/sl=3.2/5.4, hold=96, dir=long |
| XPDUSD# | H12 | current_symbol | 0/5 | False | 4.11 | 19 | 78.95% | 500.13 | 0.38 | -1.29 | low_vola: conf=0.6, edge=0.0, tp/sl=1.6/3.2, hold=48, dir=long |

## Cost Stress

| Symbol | TF | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |
|---|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| XPTUSD# | H4 | smooth_more_trees | 1.0x | True | 13.37 | 4/4 | 3/4 | 38 | 81.58% | 0.67 | -2.98 |
| XPTUSD# | H4 | smooth_more_trees | 2.0x | True | 11.99 | 4/4 | 3/4 | 38 | 81.58% | 0.67 | -3.03 |
| XPTUSD# | H4 | smooth_more_trees | 3.0x | True | 10.61 | 4/4 | 3/4 | 38 | 81.58% | 0.67 | -3.09 |
| XPTUSD# | H4 | smooth_more_trees | 4.0x | True | 9.23 | 4/4 | 3/4 | 38 | 78.95% | 0.67 | -3.15 |
| XPTUSD# | H4 | smooth_more_trees | 5.0x | False | 7.85 | 4/4 | 3/4 | 38 | 76.32% | 0.67 | -3.20 |
| XPDUSD# | H12 | current_symbol | 1.0x | False | 5.90 | 4/4 | 3/4 | 19 | 78.95% | 0.42 | -1.18 |
| XPDUSD# | H12 | current_symbol | 2.0x | False | 5.01 | 4/4 | 3/4 | 19 | 78.95% | 0.42 | -1.24 |
| XPDUSD# | H12 | current_symbol | 3.0x | False | 4.11 | 4/4 | 3/4 | 19 | 78.95% | 0.38 | -1.29 |
| XPDUSD# | H12 | current_symbol | 4.0x | False | 3.21 | 3/4 | 2/4 | 19 | 78.95% | -0.16 | -1.50 |
| XPDUSD# | H12 | current_symbol | 5.0x | False | 2.31 | 3/4 | 2/4 | 19 | 78.95% | -0.70 | -1.78 |
