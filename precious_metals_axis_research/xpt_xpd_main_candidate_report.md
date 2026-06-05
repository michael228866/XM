# XPT / XPD Main Candidate Optimization

H1 symbol models with profile search, regime filters, and 1x-5x spread stress.

## Selected

| Symbol | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| XPTUSD# | current_symbol | 1/5 | False | 13.51 | 69 | 81.16% | 250.92 | -2.33 | -4.28 | low_vola: conf=0.6, edge=0.0, tp/sl=3.2/5.2, hold=288, dir=long |
| XPDUSD# | smooth_more_trees | 0/5 | False | 8.65 | 34 | 79.41% | 500.50 | 0.00 | -2.43 | stable_combo: conf=0.66, edge=0.0, tp/sl=2.0/3.4, hold=288, dir=both |

## Cost Stress

| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| XPTUSD# | current_symbol | 1.0x | True | 21.52 | 4/4 | 4/4 | 69 | 81.16% | 0.97 | -3.77 |
| XPTUSD# | current_symbol | 2.0x | False | 17.52 | 3/4 | 3/4 | 69 | 81.16% | -0.68 | -4.02 |
| XPTUSD# | current_symbol | 3.0x | False | 13.51 | 3/4 | 3/4 | 69 | 81.16% | -2.33 | -4.28 |
| XPTUSD# | current_symbol | 4.0x | False | 9.51 | 3/4 | 3/4 | 69 | 79.71% | -3.99 | -4.82 |
| XPTUSD# | current_symbol | 5.0x | False | 5.51 | 3/4 | 3/4 | 69 | 71.01% | -5.64 | -5.88 |
| XPDUSD# | smooth_more_trees | 1.0x | False | -8.31 | 1/4 | 1/4 | 108 | 62.04% | -9.12 | -11.46 |
| XPDUSD# | smooth_more_trees | 2.0x | False | -0.94 | 2/4 | 1/4 | 66 | 66.67% | -6.37 | -6.80 |
| XPDUSD# | smooth_more_trees | 3.0x | False | 8.65 | 3/4 | 2/4 | 34 | 79.41% | 0.00 | -2.43 |
| XPDUSD# | smooth_more_trees | 4.0x | False | 6.31 | 2/4 | 2/4 | 22 | 77.27% | 0.00 | -1.27 |
| XPDUSD# | smooth_more_trees | 5.0x | False | 5.39 | 1/4 | 1/4 | 18 | 77.78% | -0.12 | -1.29 |
