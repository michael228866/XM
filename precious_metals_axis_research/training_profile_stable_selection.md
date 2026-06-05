# Stable Training Profile Selection

Selects profiles by cost-stress stability first, then 3x R.

## Selected

| Symbol | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| SILVER# | current_symbol | 3/5 | True | 17.31 | 64 | 65.63% | 1.79 | 1.56 | -2.77 | conf=0.52, edge=0.0, tp/sl=6.0/6.0, hold=336, dir=long |
| XAUEUR# | smooth_more_trees | 5/5 | True | 12.39 | 36 | 83.33% | 3.79 | 2.75 | -2.19 | conf=0.54, edge=0.0, tp/sl=2.2/4.2, hold=288, dir=both |

## Cost Stress

| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| SILVER# | current_symbol | 1.0x | True | 21.98 | 5/5 | 5/5 | 73 | 64.38% | 1.88 | -2.76 |
| SILVER# | current_symbol | 2.0x | True | 19.71 | 5/5 | 5/5 | 73 | 64.38% | 1.72 | -2.76 |
| SILVER# | current_symbol | 3.0x | True | 17.31 | 5/5 | 5/5 | 64 | 65.63% | 1.56 | -2.77 |
| SILVER# | current_symbol | 4.0x | False | 11.40 | 5/5 | 2/5 | 44 | 63.64% | 0.62 | -2.78 |
| SILVER# | current_symbol | 5.0x | False | 6.53 | 5/5 | 1/5 | 28 | 64.29% | 0.34 | -2.80 |
| XAUEUR# | smooth_more_trees | 1.0x | True | 12.95 | 3/3 | 3/3 | 36 | 83.33% | 3.15 | -2.05 |
| XAUEUR# | smooth_more_trees | 2.0x | True | 12.67 | 3/3 | 3/3 | 36 | 83.33% | 2.95 | -2.12 |
| XAUEUR# | smooth_more_trees | 3.0x | True | 12.39 | 3/3 | 3/3 | 36 | 83.33% | 2.75 | -2.19 |
| XAUEUR# | smooth_more_trees | 4.0x | True | 12.11 | 3/3 | 3/3 | 36 | 83.33% | 2.54 | -2.27 |
| XAUEUR# | smooth_more_trees | 5.0x | True | 11.83 | 3/3 | 3/3 | 36 | 83.33% | 2.34 | -2.35 |
