# Training Profile Optimization

Compares XGBoost training profiles and then retunes execution parameters at 3x spread cost.

## Best By Symbol

| Symbol | Profile | Gate | R | Trades | Win | PF | Worst R | DD | Params |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| SILVER# | conservative_shallow | True | 17.55 | 60 | 71.67% | 2.97 | 1.81 | -3.05 | conf=0.52, edge=0.0, tp/sl=4.4/5.2, hold=216, dir=long |
| XAUEUR# | balanced_regularized | True | 11.85 | 30 | 86.67% | 335.47 | 2.78 | -1.35 | conf=0.58, edge=0.0, tp/sl=2.6/4.8, hold=288, dir=both |

## Cost Stress Of Selected Profiles

| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| SILVER# | conservative_shallow | 1.0x | False | 19.48 | 5/5 | 4/5 | 70 | 67.14% | 1.74 | -3.04 |
| SILVER# | conservative_shallow | 2.0x | False | 15.77 | 5/5 | 4/5 | 71 | 66.20% | 1.27 | -3.15 |
| SILVER# | conservative_shallow | 3.0x | True | 17.55 | 5/5 | 5/5 | 60 | 71.67% | 1.81 | -3.05 |
| SILVER# | conservative_shallow | 4.0x | False | 13.34 | 5/5 | 5/5 | 55 | 69.09% | 0.79 | -4.16 |
| SILVER# | conservative_shallow | 5.0x | False | 8.69 | 5/5 | 3/5 | 37 | 67.57% | 0.13 | -3.06 |
| XAUEUR# | balanced_regularized | 1.0x | True | 12.27 | 3/3 | 3/3 | 30 | 86.67% | 3.07 | -1.33 |
| XAUEUR# | balanced_regularized | 2.0x | True | 12.06 | 3/3 | 3/3 | 30 | 86.67% | 2.92 | -1.34 |
| XAUEUR# | balanced_regularized | 3.0x | True | 11.85 | 3/3 | 3/3 | 30 | 86.67% | 2.78 | -1.35 |
| XAUEUR# | balanced_regularized | 4.0x | True | 11.65 | 3/3 | 3/3 | 30 | 86.67% | 2.64 | -1.37 |
| XAUEUR# | balanced_regularized | 5.0x | True | 11.44 | 3/3 | 3/3 | 30 | 86.67% | 2.49 | -1.38 |
