# Final SILVER / XAUEUR Robustness

Fixed candidates tested across 1x-5x spread cost plus 3x-cost parameter neighborhoods.

## Cost Stress

| Symbol | Cost | Gate | R | Positive | Passed | Trades | Win | PF | Worst R | DD | Recent R |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SILVER# | 1.0x | True | 21.74 | 5/5 | 5/5 | 71 | 63.38% | 2.07 | 1.70 | -2.93 | 3.33 |
| XAUEUR# | 1.0x | True | 13.81 | 3/3 | 3/3 | 41 | 82.93% | 334.71 | 1.76 | -2.35 | 6.28 |
| SILVER# | 2.0x | True | 20.75 | 5/5 | 5/5 | 66 | 65.15% | 2.37 | 1.41 | -3.19 | 3.24 |
| XAUEUR# | 2.0x | True | 13.54 | 3/3 | 3/3 | 41 | 82.93% | 334.69 | 1.58 | -2.38 | 6.24 |
| SILVER# | 3.0x | True | 16.68 | 5/5 | 5/5 | 62 | 64.52% | 2.05 | 1.11 | -2.95 | 3.15 |
| XAUEUR# | 3.0x | True | 13.27 | 3/3 | 3/3 | 41 | 82.93% | 334.68 | 1.40 | -2.41 | 6.19 |
| SILVER# | 4.0x | False | 9.37 | 4/5 | 3/5 | 56 | 60.71% | 201.16 | -3.12 | -5.99 | 3.06 |
| XAUEUR# | 4.0x | True | 12.99 | 3/3 | 3/3 | 41 | 82.93% | 334.66 | 1.21 | -2.44 | 6.15 |
| SILVER# | 5.0x | False | 7.60 | 4/5 | 2/5 | 37 | 62.16% | 201.05 | -2.63 | -2.95 | 2.98 |
| XAUEUR# | 5.0x | True | 12.72 | 3/3 | 3/3 | 41 | 82.93% | 334.64 | 1.03 | -2.47 | 6.11 |

## Neighborhood

| Symbol | Variants | Gate Passed | Pass Rate | Median R | Best R | Worst R |
|---|---:|---:|---:|---:|---:|---:|
| SILVER# | 288 | 32 | 11.11% | 10.34 | 16.89 | 3.61 |
| XAUEUR# | 216 | 12 | 5.56% | 6.10 | 13.39 | 2.08 |

## Verdict

Both candidates pass the 3x-cost final gate. Keep 4x/5x results as risk limits; promote only after live-paper logging from MT5 is clean.
