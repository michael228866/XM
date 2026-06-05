# Long TF Cost-Aware Walk-Forward

Each fold retrains the model and evaluates the next window with CSV spread cost.

## Summary

| Symbol | Candidate | Positive | Passed | Total R | Trades | Win | Worst Fold R | Max DD R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SILVER# | silver_optimized | 4/4 | 2/4 | 25.43 | 186 | 61.29% | 1.01 | -9.89 |
| SILVER# | silver_best | 3/4 | 3/4 | 16.90 | 119 | 70.59% | -1.24 | -4.53 |
| XPTUSD# | xpt_wide_sl | 3/4 | 1/4 | 4.63 | 262 | 65.65% | -5.10 | -11.67 |
| XPTUSD# | xpt_best | 3/4 | 0/4 | -3.33 | 278 | 62.23% | -9.27 | -15.15 |
| XPTUSD# | xpt_long_hold | 2/4 | 0/4 | -7.47 | 266 | 61.65% | -9.96 | -14.77 |

## Folds

| Symbol | Fold | Candidate | Period | R | Win | PF | Trades | DD R | Pass |
|---|---|---|---|---:|---:|---:|---:|---:|:---:|
| XPTUSD# | fold_1 | xpt_best | 2023-12-26T08:00:00 -> 2024-07-24T23:00:00 | 0.92 | 61.11% | 1.06 | 36 | -5.71 | False |
| XPTUSD# | fold_1 | xpt_wide_sl | 2023-12-26T08:00:00 -> 2024-07-24T23:00:00 | 0.13 | 61.11% | 1.01 | 36 | -5.48 | False |
| XPTUSD# | fold_1 | xpt_long_hold | 2023-12-26T08:00:00 -> 2024-07-24T23:00:00 | -3.04 | 55.56% | 0.84 | 36 | -8.18 | False |
| XPTUSD# | fold_2 | xpt_best | 2024-07-25T01:00:00 -> 2025-02-24T08:00:00 | -9.27 | 60.87% | 0.62 | 46 | -9.80 | False |
| XPTUSD# | fold_2 | xpt_wide_sl | 2024-07-25T01:00:00 -> 2025-02-24T08:00:00 | -5.10 | 64.10% | 0.69 | 39 | -6.90 | False |
| XPTUSD# | fold_2 | xpt_long_hold | 2024-07-25T01:00:00 -> 2025-02-24T08:00:00 | -9.96 | 60.87% | 0.60 | 46 | -10.47 | False |
| XPTUSD# | fold_3 | xpt_best | 2025-02-24T09:00:00 -> 2025-09-22T20:00:00 | 1.62 | 65.71% | 1.12 | 35 | -4.99 | False |
| XPTUSD# | fold_3 | xpt_wide_sl | 2025-02-24T09:00:00 -> 2025-09-22T20:00:00 | 3.75 | 72.22% | 1.37 | 36 | -3.77 | True |
| XPTUSD# | fold_3 | xpt_long_hold | 2025-02-24T09:00:00 -> 2025-09-22T20:00:00 | 1.59 | 68.57% | 1.11 | 35 | -4.99 | False |
| XPTUSD# | fold_4 | xpt_best | 2025-09-22T21:00:00 -> 2026-05-14T04:00:00 | 3.40 | 62.11% | 1.05 | 161 | -15.15 | False |
| XPTUSD# | fold_4 | xpt_wide_sl | 2025-09-22T21:00:00 -> 2026-05-14T04:00:00 | 5.85 | 65.56% | 1.10 | 151 | -11.67 | False |
| XPTUSD# | fold_4 | xpt_long_hold | 2025-09-22T21:00:00 -> 2026-05-14T04:00:00 | 3.95 | 61.74% | 1.06 | 149 | -14.77 | False |
| SILVER# | fold_1 | silver_best | 2022-08-26T08:00:00 -> 2023-07-24T19:00:00 | 12.99 | 77.50% | 2.22 | 40 | -3.46 | True |
| SILVER# | fold_1 | silver_optimized | 2022-08-26T08:00:00 -> 2023-07-24T19:00:00 | 14.30 | 65.15% | 1.48 | 66 | -9.21 | True |
| SILVER# | fold_2 | silver_best | 2023-07-24T20:00:00 -> 2024-06-19T04:00:00 | 1.87 | 72.41% | 1.16 | 29 | -3.20 | True |
| SILVER# | fold_2 | silver_optimized | 2023-07-24T20:00:00 -> 2024-06-19T04:00:00 | 4.13 | 62.00% | 1.15 | 50 | -9.89 | False |
| SILVER# | fold_3 | silver_best | 2024-06-19T05:00:00 -> 2025-05-16T11:00:00 | -1.24 | 42.86% | 0.64 | 7 | -3.41 | False |
| SILVER# | fold_3 | silver_optimized | 2024-06-19T05:00:00 -> 2025-05-16T11:00:00 | 1.01 | 57.89% | 1.11 | 19 | -5.72 | False |
| SILVER# | fold_4 | silver_best | 2025-05-16T12:00:00 -> 2026-05-14T04:00:00 | 3.28 | 67.44% | 1.18 | 43 | -4.53 | True |
| SILVER# | fold_4 | silver_optimized | 2025-05-16T12:00:00 -> 2026-05-14T04:00:00 | 5.99 | 56.86% | 1.25 | 51 | -6.44 | True |
