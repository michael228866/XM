# E-008 volatility forecasting, layer 1 (H-011)

Outcome: **refuted**

refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)

Statistical forecast accuracy only; says nothing about economic value (H-018).
Evidence class: development (prices 2018+ were inspected before; these forecasts were not). The 2024+ rows are reported and were never used to choose a model.

| protocol / horizon / window | days | QLIKE RW | HIST | EWMA_094 | EWMA_FIT | HAR | HAR/EWMA_FIT | DM t (QLIKE) | DM t (MSE) | years HAR better | by regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FROZEN|h1|eval_2018_2023 | 1515 | 0.54985 | 0.55649 | 0.32643 | 0.3396 | 0.38687 | 1.1392 | 2.94 | -0.38 | 1/6 | {'low': 1.1849, 'mid': 1.1215, 'high': 0.74} |
| FROZEN|h1|check_2024_on | 730 | 0.36765 | 0.59918 | 0.2922 | 0.31337 | 0.29675 | 0.947 | -0.97 | -1.32 | 2/3 | {'low': 1.0135, 'mid': 1.0443, 'high': 0.8744} |
| RECURSIVE|h1|eval_2018_2023 | 1515 | 0.54985 | 0.55649 | 0.32643 | 0.32933 | 0.32435 | 0.9849 | -0.65 | -0.49 | 5/6 | {'low': 0.9787, 'mid': 1.0249, 'high': 0.8487} |
| RECURSIVE|h1|check_2024_on | 730 | 0.36765 | 0.59918 | 0.2922 | 0.2742 | 0.25303 | 0.9228 | -2.65 | -1.61 | 3/3 | {'low': 0.9151, 'mid': 0.9996, 'high': 0.9015} |
| FROZEN|h5|eval_2018_2023 | 1515 | 0.22253 | 0.38672 | 0.18205 | 0.19128 | 0.24127 | 1.2613 | 3.02 | 1.09 | 1/6 | {'low': 1.3083, 'mid': 1.2771, 'high': 0.7917} |
| FROZEN|h5|check_2024_on | 726 | 0.2008 | 0.46546 | 0.20714 | 0.22393 | 0.21821 | 0.9745 | -0.33 | -0.32 | 2/3 | {'low': 0.9345, 'mid': 1.1211, 'high': 0.9568} |
| RECURSIVE|h5|eval_2018_2023 | 1515 | 0.22253 | 0.38672 | 0.18205 | 0.18842 | 0.1899 | 1.0079 | 0.2 | 0.75 | 4/6 | {'low': 0.9944, 'mid': 1.0747, 'high': 0.891} |
| RECURSIVE|h5|check_2024_on | 726 | 0.2008 | 0.46546 | 0.20714 | 0.20009 | 0.17979 | 0.8985 | -1.95 | -1.49 | 3/3 | {'low': 0.8056, 'mid': 0.9807, 'high': 0.9257} |
| FROZEN|h22|eval_2018_2023 | 1515 | 0.21119 | 0.30668 | 0.18791 | 0.18667 | 0.20588 | 1.1029 | 0.46 | 1.07 | 2/6 | {'low': 1.0129, 'mid': 1.7365, 'high': 0.777} |
| FROZEN|h22|check_2024_on | 709 | 0.26923 | 0.39236 | 0.23574 | 0.23717 | 0.21329 | 0.8993 | -0.78 | -0.65 | 1/3 | {'low': 0.5857, 'mid': 1.3788, 'high': 1.0193} |
| RECURSIVE|h22|eval_2018_2023 | 1515 | 0.21119 | 0.30668 | 0.18791 | 0.18896 | 0.18004 | 0.9528 | -0.5 | 1.15 | 3/6 | {'low': 0.894, 'mid': 1.2786, 'high': 0.9138} |
| RECURSIVE|h22|check_2024_on | 709 | 0.26923 | 0.39236 | 0.23574 | 0.23717 | 0.21281 | 0.8973 | -1.06 | 0.4 | 2/3 | {'low': 0.6452, 'mid': 0.8981, 'high': 1.0455} |

Primary-cell detail (FROZEN h=1 2018-2023, HAR vs EWMA_FIT): DM t lag 22 2.26, by weekday {0: 1.2345, 1: 1.1727, 2: 1.1375, 3: 1.1203, 4: 1.067}, HAR at floor 0 days.

FROZEN parameters (fit ['2014-01-01', '2018-01-01']): {1: {'beta': [5.2e-05, 0.027644, 0.113689, 0.558551], 'lambda': 0.96}, 5: {'beta': [5.4e-05, 0.042989, -0.017381, 0.657862], 'lambda': 0.96}, 22: {'beta': [6.1e-05, 0.005627, 0.063476, 0.586136], 'lambda': 0.96}, 'parkinson_scale': 0.4576}.
Days used: 3334 complete server days (>= 20 H1 bars). Full MSE/RMSE/MAE in metrics.json.