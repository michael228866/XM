# E-010 GVZ information increment (H-014, volatility clause)

Outcome: **refuted**

refuted: general: refuted (A/R 1.1248, DM t 3.09, Sidak p 1.0, gain -12.48% ub95 -5.83%, years 2/6, lag2 A/R 1.1183, A/P 1.116); extreme: refuted (A/R 1.5593, DM t 5.4, Sidak p 1.0, gain -55.93% ub95 -38.91%, years 0/6, lag2 A/R 1.4282, A/P 1.5058); regime_up: refuted (A/R 1.1101, DM t 2.47, Sidak p 1.0, gain -11.01% ub95 -3.68%, years 2/6, lag2 A/R 1.1025, A/P 1.1069)

Statistical only; no economic value is claimed (H-018). Primary rows: general, extreme, regime_up on eval_2018_2023. general_h1, general_h22 and all check_2024_on rows are descriptive.
Evidence class: development (GOLD# 2018+ was inspected before; GVZ was never used).

| question / window | n | loss R | A | P | A_lag2 | A/R | DM t | DM t lag22 | Sidak p (A better) | gain % | gain ub95 % | years A better | A/P | A/R by regime | event rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| general|eval_2018_2023 | 1515 | 0.20284 | 0.22816 | 0.20443 | 0.22684 | 1.1248 | 3.09 | 2.34 | 1.0 | -12.48 | -5.83 | 2/6 | 1.116 | {'low': 0.9978, 'mid': 1.2587, 'high': 1.8229} |  |
| general|check_2024_on | 726 | 0.20858 | 0.2039 | 0.20987 | 0.20619 | 0.9775 | -0.14 | -0.12 | 0.8278 | 2.25 | 28.31 | 1/3 | 0.9715 | {'low': 1.1562, 'mid': 1.0231, 'high': 0.8903} |  |
| extreme|eval_2018_2023 | 1515 | 0.13491 | 0.21037 | 0.1397 | 0.19268 | 1.5593 | 5.4 | 3.84 | 1.0 | -55.93 | -38.91 | 0/6 | 1.5058 | {'low': 1.0936, 'mid': 2.0803, 'high': 2.8147} | 0.0284 |
| extreme|check_2024_on | 726 | 0.3538 | 0.43592 | 0.35379 | 0.41614 | 1.2321 | 3.15 | 2.43 | 1.0 | -23.21 | -11.1 | 0/3 | 1.2321 | {'low': 1.4933, 'mid': 0.915, 'high': 1.2422} | 0.1667 |
| regime_up|eval_2018_2023 | 1515 | 0.50695 | 0.56276 | 0.50843 | 0.55891 | 1.1101 | 2.47 | 2.47 | 1.0 | -11.01 | -3.68 | 2/6 | 1.1069 | {'low': 0.9846, 'mid': 1.3996, 'high': 2.2438} | 0.1815 |
| regime_up|check_2024_on | 709 | 0.52727 | 0.53008 | 0.52176 | 0.52927 | 1.0053 | 0.06 | 0.06 | 0.8925 | -0.53 | 13.7 | 2/3 | 1.016 | {'low': 0.9749, 'mid': 0.9335, 'high': 1.0794} | 0.2426 |
| general_h1|eval_2018_2023 | 1515 | 0.35272 | 0.38406 | 0.3531 | 0.38035 | 1.0889 | 3.87 | 3.04 | 1.0 | -8.89 | -5.11 | 1/6 | 1.0877 | {'low': 1.018, 'mid': 1.1799, 'high': 1.4477} |  |
| general_h1|check_2024_on | 730 | 0.30463 | 0.31685 | 0.30504 | 0.3168 | 1.0401 | 0.45 | 0.41 | 0.965 | -4.01 | 10.72 | 1/3 | 1.0387 | {'low': 1.1257, 'mid': 1.0503, 'high': 0.9851} |  |
| general_h22|eval_2018_2023 | 1515 | 0.17635 | 0.19865 | 0.17741 | 0.19847 | 1.1265 | 2.84 | 2.84 | 1.0 | -12.65 | -5.32 | 2/6 | 1.1198 | {'low': 0.987, 'mid': 1.3333, 'high': 1.8328} |  |
| general_h22|check_2024_on | 709 | 0.19669 | 0.1809 | 0.19843 | 0.18311 | 0.9197 | -0.48 | -0.48 | 0.6788 | 8.03 | 35.46 | 2/3 | 0.9117 | {'low': 0.9414, 'mid': 1.1334, 'high': 0.8709} |  |

Level controls (QLIKE; descriptive):

| question / window | R | A | EWMA_094 | RW5 | HIST | IV_SCALED |
|---|---|---|---|---|---|---|
| general|eval_2018_2023 | 0.20284 | 0.22816 | 0.18205 | 0.22253 | 0.38672 | 0.3739 |
| general|check_2024_on | 0.20858 | 0.2039 | 0.20714 | 0.2008 | 0.46546 | 0.31619 |
| general_h1|eval_2018_2023 | 0.35272 | 0.38406 | 0.32643 | 0.32562 | 0.55649 | 0.5326 |
| general_h1|check_2024_on | 0.30463 | 0.31685 | 0.2922 | 0.27337 | 0.59918 | 0.43821 |
| general_h22|eval_2018_2023 | 0.17635 | 0.19865 | 0.18791 | 0.26004 | 0.30668 | 0.34368 |
| general_h22|check_2024_on | 0.19669 | 0.1809 | 0.23574 | 0.26671 | 0.39236 | 0.27754 |

FROZEN coefficients (fit ['2014-01-01', '2018-01-01']; constant first): {'general': {'train_rows': 1019, 'R': {'beta': [-2.6711, 0.69], 'iterations': 5, 'grad': 1.3e-14}, 'A': {'beta': [-0.3589, 0.4582, 0.4731], 'iterations': 5, 'grad': 1.8e-13}, 'P': {'beta': [-2.724, 0.5096, 0.1739], 'iterations': 5, 'grad': 9.7e-15}, 'A_lag2': {'beta': [-0.5087, 0.4816, 0.4343], 'iterations': 5, 'grad': 1.8e-13}, 'iv_scale': 1.6172}, 'extreme': {'train_rows': 827, 'train_events': 127, 'R': {'beta': [-20.9298, 1.5307, -3.9698], 'iterations': 7, 'grad': 8.1e-16}, 'A': {'beta': [4.5655, 0.2309, -1.9199, 2.1977], 'iterations': 7, 'grad': 1.1e-13}, 'P': {'beta': [-20.5682, 0.0595, -3.8198, 1.3697], 'iterations': 7, 'grad': 1.2e-15}, 'A_lag2': {'beta': [-0.0796, 0.452, -2.2892, 1.8075], 'iterations': 7, 'grad': 3.3e-14}}, 'regime_up': {'train_rows': 1002, 'train_events': 206, 'R': {'beta': [-9.6677, 0.9081, -1.8343], 'iterations': 6, 'grad': 8.7e-13}, 'A': {'beta': [-0.8694, 0.162, -1.7479, 1.5906], 'iterations': 6, 'grad': 3.8e-12}, 'P': {'beta': [-9.8965, 0.5139, -1.7224, 0.252], 'iterations': 6, 'grad': 1.5e-12}, 'A_lag2': {'beta': [-1.6017, 0.1952, -1.7225, 1.4543], 'iterations': 6, 'grad': 2.4e-12}}, 'general_h1': {'train_rows': 1023, 'R': {'beta': [-2.4703, 0.7134], 'iterations': 6, 'grad': 1e-15}, 'A': {'beta': [-0.095, 0.4532, 0.5071], 'iterations': 6, 'grad': 5.9e-15}, 'P': {'beta': [-2.4851, 0.6557, 0.0558], 'iterations': 6, 'grad': 3.8e-15}, 'A_lag2': {'beta': [-0.2264, 0.4812, 0.4659], 'iterations': 6, 'grad': 1.7e-15}, 'iv_scale': 1.6067}, 'general_h22': {'train_rows': 1002, 'R': {'beta': [-3.0201, 0.6488], 'iterations': 5, 'grad': 4.5e-15}, 'A': {'beta': [-1.0512, 0.484, 0.3717], 'iterations': 5, 'grad': 2.8e-15}, 'P': {'beta': [-3.0638, 0.5188, 0.1246], 'iterations': 5, 'grad': 6.2e-15}, 'A_lag2': {'beta': [-1.1872, 0.4973, 0.3443], 'iterations': 4, 'grad': 3.5e-09}, 'iv_scale': 1.643}}.

GVZ alignment (lag 1): 3270/3270 server days usable, staleness {1: 2540, 2: 21, 3: 624, 4: 85}, same-day or future values used: 0.