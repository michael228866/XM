# E-006 swap avoidance (H-008)

Outcome: **inconclusive**

Sharpe INTRADAY_LONG vs ALWAYS_LONG (cost x1): 2018-2022 0.152 vs -0.029, 2023+ 0.999 vs 0.773; better in both windows at cost x1.5, live31 and low swap: False

Evidence class: development (gold prices 2018+ were inspected before; sessions were not decomposed).

## Base scenario (cost x1, swap x1)

| strategy | window | days | net bp/day | t | Sharpe | Calmar | max DD % | annual % | positive years |
|---|---|---|---|---|---|---|---|---|---|
| ALWAYS_LONG | discovery_2018_2022 | 1279 | -0.164 | -0.07 | -0.029 | -0.012 | -33.14 | -0.41 | 2/5 |
| ALWAYS_LONG | check_2023_on | 996 | 6.018 | 1.54 | 0.773 | 0.471 | -32.21 | 15.17 | 3/4 |
| INTRADAY_LONG | discovery_2018_2022 | 1279 | 0.851 | 0.34 | 0.152 | 0.078 | -27.61 | 2.15 | 2/5 |
| INTRADAY_LONG | check_2023_on | 996 | 7.619 | 1.99 | 0.999 | 0.632 | -30.38 | 19.2 | 3/4 |
| OVERNIGHT_LONG | discovery_2018_2022 | 1279 | -3.624 | -10.53 | -4.674 | -0.245 | -37.21 | -9.13 | 0/5 |
| OVERNIGHT_LONG | check_2023_on | 996 | -3.238 | -5.73 | -2.88 | -0.294 | -27.79 | -8.16 | 0/4 |

## Cost and swap sensitivity (net bp/day / Sharpe)

| strategy | scenario | discovery_2018_2022 | check_2023_on |
|---|---|---|---|
| ALWAYS_LONG | cost x1.0 | -0.164 / -0.029 | 6.018 / 0.773 |
| ALWAYS_LONG | cost x1.5 | -0.164 / -0.029 | 6.018 / 0.773 |
| ALWAYS_LONG | cost x2.0 | -0.164 / -0.029 | 6.018 / 0.773 |
| ALWAYS_LONG | cost live31 | -0.164 / -0.029 | 6.018 / 0.773 |
| ALWAYS_LONG | swap x0.5 | 1.198 / 0.215 | 7.378 / 0.947 |
| INTRADAY_LONG | cost x1.0 | 0.851 / 0.152 | 7.619 / 0.999 |
| INTRADAY_LONG | cost x1.5 | 0.196 / 0.035 | 7.214 / 0.946 |
| INTRADAY_LONG | cost x2.0 | -0.459 / -0.082 | 6.809 / 0.893 |
| INTRADAY_LONG | cost live31 | 0.06 / 0.011 | 7.257 / 0.951 |
| INTRADAY_LONG | swap x0.5 | 0.851 / 0.152 | 7.619 / 0.999 |
| OVERNIGHT_LONG | cost x1.0 | -3.624 / -4.674 | -3.238 / -2.88 |
| OVERNIGHT_LONG | cost x1.5 | -4.279 / -5.517 | -3.643 / -3.241 |
| OVERNIGHT_LONG | cost x2.0 | -4.935 / -6.346 | -4.048 / -3.601 |
| OVERNIGHT_LONG | cost live31 | -4.415 / -5.692 | -3.6 / -3.199 |
| OVERNIGHT_LONG | swap x0.5 | -2.262 / -2.939 | -1.878 / -1.673 |

Sessions (server hours): {'ASIA': (1, 9), 'LONDON': (9, 15), 'NY': (15, 19), 'LATE': (19, 24)}; trials 1; swap long per night (points) -78.59 = -7.09%/yr of price, applied as a rate in every year.