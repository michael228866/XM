# E-007 volatility-managed long at equal risk budget (H-005)

Outcome: **refuted**

Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65

Evidence class: development (gold prices were inspected before; this rule was not).

| window | scenario | strategy | net bp/day | Sharpe | Calmar | max DD % | annual % |
|---|---|---|---|---|---|---|---|
| discovery_2018_2023 | cost x1.0 | MANAGED | 0.258 | 0.051 | 0.022 | -29.6 | 0.65 |
| discovery_2018_2023 | cost x1.0 | CONST_EQUAL_EXPOSURE | 0.616 | 0.112 | 0.047 | -32.7 | 1.55 |
| discovery_2018_2023 | cost x1.0 | CONST_VOL_MATCHED | 0.567 | 0.112 | 0.047 | -30.41 | 1.43 |
| discovery_2018_2023 | cost x1.0 | SHUFFLED | - | 0.087 (share >= managed 0.65) | - | - | - |
| discovery_2018_2023 | cost x2.0 | MANAGED | 0.231 | 0.046 | 0.02 | -29.74 | 0.58 |
| discovery_2018_2023 | cost x2.0 | CONST_EQUAL_EXPOSURE | 0.616 | 0.112 | 0.047 | -32.7 | 1.55 |
| discovery_2018_2023 | cost x2.0 | CONST_VOL_MATCHED | 0.567 | 0.112 | 0.047 | -30.41 | 1.43 |
| discovery_2018_2023 | cost x2.0 | SHUFFLED | - | 0.081 (share >= managed 0.64) | - | - | - |
| discovery_2018_2023 | cost x3.0 | MANAGED | 0.204 | 0.04 | 0.017 | -29.88 | 0.51 |
| discovery_2018_2023 | cost x3.0 | CONST_EQUAL_EXPOSURE | 0.615 | 0.112 | 0.047 | -32.7 | 1.55 |
| discovery_2018_2023 | cost x3.0 | CONST_VOL_MATCHED | 0.566 | 0.112 | 0.047 | -30.4 | 1.43 |
| discovery_2018_2023 | cost x3.0 | SHUFFLED | - | 0.074 (share >= managed 0.64) | - | - | - |
| discovery_2018_2023 | swap x0.5 | MANAGED | 1.632 | 0.322 | 0.179 | -22.98 | 4.11 |
| discovery_2018_2023 | swap x0.5 | CONST_EQUAL_EXPOSURE | 1.989 | 0.361 | 0.185 | -27.16 | 5.01 |
| discovery_2018_2023 | swap x0.5 | CONST_VOL_MATCHED | 1.83 | 0.361 | 0.183 | -25.15 | 4.61 |
| discovery_2018_2023 | swap x0.5 | SHUFFLED | - | 0.33 (share >= managed 0.525) | - | - | - |
| check_2024_on | cost x1.0 | MANAGED | 7.472 | 1.378 | 0.964 | -19.53 | 18.83 |
| check_2024_on | cost x1.0 | CONST_EQUAL_EXPOSURE | 5.379 | 0.854 | 0.553 | -24.53 | 13.56 |
| check_2024_on | cost x1.0 | CONST_VOL_MATCHED | 4.632 | 0.854 | 0.547 | -21.35 | 11.67 |
| check_2024_on | cost x1.0 | SHUFFLED | - | 0.815 (share >= managed 0.005) | - | - | - |
| check_2024_on | cost x2.0 | MANAGED | 7.452 | 1.374 | 0.96 | -19.55 | 18.78 |
| check_2024_on | cost x2.0 | CONST_EQUAL_EXPOSURE | 5.378 | 0.854 | 0.552 | -24.53 | 13.55 |
| check_2024_on | cost x2.0 | CONST_VOL_MATCHED | 4.631 | 0.854 | 0.547 | -21.35 | 11.67 |
| check_2024_on | cost x2.0 | SHUFFLED | - | 0.81 (share >= managed 0.005) | - | - | - |
| check_2024_on | cost x3.0 | MANAGED | 7.432 | 1.37 | 0.957 | -19.57 | 18.73 |
| check_2024_on | cost x3.0 | CONST_EQUAL_EXPOSURE | 5.378 | 0.854 | 0.552 | -24.53 | 13.55 |
| check_2024_on | cost x3.0 | CONST_VOL_MATCHED | 4.63 | 0.854 | 0.547 | -21.35 | 11.67 |
| check_2024_on | cost x3.0 | SHUFFLED | - | 0.806 (share >= managed 0.005) | - | - | - |
| check_2024_on | swap x0.5 | MANAGED | 8.481 | 1.563 | 1.158 | -18.46 | 21.37 |
| check_2024_on | swap x0.5 | CONST_EQUAL_EXPOSURE | 6.388 | 1.014 | 0.699 | -23.01 | 16.1 |
| check_2024_on | swap x0.5 | CONST_VOL_MATCHED | 5.504 | 1.014 | 0.693 | -20.0 | 13.87 |
| check_2024_on | swap x0.5 | SHUFFLED | - | 0.966 (share >= managed 0.005) | - | - | - |

EWMA lambda 0.94, cap 2.0, spread 8e-05 per unit of weight change, swap long per night (points) -78.59, mean weight 2018-2023 1.003.