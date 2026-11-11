# E-005 session direction (H-004)

Outcome: **refuted**

sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)

Evidence class: development (gold prices 2018+ were inspected before; sessions were not decomposed).

## Base scenario (cost x1, swap x1)

| strategy | window | days | net bp/day | t | Sharpe | Calmar | max DD % | annual % | positive years |
|---|---|---|---|---|---|---|---|---|---|
| ASIA_long | discovery_2018_2022 | 1279 | 1.303 | 1.32 | 0.584 | 0.323 | -10.16 | 3.28 | 4/5 |
| ASIA_long | check_2023_on | 996 | 4.703 | 2.23 | 1.121 | 0.662 | -17.91 | 11.85 | 4/4 |
| ASIA_long_excess | discovery_2018_2022 | 1279 | 0.551 | 0.55 | 0.242 | 0.112 | -12.42 | 1.39 | 2/5 |
| ASIA_long_excess | check_2023_on | 996 | 1.771 | 0.83 | 0.417 | 0.335 | -13.31 | 4.46 | 4/4 |
| ASIA_short | discovery_2018_2022 | 1279 | -3.814 | -3.86 | -1.713 | -0.235 | -40.83 | -9.61 | 0/5 |
| ASIA_short | check_2023_on | 996 | -6.321 | -3.0 | -1.507 | -0.299 | -53.31 | -15.93 | 0/4 |
| ASIA_short_excess | discovery_2018_2022 | 1279 | -3.062 | -3.04 | -1.351 | -0.222 | -34.73 | -7.72 | 0/5 |
| ASIA_short_excess | check_2023_on | 996 | -3.389 | -1.59 | -0.798 | -0.25 | -34.16 | -8.54 | 0/4 |
| LONDON_long | discovery_2018_2022 | 1279 | -1.409 | -1.15 | -0.51 | -0.138 | -25.64 | -3.55 | 1/5 |
| LONDON_long | check_2023_on | 996 | 2.783 | 1.56 | 0.785 | 0.786 | -8.92 | 7.01 | 4/4 |
| LONDON_long_excess | discovery_2018_2022 | 1279 | -1.973 | -1.6 | -0.709 | -0.2 | -24.85 | -4.97 | 1/5 |
| LONDON_long_excess | check_2023_on | 996 | 0.584 | 0.32 | 0.163 | 0.088 | -16.68 | 1.47 | 1/4 |
| LONDON_short | discovery_2018_2022 | 1279 | -0.84 | -0.68 | -0.303 | -0.083 | -25.39 | -2.12 | 2/5 |
| LONDON_short | check_2023_on | 996 | -4.386 | -2.46 | -1.237 | -0.289 | -38.25 | -11.05 | 0/4 |
| LONDON_short_excess | discovery_2018_2022 | 1279 | -0.276 | -0.22 | -0.099 | -0.036 | -19.1 | -0.7 | 2/5 |
| LONDON_short_excess | check_2023_on | 996 | -2.186 | -1.22 | -0.611 | -0.225 | -24.54 | -5.51 | 1/4 |
| NY_long | discovery_2018_2022 | 1279 | 0.182 | 0.11 | 0.048 | 0.032 | -14.4 | 0.46 | 2/5 |
| NY_long | check_2023_on | 996 | -2.607 | -1.19 | -0.601 | -0.245 | -26.79 | -6.57 | 1/4 |
| NY_long_excess | discovery_2018_2022 | 1279 | -0.194 | -0.12 | -0.052 | -0.031 | -15.65 | -0.49 | 2/5 |
| NY_long_excess | check_2023_on | 996 | -4.073 | -1.86 | -0.936 | -0.278 | -36.91 | -10.26 | 0/4 |
| NY_short | discovery_2018_2022 | 1279 | -2.377 | -1.42 | -0.631 | -0.191 | -31.35 | -5.99 | 1/5 |
| NY_short | check_2023_on | 996 | 1.004 | 0.46 | 0.231 | 0.154 | -16.43 | 2.53 | 3/4 |
| NY_short_excess | discovery_2018_2022 | 1279 | -2.001 | -1.19 | -0.53 | -0.176 | -28.73 | -5.04 | 0/5 |
| NY_short_excess | check_2023_on | 996 | 2.47 | 1.13 | 0.568 | 0.5 | -12.46 | 6.23 | 3/4 |
| LATE_long | discovery_2018_2022 | 1279 | -2.625 | -2.94 | -1.306 | -0.214 | -30.94 | -6.62 | 0/5 |
| LATE_long | check_2023_on | 996 | 0.242 | 0.17 | 0.086 | 0.031 | -19.64 | 0.61 | 3/4 |
| LATE_long_excess | discovery_2018_2022 | 1279 | -3.095 | -3.44 | -1.528 | -0.226 | -34.44 | -7.8 | 0/5 |
| LATE_long_excess | check_2023_on | 996 | -1.59 | -1.11 | -0.559 | -0.175 | -22.88 | -4.01 | 1/4 |
| LATE_short | discovery_2018_2022 | 1279 | 0.208 | 0.23 | 0.104 | 0.063 | -8.34 | 0.52 | 3/5 |
| LATE_short | check_2023_on | 996 | -1.85 | -1.3 | -0.655 | -0.161 | -29.03 | -4.66 | 1/4 |
| LATE_short_excess | discovery_2018_2022 | 1279 | 0.678 | 0.75 | 0.335 | 0.31 | -5.52 | 1.71 | 4/5 |
| LATE_short_excess | check_2023_on | 996 | -0.017 | -0.01 | -0.006 | -0.003 | -13.26 | -0.04 | 2/4 |
| ALWAYS_LONG | discovery_2018_2022 | 1279 | -0.164 | -0.07 | -0.03 | -0.012 | -33.14 | -0.41 | 2/5 |
| ALWAYS_LONG | check_2023_on | 996 | 6.018 | 1.54 | 0.773 | 0.471 | -32.21 | 15.16 | 3/4 |

## Cost and swap sensitivity (net bp/day / Sharpe)

| strategy | scenario | discovery_2018_2022 | check_2023_on |
|---|---|---|---|
| ASIA_long | cost x1.0 | 1.303 / 0.584 | 4.703 / 1.121 |
| ASIA_long | cost x1.5 | 0.675 / 0.303 | 4.299 / 1.025 |
| ASIA_long | cost x2.0 | 0.047 / 0.021 | 3.895 / 0.929 |
| ASIA_long | cost live31 | 0.495 / 0.222 | 4.339 / 1.035 |
| ASIA_long | swap x0.5 | 1.303 / 0.584 | 4.703 / 1.121 |
| ASIA_long_excess | cost x1.0 | 0.551 / 0.242 | 1.771 / 0.417 |
| ASIA_long_excess | cost x1.5 | -0.077 / -0.034 | 1.367 / 0.322 |
| ASIA_long_excess | cost x2.0 | -0.705 / -0.309 | 0.963 / 0.227 |
| ASIA_long_excess | cost live31 | -0.257 / -0.113 | 1.407 / 0.332 |
| ASIA_long_excess | swap x0.5 | 0.551 / 0.242 | 1.771 / 0.417 |
| ASIA_short | cost x1.0 | -3.814 / -1.713 | -6.321 / -1.507 |
| ASIA_short | cost x1.5 | -4.442 / -1.995 | -6.725 / -1.603 |
| ASIA_short | cost x2.0 | -5.07 / -2.277 | -7.129 / -1.7 |
| ASIA_short | cost live31 | -4.622 / -2.076 | -6.685 / -1.594 |
| ASIA_short | swap x0.5 | -3.814 / -1.713 | -6.321 / -1.507 |
| ASIA_short_excess | cost x1.0 | -3.062 / -1.351 | -3.389 / -0.798 |
| ASIA_short_excess | cost x1.5 | -3.69 / -1.629 | -3.793 / -0.894 |
| ASIA_short_excess | cost x2.0 | -4.318 / -1.907 | -4.197 / -0.989 |
| ASIA_short_excess | cost live31 | -3.87 / -1.707 | -3.753 / -0.884 |
| ASIA_short_excess | swap x0.5 | -3.062 / -1.351 | -3.389 / -0.798 |
| LONDON_long | cost x1.0 | -1.409 / -0.51 | 2.783 / 0.785 |
| LONDON_long | cost x1.5 | -1.971 / -0.714 | 2.382 / 0.672 |
| LONDON_long | cost x2.0 | -2.533 / -0.918 | 1.981 / 0.559 |
| LONDON_long | cost live31 | -2.292 / -0.829 | 2.412 / 0.68 |
| LONDON_long | swap x0.5 | -1.409 / -0.51 | 2.783 / 0.785 |
| LONDON_long_excess | cost x1.0 | -1.973 / -0.709 | 0.584 / 0.163 |
| LONDON_long_excess | cost x1.5 | -2.535 / -0.912 | 0.183 / 0.051 |
| LONDON_long_excess | cost x2.0 | -3.097 / -1.115 | -0.218 / -0.061 |
| LONDON_long_excess | cost live31 | -2.856 / -1.027 | 0.213 / 0.059 |
| LONDON_long_excess | swap x0.5 | -1.973 / -0.709 | 0.584 / 0.163 |
| LONDON_short | cost x1.0 | -0.84 / -0.303 | -4.386 / -1.237 |
| LONDON_short | cost x1.5 | -1.402 / -0.505 | -4.786 / -1.35 |
| LONDON_short | cost x2.0 | -1.965 / -0.707 | -5.187 / -1.464 |
| LONDON_short | cost live31 | -1.723 / -0.621 | -4.756 / -1.342 |
| LONDON_short | swap x0.5 | -0.84 / -0.303 | -4.386 / -1.237 |
| LONDON_short_excess | cost x1.0 | -0.276 / -0.099 | -2.186 / -0.611 |
| LONDON_short_excess | cost x1.5 | -0.838 / -0.3 | -2.587 / -0.723 |
| LONDON_short_excess | cost x2.0 | -1.401 / -0.5 | -2.988 / -0.835 |
| LONDON_short_excess | cost live31 | -1.159 / -0.415 | -2.557 / -0.715 |
| LONDON_short_excess | swap x0.5 | -0.276 / -0.099 | -2.186 / -0.611 |
| NY_long | cost x1.0 | 0.182 / 0.048 | -2.607 / -0.601 |
| NY_long | cost x1.5 | -0.367 / -0.098 | -3.008 / -0.693 |
| NY_long | cost x2.0 | -0.916 / -0.244 | -3.409 / -0.786 |
| NY_long | cost live31 | -0.717 / -0.191 | -2.978 / -0.686 |
| NY_long | swap x0.5 | 0.182 / 0.048 | -2.607 / -0.601 |
| NY_long_excess | cost x1.0 | -0.194 / -0.052 | -4.073 / -0.936 |
| NY_long_excess | cost x1.5 | -0.743 / -0.197 | -4.474 / -1.028 |
| NY_long_excess | cost x2.0 | -1.292 / -0.342 | -4.875 / -1.121 |
| NY_long_excess | cost live31 | -1.093 / -0.29 | -4.444 / -1.022 |
| NY_long_excess | swap x0.5 | -0.194 / -0.052 | -4.073 / -0.936 |
| NY_short | cost x1.0 | -2.377 / -0.631 | 1.004 / 0.231 |
| NY_short | cost x1.5 | -2.925 / -0.777 | 0.604 / 0.139 |
| NY_short | cost x2.0 | -3.474 / -0.923 | 0.203 / 0.047 |
| NY_short | cost live31 | -3.275 / -0.87 | 0.634 / 0.146 |
| NY_short | swap x0.5 | -2.377 / -0.631 | 1.004 / 0.231 |
| NY_short_excess | cost x1.0 | -2.001 / -0.53 | 2.47 / 0.568 |
| NY_short_excess | cost x1.5 | -2.549 / -0.675 | 2.07 / 0.476 |
| NY_short_excess | cost x2.0 | -3.098 / -0.82 | 1.669 / 0.384 |
| NY_short_excess | cost live31 | -2.899 / -0.768 | 2.1 / 0.483 |
| NY_short_excess | swap x0.5 | -2.001 / -0.53 | 2.47 / 0.568 |
| LATE_long | cost x1.0 | -2.625 / -1.306 | 0.242 / 0.086 |
| LATE_long | cost x1.5 | -3.23 / -1.606 | -0.159 / -0.056 |
| LATE_long | cost x2.0 | -3.834 / -1.905 | -0.561 / -0.199 |
| LATE_long | cost live31 | -3.461 / -1.722 | -0.126 / -0.045 |
| LATE_long | swap x0.5 | -2.625 / -1.306 | 0.242 / 0.086 |
| LATE_long_excess | cost x1.0 | -3.095 / -1.528 | -1.59 / -0.559 |
| LATE_long_excess | cost x1.5 | -3.7 / -1.826 | -1.992 / -0.7 |
| LATE_long_excess | cost x2.0 | -4.304 / -2.122 | -2.394 / -0.842 |
| LATE_long_excess | cost live31 | -3.931 / -1.941 | -1.959 / -0.689 |
| LATE_long_excess | swap x0.5 | -3.095 / -1.528 | -1.59 / -0.559 |
| LATE_short | cost x1.0 | 0.208 / 0.104 | -1.85 / -0.655 |
| LATE_short | cost x1.5 | -0.396 / -0.197 | -2.251 / -0.797 |
| LATE_short | cost x2.0 | -1.001 / -0.498 | -2.653 / -0.94 |
| LATE_short | cost live31 | -0.628 / -0.313 | -2.218 / -0.785 |
| LATE_short | swap x0.5 | 0.208 / 0.104 | -1.85 / -0.655 |
| LATE_short_excess | cost x1.0 | 0.678 / 0.335 | -0.017 / -0.006 |
| LATE_short_excess | cost x1.5 | 0.074 / 0.036 | -0.419 / -0.147 |
| LATE_short_excess | cost x2.0 | -0.531 / -0.262 | -0.821 / -0.288 |
| LATE_short_excess | cost live31 | -0.158 / -0.078 | -0.385 / -0.135 |
| LATE_short_excess | swap x0.5 | 0.678 / 0.335 | -0.017 / -0.006 |
| ALWAYS_LONG | cost x1.0 | -0.164 / -0.03 | 6.018 / 0.773 |
| ALWAYS_LONG | cost x1.5 | -0.164 / -0.03 | 6.018 / 0.773 |
| ALWAYS_LONG | cost x2.0 | -0.164 / -0.03 | 6.018 / 0.773 |
| ALWAYS_LONG | cost live31 | -0.164 / -0.03 | 6.018 / 0.773 |
| ALWAYS_LONG | swap x0.5 | 1.198 / 0.215 | 7.378 / 0.947 |

Sessions (server hours): {'ASIA': (1, 9), 'LONDON': (9, 15), 'NY': (15, 19), 'LATE': (19, 24)}; trials 8; swap long per night (points) -78.59 = -7.09%/yr of price, applied as a rate in every year.