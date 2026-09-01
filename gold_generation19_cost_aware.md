# Generation 19 - Cost-Aware Dynamic Break-Even

Status: `research_only`. No new signal family or ML architecture was introduced.

## Selection-period comparison

| Candidate | Trades | Trades/day | Realized win | TP-first | PF | Mean-R | PnL | Max DD | Break-even | Edge | Avg spread/ATR | Avg cost R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen17 repriced parent | 206 | 0.114 | 60.19% | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | 60.34% | -0.14% | 0.1565 | 0.1088 |
| gen19_dynamic_existing_portfolio | 784 | 0.434 | 51.40% | 51.02% | 0.748 | -0.1254 | -762.45 | -77.27% | 58.59% | -7.18% | 0.1775 | 0.1182 |
| gen19_dynamic_short_trend | 390 | 0.216 | 48.21% | 47.95% | 0.774 | -0.1218 | -504.78 | -53.75% | 54.59% | -6.39% | 0.2345 | 0.1555 |

## Chronological fold results

| Candidate | Fold | Margin | Trades | Trades/day | Win | PF | Mean-R | PnL | DD | Edge | Stress PF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gen17 repriced | 2018_2020 | n/a | 47 | 0.061 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% | -11.87% | 0.590 |
| gen19_dynamic_short_trend | 2018_2020 | 8.00% | 66 | 0.085 | 48.48% | 0.857 | -0.0768 | -74.91 | -15.20% | -3.87% | 0.797 |
| gen19_dynamic_existing_portfolio | 2018_2020 | 8.00% | 469 | 0.605 | 47.76% | 0.662 | -0.1812 | -707.16 | -71.63% | -10.23% | 0.626 |
| Gen17 repriced | 2021_2022 | n/a | 132 | 0.256 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% | 2.36% | 1.044 |
| gen19_dynamic_short_trend | 2021_2022 | 8.00% | 197 | 0.382 | 58.38% | 0.912 | -0.0375 | -110.28 | -21.27% | -2.23% | 0.864 |
| gen19_dynamic_existing_portfolio | 2021_2022 | 8.00% | 204 | 0.395 | 58.33% | 0.914 | -0.0366 | -111.61 | -22.20% | -2.17% | 0.867 |
| Gen17 repriced | 2023_2024 | n/a | 27 | 0.052 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% | 7.71% | 1.328 |
| gen19_dynamic_short_trend | 2023_2024 | 2.00% | 127 | 0.246 | 32.28% | 0.616 | -0.2758 | -398.32 | -41.73% | -11.36% | 0.561 |
| gen19_dynamic_existing_portfolio | 2023_2024 | 8.00% | 111 | 0.215 | 54.05% | 0.885 | -0.0529 | -86.89 | -12.88% | -3.01% | 0.854 |

## Separate paired exit-economics diagnostic

| Profile | TP/SL ATR | Trades | Win | PF | Mean-R | Break-even | Edge |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_13_16 | 1.3/1.6 | 206 | 60.19% | 0.994 | -0.0024 | 60.34% | -0.14% |
| tp15_sl16 | 1.5/1.6 | 206 | 54.85% | 0.909 | -0.0417 | 57.19% | -2.34% |
| tp16_sl16 | 1.6/1.6 | 206 | 53.88% | 0.932 | -0.0320 | 55.63% | -1.74% |
| tp13_sl15 | 1.3/1.5 | 206 | 57.77% | 0.952 | -0.0206 | 58.96% | -1.19% |

Exit profiles are diagnostic only: they were viewed on development folds and cannot support an OOS selection claim.
