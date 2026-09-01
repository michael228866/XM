# Generation 20 - Direct Net-Edge Learning

Status: `research_only`. Phase 1-5 use only the fixed Gen17 executable cohort.

## Fixed-cohort selection summary

| Policy | Trades | Retention | Trades/day | Realized WR | TP-first | PF | Mean-R | PnL | Max DD | Break-even edge | Stress PF | Phase 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gen17 observed-cost parent | 206 | 100.00% | 0.114 | 60.19% | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | -0.14% | 0.943 | parent |
| e_net_top75_past | 206 | 100.00% | 0.114 | 60.19% | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | -0.14% | 0.943 | FAIL |
| p_net_ge_050 | 96 | 46.60% | 0.053 | 56.25% | 56.25% | 0.876 | -0.0553 | -77.98 | -18.22% | -3.23% | 0.833 | FAIL |
| e_net_positive | 38 | 18.45% | 0.021 | 52.63% | 52.63% | 0.765 | -0.1137 | -61.36 | -12.77% | -6.60% | 0.725 | FAIL |
| joint_positive | 34 | 16.50% | 0.019 | 50.00% | 50.00% | 0.691 | -0.1575 | -74.63 | -12.27% | -9.12% | 0.654 | FAIL |

## Chronological folds

| Policy | Fold | Trades | Trades/day | WR | PF | Mean-R | Edge | Losers removed | Winners removed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_net_ge_050 | 2018_2020 | 39 | 0.050 | 48.72% | 0.628 | -0.1944 | -11.50% | 4 | 4 |
| e_net_positive | 2018_2020 | 13 | 0.017 | 53.85% | 0.810 | -0.0894 | -5.18% | 18 | 16 |
| joint_positive | 2018_2020 | 12 | 0.015 | 50.00% | 0.688 | -0.1592 | -9.26% | 18 | 17 |
| e_net_top75_past | 2018_2020 | 47 | 0.061 | 48.94% | 0.618 | -0.1987 | -11.87% | 0 | 0 |
| p_net_ge_050 | 2021_2022 | 36 | 0.070 | 55.56% | 0.856 | -0.0654 | -3.80% | 33 | 63 |
| e_net_positive | 2021_2022 | 20 | 0.039 | 50.00% | 0.675 | -0.1659 | -9.68% | 39 | 73 |
| joint_positive | 2021_2022 | 17 | 0.033 | 47.06% | 0.608 | -0.2120 | -12.31% | 40 | 75 |
| e_net_top75_past | 2021_2022 | 132 | 0.256 | 62.88% | 1.105 | 0.0398 | 2.36% | 0 | 0 |
| p_net_ge_050 | 2023_2024 | 21 | 0.041 | 71.43% | 1.758 | 0.2205 | 12.72% | 3 | 3 |
| e_net_positive | 2023_2024 | 5 | 0.010 | 60.00% | 1.078 | 0.0319 | 1.82% | 7 | 15 |
| joint_positive | 2023_2024 | 5 | 0.010 | 60.00% | 1.078 | 0.0319 | 1.82% | 7 | 15 |
| e_net_top75_past | 2023_2024 | 27 | 0.052 | 66.67% | 1.392 | 0.1330 | 7.71% | 0 | 0 |

Phase 6 executed: False.

All 2025 and recent results are development diagnostics, not untouched OOS evidence.
