# E-001 edge decomposition (H-001)

Outcome: **refuted** (H-001 = candidate edges are drift exposure)

timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK

Evidence class: reuse_contaminated (both windows were inspected in earlier work).

| candidate | window | trades | mean R | always-in R | drift part | timing residual | residual 95% CI (month bootstrap) | month-matched random >= | share positive timing years |
|---|---|---|---|---|---|---|---|---|---|
| H1_C_frozen | reliable_2018_2023 | 363 | 0.41 | -0.101 | -0.0318 | 0.4415 | [0.31, 0.5785] | 0.0 | 1.0 |
| H1_C_frozen | recent_2024_on | 274 | 0.132 | 0.053 | 0.0486 | 0.0831 | [-0.052, 0.2089] | 0.15 | 1.0 |
| COMBO_A_k2 | reliable_2018_2023 | 835 | -0.038 | -0.101 | -0.0204 | -0.0175 | [-0.0943, 0.0561] | 0.644 | 0.5 |
| COMBO_A_k2 | recent_2024_on | 552 | 0.147 | 0.053 | 0.153 | -0.0062 | [-0.0949, 0.0789] | 0.538 | 0.0 |
| EMA_BREAKOUT_20_50_20 | reliable_2018_2023 | 706 | -0.099 | -0.101 | -0.031 | -0.0684 | [-0.1454, 0.0164] | 0.902 | 0.0 |
| EMA_BREAKOUT_20_50_20 | recent_2024_on | 405 | 0.239 | 0.053 | 0.1342 | 0.1045 | [-0.0082, 0.2362] | 0.068 | 0.667 |
| H1_RSI2_L | reliable_2018_2023 | 1136 | -0.12 | -0.101 | -0.0446 | -0.0758 | [-0.1438, -0.0064] | 0.976 | 0.167 |
| H1_RSI2_L | recent_2024_on | 614 | 0.103 | 0.053 | 0.1277 | -0.0242 | [-0.116, 0.0621] | 0.7 | 0.667 |
| RANDOM_2PCT | reliable_2018_2023 | 547 | -0.157 | -0.101 | -0.1222 | -0.0351 | [-0.1244, 0.0584] | 0.748 | 0.333 |
| RANDOM_2PCT | recent_2024_on | 270 | 0.171 | 0.053 | 0.0609 | 0.1098 | [-0.0323, 0.2662] | 0.13 | 1.0 |

Trials counted in family for Sidak adjustment: 1. Swap per H1 bar (price): -0.0478.