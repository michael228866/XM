# E-002 information horizon (H-002)

Outcome: **inconclusive**

weekly-horizon verdicts: {'REAL10Y_CHG_20D': 'refuted', 'COT_MM_NET_PCT_OI': 'inconclusive'}

Evidence class: development (gold prices 2016-2024 were inspected before; the weekly factor tests were not).

| signal | window | weeks | Spearman IC | IC t | long/flat Sharpe | always-long Sharpe | long/flat annual | always-long annual |
|---|---|---|---|---|---|---|---|---|
| REAL10Y_CHG_20D | discovery_2016_2022 | 364 | -0.0486 | -0.93 | 0.14 | -0.2 | 0.0131 | -0.027 |
| REAL10Y_CHG_20D | check_2023_2024 | 105 | 0.1117 | 1.14 | 0.37 | 0.61 | 0.0353 | 0.0833 |
| COT_MM_NET_PCT_OI | discovery_2016_2022 | 364 | -0.0685 | -1.31 | -0.29 | -0.2 | -0.0297 | -0.027 |
| COT_MM_NET_PCT_OI | check_2023_2024 | 105 | -0.0703 | -0.72 | 1.15 | 0.61 | 0.0902 | 0.0833 |

Costs: spread 0.00008 per change, swap -0.000298 of price per night while long (today's rate).
Trials counted in family: 1.