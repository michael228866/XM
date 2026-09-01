# Generation 19 - Transaction-cost audit

All outputs remain `research_only`. Historical zero spreads are treated as missing and fall back to 30 points.

## Unit and execution audit

| Item | Result |
|---|---|
| GOLD# chart mode | bid |
| Digits / point / tick size | 2 / 0.01 / 0.01 |
| Fixed spread assumption | 30 points = 0.30 price |
| Extra cost assumption | 5 points = 0.05 price |
| Cost double-counted | No: spread is deducted once from price PnL; the denominator only expresses R in total stop cash-risk units. |
| Commission/slippage/swap | 5 points is an opaque extra-cost allowance; recorded commission/swap/fee are reported below. |

## Historical spread coverage

| Year | Rows | Zero/missing | Median positive | p90 | p95 | Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 194942 | 100.00% | n/a | n/a | n/a | n/a |
| 2015 | 352809 | 100.00% | n/a | n/a | n/a | n/a |
| 2016 | 353757 | 100.00% | n/a | n/a | n/a | n/a |
| 2017 | 352361 | 100.00% | n/a | n/a | n/a | n/a |
| 2018 | 351667 | 78.07% | 22.0 | 26.0 | 28.0 | 22.465777224577646 |
| 2019 | 352158 | 0.00% | 22.0 | 26.0 | 27.0 | 21.91433739695413 |
| 2020 | 354255 | 0.00% | 23.0 | 57.0 | 101.0 | 34.77253608992418 |
| 2021 | 353758 | 0.01% | 25.0 | 26.0 | 26.0 | 23.788708026844727 |
| 2022 | 354439 | 0.00% | 18.0 | 25.0 | 28.0 | 18.023157722485394 |
| 2023 | 353042 | 0.00% | 13.0 | 20.0 | 22.0 | 14.620515406098992 |
| 2024 | 354978 | 0.00% | 18.0 | 21.0 | 23.0 | 18.564023685974906 |
| 2025 | 353069 | 0.00% | 17.0 | 20.0 | 21.0 | 18.001379333784616 |
| 2026 | 123360 | 0.00% | 20.0 | 29.0 | 31.0 | 22.853226329442283 |

## Gen17 short trend-continuation repricing

| Cost method | Trades | Win rate | PF | Mean-R | Sum-R | Avg cost R |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 30 + 5 points | 206 | 60.19% | 0.8886 | -0.045211 | n/a | 0.141944 |
| Observed spread, zero fallback 30, +5 | 206 | 60.19% | 0.9941 | -0.002393 | -0.492883 | 0.108792 |

The fixed 0.1419R drag is internally correct for the fixed-cost formula, but it is not the observed historical average. Repricing improves Gen17 materially while remaining slightly negative; it does not create a champion by itself.
