# Generation 21 - New Information Source Study

Status: `research_only`. No TP/SL, cost, signal-universe, execution, or model-hyperparameter change.

## Information-source availability

| Family | Status | Evidence / treatment |
|---|---|---|
| Microstructure | tested | GOLD M1 tick volume, sparse positive real volume, observed spread dynamics, and completed-bar short RV; tick path unavailable |
| Cross-market | partial; XAG tested | Complete local SILVER# M1; no aligned historical DXY/yields/VIX/WTI files |
| Economic events | unavailable | No timestamped historical calendar/release file; no dates or surprises fabricated |

## Ablation summary

| Version | Mean E(net-R) Spearman | Fold Spearman | Gain vs control | Positive/improved folds | Mean P(net+) Spearman | Brier | ECE | Info gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| A_technical_control | 0.034 | 0.019, 0.019, 0.065 | +0.000 | 3/0 | -0.068 | 0.276 | 0.153 | FAIL |
| B_microstructure | -0.048 | -0.135, -0.020, 0.012 | -0.082 | 1/0 | -0.139 | 0.274 | 0.140 | FAIL |
| C_cross_market_xag | -0.008 | -0.106, 0.053, 0.031 | -0.042 | 2/1 | -0.039 | 0.267 | 0.143 | FAIL |
| E_microstructure_plus_xag | -0.049 | -0.140, -0.029, 0.022 | -0.083 | 1/0 | -0.099 | 0.273 | 0.147 | FAIL |
| D_event_context | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT TESTABLE |
| F_all_justified | same as E | same as E | same as E | same as E | same as E | same as E | same as E | same as E |

## Usable fixed-cohort observations

Models retain NaN as missing; `complete` means every added family feature was genuinely observed/derivable.

| Fold | Version | Model-scored | Any new information | Complete all added features |
|---|---|---:|---:|---:|
| 2018_2020 | A_technical_control | 47 | 47 | 47 |
| 2018_2020 | B_microstructure | 47 | 47 | 0 |
| 2018_2020 | C_cross_market_xag | 47 | 45 | 45 |
| 2018_2020 | E_microstructure_plus_xag | 47 | 47 | 0 |
| 2021_2022 | A_technical_control | 132 | 132 | 132 |
| 2021_2022 | B_microstructure | 132 | 132 | 0 |
| 2021_2022 | C_cross_market_xag | 132 | 132 | 132 |
| 2021_2022 | E_microstructure_plus_xag | 132 | 132 | 0 |
| 2023_2024 | A_technical_control | 27 | 27 | 27 |
| 2023_2024 | B_microstructure | 27 | 27 | 0 |
| 2023_2024 | C_cross_market_xag | 27 | 27 | 27 |
| 2023_2024 | E_microstructure_plus_xag | 27 | 27 | 0 |

## Frozen executable control by fold

All ablations rank the same fixed executable cohort. Trade metrics therefore remain the control metrics until Phase 6 is allowed.

| Fold | Trades | Trades/day | Realized WR | PF | Mean-R | PnL | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | 47 | 0.061 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% |
| 2021_2022 | 132 | 0.256 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% |
| 2023_2024 | 27 | 0.052 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% |

## Candidate construction

Not executed: no new information family passed the predeclared cross-regime discrimination gate.

## Untouched forward protocol

Cutoff: `2026-09-01T02:00:00+00:00`.

All prior history remains development data. No production change was made.
