# TICKVOL Information Study

Status: **research_only_no_candidate**

No Generation 22, candidate, champion, production, TP/SL, execution, or gemini.py change was made.

## Incremental chronological OOS ranking

| Information set | Fold 1 Spearman | Fold 2 Spearman | Fold 3 Spearman | Mean Spearman | Delta vs Control | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Technical control | 0.0190 | 0.0191 | 0.0647 | 0.0342 | +0.0000 | CONTROL |
| Control + frozen H1 TV_ACCEL | -0.0449 | 0.0209 | -0.0806 | -0.0349 | -0.0691 | FAIL |
| Control + minimal TICKVOL family | -0.0490 | 0.0086 | 0.0116 | -0.0096 | -0.0439 | FAIL |
| Control + predeclared TICKVOL interactions | -0.0209 | 0.0870 | -0.0079 | 0.0194 | -0.0149 | FAIL |

### Positive-net-R ranking and calibration

| Information set | Mean P(net-R>0) Spearman | Mean Brier | Mean ECE |
|---|---:|---:|---:|
| Technical control | -0.0678 | 0.2762 | 0.1533 |
| Control + frozen H1 TV_ACCEL | -0.1092 | 0.2693 | 0.1446 |
| Control + minimal TICKVOL family | -0.0475 | 0.2611 | 0.1403 |
| Control + predeclared TICKVOL interactions | -0.0308 | 0.2608 | 0.1512 |

## Frozen H1 result

| Fold | Trades | Spearman net-R | Spearman positive net-R | Spearman TP-first |
|---|---:|---:|---:|---:|
| 2018_2020 | 47 | 0.0967 | 0.0471 | 0.0471 |
| 2021_2022 | 132 | 0.1479 | 0.0722 | 0.0722 |
| 2023_2024 | 27 | 0.1941 | 0.2320 | 0.2320 |

Mean fold Spearman: **0.1462**. Blocked permutation p-value: **0.0932**.

Strict decile monotonicity: **False**; adjacent increase fraction: **55.6%**.

## Model-free minimal ablation

| Feature | Mean fold Spearman | Positive folds | Pooled decile trend | Adjacent increases | Strict monotonic |
|---|---:|---:|---:|---:|---|
| TV_LOG_LEVEL | 0.0771 | 2/3 | 0.6727 | 66.7% | False |
| TV_VELOCITY | 0.0418 | 3/3 | 0.2848 | 33.3% | False |
| TV_ACCEL | 0.1462 | 3/3 | 0.5273 | 55.6% | False |
| TV_PCTL_60 | -0.0292 | 2/3 | -0.1515 | 44.4% | False |
| TV_PCTL_1440 | 0.0051 | 2/3 | 0.5152 | 55.6% | False |
| TV_Z_60 | -0.0294 | 2/3 | 0.0545 | 55.6% | False |
| TV_Z_1440 | 0.0409 | 1/3 | 0.5273 | 44.4% | False |
| TV_BURST_LOG_RATIO_60_1440 | 0.0092 | 1/3 | 0.5152 | 55.6% | False |
| TV_ACCEL_X_ATR_PCTL | 0.1393 | 3/3 | 0.4061 | 44.4% | False |
| TV_ACCEL_X_TREND_EFF | 0.1061 | 3/3 | 0.1636 | 55.6% | False |
| TV_ACCEL_X_RV_PCTL | 0.1580 | 3/3 | 0.5636 | 66.7% | False |

## Data audit

The file contains a mixed H1/M1 2014 segment. TICKVOL modeling starts conservatively at 2015-01-01; all three requested folds have observed, nonzero M1 TICKVOL. Large annual median shifts remain, so absolute levels are nonstationary.

| Year | File rows | Valid M1 rows | Missing | Zero | p25 | Median | p75 | p90 | p95 | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2014 | 194942 | 192200 | 0 | 0 | 9 | 19 | 37 | 63 | 91 | no |
| 2015 | 352809 | 352809 | 0 | 0 | 16 | 34 | 63 | 104 | 138 | yes |
| 2016 | 353757 | 353757 | 0 | 0 | 36 | 67 | 111 | 166 | 208 | yes |
| 2017 | 352361 | 352361 | 0 | 0 | 69 | 123 | 198 | 294 | 364 | yes |
| 2018 | 351667 | 351667 | 0 | 0 | 25 | 58 | 116 | 196 | 260 | yes |
| 2019 | 352158 | 352158 | 0 | 0 | 13 | 25 | 45 | 72 | 91 | yes |
| 2020 | 354255 | 354255 | 0 | 0 | 25 | 43 | 67 | 91 | 106 | yes |
| 2021 | 353758 | 353758 | 0 | 0 | 38 | 66 | 110 | 214 | 334 | yes |
| 2022 | 354439 | 354439 | 0 | 0 | 40 | 69 | 110 | 161 | 199 | yes |
| 2023 | 353042 | 353042 | 0 | 0 | 40 | 67 | 106 | 169 | 217 | yes |
| 2024 | 354978 | 354978 | 0 | 0 | 55 | 99 | 163 | 243 | 299 | yes |
| 2025 | 353069 | 353069 | 0 | 0 | 98 | 153 | 232 | 324 | 382 | yes |
| 2026 | 123360 | 123360 | 0 | 0 | 172 | 254 | 362 | 464 | 513 | yes |

### Structural changes

- 2014: Mixed H1 and M1 granularity; exclude the full year from modeling
- 2015: Annual median shifted materially versus prior year (1.789x)
- 2016: Annual median shifted materially versus prior year (1.971x)
- 2017: Annual median shifted materially versus prior year (1.836x)
- 2018: Annual median shifted materially versus prior year (0.472x)
- 2019: Annual median shifted materially versus prior year (0.431x)
- 2020: Annual median shifted materially versus prior year (1.720x)
- 2021: Annual median shifted materially versus prior year (1.535x)
- 2025: Annual median shifted materially versus prior year (1.545x)
- 2026: Annual median shifted materially versus prior year (1.660x)
- 2026: Partial year ending 2026-05-08

The export contains no source identifier, so activity changes cannot be conclusively separated from broker-feed changes. The discontinuities are treated as data drift, not predictive signals.

## Answers

1. Frozen H1 reproduced the three positive raw development-fold correlations exactly: [0.09666975023126735, 0.14787663120326838, 0.19413919413919414].
2. The fold-relative realized Mean-R deciles were not strictly monotonic (55.6% adjacent increases).
3. Control + H1 changed mean OOS E(net-R) Spearman by -0.0691; it improved 1/3 folds.
4. The strongest raw model-free TICKVOL feature was TV_ACCEL_X_RV_PCTL by mean fold Spearman; this ranking is diagnostic, not a selected trading feature.
5. The best predeclared information set was D_control_plus_predeclared_interactions, improving 1/3 chronological regimes.
6. The frozen cohort itself is only short trend-continuation. In the broader static-family diagnostic, the largest mean fold correlation was only 0.0200 for short_mean_reversion; the apparent fixed-cohort relationship therefore does not generalize broadly across direction/expert.
7. Model-free support for frozen H1 was insufficient; XGBoost gain was not used as acceptance evidence.
8. Generation 22 is not justified; no Generation 22 artifact or strategy candidate was created.

## Forward-data protection

Cutoff: `2026-09-01T02:00:00Z`. No post-cutoff data or outcomes were opened or scored by this study.

Generation 22 justified: **False**.
