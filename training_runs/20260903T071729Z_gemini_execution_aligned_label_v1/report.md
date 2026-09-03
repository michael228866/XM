# GEMINI EXECUTION-ALIGNED LABEL MODEL VALIDATION V1

Status: **development paired-label research**. No operational artifact was changed.

## Paired executable economics under the same S5 simulator

| Model | Fold | Trades | Trades/day | Realized WR | TP-first WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_legacy_label | 2018_2020 | 570 | 0.5201 | 30.35% | 18.07% | 0.4493 | -0.3685 | -210.02 | -211.98 | 0.3999 |
| C0_legacy_label | 2021_2022 | 195 | 0.2671 | 30.26% | 22.56% | 0.4256 | -0.4174 | -81.39 | -87.09 | 0.3847 |
| C0_legacy_label | 2023_2024 | 65 | 0.0889 | 35.38% | 29.23% | 0.6997 | -0.1983 | -12.89 | -14.84 | 0.6453 |
| C0_legacy_label | pooled | 830 | 0.3246 | 30.72% | 20.00% | 0.4624 | -0.3666 | -304.31 | -305.08 | 0.4146 |
| C1_execution_aligned_label | 2018_2020 | 498 | 0.4544 | 58.03% | 56.83% | 0.8667 | -0.0565 | -28.14 | -49.51 | 0.8246 |
| C1_execution_aligned_label | 2021_2022 | 24 | 0.0329 | 58.33% | 58.33% | 0.8754 | -0.0529 | -1.27 | -6.85 | 0.8292 |
| C1_execution_aligned_label | 2023_2024 | 167 | 0.2285 | 52.10% | 52.10% | 0.7090 | -0.1412 | -23.58 | -24.30 | 0.6721 |
| C1_execution_aligned_label | pooled | 689 | 0.2695 | 56.60% | 55.73% | 0.8247 | -0.0769 | -52.99 | -59.58 | 0.7838 |

## C1 minus C0 pooled delta

```json
{
  "trades": -141,
  "trades_per_day": -0.05514274540477121,
  "realized_wr": 0.258808820186406,
  "pf": 0.36235043163714376,
  "mean_r": 0.28972866227610417,
  "pnl_r": 251.31845012471848,
  "max_dd_r": 245.5027076984377
}
```

## Label attribution

C0 positive prevalence: 16.39%.
C1 positive prevalence: 34.07%.
Disagreement: 25.14% (401,271 rows).
Cause counts in label_comparison.csv are diagnostics and may overlap; no counterfactual label was trained.

## Probability and ranking diagnostics

- C0_legacy_label: p>=0.75=0.90%; ROC-AUC=0.4839; Brier=0.2588; Spearman(score, net-R)=-0.0930; top-decile Mean-R=-0.3338, PF=0.5093.
- C1_execution_aligned_label: p>=0.75=0.25%; ROC-AUC=0.5854; Brier=0.2474; Spearman(score, net-R)=0.2901; top-decile Mean-R=-0.1336, PF=0.7215.

## Marginal executable trades

Common exact entries: 4; C0-only: 826; C1-only: 685; changed-entry pairs: 19.
C0-only economics: {"trades": 826, "wins": 252, "losses": 574, "realized_wr": 0.3050847457627119, "average_winner_r": 1.0304720913420196, "average_loser_r": -0.9843112686216922, "payoff_ratio": 1.0468965704161504, "break_even_wr": 0.48854446993220185, "break_even_adjusted_edge": -0.18345972416948997, "pf": 0.45961312847538305, "mean_r": -0.3696315994802208, "pnl_r": -305.31570117066235, "max_dd_r": -307.100193080459, "gross_profit_r": 259.6789670181889, "gross_loss_r": 564.9946681888513}.
C1-only economics: {"trades": 685, "wins": 387, "losses": 298, "realized_wr": 0.564963503649635, "average_winner_r": 0.6389679028586083, "average_loser_r": -1.010999427692031, "payoff_ratio": 0.6320160876028207, "break_even_wr": 0.6127390579028209, "break_even_adjusted_edge": -0.047775554253185915, "pf": 0.8207725701419181, "mean_r": -0.0788281037167063, "pnl_r": -53.99725104594382, "max_dd_r": -60.58901531764887, "gross_profit_r": 247.2805784062814, "gross_loss_r": 301.27782945222526}.

## Decision

C1 economic viability: **False**.
C1 full quality floor: **False**.
C1 materially improves C0: **True**.
Shadow candidate frozen: **False**.
Classification: **execution_alignment_alone_does_not_recover_robust_alpha**.

## Single next research hypothesis (not implemented)

NEW TIMESTAMP-ALIGNED ALPHA INFORMATION: preregister one external entry-time information family and test incremental cross-regime discrimination under the frozen execution-aligned target before any further model or threshold tuning.

## Evidence limits

All folds are previously inspected development data. The paired causal attribution may be internally valid, but it is not untouched final OOS proof. Historical CSV identities are hashed; their large raw snapshots are excluded from Git and durable remote raw-data preservation is not claimed.

## Operational safety

`gemini.py` and `gold_long_recent_candidate_xgb.json` remained byte-identical.
