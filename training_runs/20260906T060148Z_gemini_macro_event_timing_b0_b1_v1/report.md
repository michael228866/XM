# GEMINI MACRO EVENT TIMING B0 VS B1 V1

Status: **FAIL macro timing family; research_only**. Historical development evidence only.

## B0 reproduction gate

```json
{
  "pass": true,
  "reference": {
    "scored_rows": 2474297,
    "prediction_count": 2474297,
    "trades": 689,
    "realized_wr": 0.5660377358490566,
    "pf": 0.8247098331219882,
    "mean_r": -0.07690539315994348,
    "pnl_r": -52.98781588720106,
    "max_dd_r": -59.57958015890608,
    "cost_stress_pf": 0.7838186120438296,
    "spearman_score_realized_net_r": 0.29006366478201323,
    "top_decile_mean_r": -0.13362516057222326,
    "top_decile_pf": 0.7214634681339611,
    "top_quintile_mean_r": -0.15795575250743074,
    "top_quintile_pf": 0.6956121020778562
  },
  "reproduced": {
    "scored_rows": 2474297,
    "prediction_count": 2474297,
    "trades": 689,
    "realized_wr": 0.5660377358490566,
    "pf": 0.8247098331219882,
    "mean_r": -0.07690539315994348,
    "pnl_r": -52.98781588720106,
    "max_dd_r": -59.57958015890608,
    "cost_stress_pf": 0.7838186120438296,
    "spearman_score_realized_net_r": 0.29006366478201323,
    "top_decile_mean_r": -0.13362516057222326,
    "top_decile_pf": 0.7214634681339611,
    "top_quintile_mean_r": -0.15795575250743074,
    "top_quintile_pf": 0.6956121020778562
  },
  "differences": {
    "scored_rows": 0,
    "prediction_count": 0,
    "trades": 0,
    "realized_wr": 0.0,
    "pf": 0.0,
    "mean_r": 0.0,
    "pnl_r": 0.0,
    "max_dd_r": 0.0,
    "cost_stress_pf": 0.0,
    "spearman_score_realized_net_r": 0.0,
    "top_decile_mean_r": 0.0,
    "top_decile_pf": 0.0,
    "top_quintile_mean_r": 0.0,
    "top_quintile_pf": 0.0
  },
  "reference_probability_distribution": {
    "mean": 0.5047674962576704,
    "median": 0.494555801153183,
    "p75": 0.570514440536499,
    "p90": 0.6531402349472046,
    "p95": 0.685541009902954,
    "p99": 0.722646541595459,
    "max": 0.861997663974762
  },
  "reproduced_probability_distribution": {
    "mean": 0.5047674962576704,
    "median": 0.494555801153183,
    "p75": 0.570514440536499,
    "p90": 0.6531402349472046,
    "p95": 0.685541009902954,
    "p99": 0.722646541595459,
    "max": 0.861997663974762
  },
  "probability_differences": {
    "mean": 0.0,
    "median": 0.0,
    "p75": 0.0,
    "p90": 0.0,
    "p95": 0.0,
    "p99": 0.0,
    "max": 0.0
  },
  "tolerances": {
    "scored_rows": 0,
    "prediction_count": 0,
    "trades": 2,
    "realized_wr": 0.005,
    "pf": 0.02,
    "mean_r": 0.01,
    "spearman_score_realized_net_r": 0.005,
    "probability_stat": 0.005
  },
  "material_failures": []
}
```

## Fixed S5 executable results

| Model | Fold | Trades | Trades/day | WR | TP-first WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_technical_control | 2018_2020 | 498 | 0.4544 | 58.03% | 56.83% | 0.8667 | -0.0565 | -28.14 | -49.51 | 0.8246 |
| B0_technical_control | 2021_2022 | 24 | 0.0329 | 58.33% | 58.33% | 0.8754 | -0.0529 | -1.27 | -6.85 | 0.8292 |
| B0_technical_control | 2023_2024 | 167 | 0.2285 | 52.10% | 52.10% | 0.7090 | -0.1412 | -23.58 | -24.30 | 0.6721 |
| B0_technical_control | pooled | 689 | 0.2695 | 56.60% | 55.73% | 0.8247 | -0.0769 | -52.99 | -59.58 | 0.7838 |
| B1_technical_plus_macro | 2018_2020 | 432 | 0.3942 | 56.02% | 55.32% | 0.8129 | -0.0828 | -35.79 | -37.75 | 0.7716 |
| B1_technical_plus_macro | 2021_2022 | 14 | 0.0192 | 57.14% | 57.14% | 0.8416 | -0.0691 | -0.97 | -2.89 | 0.7992 |
| B1_technical_plus_macro | 2023_2024 | 181 | 0.2476 | 48.62% | 48.62% | 0.6271 | -0.1942 | -35.15 | -35.54 | 0.5956 |
| B1_technical_plus_macro | pooled | 627 | 0.2452 | 53.91% | 53.43% | 0.7534 | -0.1147 | -71.91 | -71.91 | 0.7153 |

## Decision

```json
{
  "b1_minus_b0": {
    "trades": -62,
    "trades_per_day": -0.02424716464606963,
    "realized_wr": -0.026962775721464882,
    "pf": -0.07128473700976745,
    "mean_r": -0.03777886798380464,
    "pnl_r": -18.91921584992901,
    "max_dd_r": -12.327451578223958,
    "cost_stress_pf": -0.06848240132000749
  },
  "fold_comparisons": [
    {
      "fold": "2018_2020",
      "spearman_improved": false,
      "spearman_delta": -0.005599222822970706,
      "top_decile_pf_improved": false,
      "top_decile_mean_r_improved": false,
      "top_quintile_pf_improved": false,
      "top_quintile_mean_r_improved": false
    },
    {
      "fold": "2021_2022",
      "spearman_improved": true,
      "spearman_delta": 1.6730724663072127e-05,
      "top_decile_pf_improved": false,
      "top_decile_mean_r_improved": false,
      "top_quintile_pf_improved": false,
      "top_quintile_mean_r_improved": false
    },
    {
      "fold": "2023_2024",
      "spearman_improved": false,
      "spearman_delta": -0.0014788125681489128,
      "top_decile_pf_improved": false,
      "top_decile_mean_r_improved": false,
      "top_quintile_pf_improved": false,
      "top_quintile_mean_r_improved": false
    }
  ],
  "folds_with_improved_spearman": 1,
  "folds_with_improved_top_decile_pf": 0,
  "folds_with_improved_top_decile_mean_r": 0,
  "folds_with_improved_top_quintile_pf": 0,
  "folds_with_improved_top_quintile_mean_r": 0,
  "pooled_spearman_delta": -0.0032630983908250033,
  "incremental_discrimination": false,
  "high_score_cohort_positive_expectancy": false,
  "b1_economic_viability": false,
  "b1_full_quality_gate": false,
  "catastrophic_folds": [
    "2018_2020",
    "2023_2024"
  ],
  "classification": "FAIL macro timing family",
  "single_next_research_hypothesis": "Test one genuinely new timestamp-aligned external information family under the same execution-aligned target.",
  "shadow_candidate_frozen": false
}
```

Marginal trades, probability/ranking diagnostics, event-state attribution and complete fold provenance are retained as CSV/JSON artifacts.

No threshold, event subset, interaction, window, parameter or calibration was searched. No production artifact was changed.

## Independent validator

Internal methodology: **PASS**. Final untouched validity: **FAIL**.

Two earlier validator attempts returned internal FAIL because the validator incorrectly expected
training-only sklearn constructor parameters to survive XGBoost raw-model reload and used an
overly strict tolerance after float32 OOF serialization. Both failed attempts are retained. Only
validator methodology was corrected; models, predictions, trades and candidate decisions were not
changed or regenerated.
