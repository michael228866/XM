# GEMINI CORE GATE OPTIMIZATION V1 — adversarial validation

Overall: **FAIL**

Internal chronological validation quality: **PASS**
Final untouched-test validity: **FAIL**

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | oof_model_provenance.json folds=3; training_script.py:340 | Every fold model ends before its scored fold and its latest training label bar is also earlier. | none |
| feature leakage | PASS | drl_trading_v2.py:138; manifest.model.features=31 | The frozen incumbent feature family is shifted one completed bar; no outcome or exit field is an input. | none |
| label maturity | PASS | manifest.data.purge_details; oof_model_provenance.json latest_training_label_bar | A full 240-source-row purge separates model labels from each scoring fold. | none |
| OOF predictions | PASS | oof_predictions.npz rows=2474297; fold replicas=3 | All retained scores come from a model fitted only before that score's fold. | none |
| calibration | PASS | manifest.model.calibration_method=none | No calibration layer is fitted, selected, or claimed. | none |
| threshold selection | PASS | manifest.search.preregistered_at_utc; candidates.csv rows=80 | Exactly the pre-registered 5×4 development search was evaluated and no candidate was selected. | none |
| purge/embargo | PASS | training_script.py:340; label_horizon_rows=240 | Purged labels end before each scored fold; no duplicated scoring timestamps were found. | none |
| holdout contamination | PASS | manifest.evidence_status.claim=development_chronological_oof_only | 2018–2024 is explicitly development evidence; no historical holdout or untouched-test claim is made. | none |
| recent-period reuse | PASS | manifest.data.data_end_utc=2024-12-31T20:00:00 | No post-2025 entry or post-2026-09-01 outcome was used; the old cutoff remains marked contaminated. | none |
| execution alignment | PASS | training_script.py:451,523; trade_ledger.csv.gz | Rising-edge episodes, one-position occupancy, stop-first first touch, timeout, and fold metrics reconcile. | none |
| cost assumptions | PASS | training_script.py:384; nominal extra=5 points; stress extra=10 points | Observed positive spread or a 30-point fallback is applied once, and both nominal/stress rewards recompute exactly. | none |
| multiple-testing risk | FAIL | README.md documents Gen1–21 and repeated use of 2018–2024; no untouched final test exists. | The 20 choices were pre-registered, but the historical regimes were repeatedly inspected in prior research. This cannot support promotion. | Only a completely frozen candidate evaluated on subsequently collected untouched forward data can clear this check. |

## Metric reconciliation

All 20 candidates × 3 folds plus pooled rows were independently recomputed from the retained executable trade ledger.

| Control interval | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | 189 | 0.1724 | 25.40% | 0.372 | -0.4460 | -84.29 | -88.42 | 0.332 |
| 2021_2022 | 0 | 0.0000 | 0.00% | 0.000 | 0.0000 | 0.00 | 0.00 | 0.000 |
| 2023_2024 | 0 | 0.0000 | 0.00% | 0.000 | 0.0000 | 0.00 | 0.00 | 0.000 |
| pooled | 189 | 0.0739 | 25.40% | 0.372 | -0.4460 | -84.29 | -88.42 | 0.332 |

Arithmetic reconciliation: **PASS**.

No holdout or recent interval is present or claimed. The 60% target and economic guardrails were not met by any configuration.

## Validation conclusion

None can rescue this run: all candidates fail economics. A future untouched test would be relevant only after a separate development run freezes a passing candidate.

Submitted performance claim: `valid_development_failure_invalid_for_promotion`.
