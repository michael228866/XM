# GEMINI EXECUTION-ALIGNED LABEL MODEL VALIDATION V1 - independent validation

Overall: **FAIL**

Internal methodology: **PASS**

Final untouched-test validity: **FAIL**

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | fold_model_provenance.json; paired_oof_predictions.npz | Every fold uses trailing 18 months and latest C0/C1 label information is strictly earlier than score start; feature bars precede decisions. | none |
| feature leakage | PASS | 31-feature list/hashes; dependency SHA-256 inventory | Both models use the same shifted 31-feature matrices and producing dependencies match their executed hashes. | none |
| label maturity | PASS | C0 240-row maturity; C1 stored exit maturity; strict fold assertions | The common training cohort admits a row only after both labels mature strictly before scoring; C1 target equals net R > 0. | none |
| OOF predictions | PASS | six retained fold models; paired OOF score artifact | Each scored fold has separate C0/C1 models trained solely on the preceding purged 18-month cohort. | none |
| calibration | PASS | manifest.model.calibration_method | No calibration is fit, selected, or applied. | none |
| threshold selection | PASS | manifest.paired_design/search; model_comparison.csv | Threshold 0.75, one C1 label, parameters, folds, features, and window are preregistered and never searched. | none |
| purge/embargo | PASS | latest C0/C1 information times per fold | The shared purge uses the later of C0 and C1 maturity; no label interval reaches score start. | none |
| holdout contamination | PASS | manifest.evidence_status | All historical folds are explicitly development evidence and no holdout claim is made. | none |
| recent-period reuse | PASS | manifest.evidence_status.previous_forward_status | The inspected former forward interval remains contaminated and is not used as a fresh final test. | none |
| execution alignment | PASS | paired OOF OHLC cohort; exact preserved S5; independently reconstructed trade ledger | C0 and C1 traverse the same S5 barwise state machine with next-open entry, entry-bar HIGH/LOW stop-first, timeout, occupancy, sessions, and risk state. | none |
| cost assumptions | PASS | trade_ledger.csv gross/spread/extra-cost/net-R/stress-R | Observed/fallback spread and both nominal and stress costs reconcile per trade. | none |
| multiple-testing risk | FAIL | manifest.evidence_status; repository historical research record | The experiment is a single preregistered label hypothesis, but every historical fold has been inspected previously and no untouched final test exists. | Freeze a qualified candidate first, then collect genuinely new post-freeze shadow data; do not relabel historical data. |

## Independent reconciliation

Paired fold/data provenance: **PASS**.
OOF cohort identity: **PASS**.
S5 trade-ledger reconstruction: **PASS**.
Metric recomputation: **PASS**.
Cost arithmetic: **PASS**.
Operational artifacts unchanged: **True**.

The paired historical hypothesis is internally interpretable, but all scored folds are development data and cannot establish final untouched validity.

## Smallest validation-only correction

No internal rerun is required; final validity requires new data collected only after a completely frozen candidate cutoff.

Submitted claim: `valid_internal_paired_label_attribution_invalid_for_final_strategy_claim`.
