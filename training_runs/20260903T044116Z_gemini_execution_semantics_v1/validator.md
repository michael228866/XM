# GEMINI EXECUTION SEMANTICS RECONCILIATION V1 - independent validation

Overall: **FAIL**

Internal diagnostic methodology: **PASS**

Final untouched-test validity: **FAIL**

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | cohort_manifest.json; primary feature_bar_time < decision_time; prior W0 OOF source | The operational artifact was fit before the primary score boundary; secondary predictions are preserved chronological OOF; completed-bar times precede decisions. | none |
| feature leakage | PASS | primary_frozen_cohort.csv.gz all_features_finite; 31 shifted features | One precomputed probability vector uses completed-bar shifted features; no exit/outcome field enters scoring. | none |
| label maturity | PASS | manifest.model.trained=false; manifest.model.label_definition | No model fitting or label use occurs in this execution-only diagnostic. | none |
| OOF predictions | PASS | cohort_manifest probability_freeze; exact artifact primary; prior W0 secondary | The exact pre-test artifact scores primary once; secondary reuses byte-preserved W0 OOF scores; S0-S5 share each vector. | none |
| calibration | PASS | manifest.model.calibration_method=none | No calibrator or probability adjustment is fitted or selected. | none |
| threshold selection | PASS | manifest.search; six predeclared simulator definitions | Threshold 0.75 and every strategy parameter are fixed; simulator variants are causal definitions, not selected candidates. | none |
| purge/embargo | PASS | manifest.model.trained=false; retained legacy training code; prior W0 provenance | No fit boundary exists in this run; the existing artifact was fitted before primary and secondary was already purged OOF. | none |
| holdout contamination | PASS | manifest.evidence_status | The primary and secondary intervals are explicitly development diagnostics and support no untouched or promotion claim. | none |
| recent-period reuse | PASS | manifest.evidence_status.previous_forward_status | The prior forward interval remains contaminated and is used only for the requested simulator reconciliation. | none |
| execution alignment | PASS | simulator definitions; trade ledger; transition/trade identity/timestamp tables | Every transition is isolated, trades use deterministic episode/ordinal matching, and S1-S5 HIGH/LOW stop-first plus S5 live timing reconcile. | none |
| cost assumptions | PASS | trade_ledger.csv gross_price/spread/extra_cost/net_r/risk_budget | Fixed and observed/fallback costs reconcile trade by trade; sizing is separated from unit-R economics. | none |
| multiple-testing risk | FAIL | all primary/secondary history previously inspected; no untouched final interval | S0-S5 were predeclared and none is selected, but these development periods have been repeatedly inspected and cannot support a final strategy claim. | Freeze a future experiment before collecting genuinely new forward data; do not reuse this run for final selection. |

## Reconciliation

Metric recomputation: **PASS**.
Frozen cohort identity: **PASS**.
Simulator transition isolation: **PASS**.
Trade matching: **PASS**.
HIGH/LOW stop-first: **PASS**.
Cost arithmetic: **PASS**.
Cooldown reconstruction: **PASS**.

All six definitions are diagnostic semantics, not candidates. The submitted internal attribution is valid only for the retained development cohorts and documented S5 approximations.

## Validation conclusion

No internal rerun is required; only final multiple-testing/untouched validity fails, which historical recomputation cannot repair.

Submitted claim: `valid_internal_execution_attribution_invalid_for_final_strategy_claim`.
