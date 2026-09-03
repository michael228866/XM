# Independent walk-forward validation

Overall: FAIL

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| feature leakage | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| label maturity | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| OOF predictions | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| calibration | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| threshold selection | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| purge/embargo | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| holdout contamination | FAIL | 2018-2024 outcomes were repeatedly inspected; no untouched final interval is used | frozen paired design and readiness stop were independently verified | evaluate a completely frozen candidate on genuinely new future data |
| recent-period reuse | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| execution alignment | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| cost assumptions | PASS | data-readiness gate failed and fitting/execution were correctly not run | data-readiness gate failed and fitting/execution were correctly not run | none |
| multiple-testing risk | FAIL | 2018-2024 outcomes were repeatedly inspected; no untouched final interval is used | frozen paired design and readiness stop were independently verified | evaluate a completely frozen candidate on genuinely new future data |

## Independent conclusions

Internal methodology: `PASS`.
Final untouched validity: `FAIL`.
Data-readiness gate: `FAIL`.
The overall FAIL is retained because no untouched final interval exists; this does not overturn a correctly enforced internal data-readiness stop.

Submitted historical performance claim: `invalid as final untouched evidence`.
