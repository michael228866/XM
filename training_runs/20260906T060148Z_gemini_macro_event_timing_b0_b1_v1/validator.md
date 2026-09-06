# Independent walk-forward validation

Overall: FAIL

Internal methodology: PASS

Final untouched validity: FAIL

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | fold provenance + OOF feature times | 18-month training and mature labels precede every score fold | none |
| feature leakage | PASS | certified matrix full reconstruction + six macro hashes | release<=decision and B1 is exactly B0 plus frozen matrix | none |
| label maturity | PASS | y hashes + exact maturity endpoints | same standalone S5 net-R target and strict maturity purge | none |
| OOF predictions | PASS | six fold models + OOF arrays | each fold has distinct preceding-history B0/B1 models | none |
| calibration | PASS | manifest.model.calibration_method | no calibration was fitted | none |
| threshold selection | PASS | two candidate rows + fixed manifest | 0.75 and whole eight-feature family fixed; no subset/window/parameter search | none |
| purge/embargo | PASS | latest label information per fold | later legacy/S5 maturity is strictly before score start | none |
| holdout contamination | PASS | manifest evidence classification | all folds explicitly remain development evidence | none |
| recent-period reuse | PASS | manifest evidence status | inspected former forward data is not a fresh test | none |
| execution alignment | PASS | independent S5 ledgers/metrics/exact entry sets | same S5 state machine and reconciled marginal trades | none |
| cost assumptions | PASS | trade ledger arithmetic + stress metrics | observed/fallback spread and fixed nominal/stress costs reconcile | none |
| multiple-testing risk | FAIL | historical development record | failed: single fixed paired test, but no untouched final interval exists | only a frozen future shadow interval can establish final validity |

## Independent special checks

- Foundation immutable: True
- Six timestamp hashes: True
- B0/B1 paired identity: True
- B0 reproduction: True
- S5 ledger/metrics: True
- Marginal trade identity: True
- Ranking diagnostics: True

All submitted metrics are historical development evidence. Final strategy validity remains FAIL regardless of internal metric quality.

None for internal methodology; final untouched validity requires genuinely future data after a fully qualified frozen shadow candidate.
