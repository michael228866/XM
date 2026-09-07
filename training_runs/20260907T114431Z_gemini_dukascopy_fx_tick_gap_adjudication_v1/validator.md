# Independent DATA-only validator

Overall: **PASS**

Internal methodology: **PASS**

Data certification: **FAIL**

Foundation ready: **NO**

| Check | Verdict | Evidence |
|---|---|---|
| old failed runs byte-identical | PASS | ['20260906T104638Z_gemini_usd_fx_pressure_foundation_v1', '20260906T180930Z_gemini_usd_fx_pressure_source_reconciliation_v1'] |
| exact 554 input gap universe | PASS | {'intervals': 554, 'minutes': 703331} |
| Dukascopy-only sourcing | PASS | {'requests': 12823} |
| repeated acquisition evidence | PASS | {'pair_hours': 12823, 'errors': []} |
| BID tick timestamps and aggregation | PASS | UTC floor-minute OHLC |
| native-M1 equivalence recomputed | PASS | {'gate': 'FAIL', 'native_overlap_expected': 57365, 'compared_bars': 57365, 'mismatch_bars': 1, 'mismatch_rate': 1.743223219733287e-05} |
| native-M1 equivalence gate | FAIL | {'gate': 'FAIL', 'native_overlap_expected': 57365, 'compared_bars': 57365, 'mismatch_bars': 1, 'mismatch_rate': 1.743223219733287e-05} |
| gap classifications reproduce raw ticks | PASS | {'verified_no_tick_interval': {'intervals': 530, 'minutes': 653215}, 'native_m1_missing_but_ticks_present': {'intervals': 0, 'minutes': 0}, 'retrieval_error': {'intervals': 229, 'minutes': 43290}, 'inconsistent_provider_history': {'intervals': 8, 'minutes': 6826}} |
| Dukascopy tick retrieval | FAIL | {'verified_no_tick_hour': 10856, 'ticks': 1184, 'retrieval_error': 783} |
| six GOLD timestamp hashes | PASS | {'fold1_train': '6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6', 'fold1_score': '47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94', 'fold2_train': '691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44', 'fold2_score': '1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8', 'fold3_train': '3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b', 'fold3_score': '1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449'} |
| exact affected GOLD timestamps and horizons | PASS | {'records': 36697, 'unique_exact_rows': 8476} |
| exact 8476-row reconciliation | PASS | {'previous': 8476, 'verified_no_tick': 1545, 'tick_reconstruction': 0, 'remaining': 6931} |
| same-provider repaired source reconstruction | PASS | ['EUR/USD', 'GBP/USD', 'USD/JPY'] |
| 5-minute causal matrix reconstruction | PASS | {'matrix_exists': False} |
| remaining unresolved rows are zero | FAIL | 6931 |
| one Dukascopy provider | PASS | 1 |
| no model or outcome access | PASS | [] |
| production unchanged | PASS | {'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07', 'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2'} |

## Walk-forward-validator contract

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | UTC completed-bar causality | Applicable evidence reproduced | None |
| feature leakage | PASS | fixed causal feature reconstruction | Applicable evidence reproduced | None |
| label maturity | PASS | not applicable: no labels | Applicable evidence reproduced | None |
| OOF predictions | PASS | not applicable: no predictions | Applicable evidence reproduced | None |
| calibration | PASS | not applicable: no calibration | Applicable evidence reproduced | None |
| threshold selection | PASS | not applicable: no selection | Applicable evidence reproduced | None |
| purge/embargo | PASS | not applicable: no fitting | Applicable evidence reproduced | None |
| holdout contamination | PASS | no outcomes accessed | Applicable evidence reproduced | None |
| recent-period reuse | PASS | no strategy evidence claimed | Applicable evidence reproduced | None |
| execution alignment | PASS | not applicable: no strategy execution | Applicable evidence reproduced | None |
| cost assumptions | PASS | not applicable: no economics | Applicable evidence reproduced | None |
| multiple-testing risk | PASS | one frozen data hypothesis | Applicable evidence reproduced | None |

No OOS strategy metrics exist in this data-only run. The 60% WR target and economic guardrails are not applicable.

Submitted data-foundation claim: **invalid**.
