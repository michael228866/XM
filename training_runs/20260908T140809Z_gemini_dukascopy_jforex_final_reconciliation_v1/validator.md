Overall: FAIL

Independent audit of the frozen data-only acquisition.

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| feature leakage | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| label maturity | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| OOF predictions | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| calibration | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| threshold selection | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| purge/embargo | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| holdout contamination | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| recent-period reuse | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| execution alignment | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| cost assumptions | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| multiple-testing risk | PASS | Frozen historical data-only request universe; scripts and raw responses | No fitting, strategy execution, outcome selection or untouched-performance claim; nonapplicable trading checks pass for this limited scope | None |
| 783 error-hour identities | PASS | 783 | Independent raw-data check | None |
| Complete mechanism inventory | PASS | 2352 | Independent raw-data check | None |
| Raw serialized hashes | PASS | [] | Independent raw-data check | None |
| Raw serialized counts | PASS | [] | Independent raw-data check | None |
| getTicks/readTicks sequence equality | PASS | [] | Independent raw-data check | None |
| Inclusive retrieval boundaries | PASS | [] | Independent raw-data check | None |
| Half-open M1 and chronological tie handling | PASS | [] | Independent raw-data check | None |
| API success versus internal retrieval errors | FAIL | {"requested_hours_with_internal_download_errors": 9, "request_ids": ["R0223", "R0487", "R0488", "R0633", "R0634", "R0635", "R0636", "R0650", "R0663"], "reason": "Returned success/empty data does not independently prove absence after SDK internal HTTP errors."} | Independent raw-data check | Do not certify this claim; see evidence limitation |
| Six GOLD timestamp hashes | PASS | {"fold1_train": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6", "fold1_score": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94", "fold2_train": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44", "fold2_score": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8", "fold3_train": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b", "fold3_score": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449"} | Independent raw-data check | None |
| Exact 6931-row identity | PASS | 6931 | Independent raw-data check | None |
| Independent row reconciliation | PASS | {"resolved": 15, "remaining": 6916, "new": 0} | Independent raw-data check | None |
| Reported hour impact is supported | FAIL | {"reported": 721, "conservative_upper_bound": 13, "reason": "Original counter counted gap hours, not verified feature impact; bound is not an exact causal count."} | Independent raw-data check | Do not certify this claim; see evidence limitation |
| Original tolerance preserved | PASS | 1e-05 | Independent raw-data check | None |
| Mismatch A/C same-provider arbitration | PASS | {"A": {"open": 1.25433, "high": 1.25439, "low": 1.25426, "close": 1.25433}, "B": {"open": 1.25433, "high": 1.25439, "low": 1.25426, "close": 1.25428}, "C": {"open": 1.25433, "high": 1.25439, "low": 1.25426, "close": 1.25433}, "JForex_ticks": {"open": 1.25433, "high": 1.25439, "low": 1.25426, "close": 1.25428}} | Independent raw-data check | None |
| Full previous HTTP mismatch window retained | FAIL | [{"hour_utc": "2023-12-10T22:00:00+00:00", "acquisition": 1, "status": 200, "sha256": "b0016610a48667048bf56733df33e197cc5a6627029349af6ce393cba4b70003"}, {"hour_utc": "2023-12-10T22:00:00+00:00", "acquisition": 2, "status": 200, "sha256": "b0016610a48667048bf56733df33e197cc5a6627029349af6ce393cba4b70003"}] | Independent raw-data check | Do not certify this claim; see evidence limitation |
| Prior finalized archives byte-identical | PASS | {"20260907T114431Z_gemini_dukascopy_fx_tick_gap_adjudication_v1": "1718f20a7b08cd388fb8b4b9a18b9a37fbfbe5cfbbc73256682f57761efa1f37", "20260906T180930Z_gemini_usd_fx_pressure_source_reconciliation_v1": "db0abeb4958382d09a9e4600768a42e67ac13f43853a454506781c6639144a7f"} | Independent raw-data check | None |
| Production unchanged | PASS | {"gemini.py": "0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07", "gold_long_recent_candidate_xgb.json": "2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2"} | Independent raw-data check | None |

No strategy performance claim exists. Final untouched-test validity: not applicable.
Data certification FAIL. Stop this foundation path under the requested stopping rule.
Evidence is insufficient for zero-quote certification and full mismatch root-cause closure; no new data family or model is selected.
