# Independent data validator

Overall: FAIL

Internal methodology: FAIL (final adversarial integration review; automated arithmetic PASS retained in validator_automated.json)

Required correction: do not equate release-month gaps with missing releases, or whole-stream uncertified rows with exact impacted rows. See source_review_notes.md. FOMC-specific source validation remains PASS. The confirmed state-level PCE contamination prevents readiness independently of this methodology issue.

| Check | Verdict | Evidence |
|---|---|---|
| official source and raw byte provenance | PASS | 21 |
| unchanged canonical definition | PASS | ['classification', 'denominator', 'event', 'exclusions', 'minimum_coverage_each_fold', 'review_protocol', 'strategy_evaluation_authorized', 'timezone', 'training_authorized'] |
| all-event identity / meeting / release-time / DST audit | PASS | ['FOMC_2016-01-27_scheduled', 'FOMC_2016-03-16_scheduled', 'FOMC_2016-04-27_scheduled', 'FOMC_2016-06-15_scheduled', 'FOMC_2016-07-27_scheduled', 'FOMC_2016-09-21_scheduled', 'FOMC_2016-11-02_scheduled', 'FOMC_2016-12-14_scheduled', 'FOMC_2017-02-01_scheduled', 'FOMC_2017-03-15_scheduled', 'FOMC_2017-05-03_scheduled', 'FOMC_2017-06-14_scheduled', 'FOMC_2017-07-26_scheduled', 'FOMC_2017-09-20_scheduled', 'FOMC_2017-11-01_scheduled', 'FOMC_2017-12-13_scheduled'] |
| scheduled and unscheduled universe / exclusions | PASS | {'official_statement_links': 16, 'canonical': 16} |
| no duplicate identities or verified timestamps | PASS | 16 |
| 95 percent coverage arithmetic | PASS | [{'scope': '2016', 'expected': '8', 'verified': '8', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'gate_pass': 'True'}, {'scope': '2017', 'expected': '8', 'verified': '8', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'gate_pass': 'True'}, {'scope': '2016_2017', 'expected': '16', 'verified': '16', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'gate_pass': 'True'}] |
| every exact required training timestamp / timezone | PASS | {'rows': 530218, 'hash': '6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6'} |
| CPI readiness evidence accounting | PASS | {'sufficient': False, 'missing_months': ['2016-06'], 'ambiguous_sources': []} |
| EMPLOYMENT readiness evidence accounting | PASS | {'sufficient': True, 'missing_months': [], 'ambiguous_sources': []} |
| PCE readiness evidence accounting | PASS | {'sufficient': False, 'missing_months': ['2016-07', '2017-02', '2017-04', '2017-07'], 'ambiguous_sources': ['https://www.bea.gov/news/2015/personal-consumption-expenditures-state-1997-2014', 'https://www.bea.gov/news/2016/personal-consumption-expenditures-state-2015', 'https://www.bea.gov/news/2017/personal-consumption-expenditures-state-2016']} |
| FOMC readiness evidence accounting | PASS | {'sufficient': True, 'missing_months': [], 'ambiguous_sources': []} |
| missing history remains uncertified / never zero event features | PASS | {'uncertified_rows': 530218, 'method': 'conservative full-stream certificate, not exact missing-event impact attribution'} |
| existing finalized runs unchanged | PASS | ['20260903T154042Z_gemini_macro_event_timing_v1', '20260905T153912Z_gemini_fomc_canonical_foundation_v1'] |
| operational artifacts unchanged | PASS | {'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07', 'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2'} |
| no models / labels / strategy performance | PASS | DATE/TIME columns only; no label or model pipeline |

Every required row constructable: FAIL
No strategy-validity claim. Do not train until all data gaps are independently resolved.
