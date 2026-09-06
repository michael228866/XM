# Independent DATA Validator

Internal methodology: FAIL
Final untouched strategy validity: FAIL (not a performance study).

| Check | Verdict | Evidence |
|---|---|---|
| national PCE identity and retained official source bytes | PASS | 119 canonical releases verified |
| national PCE reference sequence continuity | PASS | {'missing_reference_periods': [], 'errors': [], 'method': 'Reference periods from official national release content, NOT public-release calendar month presence. February/March 2019 jointly published once.'} |
| 2019 HTML timezone conflict explicitly source-resolved | PASS | Official actual-release PDF page 1 EDT independently read; preceding official EDT announcement corroborates. HTML EST defect disclosed, not silently assumed away. |
| official universe and reference-period sequence, not release-month presence | PASS | Official sitemap universe exhaustively identity reviewed; reference periods and release timestamps separately retained; see pce_sequence_review.json |
| state/regional PCE exclusion | PASS | 2015, 2016 and 2017 state releases classified by official title; not excluded by date alone |
| six exact retained timestamp hashes and UTC conversion | PASS | Compared all six to immutable C1 provenance hashes; exact deduplicated union |
| eight-feature causal as-of construction | FAIL | Independent backward as-of join: all 3004515 rows; release <= decision; 15/60/240/1440 fixed boundaries |
| combined dataset and feature matrix hashes | PASS | CSV bytes, logical matrix dtype/shape/bytes and NPZ bytes independently SHA-256 checked |
| CPI retention-gap mathematical equivalence | PASS | Whole conservative release envelope expires before earliest required timestamp; no pre-event features; original source still incomplete |
| inherited CPI/EMPLOYMENT exact official-source inventory coverage | FAIL | ['https://www.bls.gov/news.release/archives/empsit_01092015.htm'] |
| exact PCE changed-row accounting, not whole-stream withholding | FAIL | Independently compared all eight features on exact timestamps; per-block and deduplicated union |
| unique constructability and explicit unresolved-event accounting | PASS | No blanket 530218 withholding count; CPI known neutral, canonical PCE sequence complete |
| FOMC immutable reuse including unscheduled statements | PASS | 74 unchanged FOMC records, including 3 unscheduled |
| all previous finalized runs byte-identical | PASS | 9 complete archived file inventories |
| operational code/model unchanged | PASS | {'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07', 'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2'} |
| data-only implementation, no fitting or outcome access | PASS | Manual code-path review plus forbidden operation checks; only DATE/TIME raw GOLD columns; archived macro and timestamp arrays only |
| pre-run Git and immutable executed code | FAIL | {'pre_run_git_commit': '57d47fd95b9f86445cbe3a0c43cf80be0775bc34', 'pre_run_git_dirty': False, 'head_sha': '57d47fd95b9f86445cbe3a0c43cf80be0775bc34', 'origin_main_sha': '57d47fd95b9f86445cbe3a0c43cf80be0775bc34'} |
| serialization correction leaves constructed data unchanged | PASS | Initial summary serialization failed on NumPy int64; int() output-only correction. Original script and error retained; exact matrix, timestamp and impact artifact hashes must be unchanged. |

## Walk-forward-validator applicability

| Check | Verdict | Evidence / scope |
|---|---|---|
| chronology | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| feature leakage | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| label maturity | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| OOF predictions | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| calibration | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| threshold selection | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| purge/embargo | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| holdout contamination | FAIL | No untouched strategy claim supported; historical development only. |
| recent-period reuse | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| execution alignment | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| cost assumptions | FAIL | Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes. |
| multiple-testing risk | FAIL | No untouched strategy claim supported; historical development only. |

No OOS trading metrics calculated. Submitted performance claim: none; final performance validity invalid/not established.
Correction required: resolve failed data checks; no training authorized.
