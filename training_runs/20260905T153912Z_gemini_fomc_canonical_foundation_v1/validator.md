# Independent FOMC data validator

Internal methodology: PASS

| Check | Verdict | Evidence |
|---|---|---|
| official-source-only provenance | PASS | All source hosts checked |
| retained raw bytes | PASS | Independent SHA-256 of every successful response |
| one event per decision / HTML PDF deduplication | PASS | 58 events; 58 verified timestamps |
| actual canonical statement identity / meeting relation | PASS | Titles, contents, meeting quotes and release evidence independently verified for every event |
| release-time evidence / timezone / DST | PASS | Official date and time parsed independently; EST/EDT arithmetic equals IANA UTC |
| ambiguous events retained in denominator | PASS | Expected and verified sets compared |
| explicit exclusions | PASS | {'minutes_excluded': 112, 'SEP_excluded': 32, 'strategy_document_excluded': 7, 'balance_sheet_companion_excluded': 4, 'other_non_statement_monetary_release_excluded': 3, 'implementation_note_excluded': 32, 'duplicate_representation': 32} |
| denominator rebuilt from official statement links | PASS | 60 official statement links reviewed including PDF companions |
| every prior item reconciled | PASS | 104 prior URLs independently enumerated |
| 95 percent gate arithmetic / denominator reconciliation | PASS | [{'fold': '2018_2020', 'expected': '26', 'verified': '26', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'first_timestamp': '2018-01-31T19:00:00+00:00', 'last_timestamp': '2020-12-16T19:00:00+00:00', 'gate_pass': 'True', 'prior_expected': '34', 'denominator_difference': '8', 'prior_exclusions': '{"SEP_excluded": 0, "balance_sheet_companion_excluded": 2, "canonical_statement_retained": 26, "duplicate_representation": 0, "implementation_note_excluded": 0, "minutes_excluded": 0, "other_non_statement_monetary_release_excluded": 3, "strategy_document_excluded": 3, "unresolved": 0}'}, {'fold': '2021_2022', 'expected': '16', 'verified': '16', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'first_timestamp': '2021-01-27T19:00:00+00:00', 'last_timestamp': '2022-12-14T19:00:00+00:00', 'gate_pass': 'True', 'prior_expected': '36', 'denominator_difference': '20', 'prior_exclusions': '{"SEP_excluded": 0, "balance_sheet_companion_excluded": 2, "canonical_statement_retained": 16, "duplicate_representation": 0, "implementation_note_excluded": 16, "minutes_excluded": 0, "other_non_statement_monetary_release_excluded": 0, "strategy_document_excluded": 2, "unresolved": 0}'}, {'fold': '2023_2024', 'expected': '16', 'verified': '16', 'coverage_pct': '100.0', 'ambiguous': '0', 'missing': '0', 'duplicates': '0', 'first_timestamp': '2023-02-01T19:00:00+00:00', 'last_timestamp': '2024-12-18T19:00:00+00:00', 'gate_pass': 'True', 'prior_expected': '34', 'denominator_difference': '18', 'prior_exclusions': '{"SEP_excluded": 0, "balance_sheet_companion_excluded": 0, "canonical_statement_retained": 16, "duplicate_representation": 0, "implementation_note_excluded": 16, "minutes_excluded": 0, "other_non_statement_monetary_release_excluded": 0, "strategy_document_excluded": 2, "unresolved": 0}'}] |
| independent multiyear and unscheduled samples | PASS | Independently checked 58 events, all years and unscheduled actions |
| prior finalized run byte identity | PASS | Every prior file hash compared; prior archive validation also run |
| operational artifacts unchanged | PASS | gemini.py and operational model pre/current SHA-256 equality |
| no training or strategy evaluation | PASS | Data-only script has no model/market loaders, fit calls, simulator or model artifacts |
| pre-run clean committed provenance | PASS | Pre-run commit and immutable executed snapshot |

No strategy-performance or untouched-test validity is claimed.
