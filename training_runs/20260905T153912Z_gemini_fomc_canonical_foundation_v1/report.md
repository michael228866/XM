# GEMINI FOMC CANONICAL TIMESTAMP FOUNDATION V1

Status: `PASS` (data only)

Expected universe is reviewed from official FOMC meeting/policy records. No fixed annual denominator is used.

| Fold | Prior expected | Canonical | Verified | Coverage |
|---|---:|---:|---:|---:|
| 2018_2020 | 34 | 26 | 26 | 100.00% |
| 2021_2022 | 36 | 16 | 16 | 100.00% |
| 2023_2024 | 34 | 16 | 16 | 100.00% |

Annual counts: {"2018": 8, "2019": 8, "2020": 10, "2021": 8, "2022": 8, "2023": 8, "2024": 8}

prior_denominator_inflation_confirmed = True

Reconciliation categories (prior items): {"SEP_excluded": 0, "balance_sheet_companion_excluded": 4, "canonical_statement_retained": 58, "duplicate_representation": 0, "implementation_note_excluded": 32, "minutes_excluded": 0, "other_non_statement_monetary_release_excluded": 3, "strategy_document_excluded": 7, "unresolved": 0}

See official_calendar_links.csv, document_review.csv, reviewed_events.csv and retained sources for each classification and timestamp quote.

No GOLD bars, strategy labels, predictions, or performance were loaded. No new forward cutoff. No B0/B1 training authorization.

## Scope and official universe reconstruction

This is a data-only release-timestamp foundation, not a model/strategy experiment.
The official 2018, 2019 and 2020 historical pages and the 2021-2024 sections of
the official meeting calendar enumerate the decisions. The unavailable annual
historical2021-2024 URLs returned 404; their error records are retained. Their
meeting records are present in the official current calendar, so these failed
index URLs are not missing decision statements.

Sources:
- [2018 official history](https://www.federalreserve.gov/monetarypolicy/fomchistorical2018.htm)
- [2019 official history](https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm)
- [2020 official history](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm)
- [2021-2024 official meeting calendar sections](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)

All 58 actual statement titles, public-release dates/times and meeting
relationships have source-backed annotations. No annual count was imposed.
2018 and 2019 have eight scheduled statements each; 2021-2024 have eight each.
2020 has seven scheduled and three unscheduled statements (ten total).
The scheduled March 17-18, 2020 meeting was cancelled and contributes no event.
The March 2 emergency meeting's statement was released on March 3: meeting
date and release date are deliberately different.

| Unscheduled statement | Official local release | UTC | Official meeting record |
|---|---|---|---|
| [2020-03-03](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm) | 10:00 EST | 2020-03-03 15:00Z | March 2 unscheduled meeting |
| [2020-03-15](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm) | 17:00 EDT | 2020-03-15 21:00Z | March 15 unscheduled meeting |
| [2020-03-23](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200323a.htm) | 08:00 EDT | 2020-03-23 12:00Z | March 23 notation vote |

March 23 explicitly identifies an FOMC statement and a Committee monetary-policy
vote. It is included even though its decision concerns purchases/credit transmission,
rather than a newly changed target-rate range. By contrast the separate 2019-10-11
implementation statement explicitly describes purely technical measures, not a
change in monetary-policy stance. The frozen definition excludes operational
implementation releases; it is not another canonical statement.
March 19 and March 31, 2020 liquidity/FIMA facility announcements are linked as
Press Release, not Statement, in the official history and are excluded under
the same fixed scope. No exclusion is based on an a/a1/b/c suffix.

## Exact prior-denominator reconciliation

| Fold | Prior denominator | Canonical denominator | Reduction | Cause |
|---|---:|---:|---:|---|
| 2018-2020 | 34 | 26 | 8 | 3 strategy + 2 balance-sheet companions + 3 other operational/facility releases |
| 2021-2022 | 36 | 16 | 20 | 16 implementation notes + 2 strategy + 2 balance-sheet companions |
| 2023-2024 | 34 | 16 | 18 | 16 implementation notes + 2 strategy companions |

Thus 104 prior FOMC-like documents become 58 actual decisions after excluding
46 non-statement documents. The prior expected universe already contained
all 58 canonical statement URLs; zero previously undiscovered decision dates
were added. No SEP/minutes/PDF duplicates were in those original 104 URLs.

The prior retained timestamp dataset used last-write-wins deduplication by
(event_type, release_timestamp_utc). Eight canonical HTML identities were
replaced by companion-document identities with the same timestamp:
2018-01-31, 2019-01-30, 2019-03-20, 2021-01-27, 2022-01-26,
2022-05-04, 2023-02-01 and 2024-01-31.
The reconciliation calls these eight 'previously_missing_canonical_statement_recovered':
this means recovered statement identity in the retained verified dataset,
NOT eight new timestamps or newly discovered policy decisions.
The prior retained timestamp counts were 30/16/16; the first fold's 30 included
four out-of-universe dates. Canonical verified counts are 26/16/16.
A timestamp that can be parsed is not necessarily a canonical FOMC event.

For completeness the expanded document review also excludes 32 statement PDF
representations, 32 SEP HTML/PDF documents and 112 minutes HTML/PDF documents.
These are representation-level document counts, not numbers of economic events.
They do not contribute to the 46-document reduction of the previous denominator.
Supplemental minutes rows identify their meeting/reference date in document_date
and explicitly mark date_semantics; they are never used as release timestamps.
Review annotations were corrected before final independent validation to
distinguish adjacent Projection Materials and Minutes links using the actual
HTML identity and the link's own group, not broad neighboring text.

## Retention, causality and limits

284 successful official raw HTML/PDF responses are retained under sources/,
including 4 successful index documents; 4 unavailable annual-index requests
remain explicit error records. Every successful response has URL, acquisition UTC,
path and SHA-256. The expected universe is not reduced for retrieval/time ambiguity:
any such canonical event would remain in the denominator. Here all 58 have exact
official time evidence; ambiguous/missing/duplicate canonical rows are all zero.

This reconstructs historical official public release times, not a vintage
advance-announcement calendar. It does NOT prove that a future minutes_to_event
feature was knowable beforehand, especially for emergency releases. Unscheduled
events must not be retrospectively announced to a model before public release.
A future experiment must separately prove advance schedule availability for
pre-event features, causal as-of availability and GOLD broker-time alignment.
No delivery/transport latency was inferred from website release text.

All historical intervals remain development data. No GOLD bars, strategy labels,
WR/PF/PnL, model predictions or untouched-forward stream were opened by this run.
No new forward cutoff was created. The data validator's PASS is not an untouched
strategy-test PASS and not a production-promotion gate.

The prior finalized run remains immutable and its original FAIL remains preserved.
gemini.py and gold_long_recent_candidate_xgb.json retain pre-run SHA-256 identity.
No model was loaded, trained or saved. No B0/B1 rerun is authorized.

Single next action: wait for an explicit request for a separate formal macro-event
B0/B1 experiment; that experiment must have its own preregistration, causal
feature-alignment audit, run ID and validation. Stop this task after data archival.
