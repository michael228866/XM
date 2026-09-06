# Source identity and integration review

## Scope and preservation

One data-only run. Existing finalized archives and production artifacts are hash-protected. No prices, technical features, labels, predictions, returns or model binaries are loaded. DATE/TIME columns alone reconstruct the six exact retained C1 timestamp hashes. All historical data remains development history. No forward cutoff is created or changed.

## National PCE

Official BEA sitemap bytes enumerate the national release series, augmented by all inherited PCE URLs to audit contamination and title variants. All 143 retained BEA release documents were identity reviewed. State PCE, state income, and county/metropolitan income are not national PCE events. The 2015-12-01 state PCE record remains retained in the audit but is excluded by title, not by date. Every exclusion has an individual feature-impact comparison using its inherited timestamp.

The March 29, 2019 release uses the title `Personal Income, February 2019; Personal Outlays, January 2019`. Its body explicitly identifies national `Personal Income and Outlays, January 2019`, so it qualifies. The March 1 release provides December 2018 PCE, not January PCE. The April 29 release jointly publishes new February and March 2019 PCE. These are three official release events, not four. Reference-period continuity is checked from May 2016 through November 2024, not by public-release month. The May 2016 reference release is retained as the prior anchor although its June 29 12:30 UTC release precedes the minimum lookback start by 8.5 hours.

The October 31, 2019 HTML header incorrectly says EST. Its linked official full-release PDF says EDT; the preceding official August report's next-release announcement independently says EDT. Raw PDF bytes and the contradiction are retained in `pce_timezone_reconciliation.json`. Canonical UTC is 12:30, based on the release PDF, not an assumed fixed 08:30 time. Other real 10:00 releases are retained as 10:00.

## CPI

The inherited June 16, 2016 source was HTTP 200 with zero bytes. Current official BLS acquisition returned HTTP 403; the browser's official page also contained no release text. This is not source-complete. All possible times on the suspect date are bounded conservatively by June 16 00:00 UTC through June 18 00:00 UTC. Every effect expires by June 19 00:00 UTC under the frozen 1440-minute clip and 240-minute flags. Required rows begin June 30 21:00 UTC; there are no pre-event features. Exact interval intersection and insertion-equivalence tests certify zero affected rows, without claiming recovered source provenance.

## Integration semantics

The frozen implementation selects the latest qualifying release at or before decision time, using alphabetical-last family for simultaneous releases as in the retained macro feature code. All canonical simultaneous events remain in the CSV. Only the eight requested features are produced. Per-block counts may overlap across train/score blocks; the global affected-row number is a deduplicated UTC timestamp union. Individual excluded-event effects are isolated deletions from the same inherited stream, and need not sum to the joint correction.

## Methodology review before computation

The original pre-run experiment script remains byte-identical. The independent validator was extended before feature construction to accept an official body-verified national identity when the 2019 shutdown split the title, verify the reference-period sequence explicitly, corroborate the HTML/PDF timezone conflict, and check all inherited CPI/EMPLOYMENT source records. This is data-methodology review, not outcome-driven model or feature selection. The original validator snapshot is retained, with the executed revision separately frozen and committed before use.

No missing-release conclusions use public-release month presence. No whole-stream withholding count is labeled an exact affected-row count. Data readiness does not authorize training.
