# GEMINI MACRO EVENT INTEGRATION FOUNDATION V1

Data-only; no training, no strategy evaluation.

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


## Results

```json
{
  "run_id": "20260905T171629Z_gemini_macro_event_integration_foundation_v1",
  "run_status": "pending_independent_validator",
  "pce": {
    "inherited_rows": 143,
    "canonical_national_releases": 119,
    "canonical_national_in_required_history": 101,
    "state_level_excluded": 19,
    "other_noncanonical_excluded": 5,
    "missing_canonical_national_releases": [],
    "ambiguous_canonical_releases": 0,
    "exact_affected_rows": 28728,
    "history_sufficient": true
  },
  "cpi": {
    "official_timestamp_recovered": false,
    "source_retention_complete": false,
    "cpi_20160616_exact_feature_rows_affected": 0,
    "feature_impact_neutral": true,
    "classification": "proven_feature_impact_neutral",
    "possible_release_utc_envelope": [
      "2016-06-16 00:00:00+00:00",
      "2016-06-18 00:00:00+00:00"
    ],
    "maximum_effect_end_utc": "2016-06-19 00:00:00+00:00",
    "earliest_required_row_utc": "2016-06-30 21:00:00+00:00",
    "proof": "For every possible release in the conservative envelope, every required row is >1440 minutes later; clipped age=1 and all flags=0. A later verified event supersedes it. No pre-event features. Exact intersection empty; endpoint insertion regression identical.",
    "inherited_zero_byte_defect_remains": true,
    "original_source_retention_complete": false
  },
  "employment_history_sufficient": true,
  "fomc_history_sufficient": true,
  "blocks": [
    {
      "block": "fold1_train",
      "rows": 530218,
      "first_broker": "2016-07-01 00:00:00",
      "last_broker": "2017-12-29 19:55:00",
      "first_utc": "2016-06-30T21:00:00+00:00",
      "last_utc": "2017-12-29T17:55:00+00:00",
      "timestamp_sha256": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6",
      "utc_timestamp_sha256": "c58e0c0472b696ec08c39ba9085223da0dacc52d169c67a5965f123351f1d529",
      "constructable_rows": 530218
    },
    {
      "block": "fold1_score",
      "rows": 1058080,
      "first_broker": "2018-01-02 00:00:00",
      "last_broker": "2020-12-31 18:50:00",
      "first_utc": "2018-01-01T22:00:00+00:00",
      "last_utc": "2020-12-31T16:50:00+00:00",
      "timestamp_sha256": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94",
      "utc_timestamp_sha256": "6f6d1ac33002553e50f66c92b9bf813229e02ce1e43ad1c2c0fcb86850ccca0e",
      "constructable_rows": 1058080
    },
    {
      "block": "fold2_train",
      "rows": 532563,
      "first_broker": "2019-07-01 01:00:00",
      "last_broker": "2020-12-31 14:50:00",
      "first_utc": "2019-06-30T22:00:00+00:00",
      "last_utc": "2020-12-31T12:50:00+00:00",
      "timestamp_sha256": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44",
      "utc_timestamp_sha256": "d4384b492b494fd41574009134dddf086d72b5a52956dc1102c8af16e5f19fa5",
      "constructable_rows": 532563
    },
    {
      "block": "fold2_score",
      "rows": 708197,
      "first_broker": "2021-01-04 01:00:00",
      "last_broker": "2022-12-30 23:57:00",
      "first_utc": "2021-01-03T23:00:00+00:00",
      "last_utc": "2022-12-30T21:57:00+00:00",
      "timestamp_sha256": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8",
      "utc_timestamp_sha256": "baad84131fd1d5761f3222bc2805a95edba452ad9e0cc64bf5ff82906b34594a",
      "constructable_rows": 708197
    },
    {
      "block": "fold3_train",
      "rows": 533580,
      "first_broker": "2021-07-01 01:00:00",
      "last_broker": "2022-12-30 19:57:00",
      "first_utc": "2021-06-30T22:00:00+00:00",
      "last_utc": "2022-12-30T17:57:00+00:00",
      "timestamp_sha256": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b",
      "utc_timestamp_sha256": "094671c7b2f1e401c53b40f4b2f5f8fa32727f46854058366cb43570f3e7988c",
      "constructable_rows": 533580
    },
    {
      "block": "fold3_score",
      "rows": 708020,
      "first_broker": "2023-01-03 01:00:00",
      "last_broker": "2024-12-31 20:00:00",
      "first_utc": "2023-01-02T23:00:00+00:00",
      "last_utc": "2024-12-31T18:00:00+00:00",
      "timestamp_sha256": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449",
      "utc_timestamp_sha256": "2117f5f1c0226c1e4b48ade4a77004a6189d2ffa36f11fe7db896a654a69df43",
      "constructable_rows": 708020
    }
  ],
  "macro_event_dataset_sha256": "cd212f31ca0f9b8294c485e20aa863216eb859a6825540a82d40a46f81162592",
  "event_feature_matrix_sha256": "8f0bf303448bccb8ae418cd9b980447a835704731875336dddd4e0a64f74aac1",
  "event_feature_archive_sha256": "85506f203de20ecb6b34c3c91cd9a60b7befc6195197c7f5a44f36487ea9139c",
  "unique_required_rows": 3004515,
  "incomplete_event_history_rows_exact": 0,
  "causal_alignment": "PASS",
  "data_foundation_ready": false,
  "model_training_performed": false,
  "strategy_evaluation_performed": false,
  "gemini_py_changed": false,
  "operational_model_changed": false,
  "prior_runs_unchanged": true,
  "tie_policy": "Alphabetical-last family on simultaneous release; frozen prior feature convention",
  "next_action": "Wait for explicit B0/B1 authorization only if independent data validator passes; otherwise resolve documented data defects without training."
}
```

## Independent final certification

{
  "internal_methodology": "FAIL",
  "final_untouched_validity": "FAIL",
  "final_untouched_reason": "Data foundation only; no untouched strategy evaluation or performance claim.",
  "checks": [
    {
      "check": "national PCE identity and retained official source bytes",
      "verdict": "PASS",
      "evidence": "119 canonical releases verified"
    },
    {
      "check": "national PCE reference sequence continuity",
      "verdict": "PASS",
      "evidence": {
        "missing_reference_periods": [],
        "errors": [],
        "method": "Reference periods from official national release content, NOT public-release calendar month presence. February/March 2019 jointly published once."
      }
    },
    {
      "check": "2019 HTML timezone conflict explicitly source-resolved",
      "verdict": "PASS",
      "evidence": "Official actual-release PDF page 1 EDT independently read; preceding official EDT announcement corroborates. HTML EST defect disclosed, not silently assumed away."
    },
    {
      "check": "official universe and reference-period sequence, not release-month presence",
      "verdict": "PASS",
      "evidence": "Official sitemap universe exhaustively identity reviewed; reference periods and release timestamps separately retained; see pce_sequence_review.json"
    },
    {
      "check": "state/regional PCE exclusion",
      "verdict": "PASS",
      "evidence": "2015, 2016 and 2017 state releases classified by official title; not excluded by date alone"
    },
    {
      "check": "six exact retained timestamp hashes and UTC conversion",
      "verdict": "PASS",
      "evidence": "Compared all six to immutable C1 provenance hashes; exact deduplicated union"
    },
    {
      "check": "eight-feature causal as-of construction",
      "verdict": "FAIL",
      "evidence": "Independent backward as-of join: all 3004515 rows; release <= decision; 15/60/240/1440 fixed boundaries"
    },
    {
      "check": "combined dataset and feature matrix hashes",
      "verdict": "PASS",
      "evidence": "CSV bytes, logical matrix dtype/shape/bytes and NPZ bytes independently SHA-256 checked"
    },
    {
      "check": "CPI retention-gap mathematical equivalence",
      "verdict": "PASS",
      "evidence": "Whole conservative release envelope expires before earliest required timestamp; no pre-event features; original source still incomplete"
    },
    {
      "check": "inherited CPI/EMPLOYMENT exact official-source inventory coverage",
      "verdict": "FAIL",
      "evidence": [
        "https://www.bls.gov/news.release/archives/empsit_01092015.htm"
      ]
    },
    {
      "check": "exact PCE changed-row accounting, not whole-stream withholding",
      "verdict": "FAIL",
      "evidence": "Independently compared all eight features on exact timestamps; per-block and deduplicated union"
    },
    {
      "check": "unique constructability and explicit unresolved-event accounting",
      "verdict": "PASS",
      "evidence": "No blanket 530218 withholding count; CPI known neutral, canonical PCE sequence complete"
    },
    {
      "check": "FOMC immutable reuse including unscheduled statements",
      "verdict": "PASS",
      "evidence": "74 unchanged FOMC records, including 3 unscheduled"
    },
    {
      "check": "all previous finalized runs byte-identical",
      "verdict": "PASS",
      "evidence": "9 complete archived file inventories"
    },
    {
      "check": "operational code/model unchanged",
      "verdict": "PASS",
      "evidence": {
        "gemini.py": "0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07",
        "gold_long_recent_candidate_xgb.json": "2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2"
      }
    },
    {
      "check": "data-only implementation, no fitting or outcome access",
      "verdict": "PASS",
      "evidence": "Manual code-path review plus forbidden operation checks; only DATE/TIME raw GOLD columns; archived macro and timestamp arrays only"
    },
    {
      "check": "pre-run Git and immutable executed code",
      "verdict": "FAIL",
      "evidence": {
        "pre_run_git_commit": "57d47fd95b9f86445cbe3a0c43cf80be0775bc34",
        "pre_run_git_dirty": false,
        "head_sha": "57d47fd95b9f86445cbe3a0c43cf80be0775bc34",
        "origin_main_sha": "57d47fd95b9f86445cbe3a0c43cf80be0775bc34"
      }
    },
    {
      "check": "serialization correction leaves constructed data unchanged",
      "verdict": "PASS",
      "evidence": "Initial summary serialization failed on NumPy int64; int() output-only correction. Original script and error retained; exact matrix, timestamp and impact artifact hashes must be unchanged."
    }
  ]
}

MACRO EVENT B0/B1 DATA FOUNDATION READY = NO
