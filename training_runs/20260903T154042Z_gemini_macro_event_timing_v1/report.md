# GEMINI MACRO EVENT TIMING INFORMATION V1

Status: `fail`

## Data-readiness gate

Gate: `FAIL`
Verified events: `459`; source fetches: `534`.

Failures:

- 2018_2020/FOMC: expected=34, verified=30, coverage=88.2%, minimum=21
- 2021_2022/FOMC: expected=36, verified=16, coverage=44.4%, minimum=14
- 2023_2024/FOMC: expected=34, verified=16, coverage=47.1%, minimum=14

Training was correctly stopped before fitting B0/B1. No partially fabricated or unofficial timestamps were substituted.

## Evidence classification

All 2018-2024 folds are repeatedly inspected development evidence, not an untouched final test.
