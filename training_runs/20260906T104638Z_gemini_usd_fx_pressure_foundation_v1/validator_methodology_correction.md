# Validator methodology correction

The first independent validation attempt incorrectly returned internal
methodology `FAIL` for two validator-only reasons:

1. It merged quote-closure intervals from all three instruments within a
   five-minute tolerance, while the frozen builder used EUR/USD intervals as
   the canonical interval boundaries after cross-pair confirmation.
2. It treated the presence of unresolved source-history rows as a methodology
   failure instead of a data-certification failure.

No source archive, timestamp universe, feature definition, feature value,
missingness mask, operational artifact, or readiness criterion was changed.
The corrected validator reconstructs the builder's exact closure boundaries,
checks missingness masks independently, and reports unresolved source rows as
a data-certification failure.

The original validator and stdout are retained as
`validator_script_initial.py` and `validator_initial_failure_stdout.log`.
The corrected independent result is:

- internal methodology: `PASS`
- data certification: `FAIL`
- data foundation ready: `NO`
- unresolved source rows: `148867`
