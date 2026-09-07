# Validator correction record

The first independent validation run was preserved as
`validator_initial_fail.md` and `validator_initial_fail_stdout.log`.

The initial validator loaded `minute_open_utc` from CSV and converted the
result directly with `astype("int64")`. In the installed pandas version the
parsed timestamps retained microsecond resolution, while every frozen GOLD
timestamp and interval boundary uses nanoseconds. The resulting 1,000-fold
unit mismatch prevented verified no-tick closure intervals from matching any
GOLD timestamp.

The correction converts each parsed timestamp through `pd.Timestamp(...).value`,
which explicitly returns nanoseconds. No data source, request, gap, feature,
horizon, staleness rule, tolerance, classification, or gate was changed.

- Initial validator SHA-256: `7eb43b6f75f05447611f8b50c67a405ab9b2536edb654ede9530189c0c5e655f`
- Corrected validator SHA-256: `7a3f1bcad072811f5ca12971e045e82afe62dcf6a6537861b592e5de73a0db88`
- Initial exact-row result: 0 verified-no-tick resolutions; 8,476 remaining
- Corrected exact-row result: 1,545 verified-no-tick resolutions; 6,931 remaining
- Corrected internal methodology: PASS
- Corrected data certification: FAIL

The data conclusion remains a failure: retrieval is incomplete, one native-M1
equivalence mismatch remains, and 6,931 exact rows remain unresolved.
