# Independent audit corrections (no data/model changes)

Initial independent validator FAIL is retained in initial_validator_results.json, initial_validator.md and initial_validator_stdout.txt.

1. pandas parses the release list as datetime64[us] in this environment, whereas exact GOLD timestamps are nanoseconds. The independent merge incorrectly compared integer microseconds to integer nanoseconds. Explicit as_unit('ns') fixes the validator only. A before-release and 15/60/240/1440 boundary regression was added. The producer and frozen matrix remain unchanged.
2. empsit_01092015.htm has retained positive-byte source provenance but no inherited event timestamp. This is far before the June 2016 required feature history. The validator now proves its exact affected row count is zero using an intentionally conservative date-plus-three-days effect envelope. This discloses, rather than silently erases, the old timestamp omission. It is not a concrete problem on any exact required row. CPI has the same already documented neutral-gap proof. No event month is used as a missing-release test.
3. PowerShell's default Get-Content encoding altered one non-ASCII character in an acquisition-only annotation string when creating the supplemental script snapshot. The faulty snapshot remains archived. A new UTF-8 exact snapshot now matches the executed root file and its committed Git version byte-for-byte. The original pre-run training_script.py was never altered.

No canonical event, timestamp, feature definition, computed feature value or affected-row table was changed. Hash equality to all pre-correction constructed data is mandatory. These corrections do not use strategy outcomes, labels, predictions or tuning.
