# Independent CFTC GOLD COT DATA Validator

Overall: **PASS**

Validator independently reparsed official ZIP bytes and recomputed all observations, availability timestamps, and six matrices. It did not import the foundation script.

| Check | Verdict |
|---|---|
| official_cftc_only | PASS |
| source_hash_provenance | PASS |
| schema_audit | PASS |
| gold_contract_identity | PASS |
| observation_integrity | PASS |
| submitted_schema_mapping_identity | PASS |
| six_gold_timestamp_hashes | PASS |
| causal_availability | PASS |
| future_leak_rows_zero | PASS |
| unresolved_source_rows_zero | PASS |
| feature_order_exact | PASS |
| independent_observation_identity | PASS |
| element_by_element_matrix_identity | PASS |
| independent_per_block_hash_identity | PASS |
| independent_logical_hash_identity | PASS |
| all_5_features_finite | PASS |
| all_6_blocks_constructable | PASS |
| operational_files_unchanged | PASS |
| no_model_or_strategy_evaluation | PASS |

Independent unresolved source rows: **0**
Independent future-leak rows: **0**
