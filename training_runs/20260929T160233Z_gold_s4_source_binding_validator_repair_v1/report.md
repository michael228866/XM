# S4 source-binding validator repair

Status: PASS

The manifest binds raw working-tree bytes, identical to the sealed discovery snapshot; v1 also incorrectly requires that raw SHA to equal the LF-normalized Git blob SHA. The discovery file has 472 CRLF and 23 LF line endings. Only CRLF-to-LF normalization differs; no obsolete source or role substitution.

Target: 20260929T131155Z_gold_s4_secondary_improvement_v1

Original FAIL retained. Research remains IMPROVEMENT_FOUND / INTERESTING. No promotion; no production change; holdout unused. The original aborted manifest remains historical; any effective PASS is a separate amendment.

{
  "overall": "PASS",
  "run_id": "20260929T131155Z_gold_s4_secondary_improvement_v1",
  "full_validation_completed": true,
  "model_training_executed": false,
  "research_search_executed": false,
  "holdout_used": false,
  "production_changed": false,
  "production_promoted": false,
  "candidate_count": 16,
  "record_count": 22,
  "research_result": "IMPROVEMENT_FOUND",
  "selected_candidate": "F5_NO_REDUNDANT_HTF",
  "failed_checks": [],
  "original_inventory_unchanged": true,
  "source_binding_schema_status": "PASS"
}

## Archive completion

Full independent revalidation PASS preceded an archive-only metadata error. The initial finalizer rejected status completed and missing non-training metadata. Original failure and manifest are retained in archive_finalize_failure_v1.json and manifest_before_archive_completion_v1.json. Completed using the normal pass finalization workflow without rerunning validation. Additional source representation and selection receipts are included.
