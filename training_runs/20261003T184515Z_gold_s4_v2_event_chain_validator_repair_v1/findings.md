# GOLD S4 v2 event-chain repair

Status: PASS

REFERENCE_PASS_PAYLOAD_WRAPPER_MISMATCH: the archived trainer hashes the unwrapped control record, but the archived validator hashes dict(status="PASS", **control). The status wrapper belongs only to reference_control.json; all original event hashes, links, sequence, timestamps and candidate transitions are valid.

Original USER training completed; 27 candidates and 56 research events exist. The repair performed no model fitting, training or candidate search. Existing metrics were independently recomputed only for validation. The original NOT_RUN console/combined result is an error-handler state, not evidence of absent research. The original candidate CSV is header-only because final validation aborted metadata completion; candidate_results.jsonl remains the authoritative complete research record.

Research: IMPROVEMENT_FOUND; selected: A_NO_LONG_HTF; gate: INTERESTING. Economics: NEGATIVE_EXPECTANCY. Historical development only; no promotion.

Original target tree remains external, sealed and untouched. Any effective PASS requires a separate external amendment after archive push.

{
  "overall": "PASS",
  "target_run_id": "20261003T172302Z_gold_s4_secondary_improvement_v2",
  "full_validation_completed": true,
  "model_training_executed_by_repair": false,
  "research_search_executed_by_repair": false,
  "holdout_used": false,
  "production_changed": false,
  "production_promoted": false,
  "candidate_count": 27,
  "pareto_frontier": [
    "A_NO_LONG_HTF",
    "A_NO_VOL_BODY",
    "A_NO_VOLATILITY"
  ],
  "selected_candidate": "A_NO_LONG_HTF",
  "research_result": "IMPROVEMENT_FOUND",
  "tie_break": [
    {
      "candidate_id": "A_NO_LONG_HTF",
      "ordered_key": [
        -0.5717916137229987,
        -0.30778255768478685,
        -0.8093582289789337,
        -0.8521940468784924,
        0.06408993365369865,
        52.34913835149044,
        "A_NO_LONG_HTF"
      ]
    }
  ],
  "failed_checks": [],
  "event_chain_integrity_status": "PASS",
  "original_tree_before": "8b9ff3ace10ff482fff8f9ce5b63f909444baaa9efbdb7fa00c356961e7f7e48",
  "original_tree_after": "8b9ff3ace10ff482fff8f9ce5b63f909444baaa9efbdb7fa00c356961e7f7e48",
  "candidate_gate": "INTERESTING",
  "selected_metrics": {
    "trades": 787,
    "wins": 450,
    "losses": 337,
    "realized_win_rate": 0.5717916137229987,
    "trades_per_day": 0.30778255768478685,
    "profit_factor": 0.8521940468784924,
    "mean_r": -0.06408993365369865,
    "pnl_r": -50.438777785460836,
    "max_drawdown_r": -52.34913835149044,
    "stress_pf": 0.8093582289789337
  },
  "selected_economic_status": "NEGATIVE_EXPECTANCY"
}
