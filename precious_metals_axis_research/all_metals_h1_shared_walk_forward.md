# All Metals Shared Walk-Forward

Research-only. Retrains the shared H1 model on rolling folds and stress-tests candidate parameters with 3x spread.

| Symbol | Candidate | Verdict | Stress R | Positive | Passed | Trades | Win | PF | Worst R | Max DD |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XAUEUR# | xaueur_shared_best | walk_forward_candidate | 5.53 | 2/2 | 2/2 | 21 | 76.19% | 2.69 | 2.37 | -1.87 |
| XPTUSD# | xpt_shared_best | walk_forward_candidate | 2.82 | 2/2 | 1/2 | 20 | 70.00% | 500.11 | 1.32 | -2.96 |
| GOLD# | gold_shared_best | failed_walk_forward | 0.64 | 2/2 | 0/2 | 25 | 48.00% | 1.06 | 0.10 | -5.91 |
| XPDUSD# | xpd_watch | failed_walk_forward | 0.61 | 1/2 | 1/2 | 10 | 80.00% | 1.35 | -1.59 | -1.59 |
