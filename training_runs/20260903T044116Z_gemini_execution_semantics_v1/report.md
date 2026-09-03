# GEMINI EXECUTION SEMANTICS RECONCILIATION V1

Status: **research_only diagnostic**. No model was trained, no strategy candidate was selected, and no operational artifact changed.

## Primary exact-artifact frozen cohort

Rows: 83,911; SHA-256: `238ccc09196dfd3cfd9a70a003afd74245904f2e6eab2dc8f227373b36b82303`.

| Simulator | Trades | Trades/day | WR | PF sized | Mean-R | PnL-R | Max DD-R | TP | SL | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 15 | 0.1761 | 66.67% | 1.3447 | 0.1693 | 2.54 | -4.57 | 9 | 5 | 1 |
| S1 | 16 | 0.1878 | 62.50% | 1.1125 | 0.0488 | 0.78 | -3.21 | 9 | 6 | 1 |
| S2 | 17 | 0.1995 | 64.71% | 1.2334 | 0.0906 | 1.54 | -3.02 | 10 | 6 | 1 |
| S3 | 16 | 0.1878 | 62.50% | 1.4825 | 0.1611 | 2.58 | -2.02 | 10 | 5 | 1 |
| S4 | 16 | 0.1878 | 56.25% | 1.1147 | 0.0496 | 0.79 | -3.27 | 9 | 6 | 1 |
| S5 | 17 | 0.1995 | 52.94% | 0.9271 | -0.0266 | -0.45 | -3.23 | 9 | 7 | 1 |

## Secondary W0 historical-replica cohort

This cohort is reported separately and is never pooled with the exact-artifact primary cohort.

| Simulator | Trades | Trades/day | WR | PF sized | Mean-R | PnL-R | Max DD-R | TP | SL | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 194 | 0.1772 | 32.47% | 0.4746 | -0.3677 | -71.34 | -76.78 | 39 | 103 | 52 |
| S1 | 203 | 0.1854 | 31.03% | 0.4504 | -0.3511 | -71.28 | -75.48 | 42 | 114 | 47 |
| S2 | 200 | 0.1827 | 30.00% | 0.4561 | -0.3521 | -70.42 | -74.62 | 43 | 112 | 45 |
| S3 | 200 | 0.1827 | 30.50% | 0.4861 | -0.3186 | -63.73 | -68.69 | 43 | 112 | 45 |
| S4 | 237 | 0.2165 | 29.96% | 0.4850 | -0.3490 | -82.72 | -89.25 | 46 | 136 | 55 |
| S5 | 240 | 0.2192 | 30.00% | 0.4726 | -0.3417 | -82.01 | -88.16 | 48 | 138 | 54 |

## S0 reproduction

Old result reproduced within frozen tolerances: **True**.
Residuals: trades +0, WR -0.0000%, PF +0.0000, account PnL -0.0000.

## Primary transition attribution

| Transition | Δ trades | Δ WR | Δ PF | Δ Mean-R | Δ PnL-R | Retained | Removed | Added | PnL sign changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0->S1 | +1 | -4.17% | -0.2322 | -0.1204 | -1.76 | 15 | 0 | 1 | 0 |
| S1->S2 | +1 | +2.21% | +0.1209 | +0.0418 | +0.76 | 16 | 0 | 1 | 0 |
| S2->S3 | -1 | -2.21% | +0.2491 | +0.0705 | +1.04 | 16 | 1 | 0 | 1 |
| S3->S4 | +0 | -6.25% | -0.3678 | -0.1115 | -1.78 | 14 | 2 | 2 | 0 |
| S4->S5 | +1 | -3.31% | -0.1875 | -0.0762 | -1.25 | 15 | 1 | 2 | 3 |

## Intrabar causal attribution

S0 trades=15; outcome changed=0; SL before close-only TP=0; intrabar TP missed by close-only=0; same-bar both reachable=0; timeout classification changed=0.

## Entry state machine

Legacy opportunities=22; core-gate opportunities=15; S5 entries=17; S5-only=3; delayed within persistent signal=5; re-entry after prior close in same episode=2; occupancy-blocked eligible rows=24.

## Timezone/session reconciliation

Contemporaneous live log rows imply broker/API wall-time offset UTC+3. S0-S5 use the exact hour that gemini.py uses. Correcting to inferred actual UTC would change 45 qualifying rows and 8 S5 trade identities; this alternative is diagnostic only.

## Required conclusions

1. Original ~15 trade / 66.67% / PF ~1.34 reproduced: True.
2. Residual mismatch: trades +0, WR -0.0000%, PF +0.0000, account PnL -0.0000.
3. Close-only to HIGH/LOW first-touch: WR -4.17%, PF -0.2322.
4. Intrabar ordering changed 0 S0 outcomes; 0 first-touch bars could reach both barriers.
5. Entry-state-machine transition S1->S2 changed trades by +1, WR by +2.21%, PF by +0.1209.
6. RSI/session/filter ordering: S5 blocked session=49, RSI-floor=8, RSI-range=30, spread=0 raw rows; exact trade deltas are retained in transition_attribution.csv.
7. Spread/cost semantics S2->S3: WR -2.21%, PF +0.2491; pure identical-identity cost ΔPF=+0.0046, ΔMean-R=+0.0017.
8. Risk/cooldown S3->S4: trades +0, WR -6.25%, PF -0.3678; cooldown-blocked rows=20.
9. Timezone/session interpretation changes 45 qualifying rows and 8 S5 trade identities under the diagnostic UTC-corrected alternative.
10. S5 primary: trades=17, trades/day=0.1995, WR=52.94%, PF=0.9271, Mean-R=-0.0266, PnL-R=-0.45, Max DD-R=-3.23.
11. S5 economically positive on the recent cohort: False.
12. Old 66-70% interpretation: a combination of recent-sample signal quality, selection conditioning, and simulator optimism/execution mismatch; it is not execution-robust alpha.
13. The historical 25.40% OOF weakness remains relevant as cross-regime architecture evidence; simulator reconciliation does not turn fold-specific historical replicas into the exact current artifact or erase their poor ranking/economics.
14. Single next research hypothesis: EXECUTION-ALIGNED LABEL/MODEL VALIDATION: test whether a model trained on HIGH/LOW stop-first executable outcomes contains alpha, without threshold tuning.

## Reproducibility limits

The primary M1 row count and S0 metrics are directly reconciled against the old report. The original raw MT5 snapshot was not retained, so the newly fetched broker history cannot be proven byte-identical. S5 is live-equivalent conditional on the one frozen probability cohort; historical tick path, broker fill/slippage, lot rounding, and exact account state are unavailable and explicitly excluded.

## Operational safety

`gemini.py` and `gold_long_recent_candidate_xgb.json` remained byte-identical before and after the run.
