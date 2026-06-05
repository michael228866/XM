# Four-Metal Main Promotion Plan

Research-only promotion plan. This does not modify `gemini.py` or place live orders.

| Symbol | Role | Weight | TF | 3x R / PnL | Trades | Win | PF | DD | Gate Scope |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| GOLD# | main_anchor | 35.00% | M1 | 4461.81 | 261 | 70.88% | 1.82 | -30.60% | legacy GOLD gate |
| SILVER# | main_core | 25.00% | H1 | 17.31R | 64 | 65.63% | 1.79 | -2.77R | passes 1x-3x; fails 4x-5x |
| XPTUSD# | slow_main | 20.00% | H4 | 12.97R | 44 | 79.55% | 251.65 | -2.71R | passes 1x-5x |
| XPDUSD# | slow_main | 20.00% | H12 | 9.22R | 27 | 59.26% | 7.72 | -2.07R | passes 1x-3x; fails 4x-5x |

## Promotion Notes

- `GOLD#` and `SILVER#` are the faster/core main lines.
- `XPTUSD#` and `XPDUSD#` are slow-main lines, so the minimum fold-trade gate is lower by timeframe.
- Forward paper logging is still required before live deployment.
