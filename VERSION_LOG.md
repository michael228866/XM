# Version Log

## 2026-05-18 - Meta-Regime Overlay Demo Version

### Summary

Built a practical GOLD# trading version using the existing final barrier model plus a saved meta-regime overlay for dynamic position sizing.

### Key Files

- `barrier_meta_overlay.py`: shared overlay feature, model save/load, and risk multiplier utilities.
- `barrier_meta_overlay_train.py`: trains and saves `gold_meta_regime_xgb.json` plus `gold_meta_regime_overlay.json`.
- `barrier_meta_overlay_backtest.py`: loads the saved main model and saved meta model, then compares baseline vs overlay.
- `barrier_meta_overlay_optimize.py`: scans practical overlay risk rules.
- `barrier_meta_overlay_stress.py`: cost stress test for the saved overlay.
- `gemini.py`: MT5 demo/live runner with meta overlay, dynamic lot sizing, and CSV signal logging.

### Selected Rule

```text
risk_per_trade = 0.028
risk_rule = (0.40, 0.56, 0.72, 1.00, 1.45, 1.65)
```

Interpretation:

- Low meta quality: keep base risk at 1.00x instead of over-filtering.
- Medium/high quality: increase position size to 1.45x / 1.65x.
- Keep max loss streak unchanged in backtest.

### Validation Result

Formal baseline:

```text
1000 -> 3186.93
ROI +218.69%
Trades 277
DD -23.38%
PF 1.61
Max loss streak 3
```

Meta overlay:

```text
1000 -> 4325.27
ROI +332.53%
Trades 274
DD -28.60%
PF 1.70
Max loss streak 3
```

### Cost Stress

```text
extra_cost 5.0  : 1000 -> 4325.27, PF 1.70
extra_cost 7.5  : 1000 -> 3828.40, PF 1.65
extra_cost 10.0 : 1000 -> 3967.96, PF 1.67
extra_cost 12.5 : 1000 -> 3623.56, PF 1.62
```

### Live/Demo Notes

`gemini.py` now:

- Loads `gold_barrier_final_xgb.json`.
- Loads `gold_meta_regime_xgb.json` and `gold_meta_regime_overlay.json`.
- Uses dynamic risk multiplier in lot sizing.
- Keeps spread, session, RSI, daily loss, and max-position guards.
- Writes signal/order status to `gemini_signal_log.csv`.

Current setting:

```text
USE_META_OVERLAY = True
DRY_RUN = False
```
