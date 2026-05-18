# GOLD Meta-Regime Trading System

這個資料夾保存 GOLD# 交易模型、回測研究腳本、meta-regime 動態倉位 overlay，以及 MT5 demo/live 測試入口。

## Current Version

目前建議測試版本：

- Main model: `gold_barrier_final_xgb.json`
- Meta-regime model: `gold_meta_regime_xgb.json`
- Meta config: `gold_meta_regime_overlay.json`
- Live/demo runner: `gemini.py`
- Mode: `USE_META_OVERLAY = True`, `DRY_RUN = False`

## Latest Backtest

Formal baseline:

```text
1000 -> 3186.93
ROI +218.69%
Trades 277
Trades/year 155.0
Max DD -23.38%
PF 1.61
Max loss streak 3
```

Meta overlay:

```text
1000 -> 4325.27
ROI +332.53%
Trades 274
Trades/year 153.3
Max DD -28.60%
PF 1.70
Max loss streak 3
```

## Main Commands

Train/save the meta-regime overlay:

```powershell
python barrier_meta_overlay_train.py
```

Backtest the saved main model plus saved meta overlay:

```powershell
python barrier_meta_overlay_backtest.py
```

Run cost stress:

```powershell
python barrier_meta_overlay_stress.py
```

Run MT5 demo/live test:

```powershell
python gemini.py
```

## Runtime Logs

`gemini.py` writes live/demo signal records to:

```text
gemini_signal_log.csv
```

This file is intentionally ignored by Git because it grows during runtime and may contain account/runtime details. Keep it locally for forward-test analysis.

## Data Policy

Large historical files like `GOLD#_M1_*.csv` are intentionally ignored. They are required for retraining/backtesting, but they should be stored separately or tracked with Git LFS/artifact storage if needed.
