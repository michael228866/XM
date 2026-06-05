# High Win / High Profit Research

This folder is isolated from the live `gemini.py` strategy.

Goal:
- Search for research candidates that keep win rate high while improving net profit.
- Use both validation and test windows so a candidate must survive more than one period.
- Write results only inside this folder.

Run:

```powershell
python .\high_win_profit_research\analyze_high_win_profit.py
```

Check whether other symbols have enough local data for this pipeline:

```powershell
python .\high_win_profit_research\check_cross_asset_readiness.py
```

Probe MT5 symbols for M1 spread cost versus recent ATR:

```powershell
python .\high_win_profit_research\mt5_cross_asset_probe.py
```

Outputs:
- `high_win_profit_candidates.csv`
- `high_win_profit_candidates.json`
- `high_win_profit_report.md`
- `best_candidate.json`
- `cross_asset_readiness.csv`
- `mt5_cross_asset_probe.csv`

The script does not modify live settings, model files, or trade logs.

Current acceptance gate:
- Validation must improve materially versus the current meta-overlay baseline.
- Test PnL must improve by at least 800 versus the current meta-overlay baseline.
- Test win rate must be at least 70%.
- Test profit factor must be at least 1.75.
- Test drawdown must stay within 36%.
