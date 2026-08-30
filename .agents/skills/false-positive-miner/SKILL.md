---
name: false-positive-miner
description: Analyze losing GOLD# executable signals and validate filters or meta-models by losers removed versus winners accidentally removed. Use for false-positive clustering and precision-filter research, not for generic feature importance or live trade intervention.
---

# False-Positive Miner

Find stable, entry-time characteristics of losing executable trades and remove losers without sacrificing comparable winners.

## Repository contract

- Start from a named candidate and its out-of-fold signals. Gen12's non-overlapping executable events are the preferred baseline.
- Use first-touch rewards and exit offsets from `gold_generation11_execution_aligned.py`; a losing signal is a mature executable event with net reward less than or equal to zero after the stated costs.
- Analyze the exact signal universe used by the backtest. Do not compare row counts from one sampling rule with trade counts from another.
- Keep discovery, calibration, and evaluation chronological. Features, percentiles, regimes, and model probabilities must be known at entry time.

## Analysis

Break losses and comparable wins down by direction, expert, session, volatility regime, and stable entry-time features. Useful repository features include ATR or `VOLA_RATIO`, RSI, `BIAS_20`, candle body, ROC, higher-timeframe trend, distance from trend, weekday, and base P(win)/Expected-R. Analyze spread only when historical spread was actually captured; do not substitute a future or constant value and call it observed.

Prefer the simplest stable separator:

1. a transparent filter supported across folds;
2. a small interaction or regime-specific threshold;
3. only then a meta-model trained on out-of-fold base predictions.

Never train the meta-filter on in-sample probabilities from the base model. Do not use future excursion, exit reason, or TP/SL outcome as an input feature.

## Filter evaluation

For each proposed filter, compute identities from the same mature executable opportunity set:

- losers before, removed, and retained;
- winners before, accidentally removed, and retained;
- loser-removal and winner-removal rates;
- executable trades, win rate, PF, PnL/mean R, and max drawdown before and after;
- results by walk-forward fold, holdout, recent interval, and cost stress.

Prefer filters with a materially higher loser-removal rate than winner-removal rate. Reject filters whose apparent benefit comes from deleting nearly all signals, fails any historical period, or only improves the recent interval.

Keep outputs `research_only` and leave `gemini.py` unchanged unless the repository promotion gate passes.
