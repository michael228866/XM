# GOLD Axis Smoke Test

This is a research-only check. It does not modify `gemini.py`.

Pass: `True`

## Test Window

| Version | PnL | Win | PF | Trades | DD | Max Loss Streak |
|---|---:|---:|---:|---:|---:|---:|
| Current meta overlay | 3325.27 | 73.36% | 1.70 | 274 | -28.60% | 3 |
| GOLD axis candidate | 4461.81 | 70.88% | 1.82 | 261 | -30.60% | 5 |

## Validation Window

| Version | PnL | Win | PF | Trades | DD | Max Loss Streak |
|---|---:|---:|---:|---:|---:|---:|
| Current meta overlay | -286.14 | 51.16% | 0.88 | 303 | -37.76% | 6 |
| GOLD axis candidate | -125.47 | 51.01% | 0.96 | 298 | -34.12% | 6 |

## Candidate Params

```json
{
  "threshold": 0.525,
  "edge_threshold": 0.0,
  "tp_atr": 1.3,
  "sl_atr": 2.0,
  "max_hold": 180,
  "risk_per_trade": 0.028,
  "max_daily_loss_pct": 0.05
}
```
