# XPD Alternate Target / Exit Search

Research-only test using direction and dominant-swing targets instead of the original clean barrier target.

## Selected

| TF | Target | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| H12 | future_close h=8 | 3/5 | True | 9.22 | 27 | 59.26% | 7.72 | 1.70 | -2.07 | low_vola: exit=time_stop, conf=0.46, edge=0.0, tp/sl=1.4/2.8, hold=14, dir=long |

## Cost Stress

| Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | True | 11.37 | 4/4 | 3/4 | 27 | 59.26% | 1.73 | -2.07 |
| 2.0x | True | 10.30 | 4/4 | 3/4 | 27 | 59.26% | 1.73 | -2.07 |
| 3.0x | True | 9.22 | 4/4 | 3/4 | 27 | 59.26% | 1.70 | -2.07 |
| 4.0x | False | 8.15 | 4/4 | 2/4 | 27 | 51.85% | 1.18 | -2.07 |
| 5.0x | False | 7.08 | 4/4 | 2/4 | 27 | 51.85% | 0.66 | -2.07 |
