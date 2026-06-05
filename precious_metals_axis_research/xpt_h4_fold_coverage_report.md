# XPTUSD H4 Fold-Coverage Refinement

Research-only refinement focused on increasing fold-level trade coverage.

## Selected

| Profile | Strict Gate | 3x R | Trades | Min Fold Trades | Positive | Passed | Win | PF | Worst R | DD | Params |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current_symbol | False | 12.97 | 44 | 2 | 4/4 | 3/4 | 79.55% | 251.65 | 1.60 | -2.71 | none: conf=0.52, edge=0.0, tp/sl=3.2/4.4, hold=96, dir=long |

## Cost Stress

| Cost | Gate | Strict Gate | R | Trades | Min Fold Trades | Positive | Passed | Win | Worst R | DD |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | True | False | 16.80 | 44 | 2 | 4/4 | 3/4 | 79.55% | 1.60 | -2.58 |
| 2.0x | True | False | 14.89 | 44 | 2 | 4/4 | 3/4 | 79.55% | 1.60 | -2.64 |
| 3.0x | True | False | 12.97 | 44 | 2 | 4/4 | 3/4 | 79.55% | 1.60 | -2.71 |
| 4.0x | True | False | 11.06 | 44 | 2 | 4/4 | 3/4 | 77.27% | 1.60 | -2.77 |
| 5.0x | True | False | 9.15 | 44 | 2 | 4/4 | 3/4 | 77.27% | 1.60 | -2.83 |

## Fold Details

- fold_1: R=1.60, trades=2, win=100.00%, PF=999.00, pass=False
- fold_2: R=3.52, trades=12, win=83.33%, PF=2.82, pass=True
- fold_3: R=2.80, trades=8, win=87.50%, PF=2.87, pass=True
- fold_4: R=5.05, trades=22, win=72.73%, PF=1.91, pass=True
