# GEMINI US TREASURY REAL RATE B0 B1 V1

Status: **FAIL**

Archival formalization of the already-observed frozen B0/B1 comparison. No post-outcome tuning, feature subset search, threshold search, lag change, transform search, weighting search, or production promotion was performed.

- B0: 31 frozen execution-aligned technical features.
- B1: same 31 plus certified Treasury 5 features.
- XGBoost 3.2.0; 220 trees; learning_rate 0.05; depth 4; min_child_weight 80; subsample/colsample 0.85; random_state 42.
- Same 18-month training windows, three chronological folds, C1 target, S5 simulator, threshold 0.75.

## B0 exact reproduction
PASS — trades=689, WR=0.566038, PF=0.824710, Mean-R=-0.076905, PnL=-52.987816R, stress PF=0.783819.

## B1 Treasury
trades=3739, WR=0.439155, PF=0.612615, Mean-R=-0.216986, PnL=-811.311023R, stress PF=0.564252.

Delta B1-B0: {'trades': 3050, 'trades_per_day': 1.1928040672663276, 'realized_wr': -0.1268828816099552, 'pf': -0.21209527105435966, 'mean_r': -0.1400807055505603, 'pnl_r': -758.3232071913725, 'max_dd_r': -758.1020292133582, 'cost_stress_pf': -0.21956682879562506}

## Decision
**US TREASURY DIRECT-FEATURE FAMILY = FAIL**
Failure reasons: WR<60%, PF<=1.05, Mean-R<=0, PnL-R<=0, BE-edge<=0, stress-PF<=1.
Stopping rule: no Treasury subset/lag/threshold/transform/interaction/weight tuning. Family closed on FAIL.
No production artifact changed.
