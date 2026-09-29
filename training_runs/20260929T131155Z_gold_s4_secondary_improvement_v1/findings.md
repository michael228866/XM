# Historical S4 improvement research

========================================
XM GOLD S4 Improvement Train + Validate
========================================
RUN_ID:
20260929T131155Z_gold_s4_secondary_improvement_v1
Reference:
PASS
Research:
PASS
Validation:
FAIL
Final:
FAIL
Result:
IMPROVEMENT_FOUND
REFERENCE
{
  "trades": 743,
  "wins": 421,
  "losses": 322,
  "realized_win_rate": 0.566621803499327,
  "trades_per_day": 0.2905748924520923,
  "profit_factor": 0.8271090428367022,
  "mean_r": -0.07580700838089183,
  "pnl_r": -56.32460722700263,
  "max_drawdown_r": -62.009878061758215,
  "stress_pf": 0.7857575699558063
}
NEW CANDIDATE
{
  "trades": 768,
  "wins": 438,
  "losses": 330,
  "realized_win_rate": 0.5703125,
  "trades_per_day": 0.30035197497066873,
  "profit_factor": 0.8436457240683531,
  "mean_r": -0.06800170013341929,
  "pnl_r": -52.22530570246601,
  "max_drawdown_r": -56.4297416185602,
  "stress_pf": 0.8016509741596182
}
Delta WR: 0.0036906965006729964
Delta Trades/Day: 0.009777082518576452
Gate: INTERESTING
Candidate Model: D:\XM\數據\training_runs\20260929T131155Z_gold_s4_secondary_improvement_v1\models\F5_NO_REDUNDANT_HTF\fold3.json
SHA256: 617e5724e2405e7423ec61b4104c4272d9094ed1fe29d6da95735326baf18753
ValueError: ValueError: source_binding:gold_independent_secondary_classifier_v1.py
Production:
未變更
Locked Future Holdout:
未使用
========================================

Historical development evidence only; no promotion.
