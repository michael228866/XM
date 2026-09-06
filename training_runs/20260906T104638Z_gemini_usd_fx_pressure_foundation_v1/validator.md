# Independent DATA-only validator

Internal methodology: **PASS**

Data certification: **FAIL**

USD FX PRESSURE DATA FOUNDATION READY: **NO**

| Check | Verdict | Evidence |
|---|---|---|
| economic symbol identity | PASS | {'selected': {'EUR/USD': 'EURUSD#', 'GBP/USD': 'GBPUSD#', 'USD/JPY': 'USDJPY#'}, 'errors': []} |
| historical source coverage | PASS | {'rows': {'EUR/USD': 3162754, 'GBP/USD': 3019070, 'USD/JPY': 3011482}, 'errors': [], 'coverage_ok': True} |
| MT5 UTC request and filter semantics | PASS | {'mismatches': [], 'source': 'MQL5 copy_rates_range UTC contract'} |
| timezone and returned-epoch conversion | PASS | {'errors': [], 'mapping': 'MT5 Python returned bar-open epoch is UTC without broker-wall shift'} |
| exact six timestamp hashes | PASS | {'fold1_train': {'rows': 530218, 'hash': '6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6'}, 'fold1_score': {'rows': 1058080, 'hash': '47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94'}, 'fold2_train': {'rows': 532563, 'hash': '691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44'}, 'fold2_score': {'rows': 708197, 'hash': '1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8'}, 'fold3_train': {'rows': 533580, 'hash': '3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b'}, 'fold3_score': {'rows': 708020, 'hash': '1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449'}} |
| bar-open versus completed-bar semantics | PASS | {'causal': True, 'maximum_age_minutes': 5.0} |
| causal as-of and no incomplete-bar use | PASS | {'causal': True, 'maximum_age_minutes': 5.0} |
| fixed return construction and USD orientation | PASS | {'features': ['USD_PRESSURE_1M', 'USD_PRESSURE_5M', 'USD_PRESSURE_15M', 'USD_PRESSURE_60M', 'USD_DISPERSION_15M'], 'horizons': [1, 5, 15, 60], 'signs': [-1.0, -1.0, 1.0]} |
| equal weighting and 15-minute sample dispersion | PASS | Independent mean and numpy std(ddof=1) reconstruction |
| staleness, NaN semantics, and no forward fill | PASS | {'unknown_rows': 148867, 'legitimate_nan_rows': 65277, 'max_staleness': 5} |
| source-gap resolution and uniquely known row state | FAIL | {'unresolved_source_rows': 148867} |
| frozen matrix and source hashes | PASS | {'logical': 'da226ee13e8470a919cc4b08770e76c376911e8d4613e0bc5ffc98a8395fdf81', 'archive': 'f7192c75813257c5e4d75b9ccdbec6edafbc84c998c51b21755b79aa852112f5'} |
| previous finalized runs unchanged | PASS | 12 archives |
| operational artifacts unchanged | PASS | {'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07', 'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2'} |
| data-only and no outcome access | PASS | {'forbidden_hits': []} |
| no feature or model search | PASS | {'family': 'USD FX PRESSURE', 'economic_instruments': ['EUR/USD', 'GBP/USD', 'USD/JPY'], 'horizons_minutes': [1, 5, 15, 60], 'features': ['USD_PRESSURE_1M', 'USD_PRESSURE_5M', 'USD_PRESSURE_15M', 'USD_PRESSURE_60M', 'USD_DISPERSION_15M'], 'orientation': {'EUR/USD': -1, 'GBP/USD': -1, 'USD/JPY': 1}, 'aggregation': 'equal arithmetic mean; sample standard deviation ddof=1 for 15m dispersion', 'max_external_bar_staleness_minutes': 5, 'bar_semantics': 'MT5 Python M1 time is UTC bar OPEN; information time is open plus one minute', 'asof_semantics': 'latest completed close <= anchor; both t and t-h endpoints <=5 wall-clock minutes stale', 'normalization': 'none', 'feature_or_horizon_search': False, 'model_training': False, 'strategy_evaluation': False} |
| clean pushed pre-run and immutable execution | PASS | {'pre_run_git_commit': 'c2e663616cd12838da6bc27181e4cd29e06a2ecc', 'pre_run_git_dirty': False, 'head_sha': 'c2e663616cd12838da6bc27181e4cd29e06a2ecc', 'origin_main_sha': 'c2e663616cd12838da6bc27181e4cd29e06a2ecc', 'head_equals_origin_main': True} |

## Walk-forward-validator scope

| Check | Verdict |
|---|---|
| chronology | PASS |
| feature leakage | PASS |
| label maturity | PASS |
| OOF predictions | PASS |
| calibration | PASS |
| threshold selection | PASS |
| purge/embargo | PASS |
| holdout contamination | PASS |
| recent-period reuse | PASS |
| execution alignment | PASS |
| cost assumptions | PASS |
| multiple-testing risk | PASS |

No model, performance, alpha, threshold, or production conclusion was evaluated.
