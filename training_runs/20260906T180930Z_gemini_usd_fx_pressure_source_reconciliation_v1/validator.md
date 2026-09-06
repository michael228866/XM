# Independent DATA-only validator

Internal methodology: **PASS**

Data certification: **FAIL**

Foundation ready: **NO**

| Check | Verdict | Evidence |
|---|---|---|
| source hierarchy followed | PASS | {'XM': {'attempted': True, 'certification': 'FAIL', 'deterministic_retrieval': True, 'attempts': [{'attempt': 1, 'hashes': {'EUR/USD': '91ce86b1c818f375b1a32ed57ae17646f4752ed8a55115219cfa52169f1d9c8b', 'GBP/USD': '3019f3e77b8a5d7f537bd2b11d4f602255ae1d9b9c688c217f474c5ae5907952', 'USD/JPY': '25f8e403223a601d953c9b4d40c09c3a4cec95683c73c1a12ef478509d0b809b'}, 'rows': {'EUR/USD': 3162749, 'GBP/USD': 3019065, 'USD/JPY': 3011477}}, {'attempt': 2, 'hashes': {'EUR/USD': '91ce86b1c818f375b1a32ed57ae17646f4752ed8a55115219cfa52169f1d9c8b', 'GBP/USD': '3019f3e77b8a5d7f537bd2b11d4f602255ae1d9b9c688c217f474c5ae5907952', 'USD/JPY': '25f8e403223a601d953c9b4d40c09c3a4cec95683c73c1a12ef478509d0b809b'}, 'rows': {'EUR/USD': 3162749, 'GBP/USD': 3019065, 'USD/JPY': 3011477}}], 'available_terminal_paths': ['D:\\XM2\\terminal64.exe'], 'unresolved_source_rows': 151024, 'unexplained_gap_count': 5117}, 'Dukascopy': {'attempted': True, 'certification': 'FAIL', 'status': 'candidate', 'raw_db': 'dukascopy_raw_responses.sqlite', 'raw_db_sha256': '998a684ed0bfb99886c62a0c7456904658040705a65dbba29a5f9e71128d7ecf', 'metadata': {'EUR/USD': 'dukascopy_metadata_EURUSD.json', 'GBP/USD': 'dukascopy_metadata_GBPUSD.json', 'USD/JPY': 'dukascopy_metadata_USDJPY.json'}, 'price_convention': 'native M1 BID OHLC', 'timestamp_convention': 'UTC bar-open milliseconds; information at open+1m', 'unresolved_source_rows': 8476}, 'TrueFX': {'attempted': True, 'certification': 'FAIL', 'reason': 'Official reproducible full-range 2016-2024 native M1 endpoint unavailable; hierarchy exhausted without splicing.'}} |
| no provider splicing and same provider for all pairs | PASS | {'provider': None, 'count': 0} |
| no outcome-based provider selection | PASS | provenance, completeness, timestamps and causality only |
| six frozen timestamp hashes | PASS | {'fold1_train': '6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6', 'fold1_score': '47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94', 'fold2_train': '691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44', 'fold2_score': '1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8', 'fold3_train': '3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b', 'fold3_score': '1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449'} |
| historical completeness and source hashes | FAIL | {'rows': {'EUR/USD': 3161876, 'GBP/USD': 3170774, 'USD/JPY': 3168474}, 'unresolved': 8476} |
| closure classification evidence | FAIL | {'unexplained_gap_count': 554} |
| causal completed-bar and staleness | PASS | {'max_staleness_minutes': 5} |
| return horizons and sign orientation | PASS | {'horizons': (1, 5, 15, 60), 'signs': (-1.0, -1.0, 1.0)} |
| equal mean and ddof=1 dispersion | PASS | ['USD_PRESSURE_1M', 'USD_PRESSURE_5M', 'USD_PRESSURE_15M', 'USD_PRESSURE_60M', 'USD_DISPERSION_15M'] |
| no interpolation or unlimited forward fill | PASS | {'unknown_rows': 8476} |
| new matrix identity | PASS | 18f710621c13b032c1656b72e5de85c01062967614b8b1461f10026505fcf24f |
| previous failed run byte-identical | PASS | 20260906T104638Z_gemini_usd_fx_pressure_foundation_v1 |
| operational artifacts unchanged | PASS | {'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07', 'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2'} |
| no model label outcome or performance access | PASS | [] |
