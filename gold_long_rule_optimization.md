# GOLD long-rule optimization

The incumbent model is unchanged. Candidate selection uses development and validation only.

| Fold | Version | Trades | Win | PF | PnL | DD |
|---|---|---:|---:|---:|---:|---:|
| recent_development | baseline | 14 | 64.29% | 0.85 | -42.70 | -21.33% |
| recent_development | candidate | 26 | 73.08% | 1.98 | 337.89 | -17.49% |
| recent_validation | baseline | 53 | 69.81% | 1.29 | 189.65 | -20.66% |
| recent_validation | candidate | 61 | 73.77% | 1.55 | 398.71 | -17.09% |
| historical_reference | baseline | 307 | 68.73% | 1.46 | 2076.97 | -59.50% |
| historical_reference | candidate | 434 | 64.98% | 1.19 | 657.53 | -42.59% |
| recent_forward | baseline | 116 | 60.34% | 0.73 | -245.17 | -31.95% |
| recent_forward | candidate | 150 | 59.33% | 0.77 | -239.58 | -35.60% |

Historical 10-point cost stress: `PASS`
Recent 10-point cost stress: `FAIL`
Promotion gate: `FAIL`

Selected parameters: `{"augment_threshold": 0.5, "augment_quality_floor": 0.55, "augment_edge_threshold": 0.2, "tp_atr": 1.3, "sl_atr": 2.0, "max_hold": 120, "session_profile": "approved_expanded"}`
