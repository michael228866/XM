# Precious Metals Composite Plan

Research-only plan. It does not modify `gemini.py`.

## Current Composite

| Symbol | Role | Status | Strategy | Key Result | Weight Notes |
|---|---|---|---|---|---|
| GOLD# | anchor | enabled_research | existing_gold_high_profit_candidate | PnL 4461.81, win 70.88%, PF 1.82 | enabled in research weights |
| SILVER# | core_satellite | enabled_research | strict_stress_h1_candidate | 3x spread 22.13R, win 67.72%, PF 1.62 | enabled in research weights |
| XAUEUR# | gold_cross_validation | watchlist | disabled_until_revalidated | failed_cost_aware_walk_forward | 0% until revalidated |
| XPTUSD# | platinum_watchlist | watchlist | disabled_until_revalidated | failed_walk_forward_needs_regime_filter | 0% until revalidated |
| XPDUSD# | palladium_watchlist | watchlist | disabled_until_revalidated | sample_too_thin | 0% until revalidated |
| GAUCNH# | supplemental_gold_cross | watchlist | disabled_until_revalidated | history_too_short | 0% until revalidated |

## Research Weights

| Book | GOLD# | SILVER# | Notes |
|---|---:|---:|---|
| conservative | 55% | 45% | Use until combined paper test is stable |
| silver_tilt | 40% | 60% | Higher non-GOLD exposure, research only |

## Decision

- Current composite candidate is GOLD# + SILVER# only.
- XAUEUR#, XPTUSD#, XPDUSD#, and GAUCNH# stay disabled until they pass cost-aware walk-forward and stress checks.
- Next step is a combined paper portfolio runner, not live deployment.
