# GOLD causal context protocol

Proposed policy: PREDECLARED_CAUSAL_REINITIALIZATION.
Current approval: false; effective prefix policy: UNRESOLVED.

Pre-boundary native historical bars may be used as CAUSAL_CONTEXT_ONLY once
source, timestamp and native-timeframe compatibility are certified. They never
become holdout evidence. No requirement forces Monthly context to begin after
activation. The prior 21-month warning was conditional on future-only context.

The fixed proposed context envelope starts 2024-01-01T00:00:00Z and ends before
2026-09-24T00:00:00Z. These are preregistered bounds, not a claim that certified
data exist. The envelope predates any boundary established by this task and
provides room for at least 21 native Monthly bars. No market prices or outcomes
were used to select it. The selected rows, closure times and content hashes must
be sealed before approval; currently no compatible sealed context is available.

Within that envelope, use the latest 4096 certified closed M1 rows and latest
21 certified closed bars of each native higher timeframe, in chronological
order. Each context row and its closure must precede the evidence boundary.
After this seed, continue state causally without resets. Gaps and inconsistent
source semantics block approval; no resampling, synthetic bars or imputation.

Only MACD_HIST among the selected 31 features has recursive state. Freeze
pandas 3.0.5, float64, ewm(span=12/26, adjust=False, min_periods=0,
ignore_na=False), MACD=EMA12-EMA26, span9 signal then MACD minus signal.
Price EMAs seed from the first valid close; signal starts at zero with valid
first close. Reject missing context instead of silently filling it. Native
rolling20 trends retain native shift1, backward as-of and final M1 shift1.
The complete feature loader is hashed but never imported/executed here because
it also computes future labels. Model input conversion remains frozen float32.

First feature validity requires complete native context, valid shifted inputs,
and causal closure checks. Evaluation eligibility additionally requires frozen
protocol, activation, valid timezone coverage and a sealed strictly later M1
evidence bar. Capture start, holdout start and evaluation start are separate.
Compatible pre-boundary context can permit evaluation at the first evidence
bar; otherwise it must wait. No eligibility is inferred from strategy outcomes.

Reinitialization is never historically equivalent. Prior warmup254 mismatch
and identical-prefix4096 positive control remain characterization only.
User authorization permits methodology approval, but cannot supply the missing
source/timezone compatibility evidence. Hence approved=false remains necessary.
