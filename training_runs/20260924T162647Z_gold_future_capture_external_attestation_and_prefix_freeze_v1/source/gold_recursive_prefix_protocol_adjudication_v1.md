# Recursive prefix adjudication

Decision: UNRESOLVED. A deterministic reinitialization policy is representable in
v2, but is not approved for capture until its exact source range and immutable
decision are bound. No market dataset or strategy outcome was read.

## Historical semantics inspected as source text

`drl_trading_v2.py` computes CLOSE EMA12 and EMA26 using pandas
`ewm(span=..., adjust=False).mean()`. With nonmissing input the initial state is
the first close. MACD is EMA12 minus EMA26; its span-9 recursive signal starts
at the first MACD (zero when both close EMAs share their first input). MACD_HIST
is MACD minus this signal. These three recursive states are path-dependent.

For two different starting EMA states, the residual difference after n updates
is `(1-alpha)^n * initial_difference`. A finite warmup can reduce it, but does not
prove exact historical equivalence. Floating-point rounding is not a provenance
proof. Exact reconstruction requires identical certified causal input history,
ordering, timestamp handling, missing-value treatment and initialization.

Other inspected features use finite operations: ATR rolling 14, RSI gain/loss
rolling 14, Bollinger/BIAS rolling 20, ROC lag 5, candle body ratio, ATR rolling
240 ratio, and clock/calendar features. The 20 native trends use rolling 20,
one native-bar lag, as-of alignment and the final M1 feature lag. Eleven M1
features plus 20 native trends remain unchanged. No feature or threshold is tuned.

## Predeclared causal reinitialization proposal

The exact recurrence equations are in gold_recursive_prefix_protocol_v2.json.
Proposed warmup is fixed at 4096 certified M1 rows and 21 closed native rows per
required higher timeframe, before any eligible evaluation row. This is a fixed
engineering proposal, not an outcome-selected parameter. It does not claim
historical equivalence and does not alter feature definitions or lags.

Approval requires an exact inclusive-start/exclusive-end UTC context range,
same source/symbol, sealed context hashes, complete native context, causal closure
times, pinned historical feature source, and a separately immutable decision.
No range is assigned from observed performance. Data before eligibility remain
context only; no historical row is relabelled untouched. No recurrence is computed
on real market data in this task. The current range/approval remain null.

An approved reinitialization would be a new future-evaluation initialization
protocol. It would require review of input-distribution compatibility before any
promotion claim. It cannot retroactively make the historical S4 result untouched.
