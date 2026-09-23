# GOLD recursive prefix proposal

Status: unapproved; operational prefix policy remains UNRESOLVED.

The proposal preserves the prior fixed 4096 M1 and 21 native bars per higher
timeframe. This is an engineering initialization choice, never an optimized
warmup or a claim of historical equivalence. Native Monthly context can require
21 months if collected only after activation. Raw capture and evaluation starts
must therefore be recorded separately; 365 certified evaluation days cannot
include warmup. No exact source range, immutable approval or eligible date is assigned.

Only M1 MACD_HIST of the 31 selected features is recursive. EMA12 and EMA26
start with the first valid close; their difference initializes the span9 signal
at zero for a valid first row. pandas adjust=False, min_periods=0, ignore_na=False
apply. Missing observations use pandas absolute-position weighting. Rolling
windows require their default full valid window; native comparisons with NaN
historically yield -1 before shifting. Native EMA/MACD are calculated by the
historical helper but discarded from the selected 20 native trend inputs.

The historical loader sorts source filenames, takes the last file for each
timeframe, sorts naive timestamps, applies native lag1, backward as-of alignment,
then all-feature M1 lag1 and dropna. It must never be imported in this task:
the full loader also constructs future labels. Source inspection only follows
S4 -> discovery -> B0 reconstruction -> C1 -> barrier_final_train -> drl_trading_v2.
No wrappers or label logic are executed.

The archived B0 training start is 2016-07-01 broker-naive. It is not the original
recursive seed timestamp. The archive did not retain a certified full native
raw prefix. All 21 earliest certified source timestamps therefore remain null.
Synthetic feature-state comparisons document convergence and mismatches only.
Bit equality on one synthetic fixture does not establish exact historical state.

The proposal requires a separately frozen exact source range, feature/pipeline
hashes, native closure semantics, nonmissing inputs, and explicit approval.
No models, decisions or outcomes are computed to assess this proposal.
