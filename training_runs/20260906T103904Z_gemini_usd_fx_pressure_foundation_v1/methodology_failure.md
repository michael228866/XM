# Acquisition methodology failure

This run is preserved as a meaningful aborted data-only execution.

The broker symbols resolved correctly, and MT5 returned bars with success. The executed script
incorrectly shifted UTC request times and then subtracted the broker offset from returned epochs.
That caused every returned bar to be discarded by the post-request UTC filter.

An independent timestamp-only probe confirmed that the first EURUSD# chunk returned 256 bars whose
raw epochs exactly spanned the UTC query. This agrees with the official MetaTrader 5 Python contract:
rate request datetimes and returned bar-open epochs are UTC without broker-wall shift.

No model, label, prediction, trade outcome, threshold, or strategy metric was accessed. This run is
not evidence that historical FX data is unavailable. A corrected acquisition requires a new formal
run and must not overwrite this record.
