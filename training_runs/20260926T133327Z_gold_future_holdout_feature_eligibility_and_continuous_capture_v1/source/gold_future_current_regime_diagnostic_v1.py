"""Current-regime observations; retain timestamps and sanitized identity only."""
import math
import time
from datetime import datetime, timezone

from gold_future_source_identity_v4 import EXPECTED, session, verify_identity

POLICY = 'EMPIRICALLY_CERTIFIED_CURRENT_SOURCE_TIME_REGIME'
SKEW = 5


def check_samples(samples, previous=None):
    if len(samples) < 3:
        raise ValueError('INSUFFICIENT_CURRENT_OBSERVATIONS')
    prior = previous
    for sample in samples:
        identity = sample['source_identity']
        if any(identity.get(k) != v for k, v in EXPECTED.items()) or identity.get('symbol') != 'GOLD#' or identity.get('source_id') != 'XMGlobal-MT5-6_GOLD':
            raise ValueError('SOURCE_IDENTITY_MISMATCH')
        raw, observed = sample['raw_epoch'], sample['observed_utc_epoch']
        if type(raw) is not int or type(observed) not in (int, float) or not math.isfinite(observed):
            raise ValueError('INVALID_CLOCK_SAMPLE')
        offset = math.floor((raw-observed)/3600 + .5)*3600
        if offset == 7200:
            raise ValueError('PAUSE_AND_REQUIRE_REGIME_CERTIFICATION')
        if offset != 10800:
            raise ValueError('PAUSE_AND_QUARANTINE')
        if abs(raw-observed-offset) > SKEW:
            raise ValueError('CLOCK_SKEW_EXCEEDED')
        if prior and (raw <= prior['raw_epoch'] or observed <= prior['observed_utc_epoch']):
            raise ValueError('TIMESTAMP_REVERSAL_OR_STALE_TICK')
        epochs = sample['m1_raw_epochs']
        if len(epochs) < 3 or any(type(e) is not int or e % 60 for e in epochs) or any(b-a != 60 for a, b in zip(epochs, epochs[1:])):
            raise ValueError('CURRENT_M1_SPACING_INVALID')
        if not 0 <= raw-epochs[-1] < 120:
            raise ValueError('CURRENT_M1_NOT_CURRENT')
        prior = sample
    return {'status': 'PASS', 'classification': 'CURRENT_REGIME_10800_CERTIFIED',
            'sample_count': len(samples), 'current_certified_offset_seconds': 10800,
            'certified_offsets_seconds': [10800], 'uncertified_candidate_offsets_seconds': [7200],
            'policy': POLICY, 'runtime_fail_closed': True, 'max_runtime_clock_skew_seconds': SKEW,
            'monotonic': True, 'spacing': True}


def observe(mt5):
    identity = verify_identity(mt5)
    tick = mt5.symbol_info_tick('GOLD#')
    observed = datetime.now(timezone.utc).timestamp()
    if tick is None:
        raise ValueError('NO_CURRENT_TICK')
    raw = int(tick.time)
    del tick
    rates = mt5.copy_rates_from_pos('GOLD#', mt5.TIMEFRAME_M1, 0, 3)
    if rates is None:
        raise ValueError('NO_CURRENT_M1')
    epochs = [int(row['time']) for row in rates]
    del rates
    if verify_identity(mt5) != identity:
        raise ValueError('SOURCE_CHANGED_DURING_OBSERVATION')
    return {'source_identity': identity, 'raw_epoch': raw,
            'observed_utc_epoch': observed, 'm1_raw_epochs': epochs,
            'normalized_timestamp': datetime.fromtimestamp(raw-10800, timezone.utc).isoformat(),
            'observed_offset_seconds': math.floor((raw-observed)/3600+.5)*3600,
            'normalized_clock_error_seconds': raw-10800-observed}


def diagnose():
    with session() as (mt5, identity):
        samples = []
        for index in range(3):
            if index:
                time.sleep(2)
            samples.append(observe(mt5))
        try:
            decision = check_samples(samples)
            error = None
        except ValueError as failure:
            error = str(failure)
            decision = {'status': 'FAIL', 'classification': 'CONTRADICTED', 'blocker': error}
        return {'samples': samples, 'decision': decision, 'error': error,
                'source_identity': identity, 'strategy_outcome_inspected': False,
                'effective_from_utc': datetime.now(timezone.utc).isoformat(),
                'coverage_mode': 'CURRENT_REGIME_OPEN_ENDED_WHILE_RUNTIME_VALIDATION_PASSES',
                'effective_until': None,
                'clock_basis': 'OPERATIONAL_SYSTEM_UTC; 5s fixed margin over prior approximately 0.6s observations; no independent clock authority claim'}
