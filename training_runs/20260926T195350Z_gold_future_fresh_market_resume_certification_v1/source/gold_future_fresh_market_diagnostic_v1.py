"""Timestamp-only freshness gate; never writes capture state or admits bars."""
import math
from datetime import datetime, timezone

from gold_future_source_identity_v4 import EXPECTED, verify_identity

TICK_MAX_AGE = 120
BAR_MAX_AGE = 180
SKEW = 5


def observe(mt5):
    """Do not call the old observer, which infers offset even for stale data."""
    identity = verify_identity(mt5)
    tick = mt5.symbol_info_tick('GOLD#')
    clock = datetime.now(timezone.utc).timestamp()
    if tick is None:
        raise ValueError('NO_CURRENT_TICK')
    raw = int(tick.time)
    del tick
    rates = mt5.copy_rates_from_pos('GOLD#', mt5.TIMEFRAME_M1, 0, 3)
    if rates is None or len(rates) != 3:
        raise ValueError('NO_CURRENT_M1')
    epochs = [int(row['time']) for row in rates]
    del rates
    if verify_identity(mt5) != identity:
        raise ValueError('SOURCE_IDENTITY_CHANGED')
    return {'source_identity': identity, 'sample_time_system_utc': datetime.fromtimestamp(clock, timezone.utc).isoformat(),
            'system_utc_epoch': clock, 'raw_tick_epoch': raw, 'm1_raw_epochs': epochs}


def assess(samples, previous_tick, last_accepted_bar):
    """Use fixed-regime ages as a freshness screen, never as offset inference.

    Independently observed tick/bar advancement is separate fresh evidence.
    It prevents a live 7200 feed from being hidden by its apparent 3600-second
    age under 10800. A stable old observation performs no offset/skew inference.
    Two-minute tick and three-minute bar thresholds bound M1 liveness; the
    frozen five-second skew tolerance remains unchanged after freshness.
    """
    rows = []
    previous = None
    identity_pass = True
    conflict = None
    for sample in samples:
        identity = sample.get('source_identity', {})
        identity_ok = (all(identity.get(k) == v for k, v in EXPECTED.items())
                       and identity.get('source_id') == 'XMGlobal-MT5-6_GOLD'
                       and identity.get('symbol') == 'GOLD#'
                       and identity.get('digits') == 2 and identity.get('point') == .01)
        identity_pass &= identity_ok
        raw, clock, epochs = sample['raw_tick_epoch'], sample['system_utc_epoch'], sample['m1_raw_epochs']
        if (type(raw) is not int or type(clock) not in (int, float) or not math.isfinite(clock)
                or len(epochs) != 3 or any(type(e) is not int or e % 60 for e in epochs)
                or any(b-a != 60 for a, b in zip(epochs, epochs[1:]))):
            raise ValueError('INVALID_SOURCE_TIMESTAMP_SAMPLE')
        closed = max((e for e in epochs[:-1] if e+60 <= raw), default=None)
        age = clock-(raw-10800)
        bar_age = clock-(closed-10800) if closed is not None else None
        monotonic = raw >= previous_tick and (previous is None or (
            raw >= previous['raw_tick_epoch'] and clock > previous['system_utc_epoch']
            and epochs[-1] >= previous['m1_raw_epochs'][-1]))
        progression = previous is not None and (raw > previous['raw_tick_epoch'] or epochs[-1] > previous['m1_raw_epochs'][-1])
        tick_fresh = 0 <= age <= TICK_MAX_AGE or progression
        bar_fresh = closed is not None and (0 <= bar_age <= BAR_MAX_AGE or (
            progression and 60 <= raw-closed <= BAR_MAX_AGE))
        fresh = identity_ok and monotonic and tick_fresh and bar_fresh
        row = {**sample, 'raw_closed_bar_epoch': closed,
            'source_observation_age_seconds': age, 'closed_bar_age_seconds': bar_age,
            'age_basis': 'screen under existing 10800 only; not a current offset inference',
            'source_progress_observed': progression, 'tick_fresh': tick_fresh, 'closed_bar_fresh': bar_fresh,
            'source_identity_pass': identity_ok, 'monotonicity_pass': monotonic,
            'fresh': fresh, 'inferred_offset_seconds': None, 'normalized_tick_utc': None,
            'normalized_bar_utc': None, 'normalized_clock_error_seconds': None,
            'classification': 'NO_FRESH_MARKET_DATA', 'failed_rule': None}
        if not identity_ok:
            row.update(classification='TIME_RULE_CONFLICT', failed_rule='SOURCE_IDENTITY_CHANGED')
        elif not monotonic:
            row.update(classification='TIME_RULE_CONFLICT', failed_rule='TIMESTAMP_REVERSAL')
        elif fresh:
            offset = math.floor((raw-clock)/3600+.5)*3600
            error = raw-10800-clock
            row.update(inferred_offset_seconds=offset, normalized_clock_error_seconds=error,
                       normalized_tick_utc=datetime.fromtimestamp(raw-10800, timezone.utc).isoformat(),
                       normalized_bar_utc=datetime.fromtimestamp(closed-10800, timezone.utc).isoformat())
            if offset == 7200:
                row.update(classification='UNCERTIFIED_CANDIDATE_REGIME', failed_rule='OFFSET_7200_REQUIRES_SEPARATE_CERTIFICATION')
            elif offset != 10800:
                row.update(classification='TIME_RULE_CONFLICT', failed_rule='OFFSET_NOT_CERTIFIED')
            elif abs(error) > SKEW:
                row.update(classification='TIME_RULE_CONFLICT', failed_rule='CLOCK_SKEW_EXCEEDED')
            else:
                row['classification'] = 'TIME_RULE_PASS'
        if row['classification'] in {'TIME_RULE_CONFLICT', 'UNCERTIFIED_CANDIDATE_REGIME'}:
            conflict = conflict or row['classification']
        rows.append(row)
        previous = sample
    fresh_rows = [r for r in rows if r['fresh']]
    ticks = {r['raw_tick_epoch'] for r in fresh_rows}
    bars = {r['raw_closed_bar_epoch'] for r in fresh_rows if r['raw_closed_bar_epoch'] > last_accepted_bar}
    errors = [r['normalized_clock_error_seconds'] for r in fresh_rows]
    offsets = {r['inferred_offset_seconds'] for r in fresh_rows}
    latest_fresh = bool(rows and rows[-1]['fresh'])
    state = conflict or ('TIME_RULE_PASS' if latest_fresh else 'NO_FRESH_MARKET_DATA')
    regime = 'FAIL' if conflict else 'PASS' if latest_fresh and len(ticks) >= 5 and offsets == {10800} else 'PARTIAL'
    return {'samples': rows, 'classification': state, 'current_regime_status': regime,
            'source_identity_status': 'PASS' if samples and identity_pass else 'FAIL',
            'fresh_source_data_available': latest_fresh, 'fresh_tick_sample_count': len(ticks),
            'new_closed_m1_count': len(bars), 'observed_offset_seconds': next(iter(offsets)) if len(offsets) == 1 else None,
            'normalized_clock_error_seconds_min': min(errors) if errors else None,
            'normalized_clock_error_seconds_max': max(errors) if errors else None,
            'latest_source_observation_age_seconds': rows[-1]['source_observation_age_seconds'] if rows else None,
            'evidence_admitted': False, 'resume_authorized': False,
            'fresh_tick_max_age_seconds': TICK_MAX_AGE, 'fresh_closed_m1_max_age_seconds': BAR_MAX_AGE,
            'max_clock_skew_seconds': SKEW}
