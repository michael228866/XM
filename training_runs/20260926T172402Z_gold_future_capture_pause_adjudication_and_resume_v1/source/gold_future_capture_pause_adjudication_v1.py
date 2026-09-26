"""Read-only pause adjudication. This module cannot resume or admit evidence."""
import math


IDENTITY = {
    'broker_company': 'XM Global Limited',
    'broker_server': 'XMGlobal-MT5 6',
    'source_account_environment': 'demo',
    'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#',
    'digits': 2, 'point': 0.01,
}


def classify(samples, last_raw, previous_tick=None, chain_ok=True):
    """Diagnostic classification only; never a collector admission permission.

    Progress of the source clock, not its distance from local UTC, establishes
    liveness. Unchanged observations cannot certify any offset. A fresh wrong
    offset therefore cannot hide behind an age threshold. Reversal is checked
    before idle. Frozen strict sample checks still govern any future admission.
    """
    def result(state, rule, fresh=False):
        return {'classification': state, 'failed_rule': rule,
                'market_data_fresh': fresh, 'evidence_admitted': False,
                'resume_authorized': False}

    if not chain_ok:
        return result('CHAIN_CONFLICT', 'CHAIN_CONFLICT')
    if len(samples) < 3:
        return result('UNKNOWN', 'INSUFFICIENT_CURRENT_OBSERVATIONS')
    for sample in samples:
        if any(sample.get('source_identity', {}).get(k) != v
               for k, v in IDENTITY.items()):
            return result('SOURCE_IDENTITY_CONFLICT', 'SOURCE_IDENTITY_CHANGED')
        if (type(sample.get('raw_epoch')) is not int
                or type(sample.get('observed_utc_epoch')) not in (int, float)
                or not math.isfinite(sample['observed_utc_epoch'])):
            return result('TIME_RULE_CONFLICT', 'INVALID_CLOCK_SAMPLE')
        epochs = sample.get('m1_raw_epochs', [])
        if (len(epochs) < 3 or any(type(e) is not int or e % 60 for e in epochs)
                or any(b-a != 60 for a, b in zip(epochs, epochs[1:]))):
            return result('TIME_RULE_CONFLICT', 'CURRENT_M1_SPACING_INVALID')
    pairs = list(zip(samples, samples[1:]))
    if ((previous_tick is not None and samples[0]['raw_epoch'] < previous_tick)
            or any(b['raw_epoch'] < a['raw_epoch']
                   or b['observed_utc_epoch'] <= a['observed_utc_epoch']
                   or b['m1_raw_epochs'][-1] < a['m1_raw_epochs'][-1]
                   for a, b in pairs)):
        return result('TIME_RULE_CONFLICT', 'TIMESTAMP_REVERSAL')
    advancing = [b for a, b in pairs if b['raw_epoch'] > a['raw_epoch']
                 or b['m1_raw_epochs'][-1] > a['m1_raw_epochs'][-1]]
    if not advancing:
        return result('NO_FRESH_MARKET_DATA', 'NO_FRESH_TICK')
    for sample in advancing:
        delta = sample['raw_epoch'] - sample['observed_utc_epoch']
        if math.floor(delta / 3600 + .5) * 3600 != 10800:
            return result('TIME_RULE_CONFLICT', 'OFFSET_NOT_CERTIFIED', True)
        if abs(delta - 10800) > 5:
            return result('TIME_RULE_CONFLICT', 'CLOCK_SKEW_EXCEEDED', True)
        if not 0 <= sample['raw_epoch']-sample['m1_raw_epochs'][-1] < 120:
            return result('TIME_RULE_CONFLICT', 'CURRENT_M1_NOT_CURRENT', True)
    latest = samples[-1]
    closed = [e for e in latest['m1_raw_epochs'][:-1]
              if e+60 <= latest['raw_epoch']]
    if not closed or max(closed) <= last_raw:
        return result('NO_FRESH_MARKET_DATA', 'NO_NEW_CLOSED_BAR', True)
    # The first observation may predate resumption of fresh ticks. Every sample
    # must pass the frozen checks before describing the batch as time-rule PASS.
    if any(b['raw_epoch'] <= a['raw_epoch'] for a, b in pairs):
        return result('NO_FRESH_MARKET_DATA', 'INSUFFICIENT_FRESH_SAMPLES', True)
    for sample in samples:
        delta = sample['raw_epoch']-sample['observed_utc_epoch']
        if math.floor(delta / 3600 + .5)*3600 != 10800:
            return result('TIME_RULE_CONFLICT', 'OFFSET_NOT_CERTIFIED', True)
        if abs(delta-10800) > 5:
            return result('TIME_RULE_CONFLICT', 'CLOCK_SKEW_EXCEEDED', True)
        if not 0 <= sample['raw_epoch']-sample['m1_raw_epochs'][-1] < 120:
            return result('TIME_RULE_CONFLICT', 'CURRENT_M1_NOT_CURRENT', True)
    return result('TIME_RULE_PASS', None, True)


def resume_decision(classification, original_samples_retained):
    """Current observations alone cannot establish a past pause's sole cause."""
    if classification == 'NO_FRESH_MARKET_DATA':
        return 'WAIT_FOR_FRESH_MARKET_DATA'
    if classification in {'TIME_RULE_CONFLICT', 'SOURCE_IDENTITY_CONFLICT', 'CHAIN_CONFLICT'}:
        return 'PAUSE_REMAINS_VALID'
    # A separate independently certified workflow is required even with samples.
    return 'FAIL'
