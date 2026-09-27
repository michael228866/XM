"""Synthetic diagnostic regressions; no MT5, strategy or runtime mutation."""
import copy
import hashlib
import json
from pathlib import Path

from gold_future_capture_pause_adjudication_v1 import IDENTITY, classify, resume_decision
from gold_future_current_regime_diagnostic_v1 import check_samples

ROOT = Path(__file__).resolve().parent


def tests():
    checks = {}
    raw = 1800000000
    base = [{'source_identity': dict(IDENTITY), 'raw_epoch': raw+i*2,
             'observed_utc_epoch': raw+i*2-10800+.5,
             'm1_raw_epochs': [raw-120, raw-60, raw]} for i in range(3)]
    def check(name, samples, state, rule=None, last=raw-120, **kwargs):
        answer = classify(samples, last, **kwargs)
        checks[name] = answer['classification'] == state and answer['failed_rule'] == rule
        assert checks[name], (name, answer)
        assert answer['evidence_admitted'] is False and answer['resume_authorized'] is False
    check('fresh_10800', base, 'TIME_RULE_PASS')
    stale = copy.deepcopy(base)
    for s in stale:
        s['raw_epoch'] = raw
        s['observed_utc_epoch'] += 86400
    check('stale_10800_idle', stale, 'NO_FRESH_MARKET_DATA', 'NO_FRESH_TICK')
    older = copy.deepcopy(stale)
    for s in older:
        s['observed_utc_epoch'] += 86400*10
    check('age_alone_not_conflict', older, 'NO_FRESH_MARKET_DATA', 'NO_FRESH_TICK')
    for offset in (7200, 0, 14400):
        samples = copy.deepcopy(base)
        for s in samples:
            s['observed_utc_epoch'] = s['raw_epoch']-offset+.5
        check('fresh_offset_'+str(offset), samples, 'TIME_RULE_CONFLICT', 'OFFSET_NOT_CERTIFIED')
    skew = copy.deepcopy(base)
    for s in skew:
        s['observed_utc_epoch'] += 6
    check('fresh_skew', skew, 'TIME_RULE_CONFLICT', 'CLOCK_SKEW_EXCEEDED')
    reverse = copy.deepcopy(base); reverse[-1]['raw_epoch'] = raw-1
    check('tick_reversal', reverse, 'TIME_RULE_CONFLICT', 'TIMESTAMP_REVERSAL')
    check('cross_cycle_reversal', stale, 'TIME_RULE_CONFLICT', 'TIMESTAMP_REVERSAL', previous_tick=raw+1)
    identity = copy.deepcopy(stale); identity[-1]['source_identity']['broker_server'] = 'other'
    check('stale_identity_conflict', identity, 'SOURCE_IDENTITY_CONFLICT', 'SOURCE_IDENTITY_CHANGED')
    check('no_new_closed_m1', base, 'NO_FRESH_MARKET_DATA', 'NO_NEW_CLOSED_BAR', last=raw-60)
    check('chain_conflict', stale, 'CHAIN_CONFLICT', 'CHAIN_CONFLICT', chain_ok=False)
    check('insufficient_samples', base[:1], 'UNKNOWN', 'INSUFFICIENT_CURRENT_OBSERVATIONS')
    invalid = copy.deepcopy(base); invalid[0]['observed_utc_epoch'] = float('nan')
    check('nonfinite_clock', invalid, 'TIME_RULE_CONFLICT', 'INVALID_CLOCK_SAMPLE')
    spacing = copy.deepcopy(base); spacing[-1]['m1_raw_epochs'][0] -= 60
    check('invalid_spacing', spacing, 'TIME_RULE_CONFLICT', 'CURRENT_M1_SPACING_INVALID')
    original = copy.deepcopy(stale)
    first = classify(stale, raw-120); second = classify(stale, raw-120)
    checks['idle_no_evidence_no_synthetic_no_duplicate'] = first == second and not first['evidence_admitted']
    checks['idle_inputs_unchanged'] = stale == original
    checks['missing_historical_samples_no_resume'] = resume_decision(first['classification'], False) == 'WAIT_FOR_FRESH_MARKET_DATA'
    try:
        check_samples(stale)
    except ValueError as error:
        checks['frozen_stale_offset_failure_reproduced'] = str(error) == 'PAUSE_AND_QUARANTINE'
    else:
        checks['frozen_stale_offset_failure_reproduced'] = False
    spec = json.loads((ROOT/'execution_spec_gold_future_capture_pause_adjudication_v1.json').read_text())
    for label, inventory in [('frozen_boundary_context_runtime', spec['preserved_sha256']),
                             ('production', spec['protected_sha256'])]:
        checks[label+'_unchanged'] = all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h
                                        for n, h in inventory.items())
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    return {'overall': 'PASS', 'checks': checks, 'test_count': len(checks),
            'scope': 'Synthetic diagnostic classifier plus byte integrity; no runtime amendment',
            'runtime_integration_tests': 'BLOCKED_PENDING_ORIGINAL_PAUSE_CAUSAL_PROOF',
            'healthy_idle_supervisor_proven': False}


if __name__ == '__main__':
    print(json.dumps(tests(), sort_keys=True))
