"""Synthetic freshness/clock boundaries; no live access or strategy execution."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gold_future_fresh_market_diagnostic_v1 import assess
from validate_gold_future_fresh_market_resume_certification_v1 import verify_samples


def tests():
    raw = 1800000000
    identity = {'broker_company': 'XM Global Limited', 'broker_server': 'XMGlobal-MT5 6',
                'source_account_environment': 'demo', 'source_id': 'XMGlobal-MT5-6_GOLD',
                'symbol': 'GOLD#', 'digits': 2, 'point': .01}
    base = [{'source_identity': identity.copy(), 'raw_tick_epoch': raw+i*2,
             'system_utc_epoch': raw+i*2-10800+.5,
             'm1_raw_epochs': [raw-120, raw-60, raw]} for i in range(5)]
    checks = {}
    def run(name, samples, state, **kwargs):
        for sample in samples:
            sample['sample_time_system_utc'] = datetime.fromtimestamp(sample['system_utc_epoch'], timezone.utc).isoformat()
        result = assess(samples, kwargs.get('previous_tick', raw-300), kwargs.get('last_accepted_bar', raw-180))
        checks[name] = result['classification'] == state
        assert checks[name], (name, result)
        verified = verify_samples(samples, result, kwargs.get('previous_tick', raw-300), kwargs.get('last_accepted_bar', raw-180))
        assert all(verified.values()), (name, verified)
        return result
    fresh = run('fresh_10800', base, 'TIME_RULE_PASS')
    checks['fresh_acceptable_skew'] = fresh['current_regime_status'] == 'PASS' and fresh['normalized_clock_error_seconds_min'] == -.5
    stale = copy.deepcopy(base)
    for s in stale:
        s['raw_tick_epoch'] = raw
        s['system_utc_epoch'] += 86400
    old = run('stale_tick_idle', stale, 'NO_FRESH_MARKET_DATA')
    checks['stale_no_offset_inference'] = all(r['inferred_offset_seconds'] is None for r in old['samples'])
    checks['stale_no_clock_inference'] = all(r['normalized_clock_error_seconds'] is None for r in old['samples'])
    checks['stale_no_pause'] = all(r['failed_rule'] is None for r in old['samples'])
    tampered = copy.deepcopy(old)
    tampered['samples'][0]['inferred_offset_seconds'] = 10800
    checks['independent_validator_rejects_stale_offset'] = not all(verify_samples(stale, tampered, raw-300, raw-180).values())
    checks['stale_no_snapshot'] = old['evidence_admitted'] is False
    initial = copy.deepcopy(stale)
    assess(stale, raw-300, raw-180)
    checks['idle_input_chain_unchanged'] = stale == initial
    old_bar = copy.deepcopy(base)
    for s in old_bar: s['m1_raw_epochs'] = [raw-600, raw-540, raw-480]
    run('stale_closed_M1_idle', old_bar, 'NO_FRESH_MARKET_DATA')
    for offset, state in [(7200, 'UNCERTIFIED_CANDIDATE_REGIME'), (14400, 'TIME_RULE_CONFLICT'), (0, 'TIME_RULE_CONFLICT')]:
        samples = copy.deepcopy(base)
        for s in samples: s['system_utc_epoch'] = s['raw_tick_epoch']-offset+.5
        run('fresh_offset_'+str(offset), samples, state)
    skew = copy.deepcopy(base)
    for s in skew: s['system_utc_epoch'] += 6
    run('fresh_excessive_skew', skew, 'TIME_RULE_CONFLICT')
    mismatch = copy.deepcopy(stale); mismatch[0]['source_identity']['broker_server'] = 'other'
    run('identity_mismatch', mismatch, 'TIME_RULE_CONFLICT')
    reverse = copy.deepcopy(base); reverse[-1]['raw_tick_epoch'] = raw-1
    run('timestamp_reversal', reverse, 'TIME_RULE_CONFLICT')
    run('cross_cycle_reversal', stale, 'TIME_RULE_CONFLICT', previous_tick=raw+1)
    repeated = copy.deepcopy(base)
    for s in repeated: s['raw_tick_epoch'] = raw
    duplicate = assess(repeated, raw-300, raw-60)
    checks['duplicate_not_certified'] = duplicate['current_regime_status'] != 'PASS' and duplicate['new_closed_m1_count'] == 0
    checks['no_synthetic_bars'] = all(r['raw_closed_bar_epoch'] in s['m1_raw_epochs'][:-1] for r, s in zip(fresh['samples'], base))
    checks['no_backfill_admission'] = fresh['evidence_admitted'] is False and fresh['resume_authorized'] is False
    boundary_age = copy.deepcopy(stale)
    for i,s in enumerate(boundary_age): s['system_utc_epoch'] = raw-10800+120+i*.01
    boundary = assess(boundary_age, raw-300, raw-180)
    checks['freshness_threshold_exact'] = boundary['samples'][0]['fresh'] is True and boundary['samples'][1]['fresh'] is False
    root = Path(__file__).resolve().parent
    spec = json.loads((root/'execution_spec_gold_future_fresh_market_resume_certification_v1.json').read_text())
    for name, inventory in [('boundary_context_chain_frozen', spec['preserved_sha256']), ('production', spec['protected_sha256'])]:
        checks[name+'_unchanged'] = all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h for n,h in inventory.items())
    checks['strategy_model_not_loaded'] = not any(n.split('.')[0] in {'xgboost','lightgbm','catboost','sklearn','joblib','pickle','gemini','drl_trading_v2'} for n in __import__('sys').modules)
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    return {'overall': 'PASS', 'test_count': len(checks), 'checks': checks,
            'scope': 'Diagnostic-only tests and frozen-file integrity; no operational amendment or healthy-idle proof'}


if __name__ == '__main__':
    print(json.dumps(tests(), sort_keys=True))
