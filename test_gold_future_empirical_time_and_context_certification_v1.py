"""Synthetic-only empirical time and immutable collector rejection tests."""
import contextlib
import io
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import gold_future_empirical_time_v1 as t
import gold_future_capture_collector_v3 as c
import gold_future_capture_certification_v3 as cert


def payload(offset=10800, stamp=1600000020):
    row = dict(zip(c.FIELDS, [t.normalize(stamp+offset, offset), stamp+offset, t.IDENTITY['source_id'], 'GOLD#', 'M1', 10, 12, 9, 11, 1, 2, 0]))
    return {'identity': dict(t.IDENTITY), 'clock_quality': 'INDEPENDENTLY_VERIFIED', 'timeframe': 'M1',
            'native_timeframes': list(c.TIMEFRAMES), 'native_source': True, 'resampled': False,
            'raw_epochs': [stamp+offset], 'successor_raw_epochs': [stamp+offset+60],
            'anchors': [{'raw_epoch': stamp+offset-60, 'observed_utc_epoch': stamp-60},
                        {'raw_epoch': stamp+offset+120, 'observed_utc_epoch': stamp+120}],
            'certified_offset_seconds': offset, 'regime_certification_id': 'SYNTHETIC_'+str(offset),
            'transition_recertified': False, 'rows': [row]}


def fixture(root):
    for name in ('pending', 'quarantine', 'snapshots', 'manifests'):
        (root/name).mkdir(parents=True)
    source = cert.load(c.ROOT/'gold_future_capture_source_attestation_v3.json'); source['template'] = False
    time_doc = cert.load(c.ROOT/'gold_future_capture_time_attestation_v3.json')
    time_doc.update(template=False, status='EMPIRICALLY_CERTIFIED_SOURCE_TIME_RULE')
    freeze = {'prefix_approved': True, 'outcome_inspection_prohibited': True,
              'collector_sha256': cert.sha(c.ROOT/'gold_future_capture_collector_v3.py'),
              'runtime_validator_sha256': cert.sha(c.ROOT/'gold_future_empirical_time_v1.py'),
              'source_attestation_sha256': cert.canonical(source), 'time_attestation_sha256': cert.canonical(time_doc)}
    activation = cert.load(c.ROOT/'gold_future_capture_activation_v3.json')
    activation.update(template=False, status='ACTIVE_UNVERIFIED', capture_root=str(root), collector_commit='a'*40,
                      source_attestation_sha256=cert.canonical(source), time_attestation_sha256=cert.canonical(time_doc),
                      protocol_freeze_commit='b'*40, protocol_freeze_sha256=cert.canonical(freeze),
                      protocol_freeze_effective_at_utc='2020-01-01T00:00:00+00:00',
                      activation_effective_at_utc='2020-01-01T00:00:01+00:00', prefix_policy='PREDECLARED_CAUSAL_REINITIALIZATION')
    for name, value in [('source_attestation.json', source), ('time_attestation.json', time_doc), ('activation.json', activation), ('protocol_freeze.json', freeze)]:
        (root/name).write_bytes(c.encode(value))


def self_test():
    checks = []
    def reject(name, fn):
        try:
            fn()
        except (ValueError, KeyError, TypeError, OSError):
            checks.append(name); return
        raise AssertionError('Not rejected: '+name)
    for offset in (7200, 10800):
        assert t.runtime_validate(payload(offset))['offset'] == offset
        checks.append('observed_offset_'+str(offset))
    for offset in (3600, 14400):
        p = payload(); p['anchors'] = [{'raw_epoch': 1600000000+offset, 'observed_utc_epoch': 1600000000}, {'raw_epoch': 1600000180+offset, 'observed_utc_epoch': 1600000180}]
        reject('unexpected_offset_'+str(offset), lambda p=p: t.runtime_validate(p))
    p = payload(); p['anchors'][-1]['raw_epoch'] -= 3600
    reject('ambiguous_transition', lambda: t.runtime_validate(p))
    previous = t.runtime_validate(payload(7200))
    p = payload(10800, 1600000320)
    reject('uncertified_offset_change', lambda: t.runtime_validate(p, previous))
    p['transition_recertified'] = True
    assert t.runtime_validate(p, previous)['status'] == 'PASS'
    checks.append('recertified_allowed_regime_change')
    p = payload(); p['raw_epochs'] = [p['raw_epochs'][0], p['raw_epochs'][0]-60]; p['successor_raw_epochs'] *= 2
    reject('backward_timestamp', lambda: t.runtime_validate(p))
    p = payload(); p['anchors'][-1]['raw_epoch'] += 10
    reject('excessive_skew', lambda: t.runtime_validate(p))
    p = payload(); p['clock_quality'] = 'OPERATIONAL_CHECK_ONLY'
    reject('unverified_clock', lambda: t.runtime_validate(p))
    for field in ('server', 'symbol', 'source_id', 'broker', 'environment'):
        p = payload(); p['identity'][field] = 'WRONG'
        reject('wrong_'+field, lambda p=p: t.runtime_validate(p))
    p = payload(); result = t.runtime_validate(p)
    assert result['raw_epochs'] == p['raw_epochs'] and result == t.runtime_validate(p)
    checks.extend(['raw_epoch_preserved', 'deterministic_normalization'])
    with tempfile.TemporaryDirectory(prefix='empirical_資料_') as tmp:
        root = Path(tmp)/'chain'; fixture(root)
        schema = cert.load(c.ROOT/'gold_future_capture_manifest_schema_v3.json')
        c.append_snapshot(root, payload(), schema, c.now(), c.now())
        c.append_snapshot(root, payload(stamp=1600000620), schema, c.now(), c.now())
        entries, _, tip = c.recover_chain(root, schema)
        assert len(entries) == 2 and tip['readiness_pass']
        checks.extend(['session_gap_not_filled', 'missing_bars_not_fabricated', 'no_holiday_synthesis', 'unicode_paths'])
        before = {str(p): c.sha(p.read_bytes()) for folder in ('snapshots', 'manifests') for p in (root/folder).iterdir()}
        bad = payload(stamp=1600000920); bad['anchors'][-1]['raw_epoch'] += 3600
        reject('runtime_contradiction', lambda: c.append_snapshot(root, bad, schema, c.now(), c.now()))
        assert (root/'PAUSED_TIME_RULE.json').exists() and list((root/'quarantine').iterdir())
        assert before == {str(p): c.sha(p.read_bytes()) for folder in ('snapshots', 'manifests') for p in (root/folder).iterdir()}
        checks.extend(['paused_time_rule', 'quarantined_without_chain_addition', 'old_manifests_immutable'])
        reject('no_automatic_resume', lambda: c.append_snapshot(root, payload(stamp=1600001220), schema, c.now(), c.now()))
    import test_gold_future_causal_context_inventory_v1 as context_tests
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        context_tests.self_test()
    inherited = json.loads(out.getvalue()); assert inherited['overall'] == 'PASS'
    checks.extend('context:'+name for name in inherited['checks'])
    protocol = cert.load(c.ROOT/'gold_future_capture_protocol_v3.json')
    text = (c.ROOT/'gold_future_capture_collector_v3.py').read_text(encoding='utf-8')
    assert cert.inspect_collector(text, protocol)['static_status'] == 'STATICALLY_CONFORMANT'
    assert cert.inspect_collector(text+'\nimport xgboost\n', protocol)['static_status'] == 'NOT_STATICALLY_CONFORMANT'
    assert cert.inspect_collector(text.replace("file_under(root, 'chain_tip.json'))", "file_under(root, 'snapshots/x.json'))"), protocol)['static_status'] == 'NOT_STATICALLY_CONFORMANT'
    checks.extend(['collector_static', 'no_strategy_import', 'immutable_replace_guard'])
    prior = cert.load(c.ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json')
    assert prior['pipeline_sha256'] == cert.sha(c.ROOT/'drl_trading_v2.py')
    checks.append('macd_implementation_bound')
    s = {'identity': dict(t.IDENTITY), 'regime': 'winter', 'raw_epochs': [1600000020, 1600000080], 'anchors': []}
    assert t.classify_segment(s)['classification'] == 'EMPIRICAL_INTERNAL_CONSISTENCY'
    assert t.classify_segment(s)['observed_offset_seconds'] is None
    assert t.decision([s], {'quality': 'OPERATIONAL_CHECK_ONLY'})['status'] == 'PARTIAL'
    checks.append('unanchored_history_cannot_infer_offset')
    print(json.dumps({'overall': 'PASS', 'checks': checks, 'synthetic_only': True}))


if __name__ == '__main__':
    self_test()
