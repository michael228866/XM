"""Synthetic native-source, identity privacy, causality and persistent-pause tests."""
import ast
import contextlib
import json
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gold_future_source_identity_v4 as identity
import gold_future_native_context_fetch_v1 as fetcher
import gold_future_current_regime_diagnostic_v1 as regime
import gold_future_capture_collector_v4 as collector
import gold_future_capture_certification_v4 as cert
import validate_gold_future_native_context_v1 as validator

ROOT = Path(__file__).resolve().parent
CHECKS = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def rejects(name, action):
    try:
        action()
    except (ValueError, FileExistsError, KeyError, TypeError):
        CHECKS.append(name)
    else:
        raise AssertionError(name)


class Account:
    company = 'XM Global Limited'
    server = 'XMGlobal-MT5 6'
    trade_mode = 0

    def __repr__(self):
        raise AssertionError('Full account object exposed')

    def _asdict(self):
        raise AssertionError('Full account object exposed')


def fake_api(account=None):
    return SimpleNamespace(ACCOUNT_TRADE_MODE_DEMO=0, ACCOUNT_TRADE_MODE_REAL=2,
        ACCOUNT_TRADE_MODE_CONTEST=1, account_info=lambda: account or Account(),
        terminal_info=lambda: SimpleNamespace(connected=True, path=r'D:\XM2', build=5000),
        symbol_info=lambda _: SimpleNamespace(name='GOLD#', digits=2, point=.01), __version__='synthetic')


def samples():
    return [{'source_identity': identity.verify_identity(fake_api()), 'raw_epoch': 1800000060+i*2,
             'observed_utc_epoch': 1800000060+i*2-10800+.6,
             'm1_raw_epochs': [1799999940, 1800000000, 1800000060]} for i in range(3)]


def test_identity():
    check('safe extraction', identity.extract_identity(fake_api()) == identity.EXPECTED)
    for mode, label in [(0, 'demo'), (2, 'live'), (1, 'contest'), (8, 'unknown'), (None, 'unknown')]:
        account = Account(); account.trade_mode = mode
        check('mode '+label+str(mode), identity.extract_identity(fake_api(account))['source_account_environment'] == label)
    for attr, value in [('company', 'wrong'), ('server', 'wrong'), ('trade_mode', 2), ('trade_mode', None)]:
        account = Account(); setattr(account, attr, value)
        rejects('identity reject '+attr+str(value), lambda: identity.verify_identity(fake_api(account)))
    api = fake_api(); api.account_info = lambda: None
    rejects('account unavailable', lambda: identity.verify_identity(api))
    api.account_info = lambda: (_ for _ in ()).throw(RuntimeError('synthetic'))
    rejects('account exception sanitized', lambda: identity.verify_identity(api))
    api = fake_api(); api.symbol_info = lambda _: SimpleNamespace(name='OTHER', digits=2, point=.01)
    rejects('wrong symbol', lambda: identity.verify_identity(api))
    safe = identity.verify_identity(fake_api())
    check('full account never serialized', json.loads(json.dumps(safe)) == safe)
    check('no sensitive account keys', not validator.sensitive_values(safe))
    check('sensitive detector', validator.sensitive_values({'nested': {'balance': 123}}))
    check('deny list prose permitted', not validator.sensitive_values({'documentation': 'balance forbidden'}))


def test_time():
    check('10800 accepted', regime.check_samples(samples())['status'] == 'PASS')
    for offset in [7200, 14400]:
        changed = samples()
        for s in changed:
            s['observed_utc_epoch'] = s['raw_epoch']-offset
        rejects('offset '+str(offset), lambda: regime.check_samples(changed))
    changed = samples(); changed[-1]['observed_utc_epoch'] += 6
    rejects('skew', lambda: regime.check_samples(changed))
    changed = samples(); changed[-1] = deepcopy(changed[0])
    rejects('reversal', lambda: regime.check_samples(changed))
    changed = samples(); changed[-1]['source_identity']['broker_server'] = 'wrong'
    rejects('source change', lambda: regime.check_samples(changed))
    changed = samples(); changed[-1]['observed_utc_epoch'] += 3600
    rejects('ambiguous transition', lambda: regime.check_samples(changed))
    rejects('insufficient observations', lambda: regime.check_samples(samples()[:2]))


def synthetic_rates(tf, count):
    if tf == 'Monthly':
        epochs = [int(datetime(2027+(i//12), 1+(i%12), 1, tzinfo=timezone.utc).timestamp()) for i in range(count)]
    else:
        duration = 86400 if tf == 'Daily' else 604800 if tf == 'Weekly' else int(tf[1:])*(60 if tf[0] == 'M' else 3600)
        epochs = [1800000000+i*duration for i in range(count)]
    return [dict(time=e, open=10., high=12., low=9., close=11., spread=1, tick_volume=2, real_volume=0) for e in epochs]


def context_fixture(root):
    api = fake_api()
    current = [0]
    def rates(_, constant, pos, count):
        tf = next(k for k, v in fetcher.TIMEFRAMES.items() if v == constant)
        result = synthetic_rates(tf, count)
        current[0] = result[-1]['time']+10
        return result
    api.copy_rates_from_pos = rates
    api.symbol_info_tick = lambda _: SimpleNamespace(time=current[0])
    @contextlib.contextmanager
    def session():
        yield api, identity.verify_identity(api)
    with patch.object(fetcher, 'session', session), patch.object(fetcher.subprocess, 'check_output', return_value=Path(fetcher.__file__).read_bytes()), patch.object(fetcher.time, 'time', side_effect=lambda: current[0]-10800+.6):
        return fetcher.fetch(root, 'synthetic')


def test_context(root):
    inventory = context_fixture(root)
    script_sha = validator.digest(Path(fetcher.__file__).read_bytes())
    result = validator.validate(root, 'synthetic', script_sha)
    check('native 21', result['native_timeframe_count'] == 21)
    check('M1 4096', result['m1_context_bars'] == 4096)
    check('HTF 21', all(e['row_count'] == 21 for e in inventory['entries'][1:]))
    check('raw epochs retained', all(e['last_raw_epoch'] for e in inventory['entries']))
    check('deterministic serialization', fetcher.encode(inventory) == fetcher.encode(deepcopy(inventory)))
    check('hash stability', fetcher.sha(fetcher.encode(inventory)) == fetcher.sha(fetcher.encode(deepcopy(inventory))))
    rejects('context immutable', lambda: fetcher.fetch(root, 'synthetic'))
    rejects('post-boundary context', lambda: validator.validate(root, 'synthetic', script_sha, '2000-01-01T00:00:00+00:00'))
    rates = synthetic_rates('M1', 4100)
    closed = fetcher.select_closed(rates, 'M1', rates[-1]['time']+10)
    check('forming excluded', closed[-1]['RAW_SOURCE_EPOCH'] == rates[-2]['time'])
    for tf, count in [('M1', 4096), ('H1', 21)]:
        short = synthetic_rates(tf, count)
        rejects('short '+tf, lambda: fetcher.select_closed(short, tf, short[-1]['time']+10))
    for label, changed in [('duplicate', rates[:-1]+[rates[-2]]), ('reversal', rates[::-1])]:
        rejects(label, lambda: fetcher.select_closed(changed, 'M1', rates[-1]['time']+10))
    bad = deepcopy(rates); bad[0]['high'] = 1
    rejects('malformed OHLC', lambda: fetcher.select_closed(bad, 'M1', rates[-1]['time']+10))
    prefix = cert.prefix_decision(result, inventory)
    check('prefix hash and native gates', prefix['approved'])
    check('prefix fails without native', not cert.prefix_decision({'status': 'PARTIAL'}, inventory)['approved'])
    for name, expected in [('historically_equivalent', False), ('outcome_tuning', False), ('context_role', 'CAUSAL_CONTEXT_ONLY'), ('holdout_evidence', False)]:
        check('prefix '+name, prefix[name] == expected)
    entry_path = root/'manifests'/'M1.json'
    original = entry_path.read_bytes()
    bad_entry = validator.load(entry_path); bad_entry['resampled'] = True
    entry_path.write_bytes(validator.encode(bad_entry))
    rejects('resampling manifest rejected', lambda: validator.validate(root, 'synthetic', script_sha))
    entry_path.write_bytes(original)  # Disposable synthetic fixture only.
    return inventory


def test_collector_inner(root):
    root.mkdir()
    for directory in ('protocol', 'attestations', 'pending', 'snapshots', 'manifests', 'quarantine'):
        (root/directory).mkdir()
    context = context_fixture(root/'context_seed')
    # Synthetic timestamps are in the future so sealed context precedes freeze.
    freeze = {'protocol_freeze_effective_at_utc': '2027-01-01T00:00:00+00:00',
              'prefix_approved': True, 'runtime_fail_closed': True,
              'certified_offsets_seconds': [10800], 'uncertified_candidate_offsets_seconds': [7200],
              'outcome_inspection_prohibited': True, 'code_sha256': {},
              'manifest_schema_sha256': collector.sha((ROOT/'gold_future_capture_manifest_schema_v4.json').read_bytes()),
              'context_inventory_root_sha256': context['inventory_root_sha256']}
    activation = {'status': 'ACTIVE', 'collector_version': collector.VERSION,
                  'protocol_freeze_commit': 'synthetic', 'protocol_freeze_sha256': collector.sha(collector.encode(freeze)),
                  'activation_effective_at_utc': '2027-01-01T00:00:01+00:00'}
    for path, value in [('protocol/freeze.json', freeze), ('attestations/activation.json', activation),
                        ('attestations/recertification.json', {'status': 'READY_FOR_CAPTURE_ACTIVATION', 'freeze_sha256': activation['protocol_freeze_sha256']})]:
        (root/path).write_bytes(collector.encode(value))
    raw = 1800000000
    row = dict(zip(collector.FIELDS, [raw, datetime.fromtimestamp(raw-10800, timezone.utc).isoformat(),
        'XMGlobal-MT5-6_GOLD', 'GOLD#', 'M1', 10., 12., 9., 11., 1, 2, 0]))
    payload = {'samples': samples(), 'rows': [row], 'returned_raw_epochs': [raw, raw+60],
               'native_source_confirmed': True, 'resampled': False, 'timeframe': 'M1', 'mt5_timeframe_constant': 1}
    check('future M1 payload', collector.validate_payload(payload, freeze, activation)['status'] == 'PASS')
    bad_activation = deepcopy(activation); bad_activation['activation_effective_at_utc'] = row['SOURCE_TIMESTAMP']
    rejects('boundary strictly after activation', lambda: collector.validate_payload(payload, freeze, bad_activation))
    bad_freeze = deepcopy(freeze); bad_freeze['protocol_freeze_effective_at_utc'] = row['SOURCE_TIMESTAMP']
    rejects('boundary strictly after freeze', lambda: collector.validate_payload(payload, bad_freeze, activation))
    activation_path = root/'attestations'/'activation.json'
    original = activation_path.read_bytes()
    bad_activation = deepcopy(activation); bad_activation['protocol_freeze_commit'] = None
    activation_path.write_bytes(collector.encode(bad_activation))
    rejects('freeze required', lambda: collector.binding(root))
    bad_activation = deepcopy(activation); bad_activation['activation_effective_at_utc'] = freeze['protocol_freeze_effective_at_utc']
    activation_path.write_bytes(collector.encode(bad_activation))
    rejects('activation follows freeze', lambda: collector.binding(root))
    activation_path.write_bytes(original)
    first = collector.append_snapshot(root, payload)
    check('sealed manifest chain', first['sequence'] == 0 and len(collector.recover_chain(root)[0]) == 1)
    before = (root/'snapshots'/'000000000000.json').read_bytes()
    bad = deepcopy(payload)
    for s in bad['samples']:
        s['observed_utc_epoch'] = s['raw_epoch']-7200
    rejects('7200 runtime persistent pause', lambda: collector.append_snapshot(root, bad))
    check('quarantine preserved', (root/'PAUSED_TIME_RULE.json').exists() and bool(list((root/'quarantine').iterdir())))
    rejects('no automatic resume', lambda: collector.append_snapshot(root, payload))
    check('prior evidence unchanged after pause', (root/'snapshots'/'000000000000.json').read_bytes() == before)


def test_collector(root):
    with patch.object(collector.subprocess, 'check_output', side_effect=lambda *a, **kw: (root/'protocol'/'freeze.json').read_bytes()):
        test_collector_inner(root)


def main():
    test_identity(); test_time()
    with tempfile.TemporaryDirectory(prefix='gold-v4-測試-') as name:
        test_context(Path(name)/'context_seed')
        test_collector(Path(name)/'capture')
        check('Unicode paths', '測試' in name)
    spec = validator.load(ROOT/'execution_spec_gold_future_native_context_and_current_regime_activation_v1.json')
    for path, expected in spec['protected_sha256'].items():
        check('protected '+path, validator.digest((ROOT/path).read_bytes()) == expected)
    check('prior seals unchanged', all(validator.digest((ROOT/p).read_bytes()) == h for p, h in spec['prior_seals'].items()))
    protocol = validator.load(ROOT/'gold_future_capture_protocol_v4.json')
    check('no model imports or execution', cert.static_review(protocol)['status'] == 'STATICALLY_CONFORMANT')
    print(json.dumps({'status': 'PASS', 'check_count': len(CHECKS), 'checks': CHECKS}, ensure_ascii=False))


if __name__ == '__main__':
    main()
