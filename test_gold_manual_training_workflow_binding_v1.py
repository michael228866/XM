"""Small synthetic infrastructure tests. Never imports a model implementation."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import gold_historical_training_data_manager_v1 as data
import gold_manual_training_workflow_v1 as workflow
import manual_training_launcher_v1 as launcher
from training_holdout_guard_v1 import HOLDOUT_START, check_interval, check_path

ROOT = Path(__file__).resolve().parent


def tests():
    checks = {}

    def reject(name, action):
        try:
            action()
        except (ValueError, PermissionError, UnicodeError):
            checks[name] = True
        else:
            checks[name] = False

    state = workflow.binding_status()
    checks['unresolved_default_fails_closed'] = state['workflow'] == 'NOT_READY'
    checks['no_real_auto_fetch'] = state['auto_fetch'] == 'DISABLED'
    trace = json.loads((ROOT/'gold_manual_training_provenance_v1.json').read_text(encoding='utf-8'))
    checks['provenance_candidates_retained'] = len(trace['candidates']) == 4 and trace['selected_default'] is None
    for candidate in trace['candidates']:
        label = candidate['script']['path']
        checks['script_hash:' + label] = data.sha((ROOT/label).read_bytes()) == candidate['script']['sha256']
        checks['label_hash:' + label] = data.sha((ROOT/candidate['label_pipeline']['path']).read_bytes()) == candidate['label_pipeline']['sha256']
        checks['features_hash:' + label] = data.sha(json.dumps(candidate['feature_list'], separators=(',', ':')).encode()) == candidate['feature_list_sha256']
        for item in candidate['feature_pipeline']:
            checks['pipeline_hash:' + item['path']] = data.sha((ROOT/item['path']).read_bytes()) == item['sha256']
        if 'config' in candidate:
            checks['config_hash:' + label] = data.sha((ROOT/candidate['config']['path']).read_bytes()) == candidate['config']['sha256']
    req = {'symbol': 'GOLD#', 'source_id': 'XMGlobal-MT5-6_GOLD', 'timeframe': 'M1',
           'native_or_derived': 'NATIVE', 'timestamp_semantics': 'UTC', 'coverage_certified': True,
           'start': 60, 'end': 180, 'cutoff': 180, 'expected_timestamps': [60, 120, 180],
           'api_timestamp_offset_seconds': 0,
           'broker_identity': {'company': 'SYNTHETIC', 'server': 'SYNTHETIC', 'trade_mode': 0}}
    meta = {k: req[k] for k in ('symbol', 'source_id', 'timeframe', 'native_or_derived', 'timestamp_semantics')}
    meta['source_type'] = 'MT5_NATIVE_API'
    rows = [dict(zip(data.COLUMNS, [t, 10., 12., 9., 11., 7, 2, 0])) for t in req['expected_timestamps']]
    api_calls = []
    fake = SimpleNamespace(
        initialize=lambda **kwargs: True,
        account_info=lambda: SimpleNamespace(**req['broker_identity']),
        symbol_info=lambda symbol: object() if symbol == 'GOLD#' else None,
        symbol_select=lambda symbol, selected: symbol == 'GOLD#' and selected,
        TIMEFRAME_M1=1,
        copy_rates_range=lambda symbol, timeframe, start, end: api_calls.append((symbol, timeframe, int(start.timestamp()), int(end.timestamp()))) or rows,
        shutdown=lambda: api_calls.append('shutdown'))
    with patch.dict(sys.modules, {'MetaTrader5': fake}):
        checks['mt5_adapter_exact_native_fetch'] = data.mt5_fetch(req, 60, 180) == rows and api_calls == [('GOLD#', 1, 60, 180), 'shutdown']
        with patch.object(fake, 'symbol_info', return_value=None):
            reject('mt5_no_symbol_substitution', lambda: data.mt5_fetch(req, 60, 180))
        with patch.object(fake, 'account_info', return_value=SimpleNamespace(company='WRONG', server='WRONG', trade_mode=0)):
            reject('mt5_wrong_identity', lambda: data.mt5_fetch(req, 60, 180))
        checks['mt5_shutdown_on_error'] = api_calls[-2:] == ['shutdown', 'shutdown']
    checks['complete_valid'] = data.validate_rows(rows, req, meta, complete=True) == []
    checks['partial_missing_interval'] = data.intervals(data.validate_rows(rows[:1], req, meta), req['expected_timestamps']) == [(120, 180)]
    for name, changed in [('wrong_symbol', {'symbol': 'GAUCNH#'}), ('wrong_timeframe', {'timeframe': 'H1'}),
                          ('wrong_source', {'source_id': 'OTHER'}), ('unknown_source', {'source_type': 'UNKNOWN'}),
                          ('derived_not_native', {'native_or_derived': 'DERIVED'})]:
        reject(name, lambda changed=changed: data.validate_rows(rows, req, {**meta, **changed}))
    for name, values in [('duplicates', rows + [rows[-1]]), ('non_monotonic', rows[::-1]),
                         ('schema', [{**rows[0], 'extra': 1}]),
                         ('ohlc', [{**rows[0], 'high': 1}]), ('nan', [{**rows[0], 'open': float('nan')}]),
                         ('negative_spread', [{**rows[0], 'spread': -1}]),
                         ('negative_volume', [{**rows[0], 'tick_volume': -1}]),
                         ('off_grid_time', [{**rows[0], 'time': 61}])]:
        reject(name, lambda values=values: data.validate_rows(values, req, meta))
    reject('uncertified_coverage', lambda: data.requirement({**req, 'coverage_certified': False}))
    reject('missing_clock_mapping', lambda: data.requirement({**req, 'api_timestamp_offset_seconds': None}))
    reject('post_cutoff', lambda: check_interval(60, 181, 180))
    reject('holdout_timestamp', lambda: check_interval(60, HOLDOUT_START, HOLDOUT_START))
    for part in ('snapshots', 'manifests', 'context_seed'):
        reject('holdout_path:' + part, lambda part=part: check_path(ROOT/'future_holdout/gold_s4_v4'/part, ROOT))
    with tempfile.TemporaryDirectory(prefix='manual_訓練_') as directory:
        root = Path(directory)
        cache = root/'historical_training_data'
        calls = []

        def fetch(requirements, start, end):
            calls.append((requirements['symbol'], requirements['timeframe'], start, end))
            return [r for r in rows if start <= r['time'] <= end]

        data.ensure_dataset(req, rows, meta, cache, fetch, root)
        checks['complete_no_fetch'] = calls == [] and not cache.exists()
        received, sealed = data.ensure_dataset(req, rows[:1], meta, cache, fetch, root)
        checks['missing_fetch_attempted_exact_symbol_native_tf'] = calls == [('GOLD#', 'M1', 120, 180)]
        checks['successful_fetch_sealed'] = received == rows and Path(sealed['path']).is_file() and sealed['training_eligible']
        checks['unicode_csv_roundtrip'] = data.read_csv(sealed['path'], req, sealed, root) == rows
        reject('csv_bad_hash', lambda: data.read_csv(sealed['path'], req, {**sealed, 'sha256': 'bad'}, root))
        imported = {**sealed, 'source_type': 'USER_IMPORTED_CERTIFIED_FILE'}
        checks['certified_import_fallback'] = data.read_csv(sealed['path'], req, imported, root) == rows
        try:
            data.ensure_dataset(req, [], meta, cache, lambda *_: [], root)
        except ValueError as error:
            checks['empty_fetch_rejected'] = 'MT5 無法取得這段資料' in str(error)
        try:
            def unavailable(*args):
                raise ValueError('unavailable')
            data.ensure_dataset(req, [], meta, cache, unavailable, root)
        except ValueError as error:
            checks['unavailable_visible_import_hint'] = all(x in str(error) for x in ['GOLD#', 'M1', '60 ~ 180', 'import', 'RUN_TRAINING.bat'])
        reject('fetch_wrong_order', lambda: data.ensure_dataset(req, [], meta, cache, lambda *_: rows[::-1], root))
        bad = root/'bad.csv'
        bad.write_text('bad,schema\n1,2\n', encoding='utf-8')
        reject('import_schema', lambda: data.read_csv(bad, req, {**meta, 'sha256': data.sha(bad.read_bytes())}, root))
        bad.write_bytes(b'\xff\xff')
        reject('import_encoding', lambda: data.read_csv(bad, req, {**meta, 'sha256': data.sha(bad.read_bytes())}, root))
    reject('receipt_missing', lambda: workflow.consume_receipt('not-issued'))
    with patch.object(launcher, 'user_double_click', return_value=True):
        token = workflow.issue_receipt()
        workflow.consume_receipt(token)
        checks['receipt_valid'] = True
        reject('receipt_replay', lambda: workflow.consume_receipt(token))
        token = workflow.issue_receipt()
        with patch.object(workflow.time, 'monotonic', return_value=float('inf')):
            reject('receipt_expiry', lambda: workflow.consume_receipt(token))
        token = workflow.issue_receipt()
        with patch.object(workflow.os, 'getpid', return_value=-1):
            reject('receipt_wrong_process', lambda: workflow.consume_receipt(token))
        token = workflow.issue_receipt()
        reject('manual_request_still_blocks_unapproved_training', lambda: workflow.run_manual(token))
    with patch.object(launcher, 'user_double_click', return_value=False):
        reject('automated_receipt_denied', workflow.issue_receipt)
    config = launcher.load(launcher.CONFIG)
    launcher.verify_environment(config)
    checks['protected_production_hashes'] = True
    with patch.object(launcher.sys, 'executable', str(ROOT/'wrong-python.exe')):
        reject('wrong_python', lambda: launcher.verify_environment(config))
    bat = (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['bat_no_args_root_python_pause'] = all(t in bat for t in ['cd /d "%~dp0"', '.venv\\Scripts\\python.exe', 'pause >nul', 'PYTHONUTF8=1']) and '%1' not in bat
    result = subprocess.run([sys.executable, '-B', str(ROOT/'manual_training_launcher_v1.py'), '--dry-run'], capture_output=True, encoding='utf-8', timeout=20)
    checks['dry_run_never_trains'] = result.returncode == 0 and '"training_executed": false' in result.stdout
    env = {**os.environ, 'PYTHONUTF8': '1'}
    result = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT/'RUN_TRAINING.bat')], input='\r\n', capture_output=True, encoding='utf-8', errors='replace', env=env, timeout=30)
    checks['bat_automated_failure_visible'] = '[失敗]' in result.stderr and '請按任意鍵關閉' in result.stdout
    code = 'from training_holdout_guard_v1 import install; from pathlib import Path; install(Path.cwd()); open("future_holdout/gold_s4_v4/snapshots/forbidden")'
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, env=env, timeout=10)
    checks['audit_hook_denies_before_open'] = result.returncode != 0 and b'PermissionError' in result.stderr
    checks['success_summary_format'] = all(x in workflow.summary('synthetic', 'PASS', 'fixture') for x in ['RUN_ID:\nsynthetic', 'STATUS:\nPASS', '輸出位置:\nfixture'])
    checks['status_fields'] = set(state) >= {'workflow', 'owner', 'auto_fetch', 'symbol', 'historical_data', 'last_training'}
    checks['no_model_import'] = not any(x in sys.modules for x in ['xgboost', 'sklearn', 'gold_secondary_s4_confirmation_v1'])
    return {'overall': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'test_count': len(checks), 'model_training_executed': False,
            'unresolved_requirements': ['Approved default', 'Live data/clock equivalence', 'Full user-training adapter'],
            'data_fetch_test': 'SYNTHETIC_ONLY; live MT5 not called'}


if __name__ == '__main__':
    result = tests()
    print(json.dumps(result, ensure_ascii=True))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)
