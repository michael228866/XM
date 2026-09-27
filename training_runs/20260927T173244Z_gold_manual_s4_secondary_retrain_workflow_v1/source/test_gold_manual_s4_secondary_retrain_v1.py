"""Synthetic-only tests; XGBoost fitting is replaced by an in-memory fake."""
import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import gold_manual_s4_training_data_v1 as data
import gold_manual_s4_training_output_adapter_v1 as adapter
import gold_manual_s4_secondary_retrain_v1 as trainer
import gold_manual_training_workflow_v1 as launch
import manual_training_launcher_v1 as launcher
import validate_gold_manual_s4_training_run_v1 as validator
from training_holdout_guard_v1 import check_path, check_interval, HOLDOUT_START

ROOT = Path(__file__).resolve().parent


def frozen_functions(fake_xgb):
    """Execute only selected pure/fit functions with a fake classifier, no imports."""
    source = ROOT/'gold_independent_secondary_classifier_v1.py'
    wanted = {'require', 'probabilities', 'candidate_gate', 'training_subset', 'fit_secondary', 'array_hash'}
    nodes = [n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = {'np': np, 'xgb': fake_xgb, 'hashlib': hashlib, 'CANDIDATES': (('S4_SECONDARY_P075', .75),)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in wanted})


def tests():
    checks = {}
    def reject(name, action):
        try:
            action()
        except (ValueError, PermissionError, FileExistsError, FileNotFoundError):
            checks[name] = True
        else:
            checks[name] = False
    config = data.read(trainer.CONFIG)
    frozen = trainer.verify_config(config)
    checks['provenance_discovery'] = config['discovery_run'].endswith('20260915T151723Z_gold_independent_secondary_classifier_v1')
    checks['confirmation_not_trainer'] = 'confirmation' not in trainer.train_folds.__code__.co_names
    checks['symbol_GOLD_not_GAUCNH'] = config['training_symbol'] == 'GOLD#'
    checks['features_order_exact'] = config['feature_list'] == frozen['features'] and len(config['feature_list']) == 31
    checks['features_hash_exact'] = hashlib.sha256(json.dumps(config['feature_list'], separators=(',', ':')).encode()).hexdigest() == config['feature_list_sha256']
    checks['label_binding'] = data.sha(ROOT/config['label_pipeline']) == config['label_pipeline_sha256']
    checks['seed_fold_parameters'] = config['random_seed'] == 42 and config['fold_definition'] == frozen['folds'] and config['hyperparameters'] == frozen['parameters']
    checks['data_decision'] = data.read(ROOT/'gold_manual_s4_dataset_decision_v1.json')['status'] == 'PASS'
    for item in config['required_datasets']:
        checks['legacy_hash:'+item['timeframe']] = data.validate_file(ROOT/item['filename'], item)['training_eligible']
    reject('wrong_symbol_config', lambda: trainer.verify_config({**config, 'training_symbol': 'GAUCNH#'}))
    reject('threshold_drift', lambda: trainer.verify_config({**config, 'secondary_threshold': .8}))
    reject('seed_drift', lambda: trainer.verify_config({**config, 'random_seed': 43}))
    reject('fold_drift', lambda: trainer.verify_config({**config, 'fold_definition': []}))
    reject('source_hash_drift', lambda: trainer.verify_config({**config, 'source_bindings': {'drl_trading_v2.py': '0'*64}}))
    for part in ('snapshots', 'manifests', 'context_seed'):
        reject('holdout:'+part, lambda part=part: check_path(ROOT/'future_holdout/gold_s4_v4'/part, ROOT))
    reject('post_cutoff', lambda: check_interval(0, HOLDOUT_START, HOLDOUT_START))
    reject('output_production_root', lambda: adapter.model_path(ROOT, '2023_2024'))
    fits = []
    class FakeModel:
        def __init__(self, **kwargs):
            self.parameters = kwargs
            self.fitted = False
        def fit(self, x, y, sample_weight):
            self.fitted = True
            fits.append({'rows': len(x), 'features': list(x.columns), 'target': y.copy(), 'weights': sample_weight.copy(), 'parameters': self.parameters})
        def load_model(self, path):
            assert Path(path).read_text() == 'SYNTHETIC_B0'
        def get_booster(self):
            return SimpleNamespace(feature_names=config['feature_list'], num_boosted_rounds=lambda: 220)
        def save_model(self, path):
            Path(path).write_text('SYNTHETIC_TEST_DOUBLE_NOT_A_MODEL', encoding='utf-8')
        def predict_proba(self, x):
            p = np.full(len(x), .8 if self.fitted else .2, dtype=np.float32)
            return np.column_stack([1-p, p])
    fake = SimpleNamespace(__version__='3.2.0', XGBClassifier=FakeModel)
    library = frozen_functions(fake)
    primary, secondary, union = library.candidate_gate(np.array([.749, .75, .9, .1]), np.array([.75, 1., 1., .749]), ('S4_SECONDARY_P075', .75))
    checks['exact_conditioning_boundary'] = primary.tolist() == [False, True, True, False] and secondary.tolist() == [True, False, False, False]
    checks['gate_no_overlap'] = not (primary & secondary).any() and np.array_equal(primary | secondary, union)
    reject('feature_order_drift', lambda: library.fit_secondary(pd.DataFrame(np.ones((2, 31), np.float32), columns=config['feature_list'][::-1]), np.array([0, 1], np.int8), frozen))
    reject('one_class_rejected', lambda: library.fit_secondary(pd.DataFrame(np.ones((2, 31), np.float32), columns=config['feature_list']), np.array([0, 0], np.int8), frozen))
    with tempfile.TemporaryDirectory(prefix='s4_訓練_') as tmp:
        root = Path(tmp)
        run = root/'training_runs/20200101T000000Z_gold_manual_s4_secondary_retrain_v1'
        run.mkdir(parents=True)
        data.write(run/'manifest.json', {'run_id': run.name, 'status': 'in_progress'})
        item = {'filename': 'GOLD#_M1_fixture.csv', 'symbol': 'GOLD#', 'timeframe': 'M1',
                'first_timestamp': '2020-01-01T00:00:00', 'last_timestamp': '2020-01-01T00:00:00'}
        raw = ('\t'.join(data.HEADER)+'\r\n2020.01.01\t00:00:00\t10.00\t12.00\t9.00\t11.00\t7\t0\t2\r\n').encode()
        item['sha256'] = hashlib.sha256(raw).hexdigest()
        api_calls = []
        fake_mt5 = SimpleNamespace(initialize=lambda **kw: True,
            account_info=lambda: SimpleNamespace(company='XM Global Limited', server='XMGlobal-MT5 6', trade_mode=0),
            symbol_info=lambda s: object() if s == 'GOLD#' else None, symbol_select=lambda s, enabled: s == 'GOLD#',
            TIMEFRAME_M1=1,
            copy_rates_range=lambda s, tf, a, b: api_calls.append((s, tf, a.isoformat(), b.isoformat())) or
                [dict(time=1577836800, open=10., high=12., low=9., close=11., tick_volume=7, real_volume=0, spread=2)],
            shutdown=lambda: api_calls.append('shutdown'))
        with patch.dict(sys.modules, {'MetaTrader5': fake_mt5}):
            checks['mt5_native_adapter'] = data.fetch_original_export(item) == raw
            checks['mt5_exact_symbol'] = api_calls[0][0:2] == ('GOLD#', 1)
            checks['mt5_shutdown'] = api_calls[-1] == 'shutdown'
            with patch.object(fake_mt5, 'symbol_info', return_value=None):
                reject('mt5_no_substitute', lambda: data.fetch_original_export(item))
        tiny_config = {**config, 'required_datasets': [item]}
        fetches = []
        with contextlib.redirect_stdout(io.StringIO()):
            directory, manifest = data.prepare(tiny_config, root, lambda i: fetches.append(i['filename']) or raw)
        checks['missing_auto_fetch_exact_bytes'] = fetches == [item['filename']] and manifest['training_eligible']
        _, again = data.prepare(tiny_config, root, lambda _: (_ for _ in ()).throw(AssertionError('must not fetch')))
        checks['complete_no_fetch'] = again['datasets'][0]['sha256'] == item['sha256']
        checks['unicode_cache'] = data.validate_file(directory/item['filename'], item, root, scan=True)['row_count'] == 1
        wrong = {**item, 'filename': 'GOLD#_M1_wrong.csv', 'sha256': '0'*64}
        with contextlib.redirect_stdout(io.StringIO()):
            reject('new_mt5_data_not_reproduction', lambda: data.prepare({**config, 'required_datasets': [wrong]}, root, lambda _: raw))
        reject('data_cutoff', lambda: data.validate_file(directory/item['filename'], {**item, 'last_timestamp': '2026-09-26T00:00:00'}, root))
        reject('wrong_dataset_symbol', lambda: data.validate_file(directory/item['filename'], {**item, 'symbol': 'GAUCNH#'}, root))
        reject('new_retrain_mode_denied', lambda: data.prepare({**tiny_config, 'dataset_mode': 'NEW_RETRAIN_DATASET'}, root))
        archive = root/'archive'
        archive.mkdir()
        frame = pd.DataFrame(np.ones((9, 31), np.float32), columns=config['feature_list'])
        frame['TIME_DT'] = pd.date_range('2020-01-01', periods=9, freq='min')
        target = np.array([0, 1, 0]*3, dtype=np.int8)
        folds = [(i+1, name, np.array([i*3, i*3+1]), np.array([i*3+2])) for i, (name, _, _) in enumerate(config['fold_definition'])]
        synthetic = {**frozen, 'b0_models': [], 'c1_target_sha256': {}}
        saved = {}
        for number, name, train, score in folds:
            p = archive/(name+'.txt');p.write_text('SYNTHETIC_B0')
            synthetic['b0_models'].append({'fold': name, 'path': p.name, 'sha256': data.sha(p)})
            synthetic['c1_target_sha256'][f'fold{number}_train'] = library.array_hash(target[train])
            saved[f'fold{number}_indices'] = score
            saved[f'B0_31_technical_fold{number}'] = np.full(len(score), .2, np.float32)
        np.savez(archive/'paired_oof_predictions.npz', **saved)
        library.specification = lambda: synthetic
        library.write_json = data.write
        library.write_csv = lambda p, rows: data.write(p, rows)
        def evaluate(old, helper, frame, b0, sec, ids, candidate, spec):
            checks['only_S4_evaluated'] = candidate == ('S4_SECONDARY_P075', .75)
            return {'synthetic': True}, [{'synthetic': True}], library.candidate_gate(b0, sec, candidate)
        library.evaluate = evaluate
        helper = SimpleNamespace(PRICE_COLUMNS=['TIME_DT'], reconstruct=lambda _: (frame, target, folds, []))
        old = SimpleNamespace(drl_trading_v2=SimpleNamespace(DATA_DIR=None))
        with patch.dict(sys.modules, {'xgboost': fake}), patch.object(trainer, 'model_path', side_effect=lambda r, f: adapter.model_path(r, f, root)):
            result, inventory = trainer.train_folds(run, config, library, helper, old, archive, directory)
        checks['three_secondary_only_mock_fits'] = len(fits) == 3 and all(x['rows'] == 2 for x in fits)
        checks['fit_parameters_and_weights'] = all(x['parameters'] == config['hyperparameters'] and np.array_equal(x['weights'], [1., 1.]) for x in fits)
        value = adapter.build_manifest(run, config, inventory, 's', 'c', 'd', root)
        checks['adapter_deterministic'] = value == adapter.build_manifest(run, config, inventory, 's', 'c', 'd', root)
        checks['adapter_three_fold_no_promotion'] = len(value['fold_models']) == 3 and value['production_promoted'] is False
        reject('adapter_overwrite', lambda: adapter.model_path(run, '2023_2024', root))
        reject('adapter_traversal', lambda: adapter.model_path(run, '../../gemini.py', root))
        reject('validator_artifact_escape', lambda: validator.safe_path(run, '../../gemini.py'))
        checks['validator_comparison_detects_tamper'] = validator.equal({'x': 1}, {'x': 1}) and not validator.equal({'x': 2}, {'x': 1})
        from training_holdout_guard_v1 import install
        release = install(root, write_root=run)
        try:
            reject('write_guard_production', lambda: (root/'gemini.py').write_text('forbidden'))
            (run/'allowed.txt').write_text('synthetic')
            checks['write_guard_run_allowed'] = True
        finally:
            release()
        certificate = root/'training_runs/20200102T000000Z_gold_manual_s4_secondary_retrain_workflow_v1'
        certificate.mkdir()
        data.write(certificate/'metrics.json', {'formal_run_status': 'PASS'})
        data.write(certificate/'validator.json', {'overall': 'PASS'})
        data.write(certificate/'FINALIZED.json', {'file_sha256': {n: data.sha(certificate/n) for n in ('metrics.json', 'validator.json')}})
        approved = {'approved': True, 'training_execution_owner': 'USER',
                    'bindings': {}, 'certification_run': certificate.relative_to(root).as_posix(),
                    'finalized_sha256': data.sha(certificate/'FINALIZED.json')}
        data.write(root/'gold_manual_s4_approval_v1.json', approved)
        data.write(root/'gold_manual_s4_secondary_retrain_config_v1.json', {**tiny_config, 'training_symbol': 'GOLD#'})
        receipt_bytes = (root/'gold_manual_s4_approval_v1.json').read_bytes()
        with patch.object(launch.subprocess, 'check_output', return_value=receipt_bytes):
            state = launch.binding_status(root)
            checks['certified_ready_from_cache'] = state['workflow'] == 'READY' and state['historical_data'] == 'READY' and state['owner'] == 'USER'
            data.write(certificate/'metrics.json', {'formal_run_status': 'FAIL'})
            reject('approval_tampered_certificate_denied', lambda: launch.verify_s4_approval(root))
        with patch.object(launch.subprocess, 'check_output', return_value=b'not committed'):
            reject('uncommitted_approval_denied', lambda: launch.verify_s4_approval(root))
    reject('no_codex_receipt', lambda: trainer.run_manual('missing'))
    with patch.object(launcher, 'user_double_click', return_value=False):
        reject('automated_receipt_denied', launch.issue_receipt)
    with patch.object(launcher, 'user_double_click', return_value=True):
        token = launch.issue_receipt();launch.consume_receipt(token)
        reject('receipt_replay', lambda: launch.consume_receipt(token))
        token = launch.issue_receipt()
        with patch.object(launch.time, 'monotonic', return_value=float('inf')):
            reject('receipt_expired', lambda: launch.consume_receipt(token))
    # Call the public route with a fake terminal stage only: no real training starts.
    with patch.object(trainer, 'run_manual', return_value='SYNTHETIC_ROUTED'):
        checks['launcher_routes_S4'] = launch.run_manual('fixture') == 'SYNTHETIC_ROUTED'
    bat = (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['bat_one_click_root_python_pause'] = all(x in bat for x in ('%~dp0', '.venv\\Scripts\\python.exe', 'manual_training_launcher_v1.py', 'pause >nul')) and '%1' not in bat
    env = {**os.environ, 'PYTHONUTF8': '1'}
    smoke = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT/'RUN_TRAINING.bat')], input='\r\n',
                           capture_output=True, encoding='utf-8', errors='replace', env=env, timeout=30)
    checks['bat_automated_launch_denied_and_pauses'] = '[失敗]' in smoke.stderr and '請按任意鍵關閉' in smoke.stdout
    old_config = launcher.load(launcher.CONFIG)
    with patch.object(launcher.sys, 'executable', str(ROOT/'not-approved.exe')):
        reject('wrong_interpreter', lambda: launcher.verify_environment(old_config))
    output = trainer.format_result('fixture', 'models/fixture.json', {'formal_run_status': 'PASS', 'realized_win_rate': .5, 'trades_per_day': .1, 'profit_factor': 1., 'mean_r': 0.})
    checks['summary_format'] = all(x in output for x in ('XM GOLD S4', 'Win Rate:', 'Trades/Day:', 'PF:', 'Mean-R:', '未變更'))
    tree = ast.parse((ROOT/'validate_gold_manual_s4_training_run_v1.py').read_bytes())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    checks['validator_independent'] = not any(x and ('retrain' in x or 'classifier' in x) for x in imports)
    checks['validator_does_not_fit'] = not any(isinstance(n, ast.Attribute) and n.attr == 'fit' for n in ast.walk(tree))
    checks['production_hashes'] = all(data.sha(ROOT/n) == v for n, v in config['protected_sha256'].items())
    checks['no_real_xgboost_import_or_fit'] = 'xgboost' not in sys.modules and len(fits) == 3
    return {'overall': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'test_count': len(checks),
            'model_training_executed': False, 'strategy_outcome_inspected': False,
            'fake_fit_calls': len(fits), 'synthetic_mt5_fetch_only': True}


if __name__ == '__main__':
    result = tests()
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)
