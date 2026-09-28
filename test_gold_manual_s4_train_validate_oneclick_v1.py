"""Synthetic orchestration only; never fit, predict, reconstruct or replay."""
import ast
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gold_manual_s4_train_validate_v1 as flow
import gold_manual_s4_secondary_retrain_v1 as trainer
import gold_manual_training_workflow_v1 as workflow
import manual_training_launcher_v1 as launcher
import validate_gold_manual_s4_training_run_v1 as validator
import training_run_history as history
import training_holdout_guard_v1 as guard
from gold_manual_s4_training_data_v1 import read, write, sha

ROOT = Path(__file__).resolve().parent


def tests():
    checks = {}
    def reject(name, action):
        try:
            action()
        except (ValueError, PermissionError, FileExistsError, FileNotFoundError):
            checks[name] = True
        else:
            checks[name] = False
    for train, validation, expected in [('PASS', 'PASS', 'PASS'), ('PASS', 'PARTIAL', 'PARTIAL'),
            ('PASS', 'FAIL', 'FAIL'), ('FAIL', 'NOT_RUN', 'FAIL'), ('NOT_STARTED', 'NOT_RUN', 'FAIL'),
            ('PASS', 'NOT_RUN', 'FAIL'), ('FAIL', 'PASS', 'FAIL')]:
        checks['status:'+train+':'+validation] = flow.final_status(train, validation) == expected
    with tempfile.TemporaryDirectory(prefix='s4_oneclick_') as temp:
        root = Path(temp)/'Unicode_測試'
        root.mkdir()
        run = root/'training_runs'/'fixture_exact_run'
        (run/'models').mkdir(parents=True)
        (run/'models/tiny.json').write_text('{}', encoding='utf-8')
        (run/'training_script.py').write_text('# synthetic', encoding='utf-8')
        (root/'production').write_text('protected', encoding='utf-8')
        config = {'feature_list_sha256': 'features', 'feature_pipeline_sha256': 'pipeline',
                  'label_pipeline_sha256': 'labels', 'training_symbol': 'GOLD#',
                  'training_start': '2016-07-01', 'training_end': '2023-01-01',
                  'raw_data_cutoff': '2026-05-08', 'protected_sha256': {'production': sha(root/'production')}}
        write(run/'approved_training_config.json', config)
        write(run/'training_dataset_manifest.json', {'synthetic': True})
        candidate = {'run_id': run.name, 'model_path': 'models/tiny.json', 'model_sha256': sha(run/'models/tiny.json'),
            'training_script_sha256': sha(run/'training_script.py'), 'config_sha256': sha(run/'approved_training_config.json'),
            'training_dataset_manifest_sha256': sha(run/'training_dataset_manifest.json'),
            **{k: config[k] for k in ('feature_list_sha256', 'feature_pipeline_sha256', 'label_pipeline_sha256', 'training_symbol')},
            'training_range': [config['training_start'], config['training_end']]}
        write(run/'candidate_model_manifest.json', candidate)
        flow.training_handoff(run, candidate)
        def check(name, value):
            if not value:
                raise ValueError(name)
        with patch.object(validator, 'ROOT', root):
            validator.validate_handoff(run, check)
            checks['independent_handoff_valid'] = True
            for field in ('model_sha256', 'config_sha256', 'training_dataset_manifest_sha256',
                          'feature_list_sha256', 'feature_pipeline_sha256', 'label_pipeline_sha256', 'training_symbol'):
                write(run/'candidate_model_manifest.json', {**candidate, field: 'tampered'})
                reject('validator_rejects:'+field, lambda: validator.validate_handoff(run, check))
            write(run/'candidate_model_manifest.json', candidate)
            (root/'production').write_text('tampered', encoding='utf-8')
            reject('validator_production_mismatch', lambda: validator.validate_handoff(run, check))
            (root/'production').write_text('protected', encoding='utf-8')
            reject('candidate_escape', lambda: flow.training_handoff(run, {**candidate, 'model_path': '../../production'}))
            reject('validator_escape', lambda: validator.safe_path(run, '../../production', True))
            reject('holdout_path', lambda: guard.check_path(root/'future_holdout/gold_s4_v4/secret', root))
        # A real child process executes only this tiny synthetic validator.
        script = """import json,sys,pathlib
r=pathlib.Path(sys.argv[1]); h=json.loads((r/'training_result.json').read_text())
(r/'validator_attempt.json').write_text('{}')
v={'overall':'PASS','validator_status':'PASS','run_id':r.name,'candidate_model_sha256':h['candidate_model_sha256'],'failed_checks':[]}
(r/'validator.json').write_text(json.dumps(v))
print('SYNTHETIC_VALIDATOR_ONLY')
"""
        (root/'validate_gold_manual_s4_training_run_v1.py').write_text(script, encoding='utf-8')
        (root/'training_runs/zz_unrelated_newer').mkdir()
        with patch.object(flow, 'ROOT', root):
            value = flow.validate_exact_run(run)
            checks['subprocess_exact_same_run_unicode'] = value['run_id'] == run.name and not (root/'training_runs/zz_unrelated_newer/validator.json').exists()
            reject('one_shot_no_retry', lambda: flow.validate_exact_run(run))
        checks['separate_validator_logs'] = 'SYNTHETIC_VALIDATOR_ONLY' in (run/'validator_stdout.txt').read_text() and (run/'validator_stderr.txt').exists()
        value = flow.combined_result(run, 'PASS', 'PASS', candidate, {key: 1 for key in flow.METRICS})
        checks['combined_fields'] = all(key in value for key in flow.METRICS) and read(run/'combined_result.json') == value
        checks['production_holdout_flags'] = value['production_changed'] is value['production_promoted'] is value['holdout_used'] is False
        checks['display_pass'] = all(x in flow.format_result(value) for x in ('Train + Validate 完成', 'Model SHA256', 'Historical Validation Metrics', 'Stress PF'))
        failed = flow.combined_result(None, 'PASS', 'FAIL', errors=['exact_failed_check'])
        checks['display_failure_no_success'] = '完成' not in flow.format_result(failed) and 'exact_failed_check' in flow.format_result(failed)
        partial = flow.combined_result(None, 'PASS', 'PARTIAL', errors=['exact_blocker'])
        checks['display_partial'] = 'exact_blocker' in flow.format_result(partial) and '未核准 promotion' in flow.format_result(partial)
    # Exercise the real controller with only the expensive training dependency replaced.
    for scenario in ('PASS', 'PARTIAL', 'FAIL', 'CRASH', 'TRAIN_FAIL', 'DATA_FAIL', 'ARCHIVE_FAIL'):
        with tempfile.TemporaryDirectory(prefix='s4_controller_') as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            run = root/'training_runs'/'fixture_controller'
            config = read(trainer.CONFIG)
            config['protected_sha256'] = {'protected': 'fixture'}
            (root/'protected').write_text('protected', encoding='utf-8')
            config['protected_sha256']['protected'] = sha(root/'protected')
            write(root/'config.json', config)
            for name in ('gold_manual_s4_training_data_policy_v1.json', 'validate_gold_manual_s4_training_run_v1.py'):
                (root/name).write_text('{}', encoding='utf-8')
            archive = root/'baseline'
            archive.mkdir()
            write(archive/'manifest.json', {'data': {}})
            (root/'training_runs/parent').mkdir(parents=True)
            write(root/'training_runs/parent/manifest.json', {'model': {}})
            events = []
            def create(*args, **kwargs):
                events.append('create')
                run.mkdir()
                write(run/'manifest.json', {'registry': {}})
                (run/'stdout.log').write_text('', encoding='utf-8')
                return run
            row = {'trades': 2, 'wins': 1, 'losses': 1, 'realized_wr': .5, 'trades_per_day': .1,
                   'pf': 1., 'mean_r': 0., 'pnl_r': 0., 'max_dd_r': 1., 'cost_stress_pf': .9}
            def fake_train(*args):
                events.append('train')
                print('synthetic train stdout')
                print('synthetic train stderr', file=sys.stderr)
                if scenario == 'TRAIN_FAIL':
                    raise ValueError('synthetic_training_failure')
                (run/'models').mkdir()
                (run/'models/tiny.json').write_text('{}', encoding='utf-8')
                return {'metrics': [row], 'assessment': 'SYNTHETIC'}, [{'fold': 'fixture', 'train_rows': 2}]
            def fake_publish(*args):
                events.append('save')
                return {'model_path': 'models/tiny.json', 'model_sha256': sha(run/'models/tiny.json')}
            def fake_validate(actual):
                events.append('validate')
                assert actual == run and read(run/'training_result.json')['run_id'] == run.name
                assert not (run/'FINALIZED.json').exists()
                if scenario == 'CRASH':
                    raise RuntimeError('synthetic_validator_crash')
                state = 'PASS' if scenario == 'ARCHIVE_FAIL' else scenario
                return {'validator_status': state, 'failed_checks': [] if state == 'PASS' else ['synthetic_'+state]}
            def finalize(actual, status, **kwargs):
                events.append('finalize')
                assert actual == run and (run/'combined_result.json').exists()
                assert ('validate' in events) == (scenario != 'TRAIN_FAIL')
                assert read(run/'combined_result.json')['final_status'] == ('FAIL' if scenario in ('FAIL', 'CRASH', 'TRAIN_FAIL') else ('PASS' if scenario == 'ARCHIVE_FAIL' else scenario))
                if scenario == 'ARCHIVE_FAIL':
                    return ['synthetic_archive_failure']
                write(run/'FINALIZED.json', {'synthetic': True})
                return []
            for obj, name, value in [(trainer, 'ROOT', root), (trainer, 'CONFIG', root/'config.json'),
                    (trainer, 'verify_config', lambda c: {'c1_run': 'parent'}),
                    (trainer, 'git', lambda *a: '' if a[0] == 'status' else 'main' if a[0] == 'branch' else 'fixture'),
                    (workflow, 'consume_receipt', lambda t: None), (workflow, 'verify_s4_approval', lambda: None),
                    (launcher, 'verify_environment', lambda c: None), (launcher, 'load', lambda p: {}),
                    (history, 'create_run', create), (history, 'finalize_run', finalize),
                    (history, 'register_run', lambda r: events.append('register')),
                    (trainer, 'archive_git', lambda r: events.append('archive')),
                    (trainer, 'train_folds', fake_train), (trainer, 'publish', fake_publish),
                    (trainer, 'validate_exact_run', fake_validate), (guard, 'install', lambda *a, **k: lambda: None)]:
                stack.enter_context(patch.object(obj, name, value))
            dataset = {'datasets': [], 'mt5_fetches': []}
            if scenario == 'DATA_FAIL':
                stack.enter_context(patch.object(trainer, 'prepare', side_effect=ValueError('missing_exact_dataset.csv')))
            else:
                stack.enter_context(patch.object(trainer, 'prepare', return_value=(root, dataset)))
            stack.enter_context(patch.dict(sys.modules, {'gold_independent_secondary_classifier_v1': SimpleNamespace(frozen_inputs=lambda spec: (archive, None, None))}))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    result = trainer.run_manual('SYNTHETIC_ONLY')
                except flow.WorkflowError as error:
                    result = error.result
                except ValueError as error:
                    assert scenario == 'DATA_FAIL' and 'missing_exact_dataset.csv' in str(error)
                    result = flow.combined_result(None, 'NOT_STARTED', 'NOT_RUN', errors=[str(error)])
            checks['controller:'+scenario] = result['final_status'] == (scenario if scenario in ('PASS', 'PARTIAL') else 'FAIL')
            if scenario == 'DATA_FAIL':
                checks['data_fail_never_trains_or_validates'] = events == [] and result['train_status'] == 'NOT_STARTED'
            else:
                checks['same_run_and_order:'+scenario] = events[:3] == (['create', 'train', 'finalize'] if scenario == 'TRAIN_FAIL' else ['create', 'train', 'save']) and events.index('finalize') > events.index('train')
                checks['training_logs:'+scenario] = 'synthetic train stdout' in (run/'training_stdout.txt').read_text() and 'synthetic train stderr' in (run/'training_stderr.txt').read_text()
                if scenario == 'TRAIN_FAIL':
                    checks['training_fail_skips_validator'] = 'validate' not in events and result['validator_status'] == 'NOT_RUN'
    bat = (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['bat_single_no_args_pause'] = 'manual_training_launcher_v1.py"' in bat and '%1' not in bat and 'pause >nul' in bat
    checks['no_second_bat'] = 'RUN_VALIDATION' not in bat
    with tempfile.TemporaryDirectory(prefix='s4_bat_') as temp:
        folder = Path(temp)
        fixture_bat = bat.replace('%~dp0.venv\\Scripts\\python.exe', sys.executable)
        (folder/'RUN_TRAINING.bat').write_text(fixture_bat, encoding='utf-8')
        for state in ('PASS', 'PARTIAL', 'FAIL'):
            (folder/'manual_training_launcher_v1.py').write_text('print("SYNTHETIC_'+state+'")\n', encoding='utf-8')
            smoke = subprocess.run(['cmd.exe', '/d', '/c', str(folder/'RUN_TRAINING.bat')], input='\n', capture_output=True, encoding='utf-8', errors='replace', timeout=30)
            checks['bat_pauses:'+state] = 'SYNTHETIC_'+state in smoke.stdout and '請按任意鍵關閉' in smoke.stdout
    # Launch from a non-Explorer process: production training must be denied, then pause.
    smoke = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT/'RUN_TRAINING.bat')], cwd=ROOT, input='\n', capture_output=True, encoding='utf-8', errors='replace', timeout=30)
    checks['bat_failure_visible_and_pauses'] = '請按任意鍵關閉' in smoke.stdout and 'NOT_STARTED' in smoke.stdout and '[失敗]' in smoke.stderr
    tree = ast.parse((ROOT/'validate_gold_manual_s4_training_run_v1.py').read_bytes())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    checks['validator_independent'] = not any('retrain' in n or 'train_validate' in n for n in imports)
    checks['no_fit_or_predict'] = not any(isinstance(n, ast.Attribute) and n.attr in ('fit', 'predict', 'predict_proba') for n in ast.walk(tree))
    checks['no_real_model_loaded'] = 'xgboost' not in sys.modules
    return {'overall': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'test_count': len(checks), 'model_training_executed': False, 'holdout_used': False,
            'failed_checks': [k for k, v in checks.items() if not v]}


if __name__ == '__main__':
    result = tests()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)
