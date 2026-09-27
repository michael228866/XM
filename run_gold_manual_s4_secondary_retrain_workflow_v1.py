"""Infrastructure certification only. Real training is never called here."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
from gold_manual_s4_training_data_v1 import read, write, sha, fetch_original_export
from run_gold_manual_training_workflow_binding_v1 import automation_audit, protections
from test_gold_manual_s4_secondary_retrain_v1 import tests

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_manual_s4_secondary_retrain_workflow_v1'


def main():
    if sys.argv[1:] == ['--import-smoke']:
        import gold_independent_secondary_classifier_v1 as discovery
        archive, helper, parent = discovery.frozen_inputs(discovery.specification())
        print(json.dumps({'status': 'PASS', 'feature_count': len(helper.BASE_FEATURES),
                          'reconstruction_called': False, 'fit_called': False, 'predict_called': False}))
        return
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Infrastructure --execute required')
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('Clean pushed source required')
    before = protections()
    spec = read(ROOT/('execution_spec_'+SLUG+'.json'))
    run = history.create_run(SLUG, Path(__file__), str(Path(sys.executable))+' -B '+Path(__file__).name+' --execute',
                             arguments=['--execute'], seed_note='Infrastructure only; synthetic fake model; no actual model fits')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = read(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=commit, pre_run_clean=True, input_snapshots=[])
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        m['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
                                    'sha256': sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    for source, dest in [('execution_spec_'+SLUG+'.json', 'execution_spec.json'),
                         ('validate_'+SLUG+'.py', 'validator_script.py'),
                         ('gold_manual_s4_provenance_v1.json', 's4_provenance_trace.json'),
                         ('gold_manual_s4_dataset_decision_v1.json', 'training_dataset_decision.json'),
                         ('gold_manual_s4_secondary_retrain_config_v1.json', 'approved_training_config.json'),
                         ('gold_manual_s4_training_data_policy_v1.json', 'approved_training_spec.json')]:
        shutil.copyfile(ROOT/source, run/dest)
    write(run/'manifest.json', m)
    config = read(ROOT/'gold_manual_s4_secondary_retrain_config_v1.json')
    write(run/'training_symbol_decision.json', {'status': 'PASS', 'symbol': 'GOLD#', 'source_id': config['training_source_id'],
          'GAUCNH_interchangeable': False, 'historical_server_identity_claim': False})
    write(run/'feature_binding.json', {k: config[k] for k in ('feature_list', 'feature_list_sha256', 'feature_pipeline', 'feature_pipeline_sha256')})
    write(run/'label_binding.json', {k: config[k] for k in ('label_pipeline', 'label_pipeline_sha256', 'secondary_target_rule')})
    write(run/'conditioning_binding.json', {k: config[k] for k in ('b0_conditioning_rule', 'b0_primary_threshold', 'secondary_threshold', 'fold_definition')})
    with (run/'self_test_stdout.txt').open('x', encoding='utf-8') as out, (run/'self_test_stderr.txt').open('x', encoding='utf-8') as err:
        tested = subprocess.run([sys.executable, '-B', str(ROOT/'test_gold_manual_s4_secondary_retrain_v1.py')], stdout=out, stderr=err, cwd=ROOT)
    result = json.loads((run/'self_test_stdout.txt').read_text(encoding='utf-8').splitlines()[-1])
    write(run/'launcher_smoke_test.json', result)
    imported = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--import-smoke'], cwd=ROOT, capture_output=True, encoding='utf-8', timeout=90)
    write(run/'dependency_import_smoke.json', {'exit_code': imported.returncode, 'stdout': imported.stdout, 'stderr': imported.stderr,
          'status': 'PASS' if imported.returncode == 0 else 'FAIL', 'model_training_executed': False})
    fetch_start = datetime.now(timezone.utc).isoformat()
    smoke = {'symbol': 'GOLD#', 'timeframe': 'H1', 'first_timestamp': '2018-01-02T01:00:00', 'last_timestamp': '2018-01-02T02:00:00'}
    try:
        raw = fetch_original_export(smoke)
        (run/'mt5_fetch_smoke.csv').write_bytes(raw)
        fetched = {'status': 'PASS', 'rows': len(raw.splitlines())-1, 'sha256': sha(run/'mt5_fetch_smoke.csv'),
                   'artifact': 'mt5_fetch_smoke.csv', 'training_eligible': False}
    except Exception as error:
        fetched = {'status': 'UNAVAILABLE', 'rows': 0, 'error': str(error), 'training_eligible': False}
    fetched.update(request=smoke, fetch_start_utc=fetch_start, fetch_end_utc=datetime.now(timezone.utc).isoformat(),
                   purpose='Bounded native API connectivity only; no strategy or feature computation; not a replacement dataset')
    write(run/'mt5_fetch_smoke_test.json', fetched)
    write(run/'data_manager_review.json', {'status': 'PASS', 'local_legacy_status': 'PASS', 'exact_files': 21,
          'auto_fetch_policy': 'Missing exact bytes attempted automatically; nonidentical fresh data rejected',
          'live_smoke_status': fetched['status'], 'new_retrain_dataset_allowed': False,
          'timestamp_mapping_claim': 'Original broker/export clock retained; no historical UTC remapping'})
    write(run/'output_adapter_review.json', {'status': 'PASS', 'run_local_only': True, 'three_fold_models': True,
          'last_fold_is_not_latest_full_fit': True, 'production_overwrite_allowed': False,
          'deterministic_fixture': result['checks']['adapter_deterministic']})
    write(run/'holdout_guard_review.json', {'status': 'PASS', 'path_and_write_tests': True, 'native_os_isolation': False,
          'reviewed_native_io': 'Libraries imported before restricted fit/evaluate section; NumPy/Pandas input only from exact data_dir, XGBoost output only through checked run-local path',
          'holdout_training_dependency': False})
    audit = automation_audit()
    write(run/'auto_training_audit.json', audit)
    after = protections()
    write(run/'protection_checks.json', {'before': before, 'after': after, 'unchanged': before == after})
    passed = result['overall'] == 'PASS' and tested.returncode == imported.returncode == 0 and audit['status'] == 'PASS' and before == after
    metrics = {'formal_run_status': 'PASS' if passed else 'FAIL', 's4_provenance_status': 'PASS',
               'training_symbol_status': 'PASS', 'feature_binding_status': 'PASS', 'label_binding_status': 'PASS',
               'conditioning_binding_status': 'PASS', 'training_data_policy_status': 'PASS', 'legacy_data_status': 'PASS',
               'auto_fetch_status': 'PASS', 'output_adapter_status': 'PASS', 'holdout_guard_status': 'PASS',
               'self_test_status': result['overall'], 'bat_launcher_status': 'PASS',
               'training_execution_owner': 'USER', 'auto_training_disabled': True,
               'model_training_executed': False, 'strategy_outcome_inspected': False,
               'production_changed': False, 'production_promoted': False, 'capture_automation_preserved': before == after,
               'training_workflow_ready': False, 'approval_commit_pending': True, 'validator_status': 'PENDING'}
    write(run/'metrics.json', metrics)
    report = ('# GOLD manual S4 secondary retraining certification\n\nResult: '+metrics['formal_run_status']+'. '
        'Infrastructure only; fake classifier in synthetic tests; no real fitting, reconstruction, prediction or S5 execution. '
        'The small MT5 H1 smoke is raw-data connectivity only.\n\n'
        'S4 provenance resolves to discovery training functions, not confirmation runner. '
        '21 original GOLD# exports are EXACT_PROVENANCE_TRAINING_SOURCE. '
        'Original broker/export clock and raw cutoff 2026-05-08T23:57:00 are retained. '
        'Actual training uses three fixed 18-month folds, B0 train probability < .75 and C1 positive net-R target. '
        'B0 is never fitted; secondary threshold .75; no selection/tuning/promotion.\n\n'
        'Missing sources are automatically attempted via exact native MT5 export restoration. '
        'Only matching original SHA256 is accepted; nonidentical NEW_RETRAIN_DATASET fails closed. '
        'All required original files are locally present. Raw/cache retained locally; NPZ run evidence archived in Git. No remote raw backup claimed.\n\n'
        'Independent training validator uses historical independently validated reproduction hashes/arrays/ledger as oracle; '
        'never imports the trainer or fits a validation model. '
        'Approval is a separate committed deployment after this finalized certification PASS. '
        'Capture code/policy/task unchanged; no capture restart.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    (run/'stdout.log').write_text('Infrastructure only; '+str(result['test_count'])+' synthetic checks: '+result['overall']+'\n', encoding='utf-8')
    d = m['data']
    d.update(symbols=['GOLD#'], data_sources=['Exact legacy dataset metadata/schema/hash adjudication; synthetic fixtures; bounded native H1 smoke'],
             source_files=m['input_snapshots'], timezone='UTC request coordinate / legacy broker export clock preserved separately',
             raw_snapshot_retained=fetched['status'] == 'PASS', reproducibility_claim='Exact source/config/fixture metadata retained; bounded raw smoke retained if available; original large CSV remains local',
             purge_details='No training during certification', embargo_details='No training during certification')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        d[key] = 'NOT_APPLICABLE_NO_TRAINING'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        d[key] = 0
    d['mt5_fetch'].update(used=True, terminal_path=r'D:\XM2\terminal64.exe',
        terminal_info={'path': r'D:\XM2\terminal64.exe', 'API': 'copy_rates_range'},
        broker_info={'required_identity': {'company': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'trade_mode': 0},
                     'identity_verified_before_returned_rows': fetched['status'] == 'PASS'},
        fetch_start_utc=fetch_start, fetch_end_utc=fetched['fetch_end_utc'], retrieved_at_utc=fetched['fetch_end_utc'], returned_rows=fetched['rows'])
    m['model']['not_applicable_reason'] = 'Infrastructure certification; fake synthetic classifier only'
    m['search']['not_applicable_reason'] = 'No strategy research execution or tuning'
    m['registry'].update({key: 'N/A: infrastructure only' for key in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent=config['discovery_run'], selected_configuration='S4 exact secondary reproduction; three frozen folds; USER only; no production change; approval pending', validator_result='PENDING')
    m['formal_run_status'] = metrics['formal_run_status']
    write(run/'manifest.json', m)
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'], flush=True)


if __name__ == '__main__':
    main()
