"""Synthetic infrastructure certification and subsequent metadata approval only."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
from gold_manual_s4_training_data_v1 import read, write, sha
from run_gold_manual_training_workflow_binding_v1 import protections, automation_audit

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_manual_s4_train_validate_oneclick_v1'
SPEC = 'execution_spec_'+SLUG+'.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def clean_pushed():
    head = git('rev-parse', 'HEAD')
    if (git('status', '--porcelain') or git('branch', '--show-current') != 'main'
            or git('rev-parse', '@{u}') != head or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != head):
        raise ValueError('Clean pushed main required')
    return head


def execute():
    commit = clean_pushed()
    before = protections()
    spec = read(ROOT/SPEC)
    run = history.create_run(SLUG, Path(__file__), 'python -B '+Path(__file__).name+' --execute',
                             arguments=['--execute'], seed_note='Deterministic synthetic fixtures; no real model training')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    manifest = read(run/'manifest.json')
    manifest.update(source_commit=commit, pre_run_remote_commit=commit, pre_run_clean=True, input_snapshots=[])
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        manifest['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
                                          'sha256': sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/SPEC, run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'), run/'validator_script.py')
    results = {}
    for name in ('test_gold_manual_s4_secondary_retrain_v1.py', 'test_'+SLUG+'.py'):
        with (run/(name+'.stdout.txt')).open('x', encoding='utf-8') as out, (run/(name+'.stderr.txt')).open('x', encoding='utf-8') as err:
            proc = subprocess.run([sys.executable, '-B', str(ROOT/name)], cwd=ROOT, stdout=out, stderr=err)
        result = json.loads((run/(name+'.stdout.txt')).read_text(encoding='utf-8').splitlines()[-1])
        results[name] = {'exit_code': proc.returncode, **result}
    write(run/'self_tests.json', results)
    tests = results['test_'+SLUG+'.py']['checks']
    groups = spec['test_groups']
    for name, keys in groups.items():
        write(run/(name+'.json'), {'status': 'PASS' if all(tests.get(k, False) for k in keys) else 'FAIL',
                                  'checks': {k: tests.get(k, False) for k in keys}, 'synthetic_only': True})
    audit = automation_audit()
    write(run/'auto_training_audit.json', audit)
    after = protections()
    write(run/'protection_checks.json', {'before': before, 'after': after, 'unchanged': before == after})
    passed = all(r['overall'] == 'PASS' and r['exit_code'] == 0 for r in results.values()) and before == after and audit['status'] == 'PASS'
    metrics = {'formal_run_status': 'PASS' if passed else 'FAIL', 'validator_status': 'PENDING',
               'model_training_executed': False, 'holdout_used': False, 'production_changed': False,
               'production_promoted': False, 'auto_training_disabled': audit['auto_training_disabled'],
               'approval_commit_pending': True}
    write(run/'metrics.json', metrics)
    report = ('# S4 one-click Train + Validate infrastructure certification\n\n'
              'Result: '+metrics['formal_run_status']+'. Synthetic orchestration and independent handoff checks only. '
              'No real training, prediction, reconstruction, strategy replay or locked holdout evaluation.\n\n'
              'The existing trainer already called the independent validator. This change adds explicit training_result, '
              'combined_result, separate logs and deterministic PASS/PARTIAL/FAIL display. The exact created run is passed '
              'directly to validation; latest-run discovery is used only in the status display. '
              'Mandatory validation precedes successful finalization. One-shot guard remains.\n\n'
              'Frozen training config, data policy, feature/label definitions, thresholds and production remain unchanged. '
              'Prior finalized runs including 20260928T051907Z are untouched. Older status is read from sealed evidence '
              'without retroactively creating a combined result. Only a USER Explorer/BAT launch can start real training.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    (run/'stdout.log').write_text('Synthetic infrastructure tests only\n', encoding='utf-8')
    data = manifest['data']
    data.update(symbols=['GOLD#'], data_sources=['Synthetic fixtures and frozen source metadata'],
                source_files=manifest['input_snapshots'], timezone='UTC', raw_snapshot_retained=True,
                reproducibility_claim='Synthetic fixtures retained as source; no market fetch',
                purge_details='No real training', embargo_details='No real training')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[key] = 'NOT_APPLICABLE_NO_TRAINING'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        data[key] = 0
    data['mt5_fetch'].update(used=False, not_applicable_reason='Synthetic certification; no MT5 calls')
    manifest['model']['not_applicable_reason'] = 'No real model training; fixture model bytes only'
    manifest['search']['not_applicable_reason'] = 'No strategy research or tuning'
    manifest['registry'].update({k: 'N/A: infrastructure only' for k in history.REGISTRY_FIELDS})
    manifest['registry'].update(parent_or_incumbent='GOLD_MANUAL_S4_SECONDARY_RETRAIN_V1',
        selected_configuration='Same-run train + independent validate; synthetic only; no production change; approval pending', validator_result='PENDING')
    write(run/'manifest.json', manifest)
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'])


def approve(run):
    result_commit = clean_pushed()
    run = Path(run).resolve()
    if run.parent != ROOT/'training_runs' or not run.name.endswith('_'+SLUG):
        raise ValueError('Wrong certification run')
    if history.validate_run(run):
        raise ValueError('Invalid immutable certification')
    if read(run/'validator.json')['overall'] != 'PASS' or read(run/'metrics.json')['formal_run_status'] != 'PASS':
        raise ValueError('Independent certification PASS required')
    source = read(run/'execution_spec.json')['source_files']
    for name in source:
        if sha(ROOT/name) != sha(run/'source'/name):
            raise ValueError('Certification source changed: '+name)
    approval = read(ROOT/'gold_manual_s4_approval_v1.json')
    old_seal = approval['finalized_sha256']
    spec_path = ROOT/'gold_manual_training_spec_v1.json'
    spec = read(spec_path)
    spec.update(training_script_sha256=sha(ROOT/'gold_manual_s4_secondary_retrain_v1.py'),
                validation_mandatory=True, result_contract='training_result.json + validator.json + combined_result.json')
    write(spec_path, spec)
    config_path = ROOT/'gold_manual_training_config_v1.json'
    config = read(config_path)
    config['spec']['sha256'] = sha(spec_path)
    config.update(validation_mandatory=True, same_run_id_required=True)
    write(config_path, config)
    launcher_path = ROOT/'training_launcher_config_v1.json'
    launcher = read(launcher_path)
    for key in ('script', 'validator', 'configuration'):
        item = launcher['workflow'][key]
        item['sha256'] = sha(ROOT/item['path'])
    write(launcher_path, launcher)
    approval.update(certification_run=run.relative_to(ROOT).as_posix(), finalized_sha256=sha(run/'FINALIZED.json'),
                    previous_certification_sha256=old_seal, source_commit=read(run/'manifest.json')['git_commit'],
                    result_commit=result_commit, validation_mandatory=True)
    for name in set(approval['bindings']) | set(source):
        if name != 'gold_manual_s4_approval_v1.json':
            approval['bindings'][name] = sha(ROOT/name)
    write(ROOT/'gold_manual_s4_approval_v1.json', approval)
    print('APPROVAL_PREPARED; no training executed; commit and push required')


if __name__ == '__main__':
    if sys.argv[1:] == ['--execute']:
        execute()
    elif len(sys.argv) == 3 and sys.argv[1] == '--approve':
        approve(sys.argv[2])
    else:
        raise SystemExit('Use --execute for synthetic certification or --approve <sealed run>')
