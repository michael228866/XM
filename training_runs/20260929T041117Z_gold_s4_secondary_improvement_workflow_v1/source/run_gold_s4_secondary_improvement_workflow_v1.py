"""Infrastructure-only certification and metadata approval. Never runs research."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
from gold_manual_s4_training_data_v1 import read, write, sha
from run_gold_manual_training_workflow_binding_v1 import protections, automation_audit

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_s4_secondary_improvement_workflow_v1'
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
    config = read(ROOT/'gold_s4_secondary_improvement_config_v1.json')
    run = history.create_run(SLUG, Path(__file__), 'python -B '+Path(__file__).name+' --execute',
                             arguments=['--execute'], seed_note='Deterministic synthetic infrastructure fixtures; no actual model training')
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
    write(run/'research_plan.json', config)
    write(run/'predeclared_search_space.json', config['candidates'])
    write(run/'reference_binding.json', {'status': 'PASS', 'reference_run': config['reference_run'],
          'reference_metrics': config['reference_metrics'], 'model_sha256': config['reference_model_sha256'],
          'frozen_hashes': config['frozen_hashes']})
    results = {}
    for name in spec['test_suites']:
        with (run/(name+'.stdout.txt')).open('x', encoding='utf-8') as out, (run/(name+'.stderr.txt')).open('x', encoding='utf-8') as err:
            proc = subprocess.run([sys.executable, '-B', str(ROOT/name)], cwd=ROOT, stdout=out, stderr=err)
        try:
            result = json.loads((run/(name+'.stdout.txt')).read_text(encoding='utf-8').splitlines()[-1])
        except (ValueError, IndexError):
            result = {'overall': 'FAIL', 'checks': {}, 'error': 'See suite stderr'}
        results[name] = {'exit_code': proc.returncode, **result}
    write(run/'self_tests.json', results)
    tests = results['test_gold_s4_secondary_improvement_v1.py']['checks']
    for name, keys in spec['test_groups'].items():
        write(run/(name+'.json'), {'status': 'PASS' if all(tests.get(k, False) for k in keys) else 'FAIL',
                                  'checks': {k: tests.get(k, False) for k in keys}, 'synthetic_only': True})
    audit = automation_audit()
    write(run/'auto_training_audit.json', audit)
    after = protections()
    write(run/'protection_checks.json', {'before': before, 'after': after, 'unchanged': before == after})
    passed = all(r['overall'] == 'PASS' and r['exit_code'] == 0 for r in results.values()) and before == after and audit['status'] == 'PASS'
    metrics = {'formal_run_status': 'PASS' if passed else 'FAIL', 'validator_status': 'PENDING',
               'total_predeclared_candidates': len(config['candidates']), 'model_training_executed': False,
               'strategy_replay_executed': False, 'holdout_used': False, 'production_changed': False,
               'production_promoted': False, 'auto_training_disabled': audit['auto_training_disabled'],
               'approval_commit_pending': True, 'improvement_workflow_ready': False}
    write(run/'metrics.json', metrics)
    report = ('# GOLD S4 improvement workflow certification\n\nInfrastructure result: '+metrics['formal_run_status']+'. '
        'No real training/search/prediction/replay. Synthetic fake estimators test all enabled families. '
        'Reference trainer/config and prior finalized runs are unchanged.\n\n'
        '16 fixed configurations; seven separate families including control. Six causal pointwise extensions. '
        'Stage 1 uses the first two original folds; only strict dual-KPI and safety qualifiers reach the third fold. '
        'Stage 2 evaluates the full original three-fold period. All periods are historical development data. '
        'No untouched promotion evidence or automatic promotion is claimed.\n\n'
        'Independent actual-run validator recomputes model probabilities and S5 metrics from retained chunks, '
        'checks conditioning/calibration maturity, frozen features/labels, weights, search-event integrity, '
        'Pareto and tie-breaks. No trainer/search imports and no fitting in that validator.\n\n'
        'Execution PASS and NO_IMPROVEMENT_FOUND are separate dimensions. Source is now committed, '
        'but launcher deployment awaits independent certification PASS and a separate approval commit.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    (run/'stdout.log').write_text('Infrastructure only; no real training or strategy replay\n', encoding='utf-8')
    data = manifest['data']
    data.update(symbols=['GOLD#'], data_sources=['Frozen source/provenance metadata and synthetic fixtures'],
                source_files=manifest['input_snapshots'], timezone='UTC', raw_snapshot_retained=True,
                reproducibility_claim='Synthetic fixture source retained; no market fetch',
                purge_details='No real training; calibration purge tested synthetically', embargo_details='No real training')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[key] = 'NOT_APPLICABLE_NO_TRAINING'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        data[key] = 0
    data['mt5_fetch'].update(used=False, not_applicable_reason='Synthetic infrastructure certification; no MT5 calls')
    manifest['model']['not_applicable_reason'] = 'Infrastructure only; fake estimators'
    manifest['search']['not_applicable_reason'] = 'Predeclaration only; no real search executed'
    manifest['registry'].update({k: 'N/A: infrastructure only' for k in history.REGISTRY_FIELDS})
    manifest['registry'].update(parent_or_incumbent=config['reference_run'],
        selected_configuration='16 predeclared historical candidates; dual-KPI gate; USER only; reference and production unchanged; approval pending', validator_result='PENDING')
    write(run/'manifest.json', manifest)
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'])


def approve(run):
    result_commit = clean_pushed()
    run = Path(run).resolve()
    if run.parent != ROOT/'training_runs' or not run.name.endswith('_'+SLUG):
        raise ValueError('Wrong infrastructure run')
    if history.validate_run(run) or read(run/'validator.json')['overall'] != 'PASS' or read(run/'metrics.json')['formal_run_status'] != 'PASS':
        raise ValueError('Immutable independent certification PASS required')
    receipt_path = ROOT/'gold_s4_improvement_approval_v1.json'
    if receipt_path.exists():
        raise FileExistsError('Improvement approval already exists; use a new version')
    sources = read(run/'execution_spec.json')['source_files']
    for name in sources:
        if sha(ROOT/name) != sha(run/'source'/name):
            raise ValueError('Certified source changed: '+name)
    launcher_path = ROOT/'training_launcher_config_v1.json'
    launcher = read(launcher_path)
    entry = lambda name: {'path': name, 'sha256': sha(ROOT/name)}
    launcher['workflow'].update(experiment_name='gold_s4_secondary_improvement_v1',
        script=entry('gold_s4_secondary_improvement_v1.py'), validator=entry('validate_gold_s4_secondary_improvement_run_v1.py'),
        configuration=entry('gold_s4_secondary_improvement_config_v1.json'))
    launcher['approval_status'] = 'APPROVED'
    write(launcher_path, launcher)
    bindings = {name: sha(ROOT/name) for name in sources}
    bindings.update(read(ROOT/'gold_manual_s4_secondary_retrain_config_v1.json')['protected_sha256'])
    write(receipt_path, {'approved': True, 'training_execution_owner': 'USER',
        'workflow': 'GOLD_S4_SECONDARY_IMPROVEMENT_V1', 'certification_run': run.relative_to(ROOT).as_posix(),
        'finalized_sha256': sha(run/'FINALIZED.json'), 'source_commit': read(run/'manifest.json')['git_commit'],
        'result_commit': result_commit, 'bindings': bindings, 'production_promoted': False})
    print('APPROVAL_PREPARED; commit and push required; no real training executed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--execute']:
        execute()
    elif len(sys.argv) == 3 and sys.argv[1] == '--approve':
        approve(sys.argv[2])
    else:
        raise SystemExit('Use --execute for infrastructure only or --approve <sealed infrastructure run>')
