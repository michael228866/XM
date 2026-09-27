"""Infrastructure-only certification; never trains, fetches real bars or evaluates."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
from test_gold_manual_training_workflow_binding_v1 import tests

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_manual_training_workflow_binding_v1'
load, write, sha = history.read_json, history.write_json, history.file_sha256
NAMES = ('RUN_TRAINING.bat', 'manual_training_launcher_v1.py', 'gold_manual_training_workflow_v1.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def automation_audit():
    command = r'''[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$tasks=@(Get-ScheduledTask -ErrorAction Stop | ForEach-Object {
 foreach($action in $_.Actions) {
  if(($action.Execute+' '+$action.Arguments) -match 'D:\\XM|RUN_TRAINING|manual_training|gold_manual_training') {
   [pscustomobject]@{name=$_.TaskName;execute=$action.Execute;arguments=$action.Arguments;working_directory=$action.WorkingDirectory}
  }
 }
})
$startup=@(Get-CimInstance Win32_StartupCommand -ErrorAction Stop | Where-Object {$_.Command -match 'D:\\XM|RUN_TRAINING|manual_training|gold_manual_training'} | Select-Object Name,Command,Location)
@{tasks=$tasks;startup=$startup}|ConvertTo-Json -Depth 5 -Compress'''
    live = json.loads(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], timeout=40).decode('utf-8-sig'))
    references = []
    for p in sorted(ROOT.iterdir()):
        if p.suffix.lower() not in {'.py', '.ps1', '.bat', '.cmd'}:
            continue
        lines = [i for i, line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(), 1) if any(n in line for n in NAMES)]
        if lines:
            references.append({'path': p.name, 'lines': lines})
    ci = []
    for folder in (ROOT/'.github', ROOT/'.gitlab'):
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file():
                    text = p.read_text(encoding='utf-8-sig')
                    ci.append({'path': p.relative_to(ROOT).as_posix(), 'training_reference': any(n in text for n in NAMES)})
    policy = load(ROOT/'gold_future_capture_runtime_freshness_policy_v1.json')
    runtime = {}
    for name, expected in policy['runtime_code_sha256'].items():
        raw = (ROOT/name).read_text(encoding='utf-8')
        tree = ast.parse(raw)
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else '' for n in ast.walk(tree) if isinstance(n, ast.Call)]
        runtime[name] = {'sha256': sha(ROOT/name), 'unchanged': sha(ROOT/name) == expected,
                         'trainer_reference': any(n in raw for n in NAMES),
                         'model_call': any(n in calls for n in ('fit', 'predict', 'predict_proba', 'run_training'))}
    automatic = json.dumps(live).lower()
    safe = not any(n.lower() in automatic for n in NAMES)
    safe = safe and not any(x['training_reference'] for x in ci)
    safe = safe and all(x['unchanged'] and not x['trainer_reference'] and not x['model_call'] for x in runtime.values())
    return {'status': 'PASS' if safe else 'FAIL', 'live': live, 'ci': ci,
            'repository_references': references, 'runtime_import_closure': runtime,
            'auto_training_disabled': safe,
            'reference_interpretation': 'Manual BAT/launcher/wrapper and certification/test/static audit strings; runtime closure and installed startup/task actions verified separately'}


def protections():
    expected = load(ROOT/'training_launcher_config_v1.json')['protected_sha256']
    return {'production': {n: sha(ROOT/n) for n in expected},
            'production_match': all(sha(ROOT/n) == h for n, h in expected.items()),
            'runtime_policy_sha256': sha(ROOT/'gold_future_capture_runtime_freshness_policy_v1.json'),
            'prior_seals': {p.parent.name: sha(p) for p in sorted((ROOT/'training_runs').glob('*/FINALIZED.json'))}}


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Infrastructure --execute required; never trains')
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('Clean pushed source required')
    spec = load(ROOT/('execution_spec_' + SLUG + '.json'))
    before = protections()
    run = history.create_run(SLUG, Path(__file__), 'python -B '+Path(__file__).name+' --execute',
                             arguments=['--execute'], seed_note='Synthetic deterministic infrastructure only; no fitting')
    print('RUN_ID=' + run.relative_to(ROOT).as_posix(), flush=True)
    m = load(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=commit, pre_run_clean=True, input_snapshots=[])
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        m['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
                                    'sha256': sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    for source, dest in [('execution_spec_'+SLUG+'.json', 'execution_spec.json'),
                         ('validate_'+SLUG+'.py', 'validator_script.py'),
                         ('gold_manual_training_provenance_v1.json', 'training_provenance_trace.json'),
                         ('gold_manual_training_spec_v1.json', 'approved_training_spec.json'),
                         ('gold_manual_training_config_v1.json', 'approved_training_config.json'),
                         ('gold_historical_training_data_manifest_v1.json', 'historical_data_inventory.json')]:
        shutil.copyfile(ROOT/source, run/dest)
    write(run/'manifest.json', m)
    result = tests()
    write(run/'launcher_smoke_test.json', result)
    write(run/'historical_data_fetch_smoke_test.json', {
        'status': 'PARTIAL', 'synthetic_fetch_status': result['overall'], 'live_fetch_performed': False,
        'reason': 'No approved historical symbol/range/clock contract; bounded live smoke not necessary to prove unresolved binding',
        'real_data_fetched': False, 'auto_fetch_enabled': False})
    write(run/'historical_data_requirements.json', load(ROOT/'gold_manual_training_spec_v1.json')['dataset_policy'])
    write(run/'launcher_binding.json', {'status': 'PARTIAL', 'workflow_ready': False,
                                      'single_approved_default': None, 'manual_owner': 'USER',
                                      'actual_training_executed': False, 'receipt_ttl_seconds': 30})
    write(run/'holdout_guard_review.json', {'status': 'PASS', 'native_os_isolation': False,
                                          'unapproved_worker_enabled': False,
                                          'path_and_cutoff_tests_pass': all(v for k, v in result['checks'].items() if 'holdout' in k or 'cutoff' in k or 'audit_hook' in k),
                                          'holdout_tree_read_for_training': False})
    audit = automation_audit()
    write(run/'auto_training_audit.json', audit)
    status = subprocess.run([sys.executable, '-B', str(ROOT/'gold_runtime_status_v1.py')],
                            cwd=ROOT, capture_output=True, encoding='utf-8', timeout=30)
    write(run/'status_launcher_smoke_test.json', {
        'exit_code': status.returncode, 'stdout': status.stdout, 'stderr': status.stderr,
        'status': 'PASS' if status.returncode == 0 and 'TRAINING WORKFLOW: NOT_READY' in status.stdout else 'FAIL',
        'capture_restarted': False, 'capture_health_claim': 'See observed stdout; configuration preservation is not a live-health claim'})
    after = protections()
    write(run/'protection_checks.json', {'before': before, 'after': after, 'unchanged': before == after})
    metrics = {'formal_run_status': 'PARTIAL', 'training_provenance_status': 'PARTIAL',
               'approved_training_script_status': 'PARTIAL', 'approved_training_config_status': 'PARTIAL',
               'training_data_policy_status': 'PARTIAL', 'auto_fetch_status': 'PARTIAL',
               'holdout_guard_status': 'PASS', 'self_test_status': result['overall'],
               'bat_launcher_status': 'PASS' if result['overall'] == 'PASS' else 'FAIL',
               'training_workflow_ready': False, 'model_training_executed': False,
               'strategy_outcome_inspected': False, 'production_changed': False,
               'production_promoted': False, 'capture_automation_preserved': before == after,
               'auto_training_disabled': audit['auto_training_disabled'], 'validator_status': 'PENDING'}
    write(run/'metrics.json', metrics)
    report = ('# GOLD manual training workflow binding v1\n\nPARTIAL: infrastructure tests only. '
              'No approved single default, no actual training, no strategy evaluation, no live historical fetch. '
              'Production and capture preserved.\n\n'
              'Latest related historical training is S4 confirmation 20260919T124853Z; this is not approval '
              'for a repeatable default or replacing frozen CSV reconstruction with fresh MT5 bars. '
              'Production legacy main overwrites protected model; B0 main also runs B1. '
              'Historical clock mapping, exact coverage, output adapter and independent training validator '
              'remain unresolved. See training_provenance_trace.json for all candidate hashes and native timeframes.\n\n'
              'Launcher is NOT_READY with readable Chinese failure and pause. Process-local one-use 30-second receipt '
              'prevents accidental automated entry; no static secret. Data primitives tested on synthetic bars; '
              'CSV import supports reviewed canonical schema only. No actual model import/fit/prediction/replay. '
              'CPython audit hook is not hostile native-code OS isolation.\n\n'
              'Capture status at certification is recorded separately in status_launcher_smoke_test.json. '
              'Pre-certification status observed PAUSED / heartbeat FAIL / chain PASS; heartbeat stopped at '
              '2026-09-27T10:26:00.389145+00:00, PID 31912 absent. No capture restart or mutation was performed. '
              'Capture automation preserved means unchanged code/policy/task, not healthy runtime.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    (run/'stdout.log').write_text('Infrastructure synthetic tests: '+result['overall']+'\n', encoding='utf-8')
    data = m['data']
    data.update(symbols=[], data_sources=['Synthetic tiny bar fixtures and source provenance metadata only'],
                source_files=m['input_snapshots'], timezone='UTC synthetic; real historical mapping unresolved',
                raw_snapshot_retained=True, reproducibility_claim='All synthetic test source retained; no real training data used',
                purge_details='N/A: no training', embargo_details='N/A: no training')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[key] = 'NOT_APPLICABLE_NO_TRAINING'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        data[key] = 0
    data['mt5_fetch'].update(used=False, not_applicable_reason='Synthetic fetch only; no approved real data requirements')
    m['model']['not_applicable_reason'] = 'Infrastructure only; no model fitting'
    m['search']['not_applicable_reason'] = 'No strategy search or outcome evaluation'
    m['registry'].update({k: 'N/A: infrastructure only' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='GOLD user-only launcher',
                         selected_configuration='PARTIAL; default unresolved; auto-fetch disabled; no training; no production change',
                         validator_result='PENDING')
    m['formal_run_status'] = 'PARTIAL'
    write(run/'manifest.json', m)
    print('FORMAL_RUN_STATUS=PARTIAL', flush=True)


if __name__ == '__main__':
    main()
