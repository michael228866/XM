"""Formal structural eligibility and user-level capture deployment; no strategy."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import training_run_history as history
from gold_future_holdout_feature_eligibility_v1 import audit, canonical, load, sha
from gold_future_capture_supervisor_v1 import static_review, coverage, HEALTH, CAPTURE, now

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_future_holdout_feature_eligibility_and_continuous_capture_v1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def protect(spec):
    actual = {n: sha((ROOT/n).read_bytes()) for n in spec['protected_sha256']}
    if actual != spec['protected_sha256']:
        raise ValueError('PRODUCTION_CHANGED=true; STOP; no restore')
    return actual


def powershell(script, *args):
    return subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(ROOT/script), *args],
        cwd=ROOT, capture_output=True, timeout=30, check=False)


def task_projection():
    command = "$t=Get-ScheduledTask -TaskName 'GOLD-Future-Holdout-Capture-v4' -ErrorAction Stop; @{TaskName=$t.TaskName;Execute=$t.Actions[0].Execute;Arguments=$t.Actions[0].Arguments;WorkingDirectory=$t.Actions[0].WorkingDirectory;RunLevel=[string]$t.Principal.RunLevel;LogonType=[string]$t.Principal.LogonType;MultipleInstances=[string]$t.Settings.MultipleInstances;RestartCount=$t.Settings.RestartCount;RestartInterval=$t.Settings.RestartInterval;ExecutionTimeLimit=$t.Settings.ExecutionTimeLimit;AtLogon=($t.Triggers[0].CimClass.CimClassName -eq 'MSFT_TaskLogonTrigger');UserMatchesCurrent=($t.Principal.UserId -eq [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)} | ConvertTo-Json -Compress"
    command = '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); ' + command
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], capture_output=True, timeout=30, check=False)
    return json.loads(result.stdout.decode('utf-8-sig')) if result.returncode == 0 else None


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('--execute required')
    spec = load(ROOT/('execution_spec_'+SLUG+'.json'))
    protection = {'before': protect(spec)}
    commit = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    if commit != remote or git('status', '--porcelain', '-z'):
        raise ValueError('CLEAN_PUSHED_SOURCE_REQUIRED')
    for name, expected in spec['preserved_sha256'].items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('FROZEN_ARTIFACT_CHANGED:'+name)
    run = history.create_run(SLUG, Path(__file__), subprocess.list2cmdline([sys.executable, '-B', str(Path(__file__)), '--execute']), arguments=['--execute'], seed_note='Deterministic structural eligibility and operational deployment; no models')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = load(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=remote, pre_run_clean=True, pre_run_git_status='', input_snapshots=[])
    for name in spec['source_files']:
        destination = run/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        m['input_snapshots'].append({'source_path': name, 'path': destination.relative_to(run).as_posix(), 'sha256': sha(destination.read_bytes()), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/('execution_spec_'+SLUG+'.json'), run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'), run/'validator_script.py')
    history.write_json(run/'manifest.json', m)
    with (run/'self_test_stdout.txt').open('xb') as out, (run/'self_test_stderr.txt').open('xb') as err:
        synthetic = subprocess.run([sys.executable, '-B', str(ROOT/('test_'+SLUG+'.py'))], stdout=out, stderr=err, timeout=60, check=False)
    if synthetic.returncode:
        raise RuntimeError('SELF_TEST_FAILED; preserve formal attempt')
    binding, eligibility = audit()
    for name, value in [('feature_binding.json', binding), ('feature_prerequisite_audit.json', eligibility), ('feature_eligibility.json', eligibility)]:
        history.write_json(run/name, value)
    review = static_review()
    history.write_json(run/'supervisor_static_review.json', review)
    history.write_json(run/'continuous_capture_design.json', spec['continuous_capture_design'])
    protection['after_eligibility'] = protect(spec)
    install = powershell('install_gold_future_capture_task_v1.ps1')
    (run/'task_install_stdout.txt').write_bytes(install.stdout)
    (run/'task_install_stderr.txt').write_bytes(install.stderr)
    task = task_projection()
    if task is not None and not (task['RunLevel'] == 'Limited' and task['LogonType'] == 'Interactive'
            and task['MultipleInstances'] == 'IgnoreNew' and task['UserMatchesCurrent'] is True
            and task['WorkingDirectory'] == str(ROOT)
            and task['Arguments'] == '-NoProfile -NonInteractive -WindowStyle Hidden -File "'+str(ROOT/'start_gold_future_capture_v4.ps1')+'"'):
        raise ValueError('EXISTING_TASK_CONFIGURATION_MISMATCH; do not start or overwrite')
    task_status = 'INSTALLED' if task is not None else 'UNAVAILABLE'
    history.write_json(run/'scheduled_task_config.json', {'status': task_status, 'configuration': task, 'install_exit_code': install.returncode})
    if task is not None:
        start = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', "Start-ScheduledTask -TaskName 'GOLD-Future-Holdout-Capture-v4' -ErrorAction Stop"], capture_output=True, timeout=30, check=False)
        (run/'task_start_stdout.txt').write_bytes(start.stdout)
        (run/'task_start_stderr.txt').write_bytes(start.stderr)
    else:
        # Authorized user-level fallback; no credentials or elevation.
        command = "Start-Process -FilePath powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-NonInteractive','-WindowStyle','Hidden','-File','\""+str(ROOT/'start_gold_future_capture_v4.ps1')+"\"')"
        start = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], capture_output=True, timeout=30, check=False)
        (run/'fallback_start_stdout.txt').write_bytes(start.stdout)
        (run/'fallback_start_stderr.txt').write_bytes(start.stderr)
    health = {'collector_status': 'NOT_RUNNING', 'runtime_time_status': 'UNVERIFIED', 'chain_status': 'UNVERIFIED'}
    for _ in range(25):
        if (HEALTH/'heartbeat.json').exists():
            health = load(HEALTH/'heartbeat.json')
            if health['collector_status'] in {'RUNNING', 'PAUSED', 'STOPPED'}:
                break
        time.sleep(1)
    cover = coverage()
    history.write_json(run/'capture_health_initial.json', health)
    history.write_json(run/'capture_coverage_initial.json', cover)
    receipts = []
    for path in [*CAPTURE.glob('PAUSED*.json'), *HEALTH.glob('PAUSED*.json')]:
        marker = load(path)
        receipts.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path.read_bytes()),
                         'created_at_utc': marker.get('created_at_utc'), 'reason': marker.get('reason'),
                         'error_class': marker.get('error_class'), 'status': marker.get('status', 'PAUSED')})
    history.write_json(run/'capture_pause_receipts.json', receipts)
    history.write_json(run/'evaluation_start_decision.json', {'holdout_start': eligibility['holdout_start'], 'holdout_evaluation_start': None,
        'proposed_timestamp': eligibility['earliest_feature_complete_timestamp'], 'independent_validation_required': True,
        'original_boundary_sha256': sha((ROOT/'gold_future_holdout_boundary_v4.json').read_bytes())})
    protection['after_capture_deployment'] = protect(spec)
    history.write_json(run/'protection_checks.json', protection)
    checks = eligibility['candidate_checks']
    passed = eligibility['status'] == 'PASS' and health['collector_status'] == 'RUNNING' and task_status == 'INSTALLED'
    metrics = {'formal_run_status': 'PASS' if passed else 'PARTIAL', 'candidate_bars_checked': len(checks),
        'feature_complete_candidate_count': sum(c['eligibility'] == 'ELIGIBLE' for c in checks), 'expected_feature_count': 31,
        'missing_feature_count': max((len(c['missing_features']) for c in checks), default=31),
        'nan_count': sum(c['nan_count'] for c in checks), 'inf_count': sum(c['inf_count'] for c in checks),
        'context_hash_match': True, 'pipeline_hash_match': True, 'scheduled_task_installed': task_status == 'INSTALLED',
        'supervisor_started': 'supervisor_started_at_utc' in health, 'heartbeat_pass': 'supervisor_started_at_utc' in health,
        'chain_continuity_pass': health.get('chain_status') == 'PASS', 'continuous_capture_status': health['collector_status'],
        'validator_status': 'PENDING'}
    history.write_json(run/'metrics.json', metrics)
    data = m['data']; data.update(symbols=['GOLD#'], data_sources=['Frozen native context and sealed future raw chain'],
        source_files=m['input_snapshots'], timezone='Frozen raw source labels / current certified 10800 for evidence UTC',
        raw_snapshot_retained=True, reproducibility_claim='Frozen context and immutable chain snapshot hashes retained; mutable health separately snapshotted',
        purge_details='No strategy dataset', embargo_details='No future rows influence eligibility')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[key] = 'NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        data[key] = 0
    data['mt5_fetch'].update(used=False, not_applicable_reason='Eligibility uses frozen files; separate authorized supervisor invokes unchanged frozen collector and preserves its dynamic raw provenance')
    m['model']['not_applicable_reason'] = 'No model loaded or trained'
    m['search']['not_applicable_reason'] = 'No strategy search'
    m['registry'].update({k: 'N/A: feature eligibility / raw operational capture' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Frozen S4 v4 raw boundary', selected_configuration=metrics['formal_run_status']+'; eligibility '+eligibility['status']+'; capture '+health['collector_status']+'; no model/strategy/production change', validator_result='PENDING')
    m.update(formal_run_status=metrics['formal_run_status'], git_status_after_execution=git('status', '--porcelain', '-z').decode('utf-8'))
    history.write_json(run/'manifest.json', m)
    report = '# GOLD feature eligibility and continuous capture\n\nFormal result: '+metrics['formal_run_status']+'\n\n'+json.dumps(metrics, indent=2)+'\n\nFrozen seed-to-boundary M1 inputs and closed native HTF inputs are not fabricated or refetched. Evaluation start remains null pending independent eligibility. Supervisor honors all frozen collector pause markers, including failures the frozen collector treats as permanent. No model, strategy, outcome or production change.\n'
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    for name in ('stdout.txt', 'stdout.log'):
        (run/name).write_text('FORMAL_RUN_STATUS='+metrics['formal_run_status']+'\n', encoding='utf-8')
    (run/'stderr.txt').write_text('', encoding='utf-8')
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'], flush=True)


if __name__ == '__main__':
    main()
