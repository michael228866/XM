"""One-shot independent provenance, eligibility and operational deployment checks."""
import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_future_holdout_feature_eligibility_v1 import verify, load, encode, sha


def validate(run):
    root = run.parent.parent
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root)
    m = load(run/'manifest.json'); spec = load(run/'execution_spec.json')
    commit = m['source_commit']
    checks = {'source_commit': commit == m['git_commit'] == git('rev-parse', 'HEAD').decode().strip(),
        'remote_commit': commit == m['pre_run_remote_commit'] == git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0],
        'clean_provenance': m['git_dirty'] is False and m['pre_run_clean'] is True,
        'protected': all(sha((root/n).read_bytes()) == h for n, h in spec['protected_sha256'].items()),
        'frozen_and_prior_artifacts': all(sha((root/n).read_bytes()) == h for n, h in spec['preserved_sha256'].items())}
    for entry in m['input_snapshots']:
        name = entry['source_path']; path = run/entry['path']
        checks['source:'+name] = (sha(path.read_bytes()) == entry['sha256'] == sha((root/name).read_bytes())
            and path.read_bytes().replace(b'\r\n', b'\n') == git('cat-file', 'blob', commit+':'+name).replace(b'\r\n', b'\n'))
    checks['source_inventory'] = set(spec['source_files']) == {e['source_path'] for e in m['input_snapshots']}
    checks['validator_snapshot'] = Path(__file__).read_bytes() == (run/'validator_script.py').read_bytes()
    eligibility = load(run/'feature_eligibility.json')
    binding = load(run/'feature_binding.json')
    result = verify(binding, eligibility)
    checks['independent_eligibility'] = result['overall'] == 'PASS'
    checks['audit_alias'] = eligibility == load(run/'feature_prerequisite_audit.json')
    decision = load(run/'evaluation_start_decision.json')
    checks['original_boundary_preserved'] = decision['original_boundary_sha256'] == sha((root/'gold_future_holdout_boundary_v4.json').read_bytes())
    checks['evaluation_start_pending_independent_validation'] = decision['holdout_evaluation_start'] is None and decision['proposed_timestamp'] == result['holdout_evaluation_start']
    static = load(run/'supervisor_static_review.json')
    checks['supervisor_hashes'] = static['status'] == 'PASS' and static['scanned_source_sha256'] == spec['runtime_source_sha256']
    prohibited_imports = {'xgboost', 'lightgbm', 'catboost', 'sklearn', 'joblib', 'pickle', 'gemini'}
    for name, expected in spec['runtime_source_sha256'].items():
        raw = (root/name).read_bytes(); tree = ast.parse(raw)
        imports = [a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        imports += [(n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else '' for n in ast.walk(tree) if isinstance(n, ast.Call)]
        checks['runtime_safety:'+name] = sha(raw) == expected and not prohibited_imports.intersection(imports) and not {'fit', 'predict', 'predict_proba', 'load_model', 'order_send', 'order_check'}.intersection(calls)
    task = load(run/'scheduled_task_config.json')
    if task['status'] == 'INSTALLED':
        config = task['configuration']
        checks['task_safety'] = (config['TaskName'] == 'GOLD-Future-Holdout-Capture-v4' and config['RunLevel'] == 'Limited'
            and config['LogonType'] == 'Interactive' and config['MultipleInstances'] == 'IgnoreNew' and config['AtLogon'] is True and config['UserMatchesCurrent'] is True
            and 'start_gold_future_capture_v4.ps1' in config['Arguments'] and 'gemini' not in config['Arguments'].lower()
            and config['WorkingDirectory'] == str(root) and config['Execute'].lower().endswith('powershell.exe'))
    else:
        checks['persistence_blocker_honest'] = task['install_exit_code'] != 0
    health = load(run/'capture_health_initial.json'); coverage = load(run/'capture_coverage_initial.json')
    checks['heartbeat_present'] = bool(health.get('supervisor_started_at_utc')) and health.get('process_lock_status') == 'PASS'
    checks['heartbeat_raw_only'] = not {'OPEN', 'CLOSE', 'HIGH', 'LOW', 'prices', 'features', 'signals', 'returns'}.intersection(health)
    checks['last_chain_state'] = health.get('chain_status') == 'PASS' and coverage['last_sequence'] == health.get('last_successful_snapshot_sequence')
    markers = load(run/'capture_pause_receipts.json')
    if health['collector_status'] == 'PAUSED':
        checks['pause_retained'] = bool(markers) and all(sha((root/e['path']).read_bytes()) == e['sha256'] for e in markers)
        checks['no_policy_auto_resume'] = health.get('process_running') is False
    metrics = load(run/'metrics.json')
    expected_pass = result['feature_eligibility_status'] == 'PASS' and health['collector_status'] == 'RUNNING' and task['status'] == 'INSTALLED'
    checks['verdict'] = metrics['formal_run_status'] == ('PASS' if expected_pass else 'PARTIAL')
    checks['no_outcome_metrics'] = not {'wr', 'pf', 'pnl', 'mean_r', 'trades', 'returns', 'predictions'}.intersection(k.lower() for k in metrics)
    checks['protection_stages'] = all(v == spec['protected_sha256'] for v in load(run/'protection_checks.json').values())
    allowed = '?? '+run.relative_to(root).as_posix()+'/'
    checks['only_formal_run_untracked'] = all(e.startswith(allowed) for e in git('status', '--porcelain', '-z').decode('utf-8').split('\0') if e)
    return checks, result


def main():
    run = Path(sys.argv[1]).resolve()
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as f:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'rule': 'Do not retry validation'}, f)
    try:
        checks, eligibility = validate(run)
        failed = [k for k, v in checks.items() if not v]
        result = {'overall': 'PASS' if not failed else 'FAIL', 'failed_check_names': failed, 'checks': checks, 'eligibility_validation': eligibility}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': ['validator_exception'], 'exception_type': type(error).__name__}
    (run/'validator.json').write_bytes(encode(result))
    (run/'validator.md').write_text('# Independent validator\n\n'+result['overall']+'\n\n'+json.dumps(result['failed_check_names']), encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
