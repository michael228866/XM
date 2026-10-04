"""USER launch ownership, single-use validation permits, and metadata-only status."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from gold_s4_secondary_improvement_v3_support import ROOT, CONFIG, EXPERIMENT, read, sha, require

APPROVAL = 'gold_s4_improvement_v3_approval.json'
_SESSION = None
_VALIDATION = {}


def check_policy():
    policy = read(ROOT/'manual_training_policy_v3.json')
    require(policy['training_execution_owner'] == 'USER' and policy['validation_execution_owner'] == 'USER_LAUNCHED_WORKFLOW'
            and policy['bat_user_launch_required'] is True and policy['user_bat_launch_required'] is True
            and policy['research_execution_owner'] == 'USER', 'USER ownership required')
    require(all(policy[k] is False for k in ('real_train_allowed_for_codex', 'real_validation_allowed_for_codex',
        'real_research_allowed_for_codex', 'automatic_training', 'automatic_validation', 'automatic_research')), 'Automatic real execution prohibited')


def require_session():
    if _SESSION is None or _SESSION != os.getpid() or os.environ.get('SYNTHETIC_TEST_MODE','').lower() == 'true':
        raise PermissionError('Real execution requires an active USER-launched workflow')


def begin_session(token):
    global _SESSION
    check_policy()
    if os.environ.get('SYNTHETIC_TEST_MODE','').lower() == 'true':
        raise PermissionError('Synthetic mode prohibits real execution')
    from gold_manual_training_workflow_v1 import consume_receipt
    consume_receipt(token)
    require(_SESSION is None, 'No nested session')
    _SESSION = os.getpid()


def end_session():
    global _SESSION
    _SESSION = None
    _VALIDATION.clear()


def validation_permit(run):
    require_session()
    token = secrets.token_urlsafe(32)
    _VALIDATION[token] = (Path(run).resolve(), time.monotonic()+30)
    return token


def consume_validation(token, run):
    require_session()
    value = _VALIDATION.pop(token, None)
    if value is None or value[0] != Path(run).resolve() or time.monotonic() > value[1]:
        raise PermissionError('Missing, expired, consumed, or wrong-run validation permit')


def verify_approval(root=ROOT):
    root = Path(root)
    a = read(root/APPROVAL)
    require(a['approved'] is True and a['workflow'] == 'GOLD_S4_SECONDARY_IMPROVEMENT_V3', 'V3 approval')
    require(subprocess.check_output(['git', 'show', 'HEAD:'+APPROVAL], cwd=root) == (root/APPROVAL).read_bytes(), 'Approval must be committed')
    for name, expected in a['bindings'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path) == expected, 'Approved source changed:'+name)
    run = (root/a['certification_run']).resolve()
    require(run.parent == root/'training_runs' and sha(run/'FINALIZED.json') == a['finalized_sha256'], 'Infrastructure seal')
    seal = read(run/'FINALIZED.json')['file_sha256']
    for name in ('validator.json', 'metrics.json'):
        require(sha(run/name) == seal[name], 'Certificate integrity')
    require(read(run/'validator.json')['overall'] == 'PASS' and read(run/'metrics.json')['formal_run_status'] == 'PASS', 'Infrastructure PASS required')
    return a


def status():
    ready = False
    error = None
    try:
        check_policy()
        verify_approval()
        ready = True
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        error = str(exc)
    last = 'NEVER'
    for path in sorted((ROOT/'training_runs').glob('*_'+EXPERIMENT+'/manifest.json')):
        m = read(path)
        if m.get('manual_start') is True and m.get('started_by') == 'USER_LAUNCHER' and m.get('infrastructure_only') is False:
            last = path.parent.name
    previous = read(ROOT/'training_runs'/last/'combined_result.json') if last != 'NEVER' and (ROOT/'training_runs'/last/'combined_result.json').exists() else {}
    return dict(last_research_result=previous.get('research_result','NOT_RUN'), last_candidate=(previous.get('candidate') or {}).get('candidate_id'),
                last_gate=previous.get('candidate_gate','NONE'),last_economic_status=previous.get('economic_status','NOT_RUN'),workflow='GOLD S4 Improvement v3', owner='USER', ready=ready,
                reference='A_NO_LONG_HTF', last_real_user_run=last, blocker=error)


def main():
    if sys.argv[1:] == ['--status']:
        s = status()
        print('TRAINING WORKFLOW:\n'+s['workflow']+'\nOWNER:\nUSER\nREADY:\n'+('YES' if s['ready'] else 'NO')+
              '\nREFERENCE:\n'+s['reference']+'\nLAST REAL USER RUN:\n'+s['last_real_user_run'])
        for label,key in [('LAST RESEARCH RESULT','last_research_result'),('LAST CANDIDATE','last_candidate'),('LAST GATE','last_gate'),('LAST ECONOMIC STATUS','last_economic_status')]:
            print(label+':\n'+str(s[key]))
        print('Production:\nUNCHANGED')
        if s['blocker']:
            print(s['blocker'])
        return 0
    require(not sys.argv[1:], 'Use USER double-click or --status')
    from manual_training_launcher_v1 import user_double_click, verify_environment, load, CONFIG as old_config
    require(user_double_click(), 'Only USER double-click RUN_TRAINING.bat may initiate real Train + Val')
    print('========================================\nXM GOLD S4 Improvement v3\nTrain + Validate\n========================================')
    print('[1/9] 檢查環境', flush=True)
    verify_environment(load(old_config))
    check_policy()
    verify_approval()
    from gold_manual_training_workflow_v1 import issue_receipt
    from gold_s4_secondary_improvement_v3 import run_manual, format_result
    result = run_manual(issue_receipt())
    print(format_result(result))
    return 0 if result['execution_status'] == 'PASS' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('[失敗] '+str(error), file=sys.stderr)
        raise SystemExit(1)
