"""User-only entry point, inherited Explorer receipt guard, metadata-only status."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from gold_s4_entry_edge_decomposition_v1_support import ROOT, CONFIG, EXPERIMENT, read, sha, require

APPROVAL = EXPERIMENT+'_approval.json'
_SESSION = None
_PERMITS = {}


def check_policy():
    policy = read(ROOT/'manual_entry_edge_decomposition_policy_v1.json')
    require(policy['execution_owner']=='USER' and policy['user_bat_launch_required'] is True, 'USER ownership')
    require(all(policy[k] is False for k in ('real_research_allowed_for_codex','automatic_research',
        'model_training_required','signal_retraining_allowed','entry_filtering_allowed',
        'production_promotion_allowed','holdout_access_allowed')), 'Manual execution policy')


def require_session():
    if _SESSION != os.getpid() or os.environ.get('SYNTHETIC_TEST_MODE','').lower()=='true':
        raise PermissionError('Only active USER double-click workflow may read/evaluate historical entries')


def begin_session(token):
    global _SESSION
    check_policy()
    if os.environ.get('SYNTHETIC_TEST_MODE','').lower()=='true':
        raise PermissionError('Synthetic mode prohibits historical execution')
    from gold_manual_training_workflow_v1 import consume_receipt
    consume_receipt(token)
    require(_SESSION is None, 'Nested session')
    _SESSION = os.getpid()


def end_session():
    global _SESSION
    _SESSION = None
    _PERMITS.clear()


def validation_permit(run):
    require_session()
    token = secrets.token_urlsafe(32)
    _PERMITS[token] = (Path(run).resolve(),time.monotonic()+30)
    return token


def consume_validation(token,run):
    require_session()
    value = _PERMITS.pop(token,None)
    if value is None or value[0]!=Path(run).resolve() or time.monotonic()>value[1]:
        raise PermissionError('Missing/expired/consumed/wrong-run validation permit')


def verify_approval(root=ROOT):
    root = Path(root)
    a = read(root/APPROVAL)
    require(a['approved'] is True and a['workflow']=='GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1','Workflow approval')
    require(subprocess.check_output(['git','show','HEAD:'+APPROVAL],cwd=root)==(root/APPROVAL).read_bytes(),'Committed approval')
    for name,expected in a['bindings'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path)==expected,'Approved source changed: '+name)
    run = (root/a['certification_run']).resolve()
    require(run.parent==root/'training_runs' and sha(run/'FINALIZED.json')==a['finalized_sha256'],'Certification seal')
    seal = read(run/'FINALIZED.json')['file_sha256']
    for name in ('validator.json','metrics.json'):
        require(sha(run/name)==seal[name],'Certification integrity')
    require(read(run/'validator.json')['overall']=='PASS' and read(run/'metrics.json')['formal_run_status']=='PASS','Certification PASS')
    return a


def status():
    ready, error = False, None
    try:
        check_policy()
        verify_approval()
        from gold_s4_entry_edge_decomposition_v1_support import configuration
        configuration()
        ready = True
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        error = str(exc)
    previous, last = {}, 'NEVER'
    for path in sorted((ROOT/'training_runs').glob('*_'+EXPERIMENT+'/manifest.json')):
        m = read(path)
        if m.get('manual_start') is True and m.get('started_by')=='USER_LAUNCHER':
            last = path.parent.name
            previous = read(path.parent/'combined_result.json') if (path.parent/'combined_result.json').exists() else {}
    return dict(ready=ready,blocker=error,last_real_user_run=last,last_result=previous.get('research_result','NOT_RUN'),
                positive_edge_buckets=previous.get('positive_edge_buckets',0),robust_positive_buckets=previous.get('robust_positive_buckets',0),
                negative_edge_buckets=previous.get('negative_edge_buckets',0))


def main():
    if sys.argv[1:]==['--status']:
        s=status()
        for label,value in [('RESEARCH WORKFLOW','GOLD S4 Entry Edge Decomposition v1'),('OWNER','USER'),
            ('READY','YES' if s['ready'] else 'NO'),('REFERENCE','A_NO_LONG_HTF'),('REFERENCE TRADES',787),
            ('MODEL TRAINING','DISABLED'),('ENTRY FILTERING','DISABLED'),('LAST REAL USER RUN',s['last_real_user_run']),
            ('LAST RESULT',s['last_result']),('POSITIVE EDGE BUCKETS',s['positive_edge_buckets']),
            ('ROBUST POSITIVE BUCKETS',s['robust_positive_buckets']),('NEGATIVE EDGE BUCKETS',s['negative_edge_buckets'])]:
            print(str(label)+': '+str(value))
        if s['blocker']:
            print(s['blocker'])
        return 0
    require(not sys.argv[1:], 'USER double-click or --status only')
    from manual_training_launcher_v1 import user_double_click,verify_environment,load,CONFIG as launcher_config
    require(os.environ.get('SYNTHETIC_TEST_MODE','').lower()!='true' and user_double_click(),'Only USER double-click RUN_TRAINING.bat')
    print('[1/10] Environment',flush=True)
    verify_environment(load(launcher_config))
    check_policy()
    verify_approval()
    from gold_manual_training_workflow_v1 import issue_receipt
    from gold_s4_entry_edge_decomposition_v1 import run_manual
    result = run_manual(issue_receipt())
    print(result)
    return 0 if result['execution_status']=='PASS' else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('[FAIL] '+str(error),file=sys.stderr)
        raise SystemExit(1)
