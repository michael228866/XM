"""Infrastructure-only tests; no model training or real evidence admission."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from gold_future_capture_collector_v4_1 import operational_decision
from gold_future_capture_supervisor_v1_1 import apply_cycle, publish
from manual_training_launcher_v1 import ROOT, load, verify_environment, verify_workflow, user_double_click
from training_holdout_guard_v1 import check_path


def tests():
    raw = 1800000000
    identity = {'broker_company':'XM Global Limited','broker_server':'XMGlobal-MT5 6',
                'source_account_environment':'demo','source_id':'XMGlobal-MT5-6_GOLD','symbol':'GOLD#','digits':2,'point':.01}
    base = [{'source_identity':identity.copy(),'raw_tick_epoch':raw+i*2,
             'system_utc_epoch':raw+i*2-10800+.5,'m1_raw_epochs':[raw-120,raw-60,raw]} for i in range(3)]
    stale = copy.deepcopy(base)
    for s in stale: s['raw_tick_epoch']=raw; s['system_utc_epoch']+=86400
    checks = {}; fixtures=[]
    def observe_case(name,samples,expected):
        try:
            decision=operational_decision(samples,raw-300,raw-180)
            actual=decision['collector_status']
        except ValueError as error:
            actual=str(error); decision=None
        checks[name]=actual==expected
        fixtures.append({'name':name,'samples':samples,'previous_tick':raw-300,'last_bar':raw-180,'expected':expected,'actual':actual})
        return decision
    idle=observe_case('stale_to_healthy_idle',stale,'HEALTHY_IDLE')
    checks['stale_no_admission']=idle['evidence_admitted'] is False
    checks['stale_null_offset_skew']=all(r['inferred_offset_seconds'] is None and r['normalized_clock_error_seconds'] is None for r in idle['diagnostic']['samples'])
    observe_case('fresh_10800_full_validation_required',base,'RUNNING')
    for offset,expected in [(7200,'PAUSED_TIME_RULE'),(14400,'PAUSED_TIME_RULE')]:
        s=copy.deepcopy(base)
        for row in s: row['system_utc_epoch']=row['raw_tick_epoch']-offset+.5
        observe_case('fresh_offset_'+str(offset),s,expected)
    s=copy.deepcopy(base)
    for row in s: row['system_utc_epoch']+=6
    observe_case('excessive_skew',s,'PAUSED_TIME_RULE')
    s=copy.deepcopy(base); s[-1]['raw_tick_epoch']=raw-1
    observe_case('reversal',s,'PAUSED_TIME_RULE')
    s=copy.deepcopy(stale); s[0]['source_identity']['broker_server']='other'
    observe_case('identity_mismatch',s,'PAUSED_SOURCE_IDENTITY')
    with tempfile.TemporaryDirectory(prefix='gold_idle_') as folder:
        health=Path(folder); heartbeat={'pid':os.getpid()}
        apply_cycle(heartbeat,idle,0,'2026-09-25T19:24:00+00:00')
        publish(heartbeat,health); first=load(health/'heartbeat.json')
        time.sleep(.02); publish(heartbeat,health); second=load(health/'heartbeat.json')
        checks['idle_heartbeat_updates']=first['heartbeat_updated_at_utc']<second['heartbeat_updated_at_utc']
        checks['idle_sequence_unchanged']=second['last_successful_snapshot_sequence']==0
        checks['idle_no_snapshot_created']=set(p.name for p in health.iterdir())=={'heartbeat.json'}
    config=load(ROOT/'training_launcher_config_v1.json')
    verify_environment(config)
    checks['correct_python']=Path(sys.executable).resolve()==(ROOT/config['selected_interpreter']).resolve()
    checks['unicode_root']=ROOT.name=='數據'
    bad=copy.deepcopy(config); bad['selected_interpreter']='not-present/python.exe'
    try: verify_environment(bad)
    except ValueError: checks['invalid_python_blocks']=True
    bad=copy.deepcopy(config); bad['protected_sha256']['gemini.py']='0'*64
    try: verify_environment(bad)
    except ValueError: checks['production_mismatch_blocks']=True
    try: verify_workflow(config)
    except ValueError: checks['unapproved_workflow_blocks']=True
    for name,path in [('absolute',ROOT/'future_holdout/gold_s4_v4/context_seed/raw/M1.json'),
                      ('traversal',ROOT/'future_holdout/../future_holdout/gold_s4_v4/protocol/freeze.json')]:
        try: check_path(path,ROOT)
        except PermissionError: checks['holdout_'+name+'_blocked']=True
    child="from pathlib import Path; from training_holdout_guard_v1 import install; install(Path.cwd()); open('future_holdout/gold_s4_v4/protocol/freeze.json','rb')"
    p=subprocess.run([sys.executable,'-B','-c',child],cwd=ROOT,capture_output=True,check=False)
    checks['audit_hook_blocks_open']=p.returncode!=0 and b'PermissionError' in p.stderr
    checks['non_user_scheduler_agent_rejected']=not user_double_click()
    before=set((ROOT/'training_runs').iterdir())
    p=subprocess.run([sys.executable,'-B','-c','import manual_training_launcher_v1'],cwd=ROOT,capture_output=True,check=False)
    checks['import_does_not_train']=p.returncode==0 and set((ROOT/'training_runs').iterdir())==before
    p=subprocess.run([sys.executable,'-B','manual_training_launcher_v1.py','--dry-run'],cwd=ROOT,capture_output=True,check=False)
    checks['dry_run_no_training']=p.returncode==0 and json.loads(p.stdout)['training_executed'] is False
    bat=subprocess.run(['cmd.exe','/d','/c',str(ROOT/'RUN_TRAINING.bat')],cwd=ROOT,input=b'\r\n',capture_output=True,timeout=30,check=False)
    checks['bat_no_arguments_failure_visible']='[失敗]'.encode() in bat.stdout+bat.stderr
    checks['bat_window_pause']='pause >nul' in (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['bat_no_training_created']=set((ROOT/'training_runs').iterdir())==before
    # Meaningful aborted archive fixture, explicitly not a training execution.
    import training_run_history as history
    with tempfile.TemporaryDirectory(prefix='gold_abort_fixture_') as folder:
        with patch.object(history,'RUNS_ROOT',Path(folder)):
            run=history.create_run('launcher_infrastructure_fixture',Path(__file__),'synthetic infrastructure fixture',seed_note='No training')
        (run/'stdout.txt').write_text('fixture only\n'); (run/'stderr.txt').write_text('simulated cancellation\n')
        (run/'validator.json').write_text('{"overall":"NOT_RUN_CANCELLED"}')
        errors=history.finalize_run(run,'aborted',aborted_reason='Synthetic launcher cancellation fixture; no training')
        checks['aborted_archive_preserved']=not errors and (run/'FINALIZED.json').is_file()
        checks['logs_preserved']=(run/'stdout.txt').is_file() and (run/'stderr.txt').is_file()
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    return {'overall':'PASS','checks':checks,'test_count':len(checks),'runtime_fixtures':fixtures,
            'bat_smoke_scope':'Programmatic failure-path smoke; real Explorer double-click training not executed',
            'training_executed':False,'training_workflow_status':'BLOCKED_AWAITING_APPROVAL',
            'holdout_guard_scope':'CPython audited path access; not hostile native-code isolation'}


if __name__=='__main__':
    print(json.dumps(tests(),ensure_ascii=False))
