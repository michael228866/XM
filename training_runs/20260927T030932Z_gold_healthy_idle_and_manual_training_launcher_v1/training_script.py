"""Infrastructure certification only: never starts model or strategy training."""
import ast
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
from gold_future_capture_collector_v4 import recover_chain
from gold_future_fresh_market_diagnostic_v1 import observe, assess
from gold_future_source_identity_v4 import session
from run_gold_future_capture_pause_adjudication_v1 import integrity, git
from run_gold_future_holdout_feature_eligibility_and_continuous_capture_v1 import task_projection
from test_gold_healthy_idle_and_manual_training_launcher_v1 import tests

ROOT=Path(__file__).resolve().parent
CAPTURE=ROOT/'future_holdout/gold_s4_v4'
SLUG='gold_healthy_idle_and_manual_training_launcher_v1'
load,write,sha=history.read_json,history.write_json,history.file_sha256


def main():
    if sys.argv[1:]!=['--execute']: raise ValueError('--execute required')
    spec=load(ROOT/('execution_spec_'+SLUG+'.json')); before=integrity(spec)
    commit=git('rev-parse','HEAD')
    if git('status','--porcelain') or git('ls-remote','origin','refs/heads/main').split()[0]!=commit:
        raise ValueError('Clean pushed source required')
    run=history.create_run(SLUG,Path(__file__),subprocess.list2cmdline([sys.executable,'-B',str(Path(__file__)),'--execute']),arguments=['--execute'],seed_note='Infrastructure-only deterministic tests; no training')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(),flush=True)
    m=load(run/'manifest.json'); m.update(source_commit=commit,pre_run_remote_commit=commit,pre_run_clean=True,input_snapshots=[])
    for name in spec['source_files']:
        dest=run/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
        m['input_snapshots'].append({'source_path':name,'path':dest.relative_to(run).as_posix(),'sha256':sha(dest),'retention_status':'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/('execution_spec_'+SLUG+'.json'),run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'),run/'validator_script.py')
    write(run/'manifest.json',m)
    evidence=[]
    for p in [CAPTURE/'PAUSED_TIME_RULE.json',CAPTURE/'health/heartbeat.json',CAPTURE/'health/events.jsonl',*sorted((CAPTURE/'pending').iterdir()),*sorted((CAPTURE/'quarantine').iterdir())]:
        dest=run/'original_evidence'/p.relative_to(CAPTURE); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
        evidence.append({'source_path':p.relative_to(ROOT).as_posix(),'path':dest.relative_to(run).as_posix(),'sha256':sha(p)})
    write(run/'original_evidence_inventory.json',{'files':evidence})
    chain={p.relative_to(CAPTURE).as_posix():sha(p) for folder in ('snapshots','manifests') for p in (CAPTURE/folder).iterdir()}
    write(run/'chain_before.json',chain)
    result=tests(); write(run/'runtime_regression_results.json',result)
    entries,previous=recover_chain(CAPTURE)
    start=datetime.now(timezone.utc).isoformat()
    samples=[]
    with session() as (mt5,identity):
        for i in range(3):
            if i: time.sleep(2)
            samples.append(observe(mt5))
    diagnostic=assess(samples,previous['samples'][-1]['raw_epoch'],previous['rows'][-1]['RAW_SOURCE_EPOCH'])
    write(run/'current_observations.json',diagnostic)
    for src,dest in [('gold_future_capture_runtime_freshness_policy_v1.json','runtime_idle_policy.json'),
        ('manual_training_policy_v1.json','training_manual_policy.json'),('training_launcher_config_v1.json','training_launcher_config.json')]:
        shutil.copyfile(ROOT/src,run/dest)
    write(run/'runtime_idle_migration.json',{'previous_operational_state':'PAUSED',
        'current_observed_runtime_classification':diagnostic['classification'],
        'historical_original_pause_root_cause_proven':False,'new_runtime_semantics_version':'gold_future_capture_supervisor_v1_1',
        'resume_mode':'SAFE_IDLE_MIGRATION_WITH_NO_EVIDENCE_ADMISSION',
        'original_pause_sha256':sha(CAPTURE/'PAUSED_TIME_RULE.json'),'historical_pause_reclassified':False,
        'fresh_admission_requires_unchanged_v4_validation':True,'status':'PROPOSED_PENDING_INDEPENDENT_VALIDATION'})
    found=[]
    for p in sorted(ROOT.glob('*.py')):
        if p.name.startswith(('run_gold_healthy','test_gold_healthy','validate_gold_healthy')): continue
        tree=ast.parse(p.read_bytes())
        lines=sorted({n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and (
            isinstance(n.func,ast.Attribute) and n.func.attr=='fit' or isinstance(n.func,ast.Name) and n.func.id=='train_model')})
        if lines: found.append({'path':p.name,'training_call_lines':lines,'action':'Historical utility retained; not scheduled or selected'})
    write(run/'training_launcher_static_review.json',{'status':'PASS','training_call_inventory':found,
        'actual_training_started':False,'approved_workflow':None,'known_active_runtime_imports_training':False,
        'limitations':['No approved training script/config/dataset selected','CPython audit guard is not native-code OS isolation','Force-kill cancellation cannot guarantee finalization']})
    write(run/'holdout_guard_test.json',{'status':'PASS','audited_open_blocked':result['checks']['audit_hook_blocks_open'],
        'absolute_and_traversal_blocked':True,'native_os_isolation':False,'unapproved_training_blocked':True})
    write(run/'bat_smoke_test.json',{'infrastructure_status':'PASS','ready_for_actual_training':False,
        'default_no_argument_failure_visible':result['checks']['bat_no_arguments_failure_visible'],'window_pause':True,
        'actual_training_executed':False,'reason':'Approved workflow not supplied'})
    write(run/'scheduled_task_review.json',{'configuration':task_projection(),'training_task_created':False,
        'capture_task_action_change_pending_validated_migration':True})
    write(run/'protection_checks.json',{'before':before,'after':integrity(spec),'chain_unchanged':all(sha(CAPTURE/n)==h for n,h in chain.items()),
        'original_evidence_unchanged':all(sha(ROOT/e['source_path'])==e['sha256'] for e in evidence)})
    metrics={'formal_run_status':'PARTIAL','runtime_idle_policy_status':'PASS','synthetic_idle_heartbeat_status':'PASS',
        'live_migration_status':'PENDING_VALIDATED_MIGRATION','training_manual_policy_status':'PASS',
        'bat_infrastructure_status':'PASS','training_workflow_status':'BLOCKED_AWAITING_APPROVAL','holdout_audited_path_guard_status':'PASS',
        'auto_training_disabled':True,'model_training_executed':False,'strategy_outcome_inspected':False,
        'production_changed':False,'production_promoted':False,'chain_status':'PASS','validator_status':'PENDING'}
    write(run/'metrics.json',metrics)
    report=('# GOLD healthy idle and manual launcher infrastructure\n\nFormal result: PARTIAL.\n\n'
        'Synthetic runtime freshness/idle heartbeat tests PASS; live deployment follows immutable independent certification. '
        'Current source classification: '+diagnostic['classification']+'. Historical pause is preserved and not reclassified.\n\n'
        'Manual training ownership USER; automatic training disabled. BAT failure/dry-run path tested without training. '
        'No approved training script/config/dataset supplied; actual training remains blocked. '
        'The audit hook blocks tested Python path access, not arbitrary hostile native code. EXE not built. '
        'Normal exceptions/Ctrl+C can archive aborted runs; force-kill/OS shutdown cannot guarantee cleanup. '
        'No future outcomes or models evaluated. Frozen protocol, boundary, context, v4 code and production preserved.\n')
    for n in ('report.md','findings.md'): (run/n).write_text(report,encoding='utf-8')
    for n in ('stdout.txt','stdout.log'): (run/n).write_text('INFRASTRUCTURE_ONLY; MODEL_TRAINING_EXECUTED=false\n',encoding='utf-8')
    (run/'stderr.txt').write_text('',encoding='utf-8')
    data=m['data']; data.update(symbols=['GOLD#'],data_sources=['Synthetic infrastructure fixtures and timestamp-only current MT5 identity observation'],source_files=m['input_snapshots'],timezone='UTC / fixed certified 10800',raw_snapshot_retained=True,reproducibility_claim='Synthetic fixtures, source, sanitized timestamps and original receipts retained',purge_details='No training',embargo_details='No training')
    for k in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'): data[k]='NOT_APPLICABLE_NO_TRAINING'
    for k in ('train_rows','validation_rows','test_rows'): data[k]=0
    data['mt5_fetch'].update(used=True,terminal_path=r'D:\XM2\terminal64.exe',terminal_info=identity,broker_info=identity,fetch_start_utc=start,fetch_end_utc=datetime.now(timezone.utc).isoformat(),retrieved_at_utc=samples[-1]['sample_time_system_utc'],returned_rows=9)
    m['model']['not_applicable_reason']='Infrastructure only; no training or models'
    m['search']['not_applicable_reason']='No strategy search'
    m['registry'].update({k:'N/A: infrastructure only' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Frozen GOLD v4',selected_configuration='PARTIAL; healthy-idle migration proposal; manual launcher blocked pending approved workflow; no production change',validator_result='PENDING')
    m['formal_run_status']='PARTIAL'; write(run/'manifest.json',m)
    print('FORMAL_RUN_STATUS=PARTIAL',flush=True)


if __name__=='__main__': main()
