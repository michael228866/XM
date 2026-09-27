"""Independent infrastructure validator; no execution/collector imports."""
import ast
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_future_capture_pause_adjudication_v1 import load, digest, canonical
from validate_gold_future_fresh_market_resume_certification_v1 import verify_samples


def fixture_state(fixture):
    samples=fixture['samples']; expected={'broker_company':'XM Global Limited','broker_server':'XMGlobal-MT5 6',
        'source_account_environment':'demo','source_id':'XMGlobal-MT5-6_GOLD','symbol':'GOLD#','digits':2,'point':.01}
    for s in samples:
        if any(s['source_identity'].get(k)!=v for k,v in expected.items()): return 'PAUSED_SOURCE_IDENTITY'
    latest_fresh=False; has_new=False
    for i,s in enumerate(samples):
        raw,clock,epochs=s['raw_tick_epoch'],s['system_utc_epoch'],s['m1_raw_epochs']; prior=samples[i-1] if i else None
        if raw<fixture['previous_tick'] or prior and (raw<prior['raw_tick_epoch'] or clock<=prior['system_utc_epoch'] or epochs[-1]<prior['m1_raw_epochs'][-1]): return 'PAUSED_TIME_RULE'
        progress=prior is not None and (raw>prior['raw_tick_epoch'] or epochs[-1]>prior['m1_raw_epochs'][-1])
        closed=max((e for e in epochs[:-1] if e+60<=raw),default=None)
        fresh=((0<=clock-raw+10800<=120) or progress) and closed is not None and ((0<=clock-closed+10800<=180) or (progress and 60<=raw-closed<=180))
        if fresh:
            if math.floor((raw-clock)/3600+.5)*3600!=10800 or abs(raw-10800-clock)>5: return 'PAUSED_TIME_RULE'
            has_new |= closed>fixture['last_bar']
        latest_fresh=fresh
    return 'RUNNING' if latest_fresh and has_new and len({s['raw_tick_epoch'] for s in samples})>=3 else 'HEALTHY_IDLE'


def validate(run):
    root=run.parent.parent; capture=root/'future_holdout/gold_s4_v4'
    git=lambda *a:subprocess.check_output(['git',*a],cwd=root)
    m=load(run/'manifest.json'); spec=load(run/'execution_spec.json'); commit=m['source_commit']
    checks={'source_commit':commit==m['git_commit']==git('rev-parse','HEAD').decode().strip(),
        'pushed':commit==m['pre_run_remote_commit']==git('ls-remote','origin','refs/heads/main').decode().split()[0],
        'clean_source':m['git_dirty'] is False and m['pre_run_clean'] is True,
        'preserved':all(digest(root/n)==h for n,h in spec['preserved_sha256'].items()),
        'production':all(digest(root/n)==h for n,h in spec['protected_sha256'].items()),
        'validator_snapshot':Path(__file__).read_bytes()==(run/'validator_script.py').read_bytes()}
    checks['source_inventory']=set(spec['source_files'])=={e['source_path'] for e in m['input_snapshots']}
    for e in m['input_snapshots']:
        n=e['source_path']; raw=(root/n).read_bytes(); committed=git('cat-file','blob',commit+':'+n)
        checks['source:'+n]=(digest(run/e['path'])==e['sha256']==digest(root/n) and raw.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'))
    checks['original_evidence']=all(digest(root/e['source_path'])==e['sha256']==digest(run/e['path']) for e in load(run/'original_evidence_inventory.json')['files'])
    chain=load(run/'chain_before.json')
    checks['chain_unchanged']=chain=={p.relative_to(capture).as_posix():digest(p) for folder in ('snapshots','manifests') for p in (capture/folder).iterdir()}
    previous_hash=None; last_tick=None; last_bar=None
    for i,p in enumerate(sorted((capture/'manifests').iterdir())):
        item=load(p); snapshot=capture/item['snapshot_path']; payload=load(snapshot)
        checks['chain:'+str(i)]=(item['sequence']==i and item['previous_manifest_sha256']==previous_hash
            and item['snapshot_path']==f'snapshots/{i:012d}.json' and digest(snapshot)==item['snapshot_sha256'])
        previous_hash=digest(p); last_tick=payload['samples'][-1]['raw_epoch']; last_bar=payload['rows'][-1]['RAW_SOURCE_EPOCH']
    diagnostic=load(run/'current_observations.json')
    samples=[{k:s[k] for k in ('source_identity','raw_tick_epoch','system_utc_epoch','sample_time_system_utc','m1_raw_epochs')} for s in diagnostic['samples']]
    checks.update({'current:'+k:v for k,v in verify_samples(samples,diagnostic,last_tick,last_bar).items()})
    checks['safe_idle_observed']=diagnostic['classification']=='NO_FRESH_MARKET_DATA' and diagnostic['source_identity_status']=='PASS'
    migration=load(run/'runtime_idle_migration.json')
    checks['migration_honest']=(migration['previous_operational_state']=='PAUSED'
        and migration['current_observed_runtime_classification']=='NO_FRESH_MARKET_DATA'
        and migration['historical_original_pause_root_cause_proven'] is False and migration['historical_pause_reclassified'] is False
        and migration['resume_mode']=='SAFE_IDLE_MIGRATION_WITH_NO_EVIDENCE_ADMISSION'
        and migration['original_pause_sha256']==digest(capture/'PAUSED_TIME_RULE.json'))
    policy=load(run/'runtime_idle_policy.json')
    checks['policy_bytes']=(run/'runtime_idle_policy.json').read_bytes()==(root/'gold_future_capture_runtime_freshness_policy_v1.json').read_bytes()
    checks['fixed_regime']=policy['certified_offsets_seconds']==[10800] and policy['uncertified_candidate_offsets_seconds']==[7200] and policy['max_clock_skew_seconds']==5
    pending=['gold_future_capture_supervisor_v1_1.py']; scanned={}
    while pending:
        n=pending.pop()
        if n in scanned: continue
        raw=(root/n).read_bytes(); scanned[n]=digest(root/n); tree=ast.parse(raw)
        modules=[a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]+[x.module or '' for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        calls={x.func.attr if isinstance(x.func,ast.Attribute) else x.func.id if isinstance(x.func,ast.Name) else '' for x in ast.walk(tree) if isinstance(x,ast.Call)}
        checks['runtime_safe:'+n]=(not {'xgboost','lightgbm','catboost','sklearn','joblib','pickle','gemini','manual_training_launcher_v1'}.intersection(modules)
            and not {'fit','predict','predict_proba','order_send','order_check','run_training'}.intersection(calls))
        pending.extend(mod+'.py' for mod in modules if (root/(mod+'.py')).is_file())
    checks['runtime_hash_inventory']=scanned==policy['runtime_code_sha256']
    collector=(root/'gold_future_capture_collector_v4_1.py').read_text(encoding='utf-8')
    checks['frozen_full_validation_retained']=('check_samples(frozen_samples' in collector and 'validate_payload(payload, freeze, activation, previous)' in collector
        and collector.index('validation = validate_payload')<collector.index("sealed_write(CAPTURE,CAPTURE/item['snapshot_path']"))
    checks['no_original_pause_delete']='unlink(' not in collector and 'os.remove' not in collector
    regressions=load(run/'runtime_regression_results.json')
    checks['regressions']=regressions['overall']=='PASS' and all(regressions['checks'].values()) and regressions['test_count']>=24
    for f in regressions['runtime_fixtures']: checks['independent_fixture:'+f['name']]=fixture_state(f)==f['expected']==f['actual']
    manual=load(run/'training_manual_policy.json'); config=load(run/'training_launcher_config.json')
    checks['manual_policy']=(manual['training_execution_owner']=='USER' and manual['manual_launcher_required'] is True
        and all(manual[k] is False for k in ('automatic_training_by_codex','automatic_training_by_scheduler','automatic_training_on_startup','user_command_line_required','holdout_peeking_allowed','production_promotion_automatic')))
    checks['unapproved_training_disabled']=config['approval_status']=='BLOCKED_AWAITING_APPROVED_WORKFLOW' and config['workflow'] is None
    launcher=(root/'manual_training_launcher_v1.py').read_text(encoding='utf-8')
    checks['ownership_gate']='not user_double_click()' in launcher and "'explorer.exe'" in launcher and "'cmd.exe'" in launcher
    checks['bat_no_cli_required']=all(token in (root/'RUN_TRAINING.bat').read_text(encoding='utf-8') for token in ('%~dp0','manual_training_launcher_v1.py','pause >nul','PYTHONUTF8=1'))
    checks['guard_tests']=load(run/'holdout_guard_test.json')['audited_open_blocked'] is True
    task=load(run/'scheduled_task_review.json')['configuration']
    checks['task_capture_only']=task['TaskName']=='GOLD-Future-Holdout-Capture-v4' and task['RunLevel']=='Limited' and task['UserMatchesCurrent'] is True and 'start_gold_future_capture_v4.ps1' in task['Arguments']
    metrics=load(run/'metrics.json')
    checks['honest_partial']=metrics['formal_run_status']=='PARTIAL' and metrics['live_migration_status']=='PENDING_VALIDATED_MIGRATION'
    checks['no_training_or_production']=all(metrics[k] is False for k in ('model_training_executed','strategy_outcome_inspected','production_changed','production_promoted'))
    allowed='?? '+run.relative_to(root).as_posix()+'/'
    checks['only_run_untracked']=all(e.startswith(allowed) for e in git('status','--porcelain','-z').decode('utf-8').split('\0') if e)
    return checks


def main():
    run=Path(sys.argv[1]).resolve()
    with (run/'validator_attempt.json').open('x',encoding='utf-8') as out:
        json.dump({'started_at_utc':datetime.now(timezone.utc).isoformat(),'rule':'Do not retry validation'},out)
    try:
        checks=validate(run); failed=[k for k,v in checks.items() if not v]
        result={'overall':'FAIL' if failed else 'PASS','failed_check_names':failed,'checks':checks,
                'runtime_migration_approved':not failed,'actual_training_approved':False}
    except Exception as error:
        result={'overall':'FAIL','failed_check_names':['validator_exception'],'exception_type':type(error).__name__,'runtime_migration_approved':False}
    (run/'validator.json').write_bytes(canonical(result))
    (run/'validator.md').write_text('# Independent infrastructure validator\n\n'+result['overall']+'\n\n'+json.dumps(result['failed_check_names'])+'\n',encoding='utf-8')
    print(json.dumps(result)); raise SystemExit(0 if result['overall']=='PASS' else 1)


if __name__=='__main__': main()
