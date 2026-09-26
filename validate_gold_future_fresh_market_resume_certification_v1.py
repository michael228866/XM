"""Independent one-shot fresh-market verifier: no execution implementation imports."""
import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_future_capture_pause_adjudication_v1 import canonical, digest, load


def verify_samples(raw_samples, result, last_tick, last_bar):
    expected_identity = {'broker_company':'XM Global Limited','broker_server':'XMGlobal-MT5 6',
        'source_account_environment':'demo','source_id':'XMGlobal-MT5-6_GOLD','symbol':'GOLD#','digits':2,'point':.01}
    checks = {'sample_count':len(raw_samples)==len(result['samples'])}
    fresh_ticks, fresh_bars, offsets, errors = set(),set(),set(),[]
    conflict=None
    identity_pass=bool(raw_samples)
    for i,s in enumerate(raw_samples):
        row=result['samples'][i]
        raw,clock,epochs=s['raw_tick_epoch'],s['system_utc_epoch'],s['m1_raw_epochs']
        identity=all(s['source_identity'].get(k)==v for k,v in expected_identity.items())
        identity_pass &= identity
        checks['sample_schema:'+str(i)]=(type(raw) is int and type(clock) in (int,float) and math.isfinite(clock)
            and len(epochs)==3 and all(type(e) is int and e%60==0 for e in epochs)
            and epochs[1]-epochs[0]==epochs[2]-epochs[1]==60)
        checks['system_timestamp:'+str(i)]=s['sample_time_system_utc']==datetime.fromtimestamp(clock,timezone.utc).isoformat()
        prior=raw_samples[i-1] if i else None
        advancing=prior is not None and (raw>prior['raw_tick_epoch'] or epochs[-1]>prior['m1_raw_epochs'][-1])
        monotonic=raw>=last_tick and (prior is None or (raw>=prior['raw_tick_epoch']
            and clock>prior['system_utc_epoch'] and epochs[-1]>=prior['m1_raw_epochs'][-1]))
        closed=max((e for e in epochs[:-1] if e+60<=raw),default=None)
        age=clock-raw+10800
        bar_age=clock-closed+10800 if closed is not None else None
        tick_ok=(0<=age<=120) or advancing
        bar_ok=closed is not None and ((0<=bar_age<=180) or (advancing and 60<=raw-closed<=180))
        fresh=identity and monotonic and tick_ok and bar_ok
        state='NO_FRESH_MARKET_DATA'; rule=None; offset=None; error=None
        if not identity: state,rule='TIME_RULE_CONFLICT','SOURCE_IDENTITY_CHANGED'
        elif not monotonic: state,rule='TIME_RULE_CONFLICT','TIMESTAMP_REVERSAL'
        elif fresh:
            offset=3600*math.floor((raw-clock)/3600+.5)
            error=raw-10800-clock
            fresh_ticks.add(raw); offsets.add(offset); errors.append(error)
            if closed>last_bar: fresh_bars.add(closed)
            if offset==7200: state,rule='UNCERTIFIED_CANDIDATE_REGIME','OFFSET_7200_REQUIRES_SEPARATE_CERTIFICATION'
            elif offset!=10800: state,rule='TIME_RULE_CONFLICT','OFFSET_NOT_CERTIFIED'
            elif abs(error)>5: state,rule='TIME_RULE_CONFLICT','CLOCK_SKEW_EXCEEDED'
            else: state='TIME_RULE_PASS'
        if state in ('TIME_RULE_CONFLICT','UNCERTIFIED_CANDIDATE_REGIME'): conflict=conflict or state
        checks['independent_sample:'+str(i)]=(all(row.get(k)==v for k,v in s.items())
            and row['raw_closed_bar_epoch']==closed and row['source_observation_age_seconds']==age
            and row['closed_bar_age_seconds']==bar_age and row['source_progress_observed']==advancing
            and row['source_identity_pass']==identity and row['monotonicity_pass']==monotonic
            and row['tick_fresh']==tick_ok and row['closed_bar_fresh']==bar_ok and row['fresh']==fresh
            and row['classification']==state and row['failed_rule']==rule
            and row['inferred_offset_seconds']==offset and row['normalized_clock_error_seconds']==error)
        checks['fresh_only_normalization:'+str(i)]=(row['normalized_tick_utc']==(
            datetime.fromtimestamp(raw-10800,timezone.utc).isoformat() if fresh else None)
            and row['normalized_bar_utc']==(datetime.fromtimestamp(closed-10800,timezone.utc).isoformat() if fresh else None))
    latest_fresh=bool(raw_samples and fresh)
    classification=conflict or ('TIME_RULE_PASS' if latest_fresh else 'NO_FRESH_MARKET_DATA')
    regime='FAIL' if conflict else 'PASS' if latest_fresh and len(fresh_ticks)>=5 and offsets=={10800} else 'PARTIAL'
    checks['summary']=(result['classification']==classification and result['current_regime_status']==regime
        and result['fresh_tick_sample_count']==len(fresh_ticks) and result['new_closed_m1_count']==len(fresh_bars)
        and result['fresh_source_data_available']==latest_fresh and result['source_identity_status']==('PASS' if identity_pass else 'FAIL')
        and result['observed_offset_seconds']==(next(iter(offsets)) if len(offsets)==1 else None)
        and result['normalized_clock_error_seconds_min']==(min(errors) if errors else None)
        and result['normalized_clock_error_seconds_max']==(max(errors) if errors else None))
    checks['no_admission']=result['evidence_admitted'] is False and result['resume_authorized'] is False
    return checks


def validate(run):
    root=run.parent.parent; capture=root/'future_holdout/gold_s4_v4'
    git=lambda *a:subprocess.check_output(['git',*a],cwd=root)
    m=load(run/'manifest.json'); spec=load(run/'execution_spec.json'); commit=m['source_commit']
    checks={'source_commit':commit==m['git_commit']==git('rev-parse','HEAD').decode().strip(),
        'remote_commit':commit==m['pre_run_remote_commit']==git('ls-remote','origin','refs/heads/main').decode().split()[0],
        'clean_source':m['git_dirty'] is False and m['pre_run_clean'] is True,
        'protected':all(digest(root/n)==h for n,h in spec['protected_sha256'].items()),
        'preserved':all(digest(root/n)==h for n,h in spec['preserved_sha256'].items()),
        'validator_snapshot':Path(__file__).read_bytes()==(run/'validator_script.py').read_bytes(),
        'source_inventory':set(spec['source_files'])=={e['source_path'] for e in m['input_snapshots']}}
    checks['fixed_thresholds']=(spec['fresh_tick_max_age_seconds']==120 and spec['fresh_closed_m1_max_age_seconds']==180
        and spec['max_runtime_clock_skew_seconds']==5 and spec['initial_sample_count']==5
        and spec['fresh_batch_max_wait_seconds']==130)
    for e in m['input_snapshots']:
        n=e['source_path']
        checks['source:'+n]=(digest(root/n)==e['sha256']==digest(run/e['path'])
                            and (root/n).read_bytes()==git('cat-file','blob',commit+':'+n))
    evidence=load(run/'original_evidence_inventory.json')['files']
    checks['original_evidence_bytes']=all(digest(root/e['source_path'])==e['sha256']==digest(run/e['path']) for e in evidence)
    required={(capture/n).relative_to(root).as_posix() for n in ('PAUSED_TIME_RULE.json','health/heartbeat.json','health/events.jsonl')}
    required|={p.relative_to(root).as_posix() for folder in ('pending','quarantine') for p in (capture/folder).iterdir()}
    checks['original_evidence_complete']=required=={e['source_path'] for e in evidence}
    checks['pause_preserved']=digest(run/'original_pause_snapshot.json')==digest(capture/'PAUSED_TIME_RULE.json')==spec['original_pause_expected_sha256']
    freeze=load(capture/'protocol/freeze.json'); activation=load(capture/'attestations/activation.json')
    checks['committed_freeze']=git('cat-file','blob',activation['protocol_freeze_commit']+':gold_future_locked_holdout_protocol_v4.json')==canonical(freeze)
    checks['frozen_policy']=(freeze['certified_offsets_seconds']==[10800] and freeze['uncertified_candidate_offsets_seconds']==[7200]
        and freeze['runtime_fail_closed'] is True and freeze['outcome_inspection_prohibited'] is True and freeze['prefix_approved'] is True)
    checks['frozen_code']=all(digest(root/n)==h for n,h in freeze['code_sha256'].items())
    inv=load(capture/'context_seed/gold_future_native_context_inventory_v1.json')
    checks['context_root']=inv['inventory_root_sha256']==freeze['context_inventory_root_sha256']==hashlib.sha256(canonical(inv['entries'])).hexdigest()
    checks['context_bytes']=all(digest(capture/'context_seed'/e['raw_path'])==e['raw_sha256']
        and (capture/'context_seed/manifests'/(e['timeframe']+'.json')).read_bytes()==canonical(e) for e in inv['entries'])
    current={p.relative_to(capture).as_posix():digest(p) for folder in ('snapshots','manifests') for p in (capture/folder).iterdir()}
    checks['chain_unchanged']=current==load(run/'chain_before.json')
    previous_hash=None; last_tick=None; last_bar=None; paths=set()
    for number,p in enumerate(sorted((capture/'manifests').iterdir())):
        item=load(p); snapshot=capture/item['snapshot_path']; payload=load(snapshot)
        valid=(p.name==f'{number:012d}.json' and item['sequence']==number and item['previous_manifest_sha256']==previous_hash
            and item['snapshot_path']==f'snapshots/{number:012d}.json' and digest(snapshot)==item['snapshot_sha256']
            and item['activation_sha256']==hashlib.sha256(canonical(activation)).hexdigest()
            and payload['native_source_confirmed'] is True and payload['resampled'] is False)
        for s in payload['samples']:
            valid &= abs(s['raw_epoch']-10800-s['observed_utc_epoch'])<=5 and (last_tick is None or s['raw_epoch']>last_tick)
            last_tick=s['raw_epoch']
        for row in payload['rows']:
            raw=row['RAW_SOURCE_EPOCH']
            valid &= raw in payload['returned_raw_epochs'][:-1] and raw+60<=last_tick and (last_bar is None or raw>last_bar)
            valid &= row['SOURCE_TIMESTAMP']==datetime.fromtimestamp(raw-10800,timezone.utc).isoformat()
            last_bar=raw
        checks['chain:'+str(number)]=bool(valid); paths.add(snapshot.name); previous_hash=digest(p)
    checks['no_orphan_snapshots']=paths=={p.name for p in (capture/'snapshots').iterdir()}
    observed=load(run/'fresh_samples.json'); diagnostic=load(run/'fresh_market_diagnostic.json')
    checks['observation_available']=observed['error'] is None
    checks.update(verify_samples(observed['samples'],diagnostic,last_tick,last_bar))
    checks['bounded_sampling']=5<=len(observed['samples'])<=11 and observed['samples'][-1]['system_utc_epoch']-observed['samples'][0]['system_utc_epoch']<=140
    prior=root/spec['prior_adjudication_run']; pd=load(prior/'pause_diagnostic.json'); root_cause=load(run/'stale_data_root_cause.json')
    stale=(pd['market_data_fresh'] is False and pd['current_frozen_check_failure']=='PAUSE_AND_QUARANTINE')
    # Reproduce the old rounded-offset predicate independently, without calling it.
    old_samples=load(prior/'observations.json')['samples']
    checks['prior_stale_reproduction']=(len({s['raw_epoch'] for s in old_samples})==1
        and all(math.floor((s['raw_epoch']-s['observed_utc_epoch'])/3600+.5)*3600 not in (7200,10800) for s in old_samples))
    fresh_valid=diagnostic['current_regime_status']=='PASS' and diagnostic['new_closed_m1_count']>=2 and diagnostic['source_identity_status']=='PASS'
    expected_root='STALE_SOURCE_OBSERVATION_MISCLASSIFIED_AS_TIME_RULE_CONFLICT' if stale and fresh_valid else 'NOT_PROVEN'
    checks['root_cause']=root_cause['root_cause']==expected_root and root_cause['fresh_restores_10800']==fresh_valid and root_cause['prior_finalized_sha256']==digest(prior/'FINALIZED.json')
    state=diagnostic['classification']
    decision='WAIT_FOR_FRESH_MARKET_DATA' if state=='NO_FRESH_MARKET_DATA' else 'PAUSE_REMAINS_VALID' if state in ('TIME_RULE_CONFLICT','UNCERTIFIED_CANDIDATE_REGIME') else 'FAIL'
    resume=load(run/'resume_decision.json'); policy=load(run/'runtime_freshness_policy.json')
    checks['no_premature_resume']=resume['decision']==decision and resume['resume_authorized'] is False and not (run/'resume_receipt.json').exists()
    checks['runtime_unmodified']=policy['amendment_applied'] is False and policy['status']=='PARTIAL'
    health=load(run/'capture_health_after_resume.json')
    checks['honest_health']=health['collector_status']=='PAUSED' and health['current_process_running'] is False and health['process_lock_present'] is False and health['heartbeat_status']=='FAIL'
    feature=load(run/'feature_eligibility_after_resume.json'); evaluation=load(run/'evaluation_start_decision.json')
    checks['feature_not_rerun']=feature['rerun'] is False and feature['earliest_feature_complete_timestamp'] is None and digest(root/feature['prior_result_path'])==feature['prior_result_sha256']
    checks['boundary_unchanged']=evaluation['holdout_start']=='2026-09-25T19:24:00+00:00' and evaluation['holdout_evaluation_start'] is None and evaluation['original_boundary_sha256']==digest(root/'gold_future_holdout_boundary_v4.json')
    metrics=load(run/'metrics.json')
    checks['honest_verdict']=metrics['formal_run_status']==('FAIL' if diagnostic['current_regime_status']=='FAIL' else 'PARTIAL')
    checks['no_new_evidence']=metrics['new_sealed_snapshot_count']==0 and metrics['snapshots_before']==metrics['snapshots_after']==len(paths)
    checks['no_strategy_claims']=all(metrics[k] is False for k in ('lookahead_detected','strategy_outcome_inspected','model_loaded_for_holdout','model_trained_for_holdout','strategy_executed','production_changed','production_promoted'))
    pending=['run_gold_future_fresh_market_resume_certification_v1.py']; seen=set()
    while pending:
        n=pending.pop()
        if n in seen: continue
        seen.add(n); tree=ast.parse((root/n).read_bytes())
        modules=[a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]+[x.module or '' for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        calls={x.func.attr if isinstance(x.func,ast.Attribute) else x.func.id if isinstance(x.func,ast.Name) else '' for x in ast.walk(tree) if isinstance(x,ast.Call)}
        checks['static_safety:'+n]=not {'xgboost','lightgbm','catboost','sklearn','joblib','pickle','gemini'}.intersection(m.split('.')[0] for m in modules) and not {'fit','predict','predict_proba','load_model','order_send','order_check'}.intersection(calls)
        pending.extend(mod+'.py' for mod in modules if (root/(mod+'.py')).is_file())
    regressions=load(run/'regression_results.json')
    checks['regressions']=regressions['overall']=='PASS' and regressions['test_count']>=21 and all(regressions['checks'].values())
    allowed='?? '+run.relative_to(root).as_posix()+'/'
    checks['only_formal_run_untracked']=all(e.startswith(allowed) for e in git('status','--porcelain','-z').decode('utf-8').split('\0') if e)
    return checks


def main():
    run=Path(sys.argv[1]).resolve()
    with (run/'validator_attempt.json').open('x',encoding='utf-8') as f:
        json.dump({'started_at_utc':datetime.now(timezone.utc).isoformat(),'rule':'Do not retry validation'},f)
    try:
        checks=validate(run); failed=[k for k,v in checks.items() if not v]
        result={'overall':'FAIL' if failed else 'PASS','failed_check_names':failed,'checks':checks}
    except Exception as error:
        result={'overall':'FAIL','failed_check_names':['validator_exception'],'exception_type':type(error).__name__}
    (run/'validator.json').write_bytes(canonical(result))
    (run/'validator.md').write_text('# Independent fresh-market verifier\n\n'+result['overall']+'\n\n'+json.dumps(result['failed_check_names'])+'\n',encoding='utf-8')
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(0 if result['overall']=='PASS' else 1)


if __name__=='__main__':
    main()
