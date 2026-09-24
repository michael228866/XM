"""Independent external-attestation and causal-context archive verifier; no execution imports."""
import argparse
import ast
import calendar
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def legacy_time(epoch):
    raw=datetime.fromtimestamp(epoch,timezone.utc)
    candidates=[]
    for offset in (2,3):
        candidate=raw-timedelta(hours=offset)
        bounds=[]
        for month in (3,10):
            last=datetime(candidate.year,month,calendar.monthrange(candidate.year,month)[1],1,tzinfo=timezone.utc)
            bounds.append(last-timedelta(days=(last.weekday()+1)%7))
        if offset==(3 if bounds[0]<=candidate<bounds[1] else 2):candidates.append(candidate)
    if len(candidates)!=1:raise ValueError('Ambiguous hypothesis')
    return candidates[0]


def validate(run):
    root=run.parent.parent
    git=lambda *args:subprocess.check_output(['git',*args],cwd=root)
    m=load(run/'manifest.json');spec=load(run/'execution_spec.json');s=run/'source'
    commit=m['source_commit']
    checks={'source_commit':commit==m['git_commit']==git('rev-parse','HEAD').decode().strip(),
        'remote_commit':commit==m['pre_run_remote_commit']==git('ls-remote','origin','refs/heads/main').decode().split()[0],
        'clean_provenance':m['git_dirty'] is False and m['pre_run_clean'] is True and m['pre_run_git_status']=='',
        'source_inventory':set(spec['source_files'])=={v['source_path'] for v in m['input_snapshots']}}
    allowed='?? '+run.relative_to(root).as_posix()+'/'
    checks['source_remains_clean']=all(entry.startswith(allowed) for status in (m['git_status_after_execution'],git('status','--porcelain','-z').decode('utf-8')) for entry in status.split('\0') if entry)
    for item in m['input_snapshots']:
        p=(run/item['path']).resolve()
        if not p.is_relative_to(s.resolve()):raise ValueError('Snapshot escape')
        blob=git('cat-file','blob',commit+':'+item['source_path'])
        checks['source:'+item['source_path']]=sha(p)==item['sha256']==sha(root/item['source_path']) and p.read_bytes().replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n')
    checks['preserved_inputs']=all(sha(root/n)==h for n,h in spec['preserved_sha256'].items())
    checks['prior_runs_unchanged']=not git('diff',spec['base_commit'],'--','training_runs')
    prior_run=root/spec['prior_run']
    checks['prior_seal']=sha(prior_run/'FINALIZED.json')==spec['prior_finalized_sha256'] and all(
        sha(prior_run/name)==expected for name,expected in load(prior_run/'FINALIZED.json')['file_sha256'].items())
    checks['protected_production']=m['protected_sha256_before']==m['protected_sha256_after']==spec['protected_sha256']=={n:sha(root/n) for n in spec['protected_sha256']}
    checks['script_snapshot']=sha(run/'training_script.py')==m['training_script_sha256']==sha(s/'run_gold_future_capture_external_attestation_and_prefix_freeze_v1.py')
    checks['validator_snapshot']=Path(__file__).read_bytes()==(run/'validator_script.py').read_bytes()==(s/'validate_gold_future_capture_external_attestation_and_prefix_freeze_v1.py').read_bytes()
    source=load(run/'source_attestation_snapshot.json');tz=load(run/'timezone_attestation_snapshot.json');activation=load(run/'activation_attestation_snapshot.json')
    for kind in ('source','timezone','activation'):
        name='gold_future_capture_'+kind+('_attestation' if kind!='activation' else '')+'_v2.json'
        checks['attestation_snapshot:'+kind]=(run/(kind+'_attestation_snapshot.json')).read_bytes()==(s/name).read_bytes()
    checks['identity']=source['source_id']=='XMGlobal-MT5-6_GOLD' and source['symbol']==source['symbol_exact_case']=='GOLD#' and source['broker_server_name']=='XMGlobal-MT5 6' and source['broker_name']=='XM Global Limited'
    prior=load(s/'research_evidence/gold_future_capture_protocol_adjudication_v1/prior_diagnostic_timestamp.json')
    checks['demo_evidence']=source['source_account_environment']==prior['source_account_environment']=='demo'
    checks['attestation_links']=tz['source_attestation_sha256']==canonical(source)==activation['source_attestation_sha256'] and activation['timezone_attestation_sha256']==canonical(tz)
    evidence=load(s/'research_evidence/gold_future_capture_external_attestation_and_prefix_freeze_v1/evidence_index.json')
    for item in evidence:
        checks['evidence:'+item['local_path']]=sha(s/item['local_path'])==item['sha256'] and item['authority'] in {'BROKER_DOCUMENT','FEED_DOCUMENT','SIGNED_ATTESTATION','INDEPENDENT_REVIEW'} and all(item.get(k) for k in ('source_url','retrieved_at_utc','reviewed_by','reviewed_at_utc','supported_claims','unsupported_claims','publisher','title','limitations'))
    adjudication=load(run/'timezone_decision.json')
    checks['timezone_no_overcertification']=tz['template'] is True and tz['timezone_status']=='UNRESOLVED' and tz['broker_offset_intervals']==[] and tz['recurrence_rule'] is None and tz['certified_coverage_start_utc'] is None and tz['certified_coverage_end_utc'] is None and adjudication['evidence']==evidence and adjudication['broker_offset_intervals']==[] and adjudication['recurrence_rule'] is None
    diagnostic=load(run/'timezone_diagnostic.json')
    checks['diagnostic_identity_limit']=diagnostic['source_identity_currently_verified'] is False and diagnostic['symbol']=='GOLD#'
    for sample in diagnostic['samples']:
        epochs=sample['raw_epochs'];spacing=[b-a for a,b in zip(epochs,epochs[1:])]
        start=datetime.fromisoformat(sample['query_start_utc']);end=datetime.fromisoformat(sample['query_end_utc'])
        checks['time_sample:'+sample['label']]=len(epochs)<=6 and (end-start).total_seconds()==300 and all(start.timestamp()<=v<=end.timestamp() for v in epochs) and sample['interpreted_as_utc']==[datetime.fromtimestamp(t,timezone.utc).isoformat() for t in epochs] and sample['spacing_seconds']==spacing and sample['monotonic']==all(v>0 for v in spacing) and sample['hypothesis_converted']==[legacy_time(v).isoformat() for v in epochs] and sample['offset_inference'] is None and set(sample)=={'label','query_start_utc','query_end_utc','raw_epochs','interpreted_as_utc','hypothesis_converted','spacing_seconds','monotonic','offset_inference','reason','requested_utc','sample_source','classification','timestamp_rows'}
        checks['time_projection:'+sample['label']]=sample['requested_utc']==[sample['query_start_utc'],sample['query_end_utc']] and sample['classification']=='INSUFFICIENT_DATA' and sample['sample_source']=='GOLD# M1 copy_rates_range' and sample['timestamp_rows']==[{'raw_epoch':t,'raw_epoch_interpreted_direct_utc':datetime.fromtimestamp(t,timezone.utc).isoformat(),'candidate_server_offset_seconds':int(t-legacy_time(t).timestamp()),'legacy_conversion_utc':legacy_time(t).isoformat(),'difference_from_independent_anchor_if_available':None} for t in epochs]
    checks['october_windows_present']=('error_type' in diagnostic) or {'October_before','October_after','US_autumn_before','US_autumn_after'}.issubset({sample['label'] for sample in diagnostic['samples']})
    current=diagnostic['current']
    if current:
        raw=current['raw_epoch'];observed=datetime.fromisoformat(current['observed_utc']).timestamp();converted=legacy_time(raw)
        classification='SUPPORTS_BROKER_SERVER_RULE' if abs(converted.timestamp()-observed)<=180 else 'CONTRADICTS_BROKER_SERVER_RULE' if abs(raw-observed)<=180 else 'INSUFFICIENT_DATA'
        checks['current_time']=current['classification']==classification and current['hypothesis_converted']==converted.isoformat() and abs(current['raw_minus_system_seconds']-(raw-observed))<1e-6 and abs(current['hypothesis_minus_system_seconds']-(converted.timestamp()-observed))<1e-6
    expected='UNRESOLVED' if 'error_type' in diagnostic else 'CONTRADICTS_BROKER_SERVER_RULE' if current and current['classification']=='CONTRADICTS_BROKER_SERVER_RULE' else 'INSUFFICIENT_DATA'
    checks['diagnostic_classification']=diagnostic['classification']==expected==adjudication['diagnostic_classification']
    p=load(run/'prefix_decision.json');doc=load(run/'prefix_protocol.json')
    tests=load(run/'prefix_characterization.json')
    checks['prefix_protocol_snapshot']=(run/'prefix_protocol.json').read_bytes()==(s/'gold_recursive_prefix_causal_context_protocol_v1.json').read_bytes()
    checks['prefix_characterization_identity']=(run/'prefix_characterization.json').read_bytes()==(root/spec['prior_run']/'prefix_equivalence_tests.json').read_bytes()
    fn=next(n for n in ast.parse((s/'drl_trading_v2.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='add_indicators')
    feature_hash=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
    checks['prefix_hashes']=doc['pipeline_sha256']==sha(s/'drl_trading_v2.py') and doc['feature_implementation_sha256']==feature_hash and p['context_protocol_sha256']==canonical(doc)
    checks['prefix_honesty']=p['policy']=='UNRESOLVED' and p['status']=='PARTIAL' and p['approved'] is False and doc['approved'] is False and doc['source_compatibility_certified'] is False and doc['sealed_context_sha256'] is None
    checks['causal_context_role']=doc['policy']=='PREDECLARED_CAUSAL_REINITIALIZATION' and doc['context_role']==p['context_role']=='CAUSAL_CONTEXT_ONLY' and doc['context_only'] is True and doc['holdout_evidence'] is False and doc['historically_equivalent'] is False and doc['outcome_tuning'] is False and doc['recursive_features']==['MACD_HIST']
    checks['preboundary_context_allowed']=doc['historical_context_allowed_in_principle'] is True and p['requires_post_activation_monthly_accumulation'] is False
    checks['fixed_context_envelope']=doc['context_start']=='2024-01-01T00:00:00+00:00' and doc['context_end_exclusive']=='2026-09-24T00:00:00+00:00' and datetime.fromisoformat(doc['context_start'])<datetime.fromisoformat(doc['context_end_exclusive'])<datetime.fromisoformat(m['started_at_utc'])
    frames=set(load(s/'gold_future_capture_manifest_schema_v2.json')['properties']['timeframe']['enum'])
    checks['native_context_mechanics']=set(doc['warmup_bars_by_timeframe'])==frames and all(v==(4096 if k=='M1' else 21) for k,v in doc['warmup_bars_by_timeframe'].items())
    init=doc['initialization_semantics']
    checks['initialization']=init['adjust'] is False and init['min_periods']==0 and init['ignore_na'] is False and init['pandas_version']=='3.0.5' and init['dtype']=='float64' and init['model_input_dtype']=='float32'
    checks['characterization_not_exact']=tests['synthetic_only'] is True and tests['historical_equivalence_proven'] is False and any(not row['exact_match'] for row in tests['tests'] if row['candidate_warmup']==254 and row['feature_name']=='MACD_HIST')
    research=load(run/'timezone_research.json')
    checks['support_unsent']=research['support_request_sent'] is False and 'NOT_SENT' in (s/'XM_TIMEZONE_ATTESTATION_REQUEST_V1.md').read_text(encoding='utf-8') and research['evidence']==evidence
    protection=[json.loads(line) for line in (run/'protection_checks.jsonl').read_text(encoding='utf-8').splitlines()]
    checks['production_each_stage']=all(row['pass'] is True and row['sha256']==spec['protected_sha256'] for row in protection) and {row['stage'] for row in protection}=={'archive_create','self_test:before','self_test:after','diagnostic:before','diagnostic:after','prefix_review:before','prefix_review:after','certification:before','certification:after','report_complete'}
    protocol=load(s/'gold_future_capture_protocol_v2.json');report=load(run/'capture_certification_report.json');metrics=load(run/'metrics.json')
    collector=ast.parse((s/'gold_future_capture_collector_v2.py').read_text(encoding='utf-8'))
    checks['collector_report_snapshot']=load(run/'collector_static_review.json')==report['collector_static_review']
    checks['collector_static']=hashlib.sha256(ast.dump(collector,include_attributes=False).encode()).hexdigest()==protocol['approved_collector_ast_sha256'] and report['collector_static_review']['static_status']=='STATICALLY_CONFORMANT' and report['collector_static_review']['collector_sha256']==sha(s/'gold_future_capture_collector_v2.py')
    checks['native_and_tip']=metrics['higher_timeframe_policy']==protocol['higher_timeframe_policy']=='NATIVE_20TF_REQUIRED' and protocol['chain_tip_policy']=='NON_AUTHORITATIVE_REBUILDABLE_INDEX'
    checks['readiness']=metrics['capture_certification_status']==report['readiness']['status']=='NOT_READY_MULTIPLE_BLOCKERS' and {b['status'] for b in report['readiness']['blockers']}=={'NOT_READY_SOURCE_BINDING','NOT_READY_TIMEZONE','NOT_READY_PREFIX_POLICY','NOT_READY_CAPTURE_INTEGRITY','NOT_READY_PROTOCOL_FREEZE'}
    checks['verdict']=metrics['formal_run_status']==m['formal_run_status']=='PARTIAL'
    checks['no_premature_start']=all(metrics[k] is False for k in ('protocol_frozen','capture_activated','holdout_started','strategy_outcome_inspected','model_loaded_for_holdout','model_trained_for_holdout','production_changed','production_promoted')) and all(metrics[k] is None for k in ('capture_start','holdout_start','holdout_evaluation_start'))
    allowed_calls={'initialize','shutdown','terminal_info','symbol_info','symbol_info_tick','copy_rates_range'}
    diagnostic_tree=ast.parse((s/'gold_mt5_timezone_transition_diagnostic_v2.py').read_text(encoding='utf-8'))
    mt5_calls={n.func.attr for n in ast.walk(diagnostic_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='mt5'}
    checks['mt5_allowlist']=mt5_calls==allowed_calls
    for name in ('gold_future_capture_external_attestation_and_prefix_freeze_v1.py','gold_mt5_timezone_transition_diagnostic_v2.py','gold_recursive_prefix_final_readiness_v1.py','run_gold_future_capture_external_attestation_and_prefix_freeze_v1.py'):
        t=ast.parse((s/name).read_text(encoding='utf-8'))
        imports={n.module for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}
        calls={n.func.attr for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        checks['no_execution:'+name]=not imports.intersection({'xgboost','torch','sklearn','gemini','drl_trading_v2'}) and not calls.intersection({'fit','predict','predict_proba','load_model','order_send','account_info'})
    commands=[json.loads(line) for line in (run/'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    checks['commands']=[c['label'] for c in commands]==['self_test','diagnostic','certification'] and all(c['exit_code']==0 for c in commands)
    selftest=load(run/'self_test_stdout.txt')
    checks['self_tests']=selftest['overall']=='PASS' and len(selftest['checks'])>=100 and selftest['synthetic_only'] is True
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,nargs='?');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        assert legacy_time(1893499200).utcoffset().total_seconds()==0
        assert canonical({'a':1,'b':2})==canonical({'b':2,'a':1})
        print('SELF_TEST_PASS')
    else:
        if args.run is None:parser.error('Run required')
        run=args.run.resolve()
        if (run/'FINALIZED.json').exists():raise FileExistsError('Finalized run immutable')
        with (run/'validator_attempt.json').open('x',encoding='utf-8') as f:
            json.dump({'one_shot':True,'started_at_utc':datetime.now(timezone.utc).isoformat()},f)
        try:checks=validate(run)
        except Exception as error:checks={'exception:'+type(error).__name__:False}
        failed=[k for k,v in checks.items() if not v]
        result={'overall':'FAIL' if failed else 'PASS','failed_check_names':failed,'checks':checks}
        with (run/'validator.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
        with (run/'validator.md').open('x',encoding='utf-8') as f:f.write('# Independent external attestation and prefix readiness validation\n\n'+result['overall']+'\nResearch readiness verdict remains separate. No production promotion.\n')
        print('overall='+result['overall']);print('failed_check_names='+json.dumps(failed))
        raise SystemExit(bool(failed))
