"""Run external attestation and causal-context readiness from clean committed source."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
from run_gold_future_capture_protocol_adjudication_v1 import git, sha, write

ROOT = Path(__file__).resolve().parent
SPEC = 'execution_spec_gold_future_capture_external_attestation_and_prefix_freeze_v1.json'
from gold_future_capture_external_attestation_and_prefix_freeze_v1 import evidence_review, decisions


def protection(run, label):
    spec=history.read_json(ROOT/SPEC)
    actual={name:sha(ROOT/name) for name in spec['protected_sha256']}
    with (run/'protection_checks.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps({'stage':label,'sha256':actual,'pass':actual==spec['protected_sha256']})+'\n')
    if actual!=spec['protected_sha256']:
        history.write_json(run/'metrics.json',{'formal_run_status':'FAIL','production_changed':True})
        raise RuntimeError('Protected production changed; STOP; no silent restore')


def command(run, label, args):
    protection(run,label+':before')
    argv = [sys.executable, '-B', *args]
    with (run/(label+'_stdout.txt')).open('xb') as out, (run/(label+'_stderr.txt')).open('xb') as err:
        result = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, timeout=60, check=False)
    with (run/'commands.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'label':label,'argv':argv,'exit_code':result.returncode})+'\n')
    protection(run,label+':after')
    if result.returncode:
        raise RuntimeError('Preserve failed command; do not retry: '+label)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.parse_args()
    spec=history.read_json(ROOT/SPEC)
    if git('status','--porcelain','-z'):
        raise RuntimeError('Clean committed source required')
    commit=git('rev-parse','HEAD').decode().strip()
    remote=git('ls-remote','origin','refs/heads/main').decode().split()[0]
    if commit!=remote or git('branch','--show-current').decode().strip()!='main':
        raise RuntimeError('Pushed main required')
    for name,expected in spec['preserved_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise RuntimeError('Preserved input changed: '+name)
    prior=ROOT/spec['prior_run']
    if sha(prior/'FINALIZED.json')!=spec['prior_finalized_sha256'] or any(
        sha(prior/name)!=expected for name,expected in history.read_json(prior/'FINALIZED.json')['file_sha256'].items()):
        raise RuntimeError('Prior finalized evidence changed')
    for name in spec['source_files']:
        if (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=git('cat-file','blob',commit+':'+name).replace(b'\r\n',b'\n'):
            raise RuntimeError('Uncommitted input')
    run=history.create_run(spec['experiment_name'],Path(__file__),
        subprocess.list2cmdline([sys.executable,'-B',str(Path(__file__)),'--execute']),
        arguments=['--execute'],seed_note='Deterministic synthetic fixtures; no models or outcome tuning')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(),flush=True)
    protection(run,'archive_create')
    m=history.read_json(run/'manifest.json')
    m.update(source_commit=commit,pre_run_remote_commit=remote,pre_run_clean=True,pre_run_git_status='',
        git_branch='main',git_upstream='origin/main',input_snapshots=[],
        protected_sha256_before={n:sha(ROOT/n) for n in spec['protected_sha256']},
        result_commit_binding='Commit containing immutable FINALIZED.json; no self-referential commit hash')
    for name in spec['source_files']:
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
        m['input_snapshots'].append({'source_path':name,'path':dest.relative_to(run).as_posix(),
            'sha256':sha(dest),'retention_status':'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/SPEC,run/'execution_spec.json')
    shutil.copyfile(ROOT/'validate_gold_future_capture_external_attestation_and_prefix_freeze_v1.py',run/'validator_script.py')
    history.write_json(run/'manifest.json',m)
    command(run,'self_test',['gold_future_capture_external_attestation_and_prefix_freeze_v1.py'])
    command(run,'diagnostic',['gold_mt5_timezone_transition_diagnostic_v2.py','--output',str(run/'timezone_diagnostic.json')])
    protection(run,'prefix_review:before')
    shutil.copyfile(ROOT/spec['prior_run']/'prefix_equivalence_tests.json',run/'prefix_characterization.json')
    shutil.copyfile(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json',run/'prefix_protocol.json')
    protection(run,'prefix_review:after')
    command(run,'certification',['gold_future_capture_certification_v2.py','--output',str(run/'capture_certification_report.json')])
    report=history.read_json(run/'capture_certification_report.json')
    diagnostic=history.read_json(run/'timezone_diagnostic.json')
    prefix,verdict=decisions(report,diagnostic)
    write(run/'prefix_decision.json',prefix)
    write(run/'collector_static_review.json',report['collector_static_review'])
    for kind in ('source','timezone','activation'):
        name='gold_future_capture_'+kind+('_attestation' if kind!='activation' else '')+'_v2.json'
        shutil.copyfile(ROOT/name,run/(kind+'_attestation_snapshot.json'))
    if report['readiness']['status']=='READY_FOR_CAPTURE_ACTIVATION':
        raise RuntimeError('Unexpected complete inputs: preserve run for independent freeze review')
    timezone={'status':'UNRESOLVED','timezone_policy':'UNRESOLVED','certified_coverage_start_utc':None,
        'certified_coverage_end_utc':None,'broker_offset_intervals':[],'recurrence_rule':None,
        'diagnostic_classification':diagnostic['classification'],
        'evidence':evidence_review(),
        'answers':{'server_clock':'Exact server rule unconfirmed; XM generic Forex FAQ says GMT+2 and DST may apply',
            'dst_recurrence':'UNRESOLVED','api_encoding':'Generic UTC documentation conflicts with prior local-epoch observation',
            'daily_rollover':'UNRESOLVED','weekend_schedule':'UNRESOLVED','holiday_calendar':'No current exact-source schedule retained',
            'bar_timestamp':'BAR_OPEN documented for API timeframes; exact source encoding remains unresolved'},
        'decision':'Do not fabricate intervals or complete attestations from supporting diagnostics'}
    write(run/'timezone_decision.json',timezone)
    write(run/'timezone_research.json',{'evidence':evidence_review(),'support_request':'XM_TIMEZONE_ATTESTATION_REQUEST_V1.md','support_request_sent':False,'exact_source_authority_resolved':False})
    metrics={'source_status':'PASS','self_test_status':'PASS','source_attestation_status':report['source_attestation_status'],
        'timezone_attestation_status':'UNRESOLVED','timezone_policy':'UNRESOLVED',
        'timezone_diagnostic_status':diagnostic['classification'],
        'certified_timezone_coverage_start':None,'certified_timezone_coverage_end':None,
        'higher_timeframe_policy':report['higher_timeframe_policy'],'prefix_policy':prefix['policy'],
        'prefix_certification_status':prefix['status'],'prefix_context_role':prefix['context_role'],
        'prefix_historically_equivalent':False,'prefix_outcome_tuning':False,'collector_static_status':report['collector_static_review']['static_status'],
        'capture_certification_status':report['readiness']['status'],'formal_run_status':verdict,
        'protocol_frozen':False,'capture_activated':False,'capture_start':None,'holdout_started':False,
        'holdout_start':None,'holdout_evaluation_start':None,'strategy_outcome_inspected':False,
        'model_loaded_for_holdout':False,'model_trained_for_holdout':False,'production_changed':False,'production_promoted':False}
    history.write_json(run/'metrics.json',metrics)
    m.update(formal_run_status=verdict,git_status_after_execution=git('status','--porcelain','-z').decode('utf-8'),
        protected_sha256_after={n:sha(ROOT/n) for n in spec['protected_sha256']},
        git_dirty_reason='Clean pushed source; only this new run is untracked during execution')
    if m['protected_sha256_before']!=m['protected_sha256_after'] or m['protected_sha256_after']!=spec['protected_sha256']:
        raise RuntimeError('Production hash mismatch')
    d=m['data'];d.update(symbols=['GOLD#'],data_sources=['Bounded timestamp-only MT5 diagnostics','Official document reviews','Synthetic feature-state fixture'],
        source_files=m['input_snapshots'],timezone='UNRESOLVED',raw_snapshot_retained=False,
        reproducibility_claim='Timestamp projections retained; no raw OHLC retained or certified full prefix',
        purge_details='No strategy dataset',embargo_details='No outcomes inspected')
    for key in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'):
        d[key]='NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for key in ('train_rows','validation_rows','test_rows'):d[key]=0
    samples=diagnostic['samples']
    d['mt5_fetch'].update(used=True,terminal_path=r'D:\XM2\terminal64.exe',terminal_info='Read-only connectivity check; no terminal object persisted',
        broker_info='Expected XMGlobal-MT5 6 demo from prior evidence; current account identity not queried',
        fetch_start_utc=min((s['query_start_utc'] for s in samples),default='NO_RETURNED_SAMPLES'),
        fetch_end_utc=max((s['query_end_utc'] for s in samples),default='NO_RETURNED_SAMPLES'),
        retrieved_at_utc=diagnostic['generated_at_utc'],returned_rows=sum(len(s['raw_epochs']) for s in samples))
    m['model']['not_applicable_reason']='No models loaded or trained'
    m['search']['not_applicable_reason']='No strategy search'
    m['registry'].update({k:'N/A: infrastructure readiness' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='GOLD final readiness v1 PARTIAL',
        selected_configuration='Final readiness '+verdict+'; no freeze, activation, holdout or production change/promotion',validator_result='PENDING')
    history.write_json(run/'manifest.json',m)
    narrative=('# GOLD external attestation and prefix freeze readiness\n\nFormal result: '+verdict
        +'\n\nMethodology review supports pre-boundary historical context in principle; no 21-month post-activation waiting rule is imposed. '
        +'Prefix approval remains blocked by source/timezone compatibility and absence of a sealed context inventory. '
        +'historically_equivalent=false; outcome_tuning=false; context_role=CAUSAL_CONTEXT_ONLY.\n'
        +'\nExact-source DST/epoch/session evidence remains insufficient. Support request prepared, NOT SENT. '
        +'Official generic guidance is not exact-server certification. Diagnostic='+diagnostic['classification']+'.\n'
        +'\nPrior prefix characterization retained byte-for-byte; warmup4096 was identical-prefix control, not convergence proof. '
        +'No model, strategy outcome, production change or promotion. No freeze, activation, capture or holdout boundary.\n')
    (run/'report.md').write_text(narrative,encoding='utf-8')
    (run/'findings.md').write_text(narrative+'\nAll readiness blockers retained in capture_certification_report.json.\n',encoding='utf-8')
    shutil.copyfile(run/'certification_stdout.txt',run/'stdout.txt')
    shutil.copyfile(run/'certification_stderr.txt',run/'stderr.txt')
    shutil.copyfile(run/'stdout.txt',run/'stdout.log')
    protection(run,'report_complete')
    print('FORMAL_RUN_STATUS='+verdict)


if __name__=='__main__':main()
