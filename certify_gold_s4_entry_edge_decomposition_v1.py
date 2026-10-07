"""Static/synthetic certification only. Never launches the historical workflow."""
import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import training_run_history as history
from gold_s4_entry_edge_decomposition_v1_support import (
    ROOT,CONFIG,EXPERIMENT,FLAGS,read,write,sha,require,configuration,check_binding_metadata,sources,
)

SLUG=EXPERIMENT+'_infrastructure'
SOURCES=[EXPERIMENT+'.py',CONFIG,'gold_s4_entry_edge_bucket_spec_v1.json',EXPERIMENT+'_support.py',
    EXPERIMENT+'_launcher.py','gold_s4_entry_edge_feature_inventory_v1.py','gold_s4_entry_edge_feature_inventory_v1.json',
    'gold_s4_entry_edge_reference_v1.json','validate_'+EXPERIMENT+'_run.py','manual_entry_edge_decomposition_policy_v1.json',
    'README_GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1.md','test_'+EXPERIMENT+'.py','certify_'+EXPERIMENT+'.py',
    'manual_training_launcher_v1.py','manual_training_policy_v1.json','gold_manual_training_workflow_v1.py',
    'gold_manual_s4_train_validate_v1.py','gold_manual_s4_training_data_v1.py','training_holdout_guard_v1.py',
    'training_run_history.py','RUN_TRAINING.bat','CHECK_STATUS.bat']
GROUPS=['reference_binding_review','feature_inventory_review','bucket_spec_review','causality_test',
    'sample_guard_test','univariate_logic_test','bivariate_logic_test','fold_stability_test','stress_metric_test',
    'multiple_testing_test','validator_fixture_test','event_chain_test','manual_execution_guard_test',
    'holdout_guard_test','production_protection_test','bat_static_test','reference_reproduction_spec_test']


def source_files():
    return sorted(set(SOURCES)|set(read(ROOT/CONFIG)['source_bindings']))


def bat_text(status=False):
    return ('@echo off\nsetlocal\nchcp 65001 >nul\nset "PYTHONUTF8=1"\ncd /d "%~dp0"\n'
        'rem GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1: USER only; no training or entry filtering.\n'+
        ('' if status else 'set "XM_USER_TRAINING_BAT=RUN_TRAINING_V1"\n')+
        'if not exist "%~dp0.venv\\Scripts\\python.exe" (\n  echo [FAIL] Repository Python is missing.\n  goto done\n)\n'
        '"%~dp0.venv\\Scripts\\python.exe" -B "%~dp0gold_s4_entry_edge_decomposition_v1_launcher.py"'+
        (' --status' if status else '')+'\n:done\necho Press any key to close.\npause >nul\nendlocal\n')


def clean_pushed():
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
    commit=git('rev-parse','HEAD')
    require(not git('status','--porcelain') and git('branch','--show-current')=='main' and
        git('ls-remote','origin','refs/heads/main').split()[0]==commit,'Clean pushed main required')
    return commit


def fill_manifest(m,ref,infrastructure=True):
    m['data'].update(symbols=['GOLD#'],data_sources=['Synthetic fixtures and frozen metadata' if infrastructure else 'Sealed accepted trades and decision-time feature/score snapshots'],
        source_files=([dict(path='source/'+CONFIG,sha256=sha(ROOT/CONFIG),retention_status='stored_in_run_directory')] if infrastructure else
                      [dict(path=s['path'],sha256=s['sha256'],retention_status='existing_immutable_local_archive; Git seal; not remote raw backup') for s in sources(ref)]),
        timezone='UTC',raw_snapshot_retained=True,reproducibility_claim='Synthetic only' if infrastructure else ref['retention'],
        purge_details='No fitting or selection; frozen original entries',embargo_details='No new OOS or forward-evidence claim')
    for key in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'):
        m['data'][key]='NOT_APPLICABLE_INFRASTRUCTURE_ONLY' if infrastructure else 'NOT_APPLICABLE_NO_TRAINING_OR_UNTOUCHED_TEST'
    if not infrastructure:
        m['data'].update(data_start_utc='2018-01-01',data_end_utc='2025-01-01',validation_start_utc='2018-01-01',validation_end_utc='2025-01-01')
    m['data'].update(train_rows=0,validation_rows=0 if infrastructure else 787,test_rows=0)
    m['data']['mt5_fetch']['not_applicable_reason']='No broker fetch'
    m['model'].update(trained=False,not_applicable_reason='No model load, fitting, prediction or recalibration')
    m['search'].update(performed=False,not_applicable_reason='Descriptive fixed partitions only; all cells retained in univariate/bivariate tables and multiple_testing_inventory.json; no strategy selection')
    m['registry'].update(dict.fromkeys(history.REGISTRY_FIELDS,'N/A: infrastructure' if infrastructure else 'Descriptive research only'))
    m['registry'].update(parent_or_incumbent=ref['source_run_id'],selected_configuration='Predeclared entry decomposition; no filter selected',validator_result='PENDING')
    if not infrastructure:
        p=ref['metrics']
        m['registry'].update(trades_per_day=p['trades_per_day'],realized_win_rate=p['realized_win_rate'],pf=p['profit_factor'],mean_r=p['mean_r'],pnl=p['pnl_r'],max_dd=p['max_drawdown_r'])


def execute():
    commit=clean_pushed()
    config,ref,spec,inventory=configuration()
    check_binding_metadata(ref)
    run=history.create_run(SLUG,Path(__file__),'.venv\\Scripts\\python.exe -B '+Path(__file__).name+' --execute',arguments=['--execute'],seed_note='Deterministic synthetic fixtures; no bootstrap')
    print('INFRASTRUCTURE_RUN='+run.name,flush=True)
    (run/'source').mkdir()
    provenance=[]
    for name in source_files():
        raw=(ROOT/name).read_bytes()
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        require(raw==blob or raw.replace(b'\r\n',b'\n')==blob,'Committed source representation: '+name)
        shutil.copyfile(ROOT/name,run/'source'/name)
        provenance.append(dict(path=name,raw_file_sha256=sha(ROOT/name),git_commit=commit,
            git_blob_sha256=hashlib.sha256(blob).hexdigest(),line_endings='EXACT' if raw==blob else 'CRLF_TO_LF',equivalence_status='PASS'))
    write(run/'execution_spec.json',dict(kind='INFRASTRUCTURE_ONLY',source_commit=commit,source_files=source_files(),source_bindings=provenance,
        prospective_bat={n:bat_text(n=='CHECK_STATUS.bat') for n in ('RUN_TRAINING.bat','CHECK_STATUS.bat')},**FLAGS))
    with tempfile.TemporaryDirectory() as directory:
        for name in source_files():
            if name.endswith('.py'): py_compile.compile(str(ROOT/name),cfile=str(Path(directory)/(name+'c')),doraise=True)
    with (run/'self_test_stdout.txt').open('x',encoding='utf-8') as out,(run/'self_test_stderr.txt').open('x',encoding='utf-8') as err:
        tested=subprocess.run([sys.executable,'-B',str(ROOT/('test_'+EXPERIMENT+'.py'))],cwd=ROOT,stdout=out,stderr=err)
    try: result=json.loads((run/'self_test_stdout.txt').read_text(encoding='utf-8').splitlines()[-1])
    except (ValueError,IndexError): result=dict(overall='FAIL',checks={},details={},**FLAGS)
    passed=tested.returncode==0 and result['overall']=='PASS' and all(result['checks'].get(k) for k in GROUPS) and all(result.get(k) is False for k in FLAGS)
    for group in GROUPS:
        write(run/(group+'.json'),dict(overall='PASS' if result['checks'].get(group) else 'FAIL',evidence=result.get('details',{}).get(group),synthetic_only=True,**FLAGS))
    write(run/'self_tests.json',result)
    status='PASS' if passed else 'FAIL'
    write(run/'validator.json',dict(overall=status,scope='STATIC_AND_SYNTHETIC_INDEPENDENT_FIXTURES',checks=result['checks'],failed_checks=[k for k in GROUPS if not result['checks'].get(k)],**FLAGS))
    labels = dict(REFERENCE_BINDING_STATUS='reference_binding_review',
        REFERENCE_REPRODUCTION_SPEC_STATUS='reference_reproduction_spec_test',
        FEATURE_INVENTORY_STATUS='feature_inventory_review', BUCKET_SPEC_STATUS='bucket_spec_review',
        CAUSALITY_GUARD_STATUS='causality_test', UNIVARIATE_LOGIC_STATUS='univariate_logic_test',
        BIVARIATE_LOGIC_STATUS='bivariate_logic_test', SAMPLE_GUARD_STATUS='sample_guard_test',
        FOLD_STABILITY_LOGIC_STATUS='fold_stability_test', STRESS_METRIC_STATUS='stress_metric_test',
        MULTIPLE_TESTING_INVENTORY_STATUS='multiple_testing_test', VALIDATOR_FIXTURE_STATUS='validator_fixture_test',
        EVENT_CHAIN_STATUS='event_chain_test', MANUAL_EXECUTION_GUARD_STATUS='manual_execution_guard_test',
        BAT_ONE_CLICK_STATUS='bat_static_test', HOLDOUT_GUARD_STATUS='holdout_guard_test',
        PRODUCTION_PROTECTION_STATUS='production_protection_test')
    checks = {label: 'PASS' if result['checks'].get(group) else 'FAIL' for label, group in labels.items()}
    checks.update(SOURCE_STATUS='PASS', SELF_TEST_STATUS=status, VALIDATOR_STATUS=status)
    write(run/'metrics.json',dict(formal_run_status=status,validator_status=status,approved_dimensions=len(spec['dimensions']),
        univariate_buckets=sum(len(d['labels']) for d in spec['dimensions']),bivariate_pairs=len(spec['pairs']),
        implementation_checks=checks, **FLAGS))
    report=('# GOLD S4 Entry Edge Decomposition v1 infrastructure\n\nStatus: '+status+'\n\nINFRASTRUCTURE_ONLY. '
        'No historical decomposition, filtering, backtest, model fitting, inference or holdout access. '
        'Schema/seal/source inspection only; actual missingness and reference reproduction are USER runtime gates. '
        '10 fixed dimensions and 8 fixed pairs, explicit missing/empty cells, no outcome-driven boundaries. '
        'Null PF has no finite denominator and cannot qualify for edge or stability. '
        'Independent validator recomputes joins, memberships, metrics, stress, rankings and all official tables. '
        'MFE/MAE exclude unordered exit-bar extremes. Probability is secondary-model score. '
        'BAT reviewed prospectively; installation requires this sealed PASS and source equivalence. '
        'Production unchanged. Real research remains USER ONLY.\n')
    for name in ('report.md','findings.md'): (run/name).write_text(report,encoding='utf-8')
    m=read(run/'manifest.json')
    m.update(infrastructure_only=True,manual_start=False,execution_flags=FLAGS,source_commit=commit,git_branch='main',git_remote='origin',git_ref='refs/heads/main')
    fill_manifest(m,ref)
    m['registry']['validator_result']=status
    write(run/'manifest.json',m)
    require(not history.finalize_run(run,'pass' if passed else 'fail'),'Infrastructure archive metadata')
    history.register_run(run)
    require(not history.validate_run(run),'Infrastructure provenance')
    print(json.dumps(dict(run_id=run.name,status=status,finalized_sha256=sha(run/'FINALIZED.json'),**FLAGS)))
    return 0 if passed else 1


def approve(path):
    commit=clean_pushed()
    run=Path(path).resolve()
    require(run.parent==ROOT/'training_runs' and run.name.endswith('_'+SLUG),'Infrastructure run only')
    require(not history.validate_run(run),'Sealed certification provenance')
    require(read(run/'validator.json')['overall']=='PASS' and read(run/'metrics.json')['formal_run_status']=='PASS','Certification PASS required')
    require(all(read(run/'metrics.json').get(k) is False for k in FLAGS),'No historical certification')
    for name in read(run/'execution_spec.json')['source_files']:
        require(sha(ROOT/name)==sha(run/'source'/name),'Certified source changed: '+name)
    from gold_s4_entry_edge_decomposition_v1_launcher import APPROVAL
    require(not (ROOT/APPROVAL).exists(),'No approval overwrite')
    for name in ('RUN_TRAINING.bat','CHECK_STATUS.bat'):
        value=bat_text(name=='CHECK_STATUS.bat')
        require(value==read(run/'execution_spec.json')['prospective_bat'][name],'Certified BAT text')
        (ROOT/name).write_bytes(value.replace('\n','\r\n').encode('utf-8'))
    config=read(ROOT/'training_launcher_config_v1.json')
    entry=lambda name:dict(path=name,sha256=sha(ROOT/name))
    ref=read(ROOT/'gold_s4_entry_edge_reference_v1.json')
    config.update(approval_status='APPROVED',workflow=dict(experiment_name=EXPERIMENT,script=entry(EXPERIMENT+'.py'),
        validator=entry('validate_'+EXPERIMENT+'_run.py'),configuration=entry(CONFIG),
        datasets=[dict(path=s['path'],sha256=s['sha256']) for s in sources(ref)],historical_input_mode='SEALED_ENTRY_SNAPSHOTS_USER_ONLY',native_io_review_pass=True))
    write(ROOT/'training_launcher_config_v1.json',config)
    write(ROOT/APPROVAL,dict(approved=True,workflow='GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1',execution_owner='USER',
        certification_run=run.relative_to(ROOT).as_posix(),finalized_sha256=sha(run/'FINALIZED.json'),
        source_commit=read(run/'manifest.json')['git_commit'],result_commit=commit,
        bindings={name:sha(ROOT/name) for name in source_files()+['training_launcher_config_v1.json']},**FLAGS))
    print('Approval installed. Commit/push then STOP. USER alone may double-click RUN_TRAINING.bat.')


if __name__=='__main__':
    if sys.argv[1:]==['--execute']: raise SystemExit(execute())
    if len(sys.argv)==3 and sys.argv[1]=='--approve': approve(sys.argv[2])
    else: raise SystemExit('Use --execute for STATIC/SYNTHETIC certification; --approve <sealed infrastructure run> to bind BAT')
