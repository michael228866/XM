"""Decomposition only: USER loads sealed trades; every group retains full partitions."""
import contextlib
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from gold_s4_entry_edge_decomposition_v1_support import (
    ROOT, CONFIG, EXPERIMENT, FLAGS, read, write, require, configuration, check_production,
    load_reference, reproduce, group_tables, summaries, csv_write, event, artifact_hashes,
)
from gold_s4_entry_edge_feature_inventory_v1 import attach_snapshots, availability

OFFICIAL = ['entry_edge_univariate.csv','entry_edge_bivariate.csv','entry_edge_fold_stability.csv',
    'entry_edge_score_calibration.csv','entry_edge_regime_summary.csv',
    'entry_edge_top_positive_buckets.json','entry_edge_top_negative_buckets.json',
    'entry_edge_stability_summary.json','multiple_testing_inventory.json']
FREEZE = OFFICIAL+['entry_snapshots.json','reference_ledger.json','feature_inventory.json',
                  'bucket_spec.json','reference_reproduction.json','reference_binding.json','approved_config.json']


def decompose(run,rows,spec,config):
    """Pure output operation, used also with temporary synthetic fixtures."""
    print('[6/10] Run Univariate Decomposition',flush=True)
    single,fold_single=group_tables(rows,spec,config['folds'])
    csv_write(run/'entry_edge_univariate.csv',single)
    event(run,'univariate_complete',artifact_hashes(run,['entry_edge_univariate.csv']))
    print('[7/10] Run Frozen Bivariate Decomposition',flush=True)
    paired,fold_paired=group_tables(rows,spec,config['folds'],True)
    csv_write(run/'entry_edge_bivariate.csv',paired)
    event(run,'bivariate_complete',artifact_hashes(run,['entry_edge_bivariate.csv']))
    print('[8/10] Fold Stability / Stress Analysis',flush=True)
    csv_write(run/'entry_edge_fold_stability.csv',fold_single+fold_paired)
    csv_write(run/'entry_edge_score_calibration.csv',[r for r in single if r['dimensions']==['model_score']])
    csv_write(run/'entry_edge_regime_summary.csv',[r for r in single if r['dimensions'][0] in ('volatility','short_term_trend','trend_alignment')])
    positive,negative,multiple,stability=summaries(single,paired,spec)
    for name,value in [('entry_edge_top_positive_buckets.json',positive),('entry_edge_top_negative_buckets.json',negative),
                       ('multiple_testing_inventory.json',multiple),('entry_edge_stability_summary.json',stability)]:
        write(run/name,value)
    event(run,'stability_complete',artifact_hashes(run,OFFICIAL[2:]))
    event(run,'research_freeze',artifact_hashes(run,FREEZE))
    return stability


def research(run,config,ref,spec,inventory):
    from gold_s4_entry_edge_decomposition_v1_launcher import require_session
    require_session()
    print('[3/10] Load Frozen A_NO_LONG_HTF Trade Set',flush=True)
    original,control=load_reference(ref)
    write(run/'reference_ledger.json',original)
    event(run,'reference_loaded',artifact_hashes(run,['reference_ledger.json','reference_binding.json']))
    print('[4/10] Reproduce Reference Metrics',flush=True)
    write(run/'reference_reproduction.json',reproduce(control,ref))
    event(run,'reference_reproduction_pass',artifact_hashes(run,['reference_reproduction.json']))
    print('[5/10] Audit Entry-Time Feature Availability',flush=True)
    rows=attach_snapshots(original,control,ref)
    write(run/'entry_snapshots.json',rows)
    write(run/'feature_inventory.json',availability(rows,inventory))
    event(run,'feature_inventory_frozen',artifact_hashes(run,['feature_inventory.json','entry_snapshots.json']))
    event(run,'bucket_spec_frozen',artifact_hashes(run,['bucket_spec.json']))
    return decompose(run,rows,spec,config)


def run_manual(token):
    from gold_s4_entry_edge_decomposition_v1_launcher import begin_session,end_session
    begin_session(token)
    try:
        return _run()
    finally:
        end_session()


def _run():
    from gold_s4_entry_edge_decomposition_v1_launcher import require_session,verify_approval,validation_permit
    require_session()
    import numpy  # Import before the native-library audit guard; only NPZ via Python file handles.
    import training_run_history as history
    from gold_manual_s4_train_validate_v1 import Tee
    from training_holdout_guard_v1 import install
    from validate_gold_s4_entry_edge_decomposition_v1_run import run_authorized,check_chain
    from certify_gold_s4_entry_edge_decomposition_v1 import clean_pushed,source_files,fill_manifest
    approval=verify_approval()
    config,ref,spec,inventory=configuration()
    commit=clean_pushed()
    print('[2/10] Historical Data / Provenance',flush=True)
    run=history.create_run(EXPERIMENT,Path(__file__),'RUN_TRAINING.bat (USER double-click)',seed_note='Deterministic predeclared grouping; no fitting or bootstrap')
    status=dict(run_id=run.name,execution_status='FAIL',research_result='NOT_RUN',selected_candidate=None,
        production_changed=False,production_promoted=False,holdout_used=False,failed_checks=[],**FLAGS)
    manifest=read(run/'manifest.json')
    manifest.update(manual_start=True,started_by='USER_LAUNCHER',infrastructure_only=False,source_commit=commit,
        source_bindings=approval['bindings'],git_branch='main',git_remote='origin',git_ref='refs/heads/main')
    fill_manifest(manifest,ref,infrastructure=False)
    (run/'source').mkdir()
    for name in source_files()+['training_launcher_config_v1.json']:
        shutil.copyfile(ROOT/name,run/'source'/name)
    for source,dest in [(CONFIG,'approved_config.json'),(config['reference_binding'],'reference_binding.json'),
                         (config['bucket_spec'],'bucket_spec.json'),(config['feature_inventory'],'predeclared_inventory.json')]:
        shutil.copyfile(ROOT/source,run/dest)
    write(run/'manifest.json',manifest)
    release=None
    try:
        release=install(ROOT,write_root=run)
        status.update(REAL_RESEARCH_EXECUTED=True,HISTORICAL_DATA_USED=True)
        with (run/'research_stdout.txt').open('x',encoding='utf-8') as out,(run/'research_stderr.txt').open('x',encoding='utf-8') as err,contextlib.redirect_stdout(Tee(sys.stdout,out)),contextlib.redirect_stderr(Tee(sys.stderr,err)):
            status.update(research(run,config,ref,spec,inventory))
            print('[9/10] Independent Validation',flush=True)
            validation=run_authorized(run,validation_permit(run))
            event(run,'validation_result',artifact_hashes(run,['validator.json']))
            require(validation['overall']=='PASS','Independent validation failed: '+repr(validation['failed_checks']))
            status['execution_status']='PASS'
    except BaseException as error:
        (run/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status['failed_checks']=[str(error)]
        if 'REFERENCE_TRADESET_MISMATCH' in str(error):
            status.update(research_result='NOT_RUN',error='REFERENCE_TRADESET_MISMATCH',result_category='REFERENCE_MISMATCH')
        write(run/'failure.json',status)
    finally:
        if release: release()
    try:
        check_production(config)
    except ValueError as error:
        status.update(execution_status='FAIL',production_changed=True)
        status['failed_checks'].append(str(error))
    manifest.update({k:status[k] for k in FLAGS})
    manifest['registry'].update(validator_result=status['execution_status'],selected_configuration=status['research_result'])
    write(run/'manifest.json',manifest)
    for name in ('combined_result.json','metrics.json'): write(run/name,status)
    report='# GOLD S4 Entry Edge Decomposition v1\n\n'+str(status)+'\n\nHistorical development, descriptive overlapping buckets only. No entry filter or promotion. Null PF means no loss denominator and is ineligible for edge claims. MFE/MAE are completed-bar lower-bound diagnostics. Score is secondary-model probability, not sole accepted-entry confidence. All 787 entries preserved.\n'
    for name in ('report.md','findings.md'): (run/name).write_text(report,encoding='utf-8')
    if status['execution_status']=='PASS':
        event(run,'finalization',artifact_hashes(run,['combined_result.json','metrics.json']))
        check_chain(run,complete=True)
    print('[10/10] Finalize',flush=True)
    errors=history.finalize_run(run,'research_only' if status['execution_status']=='PASS' else 'aborted',aborted_reason='; '.join(status['failed_checks']) or None)
    require(not errors,'Archive: '+repr(errors))
    history.register_run(run)
    require(not history.validate_run(run),'Preserved provenance validation')
    subprocess.run(['git','add','-f','--',run.relative_to(ROOT).as_posix(),'TRAINING_RUNS.md'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Archive user-started GOLD S4 Entry Edge Decomposition v1 '+run.name],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT).decode().split()[0]==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),'Remote archival commit')
    return status


if __name__=='__main__':
    raise SystemExit('USER must manually double-click RUN_TRAINING.bat; direct historical execution forbidden')
