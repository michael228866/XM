"""Bounded fresh-market certification; amendment/resume requires later validation."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
from gold_future_capture_collector_v4 import recover_chain
from gold_future_capture_supervisor_v1 import coverage, process_exists
from gold_future_fresh_market_diagnostic_v1 import assess, observe
from gold_future_source_identity_v4 import session
from run_gold_future_capture_pause_adjudication_v1 import integrity, git
from run_gold_future_holdout_feature_eligibility_and_continuous_capture_v1 import task_projection
from test_gold_future_fresh_market_resume_certification_v1 import tests

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout/gold_s4_v4'
SLUG = 'gold_future_fresh_market_resume_certification_v1'
SPEC = 'execution_spec_'+SLUG+'.json'
load, write, sha = history.read_json, history.write_json, history.file_sha256


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('--execute required')
    spec = load(ROOT/SPEC)
    before = integrity(spec)
    if sha(CAPTURE/'PAUSED_TIME_RULE.json') != spec['original_pause_expected_sha256']:
        raise ValueError('ORIGINAL_PAUSE_CHANGED')
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('CLEAN_PUSHED_SOURCE_REQUIRED')
    run = history.create_run(SLUG, Path(__file__), subprocess.list2cmdline(
        [sys.executable, '-B', str(Path(__file__)), '--execute']), arguments=['--execute'],
        seed_note='Fixed engineering freshness thresholds; no strategy/model/search')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = load(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=commit, pre_run_clean=True, input_snapshots=[])
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        m['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
            'sha256': sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/SPEC, run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'), run/'validator_script.py')
    write(run/'manifest.json', m)
    evidence = []
    paths = [CAPTURE/'PAUSED_TIME_RULE.json', CAPTURE/'health/heartbeat.json', CAPTURE/'health/events.jsonl']
    paths += sorted((CAPTURE/'quarantine').iterdir()) + sorted((CAPTURE/'pending').iterdir())
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError('UNSAFE_OR_MISSING_EVIDENCE')
        dest = run/'original_evidence'/path.relative_to(CAPTURE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        evidence.append({'source_path': path.relative_to(ROOT).as_posix(), 'path': dest.relative_to(run).as_posix(), 'sha256': sha(path)})
    write(run/'original_evidence_inventory.json', {'files': evidence})
    shutil.copyfile(CAPTURE/'PAUSED_TIME_RULE.json', run/'original_pause_snapshot.json')
    entries, previous = recover_chain(CAPTURE)
    chain = {p.relative_to(CAPTURE).as_posix(): sha(p) for folder in ('snapshots','manifests') for p in (CAPTURE/folder).iterdir()}
    write(run/'chain_before.json', chain)
    write(run/'regression_results.json', tests())
    last_raw = previous['rows'][-1]['RAW_SOURCE_EPOCH']
    previous_tick = previous['samples'][-1]['raw_epoch']
    samples, observation_error = [], None
    started = datetime.now(timezone.utc).isoformat()
    try:
        with session() as (mt5, identity):
            deadline = time.monotonic()+130
            for index in range(5):
                if index:
                    time.sleep(2)
                samples.append(observe(mt5))
                write(run/'fresh_samples.json', {'samples': samples, 'error': None})
            diagnostic = assess(samples, previous_tick, last_raw)
            # A stale batch ends promptly. Fresh batches may wait at most two
            # additional minutes for a second native closure; no infinite wait.
            while (diagnostic['classification'] == 'TIME_RULE_PASS'
                   and (diagnostic['fresh_tick_sample_count'] < 5 or diagnostic['new_closed_m1_count'] < 2)
                   and time.monotonic()+20 <= deadline):
                print('WAITING_FOR_SECOND_FRESH_NATIVE_M1', flush=True)
                time.sleep(20)
                samples.append(observe(mt5))
                write(run/'fresh_samples.json', {'samples': samples, 'error': None})
                diagnostic = assess(samples, previous_tick, last_raw)
    except (ValueError, RuntimeError, OSError) as error:
        observation_error = str(error) if isinstance(error, ValueError) else type(error).__name__
    write(run/'fresh_samples.json', {'samples': samples, 'error': observation_error})
    diagnostic = assess(samples, previous_tick, last_raw)
    if observation_error:
        diagnostic.update(classification='UNKNOWN', current_regime_status='PARTIAL',
                          source_identity_status='FAIL' if 'IDENTITY' in observation_error else diagnostic['source_identity_status'])
    write(run/'fresh_market_diagnostic.json', diagnostic)
    fresh_valid = (not observation_error and diagnostic['current_regime_status'] == 'PASS'
                   and diagnostic['new_closed_m1_count'] >= 2 and diagnostic['source_identity_status'] == 'PASS')
    prior = ROOT/spec['prior_adjudication_run']
    prior_d = load(prior/'pause_diagnostic.json')
    stale_reproduced = (prior_d['market_data_fresh'] is False
                       and prior_d['current_frozen_check_failure'] == 'PAUSE_AND_QUARANTINE')
    root_cause = 'STALE_SOURCE_OBSERVATION_MISCLASSIFIED_AS_TIME_RULE_CONFLICT' if fresh_valid and stale_reproduced else 'NOT_PROVEN'
    write(run/'stale_data_root_cause.json', {'root_cause': root_cause, 'stale_reproduction': stale_reproduced,
        'prior_run': spec['prior_adjudication_run'], 'prior_finalized_sha256': sha(prior/'FINALIZED.json'),
        'fresh_restores_10800': fresh_valid,
        'historical_limit': 'Original failed samples absent. New task authorizes stale reproduction plus independent fresh recertification as amendment basis.'})
    write(run/'current_regime_decision.json', {'status': diagnostic['current_regime_status'],
        'classification': diagnostic['classification'], 'observed_offset_seconds': diagnostic['observed_offset_seconds'],
        'current_certified_offset_seconds': 10800, 'certified_offsets_seconds': [10800],
        'uncertified_candidate_offsets_seconds': [7200], 'max_runtime_clock_skew_seconds': 5})
    decision = ('WAIT_FOR_FRESH_MARKET_DATA' if diagnostic['classification'] == 'NO_FRESH_MARKET_DATA'
        else 'PAUSE_REMAINS_VALID' if diagnostic['classification'] in {'TIME_RULE_CONFLICT','UNCERTIFIED_CANDIDATE_REGIME'} else 'FAIL')
    write(run/'resume_decision.json', {'decision': decision, 'resume_authorized': False,
        'next_step': 'Implement and independently certify narrow amendment before resume' if fresh_valid else 'Fresh recertification not established; preserve pause',
        'fresh_certification_ready_for_amendment': fresh_valid})
    write(run/'runtime_freshness_policy.json', {'status': 'PARTIAL', 'amendment_applied': False,
        'diagnostic_freshness_first': True, 'runtime_fail_closed': True,
        'reason': 'Separate amendment must follow independently verified fresh evidence; original collector and supervisor preserved'})
    for name in ('collector_change_review.json','supervisor_change_review.json'):
        write(run/name, {'runtime_changed': False, 'old_bytes_preserved': True, 'healthy_idle_demonstrated': False})
    task = task_projection()
    write(run/'scheduled_task_config.json', {'status': 'INSTALLED' if task else 'FAIL', 'configuration': task})
    heartbeat = load(CAPTURE/'health/heartbeat.json')
    health = dict(heartbeat, current_process_running=process_exists(heartbeat['pid']),
        process_lock_present=(CAPTURE/'health/collector.lock').exists(),
        heartbeat_age_seconds=datetime.now(timezone.utc).timestamp()-datetime.fromisoformat(heartbeat['last_cycle_completed_at_utc']).timestamp(),
        heartbeat_status='FAIL', assessment='Paused, no resume occurred; final old heartbeat is not live health')
    write(run/'capture_health_after_resume.json', health)
    write(run/'capture_coverage_after_resume.json', coverage())
    feature = ROOT/'gold_future_holdout_feature_eligibility_v1.json'
    write(run/'feature_eligibility_after_resume.json', {'status': 'PARTIAL', 'rerun': False,
        'prior_result_path': feature.name, 'prior_result_sha256': sha(feature),
        'earliest_feature_complete_timestamp': None, 'reason': 'No new sealed inputs; do not rerun frozen eligibility',
        'missing_prerequisites': ['M1 19:21/19:22/19:23 UTC', 'native closed M2/M3/M4/M6/M12']})
    write(run/'evaluation_start_decision.json', {'holdout_start':'2026-09-25T19:24:00+00:00',
        'holdout_evaluation_start':None, 'original_boundary_sha256':sha(ROOT/'gold_future_holdout_boundary_v4.json')})
    after = integrity(spec)
    chain_after = {p.relative_to(CAPTURE).as_posix(): sha(p) for folder in ('snapshots','manifests') for p in (CAPTURE/folder).iterdir()}
    if chain != chain_after or any(sha(ROOT/e['source_path']) != e['sha256'] for e in evidence):
        raise ValueError('ORIGINAL_EVIDENCE_CHANGED')
    write(run/'protection_checks.json', {'before':before,'after':after,'chain_unchanged':True,'original_evidence_unchanged':True})
    verdict = 'FAIL' if diagnostic['current_regime_status'] == 'FAIL' else 'PARTIAL'
    metrics = {'formal_run_status':verdict, 'fresh_market_certification_status':verdict,
        'fresh_sample_count':diagnostic['fresh_tick_sample_count'], 'new_closed_m1_count':diagnostic['new_closed_m1_count'],
        'snapshots_before':len(entries), 'snapshots_after':len(entries), 'latest_sequence':entries[-1]['sequence'],
        'new_sealed_snapshot_count':0, 'eligible_candidates_checked':0, 'resume_decision':decision,
        'validator_status':'PENDING', 'lookahead_detected':False, 'strategy_outcome_inspected':False,
        'model_loaded_for_holdout':False, 'model_trained_for_holdout':False, 'strategy_executed':False,
        'production_changed':False, 'production_promoted':False}
    write(run/'metrics.json', metrics)
    report = ('# GOLD fresh market resume certification\n\nFormal result: '+verdict+'\n\n'
        +json.dumps({k:v for k,v in diagnostic.items() if k != 'samples'},indent=2)+'\n\n'
        'Fixed freshness screens: tick 120 seconds, closed M1 180 seconds under existing 10800. '
        'Observed feed progression is separate fresh evidence so live 7200 cannot hide as aged 10800 data. '
        'Offset and skew are evaluated only after freshness; stale rows carry null offset/skew. '
        'Five initial observations two seconds apart; stale batches stop. Fresh batches wait at most 130 seconds total for two closures.\n\n'
        'ROOT_CAUSE='+root_cause+'. No runtime amendment or resume occurred in this certification phase. '
        'No healthy-idle claim. Original pause, quarantine, pending, context, boundary, chain and frozen runtime remain unchanged. '
        'No bars appended or backfilled. Feature eligibility not rerun without new sealed input. '
        'No model/strategy/outcome evaluation or production change.\n')
    for name in ('report.md','findings.md'):
        (run/name).write_text(report,encoding='utf-8')
    for name in ('stdout.txt','stdout.log'):
        (run/name).write_text('FORMAL_RUN_STATUS='+verdict+'\n',encoding='utf-8')
    (run/'stderr.txt').write_text('',encoding='utf-8')
    data=m['data']
    data.update(symbols=['GOLD#'],data_sources=['Read-only live timestamps and frozen pause/chain evidence'],source_files=m['input_snapshots'],
        timezone='UTC system; fixed 10800 freshness screen; no offset inference on stale rows',raw_snapshot_retained=True,
        reproducibility_claim='Timestamp projections and receipts retained; no market prices or strategy outputs collected',
        purge_details='No strategy dataset',embargo_details='No strategy dataset')
    for key in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'):
        data[key]='NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for key in ('train_rows','validation_rows','test_rows'): data[key]=0
    data['mt5_fetch'].update(used=True,terminal_path=r'D:\XM2\terminal64.exe',
        terminal_info=samples[-1]['source_identity'] if samples else {'status':'unavailable'},
        broker_info=samples[-1]['source_identity'] if samples else {'status':'unavailable'},fetch_start_utc=started,
        fetch_end_utc=datetime.now(timezone.utc).isoformat(),retrieved_at_utc=datetime.now(timezone.utc).isoformat(),returned_rows=len(samples)*3)
    m['model']['not_applicable_reason']='No model loaded or trained'
    m['search']['not_applicable_reason']='No strategy search'
    m['registry'].update({k:'N/A: fresh-market timestamp certification' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Frozen GOLD v4',selected_configuration=verdict+'; '+decision+'; no production change',validator_result='PENDING')
    m.update(formal_run_status=verdict)
    write(run/'manifest.json',m)
    print('FORMAL_RUN_STATUS='+verdict,flush=True)


if __name__ == '__main__':
    main()
