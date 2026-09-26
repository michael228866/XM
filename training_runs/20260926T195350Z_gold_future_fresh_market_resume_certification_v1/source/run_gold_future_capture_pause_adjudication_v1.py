"""Preserve pause evidence and observe source timestamps without resuming capture."""
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
from gold_future_capture_collector_v4 import recover_chain
from gold_future_capture_pause_adjudication_v1 import classify, resume_decision
from gold_future_capture_supervisor_v1 import coverage, process_exists
from gold_future_current_regime_diagnostic_v1 import observe, check_samples
from gold_future_source_identity_v4 import session
from run_gold_future_holdout_feature_eligibility_and_continuous_capture_v1 import task_projection
from test_gold_future_capture_pause_adjudication_v1 import tests

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout/gold_s4_v4'
SLUG = 'gold_future_capture_pause_adjudication_and_resume_v1'
SPEC = 'execution_spec_gold_future_capture_pause_adjudication_v1.json'
load = history.read_json
sha = history.file_sha256
write = history.write_json


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def integrity(spec):
    for name, expected in {**spec['preserved_sha256'], **spec['protected_sha256']}.items():
        if sha(ROOT/name) != expected:
            raise ValueError('PRESERVATION_MISMATCH:'+name)
    return {'preserved': True, 'protected': spec['protected_sha256']}


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('--execute required')
    spec = load(ROOT/SPEC)
    before = integrity(spec)
    if sha(CAPTURE/'PAUSED_TIME_RULE.json') != spec['original_pause_expected_sha256']:
        raise ValueError('ORIGINAL_PAUSE_CHANGED; adjudication scope must be reviewed')
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('CLEAN_PUSHED_SOURCE_REQUIRED')
    run = history.create_run(SLUG, Path(__file__), subprocess.list2cmdline(
        [sys.executable, '-B', str(Path(__file__)), '--execute']), arguments=['--execute'],
        seed_note='Deterministic timestamp-only adjudication; no strategy or model')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = load(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=commit, pre_run_clean=True, input_snapshots=[])
    for name in spec['source_files']:
        destination = run/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        m['input_snapshots'].append({'source_path': name, 'path': destination.relative_to(run).as_posix(),
                                     'sha256': sha(destination), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/SPEC, run/'execution_spec.json')
    shutil.copyfile(ROOT/'validate_gold_future_capture_pause_adjudication_v1.py', run/'validator_script.py')
    write(run/'manifest.json', m)
    # Preserve every pending file, not just the newest; no source receipts change.
    paths = [CAPTURE/'PAUSED_TIME_RULE.json', CAPTURE/'health/heartbeat.json', CAPTURE/'health/events.jsonl']
    paths += sorted((CAPTURE/'quarantine').glob('*')) + sorted((CAPTURE/'pending').glob('*'))
    evidence = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError('UNSAFE_OR_MISSING_ORIGINAL_EVIDENCE')
        relative = path.relative_to(CAPTURE)
        destination = run/'original_evidence'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        evidence.append({'source_path': path.relative_to(ROOT).as_posix(),
                         'path': destination.relative_to(run).as_posix(), 'sha256': sha(path)})
    write(run/'quarantine_evidence.json', {'files': evidence})
    shutil.copyfile(CAPTURE/'PAUSED_TIME_RULE.json', run/'pause_original.json')
    original = load(run/'pause_original.json')
    heartbeat = load(CAPTURE/'health/heartbeat.json')
    entries, previous = recover_chain(CAPTURE)
    chain_before = {p.relative_to(CAPTURE).as_posix(): sha(p)
                    for folder in ('manifests', 'snapshots') for p in (CAPTURE/folder).iterdir()}
    write(run/'chain_before.json', chain_before)
    write(run/'regression_results.json', tests())
    started = datetime.now(timezone.utc).isoformat()
    samples, observation_error = [], None
    try:
        with session() as (mt5, identity):
            for i in range(3):
                if i:
                    time.sleep(2)
                samples.append(observe(mt5))
    except (ValueError, RuntimeError, OSError) as error:
        observation_error = str(error) if isinstance(error, ValueError) else type(error).__name__
    write(run/'observations.json', {'samples': samples, 'error': observation_error,
                                   'scope': 'timestamp and sanitized identity only; no holdout admission'})
    last_raw = previous['rows'][-1]['RAW_SOURCE_EPOCH']
    adjudication = classify(samples, last_raw, previous['samples'][-1]['raw_epoch'])
    if observation_error:
        adjudication = {'classification': 'SOURCE_IDENTITY_CONFLICT' if 'IDENTITY' in observation_error else 'UNKNOWN',
                        'failed_rule': observation_error, 'market_data_fresh': False,
                        'evidence_admitted': False, 'resume_authorized': False}
    frozen_error = None
    if len(samples) == 3:
        try:
            check_samples(samples, previous['samples'][-1])
        except ValueError as error:
            frozen_error = str(error)
    latest = samples[-1] if samples else None
    closed = [e for e in latest['m1_raw_epochs'][:-1] if e+60 <= latest['raw_epoch']] if latest else []
    closed_raw = max(closed) if closed else None
    quarantines = [e for e in evidence if '/quarantine/' in e['source_path']]
    diagnostic = {
        'pause_created_at_utc': original['created_at_utc'], 'pause_reason': original['reason'],
        'collector_status': heartbeat['collector_status'], 'runtime_time_status': heartbeat['runtime_time_status'],
        'last_accepted_sequence': entries[-1]['sequence'],
        'last_accepted_source_timestamp': previous['rows'][-1]['SOURCE_TIMESTAMP'],
        'current_system_utc': utc(latest['observed_utc_epoch']) if latest else datetime.now(timezone.utc).isoformat(),
        'latest_tick_raw_epoch': latest['raw_epoch'] if latest else None,
        'latest_tick_normalized_utc': utc(latest['raw_epoch']-10800) if latest else None,
        'latest_closed_m1_raw_epoch': closed_raw,
        'latest_closed_m1_normalized_utc': utc(closed_raw-10800) if closed_raw else None,
        'tick_age_seconds': latest['observed_utc_epoch']-(latest['raw_epoch']-10800) if latest else None,
        'latest_closed_bar_age_seconds': latest['observed_utc_epoch']-(closed_raw-10800) if closed_raw else None,
        'bar_age_basis': 'age since native bar OPEN; newest returned bar excluded, successor and tick closure required',
        'inferred_offset_seconds': math.floor((latest['raw_epoch']-latest['observed_utc_epoch'])/3600+.5)*3600 if latest else None,
        'offset_inference_valid': adjudication['market_data_fresh'],
        'normalized_clock_error_seconds': latest['raw_epoch']-10800-latest['observed_utc_epoch'] if latest else None,
        'market_data_fresh': adjudication['market_data_fresh'],
        'new_closed_bar_available': closed_raw is not None and closed_raw > last_raw,
        'failed_rule_names': ['OFFSET_NOT_CERTIFIED'],
        'exact_original_predicate': 'offset = floor((raw_epoch - observed_utc_epoch)/3600 + 0.5)*3600; offset != 7200 and offset != 10800',
        'original_actual_offset_seconds': None, 'original_actual_tick_epoch': None,
        'original_actual_observed_utc_epoch': None, 'original_samples_retained': original.get('payload') is not None,
        'original_sole_cause_proven': False,
        'historical_attribution_limit': 'Original receipt payload is null; current observations cannot prove original pause sole cause',
        'certified_offsets_seconds': [10800], 'uncertified_candidate_offsets_seconds': [7200],
        'max_runtime_clock_skew_seconds': 5, 'freshness_rule': 'source tick or latest M1 epoch advances across three observations spaced two seconds; reversals conflict before idle',
        'current_frozen_check_failure': frozen_error, 'current_adjudicated_rule': adjudication['failed_rule'],
        'quarantine_artifact': quarantines[0]['source_path'], 'quarantine_sha256': quarantines[0]['sha256'],
    }
    write(run/'pause_diagnostic.json', diagnostic)
    write(run/'gold_future_capture_pause_diagnostic_v1.json', diagnostic)
    write(run/'freshness_adjudication.json', adjudication)
    decision = resume_decision(adjudication['classification'], original.get('payload') is not None)
    write(run/'resume_decision.json', {'decision': decision, 'resume_authorized': False,
        'original_sole_cause_proven': False, 'original_pause_preserved': True,
        'reason': 'Original samples were not retained. No causal proof sufficient for conditional correction or resume.'})
    write(run/'runtime_policy_decision.json', {'status': 'PARTIAL', 'runtime_fail_closed': True,
        'amendment_applied': False, 'collector_changed': False, 'supervisor_changed': False,
        'diagnostic_distinguishes_idle_from_conflict': True, 'runtime_healthy_idle_supported': False,
        'reason': 'Diagnostic stale-data failure reproduced; original pause sole-cause proof unavailable. Runtime change conditional authorization unmet.'})
    write(run/'collector_diff_review.json', {'collector_v4_changed': False, 'runtime_changed': False,
        'classification_scope': 'read-only diagnostic, not admission or operational amendment',
        'original_check_order': ['identity', 'nearest_hour_offset', 'skew', 'reversal_or_stale', 'M1_spacing', 'M1_current'],
        'stale_clock_assumption_demonstrated_synthetically': True})
    task = task_projection()
    write(run/'scheduled_task_config.json', {'status': 'INSTALLED' if task else 'FAIL', 'configuration': task})
    health = dict(heartbeat, current_process_running=process_exists(heartbeat['pid']),
                  process_lock_present=(CAPTURE/'health/collector.lock').exists(),
                  assessed_at_utc=datetime.now(timezone.utc).isoformat(),
                  heartbeat_assessment='Preserved final paused heartbeat; not updating and not a running-health PASS')
    write(run/'capture_health_after.json', health)
    write(run/'capture_coverage_after.json', coverage())
    feature_path = ROOT/'gold_future_holdout_feature_eligibility_v1.json'
    write(run/'feature_eligibility_after.json', {'status': 'PARTIAL', 'rerun': False,
        'prior_result_path': feature_path.name, 'prior_result_sha256': sha(feature_path),
        'earliest_feature_complete_timestamp': None, 'reason': 'No newly sealed evidence; frozen eligibility not rerun',
        'missing_prerequisites': ['M1 19:21/19:22/19:23 UTC', 'native closed M2/M3/M4/M6/M12'],
        'limitation': 'Existing checker requires seed-to-candidate continuity; M1-only collector supplies no future HTF. Natural eventual eligibility not proven.'})
    write(run/'evaluation_start_decision.json', {'holdout_start': '2026-09-25T19:24:00+00:00',
        'holdout_evaluation_start': None, 'original_boundary_sha256': sha(ROOT/'gold_future_holdout_boundary_v4.json')})
    after = integrity(spec)
    chain_after = {p.relative_to(CAPTURE).as_posix(): sha(p)
                   for folder in ('manifests', 'snapshots') for p in (CAPTURE/folder).iterdir()}
    if chain_after != chain_before or any(sha(ROOT/e['source_path']) != e['sha256'] for e in evidence):
        raise ValueError('ORIGINAL_EVIDENCE_CHANGED')
    write(run/'protection_checks.json', {'before': before, 'after': after, 'chain_unchanged': True,
                                        'original_evidence_unchanged': True})
    metrics = {'formal_run_status': 'PARTIAL', 'pause_adjudication_status': 'PARTIAL',
        'runtime_policy_status': 'PARTIAL', 'resume_decision': decision, 'continuous_capture_status': 'PAUSED',
        'runtime_classification': adjudication['classification'], 'chain_status': 'PASS', 'feature_eligibility_status': 'PARTIAL',
        'validator_status': 'PENDING', 'original_sole_cause_proven': False,
        'lookahead_detected': False, 'strategy_outcome_inspected': False, 'model_loaded_for_holdout': False,
        'model_trained_for_holdout': False, 'strategy_executed': False, 'production_changed': False,
        'production_promoted': False, 'evidence_admitted': False}
    write(run/'metrics.json', metrics)
    report = ('# GOLD capture pause adjudication\n\nFormal result: PARTIAL.\n\n'
        'Original failed branch: rounded offset != 7200 and != 10800; reason PAUSE_AND_QUARANTINE. '
        'Original payload is null, so its exact numeric inputs and sole cause cannot be recovered. '
        'Current source observations are separately retained and never substituted for historical proof.\n\n'
        + json.dumps(adjudication, indent=2)+'\n\n'
        'The frozen validator infers offset before testing tick progress. Synthetic stale-source reproduction confirms '
        'this general defect, but cannot prove it solely caused this particular original pause. '
        'No operational amendment, pause clearing, task restart or evidence admission occurred. '
        'Diagnostic tests do not establish a healthy idle supervisor. The heartbeat is a preserved stopped receipt.\n\n'
        'Feature eligibility remains PARTIAL; no new sealed inputs justify a rerun. '
        'The original prefix gap and absent future native HTF remain blockers under unchanged rules. '
        'HOLDOUT_START is unchanged; HOLDOUT_EVALUATION_START remains null. No model, strategy, outcome or production change.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    for name in ('stdout.txt', 'stdout.log'):
        (run/name).write_text('FORMAL_RUN_STATUS=PARTIAL\n', encoding='utf-8')
    (run/'stderr.txt').write_text('', encoding='utf-8')
    data = m['data']
    data.update(symbols=['GOLD#'], data_sources=['Preserved pause receipts and read-only current MT5 timestamps'],
        source_files=m['input_snapshots'], timezone='Operational UTC and raw source labels; stale offset not certified',
        raw_snapshot_retained=True, reproducibility_claim='Current timestamp projections and original receipts retained; original failed samples were not retained',
        purge_details='No strategy dataset', embargo_details='No strategy dataset')
    for key in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[key] = 'NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for key in ('train_rows', 'validation_rows', 'test_rows'):
        data[key] = 0
    data['mt5_fetch'].update(used=True, terminal_path=r'D:\XM2\terminal64.exe',
        terminal_info=latest['source_identity'] if latest else {'status': 'unavailable'},
        broker_info=latest['source_identity'] if latest else {'status': 'unavailable'},
        fetch_start_utc=started, fetch_end_utc=datetime.now(timezone.utc).isoformat(),
        retrieved_at_utc=diagnostic['current_system_utc'], returned_rows=sum(len(s['m1_raw_epochs']) for s in samples))
    m['model']['not_applicable_reason'] = 'No model loaded or trained'
    m['search']['not_applicable_reason'] = 'No strategy search'
    m['registry'].update({k: 'N/A: timestamp-only pause adjudication' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Frozen GOLD v4',
        selected_configuration='PARTIAL; original causal evidence unavailable; '+decision+'; pause retained; no production change',
        validator_result='PENDING')
    m.update(formal_run_status='PARTIAL', pause_adjudication_status='PARTIAL')
    write(run/'manifest.json', m)
    print('FORMAL_RUN_STATUS=PARTIAL', flush=True)


if __name__ == '__main__':
    main()
