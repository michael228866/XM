"""One-shot independent pause verifier; stdlib only, no collector/runner imports."""
import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def independent_state(samples, previous_tick, last_raw):
    expected = {'broker_company': 'XM Global Limited', 'broker_server': 'XMGlobal-MT5 6',
                'source_account_environment': 'demo', 'source_id': 'XMGlobal-MT5-6_GOLD',
                'symbol': 'GOLD#', 'digits': 2, 'point': .01}
    if len(samples) < 3:
        return 'UNKNOWN', 'INSUFFICIENT_CURRENT_OBSERVATIONS', False
    for s in samples:
        if any(s['source_identity'].get(k) != v for k, v in expected.items()):
            return 'SOURCE_IDENTITY_CONFLICT', 'SOURCE_IDENTITY_CHANGED', False
        if type(s['raw_epoch']) is not int or not math.isfinite(s['observed_utc_epoch']):
            return 'TIME_RULE_CONFLICT', 'INVALID_CLOCK_SAMPLE', False
        es = s['m1_raw_epochs']
        if len(es) < 3 or any(type(e) is not int or e % 60 for e in es) or any(es[i]-es[i-1] != 60 for i in range(1, len(es))):
            return 'TIME_RULE_CONFLICT', 'CURRENT_M1_SPACING_INVALID', False
    progressing = []
    for i, s in enumerate(samples):
        prior = samples[i-1] if i else None
        if ((not prior and s['raw_epoch'] < previous_tick) or (prior and (
                s['raw_epoch'] < prior['raw_epoch'] or s['observed_utc_epoch'] <= prior['observed_utc_epoch']
                or s['m1_raw_epochs'][-1] < prior['m1_raw_epochs'][-1]))):
            return 'TIME_RULE_CONFLICT', 'TIMESTAMP_REVERSAL', False
        if prior and (s['raw_epoch'] > prior['raw_epoch'] or s['m1_raw_epochs'][-1] > prior['m1_raw_epochs'][-1]):
            progressing.append(s)
    if not progressing:
        return 'NO_FRESH_MARKET_DATA', 'NO_FRESH_TICK', False
    for s in progressing:
        delta = s['raw_epoch']-s['observed_utc_epoch']
        if 3600*math.floor(delta/3600+.5) != 10800:
            return 'TIME_RULE_CONFLICT', 'OFFSET_NOT_CERTIFIED', True
        if abs(delta-10800) > 5:
            return 'TIME_RULE_CONFLICT', 'CLOCK_SKEW_EXCEEDED', True
        if not 0 <= s['raw_epoch']-s['m1_raw_epochs'][-1] < 120:
            return 'TIME_RULE_CONFLICT', 'CURRENT_M1_NOT_CURRENT', True
    closed = [e for e in samples[-1]['m1_raw_epochs'][:-1] if e+60 <= samples[-1]['raw_epoch']]
    if not closed or max(closed) <= last_raw:
        return 'NO_FRESH_MARKET_DATA', 'NO_NEW_CLOSED_BAR', True
    if any(samples[i]['raw_epoch'] <= samples[i-1]['raw_epoch'] for i in range(1, len(samples))):
        return 'NO_FRESH_MARKET_DATA', 'INSUFFICIENT_FRESH_SAMPLES', True
    for s in samples:
        delta = s['raw_epoch']-s['observed_utc_epoch']
        if math.floor(delta/3600+.5)*3600 != 10800:
            return 'TIME_RULE_CONFLICT', 'OFFSET_NOT_CERTIFIED', True
        if abs(delta-10800) > 5:
            return 'TIME_RULE_CONFLICT', 'CLOCK_SKEW_EXCEEDED', True
        if not 0 <= s['raw_epoch']-s['m1_raw_epochs'][-1] < 120:
            return 'TIME_RULE_CONFLICT', 'CURRENT_M1_NOT_CURRENT', True
    return 'TIME_RULE_PASS', None, True


def validate(run):
    root = run.parent.parent
    capture = root/'future_holdout/gold_s4_v4'
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root)
    spec = load(run/'execution_spec.json'); manifest = load(run/'manifest.json')
    commit = manifest['source_commit']
    checks = {
        'source_commit': commit == manifest['git_commit'] == git('rev-parse', 'HEAD').decode().strip(),
        'remote_commit': commit == manifest['pre_run_remote_commit'] == git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0],
        'clean_source': manifest['git_dirty'] is False and manifest['pre_run_clean'] is True,
        'protected': all(digest(root/n) == h for n, h in spec['protected_sha256'].items()),
        'preserved': all(digest(root/n) == h for n, h in spec['preserved_sha256'].items()),
        'validator_snapshot': Path(__file__).read_bytes() == (run/'validator_script.py').read_bytes(),
        'source_inventory': {e['source_path'] for e in manifest['input_snapshots']} == set(spec['source_files']),
    }
    for entry in manifest['input_snapshots']:
        name = entry['source_path']
        checks['source:'+name] = (digest(run/entry['path']) == entry['sha256'] == digest(root/name)
                                  and (root/name).read_bytes() == git('cat-file', 'blob', commit+':'+name))
    # Static call/import inventory of the executed runner and its local helpers.
    # Feature source is parsed/hashed only; the feature construction is not called.
    prohibited = {'xgboost', 'lightgbm', 'catboost', 'sklearn', 'joblib', 'pickle', 'gemini'}
    pending = ['run_gold_future_capture_pause_adjudication_v1.py']; scanned = set()
    while pending:
        name = pending.pop()
        if name in scanned:
            continue
        scanned.add(name)
        tree = ast.parse((root/name).read_bytes())
        modules = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        modules += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        checks['static_safety:'+name] = (not prohibited.intersection(m.split('.')[0] for m in modules)
            and not {'fit', 'predict', 'predict_proba', 'load_model', 'order_send', 'order_check'}.intersection(calls))
        pending.extend(m+'.py' for m in modules if (root/(m+'.py')).is_file())
    original = load(run/'pause_original.json')
    checks['original_pause'] = ((run/'pause_original.json').read_bytes() == (capture/'PAUSED_TIME_RULE.json').read_bytes()
        and original['reason'] == 'PAUSE_AND_QUARANTINE' and original['payload'] is None)
    evidence = load(run/'quarantine_evidence.json')['files']
    required = {p.relative_to(root).as_posix() for folder in ('quarantine', 'pending') for p in (capture/folder).iterdir()}
    required |= {(capture/p).relative_to(root).as_posix() for p in ('PAUSED_TIME_RULE.json', 'health/heartbeat.json', 'health/events.jsonl')}
    checks['original_evidence_inventory'] = {e['source_path'] for e in evidence} == required
    checks['original_evidence_bytes'] = all(digest(root/e['source_path']) == e['sha256'] == digest(run/e['path']) for e in evidence)
    d = load(run/'pause_diagnostic.json')
    source = (root/'gold_future_current_regime_diagnostic_v1.py').read_text()
    checks['exact_original_predicate'] = ('if offset == 7200:' in source and 'if offset != 10800:' in source
        and "raise ValueError('PAUSE_AND_QUARANTINE')" in source
        and d['failed_rule_names'] == ['OFFSET_NOT_CERTIFIED']
        and d['original_actual_offset_seconds'] is None and d['original_actual_tick_epoch'] is None
        and d['original_actual_observed_utc_epoch'] is None and d['original_sole_cause_proven'] is False)
    checks['diagnostic_alias'] = d == load(run/'gold_future_capture_pause_diagnostic_v1.json')
    checks['quarantine_binding'] = digest(root/d['quarantine_artifact']) == d['quarantine_sha256'] == digest(run/'pause_original.json')
    freeze = load(capture/'protocol/freeze.json')
    activation = load(capture/'attestations/activation.json')
    checks['frozen_commit'] = (git('cat-file', 'blob', activation['protocol_freeze_commit']+':gold_future_locked_holdout_protocol_v4.json')
                                == canonical(freeze))
    checks['certified_regime'] = freeze['certified_offsets_seconds'] == d['certified_offsets_seconds'] == [10800] and freeze['uncertified_candidate_offsets_seconds'] == [7200]
    checks['frozen_gates'] = freeze['runtime_fail_closed'] is True and freeze['outcome_inspection_prohibited'] is True and freeze['prefix_approved'] is True
    checks['frozen_code'] = all(digest(root/n) == h for n, h in freeze['code_sha256'].items())
    inv = load(capture/'context_seed/gold_future_native_context_inventory_v1.json')
    checks['context_root'] = inv['inventory_root_sha256'] == freeze['context_inventory_root_sha256'] == hashlib.sha256(canonical(inv['entries'])).hexdigest()
    checks['context_bytes'] = all(digest(capture/'context_seed'/e['raw_path']) == e['raw_sha256']
        and (capture/'context_seed/manifests'/(e['timeframe']+'.json')).read_bytes() == canonical(e) for e in inv['entries'])
    chain = load(run/'chain_before.json')
    current = {p.relative_to(capture).as_posix(): digest(p) for folder in ('snapshots', 'manifests') for p in (capture/folder).iterdir()}
    checks['chain_unchanged'] = current == chain
    previous_hash, previous_raw, previous_tick = None, None, None
    for number, path in enumerate(sorted((capture/'manifests').glob('*.json'))):
        item = load(path); snapshot = capture/item['snapshot_path']; payload = load(snapshot)
        valid = (path.name == f'{number:012d}.json' and item['sequence'] == number
            and item['previous_manifest_sha256'] == previous_hash and digest(snapshot) == item['snapshot_sha256']
            and item['snapshot_path'] == f'snapshots/{number:012d}.json'
            and item['activation_sha256'] == hashlib.sha256(canonical(activation)).hexdigest()
            and payload['native_source_confirmed'] is True and payload['resampled'] is False)
        for sample in payload['samples']:
            delta = sample['raw_epoch']-sample['observed_utc_epoch']
            valid &= math.floor(delta/3600+.5)*3600 == 10800 and abs(delta-10800) <= 5
            valid &= previous_tick is None or sample['raw_epoch'] > previous_tick
            previous_tick = sample['raw_epoch']
        for row in payload['rows']:
            raw = row['RAW_SOURCE_EPOCH']
            valid &= raw in payload['returned_raw_epochs'][:-1] and raw+60 <= previous_tick
            valid &= previous_raw is None or raw > previous_raw
            valid &= datetime.fromtimestamp(raw-10800, timezone.utc).isoformat() == row['SOURCE_TIMESTAMP']
            previous_raw = raw
        checks['chain:'+str(number)] = bool(valid)
        previous_hash = digest(path)
    boundary = load(root/'gold_future_holdout_boundary_v4.json')
    checks['boundary'] = boundary['holdout_start'] == '2026-09-25T19:24:00+00:00' and boundary['holdout_evaluation_start'] is None
    observations = load(run/'observations.json'); samples = observations['samples']
    state, rule, fresh = independent_state(samples, previous_tick, previous_raw)
    if observations['error']:
        state = 'SOURCE_IDENTITY_CONFLICT' if 'IDENTITY' in observations['error'] else 'UNKNOWN'
        rule = observations['error']; fresh = False
    a = load(run/'freshness_adjudication.json')
    checks['independent_classification'] = (a['classification'], a['failed_rule'], a['market_data_fresh']) == (state, rule, fresh)
    checks['no_diagnostic_admission'] = a['evidence_admitted'] is False and a['resume_authorized'] is False
    if samples:
        s = samples[-1]; raw = s['raw_epoch']; clock = s['observed_utc_epoch']
        closed = max(e for e in s['m1_raw_epochs'][:-1] if e+60 <= raw)
        checks['clock_arithmetic'] = (d['latest_tick_raw_epoch'] == raw
            and d['current_system_utc'] == datetime.fromtimestamp(clock, timezone.utc).isoformat()
            and d['latest_tick_normalized_utc'] == datetime.fromtimestamp(raw-10800, timezone.utc).isoformat()
            and d['tick_age_seconds'] == clock-raw+10800
            and d['normalized_clock_error_seconds'] == raw-10800-clock
            and d['inferred_offset_seconds'] == math.floor((raw-clock)/3600+.5)*3600
            and d['offset_inference_valid'] == fresh)
        checks['closed_bar_arithmetic'] = (d['latest_closed_m1_raw_epoch'] == closed
            and d['latest_closed_bar_age_seconds'] == clock-closed+10800
            and d['new_closed_bar_available'] == (closed > previous_raw))
    resume = load(run/'resume_decision.json')
    expected_decision = ('WAIT_FOR_FRESH_MARKET_DATA' if state == 'NO_FRESH_MARKET_DATA'
        else 'PAUSE_REMAINS_VALID' if state.endswith('CONFLICT') else 'FAIL')
    checks['resume_denied_without_original_proof'] = resume['decision'] == expected_decision and resume['resume_authorized'] is False and resume['original_sole_cause_proven'] is False
    policy = load(run/'runtime_policy_decision.json')
    checks['no_runtime_amendment'] = policy['status'] == 'PARTIAL' and policy['amendment_applied'] is False and policy['collector_changed'] is False and policy['supervisor_changed'] is False and policy['runtime_healthy_idle_supported'] is False
    health = load(run/'capture_health_after.json')
    checks['paused_health_honest'] = health['collector_status'] == 'PAUSED' and health['process_running'] is False and health['current_process_running'] is False and health['process_lock_present'] is False
    task = load(run/'scheduled_task_config.json')
    checks['task'] = task['status'] == 'INSTALLED' and task['configuration']['RunLevel'] == 'Limited' and task['configuration']['UserMatchesCurrent'] is True
    feature = load(run/'feature_eligibility_after.json')
    checks['feature_not_rerun'] = feature['rerun'] is False and feature['status'] == 'PARTIAL' and digest(root/feature['prior_result_path']) == feature['prior_result_sha256'] and feature['earliest_feature_complete_timestamp'] is None
    evaluation = load(run/'evaluation_start_decision.json')
    checks['evaluation_not_forced'] = evaluation['holdout_evaluation_start'] is None and evaluation['holdout_start'] == boundary['holdout_start'] and evaluation['original_boundary_sha256'] == digest(root/'gold_future_holdout_boundary_v4.json')
    metrics = load(run/'metrics.json')
    checks['honest_partial'] = all(metrics[k] == 'PARTIAL' for k in ('formal_run_status', 'pause_adjudication_status', 'runtime_policy_status'))
    checks['no_strategy_or_production'] = all(metrics[k] is False for k in ('lookahead_detected', 'strategy_outcome_inspected', 'model_loaded_for_holdout', 'model_trained_for_holdout', 'strategy_executed', 'production_changed', 'production_promoted', 'evidence_admitted'))
    regressions = load(run/'regression_results.json')
    checks['regressions'] = regressions['overall'] == 'PASS' and all(regressions['checks'].values()) and regressions['test_count'] >= 17 and regressions['healthy_idle_supervisor_proven'] is False
    allowed = '?? '+run.relative_to(root).as_posix()+'/'
    checks['only_run_untracked'] = all(e.startswith(allowed) for e in git('status', '--porcelain', '-z').decode('utf-8').split('\0') if e)
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'rule': 'Do not retry validation'}, stream)
    try:
        checks = validate(run)
        failed = [k for k, v in checks.items() if not v]
        result = {'overall': 'FAIL' if failed else 'PASS', 'failed_check_names': failed, 'checks': checks}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': ['validator_exception'], 'exception_type': type(error).__name__}
    (run/'validator.json').write_bytes(canonical(result))
    (run/'validator.md').write_text('# Independent pause adjudication validator\n\n'+result['overall']+'\n\n'+json.dumps(result['failed_check_names'])+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
