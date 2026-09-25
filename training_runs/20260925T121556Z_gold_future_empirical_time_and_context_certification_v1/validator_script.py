"""Independent empirical arithmetic and context verifier; no execution imports."""
import argparse
import ast
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Pure independent CSV verifier from the completed inventory; not its builder.
from validate_gold_future_causal_context_inventory_v1 import scan, read_header

SLUG = 'gold_future_empirical_time_and_context_certification_v1'
IDENTITY = {'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#', 'broker': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'environment': 'demo'}


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def classification(segment):
    epochs = segment['raw_epochs']; anchors = segment['anchors']
    deltas = [a['raw_epoch']-a['observed_utc_epoch'] for a in anchors]
    offsets = [int(math.floor(v/3600+0.5)*3600) for v in deltas]
    errors = [v-o for v, o in zip(deltas, offsets)]
    ordered = all(b > a for a, b in zip(epochs, epochs[1:]))
    spacing = all(v % 60 == 0 for v in epochs) and all((b-a) % 60 == 0 for a, b in zip(epochs, epochs[1:]))
    anchor_order = all(b['raw_epoch'] > a['raw_epoch'] and b['observed_utc_epoch'] > a['observed_utc_epoch'] for a, b in zip(anchors, anchors[1:]))
    bad = not ordered or not spacing or not anchor_order or any(o not in {7200, 10800} or abs(e) > 5 for o, e in zip(offsets, errors))
    label = 'CONTRADICTED' if bad else 'AMBIGUOUS_TRANSITION' if anchors and len(set(offsets)) != 1 else 'OPERATIONAL_UTC_ANCHORED' if anchors else 'EMPIRICAL_INTERNAL_CONSISTENCY' if epochs else 'INSUFFICIENT_DATA'
    return {'classification': label, 'sample_count': len(epochs), 'raw_epoch_min': min(epochs) if epochs else None,
            'raw_epoch_max': max(epochs) if epochs else None, 'observed_offset_seconds': offsets[0] if len(set(offsets)) == 1 else None,
            'direct_utc_offset_summary': deltas, 'normalized_utc_error_summary': errors,
            'monotonic': ordered, 'bar_spacing_valid': spacing, 'source_symbol': segment['identity']['symbol'], 'source_server': segment['identity']['server']}


def validate(run):
    root = run.parent.parent; s = run/'source'
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root)
    m = load(run/'manifest.json'); spec = load(run/'execution_spec.json'); commit = m['source_commit']
    checks = {'source_commit': commit == m['git_commit'] == git('rev-parse', 'HEAD').decode().strip(),
              'remote_commit': commit == m['pre_run_remote_commit'] == git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0],
              'clean_source': m['git_dirty'] is False and m['pre_run_clean'] is True and m['pre_run_git_status'] == '',
              'source_inventory': set(spec['source_files']) == {e['source_path'] for e in m['input_snapshots']},
              'v1_v2_unchanged': all(sha(root/n) == h for n, h in spec['preserved_sha256'].items()),
              'prior_seals': all(sha(root/n) == h for n, h in spec['prior_seals'].items()),
              'prior_runs_unchanged': not git('diff', spec['base_commit'], '--', 'training_runs')}
    allowed = '?? '+run.relative_to(root).as_posix()+'/'
    checks['only_new_run_untracked'] = all(e.startswith(allowed) for status in (m['git_status_after_execution'], git('status', '--porcelain', '-z').decode('utf-8')) for e in status.split('\0') if e)
    for e in m['input_snapshots']:
        p = (run/e['path']).resolve()
        if not p.is_relative_to(s.resolve()):
            raise ValueError('Snapshot escape')
        checks['source:'+e['source_path']] = sha(p) == e['sha256'] == sha(root/e['source_path']) and p.read_bytes().replace(b'\r\n', b'\n') == git('cat-file', 'blob', commit+':'+e['source_path']).replace(b'\r\n', b'\n')
    checks['validator_snapshot'] = Path(__file__).read_bytes() == (run/'validator_script.py').read_bytes() == (s/('validate_'+SLUG+'.py')).read_bytes()
    checks['runner_snapshot'] = sha(run/'training_script.py') == m['training_script_sha256'] == sha(s/('run_'+SLUG+'.py'))
    checks['protected_production'] = m['protected_sha256_before'] == m['protected_sha256_after'] == spec['protected_sha256'] == {n: sha(root/n) for n in spec['protected_sha256']}
    observations = load(run/'empirical_time_segments.json'); decision = load(run/'empirical_time_decision.json')
    att = load(run/'gold_future_capture_time_attestation_v3.json'); source = load(run/'source_attestation_v3.json')
    recomputed = []
    for seg in observations['segments']:
        expected = classification(seg); recomputed.append(expected)
        checks['segment:'+seg['segment_id']] = all(seg[k] == v for k, v in expected.items()) and seg['identity'] == IDENTITY
        a = datetime.fromisoformat(seg['sample_start']); b = datetime.fromisoformat(seg['sample_end'])
        if seg['regime'] != 'current':
            checks['historical_scope:'+seg['segment_id']] = not seg['anchors'] and seg['observed_offset_seconds'] is None and (b-a).total_seconds() == 300 and len(seg['raw_epochs']) <= 6 and all(a.timestamp() <= v <= b.timestamp() for v in seg['raw_epochs'])
        else:
            checks['current_scope:'+seg['segment_id']] = len(seg['anchors']) == 2 and len(seg['raw_epochs']) <= 3 and a.timestamp() == seg['anchors'][0]['observed_utc_epoch'] and b.timestamp() == seg['anchors'][-1]['observed_utc_epoch']
    regimes = {seg['regime'] for seg in observations['segments']}
    checks['bounded_windows'] = len(observations['segments']) <= 9 and ('error_type' in observations or regimes == {'winter', 'summer', 'march_before', 'march_after', 'october_before', 'october_after', 'current'})
    offsets = sorted({r['observed_offset_seconds'] for r in recomputed if r['classification'] == 'OPERATIONAL_UTC_ANCHORED'})
    bad = any(r['classification'] in {'CONTRADICTED', 'AMBIGUOUS_TRANSITION'} for r in recomputed)
    expected_status = 'FAIL' if bad else 'PARTIAL'
    policy = 'CONTRADICTED' if bad else 'EMPIRICALLY_SUPPORTED_BUT_INSUFFICIENT' if offsets else 'UNRESOLVED'
    checks['empirical_decision'] = decision['status'] == expected_status and decision['policy'] == policy and decision['observed_offsets_seconds'] == offsets and decision['allowed_offsets_seconds'] == [7200, 10800] and decision['empirical_coverage_start'] is None and decision['empirical_coverage_end'] is None and decision['certified_segments'] == []
    checks['clock_honesty'] = observations['clock']['quality'] == 'OPERATIONAL_CHECK_ONLY' and observations['clock']['max_allowed_clock_skew_seconds'] == 5 and decision['clock_quality'] == 'OPERATIONAL_CHECK_ONLY'
    checks['source_identity'] = all(source[k] == v for k, v in IDENTITY.items()) and source['template'] == (not observations['identity_verified']) and source['historical_file_origin_certified'] is False and (not observations['identity_verified'] or observations['identity'] == IDENTITY)
    checks['attestation_honesty'] = att['template'] is True and att['status'] == policy and att['official_attestation_required'] is False and att['broker_internal_semantics_claimed'] is False and att['empirical_segments'] == observations['segments'] and att['empirical_evidence'][0]['sha256'] == sha(run/'empirical_time_segments.json')
    runtime = load(run/'runtime_time_validation_spec.json')
    checks['runtime_frozen_spec'] = runtime == load(s/'gold_future_empirical_time_protocol_v1.json') and runtime['allowed_offsets_seconds'] == [7200, 10800] and runtime['runtime_validation_required'] is True and runtime['runtime_fail_closed'] is True and runtime['official_broker_attestation_required'] is False and runtime['official_evidence_role'] == 'SUPPLEMENTARY_ONLY' and runtime['holiday_evidence_role'] == 'EXPLANATORY_METADATA'
    prior = root/spec['inventory_run']; prior_seal = load(prior/'FINALIZED.json')
    for old, alias in [('gold_future_causal_context_inventory_v1.json', 'causal_context_inventory.json'), ('gold_future_causal_context_file_manifest_v1.json', 'causal_context_file_manifest.json')]:
        checks['prior_inventory:'+old] = sha(prior/old) == prior_seal['file_sha256'][old] == sha(run/old) == sha(run/alias)
    inv = load(run/'causal_context_inventory.json'); fm = load(run/'causal_context_file_manifest.json')
    frames = inv['timeframes']; expected_frames = set(load(s/'gold_future_capture_manifest_schema_v2.json')['properties']['timeframe']['enum'])
    checks['exact_21_frames'] = len(frames) == 21 and {e['timeframe'] for e in frames} == expected_frames
    for e in fm['entries']:
        checks['external_bytes:'+e['path']] = sha(e['path']) == e['sha256'] and Path(e['path']).stat().st_size == e['size']
    for e in frames:
        print('Context validation: '+e['timeframe'], flush=True)
        p = Path(e['source_path']); schema = read_header(p); computed = scan(p, schema, e['timeframe'])
        checks['context:'+e['timeframe']] = schema == e['schema'] and canonical(schema) == e['schema_sha256'] and all(e[k] == v for k, v in computed.items()) and e['required_warmup_bars'] == (4096 if e['timeframe'] == 'M1' else 21) and e['warmup_requirement_satisfied'] is True and e['resampled'] is False and e['native_source_confirmed'] is False and e['source_classification'] == 'POTENTIALLY_COMPATIBLE'
        checks['preboundary:'+e['timeframe']] = all('2024-01-01T00:00:00' <= r['timestamp'] < r['successor_timestamp'] <= '2026-09-24T00:00:00' for r in e['selected_rows'])
        checks['context_stable:'+e['timeframe']] = sha(p) == e['source_file_sha256']
    ordered = sorted(fm['entries'], key=lambda e: (e['timeframe'] or '', e['path'].replace('\\', '/').casefold()))
    root_hash = hashlib.sha256(''.join(canonical(e) for e in ordered).encode('ascii')).hexdigest()
    checks['inventory_root'] = fm['inventory_root_sha256'] == inv['inventory_root_sha256'] == root_hash
    comp = load(run/'causal_context_compatibility.json'); prefix = load(run/'prefix_protocol.json'); pd = load(run/'prefix_decision.json')
    checks['context_compatibility'] = comp['status'] == 'PASS_STRUCTURAL_ONLY' and comp['formal_status'] == 'PARTIAL' and comp['official_attestation_required'] is False and comp['time_rule_compatibility'] == 'UNRESOLVED' and comp['native_origin_certified'] is False and {e['timeframe'] for e in comp['timeframes']} == expected_frames
    checks['prefix_honesty'] = prefix == load(s/'gold_recursive_prefix_protocol_v3.json') and prefix['approved'] is False and pd['approved'] is False and pd['policy'] == 'UNRESOLVED' and pd['status'] == 'PARTIAL' and prefix['inventory_root_sha256'] == root_hash
    checks['context_never_holdout'] = all(doc['context_role'] == 'CAUSAL_CONTEXT_ONLY' and all(doc[k] is False for k in ('historically_equivalent', 'outcome_tuning', 'holdout_evidence')) for doc in (inv, prefix, pd))
    fn = next(n for n in ast.parse((s/'drl_trading_v2.py').read_text(encoding='utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators')
    checks['macd_binding'] = prefix['pipeline_sha256'] == sha(s/'drl_trading_v2.py') and prefix['feature_implementation_sha256'] == hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() and prefix['recursive_features'] == ['MACD_HIST'] and prefix['initialization_semantics'] == load(s/'gold_recursive_prefix_causal_context_protocol_v1.json')['initialization_semantics']
    protocol = load(s/'gold_future_capture_protocol_v3.json'); static = load(run/'collector_static_review.json'); report = load(run/'capture_certification_report.json')
    tree = ast.parse((s/'gold_future_capture_collector_v3.py').read_text(encoding='utf-8'))
    checks['collector_pinned'] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == protocol['approved_collector_ast_sha256'] == static['collector_ast_sha256'] and sha(s/'gold_future_capture_collector_v3.py') == static['collector_sha256'] and all(sha(s/n) == h for n, h in protocol['dependency_sha256'].items())
    replacements = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'os' and n.func.attr == 'replace']
    checks['immutable_replace_allowlist'] = len(replacements) == 1 and ast.dump(replacements[0].args[1], include_attributes=False) == ast.dump(ast.parse("file_under(root, 'chain_tip.json')", mode='eval').body, include_attributes=False)
    checks['static_report'] = static == report['collector_static_review'] and static['static_status'] == 'STATICALLY_CONFORMANT' and static['blockers'] == [] and static['runtime_validator_sha256'] == sha(s/'gold_future_empirical_time_v1.py')
    checks['new_policy_preserved_invariants'] = protocol['official_broker_attestation_required'] is False and protocol['holiday_calendar_role'] == 'EXPLANATORY_METADATA' and protocol['higher_timeframe_policy'] == 'NATIVE_20TF_REQUIRED' and protocol['chain_tip_policy'] == 'NON_AUTHORITATIVE_REBUILDABLE_INDEX'
    blockers = {'EMPIRICAL_TIME_COVERAGE_INSUFFICIENT', 'CONTEXT_TIME_OR_NATIVE_PROVENANCE_UNRESOLVED', 'PREFIX_UNAPPROVED'}
    if not observations['identity_verified']:
        blockers.add('SOURCE_IDENTITY_UNRESOLVED')
    checks['certification'] = report['formal_run_status'] == expected_status and report['readiness']['status'] == 'NOT_READY_MULTIPLE_BLOCKERS' and set(report['readiness']['blockers']) == blockers and report['time_attestation_sha256'] == canonical(att) and report['source_attestation_sha256'] == canonical(source)
    metrics = load(run/'metrics.json')
    checks['metrics_verdict'] = metrics['formal_run_status'] == m['formal_run_status'] == expected_status and metrics['empirical_time_status'] == expected_status and metrics['time_policy'] == policy and metrics['context_inventory_status'] == 'PARTIAL' and metrics['timeframe_count'] == 21 and metrics['native_timeframe_count'] == 0 and metrics['m1_context_bars'] == 4096 and metrics['m1_requirement_satisfied'] is True and metrics['htf_requirements_satisfied'] is True
    checks['no_activation_production'] = all(metrics[k] is False for k in ('prefix_approved', 'protocol_frozen', 'capture_activated', 'holdout_started', 'strategy_outcome_inspected', 'model_loaded_for_holdout', 'model_trained_for_holdout', 'production_changed', 'production_promoted')) and all(metrics[k] is None for k in ('capture_start', 'holdout_start', 'holdout_evaluation_start'))
    for n in ('gold_future_empirical_time_v1.py', 'gold_future_empirical_time_diagnostic_v1.py', 'gold_future_capture_certification_v3.py', 'gold_future_capture_collector_v3.py', 'run_'+SLUG+'.py'):
        parsed = ast.parse((s/n).read_text(encoding='utf-8'))
        imports = {a.name for node in ast.walk(parsed) if isinstance(node, ast.Import) for a in node.names} | {node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)}
        calls = {node.func.attr for node in ast.walk(parsed) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        checks['no_strategy:'+n] = not imports.intersection({'xgboost', 'torch', 'sklearn', 'gemini', 'drl_trading_v2'}) and not calls.intersection({'fit', 'predict', 'predict_proba', 'load_model', 'order_send'})
    tests = load(run/'self_test_stdout.txt')
    checks['self_tests_runtime'] = tests['overall'] == 'PASS' and tests['synthetic_only'] is True and {'paused_time_rule', 'quarantined_without_chain_addition', 'observed_offset_7200', 'observed_offset_10800', 'unexpected_offset_3600', 'unexpected_offset_14400', 'ambiguous_transition', 'raw_epoch_preserved', 'no_automatic_resume'} <= set(tests['checks'])
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    if (run/'FINALIZED.json').exists():
        raise FileExistsError('Finalized archive immutable')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as f:
        json.dump({'one_shot': True, 'started_at_utc': datetime.now(timezone.utc).isoformat()}, f)
    try:
        checks = validate(run)
    except Exception as error:
        checks = {'exception:'+type(error).__name__: False}
    failed = [k for k, v in checks.items() if not v]
    result = {'overall': 'FAIL' if failed else 'PASS', 'failed_check_names': failed, 'checks': checks}
    with (run/'validator.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
    with (run/'validator.md').open('x', encoding='utf-8') as f:
        f.write('# Independent empirical time/context validation\n\n'+result['overall']+'\nNo official guarantee or promotion claim.\n')
    print('overall='+result['overall']); print('failed_check_names='+json.dumps(failed))
    raise SystemExit(bool(failed))
