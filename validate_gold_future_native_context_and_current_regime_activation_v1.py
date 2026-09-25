"""One-shot independent v4 formal verifier; no execution-module imports."""
import ast
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_future_native_context_v1 import load, encode, digest, validate, sensitive_values, IDENTITY

SLUG = 'gold_future_native_context_and_current_regime_activation_v1'


def validate_run(run):
    root = run.parent.parent
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root)
    m = load(run/'manifest.json'); spec = load(run/'execution_spec.json')
    commit = m['source_commit']
    checks = {'source_commit': commit == m['git_commit'] == git('rev-parse', 'HEAD').decode().strip(),
              'remote_commit': commit == m['pre_run_remote_commit'] == git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0],
              'clean_provenance': m['git_dirty'] is False and m['pre_run_clean'] is True and m['pre_run_git_status'] == '',
              'protected': all(digest((root/p).read_bytes()) == h for p, h in spec['protected_sha256'].items()),
              'preserved_sources': all(digest((root/p).read_bytes()) == h for p, h in spec['preserved_sha256'].items()),
              'prior_seals': all(digest((root/p).read_bytes()) == h for p, h in spec['prior_seals'].items()),
              'strategy_source_bindings': all(digest((root/p).read_bytes()) == h for p, h in spec['strategy_binding_sha256'].items()),
              'prior_runs_unchanged': not git('diff', spec['base_commit'], '--', 'training_runs')}
    allowed = '?? '+run.relative_to(root).as_posix()+'/'
    checks['only_new_run_dirty'] = all(e.startswith(allowed) for e in git('status', '--porcelain', '-z').decode('utf-8').split('\0') if e)
    checks['source_inventory'] = set(spec['source_files']) == {e['source_path'] for e in m['input_snapshots']}
    for entry in m['input_snapshots']:
        name = entry['source_path']; path = run/entry['path']
        checks['source:'+name] = (path.resolve().is_relative_to((run/'source').resolve())
            and digest(path.read_bytes()) == entry['sha256'] == digest((root/name).read_bytes())
            and path.read_bytes().replace(b'\r\n', b'\n') == git('cat-file', 'blob', commit+':'+name).replace(b'\r\n', b'\n'))
    checks['validator_snapshot'] = Path(__file__).read_bytes() == (run/'validator_script.py').read_bytes() == (run/'source'/('validate_'+SLUG+'.py')).read_bytes()
    native = load(run/'native_context_validation.json')
    inventory = load(run/'native_context_inventory.json')
    if native['status'] == 'PASS':
        actual = validate(run/'context_seed', commit, digest((run/'source'/'gold_future_native_context_fetch_v1.py').read_bytes()))
        checks['native_independent'] = native == actual
        checks['inventory_alias'] = inventory == load(run/'context_seed'/'gold_future_native_context_inventory_v1.json')
        checks['manifest_alias'] = load(run/'native_context_file_manifest.json') == load(run/'context_seed'/'gold_future_native_context_manifest_v1.json')
    else:
        checks['native_unavailable_reported'] = bool(native.get('error'))
    diag = load(run/'current_regime_diagnostic.json')
    decision = load(run/'current_regime_decision.json')
    checks['decision_binding'] = decision == diag['decision']
    if decision['status'] == 'PASS':
        samples = diag['samples']
        checks['repeated_observations'] = len(samples) == 3
        checks['identity'] = all(all(s['source_identity'].get(k) == v for k, v in IDENTITY.items()) for s in samples)
        checks['monotonic'] = all(b['raw_epoch'] > a['raw_epoch'] and b['observed_utc_epoch'] > a['observed_utc_epoch'] for a, b in zip(samples, samples[1:]))
        for i, s in enumerate(samples):
            offset = math.floor((s['raw_epoch']-s['observed_utc_epoch'])/3600+.5)*3600
            error = s['raw_epoch']-10800-s['observed_utc_epoch']
            epochs = s['m1_raw_epochs']
            checks['arithmetic:'+str(i)] = offset == s['observed_offset_seconds'] == 10800 and abs(error) <= 5 and error == s['normalized_clock_error_seconds']
            checks['normalization:'+str(i)] = s['normalized_timestamp'] == datetime.fromtimestamp(s['raw_epoch']-10800, timezone.utc).isoformat()
            checks['M1_spacing:'+str(i)] = len(epochs) == 3 and all(e % 60 == 0 for e in epochs) and all(b-a == 60 for a, b in zip(epochs, epochs[1:])) and 0 <= s['raw_epoch']-epochs[-1] < 120
        checks['current_only'] = decision['certified_offsets_seconds'] == [10800] and decision['uncertified_candidate_offsets_seconds'] == [7200]
    else:
        checks['regime_blocker_reported'] = bool(diag.get('error'))
    policy = load(run/'runtime_time_validation_spec.json')
    checks['time_policy'] = (policy['policy'] == 'EMPIRICALLY_CERTIFIED_CURRENT_SOURCE_TIME_REGIME'
        and policy['candidate_offsets_seconds'] == [7200, 10800] and policy['certified_offsets_seconds'] == [10800]
        and policy['uncertified_candidate_offsets_seconds'] == [7200] and policy['runtime_fail_closed'] is True
        and policy['max_runtime_clock_skew_seconds'] == 5
        and policy['on_uncertified_allowed_offset'] == 'PAUSE_AND_REQUIRE_REGIME_CERTIFICATION'
        and policy['on_unexpected_offset'] == 'PAUSE_AND_QUARANTINE')
    prefix = load(run/'prefix_protocol.json')
    prior = load(run/'source'/'gold_recursive_prefix_causal_context_protocol_v1.json')
    pipeline = (run/'source'/'drl_trading_v2.py').read_bytes()
    function = [n for n in ast.walk(ast.parse(pipeline)) if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators']
    impl = digest(ast.dump(function[0], include_attributes=False).encode())
    expected_approval = native['status'] == 'PASS' and digest(pipeline) == prior['pipeline_sha256'] and impl == prior['feature_implementation_sha256']
    checks['prefix_gate'] = prefix['approved'] == expected_approval and prefix['initialization_semantics'] == prior['initialization_semantics']
    checks['prefix_honesty'] = prefix['historically_equivalent'] is False and prefix['outcome_tuning'] is False and prefix['context_role'] == 'CAUSAL_CONTEXT_ONLY' and prefix['holdout_evidence'] is False
    checks['prefix_alias'] = prefix == load(run/'prefix_decision.json')
    if expected_approval:
        checks['prefix_binding'] = prefix['inventory_root_sha256'] == inventory['inventory_root_sha256'] and prefix['feature_implementation_sha256'] == impl
    protocol = load(run/'source'/'gold_future_capture_protocol_v4.json')
    static = load(run/'collector_static_review.json')
    checks['pinned_runtime'] = static['runtime_code_sha256'] == protocol['runtime_code_sha256'] and all(digest((run/'source'/n).read_bytes()) == h for n, h in protocol['runtime_code_sha256'].items())
    forbidden_calls = {'order_send', 'order_check', 'fit', 'predict', 'predict_proba', 'load_model', '_asdict'}
    for name in protocol['runtime_code_sha256']:
        if not name.endswith('.py'):
            continue
        tree = ast.parse((run/'source'/name).read_bytes())
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else '' for n in ast.walk(tree) if isinstance(n, ast.Call)]
        modules = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        checks['no_execution:'+name] = not forbidden_calls.intersection(calls) and not any(any(x in n.lower() for x in ('xgboost', 'sklearn', 'drl_trading', 'gemini', 'torch')) for n in modules)
    certificate = load(run/'capture_certification_report.json')
    ready = native['status'] == decision['status'] == 'PASS' and expected_approval and static['status'] == 'STATICALLY_CONFORMANT'
    checks['readiness'] = (certificate['status'] == 'READY_FOR_CAPTURE_ACTIVATION') == ready
    expected_result = 'FAIL' if 'FAIL' in (native['status'], decision['status']) or static['status'] == 'FAIL' else 'PASS' if ready else 'PARTIAL'
    checks['formal_result'] = certificate['formal_run_status'] == expected_result
    checks['no_activation_during_formal'] = certificate['protocol_frozen'] is False and certificate['capture_activated'] is False
    tests = json.loads((run/'self_test_stdout.txt').read_text(encoding='utf-8'))
    checks['self_tests'] = tests['status'] == 'PASS' and tests['check_count'] >= 46
    sensitive_paths = []
    # Source snapshots and explicit spec deny lists are code/documentation, not account payloads.
    for path in run.rglob('*'):
        if not path.is_file() or 'source' in path.relative_to(run).parts or path.suffix == '.py':
            continue
        if path.name in {'execution_spec.json', 'validator_attempt.json'}:
            continue
        if path.suffix == '.json':
            if sensitive_values(load(path)):
                sensitive_paths.append(path.relative_to(run).as_posix())
        elif path.suffix in {'.txt', '.log', '.md'}:
            text = path.read_text(encoding='utf-8')
            if re.search(r'(?im)^\s*(?:login|balance|equity|margin(?:_free|_level)?|profit|credit|leverage)\s*[:=]\s*[-+0-9]', text):
                sensitive_paths.append(path.relative_to(run).as_posix())
    checks['no_sensitive_account_values'] = not sensitive_paths
    metrics = load(run/'metrics.json')
    checks['no_strategy_outcome'] = all(metrics[k] is False for k in ('strategy_outcome_inspected', 'model_loaded_for_holdout', 'model_trained_for_holdout', 'production_changed', 'production_promoted'))
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    marker = run/'validator_attempt.json'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'rule': 'Do not retry validation'}, handle)
    try:
        checks = validate_run(run)
        failed = [name for name, ok in checks.items() if not ok]
        result = {'overall': 'PASS' if not failed else 'FAIL', 'checks': checks, 'failed_check_names': failed}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': ['validator_exception'], 'exception_type': type(error).__name__}
    (run/'validator.json').write_bytes(encode(result))
    (run/'validator.md').write_text('# Independent validator\n\n'+result['overall']+'\n\n'+json.dumps(result['failed_check_names']), encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
