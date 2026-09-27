"""Independent archive/static validator. Never imports training implementation."""
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(run):
    checks = {}
    manifest = load(run/'manifest.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0]
    checks['pushed_source'] = head == remote == manifest['git_commit'] == manifest['source_commit']
    checks['pre_run_clean'] = manifest['git_dirty'] is False and manifest['pre_run_clean'] is True
    for item in manifest['input_snapshots']:
        checks['source:' + item['source_path']] = sha(run/item['path']) == item['sha256'] == sha(ROOT/item['source_path'])
    trace = load(run/'training_provenance_trace.json')
    checks['provenance_honest'] = trace['status'] == 'PARTIAL' and trace['selected_default'] is None and len(trace['candidates']) == 4
    for i, candidate in enumerate(trace['candidates']):
        bindings = [candidate['script'], candidate['label_pipeline'], *candidate['feature_pipeline']]
        bindings += [candidate[k] for k in ('seal', 'config', 'training_function_source') if k in candidate]
        if 'model' in candidate:
            bindings.append(candidate['model'])
        if candidate['classification'] == 'B0_BASELINE_RESEARCH':
            bindings += [{'path': candidate['run']+'/'+item['path'], 'sha256': item['sha256']}
                         for item in candidate['models']]
        for item in bindings:
            checks[f'candidate{i}:' + item['path']] = sha(ROOT/item['path']) == item['sha256']
        features = json.dumps(candidate['feature_list'], separators=(',', ':')).encode()
        checks[f'features{i}'] = hashlib.sha256(features).hexdigest() == candidate['feature_list_sha256'] and len(candidate['feature_list']) == 31
        checks[f'exact_symbol{i}'] = candidate['symbol'] == 'GOLD#' and candidate['source_id'] == 'XMGlobal-MT5-6_GOLD'
    checks['latest_historical_not_default'] = trace['latest_formally_executed_historical_training'].endswith('20260919T124853Z_gold_secondary_s4_confirmation_v1') and trace['latest_approved_repeatable_manual_workflow'] is None
    spec = load(run/'approved_training_spec.json')
    config = load(run/'approved_training_config.json')
    checks['null_approval_not_fabricated'] = spec['approval_status'] == 'UNRESOLVED' and all(spec[k] is None for k in ('training_script', 'training_config', 'training_symbol', 'historical_training_end_rule'))
    checks['config_binding'] = sha(ROOT/config['spec']['path']) == config['spec']['sha256'] and config['training_workflow_ready'] is False
    checks['trace_binding'] = sha(ROOT/spec['provenance_trace']['path']) == spec['provenance_trace']['sha256']
    checks['holdout_frozen'] = spec['holdout_policy'] == {'locked_tree': 'future_holdout/gold_s4_v4', 'start_utc': '2026-09-25T19:24:00+00:00', 'access': False}
    checks['data_policy_partial'] = spec['dataset_policy']['status'] == 'PARTIAL' and spec['dataset_policy']['requirements'] == [] and spec['dataset_policy']['auto_fetch_enabled'] is False
    inventory = load(run/'historical_data_inventory.json')
    checks['no_invented_eligible_data'] = inventory['training_eligible'] is False and inventory['datasets'] == []
    source = (ROOT/'gold_historical_training_data_manager_v1.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    mt5_calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'mt5'}
    checks['mt5_read_only_allowlist'] = mt5_calls == {'initialize', 'account_info', 'symbol_info', 'symbol_select', 'copy_rates_range', 'shutdown'}
    checks['no_mt5_order_or_model_calls'] = not any(isinstance(n, ast.Attribute) and n.attr in {'fit', 'predict', 'predict_proba', 'order_send', 'order_check'} for n in ast.walk(tree))
    checks['identity_sanitized'] = "('company', 'server', 'trade_mode')" in source and '_asdict' not in source
    checks['data_no_substitution_or_resampling'] = "mt5.copy_rates_range(req['symbol'], getattr(mt5, 'TIMEFRAME_' + req['timeframe'])" in source and '.resample(' not in source
    checks['certified_coverage_required'] = "req.get('coverage_certified') is not True" in source and "req['expected_timestamps']" in source
    tests = load(run/'launcher_smoke_test.json')
    checks['synthetic_tests'] = tests['overall'] == 'PASS' and tests['test_count'] >= 60 and all(tests['checks'].values())
    checks['tested_holdout_and_cutoff'] = all(tests['checks'][k] for k in ('post_cutoff', 'holdout_timestamp', 'audit_hook_denies_before_open'))
    checks['manual_context_receipt'] = all(tests['checks'][k] for k in ('receipt_replay', 'receipt_expiry', 'receipt_wrong_process', 'automated_receipt_denied'))
    wrapper = ast.parse((ROOT/'gold_manual_training_workflow_v1.py').read_bytes())
    entry = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'run_manual')
    checks['training_entry_unconditionally_blocks'] = isinstance(entry.body[-1], ast.Raise) and not any(isinstance(n, ast.Attribute) and n.attr in {'fit', 'predict', 'predict_proba'} for n in ast.walk(wrapper))
    bat = (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['bat_root_interpreter_no_args_pause'] = all(t in bat for t in ('%~dp0', '.venv\\Scripts\\python.exe', 'manual_training_launcher_v1.py', 'pause >nul', 'PYTHONUTF8=1')) and '%1' not in bat
    audit = load(run/'auto_training_audit.json')
    checks['automation_audit_pass'] = audit['status'] == 'PASS' and audit['auto_training_disabled'] is True
    serialized = json.dumps(audit['live']).lower()
    checks['no_installed_automatic_trainer'] = all(s not in serialized for s in ('run_training.bat', 'manual_training_launcher', 'gold_manual_training_workflow'))
    checks['capture_action_preserved'] = any(t['name'] == 'GOLD-Future-Holdout-Capture-v4' and 'start_gold_future_capture_v4_1.ps1' in t['arguments'] for t in audit['live']['tasks'])
    for name, item in audit['runtime_import_closure'].items():
        checks['runtime:' + name] = item['unchanged'] and not item['trainer_reference'] and not item['model_call'] and sha(ROOT/name) == item['sha256']
    protection = load(run/'protection_checks.json')
    checks['protected_unchanged'] = protection['unchanged'] and protection['before'] == protection['after'] and protection['after']['production_match']
    for name, expected in protection['after']['production'].items():
        checks['production:' + name] = sha(ROOT/name) == expected
    for name, expected in protection['after']['prior_seals'].items():
        checks['prior_seal:' + name] = sha(ROOT/'training_runs'/name/'FINALIZED.json') == expected
    metrics = load(run/'metrics.json')
    checks['partial_not_training_pass'] = metrics['formal_run_status'] == 'PARTIAL' and metrics['training_workflow_ready'] is False
    checks['no_training_outcomes_promotion'] = all(metrics[k] is False for k in ('model_training_executed', 'strategy_outcome_inspected', 'production_changed', 'production_promoted'))
    checks['no_models'] = not (run/'models').exists()
    checks['status_launcher_honest'] = load(run/'status_launcher_smoke_test.json')['status'] == 'PASS'
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    if run.parent != ROOT/'training_runs' or (run/'FINALIZED.json').exists():
        raise ValueError('New repository run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump({'rule': 'Do not retry validation', 'started_at_utc': datetime.now(timezone.utc).isoformat()}, out)
    try:
        checks = validate(run)
        failed = [k for k, value in checks.items() if not value]
        result = {'overall': 'FAIL' if failed else 'PASS', 'checks': checks, 'failed_check_names': failed,
                  'actual_training_approved': False, 'binding_result': 'PARTIAL'}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': ['validator_exception'],
                  'exception': type(error).__name__ + ': ' + str(error), 'actual_training_approved': False}
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent infrastructure validation\n\n'+result['overall']+'\n\nTraining remains unapproved; binding PARTIAL.\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
