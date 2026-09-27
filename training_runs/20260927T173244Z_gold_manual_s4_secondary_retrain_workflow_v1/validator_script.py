"""Independent infrastructure certification; no training implementation imports."""
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(run):
    checks = {}
    m = read(run/'manifest.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0]
    checks['source_pushed_clean'] = m['git_commit'] == m['source_commit'] == head == remote and m['git_dirty'] is False
    for item in m['input_snapshots']:
        checks['snapshot:'+item['source_path']] = sha(ROOT/item['source_path']) == item['sha256'] == sha(run/item['path'])
    c = read(run/'approved_training_config.json')
    ds = read(ROOT/c['discovery_spec'])
    trace = read(run/'s4_provenance_trace.json')
    checks['discovery_provenance'] = trace['status'] == 'PASS' and trace['discovery'] == c['discovery_run'] and trace['secondary_model_hashes_match_confirmation'] is True
    checks['discovery_snapshot'] = trace['discovery_source_sha256'] == sha(ROOT/c['discovery_run']/'training_script.py')
    checks['not_confirmation_trainer'] = c['discovery_run'] != c['confirmation_run']
    checks['symbol'] = c['training_symbol'] == 'GOLD#' and c['training_source_id'] == 'LEGACY_XM_GOLD_S4_20260915'
    checks['feature_order'] = c['feature_list'] == ds['features'] and len(c['feature_list']) == 31
    checks['feature_list_hash'] = hashlib.sha256(json.dumps(c['feature_list'], separators=(',', ':')).encode()).hexdigest() == c['feature_list_sha256']
    checks['feature_pipeline_hash'] = hashlib.sha256(json.dumps(c['feature_pipeline'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == c['feature_pipeline_sha256']
    for name, expected in {**c['feature_pipeline'], **c['source_bindings'], **c['archive_seals']}.items():
        checks['binding:'+name] = sha(ROOT/name) == expected
    checks['label_hash'] = sha(ROOT/c['label_pipeline']) == c['label_pipeline_sha256']
    checks['frozen_settings'] = c['hyperparameters'] == ds['parameters'] and c['fold_definition'] == ds['folds'] and c['random_seed'] == ds['seed'] == 42 and c['secondary_threshold'] == c['b0_primary_threshold'] == .75
    policy = read(run/'approved_training_spec.json')
    checks['data_policy_hash'] = sha(run/'approved_training_spec.json') == c['data_policy_sha256']
    checks['cutoff_preserved'] = policy['historical_training_cutoff'] == c['raw_data_cutoff'] == '2026-05-08T23:57:00' and policy['new_retrain_dataset_allowed'] is False
    checks['holdout_locked'] = policy['future_holdout_exclusion'] == {'path': 'future_holdout/gold_s4_v4', 'start': '2026-09-25T19:24:00+00:00', 'training_access': False}
    decision = read(run/'training_dataset_decision.json')
    checks['legacy_decision'] = decision['status'] == 'PASS' and len(decision['datasets']) == 21 and decision['model_or_strategy_computation'] is False
    parent = read(ROOT/'training_runs'/ds['c1_run']/'manifest.json')
    original = {Path(x['path']).name: x['sha256'] for x in parent['data']['source_files']}
    for expected, actual in zip(c['required_datasets'], decision['datasets']):
        checks['legacy:'+expected['filename']] = (actual['classification'] == 'EXACT_PROVENANCE_TRAINING_SOURCE' and actual['status'] == 'PASS'
            and actual['row_count'] > 0 and expected['sha256'] == actual['sha256'] == original[expected['filename']] == sha(ROOT/expected['filename'])
            and expected['first_timestamp'] == actual['first_timestamp'] and expected['last_timestamp'] == actual['last_timestamp'] <= c['raw_data_cutoff'])
    test = read(run/'launcher_smoke_test.json')
    checks['synthetic_tests'] = test['overall'] == 'PASS' and test['test_count'] >= 70 and all(test['checks'].values())
    checks['no_real_fits'] = test['model_training_executed'] is False and test['fake_fit_calls'] == 3 and test['strategy_outcome_inspected'] is False
    checks['imports_only'] = read(run/'dependency_import_smoke.json')['status'] == 'PASS'
    tree = ast.parse((ROOT/'gold_manual_s4_secondary_retrain_v1.py').read_bytes())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    checks['reuse_frozen_training_functions'] = {'reconstruct', 'training_subset', 'fit_secondary', 'evaluate'}.issubset(attrs) and 'fit' not in attrs
    checks['no_formal_runner_reuse'] = 'formal_run' not in attrs
    checks['no_B0_fit'] = not any(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'b0_model' and n.func.attr == 'fit' for n in calls)
    source = (ROOT/'gold_manual_s4_secondary_retrain_v1.py').read_text(encoding='utf-8')
    checks['manual_receipt_first'] = source.index('consume_receipt(token)') < source.index('verify_s4_approval()') < source.index('data_dir, dataset = prepare(config)') < source.index('history.create_run(')
    checks['no_auto_promotion'] = "'production_promoted': False" in source and "model.save_model(path)" in source and "path = model_path(run, name)" in source
    datatree = ast.parse((ROOT/'gold_manual_s4_training_data_v1.py').read_bytes())
    mt5calls = {n.func.attr for n in ast.walk(datatree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'mt5'}
    checks['mt5_read_only'] = mt5calls == {'initialize', 'account_info', 'symbol_info', 'symbol_select', 'copy_rates_range', 'shutdown'}
    datatext = (ROOT/'gold_manual_s4_training_data_v1.py').read_text(encoding='utf-8')
    checks['fetch_byte_identity_required'] = "hashlib.sha256(raw).hexdigest() != item['sha256']" in datatext and 'NEW_RETRAIN_DATASET' in datatext
    checks['sanitized_identity'] = "('company', 'server', 'trade_mode')" in datatext and '_asdict' not in datatext
    smoke = read(run/'mt5_fetch_smoke_test.json')
    checks['bounded_historical_smoke'] = smoke['request'] == {'symbol': 'GOLD#', 'timeframe': 'H1', 'first_timestamp': '2018-01-02T01:00:00', 'last_timestamp': '2018-01-02T02:00:00'} and smoke['training_eligible'] is False
    if smoke['status'] == 'PASS':
        checks['smoke_hash'] = sha(run/smoke['artifact']) == smoke['sha256']
    vtree = ast.parse((ROOT/'validate_gold_manual_s4_training_run_v1.py').read_bytes())
    imports = [n.module or '' for n in ast.walk(vtree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(vtree) if isinstance(n, ast.Import) for a in n.names]
    checks['training_validator_independent'] = not any('retrain' in name or 'classifier' in name for name in imports)
    checks['training_validator_no_fit'] = not any(isinstance(n, ast.Attribute) and n.attr == 'fit' for n in ast.walk(vtree))
    checks['adapter'] = read(run/'output_adapter_review.json')['status'] == 'PASS'
    checks['guard'] = read(run/'holdout_guard_review.json')['status'] == 'PASS' and test['checks']['write_guard_production']
    audit = read(run/'auto_training_audit.json')
    checks['no_auto_training'] = audit['status'] == 'PASS' and audit['auto_training_disabled'] is True
    checks['capture_preserved'] = all(x['unchanged'] and not x['model_call'] and not x['trainer_reference'] for x in audit['runtime_import_closure'].values())
    for name, x in audit['runtime_import_closure'].items():
        checks['runtime:'+name] = sha(ROOT/name) == x['sha256']
    protection = read(run/'protection_checks.json')
    checks['protection'] = protection['unchanged'] and protection['before'] == protection['after'] and protection['after']['production_match']
    for name, expected in protection['after']['production'].items():
        checks['production:'+name] = sha(ROOT/name) == expected
    for name, expected in protection['after']['prior_seals'].items():
        checks['prior_seal:'+name] = sha(ROOT/'training_runs'/name/'FINALIZED.json') == expected
    metrics = read(run/'metrics.json')
    checks['infrastructure_only'] = metrics['formal_run_status'] == 'PASS' and all(metrics[k] is False for k in ('model_training_executed', 'strategy_outcome_inspected', 'production_changed', 'production_promoted'))
    checks['approval_not_premature'] = metrics['approval_commit_pending'] is True and not (ROOT/'gold_manual_s4_approval_v1.json').exists()
    checks['no_model_created'] = not (run/'models').exists()
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    if run.parent != ROOT/'training_runs' or (run/'FINALIZED.json').exists():
        raise ValueError('New infrastructure run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump({'rule': 'Do not retry validation', 'started_at_utc': datetime.now(timezone.utc).isoformat()}, out)
    try:
        checks = validate(run)
        failed = [name for name, value in checks.items() if not value]
        result = {'overall': 'FAIL' if failed else 'PASS', 'checks': checks, 'failed_check_names': failed,
                  'manual_s4_binding_approved': not failed, 'model_training_executed': False}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': [type(error).__name__+': '+str(error)], 'manual_s4_binding_approved': False}
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent manual S4 infrastructure certification\n\n'+result['overall']+'\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
