"""Independent infrastructure certification; never imports a research trainer."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(run):
    checks = {}
    manifest = read(run/'manifest.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0]
    checks['pushed_source'] = manifest['git_commit'] == manifest['source_commit'] == head == remote and manifest['git_dirty'] is False
    for item in manifest['input_snapshots']:
        checks['source:'+item['source_path']] = sha(ROOT/item['source_path']) == sha(run/item['path']) == item['sha256']
    spec = read(run/'execution_spec.json')
    plan = read(run/'research_plan.json')
    checks['search_space_exact'] = read(run/'predeclared_search_space.json') == plan['candidates'] == read(ROOT/'gold_s4_secondary_improvement_config_v1.json')['candidates']
    checks['16_candidates_7_families'] = len(plan['candidates']) == 16 <= 100 and {x['family'] for x in plan['candidates']} == set(range(7))
    checks['fixed_folds_and_stages'] = plan['stage1_fold_numbers'] == [1, 2] and plan['stage2_fold_numbers'] == [1, 2, 3] and plan['folds'] == read(ROOT/plan['reference_config'])['fold_definition']
    checks['six_pointwise_features'] = len(plan['feature_extensions']) == 6 <= 10
    checks['reference_sha'] = plan['reference_model_sha256'] == 'e60e6808aabecc94d24d563a1e1c9b5151e3dbdb3f6362367a9d7c334a145141'
    checks['fixed_tolerance'] = plan['numeric_tolerance'] == 1e-12
    for name, expected in plan['frozen_hashes'].items():
        checks['frozen:'+name] = sha(ROOT/name) == expected
    reference = read(ROOT/plan['reference_run']/'combined_result.json')
    checks['reference_metrics'] = all(reference[k] == v for k, v in plan['reference_metrics'].items()) and reference['final_status'] == reference['validator_status'] == 'PASS'
    tests = read(run/'self_tests.json')
    for name, value in tests.items():
        checks['suite:'+name] = value['overall'] == 'PASS' and value['exit_code'] == 0 and value['model_training_executed'] is False and all(value['checks'].values())
        for key, passed in value['checks'].items():
            checks['test:'+name+':'+key] = passed is True
    improvement = tests['test_gold_s4_secondary_improvement_v1.py']['checks']
    for name, keys in spec['test_groups'].items():
        result = read(run/(name+'.json'))
        checks['group:'+name] = result['status'] == 'PASS' and result['checks'] == {k: improvement[k] for k in keys} and all(result['checks'].values())
    protected = read(run/'protection_checks.json')
    checks['protected_unchanged'] = protected['unchanged'] and protected['before'] == protected['after'] and protected['after']['production_match']
    for name, expected in protected['before']['production'].items():
        checks['production:'+name] = sha(ROOT/name) == expected
    for name, expected in protected['before']['prior_seals'].items():
        checks['prior_seal:'+name] = sha(ROOT/'training_runs'/name/'FINALIZED.json') == expected
    audit = read(run/'auto_training_audit.json')
    checks['no_automatic_training'] = audit['status'] == 'PASS' and audit['auto_training_disabled'] is True
    for name, value in audit['runtime_import_closure'].items():
        checks['capture:'+name] = sha(ROOT/name) == value['sha256'] and value['unchanged'] and not value['model_call'] and not value['trainer_reference']
    source = (ROOT/'gold_s4_secondary_improvement_v1.py').read_text(encoding='utf-8')
    checks['receipt_before_research'] = source.index('consume_receipt(token)') < source.index('verify_approval()') < source.index('data_dir, dataset = prepare(base)') < source.index('space_hash = freeze_space(run, config)') < source.index('decision, chosen, inventory = research(')
    checks['validator_before_finalization'] = source.index("str(ROOT/'validate_gold_s4_secondary_improvement_run_v1.py'), str(run)") < source.index('history.finalize_run(')
    tree = ast.parse((ROOT/'validate_gold_s4_secondary_improvement_run_v1.py').read_bytes())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    checks['independent_validator'] = not any('secondary_improvement' in x or 'secondary_retrain' in x or 'independent_secondary_classifier' in x for x in imports)
    checks['validator_no_fit'] = not any(isinstance(n, ast.Attribute) and n.attr == 'fit' for n in ast.walk(tree))
    checks['independent_replay'] = any(isinstance(n, ast.Attribute) and n.attr == 'simulate' for n in ast.walk(tree))
    metrics = read(run/'metrics.json')
    checks['infrastructure_only'] = metrics['formal_run_status'] == 'PASS' and all(metrics[k] is False for k in ('model_training_executed', 'strategy_replay_executed', 'holdout_used', 'production_changed', 'production_promoted'))
    checks['approval_pending'] = metrics['approval_commit_pending'] is True and not (ROOT/'gold_s4_improvement_approval_v1.json').exists()
    checks['no_real_model_created'] = not (run/'models').exists()
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    if run.parent != ROOT/'training_runs' or (run/'FINALIZED.json').exists():
        raise ValueError('New infrastructure run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump({'rule': 'Do not retry validation', 'at_utc': datetime.now(timezone.utc).isoformat()}, out)
    try:
        checks = validate(run)
        failed = [name for name, passed in checks.items() if not passed]
        result = {'overall': 'FAIL' if failed else 'PASS', 'checks': checks, 'failed_check_names': failed}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': [type(error).__name__+': '+str(error)]}
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent improvement infrastructure certification\n\n'+result['overall']+'\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
