"""Independent infrastructure validator; never imports or executes the trainer."""
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
    for name, expected in spec['frozen_hashes'].items():
        checks['frozen:'+name] = sha(ROOT/name) == expected
    tests = read(run/'self_tests.json')
    for name, value in tests.items():
        checks['suite:'+name] = value['overall'] == 'PASS' and value['exit_code'] == 0 and value['model_training_executed'] is False and all(value['checks'].values())
        for key, passed in value['checks'].items():
            checks['test:'+name+':'+key] = passed is True
    integration = tests['test_gold_manual_s4_train_validate_oneclick_v1.py']['checks']
    for name, keys in spec['test_groups'].items():
        result = read(run/(name+'.json'))
        checks['group:'+name] = result['status'] == 'PASS' and result['checks'] == {k: integration[k] for k in keys} and all(result['checks'].values())
    protected = read(run/'protection_checks.json')
    checks['protection'] = protected['unchanged'] and protected['before'] == protected['after'] and protected['after']['production_match']
    for name, value in protected['before']['production'].items():
        checks['production:'+name] = sha(ROOT/name) == value
    for name, value in protected['before']['prior_seals'].items():
        checks['prior_seal:'+name] = sha(ROOT/'training_runs'/name/'FINALIZED.json') == value
    audit = read(run/'auto_training_audit.json')
    checks['auto_training_disabled'] = audit['status'] == 'PASS' and audit['auto_training_disabled'] is True
    for name, value in audit['runtime_import_closure'].items():
        checks['capture:'+name] = sha(ROOT/name) == value['sha256'] and value['unchanged'] and not value['model_call'] and not value['trainer_reference']
    source = (ROOT/'gold_manual_s4_secondary_retrain_v1.py').read_text(encoding='utf-8')
    checks['exact_handoff_order'] = source.index('training_handoff(run, candidate)') < source.index('validation = validate_exact_run(run)') < source.index("history.finalize_run(run, 'research_only')")
    helper = (ROOT/'gold_manual_s4_train_validate_v1.py').read_text(encoding='utf-8')
    tree = ast.parse(helper)
    validation = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_exact_run')
    checks['no_latest_run_validation'] = not any(isinstance(n, ast.Attribute) and n.attr in ('glob', 'rglob') for n in ast.walk(validation))
    checks['validator_exact_run_argument'] = 'str(run)' in ast.unparse(validation) and 'validate_gold_manual_s4_training_run_v1.py' in ast.unparse(validation)
    vtree = ast.parse((ROOT/'validate_gold_manual_s4_training_run_v1.py').read_bytes())
    imports = [n.module or '' for n in ast.walk(vtree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(vtree) if isinstance(n, ast.Import) for a in n.names]
    checks['independent_validator'] = not any('retrain' in x or 'train_validate' in x for x in imports)
    checks['no_validator_fit_predict'] = not any(isinstance(n, ast.Attribute) and n.attr in ('fit', 'predict', 'predict_proba') for n in ast.walk(vtree))
    metrics = read(run/'metrics.json')
    checks['no_training_holdout_promotion'] = metrics['formal_run_status'] == 'PASS' and all(metrics[k] is False for k in ('model_training_executed', 'holdout_used', 'production_changed', 'production_promoted'))
    checks['no_model_artifact'] = not (run/'models').exists()
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    if run.parent != ROOT/'training_runs' or (run/'FINALIZED.json').exists():
        raise ValueError('New infrastructure run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump({'rule': 'Do not retry validation', 'started_at_utc': datetime.now(timezone.utc).isoformat()}, out)
    try:
        checks = validate(run)
        failed = [name for name, passed in checks.items() if not passed]
        result = {'overall': 'FAIL' if failed else 'PASS', 'checks': checks, 'failed_check_names': failed}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': [type(error).__name__+': '+str(error)]}
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent one-click infrastructure validation\n\n'+result['overall']+'\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
