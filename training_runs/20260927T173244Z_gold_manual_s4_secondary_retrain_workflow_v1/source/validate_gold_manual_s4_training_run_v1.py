"""Independent exact-reproduction validator; no trainer imports and no fitting."""
import csv
import hashlib
import json
import math
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


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_path(run, name, models=False):
    run = Path(run).resolve()
    path = (run/name).resolve()
    base = run/'models' if models else run
    if not path.is_relative_to(base):
        raise PermissionError('Artifact path escapes run')
    return path


def equal(left, right):
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(equal(left[k], v) for k, v in right.items())
    if isinstance(right, list):
        return isinstance(left, list) and len(left) == len(right) and all(equal(x, y) for x, y in zip(left, right))
    if isinstance(right, float):
        return isinstance(left, (float, int)) and math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def validate(run):
    """Real model/output checks, called only after a USER-started training run."""
    import numpy as np
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise ValueError(name)
    m = read(run/'manifest.json')
    c = read(run/'approved_training_config.json')
    candidate = read(run/'candidate_model_manifest.json')
    check('manual_run', m['experiment_name'] == 'gold_manual_s4_secondary_retrain_v1'
          and m['manual_start'] is True and m['started_by'] == 'USER_LAUNCHER' and m['git_dirty'] is False)
    for name, saved in [('gold_manual_s4_secondary_retrain_config_v1.json', 'approved_training_config.json'),
                        ('gold_manual_s4_secondary_retrain_v1.py', 'training_script.py'),
                        ('validate_gold_manual_s4_training_run_v1.py', 'validator_script.py')]:
        raw = subprocess.check_output(['git', 'show', m['git_commit']+':'+name], cwd=ROOT)
        check('committed:'+name, raw == (run/saved).read_bytes())
    for name, expected in {**c['source_bindings'], **c['archive_seals']}.items():
        check('source:'+name, sha(ROOT/name) == expected)
    check('feature_list', len(c['feature_list']) == 31 and hashlib.sha256(json.dumps(c['feature_list'], separators=(',', ':')).encode()).hexdigest() == c['feature_list_sha256'])
    check('feature_pipeline', canonical_hash(c['feature_pipeline']) == c['feature_pipeline_sha256'])
    check('label_pipeline', sha(ROOT/c['label_pipeline']) == c['label_pipeline_sha256'])
    ds = read(ROOT/c['discovery_spec'])
    check('frozen_config', c['hyperparameters'] == ds['parameters'] and c['fold_definition'] == ds['folds']
          and c['random_seed'] == 42 and c['secondary_threshold'] == c['b0_primary_threshold'] == .75)
    check('symbol_and_mode', c['training_symbol'] == 'GOLD#' and c['dataset_mode'] == 'REPRODUCTION_DATASET')
    dataset = read(run/'training_dataset_manifest.json')
    check('dataset_manifest_identity', sha(run/'training_dataset_manifest.json') == candidate['training_dataset_manifest_sha256'])
    check('dataset_inventory', len(dataset['datasets']) == len(c['required_datasets']) == 21 and dataset['holdout_input'] is False)
    for actual, expected in zip(dataset['datasets'], c['required_datasets']):
        path = Path(actual['path']).resolve()
        check('dataset:'+expected['filename'], path.is_relative_to(ROOT/'historical_training_data')
              and path.name == expected['filename'] and actual['symbol'] == 'GOLD#'
              and actual['timeframe'] == expected['timeframe'] and actual['training_eligible'] is True
              and sha(path) == actual['sha256'] == expected['sha256']
              and expected['last_timestamp'] <= c['raw_data_cutoff'] < '2026-09-25T00:00:00')
    check('candidate_bindings', candidate['training_script_sha256'] == sha(run/'training_script.py')
          and candidate['config_sha256'] == sha(run/'approved_training_config.json')
          and all(candidate[k] == c[k] for k in ('feature_list_sha256', 'feature_pipeline_sha256', 'label_pipeline_sha256', 'training_symbol', 'model_type', 'hyperparameters'))
          and candidate['training_range'] == [c['training_start'], c['training_end']]
          and candidate['seed'] == 42 and candidate['folds'] == c['fold_definition']
          and candidate['b0_conditioning_rule'] == c['b0_conditioning_rule'])
    check('no_promotion', candidate['production_promoted'] is False and candidate['production_overwrite_allowed'] is False)
    inventory = read(run/'model_inventory.json')
    check('three_models', len(inventory) == 3 and candidate['fold_models'] == inventory)
    original = ROOT/c['discovery_run']
    originals = read(original/'model_inventory.json')
    for item, expected in zip(inventory, originals):
        path = safe_path(run, item['path'], models=True)
        check('model:'+item['fold'], item['fold'] == expected['fold'] and sha(path) == item['sha256'] == expected['sha256'])
        check('training_identity:'+item['fold'], all(item[k] == expected[k] for k in
              ('parameters', 'features', 'seed', 'rounds', 'train_rows', 'train_indices_sha256', 'x_sha256', 'target_sha256', 'weights_sha256')))
        learner = read(path)['learner']
        check('model_structure:'+item['fold'], learner['feature_names'] == c['feature_list'] and len(learner['gradient_booster']['model']['trees']) == 220)
    check('model_set', {p.relative_to(run).as_posix() for p in (run/'models').iterdir()} == {i['path'] for i in inventory})
    check('primary_candidate_identity', candidate['model_path'] == inventory[-1]['path'] and candidate['model_sha256'] == inventory[-1]['sha256'])
    with np.load(run/'secondary_evidence.npz', allow_pickle=False) as evidence, np.load(original/'secondary_evidence.npz', allow_pickle=False) as reference:
        expected_keys = {f'fold{n}_'+suffix for n in (1, 2, 3) for suffix in
                         ('train_indices', 'score_indices', 'subset_indices', 'b0_train', 'b0_score',
                          'subset_target', 'train_ns', 'score_ns', 'secondary_score', 'sample_weight')}
        expected_keys.update('S4_SECONDARY_P075_'+suffix for suffix in ('primary', 'secondary', 'union'))
        check('complete_evidence_inventory', set(evidence.files) == expected_keys)
        for key in evidence.files:
            check('evidence:'+key, key in reference.files and np.array_equal(evidence[key], reference[key]))
        for number in (1, 2, 3):
            k = f'fold{number}'
            train, score = evidence[k+'_train_indices'], evidence[k+'_score_indices']
            check('conditioning:'+k, np.array_equal(evidence[k+'_subset_indices'], train[evidence[k+'_b0_train'] < .75])
                  and not np.intersect1d(train, score).size and evidence[k+'_train_ns'].max() < evidence[k+'_score_ns'].min())
        b0 = np.concatenate([evidence[f'fold{n}_b0_score'] for n in (1, 2, 3)])
        sec = np.concatenate([evidence[f'fold{n}_secondary_score'] for n in (1, 2, 3)])
        check('gate', np.array_equal(evidence['S4_SECONDARY_P075_union'], (b0 >= .75) | ((b0 < .75) & (sec >= .75))))
    historical = read(run/'historical_metrics.json')
    check('historical_metrics_exact', equal(historical, read(original/'metrics.json')['results']['S4_SECONDARY_P075']))
    with (original/'trade_ledger.csv').open(encoding='utf-8', newline='') as stream:
        expected_trades = [{k: v for k, v in row.items() if k != 'candidate_id'} for row in csv.DictReader(stream) if row['candidate_id'] == 'S4_SECONDARY_P075']
    with (run/'trade_ledger.csv').open(encoding='utf-8', newline='') as stream:
        trades = list(csv.DictReader(stream))
    check('ledger_exact', trades == expected_trades)
    check('ledger_cutoff', all('2018-01-01' <= row['entry_time_api'] < '2025-01-01' for row in trades))
    metrics = read(run/'metrics.json')
    pooled = historical['metrics'][-1]
    mapping = {'trades': 'trades', 'wins': 'wins', 'losses': 'losses', 'realized_win_rate': 'realized_wr',
               'trades_per_day': 'trades_per_day', 'profit_factor': 'pf', 'mean_r': 'mean_r',
               'pnl_r': 'pnl_r', 'max_drawdown_r': 'max_dd_r', 'stress_pf': 'cost_stress_pf'}
    check('display_metrics', all(equal(metrics[k], pooled[v]) for k, v in mapping.items()))
    check('production_unchanged', read(run/'production_before.json') == read(run/'production_after.json') == c['protected_sha256']
          and all(sha(ROOT/n) == v for n, v in c['protected_sha256'].items()))
    return checks


def main():
    run = Path(sys.argv[1]).resolve()
    if run.parent != ROOT/'training_runs' or (run/'FINALIZED.json').exists():
        raise ValueError('New training run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'rule': 'Do not retry validation', 'started_at_utc': datetime.now(timezone.utc).isoformat()}, stream)
    try:
        checks = validate(run)
        result = {'overall': 'PASS', 'checks': checks, 'failed_check_names': [], 'production_promoted': False}
    except Exception as error:
        result = {'overall': 'FAIL', 'failed_check_names': [str(error)], 'error_type': type(error).__name__, 'production_promoted': False}
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent S4 reproduction validator\n\n'+result['overall']+'\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
