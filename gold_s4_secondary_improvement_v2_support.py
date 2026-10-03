"""Frozen v2 rules and historical input reader. Import performs no I/O."""
import math
from pathlib import Path

from validate_gold_s4_secondary_improvement_run_v1 import read, sha, require, array_hash
from gold_s4_secondary_improvement_logic_v1 import digest, features, calibration_split, weights

ROOT = Path(__file__).resolve().parent
CONFIG = 'gold_s4_secondary_improvement_v2_config.json'
EXPERIMENT = 'gold_s4_secondary_improvement_v2'
METRICS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day',
           'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')
REFERENCE = dict(zip(METRICS, (768, 438, 330, .5703125, .30035197497066873,
    .8436457240683531, -.06800170013341929, -52.22530570246601,
    -56.4297416185602, .8016509741596182)))


def configuration(root=ROOT):
    config = read(root/CONFIG)
    for name, expected in config['source_bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Frozen source: '+name)
    ref = read(root/config['reference_binding'])
    space = read(root/config['search_space'])
    require(ref['metrics'] == REFERENCE and ref['candidate_id'] == 'F5_NO_REDUNDANT_HTF'
            and ref['model_sha256'] == '617e5724e2405e7423ec61b4104c4272d9094ed1fe29d6da95735326baf18753', 'F5 reference')
    require(config['folds'] == ref['folds'] and config['candidate_count'] == len(space['candidates']) <= 60, 'Space count')
    require(len({c['candidate_id'] for c in space['candidates']}) == len(space['candidates']), 'Unique candidates')
    require(space['candidates'][0]['candidate_id'] == 'F5_CONTROL', 'Control first')
    require(len(space['derived_features']) <= 6, 'Feature extension bound')
    for c in space['candidates']:
        require(c['threshold'] == .75 and c['seed'] == 42 and c['parameters']['random_state'] == 42, 'Frozen threshold/seed')
        require(c['features'] and len(set(c['features'])) == len(c['features']) and
                set(c['features']) <= set(ref['features'] + space['derived_features']), 'F5 centered features')
    require(sum(c['family'] == 'A' for c in space['candidates']) <= 12
            and sum(c['family'] == 'D' for c in space['candidates']) <= 20, 'Family bounds')
    for name, expected in ref['bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Reference provenance: '+name)
    amendment = read(root/ref['source_run']/'validation_amendment_v1.json')
    require(amendment['target_run_id'] == ref['source_run_id'] and amendment['effective_validation_status'] == 'PASS'
            and amendment['candidate_model_sha256'] == ref['model_sha256'], 'Effective amendment')
    check_production(config, root)
    return config, ref, space


def safe_path(root, name):
    from training_holdout_guard_v1 import check_path
    root = Path(root).resolve()
    path = check_path(root/name, ROOT)
    require(path.is_relative_to(root), 'Path escape')
    return path


def check_production(config, root=ROOT):
    require(all(sha(root/n) == h for n, h in config['protected_sha256'].items()), 'Production changed')


def economic(p):
    if p['profit_factor'] > 1 and p['mean_r'] > 0 and p['pnl_r'] > 0:
        return 'POSITIVE_EXPECTANCY'
    if p['profit_factor'] < 1 and p['mean_r'] < 0 and p['pnl_r'] < 0:
        return 'NEGATIVE_EXPECTANCY'
    return 'NEAR_BREAK_EVEN'


def gate(p, folds):
    valid = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    safe = (all(valid(p.get(k)) for k in METRICS) and p['profit_factor'] >= .8
            and p['mean_r'] >= -.10690539315994348 and p['stress_pf'] >= .75 and len(folds) == 3
            and all(all(valid(f.get(k)) for k in ('trades', 'realized_win_rate', 'profit_factor'))
                    and f['trades'] >= 10 and f['realized_win_rate'] >= .45 and f['profit_factor'] >= .7 for f in folds))
    improved = safe and p['realized_win_rate'] > REFERENCE['realized_win_rate'] and p['trades_per_day'] > REFERENCE['trades_per_day']
    label = 'NONE'
    if improved:
        label = 'TARGET' if p['realized_win_rate'] >= .6 and p['trades_per_day'] >= .5 else (
            'STRONG' if p['realized_win_rate'] >= .58 and p['trades_per_day'] >= .4 else 'INTERESTING')
    return dict(safety_pass=safe, interesting=improved, gate=label)


def select(rows):
    eligible = [r for r in rows if gate(r['pooled'], r['fold_metrics'])['safety_pass']]
    def dominates(a, b):
        return all(a[k] >= b[k] for k in ('realized_win_rate', 'trades_per_day')) and any(
            a[k] > b[k] for k in ('realized_win_rate', 'trades_per_day'))
    frontier = [r for r in eligible if not any(dominates(o['pooled'], r['pooled']) for o in eligible)]
    def rank(r):
        p = r['pooled']
        return (-p['realized_win_rate'], -p['trades_per_day'], -p['stress_pf'], -p['profit_factor'],
                -p['mean_r'], abs(p['max_drawdown_r']), r['candidate_id'])
    frontier.sort(key=rank)
    choices = [r for r in frontier if r['candidate_id'] != 'F5_CONTROL' and gate(r['pooled'], r['fold_metrics'])['interesting']]
    return dict(pareto_frontier=[r['candidate_id'] for r in frontier],
                selected_candidate=choices[0]['candidate_id'] if choices else None,
                research_result='IMPROVEMENT_FOUND' if choices else 'NO_IMPROVEMENT_FOUND',
                tie_break=[dict(candidate_id=r['candidate_id'], ordered_key=list(rank(r))) for r in choices])


def check_control(pooled, reference=REFERENCE):
    require(pooled.keys() == reference.keys() and all(isinstance(pooled[k], (int, float))
            and math.isclose(pooled[k], v, rel_tol=0, abs_tol=1e-12) for k, v in reference.items()),
            'F5_REFERENCE_CONTROL_MISMATCH')


def sample_weights(y, c):
    import numpy as np
    require(np.array_equal(np.unique(y), [0, 1]), 'Both training classes required')
    return np.ones(len(y)) if c['weighting'] == 'NONE' else weights(y, c['positive_weight_multiplier'])


def split(data, c, config, n):
    import numpy as np
    import pandas as pd
    subset = np.flatnonzero(data['b0_train'] < .75)
    if c['calibration'] == 'RAW':
        return subset, np.array([], dtype=np.int64), None
    cutoff = (pd.Timestamp(config['folds'][n-1][1])-pd.DateOffset(months=config['calibration_months'])).value
    fit, cal = calibration_split(data['train_ns'], data['maturity'], data['legacy_maturity'], subset, cutoff)
    require(len(cal) >= config['minimum_calibration_samples'] and
            np.bincount(data['target'][cal], minlength=2).min() >= config['minimum_calibration_class_samples'],
            'INSUFFICIENT_CALIBRATION_SAMPLES')
    return fit, cal, cutoff


def load_evidence(config, ref):
    """Only reachable inside the USER session, never from fixture certification."""
    from gold_s4_secondary_improvement_v2_launcher import require_session
    require_session()
    import numpy as np
    import pandas as pd
    source = ROOT/ref['source_run']
    seal = read(source/'FINALIZED.json')['file_sha256']
    def sealed(name):
        path = safe_path(source, name)
        require(sha(path) == seal[name], 'Sealed input: '+name)
        return path
    spec = read(ROOT/config['discovery_spec'])
    evidence = {}
    for item in read(sealed('evidence_index.json')):
        n = item['fold_number']
        require(n not in evidence and n in (1, 2, 3), 'Evidence fold inventory')
        with np.load(sealed(item['arrays_path']), allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        for part in ('train', 'score'):
            pieces = []
            for chunk in item[part+'_chunks']:
                with np.load(sealed(chunk['path']), allow_pickle=False) as values:
                    require(len(values['values']) == chunk['rows'], 'Chunk rows')
                    pieces.append(values['values'].copy())
            data[part+'_x'] = np.concatenate(pieces)
            require(array_hash(data[part+'_x']) == spec['feature_matrix_sha256'][f'fold{n}_{part}'], 'Certified features')
        require(array_hash(data['target']) == spec['c1_target_sha256'][f'fold{n}_train'], 'Certified labels')
        with np.load(sealed('secondary_evidence.npz'), allow_pickle=False) as original:
            for key in ('train_indices', 'score_indices', 'train_ns', 'score_ns', 'b0_train', 'b0_score'):
                require(np.array_equal(data[key], original[f'fold{n}_'+key]), 'Original chronology:'+key)
        start, end = (pd.Timestamp(t).value for t in config['folds'][n-1][1:])
        require(data['train_ns'].max() < start <= data['score_ns'].min() and data['score_ns'].max() < end
                and np.all(data['maturity'] < start) and np.all(data['legacy_maturity'] < start)
                and np.all(data['maturity'] >= data['train_ns']), 'Fold and label isolation')
        with np.load(sealed(item['price_path']), allow_pickle=False) as prices:
            data['price'] = pd.DataFrame({k: pd.to_datetime(prices[k]) if k == 'TIME_DT' else prices[k] for k in prices.files})
        require(np.array_equal(data['price'].TIME_DT.to_numpy(dtype='datetime64[ns]').astype(np.int64), data['score_ns']), 'Price chronology')
        evidence[n] = data
    require(set(evidence) == {1, 2, 3}, 'All folds required')
    return evidence
