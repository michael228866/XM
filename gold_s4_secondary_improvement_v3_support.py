"""Frozen v3 rules; import reads reference metadata, never historical inputs."""
import math
from pathlib import Path

from validate_gold_s4_secondary_improvement_run_v1 import read, sha, require, array_hash
from gold_s4_secondary_improvement_logic_v1 import digest, calibration_split, weights

ROOT = Path(__file__).resolve().parent
CONFIG = 'gold_s4_secondary_improvement_v3_config.json'
EXPERIMENT = 'gold_s4_secondary_improvement_v3'
METRICS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day',
           'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')
REFERENCE = read(ROOT/'gold_s4_a_no_long_htf_reference_v1.json')['metrics']


def configuration(root=ROOT):
    config = read(root/CONFIG)
    for name, expected in config['source_bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Frozen source: '+name)
    ref = read(root/config['reference_binding'])
    space = read(root/config['search_space'])
    require(ref['metrics'] == config['reference_metrics'] == REFERENCE
            and ref['candidate_id'] == 'A_NO_LONG_HTF', 'Exact reference metrics')
    require(digest(ref['features']) == ref['feature_set_sha256'] == config['reference_feature_set_sha256'], 'Reference features')
    require(ref['config_sha256'] == config['reference_config_sha256'], 'Reference configuration')
    require(config['folds'] == ref['folds'] and config['candidate_count'] == len(space['candidates']) <= 40, 'Space count')
    require(len({c['candidate_id'] for c in space['candidates']}) == len(space['candidates']), 'Unique candidates')
    require(space['candidates'][0]['candidate_id'] == 'A_NO_LONG_HTF_CONTROL', 'Control first')
    require(sum(c['family'] == 'H' for c in space['candidates']) <= 12
            and sum(c['family'] == 'HC' for c in space['candidates']) <= 6, 'Family bounds')
    require(len([f for f in space['derived_features'] if f.startswith('X_')]) <= 6, 'Execution feature bound')
    for c in space['candidates']:
        require(c['threshold'] in (.70,.72,.74,.75,.76,.78,.80) and c['seed'] == c['parameters']['random_state'] == 42, 'Frozen threshold/seed')
        require(c['features'] and len(set(c['features'])) == len(c['features']) and
                set(c['features']) <= set(ref['features']+space['derived_features']), 'Reference-centered features')
        require(c['calibration'] in ('RAW','PLATT','ISOTONIC') and c['weighting'] == 'BALANCED'
                and c['positive_weight_multiplier'] in (.975,1.,1.025), 'Frozen fit policy')
        if c['reuse_reference_model']:
            require(c['features'] == ref['features'] and c['parameters'] == ref['parameters']
                    and c['calibration'] == 'RAW' and c['positive_weight_multiplier'] == 1., 'Reuse exact reference model')
    for name, expected in ref['bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Reference provenance: '+name)
    selected = read(root/ref['source_run']/'selected_candidate.json')['candidate']
    require(selected['candidate_id'] == ref['candidate_id'] and selected['pooled'] == ref['metrics']
            and digest(selected['config']) == ref['config_sha256'], 'Immutable selected reference')
    seal = read(root/ref['source_run']/'FINALIZED.json')['file_sha256']
    require(all(seal[m['path']] == m['sha256'] for m in ref['models'])
            and [{k:m[k] for k in ('fold_number','path','sha256')} for m in selected['models']] == ref['models'], 'Sealed model inventory')
    amendment = read(root/ref['amendment_reference'])
    require(amendment['target_run_id'] == ref['source_run_id'] and amendment['effective_validation_status'] == 'PASS'
            and amendment['effective_final_status'] == 'PASS' and amendment['selected_candidate'] == ref['candidate_id'], 'Effective amendment')
    result = read(root/amendment['repair_run']/'target_run_revalidation.json')
    require(result['overall'] == 'PASS' and result['full_validation_completed'] is True
            and result['selected_metrics'] == ref['metrics'] and result['selected_candidate'] == ref['candidate_id']
            and result['candidate_gate'] == ref['candidate_gate'], 'Reference effective validation binding')
    check_production(config, root)
    return config, ref, space


def safe_path(root, name):
    from training_holdout_guard_v1 import check_path
    root = Path(root).resolve()
    path = check_path(root/name, ROOT)
    require(path.is_relative_to(root), 'Path escape')
    return path


def features(base, base_names, wanted):
    """Pointwise causal transforms; all arithmetic ends in float32."""
    import numpy as np
    values = {name: base[:, n] for n, name in enumerate(base_names)}
    htf = np.column_stack([values[k] for k in ('Daily_TREND','H1_TREND','H4_TREND')])
    signs = np.sign(htf)
    values.update(
        HTF_TREND_CONSENSUS=htf.mean(axis=1),
        HTF_DIRECTION_COUNT=(htf > 0).sum(axis=1),
        HTF_ALIGNMENT_RATIO=(signs == np.sign(values['MACD_HIST'])[:, None]).mean(axis=1),
        HTF_DISAGREEMENT_SCORE=((signs-signs.mean(axis=1)[:, None])**2).mean(axis=1),
        X_BODY_QUALITY=values['BODY_PCT']/(np.abs(values['VOLA_RATIO'])+np.float32(1e-6)),
        X_TRIGGER_DISTANCE=values['BIAS_20']/(np.abs(values['VOLA_RATIO'])+np.float32(1e-6)))
    result = np.column_stack([values[name] for name in wanted]).astype(np.float32)
    require(np.isfinite(result).all(), 'Nonfinite candidate features')
    return result


def signals(b0, secondary, threshold):
    return (b0 >= .75) | ((b0 < .75) & (secondary >= threshold))


def event(run, kind, space_hash, candidate_id=None, record=None):
    """Implements the frozen event_hash_spec; lifecycle receipts stay separate."""
    import json
    from datetime import datetime, timezone
    path = Path(run)/'research_events.jsonl'
    previous = json.loads(path.read_text(encoding='utf-8').splitlines()[-1]) if path.exists() else None
    require(kind in {'space_frozen','reference_pass','candidate_begin','candidate_result','research_complete'}, 'Research event type')
    require(previous is None or previous['event'] != 'research_complete', 'Research chain already frozen')
    require((previous is None) == (kind == 'space_frozen'), 'Single chain root')
    row = dict(sequence=previous['sequence']+1 if previous else 0, event=kind,
        at_utc=datetime.now(timezone.utc).isoformat(), space_sha256=space_hash,
        candidate_id=candidate_id, stage=None, record_sha256=digest(record) if record is not None else None,
        previous_sha256=previous['event_sha256'] if previous else None)
    require(previous is None or row['at_utc'] >= previous['at_utc'], 'Clock moved backwards')
    row['event_sha256'] = digest(row)
    with path.open('a', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(row, allow_nan=False)+'\n')
    return row


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
        return (-p['realized_win_rate'], -p['trades_per_day'], -p['profit_factor'], -p['stress_pf'],
                -p['mean_r'], -p['pnl_r'], abs(p['max_drawdown_r']), r['candidate_id'])
    frontier.sort(key=rank)
    choices = [r for r in frontier if r['candidate_id'] != 'A_NO_LONG_HTF_CONTROL' and gate(r['pooled'], r['fold_metrics'])['interesting']]
    return dict(pareto_frontier=[r['candidate_id'] for r in frontier],
                economic_annotations={r['candidate_id']: {k:r['pooled'][k] for k in ('profit_factor','mean_r','pnl_r','max_drawdown_r','stress_pf')} for r in frontier},
                selected_candidate=choices[0]['candidate_id'] if choices else None,
                research_result='IMPROVEMENT_FOUND' if choices else 'NO_IMPROVEMENT_FOUND',
                tie_break=[dict(candidate_id=r['candidate_id'], ordered_key=list(rank(r))) for r in choices])


def check_control(pooled, reference=REFERENCE):
    require(pooled.keys() == reference.keys() and all(isinstance(pooled[k], (int, float))
            and math.isclose(pooled[k], v, rel_tol=0, abs_tol=1e-12) for k, v in reference.items()),
            'A_NO_LONG_HTF_REFERENCE_CONTROL_MISMATCH')


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
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import numpy as np
    import pandas as pd
    source = ROOT/ref['evidence_source_run']
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
