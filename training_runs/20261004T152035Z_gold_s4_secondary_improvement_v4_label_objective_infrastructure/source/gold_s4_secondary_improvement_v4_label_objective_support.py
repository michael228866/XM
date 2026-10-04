"""Frozen v3 rules; import reads reference metadata, never historical inputs."""
import math
from pathlib import Path

from validate_gold_s4_secondary_improvement_run_v1 import read, sha, require, array_hash
from gold_s4_secondary_improvement_logic_v1 import digest, calibration_split, weights

ROOT = Path(__file__).resolve().parent
CONFIG = 'gold_s4_secondary_improvement_v4_label_objective_config.json'
EXPERIMENT = 'gold_s4_secondary_improvement_v4_label_objective'
METRICS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day',
           'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')
REFERENCE = read(ROOT/'gold_s4_a_no_long_htf_reference_v2.json')['metrics']


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
    require(config['folds'] == ref['folds'] and config['candidate_count'] == len(space['candidates']) <= 24, 'Space count')
    require(len({c['candidate_id'] for c in space['candidates']}) == len(space['candidates']), 'Unique candidates')
    require(space['candidates'][0]['candidate_id'] == 'A_NO_LONG_HTF_CONTROL', 'Control first')
    require(len(space['label_definitions']) <= 8, 'Label definition bound')
    from gold_s4_v4_label_definitions import FORMULAS
    from gold_s4_v4_label_pipeline_audit import verify_audit
    verify_audit()
    definitions = {d['label_id']:d for d in space['label_definitions']}
    require(set(definitions) == set(FORMULAS), 'Frozen label inventory')
    for label_id, definition in definitions.items():
        require(definition['human_readable_formula'] == FORMULAS[label_id]
                and definition['implementation_source_sha256'] == sha(root/'gold_s4_v4_label_definitions.py'), 'Label implementation binding')
    for c in space['candidates']:
        require(c['features'] == ref['features'] and c['parameters'] == ref['parameters'], 'Frozen features/model capacity')
        require(c['label_id'] in definitions and c['label_formula'] == FORMULAS[c['label_id']]
                and c['model_objective'] == 'binary:logistic', 'Frozen label/objective')
        require(c['calibration'] in ('RAW','ISOTONIC') and c['threshold'] == .75, 'Frozen score contract')
        require(c['threshold_grid'] == ([.75] if c['reuse_reference_model'] else [.55,.60,.65,.70,.75,.80]), 'Frozen threshold grid')
        require(c['weighting'] == ('BALANCED' if c['reuse_reference_model'] else 'NONE'), 'Frozen weights')
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
    import numpy as np
    frozen = read(ROOT/'gold_s4_a_no_long_htf_reference_v2.json')['features']
    require(wanted == frozen, 'Future outcome/new feature input rejected')
    result = np.column_stack([base[:,base_names.index(n)] for n in wanted]).astype(np.float32)
    require(np.isfinite(result).all(), 'Nonfinite frozen features')
    return result


def signals(b0, secondary, threshold):
    return (b0 >= .75) | ((b0 < .75) & (secondary >= threshold))


def event(run, kind, space_hash, candidate_id=None, record=None):
    """Implements the frozen event_hash_spec; lifecycle receipts stay separate."""
    import json
    from datetime import datetime, timezone
    path = Path(run)/'research_events.jsonl'
    previous = json.loads(path.read_text(encoding='utf-8').splitlines()[-1]) if path.exists() else None
    require(kind in {'space_frozen','reference_pass','label_definition_freeze','candidate_start','candidate_end','pareto_complete','selection_complete','research_freeze'}, 'Research event type')
    require(previous is None or previous['event'] != 'research_freeze', 'Research chain already frozen')
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
    return dict(economic_frontier=economic_frontier(rows), pareto_frontier=[r['candidate_id'] for r in frontier],
                economic_annotations={r['candidate_id']: {k:r['pooled'][k] for k in ('profit_factor','mean_r','pnl_r','max_drawdown_r','stress_pf')} for r in frontier},
                selected_candidate=choices[0]['candidate_id'] if choices else None,
                research_result='IMPROVEMENT_FOUND' if choices else 'NO_IMPROVEMENT_FOUND',
                tie_break=[dict(candidate_id=r['candidate_id'], ordered_key=list(rank(r))) for r in choices])


def check_control(pooled, reference=REFERENCE):
    require(pooled.keys() == reference.keys() and all(isinstance(pooled[k], (int, float))
            and math.isclose(pooled[k], v, rel_tol=0, abs_tol=1e-12) for k, v in reference.items()),
            'REFERENCE_CONTROL_MISMATCH')


def split(data, c, config, n):
    import pandas as pd
    from gold_s4_v4_label_definitions import partitions
    end = pd.Timestamp(config['folds'][n-1][1])
    dev = (end-pd.DateOffset(months=3)).value
    cal = (end-pd.DateOffset(months=6)).value if c['calibration'] == 'ISOTONIC' else None
    return partitions(data,end.value,dev,cal)


def economic_frontier(rows):
    valid=[r for r in rows if all(isinstance(r['pooled'].get(k),(int,float)) and math.isfinite(r['pooled'][k]) for k in ('profit_factor','mean_r'))]
    keys=('profit_factor','mean_r')
    return sorted(r['candidate_id'] for r in valid if not any(
        all(o['pooled'][k] >= r['pooled'][k] for k in keys) and any(o['pooled'][k] > r['pooled'][k] for k in keys) for o in valid))


def milestones(p):
    return dict(PF_090=p['profit_factor'] is not None and p['profit_factor'] >= .90,
                PF_100=p['profit_factor'] is not None and p['profit_factor'] >= 1.,
                MEAN_R_NONNEGATIVE=p['mean_r'] >= 0., PNL_R_POSITIVE=p['pnl_r'] > 0.)


def load_evidence(config, ref):
    """Only reachable inside the USER session, never from fixture certification."""
    from gold_s4_secondary_improvement_v4_label_objective_launcher import require_session
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
