"""Independent v3 verification. Real validation is USER-workflow-only."""
import json
import math
from pathlib import Path

from gold_s4_secondary_improvement_v3_support import ROOT, REFERENCE, configuration, check_production, safe_path
from validate_gold_s4_secondary_improvement_run_v1 import (
    read, sha, require, equal, digest, array_hash, path_in, statistics,
)


def classify(p, folds):
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    safe = all(numeric(p.get(k)) for k in REFERENCE)
    safe = safe and p['profit_factor'] >= .8 and p['mean_r'] >= -.10690539315994348 and p['stress_pf'] >= .75
    safe = safe and len(folds) == 3 and all(
        all(numeric(f.get(k)) for k in ('trades', 'realized_win_rate', 'profit_factor'))
        and f['trades'] >= 10 and f['realized_win_rate'] >= .45 and f['profit_factor'] >= .7 for f in folds)
    improved = safe and p['realized_win_rate'] > REFERENCE['realized_win_rate'] and p['trades_per_day'] > REFERENCE['trades_per_day']
    gate = 'NONE'
    if improved:
        gate = 'TARGET' if p['realized_win_rate'] >= .6 and p['trades_per_day'] >= .5 else (
            'STRONG' if p['realized_win_rate'] >= .58 and p['trades_per_day'] >= .4 else 'INTERESTING')
    return dict(safety_pass=safe, interesting=improved, gate=gate)


def economics(p):
    if p['profit_factor'] is not None and p['profit_factor'] > 1 and p['mean_r'] > 0 and p['pnl_r'] > 0:
        return 'POSITIVE_EXPECTANCY'
    if p['profit_factor'] is not None and p['profit_factor'] < 1 and p['mean_r'] < 0 and p['pnl_r'] < 0:
        return 'NEGATIVE_EXPECTANCY'
    return 'NEAR_BREAK_EVEN'


def decision(rows):
    safe = [r for r in rows if classify(r['pooled'], r['fold_metrics'])['safety_pass']]
    front = []
    for r in safe:
        wr, tpd = r['pooled']['realized_win_rate'], r['pooled']['trades_per_day']
        if not any(o['pooled']['realized_win_rate'] >= wr and o['pooled']['trades_per_day'] >= tpd and
                   (o['pooled']['realized_win_rate'] > wr or o['pooled']['trades_per_day'] > tpd) for o in safe):
            front.append(r)
    def key(r):
        p = r['pooled']
        return [-p['realized_win_rate'], -p['trades_per_day'], -p['profit_factor'], -p['stress_pf'],
                -p['mean_r'], -p['pnl_r'], abs(p['max_drawdown_r']), r['candidate_id']]
    front.sort(key=key)
    choices = [r for r in front if r['candidate_id'] != 'A_NO_LONG_HTF_CONTROL' and classify(r['pooled'], r['fold_metrics'])['interesting']]
    return dict(pareto_frontier=[r['candidate_id'] for r in front],
                economic_annotations={r['candidate_id']:{k:r['pooled'][k] for k in ('profit_factor','mean_r','pnl_r','max_drawdown_r','stress_pf')} for r in front}, selected_candidate=choices[0]['candidate_id'] if choices else None,
                research_result='IMPROVEMENT_FOUND' if choices else 'NO_IMPROVEMENT_FOUND',
                tie_break=[dict(candidate_id=r['candidate_id'], ordered_key=key(r)) for r in choices])


def validate_records(space, rows, selected):
    expected = space['candidates']
    require(1 <= len(expected) <= 40 and len({c['candidate_id'] for c in expected}) == len(expected), 'Candidate count')
    require([r['candidate_id'] for r in rows] == [c['candidate_id'] for c in expected], 'All frozen IDs in order')
    for c, row in zip(expected, rows):
        require(row['config'] == c, 'Frozen candidate config')
        require(row['gate'] == classify(row['pooled'], row['fold_metrics']), 'Fold safety gate')
        require(row['economic_status'] == economics(row['pooled']), 'Economic status')
    require(equal(rows[0]['pooled'], REFERENCE), 'A_NO_LONG_HTF_REFERENCE_CONTROL_MISMATCH')
    require(selected == decision(rows), 'Independent Pareto and selection')


def isotonic_values(raw, y, query):
    """Independent weighted PAV on unique calibration scores; no estimator fit."""
    import numpy as np
    xs, inverse, counts = np.unique(raw.astype(np.float64), return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y, minlength=len(xs))
    blocks = []
    for i, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([i, i+1, float(total), int(count)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
    levels = np.empty(len(xs))
    for start, end, total, count in blocks:
        levels[start:end] = total/count
    return np.interp(query, xs, levels)


def check_fit_positions(data, c, config, n, fit, cal, saved_weight):
    import numpy as np
    import pandas as pd
    subset = np.flatnonzero(data['b0_train'] < .75)
    cutoff = None
    if c['calibration'] == 'RAW':
        wanted, calibration = subset, np.array([], dtype=np.int64)
    else:
        cutoff = (pd.Timestamp(config['folds'][n-1][1])-pd.DateOffset(months=config['calibration_months'])).value
        wanted = subset[(data['train_ns'][subset] < cutoff) & (data['maturity'][subset] < cutoff) & (data['legacy_maturity'][subset] < cutoff)]
        calibration = subset[data['train_ns'][subset] >= cutoff]
        require(len(calibration) >= config['minimum_calibration_samples'] and
                np.bincount(data['target'][calibration], minlength=2).min() >= config['minimum_calibration_class_samples'], 'Calibration sample minimum')
    require(len(wanted) > 0 and np.array_equal(fit, wanted) and np.array_equal(cal, calibration), 'Calibration isolation')
    y = data['target'][wanted]
    require(np.array_equal(np.unique(y), [0, 1]), 'Class availability')
    expected = np.ones(len(y)) if c['weighting'] == 'NONE' else len(y)/(2.*np.bincount(y, minlength=2)[y])*np.where(y == 1, c['positive_weight_multiplier'], 1.)
    require(np.array_equal(saved_weight, expected), 'Sample weighting')
    return cutoff


def verify_model_file(run, model):
    path = path_in(run, model['path'])
    require(path.is_relative_to(run/'models') and sha(path) == model['sha256'], 'Model hash and path')
    return path


def feature_matrix(base, names, requested):
    """Independent implementation of frozen pointwise feature definitions."""
    import numpy as np
    columns = dict(zip(names, base.T))
    trends = np.asarray([columns[n] for n in ('Daily_TREND','H1_TREND','H4_TREND')]).T.copy()
    directions = np.sign(trends)
    average = directions.mean(axis=1)
    denominator = np.abs(columns['VOLA_RATIO'])+np.float32(1e-6)
    columns['HTF_TREND_CONSENSUS'] = trends.mean(axis=1)
    columns['HTF_DIRECTION_COUNT'] = np.count_nonzero(trends > 0, axis=1)
    columns['HTF_ALIGNMENT_RATIO'] = (directions == np.sign(columns['MACD_HIST'])[:, None]).sum(axis=1)/3
    columns['HTF_DISAGREEMENT_SCORE'] = np.mean((directions-average[:, None])**2, axis=1)
    columns['X_BODY_QUALITY'] = columns['BODY_PCT']/denominator
    columns['X_TRIGGER_DISTANCE'] = columns['BIAS_20']/denominator
    matrix = np.stack([columns[n] for n in requested], axis=1).astype(np.float32)
    require(np.isfinite(matrix).all(), 'Nonfinite features')
    return matrix


def check_events(events, rows, selected, space):
    """Independent exact-payload checker; no trainer event/hash helper imports."""
    from datetime import datetime, timedelta
    expected = [('space_frozen',None,None),('reference_pass',rows[0]['candidate_id'],rows[0]),
                ('candidate_result',rows[0]['candidate_id'],rows[0])]
    for row in rows[1:]:
        expected.extend([('candidate_begin',row['candidate_id'],None),('candidate_result',row['candidate_id'],row)])
    expected.append(('research_complete',None,selected))
    require(len(events) == len(expected), 'Complete frozen research event inventory')
    previous, stamp = None, None
    fields = {'sequence','event','at_utc','space_sha256','candidate_id','stage','record_sha256','previous_sha256','event_sha256'}
    for i, (e, (kind, cid, record)) in enumerate(zip(events, expected)):
        require(set(e) == fields, 'Event schema')
        at = datetime.fromisoformat(e['at_utc'])
        require(at.tzinfo is not None and at.utcoffset() == timedelta(0)
                and (stamp is None or at >= stamp), 'Event chronology')
        payload = {k:v for k,v in e.items() if k != 'event_sha256'}
        require(type(e['sequence']) is int and e['sequence'] == i and e['previous_sha256'] == previous
                and digest(payload) == e['event_sha256'] and e['space_sha256'] == digest(space)
                and e['event'] == kind and e['candidate_id'] == cid and e['stage'] is None
                and e['record_sha256'] == (digest(record) if record is not None else None), 'Frozen chronological event chain')
        previous, stamp = e['event_sha256'], at
    return True


def validate(run):
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import gold_gemini_execution_semantics_v1 as semantics
    config, ref, space = configuration()
    require(read(run/'approved_config.json') == config and read(run/'reference_binding.json') == ref
            and read(run/'predeclared_search_space.json') == space, 'Frozen input snapshots')
    require(sha(run/'validator_script.py') == sha(Path(__file__)), 'Validator snapshot')
    manifest = read(run/'manifest.json')
    require(manifest['manual_start'] is True and manifest['started_by'] == 'USER_LAUNCHER'
            and manifest['infrastructure_only'] is False and manifest['git_dirty'] is False, 'USER provenance')
    from gold_s4_secondary_improvement_v3_launcher import APPROVAL
    require(manifest['source_bindings'] == read(ROOT/APPROVAL)['bindings'], 'Complete approved source inventory')
    for name, expected in manifest['source_bindings'].items():
        require(sha(ROOT/name) == expected, 'Source identity:'+name)
    rows = [json.loads(line) for line in (run/'candidate_results.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen = read(run/'selection.json')
    validate_records(space, rows, chosen)
    events = [json.loads(line) for line in (run/'research_events.jsonl').read_text(encoding='utf-8').splitlines()]
    check_events(events, rows, chosen, space)
    require(read(run/'reference_control.json') == dict(status='PASS', **rows[0]), 'Control evidence')
    state = load_evidence(config, ref)
    inventory = set()
    for row in rows:
        c = row['config']
        require([m['fold_number'] for m in row['models']] == [1, 2, 3], 'Three model folds')
        predictions = []
        for m in row['models']:
            n = m['fold_number']
            path = verify_model_file(run, m)
            inventory.add(m['path'])
            data = state[n]
            classifier = xgb.XGBClassifier()
            classifier.load_model(path)
            require(classifier.get_booster().feature_names == c['features'] and classifier.get_booster().num_boosted_rounds() == c['parameters']['n_estimators'], 'Model feature schema and rounds')
            xscore = feature_matrix(data['score_x'], config['input_features'], c['features'])
            raw = classifier.predict_proba(pd.DataFrame(xscore, columns=c['features']))[:, 1]
            probability = raw
            if c['reuse_reference_model']:
                require(m['sha256'] == ref['models'][n-1]['sha256'], 'Frozen A_NO_LONG_HTF model')
            else:
                evidence = path_in(run, m['evidence_path'])
                require(sha(evidence) == m['evidence_sha256'], 'Candidate evidence hash')
                with np.load(evidence, allow_pickle=False) as saved:
                    require(np.array_equal(raw, saved['raw']), 'Independent prediction')
                    fit, cal = saved['fit_positions'], saved['calibration_positions']
                    cutoff = check_fit_positions(data, c, config, n, fit, cal, saved['sample_weight'])
                    xtrain = feature_matrix(data['train_x'], config['input_features'], c['features'])
                    require(array_hash(xtrain[fit]) == m['fit_feature_sha256'] and array_hash(data['target'][fit]) == m['fit_target_sha256'], 'Fit feature and label hashes')
                    params = m['model_configuration']['learner']
                    tree = params['gradient_booster']['tree_train_param']
                    for key in ('max_depth', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree', 'learning_rate'):
                        require(math.isclose(float(tree[key]), c['parameters'][key], rel_tol=0, abs_tol=1e-6), 'Regularization:'+key)
                    require(int(params['generic_param']['seed']) == c['seed'] and params['objective']['name'] == 'binary:logistic', 'Seed and objective')
                    if c['calibration'] != 'RAW':
                        calraw = classifier.predict_proba(pd.DataFrame(xtrain[cal], columns=c['features']))[:, 1]
                        require(np.array_equal(calraw, saved['calibration_raw']), 'Calibration prediction')
                        info = m['calibration']
                        require(info['cutoff_ns'] == cutoff and info['fit_count'] == len(fit) and info['calibration_count'] == len(cal), 'Calibration metadata')
                        if c['calibration'] == 'PLATT':
                            def logit(p):
                                p = np.clip(p.astype(np.float64), 1e-6, 1-1e-6)
                                return np.log(p/(1-p))
                            def calibrated(p):
                                return 1/(1+np.exp(-np.clip(logit(p)*info['coefficient']+info['intercept'], -700, 700)))
                            residual = calibrated(calraw)-data['target'][cal]
                            require(abs(residual.mean()) < .001 and abs((residual*logit(calraw)).mean()+info['coefficient']/len(cal)) < .001, 'Platt stationarity')
                            probability = calibrated(raw)
                        else:
                            xs, ys = np.asarray(info['x_thresholds']), np.asarray(info['y_thresholds'])
                            require(len(xs) == len(ys) and len(xs) > 0 and np.isfinite(xs).all() and np.isfinite(ys).all()
                                    and np.all(np.diff(xs) > 0) and np.all(np.diff(ys) >= 0), 'Isotonic monotonicity')
                            expected = isotonic_values(calraw, data['target'][cal], np.r_[raw, xs, calraw])
                            observed = np.interp(np.r_[raw, xs, calraw], xs, ys)
                            require(np.allclose(expected, observed, atol=1e-12, rtol=0), 'Independent isotonic PAV')
                            probability = np.interp(raw, xs, ys)
                    else:
                        require(m['calibration'] is None, 'Raw calibration absent')
                    require(np.allclose(probability, saved['probability'], rtol=0, atol=1e-12), 'Final probability')
            require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)), 'Probability domain')
            predictions.append(probability)
        frame = pd.concat([state[n]['price'] for n in (1, 2, 3)], ignore_index=True)
        b0 = np.concatenate([state[n]['b0_score'] for n in (1, 2, 3)])
        secondary = np.concatenate(predictions)
        union = (b0 >= .75) | ((b0 < .75) & (secondary >= c['threshold']))
        frame['buy_prob'], frame['sell_prob'] = b0, np.float32(0)
        cohort = semantics.finalize_cohort(frame, c['candidate_id'], offset_hours=0)
        times = cohort.decision_time_api.to_numpy(dtype='datetime64[ns]')
        gap = np.r_[True, np.diff(times).astype('timedelta64[s]').astype(np.int64) > 120]
        episodes = np.cumsum(union & (np.r_[False, ~union[:-1]] | gap)).astype(np.int64)-1
        episodes[~union] = -1
        cohort['raw_signal'], cohort['raw_episode_id'] = union, episodes
        trades, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        days, folds = 0, []
        for name, start, end in config['folds']:
            count = (pd.Timestamp(end)-pd.Timestamp(start)).days
            days += count
            folds.append(dict(fold=name, **statistics([t for t in trades if start <= t['entry_time_api'] < end], count)))
        require(equal(row['pooled'], statistics(trades, days)) and equal(row['fold_metrics'], folds), 'Independent historical metrics')
        ledger = path_in(run, row['ledger_path'])
        require(sha(ledger) == row['ledger_sha256'] and equal(read(ledger), trades), 'Trade ledger')
        print('Verified '+c['candidate_id'], flush=True)
    require({p.relative_to(run).as_posix() for p in (run/'models').rglob('*.json')} == inventory, 'No extra models')
    expected = next((r for r in rows if r['candidate_id'] == chosen['selected_candidate']), None)
    require(read(run/'selected_candidate.json') == dict(candidate=expected, classification='HISTORICAL_RESEARCH_CANDIDATE', production_promoted=False), 'Selection binding')
    require(read(run/'pareto_frontier.json') == chosen['pareto_frontier'], 'Pareto artifact')
    require(read(run/'training_result.json') == dict(research_result=chosen['research_result'], candidate=expected, run_id=run.name), 'Same-run handoff')
    check_production(config)
    return dict(candidate_count=len(rows), **chosen, holdout_used=False, production_changed=False, production_promoted=False)


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


def run_authorized(run, permit):
    from gold_s4_secondary_improvement_v3_launcher import consume_validation
    consume_validation(permit, run)
    require(run.parent == ROOT/'training_runs' and not (run/'FINALIZED.json').exists(), 'New USER run only')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump(dict(owner='USER_LAUNCHED_WORKFLOW', rule='Do not retry validation'), out)
    try:
        result = dict(overall='PASS', failed_checks=[], **validate(run))
    except Exception as error:
        result = dict(overall='FAIL', failed_checks=[type(error).__name__+': '+str(error)])
    with (run/'validator.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit('Real validation requires the same USER-launched v3 workflow; no direct CLI validation')
