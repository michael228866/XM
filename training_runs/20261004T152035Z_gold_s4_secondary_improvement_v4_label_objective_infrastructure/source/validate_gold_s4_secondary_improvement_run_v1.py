"""Independent audit/replay of a USER research run. Never imports the search/trainer."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day', 'profit_factor',
        'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')
REFERENCE = dict(zip(KEYS, (743, 421, 322, .566621803499327, .2905748924520923,
    .8271090428367022, -.07580700838089183, -56.32460722700263, -62.009878061758215,
    .7857575699558063)))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+np.asarray(value.shape, dtype=np.int64).tobytes()+value.tobytes()).hexdigest()


def path_in(run, name):
    path = (run/name).resolve()
    if not path.is_relative_to(run.resolve()):
        raise PermissionError('Artifact escapes run')
    return path


def require(value, name):
    if not value:
        raise ValueError(name)


def equal(a, b):
    if isinstance(b, dict):
        return isinstance(a, dict) and a.keys() == b.keys() and all(equal(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(b, float):
        return isinstance(a, (float, int)) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return a == b


def independent_gate(pooled, folds):
    safe = (all(isinstance(pooled.get(k), (float, int)) and math.isfinite(pooled[k]) for k in KEYS)
            and pooled['profit_factor'] >= .80 and pooled['mean_r'] >= -.10690539315994348 and pooled['stress_pf'] >= .75
            and bool(folds) and all(r['trades'] >= 10 and r['realized_win_rate'] >= .45
                and isinstance(r['profit_factor'], (float, int)) and math.isfinite(r['profit_factor']) and r['profit_factor'] >= .70 for r in folds))
    improved = safe and pooled['realized_win_rate'] > REFERENCE['realized_win_rate'] and pooled['trades_per_day'] > REFERENCE['trades_per_day']
    label = 'NONE'
    if improved:
        label = 'TARGET' if pooled['realized_win_rate'] >= .60 and pooled['trades_per_day'] >= .50 else ('STRONG' if pooled['realized_win_rate'] >= .58 and pooled['trades_per_day'] >= .40 else 'INTERESTING')
    return {'safety_pass': safe, 'interesting': improved, 'gate': label,
            'delta_wr': pooled['realized_win_rate']-REFERENCE['realized_win_rate'],
            'delta_trades_per_day': pooled['trades_per_day']-REFERENCE['trades_per_day']}


def independent_selection(rows):
    eligible = [r for r in rows if r['stage'] in (0, 2) and independent_gate(r['pooled'], r['fold_metrics'])['safety_pass']]
    frontier = []
    for r in eligible:
        wr, tpd = r['pooled']['realized_win_rate'], r['pooled']['trades_per_day']
        if not any(o['pooled']['realized_win_rate'] >= wr and o['pooled']['trades_per_day'] >= tpd
                   and (o['pooled']['realized_win_rate'] > wr or o['pooled']['trades_per_day'] > tpd) for o in eligible):
            frontier.append(r)
    def key(r):
        m = r['pooled']
        return [-m['realized_win_rate'], -m['trades_per_day'], -m['stress_pf'], -m['profit_factor'],
                -m['mean_r'], abs(m['max_drawdown_r']), r['candidate_id']]
    frontier.sort(key=key)
    choices = [r for r in frontier if r['stage'] == 2 and independent_gate(r['pooled'], r['fold_metrics'])['interesting']]
    return [r['candidate_id'] for r in frontier], choices[0]['candidate_id'] if choices else None, [
        {'candidate_id': r['candidate_id'], 'ordered_key': key(r)} for r in choices]


def feature_matrix(base, names, selected):
    import numpy as np
    value = {name: base[:, i] for i, name in enumerate(names)}
    high = (np.column_stack([value['H1_TREND'], value['H4_TREND'], value['Daily_TREND']])).mean(axis=1)
    low = (np.column_stack([value['M2_TREND'], value['M5_TREND'], value['M15_TREND']])).mean(axis=1)
    value['S4_RSI_CENTER'] = (value['M1_RSI']-50)/50
    value['S4_MACD_ATR'] = value['MACD_HIST']/(np.abs(value['ATR'])+np.float32(1e-6))
    value['S4_BODY_VOL'] = value['BODY_PCT']/(np.abs(value['VOLA_RATIO'])+np.float32(1e-6))
    value.update(S4_HTF_AGREEMENT=high, S4_LTF_AGREEMENT=low, S4_ALIGNMENT=high*low)
    return np.column_stack([value[n] for n in selected]).astype(np.float32)


def statistics(trades, days):
    import numpy as np
    values = np.asarray([t['net_r'] for t in trades], dtype=np.float64)
    stress = np.asarray([t['stress_r'] for t in trades], dtype=np.float64)
    def pf(x):
        gain, loss = x[x > 0].sum(), -x[x <= 0].sum()
        return float(gain/loss) if loss > 0 else (None if gain > 0 else 0.)
    curve = np.r_[0., np.cumsum(values)]
    return {'trades': len(values), 'wins': int((values > 0).sum()), 'losses': int((values <= 0).sum()),
            'realized_win_rate': float((values > 0).mean()) if len(values) else 0.,
            'trades_per_day': len(values)/max(days, 1), 'profit_factor': pf(values),
            'mean_r': float(values.mean()) if len(values) else 0., 'pnl_r': float(values.sum()),
            'max_drawdown_r': float((curve-np.maximum.accumulate(curve)).min()), 'stress_pf': pf(stress)}


def check_search_records(config, records, events):
    expected = {c['candidate_id']: c for c in config['candidates']}
    require(len(expected) == len(config['candidates']) <= 100, 'candidate_count')
    required = {('F0_REFERENCE', 0)} | {(cid, 1) for cid in expected if cid != 'F0_REFERENCE'}
    required |= {(r['candidate_id'], 2) for r in records if r['stage'] == 1 and independent_gate(r['pooled'], r['fold_metrics'])['interesting']}
    require(len(records) == len(required) and {(r['candidate_id'], r['stage']) for r in records} == required, 'missing_or_unrecorded_candidates')
    for r in records:
        c = expected[r['candidate_id']]
        for key in ('family', 'parameters', 'features', 'seed', 'threshold', 'calibration', 'positive_weight_multiplier'):
            require(r[key] == c[key], 'candidate_config:'+key)
        require(r['feature_set_sha256'] == digest(c['features']) and r['fold_definition_sha256'] == digest(config['folds'])
                and r['label_pipeline_sha256'] == config['label_pipeline_sha256'], 'candidate_bindings')
        require(equal(r['gate_result'], independent_gate(r['pooled'], r['fold_metrics'])), 'candidate_gate')
    previous = None
    kinds = []
    result_keys = []
    begun = set()
    control_passed = False
    last_time = None
    for i, e in enumerate(events):
        payload = {k: v for k, v in e.items() if k != 'event_sha256'}
        require(e['sequence'] == i and e['previous_sha256'] == previous and digest(payload) == e['event_sha256']
                and e['space_sha256'] == digest(config['candidates']), 'search_event_chain')
        previous = e['event_sha256']
        require(e['event'] in {'space_frozen', 'reference_pass', 'candidate_begin', 'candidate_result', 'research_complete'}, 'unknown_search_event')
        stamp = datetime.fromisoformat(e['at_utc'])
        require(last_time is None or stamp >= last_time, 'event_chronology')
        last_time = stamp
        kinds.append(e['event'])
        pair = (e['candidate_id'], e['stage'])
        if e['event'] == 'reference_pass':
            require(i == 1, 'reference_before_search')
            control_passed = True
        elif e['event'] == 'candidate_begin':
            require(control_passed and pair not in begun and pair in required and pair[1] != 0, 'candidate_begin_order')
            if pair[1] == 2:
                require((pair[0], 1) in result_keys, 'stage2_after_shortlist')
            begun.add(pair)
        elif e['event'] == 'candidate_result':
            require(control_passed and (pair in begun or pair == ('F0_REFERENCE', 0)), 'candidate_result_order')
            row = next(r for r in records if (r['candidate_id'], r['stage']) == pair)
            require(e['record_sha256'] == digest(row), 'candidate_result_event_hash')
            result_keys.append(pair)
    require(kinds[0] == 'space_frozen' and kinds[-1] == 'research_complete' and len(result_keys) == len(records)
            and set(result_keys) == required, 'search_freeze_complete')


def validate(run):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import gold_gemini_execution_semantics_v1 as semantics
    config = read(run/'approved_config.json')
    base = read(ROOT/config['reference_config'])
    ds = read(ROOT/base['discovery_spec'])
    require(config['reference_metrics'] == REFERENCE and config['numeric_tolerance'] == 1e-12, 'reference_constants')
    require(config['folds'] == base['fold_definition'] and config['reference_features'] == base['feature_list'], 'frozen_folds_features')
    for name, value in {**config['frozen_hashes'], **base['source_bindings'], **base['archive_seals']}.items():
        require(sha(ROOT/name) == value, 'frozen_source:'+name)
    require(read(run/'predeclared_search_space.json') == config['candidates'] == read(run/'research_plan.json')['candidates'], 'frozen_search_space')
    records = [json.loads(line) for line in (run/'candidate_results.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (run/'research_events.jsonl').read_text().splitlines()]
    check_search_records(config, records, events)
    dataset = read(run/'training_dataset_manifest.json')
    require(len(dataset['datasets']) == len(base['required_datasets']) == 21 and dataset['holdout_input'] is False, 'dataset_inventory')
    for item, expected in zip(dataset['datasets'], base['required_datasets']):
        path = Path(item['path']).resolve()
        require(path.is_relative_to(ROOT/'historical_training_data') and path.name == expected['filename']
                and item['symbol'] == 'GOLD#' and item['training_eligible'] is True
                and sha(path) == item['sha256'] == expected['sha256']
                and expected['last_timestamp'] <= base['raw_data_cutoff'] < '2026-09-25T00:00:00', 'dataset:'+expected['filename'])
    control = read(run/'reference_control.json')
    require(control['status'] == 'PASS' and equal(control['pooled'], REFERENCE), 'reference_control')
    original_models = read(ROOT/config['reference_run']/'model_inventory.json')
    require([x['sha256'] for x in control['models']] == [x['sha256'] for x in original_models], 'reference_models')
    for model in control['models']:
        require(sha(path_in(run, model['path'])) == model['sha256'], 'reference_model_hash')
    with np.load(run/'secondary_evidence.npz', allow_pickle=False) as a, np.load(ROOT/config['reference_run']/'secondary_evidence.npz', allow_pickle=False) as b:
        require(set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files), 'exact_reference_evidence')
    evidence, cached_predictions = {}, {}
    for item in read(run/'evidence_index.json'):
        n = item['fold_number']
        arrays = path_in(run, item['arrays_path'])
        require(sha(arrays) == item['arrays_sha256'], 'array_hash')
        with np.load(arrays, allow_pickle=False) as values:
            data = {k: values[k].copy() for k in values.files}
        for part in ('train', 'score'):
            pieces = []
            for chunk in item[part+'_chunks']:
                path = path_in(run, chunk['path'])
                require(sha(path) == chunk['sha256'], 'feature_chunk_hash')
                with np.load(path, allow_pickle=False) as values:
                    require(len(values['values']) == chunk['rows'], 'chunk_rows')
                    pieces.append(values['values'].copy())
            data[part+'_x'] = np.concatenate(pieces)
            require(array_hash(data[part+'_x']) == ds['feature_matrix_sha256'][f'fold{n}_{part}'], 'frozen_feature_matrix')
        require(array_hash(data['target']) == ds['c1_target_sha256'][f'fold{n}_train'], 'frozen_labels')
        with np.load(run/'secondary_evidence.npz', allow_pickle=False) as ref:
            for key in ('train_indices', 'score_indices', 'train_ns', 'score_ns', 'b0_train', 'b0_score'):
                require(np.array_equal(data[key], ref[f'fold{n}_'+key]), 'fold_isolation:'+key)
        start = pd.Timestamp(config['folds'][n-1][1]).value
        end = pd.Timestamp(config['folds'][n-1][2]).value
        require(data['train_ns'].max() < start <= data['score_ns'].min() and data['score_ns'].max() < end
                and np.all(data['maturity'] < start) and np.all(data['legacy_maturity'] < start)
                and np.all(data['maturity'] >= data['train_ns']), 'label_maturity_and_fold_cutoff')
        price_path = path_in(run, item['price_path'])
        require(sha(price_path) == item['price_sha256'], 'price_hash')
        with np.load(price_path, allow_pickle=False) as p:
            data['price'] = pd.DataFrame({key: pd.to_datetime(p[key]) if key == 'TIME_DT' else p[key] for key in p.files})
        require(np.array_equal(data['price'].TIME_DT.to_numpy(dtype='datetime64[ns]').astype(np.int64), data['score_ns']), 'price_time_alignment')
        evidence[n] = data
    require(set(evidence) == {1, 2, 3}, 'three_evidence_folds')
    expected_ids = {x['candidate_id']: x for x in config['candidates']}
    for row in records:
        c = expected_ids[row['candidate_id']]
        require(row['training_data_manifest_sha256'] == sha(run/'training_dataset_manifest.json'), 'dataset_record_binding')
        numbers = [1, 2] if row['stage'] == 1 else [1, 2, 3]
        require([x['fold_number'] for x in row['models']] == numbers, 'model_fold_inventory')
        probs = []
        for model in row['models']:
            n = model['fold_number']
            key = (c['candidate_id'], n)
            data = evidence[n]
            path = path_in(run, model['path'])
            require(path.is_relative_to(run/'models') and sha(path) == model['sha256'], 'model_hash_and_location')
            if key not in cached_predictions:
                xscore = feature_matrix(data['score_x'], config['reference_features'], c['features'])
                classifier = xgb.XGBClassifier()
                classifier.load_model(path)
                require(classifier.get_booster().feature_names == c['features'] and classifier.get_booster().num_boosted_rounds() == c['parameters']['n_estimators'], 'model_features_rounds')
                raw = classifier.predict_proba(pd.DataFrame(xscore, columns=c['features']))[:, 1]
                if c['family'] in (0, 4):
                    require(model['sha256'] == original_models[n-1]['sha256'] and np.array_equal(raw, data['reference_probability']), 'fixed_reference_probability')
                    probability = raw
                else:
                    audit_path = path_in(run, model['evidence_path'])
                    require(sha(audit_path) == model['evidence_sha256'], 'candidate_evidence_hash')
                    with np.load(audit_path, allow_pickle=False) as audit:
                        require(np.array_equal(raw, audit['raw']), 'model_prediction')
                        fit, cal = audit['fit_positions'], audit['calibration_positions']
                        subset = np.flatnonzero(data['b0_train'] < .75)
                        if c['calibration'] == 'none':
                            require(np.array_equal(fit, subset) and not len(cal) and model['calibration'] is None, 'conditioning')
                        else:
                            cutoff = (pd.Timestamp(config['folds'][n-1][1])-pd.DateOffset(months=3)).value
                            expected_fit = subset[(data['train_ns'][subset] < cutoff) & (data['maturity'][subset] < cutoff) & (data['legacy_maturity'][subset] < cutoff)]
                            expected_cal = subset[data['train_ns'][subset] >= cutoff]
                            require(np.array_equal(fit, expected_fit) and np.array_equal(cal, expected_cal) and len(fit) and len(cal), 'calibration_leakage')
                        xtrain = feature_matrix(data['train_x'], config['reference_features'], c['features'])
                        require(array_hash(xtrain[fit]) == model['fit_feature_sha256'] and array_hash(data['target'][fit]) == model['fit_target_sha256'], 'fit_feature_label_binding')
                        y = data['target'][fit]
                        configuration = model['model_configuration']['learner']
                        tree_parameters = configuration['gradient_booster']['tree_train_param']
                        for parameter in ('max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'learning_rate', 'reg_alpha', 'reg_lambda', 'gamma'):
                            require(math.isclose(float(tree_parameters[parameter]), c['parameters'][parameter], rel_tol=0, abs_tol=1e-6), 'model_parameter:'+parameter)
                        require(int(configuration['generic_param']['seed']) == c['seed'] and configuration['objective']['name'] == 'binary:logistic', 'seed_objective')
                        expected_weight = len(y)/(2.*np.bincount(y, minlength=2)[y])*np.where(y == 1, c['positive_weight_multiplier'], 1.)
                        require(np.array_equal(audit['sample_weight'], expected_weight), 'weighting_rule')
                        if c['calibration'] != 'none':
                            calraw = classifier.predict_proba(pd.DataFrame(xtrain[cal], columns=c['features']))[:, 1]
                            require(np.array_equal(calraw, audit['calibration_raw']), 'calibration_predictions')
                            info = model['calibration']
                            require(info['cutoff_ns'] == cutoff and info['fit_count'] == len(fit) and info['calibration_count'] == len(cal), 'calibration_metadata')
                            def calibrated(p):
                                p = np.clip(p.astype(np.float64), 1e-6, 1-1e-6)
                                z = np.log(p/(1-p))*info['coefficient']+info['intercept']
                                return 1/(1+np.exp(-np.clip(z, -700, 700)))
                            probability = calibrated(raw)
                            yc = data['target'][cal]
                            residual = calibrated(calraw)-yc
                            logit = np.log(np.clip(calraw.astype(np.float64), 1e-6, 1-1e-6)/(1-np.clip(calraw.astype(np.float64), 1e-6, 1-1e-6)))
                            require(abs(residual.mean()) < .001 and abs((residual*logit).mean()+info['coefficient']/len(cal)) < .001, 'calibration_stationarity')
                        else:
                            probability = raw
                        require(np.allclose(probability, audit['probability'], rtol=0, atol=1e-12), 'calibrated_probability')
                require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'probability_domain')
                cached_predictions[key] = probability
            probs.append(cached_predictions[key])
        frame = pd.concat([evidence[n]['price'] for n in numbers], ignore_index=True)
        b0 = np.concatenate([evidence[n]['b0_score'] for n in numbers])
        secondary = np.concatenate(probs)
        union = (b0 >= .75) | ((b0 < .75) & (secondary >= c['threshold']))
        frame['buy_prob'], frame['sell_prob'] = b0, np.float32(0)
        cohort = semantics.finalize_cohort(frame, c['candidate_id'], offset_hours=0)
        times = cohort.decision_time_api.to_numpy(dtype='datetime64[ns]')
        gap = np.r_[True, np.diff(times).astype('timedelta64[s]').astype(np.int64) > 120]
        episodes = np.cumsum(union & (np.r_[False, ~union[:-1]] | gap)).astype(np.int64)-1
        episodes[~union] = -1
        cohort['raw_signal'], cohort['raw_episode_id'] = union, episodes
        trades, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        days, fold_metrics = 0, []
        for n in numbers:
            name, start, end = config['folds'][n-1]
            count = (pd.Timestamp(end)-pd.Timestamp(start)).days
            days += count
            fold_metrics.append({'fold': name, **statistics([t for t in trades if start <= t['entry_time_api'] < end], count)})
        require(equal(row['pooled'], statistics(trades, days)) and equal(row['fold_metrics'], fold_metrics), 'independent_S5_metrics:'+row['candidate_id'])
        if row['stage'] != 0:
            ledger = path_in(run, row['ledger_path'])
            require(sha(ledger) == row['ledger_sha256'] and equal(read(ledger), trades), 'trade_ledger_exact')
    frontier, selected, ties = independent_selection(records)
    decision = read(run/'selected_candidate_reason.json')
    require(events[-1]['record_sha256'] == digest(decision), 'final_decision_event')
    require(read(run/'pareto_frontier.json') == frontier == decision['pareto_frontier'] and decision['selected_candidate'] == selected
            and equal(decision['tie_break'], ties), 'independent_pareto_selection')
    chosen = next((r for r in records if r['stage'] == 2 and r['candidate_id'] == selected), None)
    require(read(run/'selected_candidate.json')['candidate'] == chosen and read(run/'selected_candidate.json')['production_promoted'] is False, 'selected_candidate_binding')
    result = 'IMPROVEMENT_FOUND' if chosen else 'NO_IMPROVEMENT_FOUND'
    handoff = read(run/'training_result.json')
    require(handoff['run_id'] == run.name and handoff['train_status'] == 'PASS' and handoff['candidate'] == chosen
            and handoff['research_result'] == decision['research_result'] == result
            and handoff['holdout_used'] is handoff['production_promoted'] is False, 'same_run_result')
    require(read(run/'production_before.json') == read(run/'production_after.json') == base['protected_sha256']
            and all(sha(ROOT/n) == value for n, value in base['protected_sha256'].items()), 'production_hashes')
    model_paths = {m['path'] for row in records for m in row['models']}
    require({p.relative_to(run).as_posix() for p in (run/'models').rglob('*.json')} == model_paths, 'no_unrecorded_models')
    return {'candidate_count': len(config['candidates']), 'record_count': len(records), 'research_result': result,
            'selected_candidate': selected, 'holdout_used': False, 'production_promoted': False}


def main():
    run = Path(sys.argv[1]).resolve()
    require(run.parent == ROOT/'training_runs' and not (run/'FINALIZED.json').exists(), 'New run required')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as out:
        json.dump({'rule': 'Do not retry validation', 'at_utc': datetime.now(timezone.utc).isoformat()}, out)
    release = None
    try:
        manifest = read(run/'manifest.json')
        require(manifest['manual_start'] is True and manifest['started_by'] == 'USER_LAUNCHER' and manifest['git_dirty'] is False, 'manual_source')
        for source, snapshot in [('gold_s4_secondary_improvement_v1.py', 'training_script.py'),
                ('gold_s4_secondary_improvement_config_v1.json', 'approved_config.json'),
                ('validate_gold_s4_secondary_improvement_run_v1.py', 'validator_script.py')]:
            raw = subprocess.check_output(['git', 'show', manifest['git_commit']+':'+source], cwd=ROOT)
            require(raw == (run/snapshot).read_bytes(), 'committed_source:'+source)
        for name, expected in manifest['source_bindings'].items():
            committed = subprocess.check_output(['git', 'show', manifest['git_commit']+':'+name], cwd=ROOT)
            require(sha(ROOT/name) == expected == hashlib.sha256(committed).hexdigest(), 'source_binding:'+name)
        import numpy
        import pandas
        import xgboost
        import gold_gemini_execution_semantics_v1
        from training_holdout_guard_v1 import install
        release = install(ROOT, write_root=run)
        result = {'overall': 'PASS', 'run_id': run.name, 'failed_checks': [], **validate(run)}
    except Exception as error:
        result = {'overall': 'FAIL', 'run_id': run.name, 'failed_checks': [type(error).__name__+': '+str(error)]}
    finally:
        if release:
            release()
    (run/'validator.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (run/'validator.md').write_text('# Independent S4 improvement validation\n\n'+result['overall']+'\n', encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
