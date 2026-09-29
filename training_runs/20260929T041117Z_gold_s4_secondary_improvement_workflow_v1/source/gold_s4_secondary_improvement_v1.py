"""USER-only bounded historical S4 research; imports never train or predict."""
import contextlib
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from gold_manual_s4_training_data_v1 import ROOT, read, write, sha, prepare
from gold_s4_secondary_improvement_logic_v1 import (
    METRICS, REFERENCE, digest, gate, select, check_reference, features,
    calibration_split, weights, freeze_space, event,
)

CONFIG = ROOT/'gold_s4_secondary_improvement_config_v1.json'
EXPERIMENT = 'gold_s4_secondary_improvement_v1'


def verify_config(config):
    if (config['reference_metrics'] != REFERENCE or config['numeric_tolerance'] != 1e-12
            or config['training_symbol'] != 'GOLD#' or config['stage1_fold_numbers'] != [1, 2]
            or config['stage2_fold_numbers'] != [1, 2, 3] or len(config['candidates']) > 100):
        raise ValueError('Frozen improvement contract drift')
    for name, expected in config['frozen_hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen reference/source changed: '+name)
    import gold_manual_s4_secondary_retrain_v1 as reference
    base = read(ROOT/config['reference_config'])
    reference.verify_config(base)
    if config['folds'] != base['fold_definition'] or config['reference_features'] != base['feature_list']:
        raise ValueError('Reference folds/features changed')
    return base


def normalized(row):
    import math
    mapping = dict(zip(METRICS, ('trades', 'wins', 'losses', 'realized_wr', 'trades_per_day',
                  'pf', 'mean_r', 'pnl_r', 'max_dd_r', 'cost_stress_pf')))
    return {key: row[value] if not isinstance(row[value], float) or math.isfinite(row[value]) else None
            for key, value in mapping.items()}


def metric_set(old, trades, windows):
    import pandas as pd
    folds = []
    total = 0
    for name, start, end in windows:
        days = (pd.Timestamp(end)-pd.Timestamp(start)).days
        total += days
        local = [t for t in trades if start <= t['entry_time_api'] < end]
        folds.append({'fold': name, **normalized(old.trade_metrics(local, days))})
    return normalized(old.trade_metrics(trades, total)), folds


def save_chunks(run, name, matrix):
    import numpy as np
    output = []
    for first in range(0, len(matrix), 50000):
        path = run/'evidence'/f'{name}_{first:08d}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, values=matrix[first:first+50000])
        output.append({'path': path.relative_to(run).as_posix(), 'sha256': sha(path), 'rows': len(matrix[first:first+50000])})
    return output


def evaluate(run, config, candidate, stage, state, predictions, model_records, old, discovery):
    import numpy as np
    import pandas as pd
    numbers = config['stage1_fold_numbers'] if stage == 1 else config['stage2_fold_numbers']
    frame = pd.concat([state[n]['price'] for n in numbers], ignore_index=True)
    b0 = np.concatenate([state[n]['b0_score'] for n in numbers])
    secondary = np.concatenate([predictions[n] for n in numbers])
    union = (b0 >= .75) | ((b0 < .75) & (secondary >= candidate['threshold']))
    frame['buy_prob'], frame['sell_prob'] = b0, np.float32(0)
    cohort = discovery.gated_cohort(old.semantics, frame, candidate['candidate_id'], union)
    trades, _ = old.semantics.simulate(cohort, old.semantics.SIMULATORS[-1])
    pooled, folds = metric_set(old, trades, [config['folds'][n-1] for n in numbers])
    ledger = run/'ledgers'/f"{candidate['candidate_id']}_stage{stage}.json"
    ledger.parent.mkdir(exist_ok=True)
    discovery.write_json(ledger, trades)
    row = {'candidate_id': candidate['candidate_id'], 'family': candidate['family'], 'stage': stage,
           'parameters': candidate['parameters'], 'features': candidate['features'],
           'feature_set_sha256': digest(candidate['features']), 'label_pipeline_sha256': config['label_pipeline_sha256'],
           'training_data_manifest_sha256': sha(run/'training_dataset_manifest.json'),
           'seed': candidate['seed'], 'fold_definition_sha256': digest(config['folds']),
           'threshold': candidate['threshold'], 'positive_weight_multiplier': candidate['positive_weight_multiplier'],
           'calibration': candidate['calibration'], 'models': model_records,
           'ledger_path': ledger.relative_to(run).as_posix(), 'ledger_sha256': sha(ledger),
           'fold_metrics': folds, 'pooled': pooled, 'stress_metrics': {'stress_pf': pooled['stress_pf']},
           'gate_result': gate(pooled, folds)}
    return row


def fit_candidate(run, config, candidate, number, data, discovery):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    features_used = candidate['features']
    xtrain = features(data['train_x'], config['reference_features'], features_used)
    xscore = features(data['score_x'], config['reference_features'], features_used)
    subset = np.flatnonzero(data['b0_train'] < .75)
    calibration = np.array([], dtype=np.int64)
    fit = subset
    cutoff = None
    if candidate['calibration'] != 'none':
        cutoff = (pd.Timestamp(config['folds'][number-1][1])-pd.DateOffset(months=3)).value
        fit, calibration = calibration_split(data['train_ns'], data['maturity'], data['legacy_maturity'], subset, cutoff)
    sample_weight = weights(data['target'][fit], candidate['positive_weight_multiplier'])
    model = xgb.XGBClassifier(**candidate['parameters'])
    model.fit(pd.DataFrame(xtrain[fit], columns=features_used), data['target'][fit], sample_weight=sample_weight)
    path = run/'models'/candidate['candidate_id']/f'fold{number}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not path.resolve().is_relative_to((run/'models').resolve()):
        raise PermissionError('Candidate must be new and run-local')
    model.save_model(path)
    raw = model.predict_proba(pd.DataFrame(xscore, columns=features_used))[:, 1]
    probability = raw
    calibration_record = None
    calibration_raw = np.array([], dtype=np.float32)
    if len(calibration):
        ycal = data['target'][calibration]
        if not np.array_equal(np.unique(ycal), [0, 1]):
            raise ValueError('Calibration requires both classes')
        calibration_raw = model.predict_proba(pd.DataFrame(xtrain[calibration], columns=features_used))[:, 1]
        def logit(p):
            p = np.clip(p.astype(np.float64), 1e-6, 1-1e-6)
            return np.log(p/(1-p)).reshape(-1, 1)
        estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=42)
        estimator.fit(logit(calibration_raw), ycal)
        if int(estimator.n_iter_.max()) >= 1000:
            raise ValueError('Calibration did not converge')
        probability = estimator.predict_proba(logit(raw))[:, 1]
        calibration_record = {'coefficient': float(estimator.coef_[0, 0]), 'intercept': float(estimator.intercept_[0]),
                              'cutoff_ns': int(cutoff), 'fit_count': len(fit), 'calibration_count': len(calibration)}
    evidence = run/'evidence'/f"{candidate['candidate_id']}_fold{number}.npz"
    np.savez_compressed(evidence, raw=raw, probability=probability, fit_positions=fit,
                        calibration_positions=calibration, sample_weight=sample_weight, calibration_raw=calibration_raw)
    return probability, {'fold_number': number, 'path': path.relative_to(run).as_posix(), 'sha256': sha(path),
        'evidence_path': evidence.relative_to(run).as_posix(), 'evidence_sha256': sha(evidence),
        'fit_target_sha256': discovery.array_hash(data['target'][fit]),
        'fit_feature_sha256': discovery.array_hash(xtrain[fit]),
        'calibration': calibration_record, 'model_configuration': json.loads(model.get_booster().save_config())}


def search_candidates(config, evaluate_stage, retain, begin):
    """Bounded control flow also exercised with tiny, non-model fixtures."""
    for candidate in config['candidates'][1:]:
        for stage in (1, 2):
            begin(candidate, stage)
            row = evaluate_stage(candidate, stage)
            retain(row)
            if stage == 1 and not row['gate_result']['interesting']:
                break


def research(run, config, base, data_dir):
    """Called only inside a consumed USER receipt; tests substitute all model work."""
    import numpy as np
    import pandas as pd
    import sklearn.linear_model  # Load approved native dependencies before the IO guard.
    import gold_independent_secondary_classifier_v1 as discovery
    import gold_manual_s4_secondary_retrain_v1 as reference
    from training_holdout_guard_v1 import install
    spec = discovery.specification()
    archive, helper, old = discovery.frozen_inputs(spec)
    old.drl_trading_v2.DATA_DIR = str(data_dir)
    release = install(ROOT, write_root=run)
    try:
        frame, target, folds, identities = helper.reconstruct(old)
        proxy = SimpleNamespace(**{**vars(helper), 'reconstruct': lambda unused: (frame, target, folds, identities)})
        print('[3/8] 驗證 Reference Control', flush=True)
        control, inventory = reference.train_folds(run, base, discovery, proxy, old, archive, data_dir)
        pooled = normalized(control['metrics'][-1])
        expected = read(ROOT/config['reference_run']/'model_inventory.json')
        check_reference(pooled, inventory, expected)
        write(run/'reference_control.json', {'status': 'PASS', 'pooled': pooled, 'models': inventory,
                                            'numeric_tolerance': config['numeric_tolerance']})
        space_hash = digest(config['candidates'])
        event(run, 'reference_pass', space_hash, 'F0_REFERENCE', 0)
        print('[4/8] 執行 Improvement Research', flush=True)
        labels = old.build_execution_aligned_labels(frame)
        ns = frame.TIME_DT.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        state, evidence_index = {}, []
        with np.load(run/'secondary_evidence.npz', allow_pickle=False) as source:
            for number, name, train, score in folds:
                key = f'fold{number}'
                train_x = frame.loc[train, base['feature_list']].to_numpy(dtype=np.float32)
                score_x = frame.loc[score, base['feature_list']].to_numpy(dtype=np.float32)
                data = {'train_x': train_x, 'score_x': score_x, 'target': target[train], 'train_ns': ns[train],
                        'score_ns': ns[score], 'b0_train': source[key+'_b0_train'].copy(),
                        'b0_score': source[key+'_b0_score'].copy(), 'reference_probability': source[key+'_secondary_score'].copy(),
                        'maturity': labels['C1_MATURITY_NS'].to_numpy(dtype=np.int64)[train],
                        'legacy_maturity': ns[train+old.LEGACY_HORIZON_ROWS],
                        'price': frame.loc[score, helper.PRICE_COLUMNS].copy()}
                array_path = run/'evidence'/f'fold{number}_audit.npz'
                array_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(array_path, **{k: v for k, v in data.items() if k not in ('price', 'train_x', 'score_x')},
                                    train_indices=train, score_indices=score)
                price_path = run/'evidence'/f'fold{number}_prices.npz'
                np.savez_compressed(price_path, **{k: ns[score] if k == 'TIME_DT' else data['price'][k].to_numpy()
                                                  for k in data['price'].columns})
                evidence_index.append({'fold_number': number, 'arrays_path': array_path.relative_to(run).as_posix(),
                    'arrays_sha256': sha(array_path), 'price_path': price_path.relative_to(run).as_posix(), 'price_sha256': sha(price_path),
                    'train_chunks': save_chunks(run, key+'_train', train_x), 'score_chunks': save_chunks(run, key+'_score', score_x)})
                state[number] = data
        write(run/'evidence_index.json', evidence_index)
        records = []
        def retain(row):
            with (run/'candidate_results.jsonl').open('a', encoding='utf-8') as out:
                out.write(json.dumps(row, allow_nan=False)+'\n')
            records.append(row)
            event(run, 'candidate_result', space_hash, row['candidate_id'], row['stage'], row)
        control_candidate = config['candidates'][0]
        reference_models = [{'fold_number': n, 'path': x['path'], 'sha256': x['sha256'], 'reference_model': True}
                            for n, x in enumerate(inventory, 1)]
        # Control metrics are the original full-period replay, not a new optimization.
        retain({'candidate_id': 'F0_REFERENCE', 'family': 0, 'stage': 0, 'parameters': control_candidate['parameters'],
                'features': control_candidate['features'], 'feature_set_sha256': digest(control_candidate['features']),
                'label_pipeline_sha256': config['label_pipeline_sha256'], 'training_data_manifest_sha256': sha(run/'training_dataset_manifest.json'),
                'seed': 42, 'fold_definition_sha256': digest(config['folds']), 'threshold': .75,
                'positive_weight_multiplier': 1., 'calibration': 'none', 'models': reference_models,
                'pooled': pooled, 'fold_metrics': [{'fold': x['fold'], **normalized(x)} for x in control['metrics'][:-1]],
                'stress_metrics': {'stress_pf': pooled['stress_pf']}, 'gate_result': gate(pooled, [normalized(x) for x in control['metrics'][:-1]])})
        candidate_state = {}
        def stage_evaluation(candidate, stage):
            predictions, models = candidate_state.setdefault(candidate['candidate_id'], ({}, []))
            print(f"{candidate['candidate_id']} Stage {stage}", flush=True)
            for n in ([1, 2] if stage == 1 else [3]):
                if candidate['family'] == 4:
                    predictions[n] = state[n]['reference_probability']
                    models.append(reference_models[n-1])
                else:
                    predictions[n], audit = fit_candidate(run, config, candidate, n, state[n], discovery)
                    models.append(audit)
            return evaluate(run, config, candidate, stage, state, predictions, list(models), old, discovery)
        search_candidates(config, stage_evaluation, retain,
                          lambda c, stage: event(run, 'candidate_begin', space_hash, c['candidate_id'], stage))
        if read(run/'predeclared_search_space.json') != config['candidates']:
            raise ValueError('Search space changed during research')
        print('[5/8] 建立 Pareto Frontier', flush=True)
        decision = select(records)
        write(run/'pareto_frontier.json', decision['pareto_frontier'])
        print('[6/8] 選擇 Historical Candidate', flush=True)
        chosen = next((r for r in records if r['stage'] == 2 and r['candidate_id'] == decision['selected_candidate']), None)
        write(run/'selected_candidate.json', {'classification': 'HISTORICAL_RESEARCH_CANDIDATE', 'candidate': chosen, 'production_promoted': False})
        write(run/'selected_candidate_reason.json', decision)
        for name, field in [('fold_metrics', 'fold_metrics'), ('pooled_metrics', 'pooled'), ('stress_metrics', 'stress_metrics')]:
            write(run/(name+'.json'), [{'candidate_id': r['candidate_id'], 'stage': r['stage'], field: r[field]} for r in records])
        with (run/'candidate_summary.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['candidate_id', 'family', 'stage', *METRICS, 'gate'])
            writer.writeheader()
            writer.writerows({'candidate_id': r['candidate_id'], 'family': r['family'], 'stage': r['stage'], **r['pooled'], 'gate': r['gate_result']['gate']} for r in records)
        event(run, 'research_complete', space_hash, record=decision)
        return decision, chosen, inventory
    finally:
        release()


def run_manual(token):
    from gold_manual_training_workflow_v1 import consume_receipt
    from gold_s4_secondary_improvement_launcher_v1 import verify_approval
    from manual_training_launcher_v1 import verify_environment, load, CONFIG as launcher_config
    from gold_manual_s4_train_validate_v1 import Tee
    import training_run_history as history
    from gold_manual_s4_secondary_retrain_v1 import archive_git, git
    consume_receipt(token)
    verify_environment(load(launcher_config))
    verify_approval()
    config = read(CONFIG)
    base = verify_config(config)
    commit = git('rev-parse', 'HEAD')
    if (git('status', '--porcelain') or git('branch', '--show-current') != 'main'
            or git('rev-parse', '@{u}') != commit or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit):
        raise ValueError('Clean pushed main required')
    print('[2/8] 檢查歷史資料', flush=True)
    data_dir, dataset = prepare(base)
    run = history.create_run(EXPERIMENT, Path(__file__), 'RUN_TRAINING.bat (USER double-click)',
                             arguments=[], seeds={'research': 42}, seed_note='16 predeclared candidates; frozen chronological folds')
    space_hash = freeze_space(run, config)
    event(run, 'space_frozen', space_hash)
    write(run/'research_plan.json', config)
    write(run/'training_dataset_manifest.json', dataset)
    shutil.copyfile(CONFIG, run/'approved_config.json')
    shutil.copyfile(ROOT/'validate_gold_s4_secondary_improvement_run_v1.py', run/'validator_script.py')
    manifest = read(run/'manifest.json')
    manifest.update(manual_start=True, started_by='USER_LAUNCHER', source_commit=commit, space_sha256=space_hash,
                    plan_sha256=sha(CONFIG), source_bindings={n: sha(ROOT/n) for n in read(ROOT/'gold_s4_improvement_approval_v1.json')['bindings']})
    write(run/'manifest.json', manifest)
    status = {'workflow': config['workflow_version'], 'run_id': run.name, 'reference_status': 'NOT_RUN',
              'train_status': 'FAIL', 'validator_status': 'NOT_RUN', 'final_status': 'FAIL',
              'execution_status': 'FAIL', 'research_result': 'NOT_RUN', 'candidate_gate': 'NONE',
              'reference': REFERENCE, 'candidate': None, 'production_changed': False,
              'production_promoted': False, 'holdout_used': False, 'failed_checks': []}
    before = {n: sha(ROOT/n) for n in base['protected_sha256']}
    write(run/'production_before.json', before)
    try:
        with (run/'training_stdout.txt').open('x', encoding='utf-8') as out, (run/'training_stderr.txt').open('x', encoding='utf-8') as err, contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
            decision, chosen, inventory = research(run, config, base, data_dir)
        after = {n: sha(ROOT/n) for n in base['protected_sha256']}
        write(run/'production_after.json', after)
        if before != after or after != base['protected_sha256']:
            raise ValueError('Production hash changed')
        status.update(reference_status='PASS', train_status='PASS', research_result=decision['research_result'],
                      candidate=chosen, candidate_gate=chosen['gate_result']['gate'] if chosen else 'NONE')
        write(run/'training_result.json', {**status, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
        print('[7/8] Independent Validation', flush=True)
        status['validator_status'] = 'FAIL'
        with (run/'validator_stdout.txt').open('x', encoding='utf-8') as out, (run/'validator_stderr.txt').open('x', encoding='utf-8') as err:
            process = subprocess.run([sys.executable, '-B', str(ROOT/'validate_gold_s4_secondary_improvement_run_v1.py'), str(run)], cwd=ROOT, stdout=out, stderr=err)
        validation = read(run/'validator.json')
        if validation['run_id'] != run.name:
            raise ValueError('VALIDATOR_RUN_ID_MISMATCH')
        if process.returncode or validation['overall'] != 'PASS':
            raise ValueError('; '.join(validation.get('failed_checks', [])) or 'Independent validator failed')
        status.update(validator_status='PASS', final_status='PASS', execution_status='PASS')
        manifest = read(run/'manifest.json')
        manifest['data'] = read(ROOT/config['reference_run']/'manifest.json')['data']
        manifest['data']['source_files'] = dataset['datasets']
        manifest['data']['data_sources'] = ['Certified exact historical S4 source; predeclared development research']
        if dataset.get('mt5_fetches'):
            manifest['data']['mt5_fetch'] = {'used': True, 'terminal_path': r'D:\XM2\terminal64.exe',
                'terminal_info': {'requested_path': r'D:\XM2\terminal64.exe'},
                'broker_info': {'company': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'trade_mode': 0},
                'fetch_start_utc': dataset['preparation_started_at_utc'], 'fetch_end_utc': dataset['preparation_finished_at_utc'],
                'retrieved_at_utc': dataset['preparation_finished_at_utc'],
                'returned_rows': sum(x['returned_rows'] for x in dataset['mt5_fetches']),
                'requests': dataset['mt5_fetches'], 'acceptance': 'Original export exact hash restoration only'}
        manifest['model'] = read(ROOT/config['reference_run']/'manifest.json')['model']
        # Overall archive identity is always the retained exact control; candidate bundles are separately inventoried.
        manifest['model'].update(artifact_path=inventory[-1]['path'], artifact_sha256=inventory[-1]['sha256'])
        (run/'model.sha256').write_text(inventory[-1]['sha256']+'  '+inventory[-1]['path']+'\n', encoding='utf-8')
        manifest['search'].update(performed=True, predefined_search_space=config['candidates'], candidate_results_file='candidate_summary.csv')
        # Repository candidate CSV schema differs; preserve its required columns as well.
        with (run/'candidates.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=history.CANDIDATE_COLUMNS)
            writer.writeheader()
            for row in [json.loads(line) for line in (run/'candidate_results.jsonl').read_text().splitlines()]:
                item = {key: 'N/A: see candidate_results.jsonl' for key in history.CANDIDATE_COLUMNS}
                item.update(candidate_id=row['candidate_id'], parameters=json.dumps(row['parameters'], sort_keys=True),
                            fold='stage'+str(row.get('stage', 0))+'_pooled', qualification_verdict=row.get('gate_result', {}).get('gate', 'NONE'))
                for key, source in [('executable_trades', 'trades'), ('trades_per_day', 'trades_per_day'),
                        ('realized_wr', 'realized_win_rate'), ('pf', 'profit_factor'), ('mean_r', 'mean_r'),
                        ('pnl', 'pnl_r'), ('max_dd', 'max_drawdown_r'), ('cost_stress_result', 'stress_pf')]:
                    item[key] = row.get('pooled', {}).get(source)
                writer.writerow(item)
        manifest['search']['candidate_results_file'] = 'candidates.csv'
        manifest['registry'].update({k: 'N/A: see candidate_results.jsonl' for k in history.REGISTRY_FIELDS})
        manifest['registry'].update(parent_or_incumbent=config['reference_run'], selected_configuration=decision['selected_candidate'] or 'NO_IMPROVEMENT_FOUND', validator_result='PASS')
        summary = chosen['pooled'] if chosen else REFERENCE
        manifest['registry'].update(trades_per_day=summary['trades_per_day'], realized_win_rate=summary['realized_win_rate'],
            pf=summary['profit_factor'], mean_r=summary['mean_r'], pnl=summary['pnl_r'], max_dd=summary['max_drawdown_r'])
        write(run/'manifest.json', manifest)
    except BaseException as error:
        import traceback
        with (run/'training_stderr.txt').open('a', encoding='utf-8') as out:
            traceback.print_exc(file=out)
        status['reference_status'] = 'PASS' if (run/'reference_control.json').exists() else 'FAIL'
        if not (run/'reference_control.json').exists():
            write(run/'reference_control.json', {'status': 'FAIL', 'reason': str(error)})
        status['failed_checks'] = [type(error).__name__+': '+str(error)]
        status['production_changed'] = any(not (ROOT/n).exists() or sha(ROOT/n) != h for n, h in base['protected_sha256'].items())
        write(run/'failure.json', status)
    write(run/'combined_result.json', status)
    write(run/'metrics.json', {'formal_run_status': status['final_status'], **status})
    for name in ('report.md', 'findings.md'):
        (run/name).write_text('# Historical S4 improvement research\n\n'+format_result(status)+'\n\nHistorical development evidence only; no promotion.\n', encoding='utf-8')
    for name in ('training_stdout.txt', 'training_stderr.txt', 'validator_stdout.txt', 'validator_stderr.txt'):
        (run/name).touch(exist_ok=True)
    if not (run/'training_result.json').exists():
        write(run/'training_result.json', status)
    print('[8/8] Finalize', flush=True)
    try:
        errors = history.finalize_run(run, 'research_only' if status['final_status'] == 'PASS' else 'aborted',
                                      aborted_reason='; '.join(status['failed_checks']) or None)
        if errors:
            raise ValueError('Archive validation failed: '+'; '.join(errors))
        history.register_run(run)
        archive_git(run)
    except Exception as error:
        error.result = {**status, 'final_status': 'FAIL', 'execution_status': 'FAIL',
                        'failed_checks': [*status['failed_checks'], 'Archive: '+str(error)]}
        raise
    return status


def format_result(status):
    lines = ['='*40, 'XM GOLD S4 Improvement Train + Validate', '='*40, 'RUN_ID:', status.get('run_id') or 'N/A']
    for label, key in [('Reference', 'reference_status'), ('Research', 'train_status'), ('Validation', 'validator_status'),
                       ('Final', 'final_status'), ('Result', 'research_result')]:
        lines.extend([label+':', str(status.get(key, 'NOT_RUN'))])
    lines.extend(['REFERENCE', json.dumps(REFERENCE, indent=2)])
    chosen = status.get('candidate')
    if chosen:
        lines.extend(['NEW CANDIDATE', json.dumps(chosen['pooled'], indent=2),
                      'Delta WR: '+str(chosen['gate_result']['delta_wr']),
                      'Delta Trades/Day: '+str(chosen['gate_result']['delta_trades_per_day']),
                      'Gate: '+chosen['gate_result']['gate'],
                      'Candidate Model: '+str(ROOT/'training_runs'/status['run_id']/chosen['models'][-1]['path']),
                      'SHA256: '+chosen['models'][-1]['sha256']])
    elif status.get('research_result') == 'NO_IMPROVEMENT_FOUND':
        lines.extend(['沒有找到同時提高 WR 與 Trades/Day 且通過安全條件的候選模型。',
                      'Reference 保持最佳已知 historical candidate（本次有界搜尋範圍內）。'])
    lines.extend(status.get('failed_checks', []))
    lines.extend(['Production:', '未變更' if not status.get('production_changed') else 'HASH FAILURE',
                  'Locked Future Holdout:', '未使用', '='*40])
    return '\n'.join(lines)


if __name__ == '__main__':
    raise SystemExit('USER double-click RUN_TRAINING.bat required; no direct automatic research execution')
