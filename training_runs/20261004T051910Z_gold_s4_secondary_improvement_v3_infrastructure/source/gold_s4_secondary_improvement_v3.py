"""Real v3 execution requires a USER receipt; imports never execute research."""
import contextlib
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from gold_s4_secondary_improvement_v3_support import (
    ROOT, CONFIG, EXPERIMENT, REFERENCE, METRICS, configuration, require, read, sha,
    digest, features, sample_weights, split, array_hash, gate, select, economic,
    check_control, check_production, load_evidence, event, signals as entry_signals,
)
from gold_manual_s4_training_data_v1 import write


def fit_candidate(run, config, c, n, data):
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    train = features(data['train_x'], config['input_features'], c['features'])
    score = features(data['score_x'], config['input_features'], c['features'])
    fit, cal, cutoff = split(data, c, config, n)
    weight = sample_weights(data['target'][fit], c)
    model = xgb.XGBClassifier(**c['parameters'])
    model.fit(pd.DataFrame(train[fit], columns=c['features']), data['target'][fit], sample_weight=weight)
    path = run/'models'/c['candidate_id']/f'fold{n}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Never overwrite model')
    model.save_model(path)
    raw = model.predict_proba(pd.DataFrame(score, columns=c['features']))[:, 1]
    probability, info, calraw = raw, None, np.array([], dtype=np.float32)
    if len(cal):
        calraw = model.predict_proba(pd.DataFrame(train[cal], columns=c['features']))[:, 1]
        y = data['target'][cal]
        info = dict(cutoff_ns=int(cutoff), fit_count=len(fit), calibration_count=len(cal))
        if c['calibration'] == 'PLATT':
            def logit(p):
                p = np.clip(p.astype(np.float64), 1e-6, 1-1e-6)
                return np.log(p/(1-p)).reshape(-1, 1)
            estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=42)
            estimator.fit(logit(calraw), y)
            require(int(estimator.n_iter_.max()) < 1000, 'Calibration convergence')
            probability = estimator.predict_proba(logit(raw))[:, 1]
            info.update(coefficient=float(estimator.coef_[0, 0]), intercept=float(estimator.intercept_[0]))
        else:
            estimator = IsotonicRegression(out_of_bounds='clip')
            estimator.fit(calraw.astype(np.float64), y)
            probability = estimator.predict(raw.astype(np.float64))
            info.update(x_thresholds=estimator.X_thresholds_.tolist(), y_thresholds=estimator.y_thresholds_.tolist())
    evidence = run/'evidence'/f"{c['candidate_id']}_fold{n}.npz"
    evidence.parent.mkdir(exist_ok=True)
    np.savez_compressed(evidence, raw=raw, probability=probability, fit_positions=fit,
                        calibration_positions=cal, sample_weight=weight, calibration_raw=calraw)
    return probability, dict(fold_number=n, path=path.relative_to(run).as_posix(), sha256=sha(path),
        evidence_path=evidence.relative_to(run).as_posix(), evidence_sha256=sha(evidence),
        fit_feature_sha256=array_hash(train[fit]), fit_target_sha256=array_hash(data['target'][fit]),
        calibration=info, model_configuration=json.loads(model.get_booster().save_config()))


def metrics(trades, days):
    import numpy as np
    net = np.array([t['net_r'] for t in trades], dtype=float)
    stress = np.array([t['stress_r'] for t in trades], dtype=float)
    def pf(v):
        loss = -v[v <= 0].sum()
        gain = v[v > 0].sum()
        return float(gain/loss) if loss else (None if gain > 0 else 0.)
    curve = np.r_[0., net.cumsum()]
    return dict(trades=len(net), wins=int((net > 0).sum()), losses=int((net <= 0).sum()),
                realized_win_rate=float((net > 0).mean()) if len(net) else 0., trades_per_day=len(net)/days,
                profit_factor=pf(net), mean_r=float(net.mean()) if len(net) else 0., pnl_r=float(net.sum()),
                max_drawdown_r=float((curve-np.maximum.accumulate(curve)).min()), stress_pf=pf(stress))


def evaluate(run, config, c, state, probabilities, models):
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import numpy as np
    import pandas as pd
    import gold_gemini_execution_semantics_v1 as semantics
    frame = pd.concat([state[n]['price'] for n in (1, 2, 3)], ignore_index=True)
    b0 = np.concatenate([state[n]['b0_score'] for n in (1, 2, 3)])
    secondary = np.concatenate(probabilities)
    signals = entry_signals(b0, secondary, c['threshold'])
    frame['buy_prob'], frame['sell_prob'] = b0, np.float32(0)
    cohort = semantics.finalize_cohort(frame, c['candidate_id'], offset_hours=0)
    times = cohort.decision_time_api.to_numpy(dtype='datetime64[ns]')
    gap = np.r_[True, np.diff(times).astype('timedelta64[s]').astype(np.int64) > 120]
    episode = np.cumsum(signals & (np.r_[False, ~signals[:-1]] | gap)).astype(np.int64)-1
    episode[~signals] = -1
    cohort['raw_signal'], cohort['raw_episode_id'] = signals, episode
    trades, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
    days, folds = 0, []
    for name, start, end in config['folds']:
        count = (pd.Timestamp(end)-pd.Timestamp(start)).days
        days += count
        folds.append(dict(fold=name, **metrics([t for t in trades if start <= t['entry_time_api'] < end], count)))
    pooled = metrics(trades, days)
    ledger = run/'ledgers'/(c['candidate_id']+'.json')
    ledger.parent.mkdir(exist_ok=True)
    write(ledger, trades)
    return dict(candidate_id=c['candidate_id'], config=c, models=models, pooled=pooled, fold_metrics=folds,
                gate=gate(pooled, folds), economic_status=economic(pooled) if pooled['profit_factor'] is not None else 'NEAR_BREAK_EVEN',
                ledger_path=ledger.relative_to(run).as_posix(), ledger_sha256=sha(ledger))


def reference_predictions(run, config, c, state, ref):
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import pandas as pd
    import xgboost as xgb
    probabilities, models = [], []
    for m in ref['models']:
        src = ROOT/ref['source_run']/m['path']
        require(sha(src) == m['sha256'], 'A_NO_LONG_HTF model hash')
        path = run/'models'/c['candidate_id']/f"fold{m['fold_number']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
        model = xgb.XGBClassifier()
        model.load_model(path)
        x = features(state[m['fold_number']]['score_x'], config['input_features'], c['features'])
        probabilities.append(model.predict_proba(pd.DataFrame(x, columns=c['features']))[:, 1])
        models.append(dict(fold_number=m['fold_number'], path=path.relative_to(run).as_posix(), sha256=sha(path)))
    return probabilities, models


def research(run, config, ref, space):
    from gold_s4_secondary_improvement_v3_launcher import require_session
    require_session()
    import pandas as pd
    import xgboost as xgb
    state = load_evidence(config, ref)
    write(run/'data_inventory.json', [dict(fold=n, train_rows=len(d['target']), score_rows=len(d['score_ns']),
          train_start=str(pd.Timestamp(int(d['train_ns'].min()))), train_end=str(pd.Timestamp(int(d['train_ns'].max()))),
          score_start=str(pd.Timestamp(int(d['score_ns'].min()))), score_end=str(pd.Timestamp(int(d['score_ns'].max())))) for n, d in state.items()])
    space_hash = digest(space)
    event(run, 'space_frozen', space_hash)
    c = space['candidates'][0]
    print('[3/9] 驗證 A_NO_LONG_HTF Reference', flush=True)
    probabilities, models = reference_predictions(run, config, c, state, ref)
    control = evaluate(run, config, c, state, probabilities, models)
    check_control(control['pooled'])
    write(run/'reference_control.json', dict(status='PASS', **control))
    event(run, 'reference_pass', space_hash, c['candidate_id'], record=control)
    records = []
    def retain(row):
        with (run/'candidate_results.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, allow_nan=False)+'\n')
        records.append(row)
        event(run, 'candidate_result', space_hash, row['candidate_id'], record=row)
    retain(control)
    for c in space['candidates'][1:]:
        print('[4/9] 執行 v3 Candidate Training: '+c['candidate_id'], flush=True)
        event(run, 'candidate_begin', space_hash, c['candidate_id'])
        if c['reuse_reference_model']:
            predictions, inventory = reference_predictions(run, config, c, state, ref)
        else:
            predictions, inventory = [], []
            for n in (1, 2, 3):
                p, model = fit_candidate(run, config, c, n, state[n])
                predictions.append(p)
                inventory.append(model)
        print('[5/9] Historical Validation: '+c['candidate_id'], flush=True)
        retain(evaluate(run, config, c, state, predictions, inventory))
    require(read(run/'predeclared_search_space.json') == space, 'Search space mutated')
    print('[6/9] Safety Gates', flush=True)
    print('[7/9] 建立 Pareto Frontier', flush=True)
    decision = select(records)
    write(run/'selection.json', decision)
    write(run/'pareto_frontier.json', decision['pareto_frontier'])
    chosen = next((r for r in records if r['candidate_id'] == decision['selected_candidate']), None)
    write(run/'selected_candidate.json', dict(candidate=chosen, classification='HISTORICAL_RESEARCH_CANDIDATE', production_promoted=False))
    event(run, 'research_complete', space_hash, record=decision)
    return decision, chosen, records


def run_manual(token):
    from gold_s4_secondary_improvement_v3_launcher import begin_session, end_session, verify_approval, validation_permit
    begin_session(token)
    try:
        return _run()
    finally:
        end_session()


def _run():
    from gold_s4_secondary_improvement_v3_launcher import require_session, verify_approval, validation_permit
    require_session()
    import training_run_history as history
    from gold_manual_s4_train_validate_v1 import Tee
    from training_holdout_guard_v1 import install
    import numpy
    import pandas
    import xgboost
    import sklearn.linear_model
    import sklearn.isotonic
    import gold_gemini_execution_semantics_v1
    from validate_gold_s4_secondary_improvement_v3_run import run_authorized
    approval = verify_approval()
    config, ref, space = configuration()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    commit = git('rev-parse', 'HEAD')
    require(not git('status', '--porcelain') and git('branch', '--show-current') == 'main'
            and git('ls-remote', 'origin', 'refs/heads/main').split()[0] == commit, 'Clean pushed main required')
    print('[2/9] 檢查 Historical Data', flush=True)
    run = history.create_run(EXPERIMENT, Path(__file__), 'RUN_TRAINING.bat (USER double-click)',
                             seeds={'research': '42'}, seed_note='23 frozen candidates; three original folds')
    for source, dest in [(CONFIG, 'approved_config.json'), (config['search_space'], 'predeclared_search_space.json'),
                         (config['reference_binding'], 'reference_binding.json'), ('validate_gold_s4_secondary_improvement_v3_run.py', 'validator_script.py')]:
        shutil.copyfile(ROOT/source, run/dest)
    manifest = read(run/'manifest.json')
    manifest.update(manual_start=True, started_by='USER_LAUNCHER', infrastructure_only=False,
                    source_bindings=approval['bindings'], source_commit=commit)
    write(run/'manifest.json', manifest)
    status = dict(run_id=run.name, execution_status='FAIL', final_status='FAIL', research_result='NOT_RUN',
                  candidate_gate='NONE', economic_status='NEGATIVE_EXPECTANCY', candidate=None,
                  production_changed=False, production_promoted=False, holdout_used=False, failed_checks=[])
    release = None
    try:
        release = install(ROOT, write_root=run)
        with (run/'training_stdout.txt').open('x', encoding='utf-8') as out, (run/'training_stderr.txt').open('x', encoding='utf-8') as err, contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
            decision, chosen, records = research(run, config, ref, space)
        write(run/'training_result.json', dict(research_result=decision['research_result'], candidate=chosen, run_id=run.name))
        status.update(research_result=decision['research_result'], candidate=chosen,
                      candidate_gate=chosen['gate']['gate'] if chosen else 'NONE',
                      economic_status=chosen['economic_status'] if chosen else economic(REFERENCE))
        fill_manifest(manifest, run, ref, space, records, chosen, history)
        manifest['registry']['validator_result'] = 'PENDING'
        write(run/'manifest.json', manifest)
        print('[8/9] Independent Validation', flush=True)
        with (run/'validator_stdout.txt').open('x', encoding='utf-8') as out, (run/'validator_stderr.txt').open('x', encoding='utf-8') as err, contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
            result = run_authorized(run, validation_permit(run))
        require(result['overall'] == 'PASS', 'Independent validator failed: '+repr(result))
        check_production(config)
        status.update(execution_status='PASS', final_status='PASS', research_result=decision['research_result'], candidate=chosen,
                      candidate_gate=chosen['gate']['gate'] if chosen else 'NONE',
                      economic_status=chosen['economic_status'] if chosen else economic(REFERENCE))
        manifest['registry']['validator_result'] = 'PASS'
    except BaseException as error:
        import traceback
        with (run/'failure_traceback.txt').open('a', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        status['failed_checks'] = [str(error)]
        manifest['registry']['validator_result'] = 'FAIL'
        write(run/'failure.json', status)
    finally:
        if release:
            release()
    check_production(config)
    write(run/'manifest.json', manifest)
    write(run/'combined_result.json', status)
    write(run/'metrics.json', status)
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(format_result(status)+'\nHistorical development only; not untouched promotion evidence.\n', encoding='utf-8')
    print('[9/9] Finalize', flush=True)
    errors = history.finalize_run(run, 'research_only' if status['execution_status'] == 'PASS' else 'aborted',
                                  aborted_reason='; '.join(status['failed_checks']) or None)
    require(not errors, 'Archive metadata: '+repr(errors))
    history.register_run(run)
    subprocess.run(['git', 'add', '-f', '--', run.relative_to(ROOT).as_posix(), 'TRAINING_RUNS.md'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'Archive user-started GOLD S4 improvement v3 '+run.name], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    require(git('ls-remote', 'origin', 'refs/heads/main').split()[0] == git('rev-parse', 'HEAD'), 'Remote archive verification')
    return status


def fill_manifest(m, run, ref, space, records, chosen, history):
    m['data'].update(symbols=['GOLD#'], data_sources=['Frozen certified historical feature/label/price evidence'],
        source_files=[dict(path=ref[key]+'/FINALIZED.json', sha256=ref['bindings'][ref[key]+'/FINALIZED.json'], retention_status='existing_immutable_archive')
                      for key in ('source_run','evidence_source_run')],
        timezone='UTC', raw_snapshot_retained=True, reproducibility_claim='sealed certified evidence',
        purge_details='Original C1 and legacy maturity before training end; inner calibration purged', embargo_details='Original frozen folds')
    inputs = read(run/'data_inventory.json')
    m['data'].update(data_start_utc=min(x['train_start'] for x in inputs), data_end_utc=max(x['score_end'] for x in inputs),
        train_start_utc=min(x['train_start'] for x in inputs), train_end_utc=max(x['train_end'] for x in inputs),
        validation_start_utc=min(x['score_start'] for x in inputs), validation_end_utc=max(x['score_end'] for x in inputs),
        test_start_utc='NOT_APPLICABLE_NO_UNTOUCHED_TEST', test_end_utc='NOT_APPLICABLE_NO_UNTOUCHED_TEST',
        train_rows=sum(x['train_rows'] for x in inputs), validation_rows=sum(x['score_rows'] for x in inputs), test_rows=0)
    m['data']['mt5_fetch']['not_applicable_reason'] = 'No broker fetch; sealed inputs only'
    representative = chosen or records[1]
    model = representative['models'][-1]
    c = representative['config']
    m['model'].update(trained=True, model_type='XGBClassifier', parameters=c['parameters'],
        boosted_rounds_or_estimators=c['parameters']['n_estimators'], features=c['features'], feature_count=len(c['features']),
        label_definition='Frozen C1 target; C1 and legacy maturity purges', horizon='Frozen discovery horizon',
        label_tp_sl_semantics='Frozen discovery label semantics', execution_tp_sl_semantics='Frozen S5 execution',
        calibration_method=c['calibration'], artifact_path=model['path'], artifact_sha256=model['sha256'],
        retention_status='stored_in_run_directory; representative only, not a production selection')
    (run/'model.sha256').write_text(model['sha256']+'  '+model['path']+'\n', encoding='utf-8')
    m['candidate_training_executed'] = True
    m['search'].update(performed=True, predefined_search_space=space, candidate_results_file='candidates.csv')
    with (run/'candidates.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=history.CANDIDATE_COLUMNS)
        writer.writeheader()
        for row in records:
            item = dict.fromkeys(history.CANDIDATE_COLUMNS, 'N/A: see candidate_results.jsonl')
            item.update(candidate_id=row['candidate_id'], parameters=json.dumps(row['config'], sort_keys=True), fold='pooled_three_folds', qualification_verdict=row['gate']['gate'])
            for key, source in [('executable_trades','trades'), ('trades_per_day','trades_per_day'), ('realized_wr','realized_win_rate'), ('pf','profit_factor'), ('mean_r','mean_r'), ('pnl','pnl_r'), ('max_dd','max_drawdown_r'), ('cost_stress_result','stress_pf')]:
                item[key] = row['pooled'][source]
            writer.writerow(item)
    m['registry'].update(dict.fromkeys(history.REGISTRY_FIELDS, 'See candidate_results.jsonl'))
    m['registry'].update(parent_or_incumbent=ref['source_run_id'], selected_configuration=chosen['candidate_id'] if chosen else 'NO_IMPROVEMENT_FOUND', validator_result='PASS')
    p = chosen['pooled'] if chosen else REFERENCE
    m['registry'].update(trades_per_day=p['trades_per_day'], realized_win_rate=p['realized_win_rate'], pf=p['profit_factor'], mean_r=p['mean_r'], pnl=p['pnl_r'], max_dd=p['max_drawdown_r'])


def format_result(status):
    lines = ['XM GOLD S4 Improvement v3 — Train + Validate', 'RUN_ID: '+status['run_id'],
             'Execution: '+status['execution_status'], 'Research Result: '+status['research_result'],
             'Gate: '+status['candidate_gate'], 'Economic Status: '+status['economic_status'],
             'REFERENCE A_NO_LONG_HTF:', json.dumps(REFERENCE, indent=2)]
    c = status.get('candidate')
    if c:
        lines += ['NEW CANDIDATE:', json.dumps(c['pooled'], indent=2)]
        for key in ('realized_win_rate', 'trades_per_day', 'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf'):
            lines.append('Delta '+key+': '+str(c['pooled'][key]-REFERENCE[key]))
        lines += ['Candidate ID: '+c['candidate_id'], 'Model path: '+str(ROOT/'training_runs'/status['run_id']/c['models'][-1]['path']), 'SHA256: '+c['models'][-1]['sha256']]
    lines += status.get('failed_checks', [])
    lines += ['Production: 未變更', 'Locked Future Holdout: 未使用']
    return '\n'.join(lines)


if __name__ == '__main__':
    raise SystemExit('USER double-click RUN_TRAINING.bat required; direct real execution prohibited')
