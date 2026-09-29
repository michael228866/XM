"""Synthetic fixtures only: no real training, prediction, or strategy replay."""
import ast
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn.linear_model
import gold_s4_secondary_improvement_logic_v1 as rules
import gold_s4_secondary_improvement_v1 as search
import validate_gold_s4_secondary_improvement_run_v1 as independent
import manual_training_launcher_v1 as launcher
import gold_manual_training_workflow_v1 as receipts
from gold_manual_s4_training_data_v1 import read, write, sha
from training_holdout_guard_v1 import check_path, check_interval, HOLDOUT_START, install

ROOT = Path(__file__).resolve().parent


def tests():
    checks = {}
    def rejected(name, function):
        try:
            function()
        except (ValueError, PermissionError, FileExistsError, FileNotFoundError):
            checks[name] = True
        else:
            checks[name] = False
    config = read(search.CONFIG)
    base = search.verify_config(config)
    checks['reference_immutable'] = all(sha(ROOT/n) == h for n, h in config['frozen_hashes'].items())
    checks['reference_constants'] = rules.REFERENCE == config['reference_metrics'] == independent.REFERENCE
    checks['family_count_bounded'] = len(config['candidates']) == 16 and {c['family'] for c in config['candidates']} == set(range(7))
    checks['deterministic_ids'] = len({c['candidate_id'] for c in config['candidates']}) == 16 and rules.digest(config['candidates']) == rules.digest(copy.deepcopy(config['candidates']))
    checks['model_family'] = [c['parameters']['max_depth'] for c in config['candidates'] if c['family'] == 1] == [3, 5, 4]
    checks['weighting_family'] = [c['positive_weight_multiplier'] for c in config['candidates'] if c['family'] == 2] == [.9, 1.1]
    checks['coarse_thresholds_fixed_reference'] = sorted([c['threshold'] for c in config['candidates'] if c['family'] in (0, 4)]) == [.70, .725, .75, .775, .80] and all(c['parameters'] == base['hyperparameters'] and c['features'] == base['feature_list'] for c in config['candidates'] if c['family'] == 4)
    checks['ablation_predeclared'] = len([c for c in config['candidates'] if c['family'] == 5]) == 3 and all(set(c['features']) < set(base['feature_list']) for c in config['candidates'] if c['family'] == 5)
    checks['six_extensions_only'] = len(config['feature_extensions']) == 6 <= 10
    checks['folds_labels_frozen'] = config['folds'] == base['fold_definition'] and config['label_pipeline_sha256'] == base['label_pipeline_sha256']
    rules.check_reference(rules.REFERENCE, [{'sha256': 'x'}], [{'sha256': 'x'}])
    checks['reference_control_accepts_exact'] = True
    rejected('reference_mismatch_stops', lambda: rules.check_reference({**rules.REFERENCE, 'realized_win_rate': .57}, [{'sha256': 'x'}], [{'sha256': 'x'}]))
    rejected('reference_model_mismatch', lambda: rules.check_reference(rules.REFERENCE, [{'sha256': 'x'}], [{'sha256': 'y'}]))
    good = {**rules.REFERENCE, 'realized_win_rate': .57, 'trades_per_day': .31}
    folds = [{'trades': 100, 'realized_win_rate': .55, 'profit_factor': .85} for _ in range(3)]
    checks['interesting'] = rules.gate(good, folds)['gate'] == 'INTERESTING'
    for name, key, value in [('requires_wr', 'realized_win_rate', rules.REFERENCE['realized_win_rate']),
            ('requires_tpd', 'trades_per_day', rules.REFERENCE['trades_per_day']), ('pf_guard', 'profit_factor', .79),
            ('mean_guard', 'mean_r', -.107), ('stress_guard', 'stress_pf', .74)]:
        checks[name] = rules.gate({**good, key: value}, folds)['gate'] == 'NONE'
    for name, key, value in [('fold_trades', 'trades', 9), ('fold_wr', 'realized_win_rate', .44), ('fold_pf', 'profit_factor', .69)]:
        altered = copy.deepcopy(folds)
        altered[0][key] = value
        checks[name] = not rules.gate(good, altered)['safety_pass']
    checks['strong'] = rules.gate({**good, 'realized_win_rate': .58, 'trades_per_day': .40}, folds)['gate'] == 'STRONG'
    checks['target'] = rules.gate({**good, 'realized_win_rate': .60, 'trades_per_day': .50}, folds)['gate'] == 'TARGET'
    for i, values in enumerate([rules.REFERENCE, good, {**good, 'profit_factor': .79}, {**good, 'realized_win_rate': .60, 'trades_per_day': .50}]):
        checks['independent_gate:'+str(i)] = rules.gate(values, folds) == independent.independent_gate(values, folds)
    def row(cid, wr, tpd, stage=2):
        return {'candidate_id': cid, 'stage': stage, 'pooled': {**good, 'realized_win_rate': wr, 'trades_per_day': tpd}, 'fold_metrics': folds}
    rows = [row('a', .58, .4), row('b', .59, .4), row('c', .58, .45), row('d', .59, .4)]
    decision = rules.select(rows)
    checks['pareto_domination'] = decision['pareto_frontier'] == ['b', 'd', 'c']
    checks['deterministic_tie_break'] = decision['selected_candidate'] == 'b' and len(decision['tie_break']) == 3
    checks['independent_pareto'] = independent.independent_selection(rows) == (decision['pareto_frontier'], decision['selected_candidate'], decision['tie_break'])
    checks['no_improvement_valid'] = rules.select([row('ref', rules.REFERENCE['realized_win_rate'], rules.REFERENCE['trades_per_day'], 0)])['research_result'] == 'NO_IMPROVEMENT_FOUND'
    x = np.arange(20*31, dtype=np.float32).reshape(20, 31)+1
    wanted = base['feature_list']+list(config['feature_extensions'])
    derived = rules.features(x, base['feature_list'], wanted)
    checks['independent_features'] = np.array_equal(derived, independent.feature_matrix(x, base['feature_list'], wanted))
    changed = x.copy()
    changed[10:] *= 10
    checks['causal_future_perturbation'] = np.array_equal(derived[:10], rules.features(changed, base['feature_list'], wanted)[:10])
    checks['causal_prefix_invariance'] = np.array_equal(derived[:10], rules.features(x[:10], base['feature_list'], wanted))
    times = np.arange(20)
    fit, calibration = rules.calibration_split(times, times+2, times+3, np.arange(20), 15)
    checks['calibration_maturity_purge'] = np.array_equal(fit, np.arange(12)) and np.array_equal(calibration, np.arange(15, 20))
    rejected('calibration_empty_rejected', lambda: rules.calibration_split(times, times+20, times+20, np.arange(20), 15))
    checks['balanced_weights_deterministic'] = np.array_equal(rules.weights(np.array([0, 0, 1], dtype=np.int8), 1.1), [.75, .75, 1.5*1.1])
    with tempfile.TemporaryDirectory(prefix='s4_improvement_') as temp:
        root = Path(temp)/'測試'
        root.mkdir()
        freeze = rules.freeze_space(root, config)
        rejected('freeze_no_overwrite', lambda: rules.freeze_space(root, config))
        rejected('max100_enforced', lambda: rules.freeze_space(root, {**config, 'candidates': config['candidates']*7}))
        rules.event(root, 'space_frozen', freeze)
        rules.event(root, 'reference_pass', freeze, 'F0_REFERENCE', 0)
        records = []
        def record(c, stage, improved):
            pooled = good if improved else rules.REFERENCE
            return {'candidate_id': c['candidate_id'], 'family': c['family'], 'parameters': c['parameters'],
                'features': c['features'], 'seed': c['seed'], 'threshold': c['threshold'], 'calibration': c['calibration'],
                'positive_weight_multiplier': c['positive_weight_multiplier'], 'stage': stage,
                'feature_set_sha256': rules.digest(c['features']), 'label_pipeline_sha256': config['label_pipeline_sha256'],
                'fold_definition_sha256': rules.digest(config['folds']), 'pooled': pooled, 'fold_metrics': folds,
                'gate_result': rules.gate(pooled, folds)}
        def retain(value):
            records.append(value)
            rules.event(root, 'candidate_result', freeze, value['candidate_id'], value['stage'], value)
        retain(record(config['candidates'][0], 0, False))
        search.search_candidates(config, lambda c, stage: record(c, stage, c['family'] == 1), retain,
                                 lambda c, stage: rules.event(root, 'candidate_begin', freeze, c['candidate_id'], stage))
        final = rules.select(records)
        rules.event(root, 'research_complete', freeze, record=final)
        events = [json.loads(s) for s in (root/'research_events.jsonl').read_text().splitlines()]
        independent.check_search_records(config, records, events)
        checks['two_stage_shortlist_and_losers_retained'] = len(records) == 19 and len([r for r in records if r['stage'] == 2]) == 3
        checks['freeze_before_evaluation_unicode'] = events[0]['event'] == 'space_frozen' and events[1]['event'] == 'reference_pass'
        rejected('losing_candidate_removal_detected', lambda: independent.check_search_records(config, records[:-1], events))
        altered = copy.deepcopy(records)
        altered[1]['threshold'] = .99
        rejected('posthoc_config_detected', lambda: independent.check_search_records(config, altered, events))
        altered = copy.deepcopy(events)
        altered[0]['space_sha256'] = 'tampered'
        rejected('posthoc_space_detected', lambda: independent.check_search_records(config, records, altered))
        rejected('holdout_path', lambda: check_path(root/'future_holdout/gold_s4_v4/rows', root))
        rejected('future_interval', lambda: check_interval(0, HOLDOUT_START, HOLDOUT_START))
        rejected('artifact_escape', lambda: independent.path_in(root, '../production'))
        run = root/'run'
        run.mkdir()
        release = install(root, write_root=run)
        try:
            rejected('production_write_guard', lambda: (root/'gemini.py').write_text('blocked'))
            (run/'allowed').write_text('fixture')
            checks['run_local_write'] = True
        finally:
            release()
        # Run the reference-control branch with an intentionally wrong synthetic result.
        fake_old = SimpleNamespace(drl_trading_v2=SimpleNamespace(DATA_DIR=''))
        fake_helper = SimpleNamespace(reconstruct=lambda old: (None, None, [], []))
        fake_discovery = SimpleNamespace(specification=lambda: {}, frozen_inputs=lambda s: (root, fake_helper, fake_old))
        import gold_manual_s4_secondary_retrain_v1 as reference
        import training_holdout_guard_v1 as guard
        wrong = {'metrics': [{'trades': 0, 'wins': 0, 'losses': 0, 'realized_wr': 0., 'trades_per_day': 0.,
                             'pf': 0., 'mean_r': 0., 'pnl_r': 0., 'max_dd_r': 0., 'cost_stress_pf': 0.}]}
        (root/'reference').mkdir()
        write(root/'reference/model_inventory.json', [{'sha256': 'expected'}])
        with patch.dict(sys.modules, {'gold_independent_secondary_classifier_v1': fake_discovery}), patch.object(reference, 'train_folds', return_value=(wrong, [{'sha256': 'wrong'}])) as control, patch.object(search, 'ROOT', root), patch.object(guard, 'install', return_value=lambda: None), patch.object(search, 'fit_candidate') as fit_call, contextlib.redirect_stdout(io.StringIO()):
            rejected('control_failure_stops_search', lambda: search.research(run, {**config, 'reference_run': 'reference'}, base, root))
            checks['reference_internal_path_called'] = control.call_count == 1 and fit_call.call_count == 0
    with patch.object(launcher, 'verify_environment'), patch.object(receipts, 'issue_receipt', return_value='synthetic'), patch.object(search, 'run_manual', return_value={'final_status': 'PASS'}) as call:
        checks['launcher_routes_improvement'] = launcher.run_training({'workflow': {'experiment_name': 'gold_s4_secondary_improvement_v1'}})['final_status'] == 'PASS' and call.call_args.args == ('synthetic',)
    # Exercise every fitting-family implementation with fake classifiers only.
    import pandas as pd
    import sklearn.linear_model
    fit_calls = []
    class FakeXGB:
        def __init__(self, **parameters):
            self.parameters = parameters
        def fit(self, x, y, sample_weight):
            fit_calls.append((self.parameters, list(x.columns), len(y), sample_weight.copy()))
            return self
        def save_model(self, path):
            Path(path).write_text('{"synthetic":true}', encoding='utf-8')
        def predict_proba(self, x):
            return np.column_stack([np.full(len(x), .4, dtype=np.float32), np.full(len(x), .6, dtype=np.float32)])
        def get_booster(self):
            return SimpleNamespace(save_config=lambda: '{}')
    class FakeLogistic:
        def __init__(self, **kwargs):
            self.coef_, self.intercept_, self.n_iter_ = np.array([[0.]]), np.array([0.]), np.array([1])
        def fit(self, x, y):
            return self
        def predict_proba(self, x):
            return np.full((len(x), 2), .5)
    times = pd.date_range('2016-07-01', '2017-12-30', periods=80).to_numpy(dtype='datetime64[ns]').astype(np.int64)
    fixture = {'train_x': np.ones((80, 31), dtype=np.float32), 'score_x': np.ones((5, 31), dtype=np.float32),
               'target': np.tile(np.array([0, 1], dtype=np.int8), 40), 'b0_train': np.full(80, .5),
               'train_ns': times, 'maturity': times+90*60*10**9, 'legacy_maturity': times+90*60*10**9}
    with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'xgboost': SimpleNamespace(XGBClassifier=FakeXGB)}), patch.object(sklearn.linear_model, 'LogisticRegression', FakeLogistic):
        root = Path(temp)
        (root/'evidence').mkdir()
        for candidate in config['candidates']:
            if candidate['family'] in (0, 4):
                continue
            probability, audit = search.fit_candidate(root, config, candidate, 1, fixture, SimpleNamespace(array_hash=independent.array_hash))
            checks['synthetic_fit:'+candidate['candidate_id']] = len(probability) == 5 and audit['sha256'] == sha(root/audit['path']) and fit_calls[-1][0] == candidate['parameters'] and fit_calls[-1][1] == candidate['features']
            if candidate['family'] == 3:
                checks['calibration_is_inner_not_outer'] = audit['calibration']['calibration_count'] > 0 and audit['calibration']['fit_count']+audit['calibration']['calibration_count'] <= 80 and np.array_equal(probability, np.full(5, .5))
    rejected('no_codex_training_receipt', lambda: receipts.consume_receipt('invalid'))
    import training_run_history as history
    import gold_manual_s4_secondary_retrain_v1 as reference
    import gold_s4_secondary_improvement_launcher_v1 as approval
    for scenario in ('PASS', 'VALIDATOR_FAIL', 'WRONG_RUN', 'CRASH', 'REFERENCE_FAIL'):
        with tempfile.TemporaryDirectory(prefix='improvement_controller_') as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            run = root/'training_runs'/'synthetic_improvement'
            local_config = copy.deepcopy(config)
            local_config['reference_run'] = 'training_runs/reference'
            (root/'training_runs/reference').mkdir(parents=True)
            write(root/'training_runs/reference/manifest.json', {'data': {}, 'model': {}})
            write(root/'config.json', local_config)
            write(root/'gold_s4_improvement_approval_v1.json', {'bindings': {}})
            (root/'validate_gold_s4_secondary_improvement_run_v1.py').write_text('# synthetic')
            (root/'protected').write_text('unchanged')
            local_base = {**base, 'protected_sha256': {'protected': sha(root/'protected')}}
            events = []
            def create(*args, **kwargs):
                events.append('create')
                run.mkdir()
                write(run/'manifest.json', {'data': {}, 'model': {}, 'search': {}, 'registry': {}})
                return run
            def fake_research(actual, *args):
                events.append('research')
                assert actual == run and (run/'predeclared_search_space.json').exists()
                if scenario == 'REFERENCE_FAIL':
                    raise ValueError('REFERENCE_CONTROL_MISMATCH')
                write(run/'reference_control.json', {'status': 'PASS'})
                (run/'models').mkdir()
                (run/'models/control.json').write_text('{}')
                (run/'candidate_results.jsonl').write_text(json.dumps({'candidate_id': 'F0_REFERENCE', 'parameters': {}})+'\n')
                return {'research_result': 'NO_IMPROVEMENT_FOUND', 'selected_candidate': None}, None, [{'path': 'models/control.json', 'sha256': sha(run/'models/control.json')}]
            def validate_process(command, **kwargs):
                events.append('validate')
                assert Path(command[-1]) == run and read(run/'training_result.json')['run_id'] == run.name
                if scenario == 'CRASH':
                    raise RuntimeError('fixture_validator_crash')
                failed = scenario == 'VALIDATOR_FAIL'
                write(run/'validator.json', {'overall': 'FAIL' if failed else 'PASS',
                    'run_id': 'wrong' if scenario == 'WRONG_RUN' else run.name,
                    'failed_checks': ['fixture_check'] if failed else []})
                return SimpleNamespace(returncode=1 if failed else 0)
            def finalize(actual, status, **kwargs):
                events.append('finalize')
                combined = read(actual/'combined_result.json')
                assert ('validate' in events) == (scenario != 'REFERENCE_FAIL')
                assert combined['final_status'] == ('PASS' if scenario == 'PASS' else 'FAIL')
                return []
            for obj, name, value in [(search, 'ROOT', root), (search, 'CONFIG', root/'config.json'),
                    (search, 'verify_config', lambda c: local_base), (search, 'prepare', lambda c: (root, {'datasets': []})),
                    (receipts, 'consume_receipt', lambda t: None), (approval, 'verify_approval', lambda: None),
                    (launcher, 'verify_environment', lambda c: None), (launcher, 'load', lambda p: {}),
                    (reference, 'git', lambda *a: '' if a[0] == 'status' else 'main' if a[0] == 'branch' else 'commit'),
                    (reference, 'archive_git', lambda p: events.append('archive')), (search, 'research', fake_research),
                    (history, 'create_run', create), (history, 'finalize_run', finalize),
                    (history, 'register_run', lambda p: events.append('register')),
                    (search.subprocess, 'run', validate_process)]:
                stack.enter_context(patch.object(obj, name, value))
            with contextlib.redirect_stdout(io.StringIO()):
                status = search.run_manual('synthetic')
            checks['orchestration:'+scenario] = status['final_status'] == ('PASS' if scenario == 'PASS' else 'FAIL')
            checks['finalize_order:'+scenario] = events[-3:] == ['finalize', 'register', 'archive']
            if scenario == 'PASS':
                checks['no_improvement_execution_PASS'] = status['research_result'] == 'NO_IMPROVEMENT_FOUND' and status['train_status'] == status['validator_status'] == 'PASS'
            if scenario == 'REFERENCE_FAIL':
                checks['reference_failure_NOT_RUN'] = status['research_result'] == 'NOT_RUN' and 'validate' not in events
    source = (ROOT/'gold_s4_secondary_improvement_v1.py').read_text(encoding='utf-8')
    checks['same_run_validator_argument'] = "'validate_gold_s4_secondary_improvement_run_v1.py'), str(run)" in source
    checks['validator_before_finalization'] = source.index("process = subprocess.run(") < source.index('history.finalize_run(')
    checks['validator_failure_blocks_pass'] = "if process.returncode or validation['overall'] != 'PASS':" in source and 'VALIDATOR_RUN_ID_MISMATCH' in source
    tree = ast.parse((ROOT/'validate_gold_s4_secondary_improvement_run_v1.py').read_bytes())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    checks['independent_validator_no_trainer_import'] = not any('secondary_improvement' in n or 'secondary_retrain' in n or 'independent_secondary_classifier' in n for n in imports)
    checks['validator_never_fits'] = not any(isinstance(n, ast.Attribute) and n.attr == 'fit' for n in ast.walk(tree))
    policy = read(ROOT/'manual_training_policy_v1.json')
    checks['no_automatic_training_policy'] = policy['training_execution_owner'] == 'USER' and all(policy[k] is False for k in ('automatic_training_by_codex', 'automatic_training_by_scheduler', 'automatic_training_on_startup'))
    bat = (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
    checks['single_bat_no_args_pause'] = 'manual_training_launcher_v1.py' in bat and 'RUN_VALIDATION' not in bat and '%1' not in bat and 'pause >nul' in bat
    smoke = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT/'RUN_TRAINING.bat')], cwd=ROOT, input='\n', capture_output=True, encoding='utf-8', errors='replace', timeout=30)
    checks['bat_automation_denied_and_paused'] = '請按任意鍵關閉' in smoke.stdout and '[失敗]' in smoke.stderr
    checks['no_automatic_promotion'] = all(c.get('production_promoted', False) is False for c in [config])
    return {'overall': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'test_count': len(checks), 'failed_checks': [k for k, v in checks.items() if not v],
            'model_training_executed': False, 'strategy_replay_executed': False, 'holdout_used': False}


if __name__ == '__main__':
    value = tests()
    print(json.dumps(value, ensure_ascii=False))
    raise SystemExit(0 if value['overall'] == 'PASS' else 1)
