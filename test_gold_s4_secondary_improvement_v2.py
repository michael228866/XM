"""Synthetic/static certification only. No real dataset/model is opened or fitted."""
import ast
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.isotonic import IsotonicRegression
import gold_s4_secondary_improvement_v2_support as rules
import gold_s4_secondary_improvement_v2_launcher as launcher
import validate_gold_s4_secondary_improvement_v2_run as validator


def rejected(call):
    try:
        call()
    except (ValueError, PermissionError, KeyError, FileNotFoundError):
        return True
    return False


def synthetic_guard():
    """Permanent for this short-lived test process; disallow any real input access."""
    allowed_metadata = {'FINALIZED.json', 'validation_amendment_v1.json', 'validator_recheck_v1.json'}
    def hook(event, args):
        if event not in {'open', 'os.listdir', 'os.scandir'} or not args or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        for folder in ('future_holdout', 'historical_training_data', 'training_runs'):
            if path.is_relative_to(rules.ROOT/folder):
                if folder == 'training_runs' and event == 'open' and path.name in allowed_metadata:
                    mode = args[1] or ''
                    if not any(c in mode for c in 'wax+'):
                        return
                raise PermissionError('SYNTHETIC_TEST_MODE prohibits real inputs')
        if path.parent == rules.ROOT and path.suffix.lower() in {'.csv', '.npz', '.parquet'}:
            raise PermissionError('Synthetic tests cannot read historical exports')
    sys.addaudithook(hook)


def row(cid, wr, tpd, config=None):
    pooled = {**rules.REFERENCE, 'realized_win_rate': wr, 'trades_per_day': tpd}
    folds = [dict(trades=50, realized_win_rate=.55, profit_factor=.85) for _ in range(3)]
    return dict(candidate_id=cid, config=config or {}, pooled=pooled, fold_metrics=folds,
                gate=rules.gate(pooled, folds), economic_status=rules.economic(pooled))


def run_tests():
    os.environ['SYNTHETIC_TEST_MODE'] = 'true'
    synthetic_guard()
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    config, ref, space = rules.configuration()
    check('reference_binding_review', ref['metrics'] == rules.REFERENCE and len(ref['features']) == 27)
    check('search_space_review', len(space['candidates']) == 27)
    check('candidate_family_review', {c['family'] for c in space['candidates']} == {'CONTROL', 'A', 'B', 'C', 'D', 'E'})
    rows = [row('F5_CONTROL', .5703125, .30035197497066873), row('A', .58, .31), row('B', .59, .29), row('C', .56, .40)]
    check('selection_logic_test', rules.select(rows)['selected_candidate'] == 'A')
    check('pareto_logic_test', set(rules.select(rows)['pareto_frontier']) == {'A', 'B', 'C'})
    check('independent_decision', rules.select(rows) == validator.decision(rows))
    check('no_improvement', rules.select([rows[0], rows[2], rows[3]])['research_result'] == 'NO_IMPROVEMENT_FOUND')
    check('strict_primary_gates', not rules.gate(rules.REFERENCE, rows[0]['fold_metrics'])['interesting'])
    bad = row('bad', .65, .7)
    bad['fold_metrics'][0]['profit_factor'] = .69
    check('collapsed_fold', not rules.gate(bad['pooled'], bad['fold_metrics'])['safety_pass'])
    for wr, tpd, label in [(.58, .4, 'STRONG'), (.6, .5, 'TARGET'), (.58, .31, 'INTERESTING')]:
        check('gate_'+label, rules.gate(row('x',wr,tpd)['pooled'], rows[0]['fold_metrics'])['gate'] == label)
    for pf, mean, pnl, label in [(1.1,.1,1,'POSITIVE_EXPECTANCY'),(.9,-.1,-1,'NEGATIVE_EXPECTANCY'),(1.,0.,0.,'NEAR_BREAK_EVEN')]:
        p = {**rules.REFERENCE, 'profit_factor':pf,'mean_r':mean,'pnl_r':pnl}
        check(label, rules.economic(p) == validator.economics(p) == label)
    tied = [row('Z',.59,.35), row('A',.59,.35)]
    check('stable_exact_tie', rules.select(tied)['selected_candidate'] == 'A')
    for field in ('stress_pf','profit_factor','mean_r','max_drawdown_r'):
        altered = copy.deepcopy(tied)
        altered[0]['pooled'][field] += .001
        check('tie_'+field, rules.select(altered)['selected_candidate'] == 'Z')
    records = [row(c['candidate_id'], .5703125 if i == 0 else .58, .30035197497066873 if i == 0 else .31, c)
               for i,c in enumerate(space['candidates'])]
    validator.validate_records(space, records, rules.select(records))
    check('validator_fixture_test', True)
    for field in ('threshold', 'features', 'parameters', 'calibration', 'weighting'):
        bad = copy.deepcopy(records)
        bad[1]['config'][field] = 'mutated'
        check('config_mutation_'+field, rejected(lambda: validator.validate_records(space, bad, rules.select(records))))
    bad = copy.deepcopy(records)
    bad[0]['pooled']['wins'] += 1
    check('reference_metric_mutation', rejected(lambda: validator.validate_records(space,bad,rules.select(records))))
    check('missing_candidate', rejected(lambda: validator.validate_records(space,records[:-1],rules.select(records))))
    bad_decision = {**rules.select(records),'selected_candidate':'not-frozen'}
    check('selection_mutation', rejected(lambda: validator.validate_records(space,records,bad_decision)))
    bad = copy.deepcopy(records); bad[1]['economic_status'] = 'POSITIVE_EXPECTANCY'
    check('economic_mutation', rejected(lambda: validator.validate_records(space,bad,rules.select(records))))
    check('control_failure', rejected(lambda: rules.check_control({**rules.REFERENCE,'trades':769})))
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40,len(config['input_features']))).astype(np.float32)
    for c in space['candidates']:
        transformed = rules.features(x,config['input_features'],c['features'])
        check('causal_'+c['candidate_id'], np.array_equal(transformed[:20],rules.features(x[:20],config['input_features'],c['features']))
              and np.array_equal(transformed,validator.feature_matrix(x,config['input_features'],c['features'])))
    check('causality_review', True)
    for mode in ('NONE','BALANCED'):
        c = {**space['candidates'][0], 'weighting':mode, 'positive_weight_multiplier':1.2}
        y = np.array([0,0,0,1,1])
        w = rules.sample_weights(y,c)
        check('weights_'+mode, np.array_equal(w,np.ones(5) if mode=='NONE' else np.array([5/6,5/6,5/6,1.5,1.5])))
    raw = np.array([.1,.1,.2,.3,.4,.5,.5,.8])
    y = np.array([0,1,1,0,1,0,1,1])
    query = np.linspace(0,1,101)
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw,y)
    check('isotonic_independent_PAV',np.allclose(validator.isotonic_values(raw,y,query),iso.predict(query),rtol=0,atol=1e-12))
    import pandas as pd
    times = pd.date_range('2017-01-01', periods=365, freq='D').to_numpy(dtype='datetime64[ns]').astype(np.int64)
    data = dict(train_ns=times, maturity=times+3600_000_000_000, legacy_maturity=times+7200_000_000_000,
                b0_train=np.zeros(365),target=np.arange(365)%2)
    tiny = {**config,'minimum_calibration_samples':10,'minimum_calibration_class_samples':2}
    c = next(c for c in space['candidates'] if c['calibration']=='PLATT')
    fit,cal,cutoff = rules.split(data,c,tiny,1)
    w = rules.sample_weights(data['target'][fit],c)
    validator.check_fit_positions(data,c,tiny,1,fit,cal,w)
    check('calibration_isolation',not np.intersect1d(fit,cal).size and data['maturity'][fit].max()<cutoff)
    check('calibration_leak_rejected',rejected(lambda:validator.check_fit_positions(data,c,tiny,1,np.r_[fit,cal[0]],cal,np.r_[w,1.])))
    check('weight_mutation_rejected',rejected(lambda:validator.check_fit_positions(data,c,tiny,1,fit,cal,w*2)))
    check('insufficient_calibration',rejected(lambda:rules.split(data,c,config,1)))
    check('manual_execution_guard_test',rejected(launcher.require_session) and rejected(lambda:launcher.begin_session('invalid')))
    check('validator_no_receipt',rejected(lambda:validator.run_authorized(Path('fake'), 'invalid')))
    with patch.object(launcher,'require_session',return_value=None):
        token = launcher.validation_permit(Path('synthetic'))
        check('permit_wrong_run',rejected(lambda:launcher.consume_validation(token,Path('different'))))
        check('permit_consumed_on_failure',rejected(lambda:launcher.consume_validation(token,Path('synthetic'))))
        token = launcher.validation_permit(Path('synthetic'))
        launcher.consume_validation(token,Path('synthetic'))
        check('permit_single_use',rejected(lambda:launcher.consume_validation(token,Path('synthetic'))))
        token = launcher.validation_permit(Path('synthetic'))
        with patch.object(launcher.time,'monotonic',return_value=float('inf')):
            check('permit_expired',rejected(lambda:launcher.consume_validation(token,Path('synthetic'))))
    check('historical_loader_guard',rejected(lambda:rules.load_evidence(config,ref)))
    from manual_training_launcher_v1 import manual_parent_chain
    check('scheduler_rejected',not manual_parent_chain([dict(Name='cmd.exe',SessionId=1,ParentProcessId=2),dict(Name='taskeng.exe',ProcessId=2)],'python'))
    check('explorer_chain',manual_parent_chain([dict(Name='cmd.exe',SessionId=1,ParentProcessId=2),dict(Name='explorer.exe',ProcessId=2)],'python'))
    from training_holdout_guard_v1 import check_path
    check('holdout_guard_test',rejected(lambda:check_path(rules.ROOT/'future_holdout/gold_s4_v4/fake',rules.ROOT)))
    check('synthetic_real_data_denied',rejected(lambda:(rules.ROOT/ref['source_run']/'secondary_evidence.npz').read_bytes()))
    check('synthetic_model_denied',rejected(lambda:(rules.ROOT/ref['source_run']/ref['models'][0]['path']).read_bytes()))
    rules.check_production(config)
    check('production_protection_test',True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root/'p').write_bytes(b'changed')
        check('production_mutation_rejected',rejected(lambda:rules.check_production({'protected_sha256':{'p':'0'*64}},root)))
        import training_run_history as history
        from gold_s4_secondary_improvement_v2 import fill_manifest, format_result
        fixture = history.create_run('synthetic_v2_archive', Path(__file__), 'synthetic fixture only',
                                     seed_note='No model training', runs_root=root/'runs', root=rules.ROOT)
        (fixture/'models').mkdir()
        model = fixture/'models/fake.json'
        model.write_text('{}', encoding='utf-8')
        record = dict(path='models/fake.json',sha256=rules.sha(model))
        validator.verify_model_file(fixture,record)
        model.write_text('{"changed":true}',encoding='utf-8')
        check('model_mutation_rejected',rejected(lambda:validator.verify_model_file(fixture,record)))
        model.write_text('{}',encoding='utf-8')
        check('model_path_escape',rejected(lambda:validator.verify_model_file(fixture,dict(path='../escape',sha256='0'*64))))
        fake = copy.deepcopy(records)
        for r in fake:
            r['models'] = [dict(path='models/fake.json', sha256=rules.sha(model))]
        (fixture/'data_inventory.json').write_text(json.dumps([dict(train_start='2000-01-01',train_end='2000-12-01',score_start='2001-01-01',score_end='2001-12-01',train_rows=5,score_rows=5)]))
        manifest = history.read_json(fixture/'manifest.json')
        fill_manifest(manifest,fixture,ref,space,fake,None,history)
        history.write_json(fixture/'manifest.json',manifest)
        history.write_json(fixture/'metrics.json',{'synthetic_only':True})
        (fixture/'report.md').write_text('Synthetic archive schema test only')
        check('archive_schema_fixture',not history.finalize_run(fixture,'research_only'))
        check('archive_integrity_fixture',not history.validate_run(fixture))
        display = format_result(dict(run_id='fixture',execution_status='PASS',research_result='NO_IMPROVEMENT_FOUND',candidate_gate='NONE',economic_status='NEGATIVE_EXPECTANCY',candidate=None))
        check('no_winner_display','NO_IMPROVEMENT_FOUND' in display and 'Model path:' not in display)
    for name in ('RUN_TRAINING.bat','CHECK_STATUS.bat'):
        text = (rules.ROOT/name).read_text(encoding='utf-8')
        check('bat_'+name,'gold_s4_secondary_improvement_v2_launcher.py' in text and 'python.exe" -B' in text
              and ('--status' in text) == (name=='CHECK_STATUS.bat'))
    check('bat_static_test',True)
    for name in ('gold_s4_secondary_improvement_v2.py','validate_gold_s4_secondary_improvement_v2_run.py'):
        ast.parse((rules.ROOT/name).read_text(encoding='utf-8'))
    check('no_real_execution',launcher._SESSION is None)
    return dict(overall='PASS',checks=checks,MODEL_TRAINING_EXECUTED=False,REAL_VALIDATION_EXECUTED=False,
                REAL_RESEARCH_EXECUTED=False,HISTORICAL_DATA_USED=False)


if __name__ == '__main__':
    print(json.dumps(run_tests(),sort_keys=True))
