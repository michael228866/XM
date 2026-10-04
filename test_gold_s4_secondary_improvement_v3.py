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
import gold_s4_secondary_improvement_v3_support as rules
import gold_s4_secondary_improvement_v3_launcher as launcher
import validate_gold_s4_secondary_improvement_v3_run as validator


def rejected(call):
    try:
        call()
    except (ValueError, PermissionError, KeyError, FileNotFoundError):
        return True
    return False


def synthetic_guard():
    """Permanent for this short-lived test process; disallow any real input access."""
    allowed_metadata = set(rules.read(rules.ROOT/'gold_s4_a_no_long_htf_reference_v1.json')['bindings'])
    def hook(event, args):
        if event not in {'open', 'os.listdir', 'os.scandir'} or not args or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        for folder in ('future_holdout', 'historical_training_data', 'training_runs'):
            if path.is_relative_to(rules.ROOT/folder):
                if folder == 'training_runs' and event == 'open' and path.relative_to(rules.ROOT).as_posix() in allowed_metadata:
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
    check('reference_binding_review', ref['metrics'] == rules.REFERENCE and len(ref['features']) == 24)
    check('search_space_review', len(space['candidates']) == 23)
    check('candidate_family_review', {c['family'] for c in space['candidates']} == {'CONTROL','H','HC','T','CAL','W','X'})
    rows = [row('A_NO_LONG_HTF_CONTROL', rules.REFERENCE['realized_win_rate'], rules.REFERENCE['trades_per_day']), row('A', .58, .31), row('B', .59, .29), row('C', .56, .40)]
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
    for field in ('stress_pf','profit_factor','mean_r','pnl_r','max_drawdown_r'):
        altered = copy.deepcopy(tied)
        altered[0]['pooled'][field] += .001
        check('tie_'+field, rules.select(altered)['selected_candidate'] == 'Z')
    records = [row(c['candidate_id'], rules.REFERENCE['realized_win_rate'] if i == 0 else .58, rules.REFERENCE['trades_per_day'] if i == 0 else .31, c)
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
    check('feature_causality_review', True)
    for mode in ('NONE','BALANCED'):
        c = {**space['candidates'][0], 'weighting':mode, 'positive_weight_multiplier':1.2}
        y = np.array([0,0,0,1,1])
        w = rules.sample_weights(y,c)
        check('weights_'+mode, np.array_equal(w,np.ones(5) if mode=='NONE' else np.array([5/6,5/6,5/6,1.5,1.5])))
    raw = np.array([.1,.1,.2,.3,.4,.5,.5,.8])
    y = np.array([0,1,1,0,1,0,1,1])
    query = np.linspace(0,1,101)
    expected_iso = np.interp(query,[.1,.2,.3,.4,.5,.8],[.5,.5,.5,2/3,2/3,1.])
    check('isotonic_independent_PAV',np.allclose(validator.isotonic_values(raw,y,query),expected_iso,rtol=0,atol=1e-12))
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
        from gold_s4_secondary_improvement_v3 import fill_manifest, format_result
        fixture = history.create_run('synthetic_v3_archive', Path(__file__), 'synthetic fixture only',
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
        check('bat_'+name,'gold_s4_secondary_improvement_v3_launcher.py' in text and 'python.exe" -B' in text
              and ('--status' in text) == (name=='CHECK_STATUS.bat'))
    check('bat_static_test',True)
    for name in ('gold_s4_secondary_improvement_v3.py','validate_gold_s4_secondary_improvement_v3_run.py'):
        ast.parse((rules.ROOT/name).read_text(encoding='utf-8'))
    check('no_real_execution',launcher._SESSION is None)
    checks.update(extra_tests(config,ref,space,records))
    return dict(overall='PASS',checks=checks,MODEL_TRAINING_EXECUTED=False,REAL_VALIDATION_EXECUTED=False,
                REAL_RESEARCH_EXECUTED=False,HISTORICAL_DATA_USED=False)


def extra_tests(config, ref, space, rows):
    from gold_s4_secondary_improvement_v3 import fit_candidate, research, reference_predictions
    checks = {}
    def check(name, value):
        assert value, name
        checks[name] = True
    check('fit_without_session', rejected(lambda:fit_candidate(None,None,None,None,None)))
    check('research_without_session', rejected(lambda:research(None,None,None,None)))
    check('reference_inference_without_session', rejected(lambda:reference_predictions(None,None,None,None,None)))
    check('independent_loader_guard', rejected(lambda:validator.load_evidence(config,ref)))
    a,b,c = row('A',.575,.315),row('B',.58,.300),row('C',.570,.34)
    a['pooled']['profit_factor']=.88
    b['pooled']['profit_factor']=1.05
    c['pooled']['profit_factor']=.95
    check('required_primary_example',rules.select([a,b,c])['selected_candidate']=='A')
    check('economic_cannot_override_primary',not rules.gate(b['pooled'],b['fold_metrics'])['interesting'])
    for field in rules.METRICS:
        bad={**rules.REFERENCE,field:float('nan')}
        check('nonfinite_'+field,not rules.gate(bad,a['fold_metrics'])['safety_pass'])
    for threshold in (.70,.72,.74,.75,.76,.78,.80):
        b0=np.array([.75,.749,.2]); sec=np.array([0,threshold,threshold-.001])
        check('threshold_'+str(threshold),np.array_equal(rules.signals(b0,sec,threshold),[True,True,False]))
    check('threshold_logic_test',True)
    check('weighting_logic_test',True)
    check('calibration_isolation_test',True)
    for candidate in space['candidates']:
        if candidate['family'] == 'W':
            y=np.array([0,0,1])
            expected=np.array([.75,.75,1.5*candidate['positive_weight_multiplier']])
            check('frozen_weight_'+candidate['candidate_id'],np.array_equal(rules.sample_weights(y,candidate),expected))
    zero=np.zeros((3,len(config['input_features'])),dtype=np.float32)
    check('zero_volatility_features',np.isfinite(rules.features(zero,config['input_features'],space['derived_features'])).all())
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); decision=rules.select(rows); sh=rules.digest(space)
        rules.event(root,'space_frozen',sh)
        rules.event(root,'reference_pass',sh,rows[0]['candidate_id'],record=rows[0])
        rules.event(root,'candidate_result',sh,rows[0]['candidate_id'],record=rows[0])
        for item in rows[1:]:
            rules.event(root,'candidate_begin',sh,item['candidate_id'])
            rules.event(root,'candidate_result',sh,item['candidate_id'],record=item)
        rules.event(root,'research_complete',sh,record=decision)
        events=[json.loads(line) for line in (root/'research_events.jsonl').read_text().splitlines()]
        check('event_valid',validator.check_events(events,rows,decision,space))
        check('reference_exact_payload',events[1]['record_sha256']==rules.digest(rows[0]))
        check('postfreeze_append',rejected(lambda:rules.event(root,'candidate_begin',sh,'extra')))
        for field,value in [('record_sha256','0'*64),('sequence',99),('at_utc','2000-01-01T00:00:00+00:00'),('previous_sha256','0'*64)]:
            bad=copy.deepcopy(events); bad[1][field]=value
            bad[1]['event_sha256']=validator.digest({k:v for k,v in bad[1].items() if k!='event_sha256'})
            check('event_mutation_'+field,rejected(lambda:validator.check_events(bad,rows,decision,space)))
        bad=copy.deepcopy(events);bad[1]['record_sha256']=rules.digest(dict(status='PASS',**rows[0]))
        check('old_wrapper_bug_rejected',rejected(lambda:validator.check_events(bad,rows,decision,space)))
        duplicate=copy.deepcopy(events); previous=None
        for e in duplicate:
            e['at_utc']=duplicate[0]['at_utc'];e['previous_sha256']=previous
            e['event_sha256']=validator.digest({k:v for k,v in e.items() if k!='event_sha256'})
            previous=e['event_sha256']
        check('duplicate_timestamps',validator.check_events(duplicate,rows,decision,space))
        check('line_endings',validator.check_events([json.loads(line) for line in ('\r\n'.join(json.dumps(e) for e in events)).splitlines()],rows,decision,space))
        for kind in ('validation_complete','archive_complete'):
            check('lifecycle_append_denied_'+kind,rejected(lambda:validator.check_events(events+[{'event':kind}],rows,decision,space)))
            (root/(kind+'.json')).write_text(json.dumps({'event':kind,'research_tip':events[-1]['event_sha256']}))
            check('separate_lifecycle_'+kind,validator.check_events(events,rows,decision,space))
    check('event_chain_test',True)
    # Exercise the actual research coordinator only with fake inputs/evaluators.
    # No estimator, history loader, inference or simulator is permitted here.
    import contextlib
    import io
    import gold_s4_secondary_improvement_v3 as trainer
    fake_data={n:dict(target=np.array([0,1]),train_ns=np.array([1,2]),
                     score_ns=np.array([3,4])) for n in (1,2,3)}
    def fake_evaluate(run, cfg, candidate, state, predictions, models):
        return copy.deepcopy(next(r for r in rows if r['candidate_id']==candidate['candidate_id']))
    with tempfile.TemporaryDirectory() as directory:
        run=Path(directory)
        (run/'predeclared_search_space.json').write_text(json.dumps(space))
        with patch.object(launcher,'require_session'), patch.object(trainer,'load_evidence',return_value=fake_data), \
             patch.object(trainer,'reference_predictions',return_value=([],[])) as reused, \
             patch.object(trainer,'fit_candidate',return_value=(np.array([]),{})) as fitted, \
             patch.object(trainer,'evaluate',side_effect=fake_evaluate), contextlib.redirect_stdout(io.StringIO()):
            chosen, _, recorded=trainer.research(run,config,ref,space)
        chain=[json.loads(line) for line in (run/'research_events.jsonl').read_text().splitlines()]
        check('trainer_validator_event_integration',validator.check_events(chain,recorded,chosen,space))
        check('control_threshold_never_fit',reused.call_count==7 and fitted.call_count==48
              and all(not call.args[2]['reuse_reference_model'] for call in fitted.call_args_list))
    with tempfile.TemporaryDirectory() as directory:
        def wrong_control(*args):
            item=copy.deepcopy(rows[0]);item['pooled']['trades']+=1;return item
        with patch.object(launcher,'require_session'), patch.object(trainer,'load_evidence',return_value=fake_data), \
             patch.object(trainer,'reference_predictions',return_value=([],[])), \
             patch.object(trainer,'fit_candidate') as fitted, patch.object(trainer,'evaluate',side_effect=wrong_control), \
             contextlib.redirect_stdout(io.StringIO()):
            mismatch=rejected(lambda:trainer.research(Path(directory),config,ref,space))
        check('reference_mismatch_stops_before_training',mismatch and fitted.call_count==0)
    competitors=[row('PF',.59,.35),row('STRESS',.59,.35)]
    competitors[0]['pooled']['profit_factor']=.99
    competitors[1]['pooled']['stress_pf']=1.1
    check('pf_precedes_stress',rules.select(competitors)['selected_candidate']=='PF'
          and validator.decision(competitors)==rules.select(competitors))
    from gold_manual_training_workflow_v1 import issue_receipt, consume_receipt
    with patch('manual_training_launcher_v1.user_double_click',return_value=False):
        check('nonuser_cannot_issue_receipt',rejected(issue_receipt))
    check('forged_user_receipt_denied',rejected(lambda:consume_receipt('synthetic-invalid')))
    tree=ast.parse((rules.ROOT/'gold_s4_secondary_improvement_v3.py').read_text(encoding='utf-8'))
    source=ast.get_source_segment((rules.ROOT/'gold_s4_secondary_improvement_v3.py').read_text(encoding='utf-8'),
                                 next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run'))
    check('completed_research_saved_before_validation',source.index('status.update(research_result=') < source.index('result = run_authorized'))
    return checks


if __name__ == '__main__':
    print(json.dumps(run_tests(),sort_keys=True))
