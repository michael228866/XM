"""Synthetic contracts only: no real dataset, fitting, inference or replay."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import gold_s4_secondary_improvement_v4_label_objective_support as rules
import gold_s4_secondary_improvement_v4_label_objective_launcher as launcher
import validate_gold_s4_secondary_improvement_v4_label_objective_run as validator
import gold_s4_v4_label_definitions as labels


def rejected(call):
    try:
        call()
    except (ValueError,PermissionError,KeyError,FileNotFoundError):
        return True
    return False


def synthetic_guard():
    allowed=set(rules.read(rules.ROOT/'gold_s4_a_no_long_htf_reference_v2.json')['bindings'])
    def hook(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or isinstance(args[0],int):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        for folder in ('training_runs','historical_training_data','future_holdout'):
            if path.is_relative_to(rules.ROOT/folder):
                if event=='open' and path.relative_to(rules.ROOT).as_posix() in allowed and not any(c in (args[1] or '') for c in 'wax+'):
                    return
                raise PermissionError('Synthetic certification denies real historical/holdout inputs')
        if path.parent==rules.ROOT and path.suffix.lower() in ('.npz','.csv','.parquet'):
            raise PermissionError('Historical exports prohibited')
    sys.addaudithook(hook)


def row(c,wr=None,tpd=None):
    p=copy.deepcopy(rules.REFERENCE)
    if wr is not None:p['realized_win_rate']=wr
    if tpd is not None:p['trades_per_day']=tpd
    folds=[dict(trades=100,realized_win_rate=.55,profit_factor=.9) for _ in range(3)]
    return dict(candidate_id=c['candidate_id'],config=c,pooled=p,fold_metrics=folds,
                gate=rules.gate(p,folds),economic_status=rules.economic(p),economic_milestones=rules.milestones(p))


def run_tests():
    os.environ['SYNTHETIC_TEST_MODE']='true'
    synthetic_guard()
    checks={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks[name]=True
    config,ref,space=rules.configuration()
    check('reference_binding_review',ref['candidate_id']=='A_NO_LONG_HTF' and ref['metrics']['trades']==787)
    from gold_s4_v4_label_pipeline_audit import verify_audit,label_source,add_label_context
    check('label_pipeline_audit_review',verify_audit()['overall']=='PASS')
    check('search_space_review',len(space['candidates'])==8 and len(space['label_definitions'])==5)
    check('frozen_features',all(c['features']==ref['features'] and c['parameters']==ref['parameters'] for c in space['candidates']))
    r=np.array([-1.,-.5,-.4,-.25,-.1,0.,.2,.25,.4,.5,.8,1.5])
    expected={
        'L0_CURRENT':[0,0,0,0,0,0,1,1,1,1,1,1],
        'L1_R_GE_025':[0,0,0,0,0,0,0,1,1,1,1,1],
        'L1_R_GE_050':[0,0,0,0,0,0,0,0,0,1,1,1],
        'L2_NONLOSS_GT_M025':[0,0,0,0,1,1,1,1,1,1,1,1],
        'L2_NONLOSS_GT_M050':[0,0,1,1,1,1,1,1,1,1,1,1]}
    for lid,wanted in expected.items():
        check('label_'+lid,np.array_equal(labels.make_label(lid,r),wanted) and np.array_equal(validator.independent_label(lid,r),wanted))
    check('label_definition_review',True)
    for invalid in ([np.nan],[np.inf],[None]):
        check('missing_label_'+str(invalid),rejected(lambda:labels.make_label('L0_CURRENT',invalid)))
    check('unsupported_label',rejected(lambda:labels.make_label('L3_STRONG_WIN',r)))
    y=np.tile([0,1],200)
    check('class_balance_test',labels.class_summary(y)==validator.balance(y))
    for bad in (np.zeros(300),np.ones(300),np.array([0,1]),np.r_[np.zeros(20000),np.ones(100)]):
        check('degenerate_'+str(len(checks)),rejected(lambda:labels.class_summary(bad)) and rejected(lambda:validator.balance(bad)))
    check('label_degeneracy_test',True)
    times=pd.date_range('2016-07-01','2017-12-31',freq='h').to_numpy(dtype='datetime64[ns]').astype(np.int64)
    data=dict(train_ns=times,feature_cutoff_time=times.copy(),maturity=times+60_000_000_000,
              legacy_maturity=times+240*60_000_000_000,b0_train=np.zeros(len(times)))
    for calibration in ('RAW','ISOTONIC'):
        c=next(c for c in space['candidates'] if c['calibration']==calibration and not c['reuse_reference_model'])
        fit,cal,dev=rules.split(data,c,config,1)
        independent=validator.partition_indices(data,c,config,1)
        check('partitions_'+calibration,all(np.array_equal(a,b) for a,b in zip((fit,cal,dev),independent)))
        check('disjoint_'+calibration,not np.intersect1d(fit,dev).size and not np.intersect1d(cal,dev).size)
        check('purged_fit_'+calibration,data['legacy_maturity'][fit].max()<times[cal[0] if len(cal) else dev[0]])
        if len(cal):check('purged_calibration',data['legacy_maturity'][cal].max()<times[dev[0]])
        for key in ('feature_cutoff_time','maturity','legacy_maturity'):
            bad=copy.deepcopy(data);bad[key][-1]=pd.Timestamp('2019-01-01').value
            check('leak_'+calibration+'_'+key,rejected(lambda:rules.split(bad,c,config,1)) and rejected(lambda:validator.partition_indices(bad,c,config,1)))
    check('temporal_leakage_review',True)
    check('calibration_isolation_test',True)
    p=np.tile([.2,.8],200)
    grid=[.55,.60,.65,.70,.75,.80]
    check('threshold_isolation_test',labels.choose_threshold(p,y,grid)==validator.threshold_choice(p,y,grid))
    check('threshold_fixed_tie',labels.choose_threshold(p,y,grid)[0]==.80)
    check('bad_development_probabilities',rejected(lambda:labels.choose_threshold(p*2,y,grid)))
    x=np.ones((4,len(config['input_features'])),dtype=np.float32)
    check('feature_identity',np.array_equal(rules.features(x,config['input_features'],ref['features']),validator.feature_matrix(x,config['input_features'],ref['features'])))
    for leak in ('future_r','future_stop_target','validation_outcome','post_decision_price'):
        wanted=ref['features'][:-1]+[leak]
        check('feature_leak_'+leak,rejected(lambda:rules.features(x,config['input_features'],wanted)) and rejected(lambda:validator.feature_matrix(x,config['input_features'],wanted)))
    rows=[row(c,None if i==0 else .58,None if i==0 else .32) for i,c in enumerate(space['candidates'])]
    decision=rules.select(rows)
    validator.validate_records(space,rows,decision)
    check('validator_fixture_test',decision==validator.decision(rows))
    check('pareto_logic_test',True)
    check('selection_logic_test',decision['selected_candidate'] is not None)
    a,b,c=[row({'candidate_id':n},wr,tpd) for n,wr,tpd in [('A',.575,.315),('B',.58,.300),('C',.570,.34)]]
    b['pooled'].update(profit_factor=1.1,mean_r=.1,pnl_r=10)
    check('dual_primary_required',rules.select([a,b,c])['selected_candidate']=='A')
    check('no_improvement',rules.select([rows[0],b,c])['selected_candidate'] is None)
    for wr,tpd,gate in [(.58,.40,'STRONG'),(.60,.50,'TARGET')]:
        candidate=row({'candidate_id':'gate'},wr,tpd)
        check('gate_'+gate,candidate['gate']['gate']==gate)
    bad=copy.deepcopy(rows);bad[1]['fold_metrics'][0]['profit_factor']=.69
    check('collapsed_fold',not rules.gate(bad[1]['pooled'],bad[1]['fold_metrics'])['safety_pass'])
    for key in ('label_id','label_formula','features','threshold_grid','class_weight_policy'):
        bad=copy.deepcopy(rows);bad[1]['config'][key]='changed'
        check('frozen_mutation_'+key,rejected(lambda:validator.validate_records(space,bad,decision)))
    check('incomplete_candidates',rejected(lambda:validator.validate_records(space,rows[:-1],decision)))
    with tempfile.TemporaryDirectory() as folder:
        run=Path(folder);sh=rules.digest(space)
        rules.event(run,'space_frozen',sh)
        rules.event(run,'reference_pass',sh,rows[0]['candidate_id'],record=rows[0])
        rules.event(run,'label_definition_freeze',sh,record=space['label_definitions'])
        rules.event(run,'candidate_end',sh,rows[0]['candidate_id'],record=rows[0])
        for item in rows[1:]:
            rules.event(run,'candidate_start',sh,item['candidate_id'])
            rules.event(run,'candidate_end',sh,item['candidate_id'],record=item)
        for name,payload in [('pareto_complete',decision['pareto_frontier']),('selection_complete',decision),('research_freeze',decision)]:
            rules.event(run,name,sh,record=payload)
        events=[json.loads(line) for line in (run/'research_events.jsonl').read_text().splitlines()]
        check('event_chain_test',validator.check_events(events,rows,decision,space))
        check('raw_reference_hash',events[1]['record_sha256']==rules.digest(rows[0]))
        for field,value in [('sequence',99),('record_sha256','0'*64),('previous_sha256','0'*64),('at_utc','2000-01-01T00:00:00+00:00')]:
            altered=copy.deepcopy(events);altered[1][field]=value
            altered[1]['event_sha256']=validator.digest({k:v for k,v in altered[1].items() if k!='event_sha256'})
            check('event_mutation_'+field,rejected(lambda:validator.check_events(altered,rows,decision,space)))
        check('postfreeze_rejected',rejected(lambda:rules.event(run,'candidate_start',sh,'extra')))
        for name in ('validation_result','finalization'):
            (run/(name+'.json')).write_text(json.dumps(dict(event=name,research_tip=events[-1]['event_sha256'])))
            check('separate_'+name,validator.check_events(events,rows,decision,space))
            check('append_'+name,rejected(lambda:validator.check_events(events+[dict(event=name)],rows,decision,space)))
    check('manual_execution_guard_test',rejected(launcher.require_session) and rejected(lambda:launcher.begin_session('fake')))
    check('no_real_label_source',rejected(label_source))
    check('no_real_label_regeneration',rejected(lambda:add_label_context(None,None,None,None)))
    check('no_real_evidence',rejected(lambda:rules.load_evidence(config,ref)))
    check('no_real_validation',rejected(lambda:validator.run_authorized(Path('fake'),'fake')))
    import gold_s4_secondary_improvement_v4_label_objective as trainer
    check('no_real_fit',rejected(lambda:trainer.fit_candidate(None,None,None,None,None)))
    with patch.object(launcher,'require_session'):
        permit=launcher.validation_permit(Path('fake'))
        launcher.consume_validation(permit,Path('fake'))
        check('permit_reuse_denied',rejected(lambda:launcher.consume_validation(permit,Path('fake'))))
    from manual_training_launcher_v1 import manual_parent_chain
    check('scheduler_rejected',not manual_parent_chain([dict(Name='cmd.exe',SessionId=1,ParentProcessId=2),dict(Name='taskeng.exe',ProcessId=2)],'python'))
    from gold_manual_training_workflow_v1 import issue_receipt
    with patch('manual_training_launcher_v1.user_double_click',return_value=False):
        check('codex_receipt_denied',rejected(issue_receipt))
    from training_holdout_guard_v1 import check_path
    check('holdout_guard_test',rejected(lambda:check_path(rules.ROOT/'future_holdout/gold_s4_v4/fake',rules.ROOT)))
    check('historical_input_denial',rejected(lambda:(rules.ROOT/ref['evidence_source_run']/'secondary_evidence.npz').read_bytes()))
    rules.check_production(config)
    check('production_protection_test',True)
    for name in ('RUN_TRAINING.bat','CHECK_STATUS.bat'):
        text=(rules.ROOT/name).read_text(encoding='utf-8')
        check('bat_'+name,'gold_s4_secondary_improvement_v4_label_objective_launcher.py' in text
            and ('--status' in text)==(name=='CHECK_STATUS.bat'))
    check('bat_static_test',True)
    checks.update(fitted_evidence_fixture(config,ref,space,data))
    import contextlib
    import io
    import gold_s4_v4_label_pipeline_audit as audit
    fake_state={n:dict(target=np.array([0,1]),train_ns=np.array([1,2]),score_ns=np.array([3,4])) for n in (1,2,3)}
    def fake_evaluate(run,cfg,c,state,predictions,models):
        return copy.deepcopy(next(r for r in rows if r['candidate_id']==c['candidate_id']))
    with tempfile.TemporaryDirectory() as directory:
        run=Path(directory);(run/'predeclared_search_space.json').write_text(json.dumps(space))
        with patch.object(launcher,'require_session'), patch.object(trainer,'load_evidence',return_value=fake_state), \
             patch.object(trainer,'reference_predictions',return_value=([],[])), patch.object(audit,'label_source',return_value=None), \
             patch.object(audit,'add_label_context',return_value=fake_state), patch.object(trainer,'evaluate',side_effect=fake_evaluate), \
             patch.object(trainer,'fit_candidate',return_value=(np.array([]),{})) as fitted, contextlib.redirect_stdout(io.StringIO()):
            selected,_,records=trainer.research(run,config,ref,space)
        events=[json.loads(line) for line in (run/'research_events.jsonl').read_text().splitlines()]
        check('actual_coordinator_event_fixture',validator.check_events(events,records,selected,space))
        check('control_not_retrained',fitted.call_count==21 and all(not call.args[2]['reuse_reference_model'] for call in fitted.call_args_list))
    with tempfile.TemporaryDirectory() as directory:
        def wrong_control(*args):
            bad=copy.deepcopy(rows[0]);bad['pooled']['trades']+=1;return bad
        with patch.object(launcher,'require_session'),patch.object(trainer,'load_evidence',return_value=fake_state), \
             patch.object(trainer,'reference_predictions',return_value=([],[])),patch.object(trainer,'evaluate',side_effect=wrong_control), \
             patch.object(audit,'add_label_context') as regenerate,patch.object(trainer,'fit_candidate') as fitted, \
             contextlib.redirect_stdout(io.StringIO()):
            stopped=rejected(lambda:trainer.research(Path(directory),config,ref,space))
        check('control_failure_stops_label_generation',stopped and not regenerate.called and not fitted.called)
    with tempfile.TemporaryDirectory() as directory:
        import training_run_history as history
        run=history.create_run('synthetic_v4_archive',Path(__file__),'Synthetic metadata only',
            runs_root=Path(directory)/'runs',root=rules.ROOT,seed_note='No fitting')
        (run/'models').mkdir();model=run/'models/fake.json';model.write_text('{}')
        fixture=copy.deepcopy(rows)
        for record in fixture:record['models']=[dict(path='models/fake.json',sha256=rules.sha(model))]
        (run/'data_inventory.json').write_text(json.dumps([dict(train_start='2000-01-01',train_end='2000-12-01',score_start='2001-01-01',score_end='2001-12-01',train_rows=5,score_rows=5)]))
        manifest=history.read_json(run/'manifest.json')
        trainer.fill_manifest(manifest,run,ref,space,fixture,None,history)
        history.write_json(run/'manifest.json',manifest);history.write_json(run/'metrics.json',dict(synthetic_only=True))
        (run/'report.md').write_text('Synthetic archive schema only')
        check('archive_schema_fixture',not history.finalize_run(run,'research_only') and not history.validate_run(run))
    return dict(overall='PASS',checks=checks,MODEL_TRAINING_EXECUTED=False,REAL_VALIDATION_EXECUTED=False,
                REAL_RESEARCH_EXECUTED=False,HISTORICAL_DATA_USED=False)


def fitted_evidence_fixture(config,ref,space,data):
    """Exercise validator's model-evidence branch with a pure fake predictor."""
    class FakePredictor:
        def predict_proba(self,frame):
            p=frame.iloc[:,0].to_numpy(dtype=np.float32)
            return np.column_stack([1-p,p])
    data=copy.deepcopy(data)
    size=len(data['train_ns'])
    data['realized_r']=np.resize(np.array([-1.,1.]),size)
    data['train_x']=np.ones((size,len(config['input_features'])),dtype=np.float32)
    data['train_x'][:,0]=np.resize(np.array([.2,.8],dtype=np.float32),size)
    classifier=FakePredictor();checks={}
    for mode in ('RAW','ISOTONIC'):
        c=next(c for c in space['candidates'] if not c['reuse_reference_model'] and c['calibration']==mode)
        y=labels.make_label(c['label_id'],data['realized_r'])
        fit,cal,dev=rules.split(data,c,config,1)
        x=rules.features(data['train_x'],config['input_features'],c['features'])
        raw=np.array([.2,.8],dtype=np.float32);devraw=x[dev,0];calraw=x[cal,0]
        probability=raw;devprob=devraw;info=None
        if len(cal):
            xs=np.array([.2,.8],dtype=np.float32).astype(float)
            info=dict(x_thresholds=xs.tolist(),y_thresholds=[0.,1.])
            probability=np.interp(raw,xs,[0.,1.]);devprob=np.interp(devraw,xs,[0.,1.])
        threshold,review=labels.choose_threshold(devprob,y[dev],c['threshold_grid'])
        membership=np.zeros(size,dtype=np.int8);membership[fit]=1;membership[cal]=2;membership[dev]=3
        saved=dict(partition_membership=membership,raw=raw,probability=probability,fit_positions=fit,
            calibration_positions=cal,development_positions=dev,sample_weight=np.ones(len(fit)),calibration_raw=calraw,
            development_raw=devraw,development_probability=devprob,target=y,feature_cutoff_time=data['feature_cutoff_time'],
            decision_time=data['train_ns'],label_outcome_time=data['maturity'])
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);path=run/'fixture.npz';np.savez_compressed(path,**saved)
            m=dict(evidence_path=path.name,evidence_sha256=rules.sha(path),label_id=c['label_id'],
                realized_r_sha256=rules.array_hash(data['realized_r']),class_balance=labels.class_summary(y[fit]),
                development_class_balance=labels.class_summary(y[dev]),calibration_class_balance=labels.class_summary(y[cal]) if len(cal) else None,
                fit_feature_sha256=rules.array_hash(x[fit]),fit_target_sha256=rules.array_hash(y[fit]),calibration=info,
                selected_threshold=threshold,threshold_development=review,
                model_configuration={'learner':{'gradient_booster':{'tree_train_param':c['parameters']},'generic_param':{'seed':42},'objective':{'name':'binary:logistic'}}})
            result=validator.verify_fitted(run,m,c,data,config,1,classifier,raw)
            assert np.array_equal(result,probability)
            checks['full_fitted_evidence_'+mode]=True
            altered={**m,'selected_threshold':.123}
            assert rejected(lambda:validator.verify_fitted(run,altered,c,data,config,1,classifier,raw))
            checks['threshold_evidence_mutation_'+mode]=True
            saved['label_outcome_time']=saved['label_outcome_time']+1
            np.savez_compressed(path,**saved);m['evidence_sha256']=rules.sha(path)
            assert rejected(lambda:validator.verify_fitted(run,m,c,data,config,1,classifier,raw))
            checks['outcome_evidence_mutation_'+mode]=True
    return checks


if __name__=='__main__':
    print(json.dumps(run_tests(),sort_keys=True))
