"""Deterministic synthetic-only checks; no historical arrays or trades are loaded."""
import ast
import contextlib
import copy
import io
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ['SYNTHETIC_TEST_MODE']='true'
import gold_s4_entry_edge_decomposition_v1_support as s
import gold_s4_entry_edge_feature_inventory_v1 as features
import validate_gold_s4_entry_edge_decomposition_v1_run as validator
import gold_s4_entry_edge_decomposition_v1_launcher as launcher
from gold_s4_entry_edge_decomposition_v1 import decompose, FREEZE
from certify_gold_s4_entry_edge_decomposition_v1 import GROUPS,bat_text

SPEC=s.read(s.ROOT/'gold_s4_entry_edge_bucket_spec_v1.json')
INVENTORY=s.read(s.ROOT/'gold_s4_entry_edge_feature_inventory_v1.json')
FOLDS=[['2018_2020','2018-01-01','2021-01-01'],['2021_2022','2021-01-01','2023-01-01'],['2023_2024','2023-01-01','2025-01-01']]


def rejected(fn):
    try: fn()
    except (ValueError,PermissionError,KeyError): return
    raise AssertionError('Expected rejection')


def fixture():
    rows=[]
    for fold,start,end in FOLDS:
        for day in range(60):
            for hour in (0,4):
                stamp=(datetime.fromisoformat(start)+timedelta(days=day,hours=hour)).isoformat()
                win=day%3!=0 if hour==0 else day%3==0
                r=1.2 if win else -1.
                row=dict(trade_id='synthetic-'+str(len(rows)),entry_index=len(rows),entry_time_api=stamp,
                    entry_time_actual_utc=stamp,entry_price=100.,exit_price=100+r,exit_reason='take_profit' if win else 'stop_loss',
                    net_r=r,stress_r=r-.05,holding_minutes=5.,mfe_completed_bar_r=1.4 if win else .2,
                    mae_completed_bar_r=.2 if win else 1.,model_probability=.82 if hour==0 else .92,
                    VOLA_RATIO=.9 if hour==0 else 1.1,M5_TREND=1 if hour==0 else -1,H1_TREND=1,
                    BIAS_20=.01,ROC_5=.01 if hour==0 else -.01,M1_RSI=55.,BODY_PCT=.7,fold=fold)
                rows.append(row)
    return rows


def prepared(directory,rows=None):
    run=Path(directory); rows=rows or fixture()
    ref=dict(accepted_trade_count=len(rows),folds=FOLDS,metrics={},fold_metrics=[])
    raw=[{k:v for k,v in row.items() if k!='model_probability'} for row in rows]
    reference=s.metrics(raw,len(raw),s.reporting_days(FOLDS))
    keys=['trades','wins','losses','realized_win_rate','trades_per_day','profit_factor','mean_r','pnl_r','max_drawdown_r','stress_pf']
    ref['metrics']={k:reference[k] for k in keys}
    for fold,start,end in FOLDS:
        local=[r for r in raw if start<=r['entry_time_api']<end]
        m=s.metrics(local,len(rows),(datetime.fromisoformat(end)-datetime.fromisoformat(start)).days)
        ref['fold_metrics'].append(dict(fold=fold,**{k:m[k] for k in keys}))
    for name,value in [('reference_ledger.json',raw),('reference_binding.json',ref),('approved_config.json',{'folds':FOLDS}),
        ('entry_snapshots.json',rows),('bucket_spec.json',SPEC),('reference_reproduction.json',reference),('feature_inventory.json',features.availability(rows,INVENTORY))]:
        s.write(run/name,value)
    for event,names in [('reference_loaded',['reference_ledger.json','reference_binding.json']),
        ('reference_reproduction_pass',['reference_reproduction.json']),('feature_inventory_frozen',['feature_inventory.json','entry_snapshots.json']),
        ('bucket_spec_frozen',['bucket_spec.json'])]:
        s.event(run,event,s.artifact_hashes(run,names))
    with contextlib.redirect_stdout(io.StringIO()): decompose(run,rows,SPEC,{'folds':FOLDS})
    return run,rows,ref


def synthetic_source_join():
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        columns=['VOLA_RATIO','M5_TREND','H1_TREND','BIAS_20','ROC_5','M1_RSI','BODY_PCT']
        source_rows=[]; originals=[]; sources=[]; paths=[]
        offset=0
        for n,((fold,start,end),count) in enumerate(zip(FOLDS,[267,260,260]),1):
            times=[]
            for position in range(count):
                dt=datetime.fromisoformat(start)+timedelta(minutes=position)
                stamp=dt.isoformat()
                from datetime import timezone
                epoch=dt.replace(tzinfo=timezone.utc).timestamp()
                row=dict(trade_id='synthetic-join-'+str(offset+position),entry_index=offset+position,
                    entry_time_api=stamp,entry_time_actual_utc=stamp,entry_price=100.,
                    sl_distance=1.,tp_distance=1.,spread_points=1.,spread_observed=True,
                    raw_episode_id=offset+position,entry_epoch=epoch,exit_epoch=epoch+30,
                    exit_price=101.,exit_reason='take_profit',net_r=.9,stress_r=.8,
                    same_bar_both_hit=False,holding_minutes=.5,mfe_completed_bar_r=.5,mae_completed_bar_r=.1)
                source_rows.append(row)
                originals.append(dict(row,exit_time_actual_utc=(dt+timedelta(seconds=30)).isoformat()))
                times.append(int(epoch*1e9))
            np.savez(root/f'audit{n}.npz',score_ns=np.array(times,dtype=np.int64))
            np.savez(root/f'prob{n}.npz',probability=np.full(count,.82))
            values=np.tile([.9,1.,1.,.01,.01,55.,.7],(count,1))
            np.savez(root/f'chunk{n}.npz',values=values)
            def desc(name):
                paths.append(name)
                return dict(path=name,sha256=s.sha(root/name),seal='FINALIZED.json')
            sources.append(dict(fold_number=n,audit=desc(f'audit{n}.npz'),probability=desc(f'prob{n}.npz'),
                score_chunks=[dict(desc(f'chunk{n}.npz'),rows=count)]))
            offset+=count
        s.write(root/'original.json',originals); s.write(root/'control.json',source_rows)
        ref=dict(accepted_trade_count=787,folds=FOLDS,feature_columns=columns,fold_sources=sources,
            original_ledger=desc('original.json'),control_ledger=desc('control.json'))
        s.write(root/'FINALIZED.json',dict(file_sha256={name:s.sha(root/name) for name in paths}))
        for source in [ref['original_ledger'],ref['control_ledger'],*[item for f in sources for item in [f['audit'],f['probability'],*f['score_chunks']]]]:
            source['seal_sha256']=s.sha(root/'FINALIZED.json')
        def synthetic_file(item):
            path=(root/item['path']).resolve()
            assert path.is_relative_to(root.resolve()) and s.sha(path)==item['sha256']
            return path
        with patch.object(launcher,'require_session',lambda:None),patch.object(s,'check_binding_metadata',lambda r:None),patch.object(s,'sealed_file',synthetic_file),patch.object(features,'sealed_file',synthetic_file),patch.object(validator,'ROOT',root):
            old,control=s.load_reference(ref)
            joined=features.attach_snapshots(old,control,ref)
            independent_old,independent=validator.independent_rows(ref)
            assert independent_old==old and independent==joined and len(joined)==787
            broken=copy.deepcopy(old); broken[0]['entry_index']=1
            rejected(lambda:features.attach_snapshots(broken,control,ref))
            broken=copy.deepcopy(old); broken[0]['entry_time_actual_utc']='2018-01-01T00:00:01'
            rejected(lambda:features.attach_snapshots(broken,control,ref))


def reference_binding_review():
    config,ref,spec,inventory=s.configuration()
    s.check_binding_metadata(ref)
    assert ref['metrics']==dict(trades=787,wins=450,losses=337,realized_win_rate=.5717916137229987,
        trades_per_day=.30778255768478685,profit_factor=.8521940468784924,mean_r=-.06408993365369865,
        pnl_r=-50.438777785460836,max_drawdown_r=-52.34913835149044,stress_pf=.8093582289789337)
    assert len(ref['feature_columns'])==31 and len(ref['fold_sources'])==3
    assert all(s.safe_path(s.ROOT,x['path']).is_file() for x in s.sources(ref))
    return 'Seal/hash metadata and retained-file existence only; no trade/array read or reference reproduction'


def feature_inventory_review():
    synthetic_source_join()
    features.validate_spec(SPEC,INVENTORY)
    rows=fixture(); rows[0]['M1_RSI']=None
    audit=features.availability(rows,INVENTORY)
    assert next(f for f in audit['fields'] if f['field_name']=='M1_RSI')['missing_rate']==1/len(rows)
    assert all(f['missing_rate'] is None for f in INVENTORY['fields'])
    assert not next(f for f in INVENTORY['fields'] if f['field_name']=='spread_points')['allowed_for_decomposition']
    for key,value in [('BODY_PCT',1.1),('model_probability',-1),('M1_RSI',101),('VOLA_RATIO',-1),('H1_TREND',0)]:
        rejected(lambda: features.check_values(dict(rows[1],**{key:value})))
    return 'Synthetic 787-row NPZ joins independently match; index/time mutation rejected; missing-rate and domain checks; real missing rates remain null'


def bucket_spec_review():
    assert len(SPEC['dimensions'])==10 and len(SPEC['pairs'])==8
    for d in SPEC['dimensions']:
        if d['kind']=='bins':
            for i,edge in enumerate(d['edges']):
                assert s.bucket({d['fields'][0]:edge},d)==d['labels'][i+1]
                assert s.bucket({d['fields'][0]:edge-1e-8},d)==d['labels'][i]
        assert s.bucket({},d)=='MISSING'
    d=SPEC['dimensions'][0]
    for hour in range(24): assert s.bucket({'entry_time_actual_utc':f'2020-01-01T{hour:02}:00:00'},d)==d['labels'][hour//4]
    return 'All declared numeric boundaries, 24 UTC hours and explicit missing buckets'


def causality_test():
    for forbidden in ['future_r','future_mfe','future_mae','exit_type','exit_reason','holding_minutes','net_r','stress_r']:
        bad=copy.deepcopy(SPEC); bad['dimensions'][0]['fields']=[forbidden]
        rejected(lambda: features.validate_spec(bad,INVENTORY))
        rejected(lambda: validator.membership({forbidden:1},bad['dimensions'][0]))
    bad=copy.deepcopy(SPEC); bad['pairs'][0].append('model_score')
    rejected(lambda: features.validate_spec(bad,INVENTORY))
    return 'Eight future/outcome fields rejected by runner and independent validator; 3-way pair rejected'


def sample_guard_test():
    for n,expected in [(7,'DESCRIPTIVE_ONLY'),(29,'DESCRIPTIVE_ONLY'),(30,'LOW_CONFIDENCE'),(49,'LOW_CONFIDENCE'),(50,'USABLE')]:
        m=dict(trades=n,profit_factor=2.,mean_r=.2,pnl_r=n*.2,stress_pf=1.5)
        assert s.classify(m,1)['sample_confidence']==expected
        if n<30: assert s.classify(m,1)['edge_classification']=='NEUTRAL_BUCKET'
        assert s.classify(m,1,True)['sample_confidence']=='DESCRIPTIVE_ONLY'
    m=dict(trades=20,profit_factor=2.,mean_r=.2,pnl_r=4,stress_pf=1.5)
    assert s.classify(m,2)['sample_confidence']=='LOW_CONFIDENCE'
    assert s.classify(m,2)['edge_classification']=='NEUTRAL_BUCKET'
    m['trades']=19; assert s.classify(m,2)['sample_confidence']=='DESCRIPTIVE_ONLY'
    return '1D 7/29/30/49/50; 2D 19/20; missing buckets never inferential'


def univariate_logic_test():
    rows=fixture(); table,folds=s.group_tables(rows,SPEC,FOLDS)
    for d in SPEC['dimensions']:
        groups=[r for r in table if r['dimensions']==[d['name']]]
        assert sum(r['trades'] for r in groups)==len(rows)
        assert sorted(i for r in groups for i in r['trade_ids'])==sorted(r['trade_id'] for r in rows)
    a=next(r for r in table if r['group_id']=='time_of_day:00-04')
    b=next(r for r in table if r['group_id']=='time_of_day:04-08')
    assert abs(a['profit_factor']-2.4)<1e-12 and abs(b['profit_factor']-.6)<1e-12
    assert a['edge_classification']=='ROBUST_POSITIVE_EDGE_BUCKET' and b['edge_classification']=='NEGATIVE_EDGE_BUCKET'
    assert a['research_candidate']=='ENTRY_FILTER_RESEARCH_CANDIDATE' and b['research_candidate']=='ENTRY_EXCLUSION_RESEARCH_CANDIDATE'
    return 'Synthetic positive/negative time buckets PF 2.4/0.6; exact full partition for all 10 dimensions'


def bivariate_logic_test():
    rows=fixture(); table,folds=s.group_tables(rows,SPEC,FOLDS,True)
    for pair in SPEC['pairs']:
        subset=[r for r in table if r['dimensions']==pair]
        assert sum(r['trades'] for r in subset)==len(rows)
        assert sorted(t for r in subset for t in r['trade_ids'])==sorted(r['trade_id'] for r in rows)
    return 'Eight complete Cartesian partitions, including empty and missing cells; no filtered ledger'


def fold_stability_test():
    rule=SPEC['stability']
    make=lambda pf,n=10:dict(trades=n,profit_factor=pf,mean_r=.1 if pf>1 else -.1)
    assert s.persistence([make(1.2),make(1.1),make(.8)],rule)['stability']=='STABLE'
    assert s.persistence([make(1.2,7),make(1.1),make(.8)],rule)['stability']=='UNSTABLE'
    assert s.persistence([make(1.2),make(1.1),make(.69)],rule)['stability']=='UNSTABLE'
    assert s.persistence([make(.9),make(.9),make(.9)],rule)['stability']=='MIXED'
    assert s.persistence([make(.6),make(.7),make(.8)],rule)['persistent_weakness']
    assert not s.persistence([make(.6),make(.7),make(1.1)],rule)['persistent_weakness']
    assert s.persistence([make(1.2),make(1.1)],rule)['stability']=='UNSTABLE'
    return 'All folds >=8, at least 2 positive folds, PF floor 0.70; persistent weakness independently constrained'


def stress_metric_test():
    rows=fixture()[:12]
    s.metrics(rows,len(rows),1)
    validator.equal(s.metrics(rows,len(rows),1),validator.accounting(rows,len(rows),1))
    for subset in [[],[dict(rows[0],net_r=0.,stress_r=0.)],[dict(rows[0],net_r=1.,stress_r=1.)],[dict(rows[0],net_r=-1.,stress_r=-1.)]]:
        validator.equal(s.metrics(subset,len(subset),1),validator.accounting(subset,len(subset),1))
    bad=copy.deepcopy(rows); bad[0]['stress_r']=float('nan')
    rejected(lambda:s.metrics(bad,len(bad),1))
    assert s.metrics([dict(rows[0],net_r=1.,stress_r=1.)],1,1)['profit_factor'] is None
    return 'Independent R/stress/drawdown math; zero trades, flats, all wins/losses and nonfinite rejection'


def multiple_testing_test():
    rows=fixture(); single,_=s.group_tables(rows,SPEC,FOLDS); paired,_=s.group_tables(rows,SPEC,FOLDS,True)
    a=s.summaries(single,paired,SPEC); b=validator.derived(single,paired,SPEC)
    validator.equal(list(a),list(b))
    assert a[2]['number_of_univariate_buckets']==sum(len(d['labels']) for d in SPEC['dimensions'])
    sizes={d['name']:len(d['labels']) for d in SPEC['dimensions']}
    assert a[2]['number_of_bivariate_cells']==sum(sizes[x]*sizes[y] for x,y in SPEC['pairs'])
    return 'All predefined cells counted even empty; inferential counts and lexicographic rankings independently checked'


def validator_fixture_test():
    mutations=0
    with tempfile.TemporaryDirectory() as folder:
        run,rows,ref=prepared(folder)
        assert validator.verify_outputs(run,rows,SPEC,INVENTORY,ref)
        for name in validator.TABLES+validator.JSONS+['feature_inventory.json','bucket_spec.json','reference_reproduction.json']:
            path=run/name; old=path.read_bytes()
            if name.endswith('.csv'):
                text=path.read_text(encoding='utf-8'); path.write_text(text.replace('synthetic-0','mutated-id',1),encoding='utf-8')
                if path.read_bytes()==old:
                    path.write_text(text.replace('0.0','9.9',1),encoding='utf-8')
            else: s.write(path,{})
            rejected(lambda:validator.verify_outputs(run,rows,SPEC,INVENTORY,ref))
            path.write_bytes(old); mutations+=1
        for field,value in [('net_r',99.),('stress_r',99.),('entry_time_api','2030-01-01T00:00:00')]:
            changed=copy.deepcopy(rows); changed[0][field]=value
            rejected(lambda:validator.verify_outputs(run,changed,SPEC,INVENTORY,ref)); mutations+=1
    return str(mutations)+' corrupted table, count, identity, R, stress, fold, boundary and audit fixtures rejected'


def event_chain_test():
    with tempfile.TemporaryDirectory() as folder:
        run,rows,ref=prepared(folder)
        assert validator.check_chain(run)
        chain=run/'research_events.jsonl'; old=chain.read_bytes()
        data=[json.loads(line) for line in chain.read_text().splitlines()]
        data[2]['records']={'wrapper':data[2]['records']}
        chain.write_text('\n'.join(json.dumps(r) for r in data)+'\n')
        rejected(lambda:validator.check_chain(run)); chain.write_bytes(old)
        rejected(lambda:s.event(run,'univariate_complete',{}))
        s.write(run/'validator.json',{'overall':'PASS'}); s.event(run,'validation_result',s.artifact_hashes(run,['validator.json']))
        s.write(run/'combined_result.json',{'execution_status':'PASS'}); s.event(run,'finalization',s.artifact_hashes(run,['combined_result.json']))
        assert validator.check_chain(run,True)
        rejected(lambda:s.event(run,'finalization',{}))
    return 'Canonical raw event payload, artifact hashes, ordered ten-event lifecycle, wrappers and append-after-finalization rejected'


def manual_execution_guard_test():
    launcher.check_policy()
    rejected(launcher.require_session)
    rejected(lambda:launcher.begin_session('fake'))
    rejected(lambda:launcher.consume_validation('fake',Path('.')))
    rejected(lambda:s.sealed_file({'path':'must-not-open'}))
    rejected(lambda:features.attach_snapshots([],[],{}))
    rejected(lambda:validator.run_authorized(Path('.'),'fake'))
    from gold_s4_entry_edge_decomposition_v1 import research
    rejected(lambda:research(None,None,None,None,None))
    with patch.object(launcher,'require_session',lambda:None):
        token=launcher.validation_permit(Path('.'))
        launcher.consume_validation(token,Path('.'))
        rejected(lambda:launcher.consume_validation(token,Path('.')))
        token=launcher.validation_permit(Path('.'))
        rejected(lambda:launcher.consume_validation(token,Path('wrong')))
    return 'All real I/O/research/validation entry points blocked without USER session; single-use/wrong-run permit checks'


def holdout_guard_test():
    from training_holdout_guard_v1 import check_path,install
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); run=root/'run'; run.mkdir()
        denied=root/'future_holdout'/'gold_s4_v4'/'not-opened.json'
        rejected(lambda:check_path(denied,root))
        release=install(root,write_root=run)
        try:
            rejected(lambda:denied.open('r'))
            rejected(lambda:(root/'production.py').write_text('forbidden'))
            (run/'allowed.txt').write_text('synthetic')
        finally: release()
    rejected(lambda:s.safe_path(s.ROOT,'../outside.json'))
    return 'Synthetic locked-path read and out-of-run write denied; no real holdout access'


def production_protection_test():
    config=s.read(s.ROOT/s.CONFIG); s.check_production(config)
    bad=copy.deepcopy(config); bad['protected_sha256']['gemini.py']='0'*64
    rejected(lambda:s.check_production(bad))
    for name in [s.EXPERIMENT+'.py',s.EXPERIMENT+'_support.py','gold_s4_entry_edge_feature_inventory_v1.py','validate_'+s.EXPERIMENT+'_run.py']:
        tree=ast.parse((s.ROOT/name).read_text(encoding='utf-8-sig'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert not set(calls)&{'fit','predict','predict_proba','simulate','replay_entries','replay_trade'}
    return 'Exact production hashes; no model fitting, inference or backtest calls in new data path'


def bat_static_test():
    run,status=bat_text(),bat_text(True)
    assert 'GOLD_S4_ENTRY_EDGE_DECOMPOSITION_V1' in run and 'XM_USER_TRAINING_BAT=RUN_TRAINING_V1' in run
    assert 'gold_s4_entry_edge_decomposition_v1_launcher.py"\n' in run and '--status' not in run
    assert '--status' in status and 'XM_USER_TRAINING_BAT=' not in status
    assert 'pause >nul' in run and '.venv\\Scripts\\python.exe' in run
    from manual_training_launcher_v1 import manual_parent_chain
    explorer=[dict(Name='cmd.exe',SessionId=1,ParentProcessId=2),dict(Name='explorer.exe',ProcessId=2)]
    assert manual_parent_chain(explorer,'python')
    assert not manual_parent_chain([dict(Name='powershell.exe'),dict(Name='explorer.exe')],'python')
    return 'Prospective exact BAT text and Explorer-vs-agent ancestry; BAT never executed'


def reference_reproduction_spec_test():
    rows=fixture()
    with tempfile.TemporaryDirectory() as folder:
        run,rows,ref=prepared(folder,rows)
        s.reproduce(rows,ref)
        changed=copy.deepcopy(rows); changed[0]['net_r']+=.01
        rejected(lambda:s.reproduce(changed,ref))
        rejected(lambda:s.reproduce(rows[:-1],ref))
        changed=copy.deepcopy(rows); changed[0]['stress_r']+=.01
        rejected(lambda:s.reproduce(changed,ref))
    return 'Synthetic exact reproduction and R/count/stress mismatches stop; real 787 reproduction not executed'


def main():
    checks={}; details={}
    for name in GROUPS:
        try:
            details[name]=globals()[name](); checks[name]=True
        except Exception as error:
            import traceback
            traceback.print_exc()
            checks[name]=False; details[name]=repr(error)
    result=dict(overall='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details,**s.FLAGS)
    print(json.dumps(result,allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__=='__main__': raise SystemExit(main())
