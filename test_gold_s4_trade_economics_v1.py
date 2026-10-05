"""Synthetic-only tests; no historical inputs, model inference or real replay."""
import copy
import json
import os
import random
import tempfile
from pathlib import Path

os.environ['SYNTHETIC_TEST_MODE'] = 'true'

from gold_s4_trade_economics_path_audit import replay_trade
from gold_s4_trade_economics_v1_support import metrics, gates, select, event, digest
from validate_gold_s4_trade_economics_v1_run import independent_trade, independent_metrics, equal, verify_chain


def tests():
    checks = {}
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True
    def reject(name, function):
        try:
            function()
        except (ValueError, PermissionError, KeyError):
            checks[name] = True
        else:
            raise AssertionError(name)
    entry = dict(trade_id='SYNTHETIC_0', entry_epoch=0, entry_price=100., sl_distance=1.,
                 tp_distance=2., spread_points=0., entry_time_api='1970-01-01T00:00:00')
    base = dict(candidate_id='E0_REFERENCE_EXECUTION', stop_multiple=1., target_r=None,
                time_minutes=90, break_even_r=None, trail_after_r=None, trail_by_r=None,
                no_progress_minutes=None)
    def compare(name, bars, changes=None):
        rule = dict(base, **(changes or {}))
        actual = replay_trade(entry, bars, rule)
        equal(actual, independent_trade(entry, bars, rule))
        checks[name+'_independent'] = True
        return actual
    row = compare('same_bar', [(0,100,103,98,100)])
    check('same_bar_stop_first', row['exit_price'] == 99 and row['same_bar_both_hit'])
    row = compare('gap', [(0,100,100.2,99.8,100), (60,97,97.5,96,97)])
    check('gap_exact_inherited_barrier_fill', row['exit_price'] == 99)
    row = compare('time_stop', [(0,100,100.1,99.8,100),(5400,100.2,103,98,100)])
    check('time_open_precedes_intrabar', row['exit_reason'] == 'timeout' and row['exit_price'] == 100.2)
    row = compare('time_gap', [(0,100,100.1,99.8,100),(6000,100.3,100.4,100.1,100.2)])
    check('time_first_available_open', row['holding_minutes'] == 100)
    row = compare('be', [(0,100,100.6,99.5,100.4),(60,100.4,100.5,99.8,100.1)], {'break_even_r':.5})
    check('be_next_bar_only', row['exit_reason'] == 'break_even' and row['bars_to_exit'] == 1)
    check('be_is_net_loss_after_cost', row['net_r'] < 0)
    row = compare('be_same_bar_old_stop', [(0,100,100.6,98.8,100.2)], {'break_even_r':.5})
    check('be_old_stop_first', row['exit_reason'] == 'stop_loss')
    row = compare('be_untriggered', [(0,100,100.4,99.5,100.2),(60,100.2,100.3,99.8,100)], {'break_even_r':.5})
    check('be_threshold_required', row['exit_reason'] == 'cohort_end')
    row = compare('trail', [(0,100,101,99.8,100.8),(60,100.8,101.4,100.6,101.2),(120,101.2,101.3,100.8,101)],
                  {'trail_after_r':1., 'trail_by_r':.5})
    check('trail_ratchet', row['exit_reason'] == 'trailing' and abs(row['exit_price']-100.9) < 1e-12)
    row = compare('no_progress', [(0,100,100.1,99.8,100),(1800,100.1,103,98,101)], {'no_progress_minutes':30})
    check('no_progress_ignores_current_high', row['exit_reason'] == 'no_progress' and row['exit_price'] == 100.1)
    row = compare('causal_extremes', [(0,100,104,98,100)])
    check('no_post_exit_extreme_claim', row['mfe_completed_bar_r'] == row['mae_completed_bar_r'] == 0)
    row = compare('fixed_r', [(0,100,100.1,98,99)], {'stop_multiple':1.5})
    check('risk_unit_never_rescaled', abs(row['net_r']+1.55) < 1e-12)
    reject('entry_price_mutation', lambda: replay_trade(entry, [(0,101,102,100,101)], base))
    reject('entry_time_mutation', lambda: replay_trade(entry, [(60,100,101,99.5,100)], base))
    reject('invalid_ohlc', lambda: replay_trade(entry, [(0,100,99,98,100)], base))
    reject('empty_path', lambda: replay_trade(entry, [], base))
    reject('nan', lambda: replay_trade(entry, [(0,100,float('nan'),99,100)], base))
    accounting = [dict(net_r=r, stress_r=r, gross_r=r, holding_minutes=i, exit_reason='timeout')
                  for i, r in enumerate((1., .5, -1., -.5))]
    m = metrics(accounting, 10)
    equal(m, independent_metrics(accounting, 10))
    check('accounting_example', m['gross_win_r'] == m['gross_loss_r'] == 1.5 and m['profit_factor'] == 1
          and m['mean_r'] == m['pnl_r'] == 0)
    accounting.append(dict(net_r=0., stress_r=0., gross_r=0., holding_minutes=1, exit_reason='break_even'))
    m = metrics(accounting, 10)
    check('flats_separate', m['trades'] == m['wins']+m['losses']+m['flat_or_breakeven'] == 5)
    equal(m, independent_metrics(accounting, 10))
    check('accounting_identity', abs(m['pnl_r']-m['mean_r']*m['trades']) < 1e-12)
    p = dict(profit_factor=1.2, mean_r=.1, pnl_r=10., stress_pf=1.05, realized_win_rate=.51,
             trades_per_day=.31, max_drawdown_r=-3.)
    folds = [dict(trades=10, profit_factor=.85, mean_r=-.15)]*3
    check('target_gate', gates(p,folds)['economic_gate'] == 'TARGET')
    check('positive_not_wr_improvement', gates(dict(p,profit_factor=1.01,mean_r=.001),folds)['economic_gate'] == 'POSITIVE')
    for key, value in [('profit_factor',1.),('mean_r',0.),('pnl_r',0.),('stress_pf',.89),('realized_win_rate',.49),('trades_per_day',.27)]:
        check('gate_'+key, gates(dict(p,**{key:value}),folds)['economic_gate'] == 'NONE')
    check('fold_catastrophe', not gates(p,[dict(folds[0],mean_r=-.151)]+folds[1:])['robustness_pass'])
    rows = [dict(candidate_id=name,execution_compatibility='PASS',pooled=dict(p,profit_factor=pf,mean_r=mean), gate=gates(dict(p,profit_factor=pf,mean_r=mean),folds))
            for name,pf,mean in [('A',1.2,.1),('B',1.1,.1),('C',1.15,.15)]]
    decision = select(rows)
    check('pareto', decision['pareto_frontier'] == ['A','C'] and decision['selected_candidate'] == 'A')
    check('no_positive_no_selection', select([])['research_result'] == 'NO_POSITIVE_EXPECTANCY_FOUND')
    rng = random.Random(42)
    for i in range(100):
        bars, price = [], 100.
        for minute in range(20):
            close = price+rng.uniform(-.3,.3)
            bars.append((minute*60,price,max(price,close)+rng.uniform(0,.4),min(price,close)-rng.uniform(0,.4),close))
            price = close
        compare('random_path_'+str(i),bars,dict(stop_multiple=rng.choice([.75,1,1.25,1.5]),
                target_r=rng.choice([.75,1,1.5,2]),time_minutes=rng.choice([5,10,90]),
                break_even_r=rng.choice([None,.5,.75]),trail_after_r=rng.choice([None,1.]),trail_by_r=.5))
    from gold_s4_trade_economics_v1_launcher import require_session, begin_session, consume_validation
    reject('no_real_session', require_session)
    reject('synthetic_no_user_session', lambda: begin_session('fake'))
    reject('no_real_validator_permit', lambda: consume_validation('fake', Path('.')))
    from training_holdout_guard_v1 import check_path
    reject('holdout', lambda: check_path(Path.cwd()/'future_holdout/gold_s4_v4/data.csv', Path.cwd()))
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        space, receipt, audit = {'candidates':['synthetic']}, {'count':1}, {'synthetic':True}
        control = rows[0]
        inventory = [('reference_signal_loaded',None,receipt),('reference_execution_pass','E0_REFERENCE_EXECUTION',control),
                     ('path_semantics_frozen',None,audit),('search_space_frozen',None,space)]
        for row in rows:
            inventory += [('candidate_start',row['candidate_id'],None),('candidate_end',row['candidate_id'],row)]
        inventory += [('economics_gate_complete',None,[r['gate'] for r in rows]),('pareto_complete',None,decision['pareto_frontier']),
                      ('selection_complete',None,decision),('research_freeze',None,decision)]
        for kind,candidate,record in inventory:
            event(run,kind,digest(space),candidate,record)
        verify_chain(run,space,receipt,control,audit,rows,decision)
        checks['event_chain_raw_payloads'] = True
        reject('no_append_after_freeze',lambda:event(run,'candidate_start',digest(space)))
        changed = copy.deepcopy(rows)
        changed[0]['pooled']['pnl_r'] += 1
        reject('event_payload_mutation',lambda:verify_chain(run,space,receipt,control,audit,changed,decision))
    from gold_s4_trade_economics_path_audit import replay_entries
    from validate_gold_s4_trade_economics_v1_run import independent_entries, independent_gate, independent_selection, validate_outputs
    from gold_s4_trade_economics_v1_support import configuration, load_inputs, ROOT, entry_projection, write, sha
    from unittest.mock import patch
    from datetime import datetime, timezone
    def entry_at(i, stamp, **changes):
        iso = datetime.fromtimestamp(stamp,timezone.utc).replace(tzinfo=None).isoformat()
        return dict(entry,trade_id='SYNTHETIC_'+str(i),entry_index=i,entry_epoch=stamp,
                    entry_time_api=iso,entry_time_actual_utc=iso,spread_observed=False,raw_episode_id=i,**changes)
    def compatible(name, entries, bars, rule=base):
        a = replay_entries(entries,bars,rule)
        b = independent_entries(entries,bars,rule)
        equal(list(a),list(b))
        checks[name+'_independent'] = True
        return a
    es = [entry_at(0,0),entry_at(1,60)]
    bars = [(0,100,100.1,99.9,100),(60,100,100.1,99.9,100),(120,100,103,99.9,102)]
    ledger,reason,conflict = compatible('overlap',es,bars)
    check('overlap_whole_candidate_failure',reason == 'REFERENCE_ENTRY_OVERLAP' and len(ledger) == 1 and conflict == es[1]['trade_id'])
    bars = [(0,100,100.1,98,99),(60,100,103,99.9,102)]
    check('cooldown_whole_candidate_failure',compatible('cooldown',es,bars)[1] == 'REFERENCE_COOLDOWN_CONFLICT')
    es = [entry_at(0,0),entry_at(1,900)]
    bars = [(0,100,100.1,98,99),(900,100,103,99.9,102)]
    check('exact_cooldown_boundary',compatible('cooldown_boundary',es,bars)[1] is None)
    # Same-timestamp CLOSE/barrier is too late to free an OPEN entry.
    es = [entry_at(0,0),entry_at(1,60)]
    bars = [(0,100,100.1,99.9,100),(60,100,103,99.9,102)]
    check('intrabar_same_time_conflict',compatible('same_time_intrabar',es,bars)[1] == 'REFERENCE_ENTRY_OVERLAP')
    es = [entry_at(0,0),entry_at(1,60,entry_price=101.)]
    bars = [(0,100,100.1,99.9,100),(60,101,101.1,100.9,101),(120,101,103.1,100.9,103)]
    check('open_exit_before_entry',compatible('same_time_open',es,bars,dict(base,time_minutes=1))[1] is None)
    es = [entry_at(i,i*900) for i in range(6)]
    bars = [(i*900,100,100.1,98,99) for i in range(6)]
    check('daily_guard_preserved',compatible('daily_guard',es,bars)[1] == 'REFERENCE_DAILY_LOSS_CONFLICT')
    equal(gates(p,folds),independent_gate(p,folds))
    equal(select(rows),independent_selection(rows))
    rejected = dict(rows[0],candidate_id='INCOMPATIBLE',execution_compatibility='FAIL')
    check('incompatible_never_frontier', 'INCOMPATIBLE' not in select(rows+[rejected])['pareto_frontier'])
    reject('real_input_denied',lambda:load_inputs({}))
    reject('real_validation_denied',lambda:__import__('validate_gold_s4_trade_economics_v1_run').run_authorized(Path('.'),'fake'))
    config,ref,space = configuration()
    check('signal_reference_review',len(ref['entries']) == 787 and ref['accepted_trade_count'] == ref['signal_count'] == 787)
    check('path_semantics_review',config['overlap_policy'] == 'SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY')
    check('search_space_review',len(space['candidates']) == 22 and not space['diagnostic_enabled'])
    check('production_protection_test',all(sha(ROOT/k) == v for k,v in config['protected_sha256'].items()))
    check('bat_static_test','gold_s4_trade_economics_v1_launcher.py' in (ROOT/'RUN_TRAINING.bat').read_text(encoding='utf-8')
          and '--status' in (ROOT/'CHECK_STATUS.bat').read_text(encoding='utf-8'))
    # Real coordinator, fake inputs only; no USER receipt or real input loader.
    import gold_s4_trade_economics_v1 as runner
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        stamp = 1514764800
        es = [entry_at(i,stamp+i*86400) for i in range(3)]
        bars = [(stamp+i*86400,100,103,99.5,102) for i in range(3)]
        cspace = dict(candidates=[base,dict(base,candidate_id='SYNTHETIC_TP',target_r=1.),
                                 dict(base,candidate_id='SYNTHETIC_OVERLAP',target_r=10.,time_minutes=10000),
                                 dict(base,candidate_id='SYNTHETIC_COOLDOWN',target_r=10.)])
        cfg = dict(folds=[['A','2018-01-01','2018-01-02'],['B','2018-01-02','2018-01-03'],['C','2018-01-03','2018-01-04']])
        rref = dict(entries=entry_projection(es),entry_stream_sha256=digest(entry_projection(es)))
        baseline = replay_entries(es,bars,base)[0]
        rref['metrics'] = independent_metrics(baseline,3)
        for name,value in [('approved_config.json',cfg),('signal_reference.json',rref),('predeclared_search_space.json',cspace),
                           ('path_audit.json',json.loads((ROOT/'gold_s4_trade_economics_path_audit.json').read_text(encoding='utf-8')))]:
            write(run/name,value)
        with patch('gold_s4_trade_economics_v1_launcher.require_session'),patch.object(runner,'load_inputs',return_value=(es,bars)),patch.object(runner,'check_control'):
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                decision,records = runner.research(run,cfg,rref,cspace)
        validate_outputs(run,cfg,rref,cspace,es,bars)
        check('validator_fixture_test',True)
        check('failed_candidates_no_economics', all(r['pooled'] is None and r['gate']['economic_gate'] == 'NONE'
              for r in records[2:]) and records[2]['fail_reason'] == 'REFERENCE_ENTRY_OVERLAP'
              and records[3]['fail_reason'] == 'REFERENCE_COOLDOWN_CONFLICT')
        path=run/'ledgers/E0_REFERENCE_EXECUTION.json'
        original=path.read_text(encoding='utf-8')
        altered=json.loads(original); altered[0]['net_r'] += .01; write(path,altered)
        reject('validator_ledger_tamper',lambda:validate_outputs(run,cfg,rref,cspace,es,bars))
        path.write_text(original,encoding='utf-8')
        decision['selected_candidate']='FAKE'; write(run/'selection.json',decision)
        reject('validator_selection_tamper',lambda:validate_outputs(run,cfg,rref,cspace,es,bars))
    with tempfile.TemporaryDirectory() as directory:
        run=Path(directory)
        with patch('gold_s4_trade_economics_v1_launcher.require_session'),patch.object(runner,'load_inputs',return_value=(es,bars)),patch.object(runner,'check_control',side_effect=ValueError('REFERENCE_EXECUTION_MISMATCH')),patch.object(runner,'evaluate',return_value={'candidate_id':'E0_REFERENCE_EXECUTION'}) as evaluation:
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                reject('reference_failure_stops_search',lambda:runner.research(run,cfg,rref,cspace))
            check('no_candidates_after_reference_failure',evaluation.call_count == 1)
    # Reconcile against unchanged original S5 on invented bars only.
    import numpy as np
    import pandas as pd
    import gold_gemini_execution_semantics_v1 as original_s5
    times = pd.date_range('2018-01-17',periods=80,freq='h')
    frame = pd.DataFrame(dict(decision_time_api=times,decision_time_actual_utc=times,
        cohort='SYNTHETIC_ONLY',raw_signal=True,raw_episode_id=np.arange(80),
        session_api=True,session_actual_utc=True,M1_RSI=50.,ATR=1.,OPEN=100.,CLOSE=100.,
        HIGH=np.where(np.arange(80)%3 == 0,100.2,103.),LOW=np.where(np.arange(80)%3 == 0,97.,99.),
        effective_spread_points=30.,spread_observed=False))
    original_ledger,_ = original_s5.simulate(frame,original_s5.SIMULATORS[-1])
    es=[]
    for row in original_ledger:
        e={k:row[k] for k in __import__('gold_s4_trade_economics_v1_support').ENTRY_FIELDS}
        e['entry_epoch']=times[row['entry_index']].timestamp()
        es.append(e)
    synthetic_bars=np.column_stack([[t.timestamp() for t in times],frame.OPEN,frame.HIGH,frame.LOW,frame.CLOSE])
    actual,reason,_=compatible('original_s5_synthetic',es,synthetic_bars)
    check('original_s5_full_accounting',reason is None and len(actual) == len(original_ledger))
    for a,b in zip(actual,original_ledger):
        for key in ('exit_price','exit_reason','net_r','stress_r','risk_mult','risk_budget','account_pnl','balance_after'):
            equal(a[key],b[key],key)
    checks['original_s5_metrics_reproduction_synthetic']=True
    # Static prohibition of model operations throughout the new real workflow.
    import ast
    for name in ('gold_s4_trade_economics_v1.py','gold_s4_trade_economics_v1_support.py',
                 'gold_s4_trade_economics_path_audit.py','validate_gold_s4_trade_economics_v1_run.py'):
        tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
        check('no_model_calls_'+name, not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and n.func.attr in ('fit','predict','predict_proba','load_model') for n in ast.walk(tree)))
    from manual_training_launcher_v1 import manual_parent_chain
    check('scheduler_rejected',not manual_parent_chain([dict(Name='cmd.exe',SessionId=1,ParentProcessId=2),
          dict(Name='taskeng.exe',ProcessId=2)],'fake'))
    import gold_s4_trade_economics_v1_launcher as launcher
    with patch.object(launcher,'require_session'):
        token=launcher.validation_permit(Path('.'))
        reject('wrong_run_permit',lambda:launcher.consume_validation(token,Path('other')))
        reject('consumed_permit',lambda:launcher.consume_validation(token,Path('.')))
    checks['no_model_training_static']=True
    for group, name in [('accounting_test','accounting_example'),('same_bar_conflict_test','same_bar_stop_first'),
                        ('gap_test','gap_exact_inherited_barrier_fill'),('time_stop_test','time_open_precedes_intrabar'),
                        ('break_even_test','be_next_bar_only'),('trailing_test','trail_ratchet'),
                        ('pareto_logic_test','pareto'),('selection_logic_test','incompatible_never_frontier'),
                        ('event_chain_test','event_chain_raw_payloads'),('manual_execution_guard_test','no_real_session'),
                        ('holdout_guard_test','holdout'),('entry_compatibility_test','overlap_whole_candidate_failure'),
                        ('cooldown_compatibility_test','cooldown_whole_candidate_failure')]:
        check(group,checks[name])
    return checks


if __name__ == '__main__':
    print(json.dumps(dict(overall='PASS', scope='STATIC_SYNTHETIC_ONLY', checks=tests(),
                         MODEL_TRAINING_EXECUTED=False, REAL_ECONOMICS_RESEARCH_EXECUTED=False,
                         REAL_VALIDATION_EXECUTED=False, REAL_RESEARCH_EXECUTED=False, HISTORICAL_DATA_USED=False)))
