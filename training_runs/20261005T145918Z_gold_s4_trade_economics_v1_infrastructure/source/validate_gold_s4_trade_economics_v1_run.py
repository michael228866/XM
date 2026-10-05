"""Independent pure exit/accounting reconstruction and USER-only validation."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from gold_s4_trade_economics_v1_support import ROOT, read, write, require, sha


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def independent_trade(entry, bars, rule):
    """Independent state transition implementation; no runner path helper calls."""
    entry_price, risk = entry['entry_price'], entry['sl_distance']
    require(risk > 0 and entry['spread_points'] >= 0, 'Entry risk')
    floor = entry_price - risk * rule['stop_multiple']
    ceiling = entry_price + (entry['tp_distance'] if rule['target_r'] is None else rule['target_r'] * risk)
    floor_name = 'stop_loss'
    favorable, adverse = 0., 0.
    last = None
    final = None
    for n, (stamp, op, hi, lo, cl) in enumerate(bars):
        require(all(math.isfinite(v) for v in (stamp, op, hi, lo, cl)), 'Finite bars')
        require(lo <= min(op, cl) <= max(op, cl) <= hi, 'OHLC')
        require(stamp >= entry['entry_epoch'] and (last is None or stamp > last), 'Chronology')
        require(n != 0 or (stamp == entry['entry_epoch'] and op == entry_price), 'Entry identity')
        elapsed = (stamp - entry['entry_epoch']) / 60
        last = stamp
        both = False
        if elapsed >= rule['time_minutes']:
            final = (stamp, op, 'timeout', n)
        elif (rule['no_progress_minutes'] is not None and elapsed >= rule['no_progress_minutes']
              and favorable / risk < .2):
            final = (stamp, op, 'no_progress', n)
        else:
            touched = (lo <= floor, hi >= ceiling)
            both = all(touched)
            if touched[0]:
                final = (stamp, floor, floor_name, n)
            elif touched[1]:
                final = (stamp, ceiling, 'take_profit', n)
        if final is not None:
            if final[2] in ('timeout', 'no_progress'):
                favorable, adverse = max(favorable, op-entry_price), max(adverse, entry_price-op)
            break
        favorable = max(favorable, hi-entry_price)
        adverse = max(adverse, entry_price-lo)
        changes = [(floor, floor_name)]
        if rule['break_even_r'] is not None and favorable >= rule['break_even_r'] * risk:
            changes.append((entry_price, 'break_even'))
        if rule['trail_after_r'] is not None and favorable >= rule['trail_after_r'] * risk:
            changes.append((entry_price + favorable - rule['trail_by_r'] * risk, 'trailing'))
        floor, floor_name = max(changes, key=lambda pair: pair[0])
    require(last is not None, 'Empty bars')
    if final is None:
        final = (last, cl, 'cohort_end', n)
        both = False
    stamp, price, reason, n = final
    denominator = risk + .01 * entry['spread_points']
    difference = price - entry_price
    return dict(entry, exit_epoch=stamp, exit_price=price, exit_reason=reason,
                same_bar_both_hit=bool(both), gross_price=difference, gross_r=difference/denominator,
                net_r=(difference-.01*(entry['spread_points']+5))/denominator,
                stress_r=(difference-.01*(entry['spread_points']+10))/denominator,
                holding_minutes=(stamp-entry['entry_epoch'])/60,
                mfe_completed_bar_r=favorable/denominator, mae_completed_bar_r=adverse/denominator, bars_to_exit=n)


def independent_metrics(ledger, days):
    import numpy as np
    require(days > 0, 'Period')
    r = np.asarray([t['net_r'] for t in ledger], dtype=float)
    stressed = np.asarray([t['stress_r'] for t in ledger], dtype=float)
    require(np.isfinite(r).all() and np.isfinite(stressed).all(), 'Finite accounting')
    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    avg_win = float(wins.mean()) if len(wins) else 0.
    avg_loss = float(-losses.mean()) if len(losses) else 0.
    n = len(r)
    equity = np.r_[0., r.cumsum()]
    stress_loss = float(-stressed[stressed < 0].sum())
    holding = [t['holding_minutes'] for t in ledger]
    reasons = ('stop_loss', 'take_profit', 'timeout', 'break_even', 'trailing', 'no_progress', 'cohort_end')
    return dict(trades=n, wins=len(wins), losses=len(losses), flat_or_breakeven=int((r == 0).sum()),
                realized_win_rate=len(wins)/n if n else 0., trades_per_day=n/days,
                profit_factor=gross_win/gross_loss if gross_loss else None,
                mean_r=float(r.mean()) if n else 0., pnl_r=float(r.sum()),
                max_drawdown_r=float((equity-np.maximum.accumulate(equity)).min()),
                stress_pf=float(stressed[stressed > 0].sum())/stress_loss if stress_loss else None,
                gross_win_r=gross_win, gross_loss_r=gross_loss, avg_win_r=avg_win, avg_loss_r=avg_loss,
                win_loss_payoff_ratio=avg_win/avg_loss if avg_loss else None,
                expected_r_per_trade=float(r.mean()) if n else 0.,
                pre_cost_pnl_r=float(np.sum([t['gross_r'] for t in ledger])),
                cost_r=float(np.sum([t['gross_r']-t['net_r'] for t in ledger])),
                avg_holding_time=float(np.mean(holding)) if n else 0.,
                median_holding_time=float(np.median(holding)) if n else 0.,
                exit_counts={reason: sum(t['exit_reason'] == reason for t in ledger) for reason in reasons})


def equal(actual, expected, context='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), context + ': keys')
        for key in expected:
            equal(actual[key], expected[key], context + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + ': length')
        for a, b in zip(actual, expected):
            equal(a, b, context)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), context + ': numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, context + ': mismatch')


def verify_chain(run, space, reference_receipt, control, audit, rows, decision):
    expected = [('reference_signal_loaded', None, reference_receipt),
                ('reference_execution_pass', 'E0_REFERENCE_EXECUTION', control),
                ('path_semantics_frozen', None, audit), ('search_space_frozen', None, space)]
    for row in rows:
        expected += [('candidate_start', row['candidate_id'], None), ('candidate_end', row['candidate_id'], row)]
    expected += [('economics_gate_complete', None, [r['gate'] for r in rows]),
                 ('pareto_complete', None, decision['pareto_frontier']),
                 ('selection_complete', None, decision), ('research_freeze', None, decision)]
    events = [json.loads(line) for line in (run/'research_events.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(events) == len(expected), 'Event inventory')
    previous, clock = None, None
    for n, (row, (kind, candidate, record)) in enumerate(zip(events, expected)):
        require(set(row) == {'sequence','event','at_utc','space_sha256','candidate_id','stage','record_sha256','previous_sha256','event_sha256'}, 'Event fields')
        require(type(row['sequence']) is int and row['sequence'] == n and row['stage'] is None, 'Event sequence')
        require(row['event'] == kind and row['candidate_id'] == candidate, 'Event order')
        require(row['record_sha256'] == (canonical(record) if record is not None else None), 'Raw event payload')
        require(row['space_sha256'] == canonical(space) and row['previous_sha256'] == previous, 'Event linkage')
        require(row['event_sha256'] == canonical({k:v for k,v in row.items() if k != 'event_sha256'}), 'Event hash')
        at = datetime.fromisoformat(row['at_utc'])
        require(at.tzinfo is not None and at.utcoffset().total_seconds() == 0 and (clock is None or at >= clock), 'UTC monotonic events')
        previous, clock = row['event_sha256'], at
    return previous


def independent_entries(entries, bars, rule):
    balance = 1000.
    ledger = []
    last_loss = None
    for entry in entries:
        now = entry['entry_epoch']
        if ledger:
            last = ledger[-1]
            if last['exit_epoch'] > now or (last['exit_epoch'] == now and last['exit_reason'] not in ('timeout','no_progress')):
                return ledger, 'REFERENCE_ENTRY_OVERLAP', entry['trade_id']
        if last_loss is not None and now < last_loss + 900:
            return ledger, 'REFERENCE_COOLDOWN_CONFLICT', entry['trade_id']
        day = int((now + 28800) // 86400)
        daily_pnl = sum(t['account_pnl'] for t in ledger if int((t['exit_epoch']+28800)//86400) == day)
        if daily_pnl <= -.05 * balance:
            return ledger, 'REFERENCE_DAILY_LOSS_CONFLICT', entry['trade_id']
        previous = [r['account_pnl'] for r in ledger[-30:]]
        pos = sum(max(0., x) for x in previous)
        neg = sum(max(0., -x) for x in previous)
        multiplier = .5 if len(previous) >= 18 and neg > 0 and pos / neg < 1.15 else 1.
        row = independent_trade(entry, bars[entry['entry_index']:], rule)
        row.update(risk_mult=multiplier, balance_before=balance, risk_budget=max(balance,0.)*.014*multiplier)
        row['account_pnl'] = row['net_r']*row['risk_budget']
        balance += row['account_pnl']
        row['balance_after'] = balance
        if row['net_r'] <= 0:
            last_loss = row['exit_epoch']
        ledger.append(row)
    return ledger, None, None


def independent_gate(p, folds):
    positive = p['profit_factor'] is not None and p['profit_factor'] > 1 and p['mean_r'] > 0 and p['pnl_r'] > 0
    safe = (p['profit_factor'] is not None and p['stress_pf'] is not None and p['stress_pf'] >= .90
            and p['realized_win_rate'] >= .50 and p['trades_per_day'] >= .2770043019163082
            and len(folds) == 3 and all(f['trades'] >= 10 and f['profit_factor'] is not None
                and f['profit_factor'] >= .85 and f['mean_r'] >= -.15 for f in folds))
    level = 'NONE'
    if positive and safe:
        level = 'POSITIVE'
        if p['profit_factor'] >= 1.1 and p['mean_r'] >= .05 and p['stress_pf'] >= 1:
            level = 'STRONG'
        if p['profit_factor'] >= 1.2 and p['mean_r'] >= .10 and p['stress_pf'] >= 1.05:
            level = 'TARGET'
    return dict(robustness_pass=bool(safe), economic_status='POSITIVE_EXPECTANCY' if positive else
                ('NEGATIVE_EXPECTANCY' if p['profit_factor'] is not None and p['stress_pf'] is not None and p['mean_r'] < 0 else 'NEAR_BREAK_EVEN'), economic_gate=level)


def independent_selection(rows):
    eligible = [r for r in rows if r['execution_compatibility'] == 'PASS' and r['gate']['robustness_pass']]
    frontier = []
    for row in eligible:
        p = row['pooled']
        dominated = False
        for other in eligible:
            q = other['pooled']
            if q['profit_factor'] >= p['profit_factor'] and q['mean_r'] >= p['mean_r'] and (q['profit_factor'],q['mean_r']) != (p['profit_factor'],p['mean_r']):
                dominated = True
        if not dominated:
            frontier.append(row)
    def key(r):
        p = r['pooled']
        return [-p['profit_factor'],-p['mean_r'],-p['stress_pf'],-p['pnl_r'],abs(p['max_drawdown_r']),-p['trades_per_day'],-p['realized_win_rate'],r['candidate_id']]
    frontier.sort(key=key)
    choices = [r for r in frontier if r['candidate_id'] != 'E0_REFERENCE_EXECUTION' and r['gate']['economic_gate'] != 'NONE']
    return dict(pareto_frontier=[r['candidate_id'] for r in frontier], selected_candidate=choices[0]['candidate_id'] if choices else None,
                research_result='POSITIVE_EXPECTANCY_FOUND' if choices else 'NO_POSITIVE_EXPECTANCY_FOUND',
                economic_gate=choices[0]['gate']['economic_gate'] if choices else 'NONE',
                tie_break=[dict(candidate_id=r['candidate_id'],ordered_key=key(r)) for r in choices])


def validate_outputs(run, config, ref, space, entries, bars):
    """Pure artifact validation, also exercised with fully synthetic fixtures."""
    equal(read(run/'approved_config.json'), config)
    equal(read(run/'signal_reference.json'), ref)
    equal(read(run/'predeclared_search_space.json'), space)
    rows = [json.loads(line) for line in (run/'candidate_results.jsonl').read_text(encoding='utf-8').splitlines()]
    require([r['candidate_id'] for r in rows] == [c['candidate_id'] for c in space['candidates']], 'Exact candidate inventory/order')
    for row, rule in zip(rows, space['candidates']):
        equal(row['config'], rule)
        expected, failure, conflict = independent_entries(entries, bars, rule)
        require(row['execution_compatibility'] == ('FAIL' if failure else 'PASS') and row['fail_reason'] == failure
                and row['conflict_entry_id'] == conflict, 'Independent execution compatibility')
        require(row['EXECUTION_COMPATIBILITY'] == row['execution_compatibility'] and row['FAIL_REASON'] == failure,
                'Explicit compatibility status')
        require(row['frozen_entry_count'] == len(entries) and row['completed_trades'] == len(expected)
                and row['entry_stream_sha256'] == ref['entry_stream_sha256'], 'Entry inventory')
        require(row['source_of_truth'] == 'SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY' and row['diagnostic'] is False, 'Official source of truth')
        path = run/'ledgers'/(rule['candidate_id']+'.json')
        require(row['ledger_path'] == path.relative_to(run).as_posix() and sha(path) == row['ledger_sha256'], 'Ledger binding')
        equal(read(path), expected, 'Independent trade ledger')
        if failure:
            equal(row['pooled'], None)
            equal(row['fold_metrics'], [])
            equal(row['gate'],dict(robustness_pass=False,economic_status='NOT_EVALUATED_INCOMPATIBLE',economic_gate='NONE'))
            continue
        folds, days = [], 0
        for name, start, end in config['folds']:
            duration = (datetime.fromisoformat(end)-datetime.fromisoformat(start)).days
            days += duration
            folds.append(dict(fold=name,**independent_metrics([e for e in expected if start <= e['entry_time_api'] < end],duration)))
        pooled = independent_metrics(expected,days)
        equal(row['pooled'],pooled)
        equal(row['fold_metrics'],folds)
        # Gate comparisons use independently rebuilt metrics, including exact boundary values.
        equal(row['gate'],independent_gate(pooled,folds))
    control = rows[0]
    require(control['execution_compatibility'] == 'PASS', 'Reference compatibility')
    equal({k:control['pooled'][k] for k in ref['metrics']},ref['metrics'])
    equal(read(run/'reference_execution.json'),control)
    decision = independent_selection(rows)
    equal(read(run/'selection.json'),decision)
    equal(read(run/'pareto_frontier.json'),[r for r in rows if r['candidate_id'] in decision['pareto_frontier']])
    signal_receipt = dict(candidate_id='A_NO_LONG_HTF',signal_count=len(entries),accepted_trade_count=len(entries),
                          entry_stream_sha256=ref['entry_stream_sha256'],direction='LONG',threshold=.75)
    equal(read(run/'reference_signal_loaded.json'), signal_receipt)
    tip = verify_chain(run,space,signal_receipt,control,read(run/'path_audit.json'),rows,decision)
    return tip


def run_authorized(run, permit):
    from gold_s4_trade_economics_v1_launcher import consume_validation
    consume_validation(permit,run)
    run = Path(run).resolve()
    require(run.parent == ROOT/'training_runs' and not (run/'FINALIZED.json').exists(), 'Open run only')
    with (run/'validator_attempt.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(started_at_utc=datetime.now(timezone.utc).isoformat(),one_shot=True),stream)
    result = dict(overall='FAIL',failed_checks=[],MODEL_TRAINING_EXECUTED=False)
    tip = None
    try:
        from gold_s4_trade_economics_v1_support import configuration, load_inputs
        config,ref,space = configuration()
        entries,bars = load_inputs(ref)
        equal(read(run/'path_audit.json'),read(ROOT/'gold_s4_trade_economics_path_audit.json'))
        tip = validate_outputs(run,config,ref,space,entries,bars)
        manifest = read(run/'manifest.json')
        require(manifest['MODEL_TRAINING_EXECUTED'] is False and manifest['manual_start'] is True
                and manifest['model']['trained'] is False, 'No model training')
        result['overall'] = 'PASS'
    except Exception as error:
        result['failed_checks'].append(str(error))
    write(run/'validator.json',result)
    (run/'validator.md').write_text('# Independent economics validation\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write(run/'validation_result.json',dict(event='validation_result',at_utc=datetime.now(timezone.utc).isoformat(),
          research_tip=tip,validator_sha256=sha(run/'validator.json'),overall=result['overall']))
    return result


if __name__ == '__main__':
    raise SystemExit('Independent real validation requires the USER workflow one-use permit')
