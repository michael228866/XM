"""Frozen metadata, accounting and gates; imports never load historical bars."""
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'gold_s4_trade_economics_v1'
CONFIG = EXPERIMENT + '_config.json'
CONTROL = 'E0_REFERENCE_EXECUTION'
ENTRY_FIELDS = ('trade_id', 'entry_index', 'entry_time_api', 'entry_time_actual_utc',
                'entry_price', 'sl_distance', 'tp_distance', 'spread_points',
                'spread_observed', 'raw_episode_id')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_path(root, name):
    from training_holdout_guard_v1 import check_path
    root = Path(root).resolve()
    path = check_path(root/name, ROOT)
    require(path.is_relative_to(root), 'Path escape')
    return path


def entry_projection(rows):
    return [{key: row[key] for key in ENTRY_FIELDS} for row in rows]


def load_inputs(ref):
    from gold_s4_trade_economics_v1_launcher import require_session
    require_session()
    import numpy as np
    parts = []
    seal = read(ROOT/ref['evidence_source_run']/'FINALIZED.json')['file_sha256']
    for item in ref['price_files']:
        path = safe_path(ROOT, item['path'])
        relative = path.relative_to(ROOT/ref['evidence_source_run']).as_posix()
        require(sha(path) == item['sha256'] == seal[relative], 'Sealed price evidence')
        with path.open('rb') as stream, np.load(stream, allow_pickle=False) as data:
            parts.append({name: data[name].copy() for name in ('TIME_DT','OPEN','HIGH','LOW','CLOSE','ATR')})
    fields = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    times = fields['TIME_DT']
    require(np.all(np.diff(times) > 0) and times.min() >= 1514764800000000000
            and times.max() < 1735689600000000000, 'Original 2018-2024 universe only')
    bars = np.column_stack([times / 1e9, *[fields[k] for k in ('OPEN','HIGH','LOW','CLOSE')]])
    entries = []
    for original in ref['entries']:
        e = dict(original)
        i = e['entry_index']
        require(0 <= i < len(bars), 'Entry index')
        expected_time = datetime.fromtimestamp(int(times[i]) / 1e9, timezone.utc).replace(tzinfo=None).isoformat()
        require(expected_time == e['entry_time_api'] == e['entry_time_actual_utc']
                and float(fields['OPEN'][i]) == e['entry_price'], 'Frozen entry price/time')
        require(math.isclose(max(float(fields['ATR'][i])*1.6,.6),e['sl_distance'],rel_tol=0,abs_tol=1e-12)
                and math.isclose(max(float(fields['ATR'][i])*1.3,1.5),e['tp_distance'],rel_tol=0,abs_tol=1e-12), 'Original ATR distances')
        e['entry_epoch'] = float(bars[i,0])
        entries.append(e)
    require(entry_projection(entries) == ref['entries'], 'All original entries retained')
    return entries, bars


def check_production(config, root=ROOT):
    for name, expected in config['protected_sha256'].items():
        require(sha(root/name) == expected, 'Production changed: ' + name)


def configuration(root=ROOT):
    from gold_s4_trade_economics_path_audit import verify_audit
    config = read(root/CONFIG)
    for name, expected in config['source_bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Source binding: ' + name)
    ref = read(root/config['reference_binding'])
    space = read(root/config['search_space'])
    audit = verify_audit(root)
    require(digest(space) == config['search_space_sha256'], 'Frozen search space')
    require(digest(ref) == config['signal_reference_sha256'], 'Frozen reference')
    require(ref['candidate_id'] == 'A_NO_LONG_HTF' and ref['threshold'] == .75 and ref['direction'] == 'LONG', 'Signal definition')
    require(digest(ref['features']) == ref['feature_set_sha256'], 'Feature set')
    require(ref['signal_count'] == ref['accepted_trade_count'] == len(ref['entries']) == 787, 'Accepted signal count')
    require(digest(entry_projection(ref['entries'])) == ref['entry_stream_sha256'], 'Entry identity')
    require(len({e['trade_id'] for e in ref['entries']}) == 787, 'Duplicate entry')
    for name, expected in ref['bindings'].items():
        require(sha(safe_path(root, name)) == expected, 'Prior reference: ' + name)
    require(sha(root/ref['reference_metadata']) == ref['reference_metadata_sha256'], 'Prior reference metadata')
    legacy = read(root/ref['reference_metadata'])
    amendment = read(root/legacy['amendment_reference'])
    require(amendment['effective_validation_status'] == 'PASS' and amendment['selected_candidate'] == ref['candidate_id'], 'Reference effective PASS')
    selected = read(root/ref['source_run']/'selected_candidate.json')['candidate']
    require(selected['pooled'] == ref['metrics'] and selected['fold_metrics'] == ref['fold_metrics'], 'Exact reference metrics')
    require(entry_projection(read(root/ref['ledger_path'])) == ref['entries'], 'Sealed accepted entry projection')
    latest = read(root/ref['latest_research_no_replacement']/'combined_result.json')
    require(latest['execution_status'] == 'PASS' and latest['research_result'] == 'NO_IMPROVEMENT_FOUND', 'Latest v4 result')
    require(audit['OVERLAP_POLICY'] == 'SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY', 'Formal single-position policy')
    require(config['overlap_policy'] == audit['OVERLAP_POLICY'], 'Overlap policy binding')
    require(config['folds'] == ref['folds'] and len(space['candidates']) == config['candidate_count'] <= 30, 'Frozen folds/count')
    require(len({c['candidate_id'] for c in space['candidates']}) == len(space['candidates']), 'Duplicate candidate')
    require(space['candidates'][0]['candidate_id'] == CONTROL, 'Control first')
    check_production(config, root)
    return config, ref, space


def metrics(rows, days):
    require(days > 0, 'Positive reporting period')
    net = [t['net_r'] for t in rows]
    stress = [t['stress_r'] for t in rows]
    require(all(math.isfinite(v) for v in net + stress), 'Nonfinite ledger')
    positive, negative = [v for v in net if v > 0], [v for v in net if v < 0]
    gain, loss = math.fsum(positive), -math.fsum(negative)
    stress_gain = math.fsum(v for v in stress if v > 0)
    stress_loss = -math.fsum(v for v in stress if v < 0)
    count, pnl = len(net), math.fsum(net)
    curve = peak = 0.
    drawdown = 0.
    for value in net:
        curve += value
        peak = max(peak, curve)
        drawdown = min(drawdown, curve - peak)
    average_win = gain / len(positive) if positive else 0.
    average_loss = loss / len(negative) if negative else 0.
    holding = [t['holding_minutes'] for t in rows]
    return dict(trades=count, wins=len(positive), losses=len(negative), flat_or_breakeven=count-len(positive)-len(negative),
                realized_win_rate=len(positive)/count if count else 0., trades_per_day=count/days,
                profit_factor=gain/loss if loss else None, mean_r=pnl/count if count else 0., pnl_r=pnl,
                max_drawdown_r=drawdown, stress_pf=stress_gain/stress_loss if stress_loss else None,
                gross_win_r=gain, gross_loss_r=loss, avg_win_r=average_win, avg_loss_r=average_loss,
                win_loss_payoff_ratio=average_win/average_loss if average_loss else None,
                expected_r_per_trade=pnl/count if count else 0.,
                pre_cost_pnl_r=math.fsum(t['gross_r'] for t in rows),
                cost_r=math.fsum(t['gross_r']-t['net_r'] for t in rows),
                avg_holding_time=statistics.mean(holding) if holding else 0.,
                median_holding_time=statistics.median(holding) if holding else 0.,
                exit_counts={kind: sum(t['exit_reason'] == kind for t in rows) for kind in
                             ('stop_loss', 'take_profit', 'timeout', 'break_even', 'trailing', 'no_progress', 'cohort_end')})


def summarize(rows, folds):
    result, days = [], 0
    for name, start, end in folds:
        duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
        days += duration
        result.append(dict(fold=name, **metrics([r for r in rows if start <= r['entry_time_api'] < end], duration)))
    return metrics(rows, days), result


def gates(pooled, folds, identity=True, overlap_ok=True):
    numeric = ('profit_factor', 'mean_r', 'pnl_r', 'stress_pf', 'realized_win_rate', 'trades_per_day')
    valid = all(isinstance(pooled.get(k), (int, float)) and math.isfinite(pooled[k]) for k in numeric)
    safe = (valid and identity and overlap_ok and pooled['realized_win_rate'] >= .5
            and pooled['trades_per_day'] >= .2770043019163082 and pooled['stress_pf'] >= .9
            and len(folds) == 3 and all(f['trades'] >= 10 and isinstance(f['profit_factor'], (int, float))
                and math.isfinite(f['profit_factor']) and f['profit_factor'] >= .85
                and math.isfinite(f['mean_r']) and f['mean_r'] >= -.15 for f in folds))
    positive = valid and pooled['profit_factor'] > 1 and pooled['mean_r'] > 0 and pooled['pnl_r'] > 0
    gate = 'NONE'
    if safe and positive:
        gate = 'POSITIVE'
        if pooled['profit_factor'] >= 1.10 and pooled['mean_r'] >= .05 and pooled['stress_pf'] >= 1:
            gate = 'STRONG'
        if pooled['profit_factor'] >= 1.20 and pooled['mean_r'] >= .10 and pooled['stress_pf'] >= 1.05:
            gate = 'TARGET'
    return dict(robustness_pass=bool(safe), economic_status='POSITIVE_EXPECTANCY' if positive else
                ('NEGATIVE_EXPECTANCY' if valid and pooled['mean_r'] < 0 else 'NEAR_BREAK_EVEN'), economic_gate=gate)


def select(rows):
    eligible = [r for r in rows if r.get('execution_compatibility') == 'PASS' and r['gate']['robustness_pass']]
    frontier = [r for r in eligible if not any(
        all(o['pooled'][k] >= r['pooled'][k] for k in ('profit_factor', 'mean_r')) and
        any(o['pooled'][k] > r['pooled'][k] for k in ('profit_factor', 'mean_r')) for o in eligible)]
    def rank(row):
        p = row['pooled']
        return (-p['profit_factor'], -p['mean_r'], -p['stress_pf'], -p['pnl_r'],
                abs(p['max_drawdown_r']), -p['trades_per_day'], -p['realized_win_rate'], row['candidate_id'])
    frontier.sort(key=rank)
    choices = sorted([r for r in frontier if r['candidate_id'] != CONTROL and r['gate']['economic_gate'] != 'NONE'], key=rank)
    chosen = choices[0] if choices else None
    return dict(pareto_frontier=[r['candidate_id'] for r in frontier], selected_candidate=chosen['candidate_id'] if chosen else None,
                research_result='POSITIVE_EXPECTANCY_FOUND' if chosen else 'NO_POSITIVE_EXPECTANCY_FOUND',
                economic_gate=chosen['gate']['economic_gate'] if chosen else 'NONE',
                tie_break=[dict(candidate_id=r['candidate_id'], ordered_key=list(rank(r))) for r in choices])


def event(run, kind, space_hash, candidate_id=None, record=None):
    path = Path(run)/'research_events.jsonl'
    previous = json.loads(path.read_text(encoding='utf-8').splitlines()[-1]) if path.exists() else None
    require(previous is None or previous['event'] != 'research_freeze', 'Research chain frozen')
    require((previous is None) == (kind == 'reference_signal_loaded'), 'Single chain root')
    row = dict(sequence=previous['sequence']+1 if previous else 0, event=kind,
               at_utc=datetime.now(timezone.utc).isoformat(), space_sha256=space_hash,
               candidate_id=candidate_id, stage=None, record_sha256=digest(record) if record is not None else None,
               previous_sha256=previous['event_sha256'] if previous else None)
    require(previous is None or row['at_utc'] >= previous['at_utc'], 'Clock moved backwards')
    row['event_sha256'] = digest(row)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    return row
