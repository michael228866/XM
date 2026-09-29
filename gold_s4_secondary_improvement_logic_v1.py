"""Predeclared S4 research rules; import performs no training or data access."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

METRICS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day',
           'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')
REFERENCE = dict(zip(METRICS, (743, 421, 322, 0.566621803499327, 0.2905748924520923,
    0.8271090428367022, -0.07580700838089183, -56.32460722700263, -62.009878061758215,
    0.7857575699558063)))
TOLERANCE = 1e-12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def gate(pooled, folds):
    valid = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    safety = (all(valid(pooled.get(k)) for k in METRICS)
              and pooled['profit_factor'] >= .80 and pooled['mean_r'] >= -.10690539315994348
              and pooled['stress_pf'] >= .75 and bool(folds)
              and all(all(valid(row.get(k)) for k in ('trades', 'realized_win_rate', 'profit_factor'))
                      and row['trades'] >= 10 and row['realized_win_rate'] >= .45
                      and row['profit_factor'] >= .70 for row in folds))
    improved = safety and pooled['realized_win_rate'] > REFERENCE['realized_win_rate'] and pooled['trades_per_day'] > REFERENCE['trades_per_day']
    name = 'NONE'
    if improved:
        name = 'TARGET' if pooled['realized_win_rate'] >= .60 and pooled['trades_per_day'] >= .50 else (
            'STRONG' if pooled['realized_win_rate'] >= .58 and pooled['trades_per_day'] >= .40 else 'INTERESTING')
    return {'safety_pass': safety, 'interesting': improved, 'gate': name,
            'delta_wr': pooled['realized_win_rate']-REFERENCE['realized_win_rate'],
            'delta_trades_per_day': pooled['trades_per_day']-REFERENCE['trades_per_day']}


def select(records):
    eligible = [r for r in records if r['stage'] in (0, 2) and gate(r['pooled'], r['fold_metrics'])['safety_pass']]
    frontier = [r for r in eligible if not any(
        other['pooled']['realized_win_rate'] >= r['pooled']['realized_win_rate']
        and other['pooled']['trades_per_day'] >= r['pooled']['trades_per_day']
        and (other['pooled']['realized_win_rate'] > r['pooled']['realized_win_rate']
             or other['pooled']['trades_per_day'] > r['pooled']['trades_per_day']) for other in eligible)]
    def rank(r):
        m = r['pooled']
        return (-m['realized_win_rate'], -m['trades_per_day'], -m['stress_pf'],
                -m['profit_factor'], -m['mean_r'], abs(m['max_drawdown_r']), r['candidate_id'])
    ordered = sorted([r for r in frontier if r['stage'] == 2 and gate(r['pooled'], r['fold_metrics'])['interesting']], key=rank)
    return {'pareto_frontier': [r['candidate_id'] for r in sorted(frontier, key=rank)],
            'selected_candidate': ordered[0]['candidate_id'] if ordered else None,
            'research_result': 'IMPROVEMENT_FOUND' if ordered else 'NO_IMPROVEMENT_FOUND',
            'tie_break': [{'candidate_id': r['candidate_id'], 'ordered_key': list(rank(r))} for r in ordered],
            'tie_break_order': ['WR desc', 'TPD desc', 'stress PF desc', 'PF desc', 'Mean-R desc',
                                'absolute drawdown asc', 'candidate ID asc only for exact numerical ties']}


def check_reference(pooled, inventory, expected_models):
    if (any(not math.isclose(pooled[k], v, rel_tol=0, abs_tol=TOLERANCE) for k, v in REFERENCE.items())
            or [x['sha256'] for x in inventory] != [x['sha256'] for x in expected_models]):
        raise ValueError('REFERENCE_CONTROL_MISMATCH')


def features(base, base_names, wanted):
    """Six pointwise transforms of already causal, frozen features; no new bars."""
    import numpy as np
    values = {name: base[:, n] for n, name in enumerate(base_names)}
    htf = np.mean(np.column_stack([values[n] for n in ('H1_TREND', 'H4_TREND', 'Daily_TREND')]), axis=1)
    ltf = np.mean(np.column_stack([values[n] for n in ('M2_TREND', 'M5_TREND', 'M15_TREND')]), axis=1)
    values.update(S4_RSI_CENTER=(values['M1_RSI']-50)/50,
                  S4_MACD_ATR=values['MACD_HIST']/(np.abs(values['ATR'])+np.float32(1e-6)),
                  S4_BODY_VOL=values['BODY_PCT']/(np.abs(values['VOLA_RATIO'])+np.float32(1e-6)),
                  S4_HTF_AGREEMENT=htf, S4_LTF_AGREEMENT=ltf, S4_ALIGNMENT=htf*ltf)
    result = np.column_stack([values[name] for name in wanted]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite candidate features')
    return result


def calibration_split(times, maturity, legacy_maturity, subset, cutoff):
    import numpy as np
    fit = subset[(times[subset] < cutoff) & (maturity[subset] < cutoff) & (legacy_maturity[subset] < cutoff)]
    calibration = subset[times[subset] >= cutoff]
    if not len(fit) or not len(calibration) or np.intersect1d(fit, calibration).size:
        raise ValueError('Invalid inner chronological calibration split')
    if times[fit].max() >= times[calibration].min():
        raise ValueError('Calibration leakage')
    return fit, calibration


def weights(target, positive_multiplier):
    import numpy as np
    if not np.array_equal(np.unique(target), [0, 1]):
        raise ValueError('Both target classes required')
    result = len(target)/(2.0*np.bincount(target, minlength=2)[target])
    return result*np.where(target == 1, positive_multiplier, 1.)


def freeze_space(run, config):
    candidates = config['candidates']
    if not 1 <= len(candidates) <= 100 or len({x['candidate_id'] for x in candidates}) != len(candidates):
        raise ValueError('Invalid bounded search space')
    path = Path(run)/'predeclared_search_space.json'
    with path.open('x', encoding='utf-8') as out:
        json.dump(candidates, out, indent=2, allow_nan=False)
        out.write('\n')
    return digest(candidates)


def event(run, kind, space_hash, candidate_id=None, stage=None, record=None):
    path = Path(run)/'research_events.jsonl'
    previous = json.loads(path.read_text(encoding='utf-8').splitlines()[-1]) if path.exists() else None
    row = {'sequence': previous['sequence']+1 if previous else 0, 'event': kind,
           'at_utc': datetime.now(timezone.utc).isoformat(), 'space_sha256': space_hash,
           'candidate_id': candidate_id, 'stage': stage, 'record_sha256': digest(record) if record else None,
           'previous_sha256': previous['event_sha256'] if previous else None}
    row['event_sha256'] = digest(row)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(row, allow_nan=False)+'\n')
    return row
