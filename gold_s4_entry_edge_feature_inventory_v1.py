"""Allowlisted entry-time snapshot audit. Historical reads require USER session."""
import math
from datetime import datetime, timezone

from gold_s4_entry_edge_decomposition_v1_support import require, finite, read, sealed_file

ALLOWED = {'entry_time_actual_utc', 'VOLA_RATIO', 'M5_TREND', 'H1_TREND',
           'BIAS_20', 'ROC_5', 'M1_RSI', 'BODY_PCT', 'model_probability'}
FORBIDDEN = {'future_r', 'future_mfe', 'future_mae', 'exit_type', 'exit_reason',
             'net_r', 'stress_r', 'holding_minutes', 'mfe_completed_bar_r', 'mae_completed_bar_r'}


def validate_spec(spec, inventory):
    approved = {f['field_name'] for f in inventory['fields'] if f['allowed_for_decomposition']
                and f['causal'] and f['decision_time_available']}
    require(approved == ALLOWED, 'Causal whitelist drift')
    dims = spec['dimensions']
    names = [d['name'] for d in dims]
    require(len(set(names)) == len(names) and len(spec['pairs']) <= 10, 'Dimension/pair count')
    require(spec['predeclared'] is True and spec['outcome_optimized'] is False and
            spec['higher_order_allowed'] is False and spec['bootstrap_enabled'] is False, 'Frozen descriptive scope')
    for d in dims:
        require(set(d['fields']) <= approved and not set(d['fields']) & FORBIDDEN, 'Noncausal grouping field')
        require(d['labels'][-1] == 'MISSING' and len(set(d['labels'])) == len(d['labels']), 'Missing/duplicate bucket')
        kind = d['kind']
        require(kind in ('bins','sign','alignment','hour','weekday'), 'Unsupported grouping')
        if kind == 'bins':
            require(all(finite(v) for v in d['edges']) and sorted(set(d['edges'])) == d['edges'] and
                    len(d['labels']) == len(d['edges'])+2, 'Bucket boundaries')
        elif kind in ('sign','alignment'):
            require(len(d['labels']) == 4, 'Sign/alignment buckets')
        elif kind == 'hour':
            require(d['timezone']=='UTC' and len(d['labels'])==7, 'Time buckets')
        elif kind == 'weekday':
            require(d['timezone']=='UTC' and len(d['labels'])==8, 'Weekday buckets')
    require(len({tuple(p) for p in spec['pairs']}) == len(spec['pairs']), 'Duplicate pair')
    for pair in spec['pairs']:
        require(len(pair)==2 and len(set(pair))==2 and set(pair)<=set(names), 'Higher-order/unknown pair')
    require(spec['stability'] == dict(required_folds=['2018_2020','2021_2022','2023_2024'],
        minimum_each_fold=8,positive_folds_required=2,minimum_pf=.70,negative_maximum_pf=1.0,
        negative_weak_folds_required=2,negative_weak_pf=.85), 'Stability rule drift')
    require(spec['sample_guards'] == dict(univariate=30,bivariate=20,strong=50), 'Sample guard drift')
    require(spec['classification'] == dict(positive_pf=1.0, robust_pf=1.10, robust_mean_r=.05,
        robust_stress_pf=.95, negative_pf=.85), 'Classification drift')
    require(spec['candidate'] == dict(positive_n=50, positive_pf=1.05, positive_mean_r=0,
        positive_stress_pf=.95, negative_n=50, negative_pf=.75, negative_mean_r=0, negative_stress_pf=.75),
        'Candidate criteria drift')


def check_values(row):
    for field in ('M1_RSI','BODY_PCT','model_probability','VOLA_RATIO','H1_TREND','M5_TREND'):
        value = row.get(field)
        if value is None:
            continue
        require(finite(value), 'Nonfinite snapshot')
        if field in ('BODY_PCT','model_probability'):
            require(0 <= value <= 1, 'Probability/body domain')
        elif field == 'M1_RSI':
            require(0 <= value <= 100, 'RSI domain')
        elif field == 'VOLA_RATIO':
            require(value >= 0, 'Volatility domain')
        else:
            require(value in (-1,1), 'Trend domain')


def attach_snapshots(original, control, ref):
    from gold_s4_entry_edge_decomposition_v1_launcher import require_session
    require_session()
    import numpy as np
    rows = [dict(r) for r in control]
    offset = 0
    for source, (fold,start,end) in zip(ref['fold_sources'],ref['folds']):
        with np.load(sealed_file(source['audit']),allow_pickle=False) as archive:
            times = archive['score_ns']
        with np.load(sealed_file(source['probability']),allow_pickle=False) as archive:
            probability = archive['probability']
        require(len(times)==len(probability) and np.all(np.diff(times)>0), 'Score chronology')
        selected = [(i,r['entry_index']-offset) for i,r in enumerate(original)
                    if start <= r['entry_time_api'] < end]
        start_ns = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()*1e9)
        end_ns = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()*1e9)
        require(len(times)>0 and int(times[0])>=start_ns and int(times[-1])<end_ns, 'Fold boundaries')
        for i,position in selected:
            require(0<=position<len(times), 'Entry index outside fold')
            timestamp = datetime.fromtimestamp(int(times[position])/1e9,timezone.utc).replace(tzinfo=None).isoformat()
            require(timestamp == original[i]['entry_time_actual_utc'], 'Feature timestamp join')
            value = float(probability[position])
            rows[i].update(model_probability=value if math.isfinite(value) else None,fold=fold,
                           feature_decision_time=timestamp,feature_snapshot={})
        first = 0
        for chunk in source['score_chunks']:
            with np.load(sealed_file(chunk),allow_pickle=False) as archive:
                values = archive['values']
            require(values.shape == (chunk['rows'],len(ref['feature_columns'])), 'Feature matrix schema')
            for i,position in selected:
                if first <= position < first+len(values):
                    snapshot = {k:float(v) if math.isfinite(float(v)) else None
                                for k,v in zip(ref['feature_columns'],values[position-first])}
                    rows[i]['feature_snapshot'] = snapshot
                    rows[i].update({k:snapshot[k] for k in ALLOWED if k in snapshot})
            first += len(values)
        require(first == len(times), 'Feature/score row alignment')
        offset += len(times)
    for row in rows:
        require(len(row.get('feature_snapshot',{}))==len(ref['feature_columns']), 'Missing entry snapshot')
        check_values(row)
    return rows


def availability(rows, inventory):
    result = []
    for field in inventory['fields']:
        item = dict(field)
        if field['allowed_for_decomposition']:
            values = [r.get(field['field_name']) for r in rows]
            item['missing_rate'] = sum(v is None for v in values)/len(rows) if rows else None
            item['missing_rate_status'] = 'MEASURED_ENTRY_SNAPSHOT'
        result.append(item)
    return dict(scope='USER_ENTRY_SNAPSHOT_AUDIT',fields=result,trade_count=len(rows))
