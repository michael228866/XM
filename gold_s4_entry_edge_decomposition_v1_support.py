"""Frozen descriptive accounting; importing this module never reads trade data."""
import bisect
import csv
import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'gold_s4_entry_edge_decomposition_v1'
CONFIG = EXPERIMENT + '_config.json'
EVENTS = ['reference_loaded', 'reference_reproduction_pass', 'feature_inventory_frozen',
          'bucket_spec_frozen', 'univariate_complete', 'bivariate_complete',
          'stability_complete', 'research_freeze', 'validation_result', 'finalization']
FLAGS = dict(MODEL_TRAINING_EXECUTED=False, REAL_RESEARCH_EXECUTED=False,
             HISTORICAL_DATA_USED=False, ENTRY_FILTERING_EXECUTED=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_path(root, name):
    from training_holdout_guard_v1 import check_path
    root = Path(root).resolve()
    path = check_path(root / name, ROOT)
    require(path.is_relative_to(root), 'Path escape')
    return path


def configuration():
    config = read(ROOT/CONFIG)
    for name, expected in config['source_bindings'].items():
        require(sha(safe_path(ROOT, name)) == expected, 'Source drift: '+name)
    ref, spec, inventory = [read(ROOT/config[k]) for k in ('reference_binding', 'bucket_spec', 'feature_inventory')]
    from gold_s4_entry_edge_feature_inventory_v1 import validate_spec
    validate_spec(spec, inventory)
    require(ref['candidate_id'] == 'A_NO_LONG_HTF' and ref['accepted_trade_count'] == 787, 'Reference binding')
    require(ref['folds'] == config['folds'], 'Fold binding')
    check_production(config)
    return config, ref, spec, inventory


def check_production(config):
    for name, expected in config['protected_sha256'].items():
        require(sha(ROOT/name) == expected, 'Production changed: '+name)


def sources(ref):
    result = [ref[k] for k in ('original_ledger', 'control_ledger', 'control_review', 'prior_validator', 'evidence_index', 'feature_schema')]
    for fold in ref['fold_sources']:
        result.extend([fold['audit'], fold['probability'], *fold['score_chunks']])
    return result


def check_binding_metadata(ref):
    seals = {}
    for item in sources(ref):
        seal_path = safe_path(ROOT, item['seal'])
        require(sha(seal_path) == item['seal_sha256'], 'Archive seal changed')
        if item['seal'] not in seals:
            seals[item['seal']] = read(seal_path)['file_sha256']
        name = safe_path(ROOT, item['path']).relative_to(seal_path.parent).as_posix()
        require(seals[item['seal']][name] == item['sha256'], 'Sealed source mismatch')
        require(safe_path(ROOT, item['path']).is_file(), 'Missing retained artifact: '+item['path'])
    schema = read(safe_path(ROOT, ref['feature_schema']['path']))
    require(schema['reference_features'] == ref['feature_columns'], 'Sealed feature column order')
    for name, expected in ref['semantic_bindings'].items():
        require(sha(safe_path(ROOT, name)) == expected, 'Semantic source changed')


def sealed_file(item):
    from gold_s4_entry_edge_decomposition_v1_launcher import require_session
    require_session()
    path = safe_path(ROOT, item['path'])
    require(sha(path) == item['sha256'], 'Sealed data changed: '+item['path'])
    return path


def load_reference(ref):
    """Only copies immutable ledgers; no simulation, signals, models or fitting."""
    check_binding_metadata(ref)
    original = read(sealed_file(ref['original_ledger']))
    control = read(sealed_file(ref['control_ledger']))
    require(len(original) == len(control) == 787, 'REFERENCE_TRADESET_MISMATCH')
    require(len({r['trade_id'] for r in original}) == 787, 'REFERENCE_TRADESET_MISMATCH')
    shared = ('trade_id', 'entry_index', 'entry_time_api', 'entry_time_actual_utc', 'entry_price',
              'exit_price', 'exit_reason', 'net_r', 'stress_r', 'sl_distance', 'tp_distance',
              'spread_points', 'spread_observed', 'raw_episode_id', 'same_bar_both_hit')
    for old, new in zip(original, control):
        for key in shared:
            a, b = old[key], new[key]
            same = math.isclose(a, b, rel_tol=0, abs_tol=1e-12) if type(a) is float and type(b) is float else a == b
            require(same, 'REFERENCE_TRADESET_MISMATCH')
        stamp = datetime.fromisoformat(old['exit_time_actual_utc']).replace(tzinfo=timezone.utc).timestamp()
        require(stamp == new['exit_epoch'], 'REFERENCE_TRADESET_MISMATCH')
    times = [r['entry_time_actual_utc'] for r in original]
    require(times == sorted(times) and len(set(times)) == 787, 'REFERENCE_TRADESET_MISMATCH')
    augmented = []
    for old, diagnostic in zip(original, control):
        duration = (datetime.fromisoformat(old['exit_time_actual_utc']) -
                    datetime.fromisoformat(old['entry_time_actual_utc'])).total_seconds() / 60
        require(duration == diagnostic['holding_minutes'], 'REFERENCE_TRADESET_MISMATCH')
        augmented.append(dict(old, **{k: diagnostic[k] for k in
            ('holding_minutes', 'mfe_completed_bar_r', 'mae_completed_bar_r')}))
    return original, augmented


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics(rows, total, days):
    require(days > 0 and total >= len(rows), 'Invalid metric denominator')
    net, stress = [[r[k] for r in rows] for k in ('net_r', 'stress_r')]
    require(all(finite(v) for v in net+stress), 'Nonfinite R')
    wins, losses = [x for x in net if x > 0], [x for x in net if x < 0]
    gain, loss = math.fsum(wins), -math.fsum(losses)
    sg, sl = math.fsum(x for x in stress if x > 0), -math.fsum(x for x in stress if x < 0)
    count = len(rows)
    avgw, avgl = (gain/len(wins) if wins else 0), (loss/len(losses) if losses else 0)
    balance = peak = dd = 0.
    for value in net:
        balance += value
        peak = max(peak, balance)
        dd = min(dd, balance-peak)
    wr = len(wins)/count if count else None
    be = avgl/(avgl+avgw) if avgl+avgw else None
    out = dict(trades=count, trade_share=count/total if total else 0, wins=len(wins), losses=len(losses),
        flat_or_breakeven=count-len(wins)-len(losses), realized_win_rate=wr,
        trades_per_day=count/days, profit_factor=gain/loss if loss else None,
        pf_status='FINITE' if loss else 'NO_LOSSES' if gain else 'NO_PROFIT_OR_LOSS',
        mean_r=math.fsum(net)/count if count else None, pnl_r=math.fsum(net), max_drawdown_r=dd,
        stress_pf=sg/sl if sl else None, stress_pf_status='FINITE' if sl else 'NO_LOSSES' if sg else 'NO_PROFIT_OR_LOSS',
        stress_mean_r=math.fsum(stress)/count if count else None, avg_win_r=avgw, avg_loss_r=avgl,
        payoff_ratio=avgw/avgl if avgl else None, break_even_wr=be,
        break_even_adjusted_wr_edge=wr-be if wr is not None and be is not None else None,
        tp_exits=sum(r['exit_reason']=='take_profit' for r in rows),
        sl_exits=sum(r['exit_reason']=='stop_loss' for r in rows),
        timeout_exits=sum(r['exit_reason']=='timeout' for r in rows),
        tp_first_wr=sum(r['exit_reason']=='take_profit' for r in rows)/count if count else None)
    for field, label in [('mfe_completed_bar_r','mfe'), ('mae_completed_bar_r','mae'),
                         ('holding_minutes','holding_time'), ('model_probability','predicted_score')]:
        values = [r[field] for r in rows if finite(r.get(field))]
        out['avg_'+label] = statistics.mean(values) if values else None
        out['median_'+label] = statistics.median(values) if values else None
        out[label+'_available_count'] = len(values)
    out['mfe_mae_ratio'] = out['avg_mfe']/out['avg_mae'] if out['avg_mae'] else None
    return out


def reporting_days(folds):
    return sum((datetime.fromisoformat(end)-datetime.fromisoformat(start)).days for _,start,end in folds)


def reproduce(rows, ref):
    try:
        result = metrics(rows, len(rows), reporting_days(ref['folds']))
        for key, value in ref['metrics'].items():
            require(finite(result[key]) and math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12), key)
        require(result['trades'] == ref['accepted_trade_count'], 'Trade count')
        for (name,start,end), expected in zip(ref['folds'], ref['fold_metrics']):
            subset = [r for r in rows if start <= r['entry_time_api'] < end]
            actual = metrics(subset, len(rows), (datetime.fromisoformat(end)-datetime.fromisoformat(start)).days)
            for key,value in expected.items():
                if key != 'fold':
                    require(finite(actual[key]) and math.isclose(actual[key],value,rel_tol=0,abs_tol=1e-12), key)
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('REFERENCE_TRADESET_MISMATCH') from error


def bucket(row, dimension):
    values = [row.get(k) for k in dimension['fields']]
    labels, kind = dimension['labels'], dimension['kind']
    if any(v is None for v in values):
        return 'MISSING'
    if kind in ('hour', 'weekday'):
        stamp = datetime.fromisoformat(values[0])
        require(stamp.tzinfo is None or stamp.utcoffset().total_seconds() == 0, 'UTC required')
        return labels[stamp.hour//4 if kind == 'hour' else stamp.weekday()]
    if not all(finite(v) for v in values):
        return 'MISSING'
    if kind == 'alignment':
        require(all(v in (-1,1) for v in values), 'Trend domain')
        return 'ALIGNED' if all(v == 1 for v in values) else 'OPPOSED' if all(v == -1 for v in values) else 'MIXED'
    if kind == 'sign':
        value, pivot = values[0], dimension['pivot']
        return labels[0 if value < pivot else 1 if value == pivot else 2]
    require(kind == 'bins', 'Unknown grouping rule')
    return labels[bisect.bisect_right(dimension['edges'], values[0])]


def classify(metric, dimensions, missing=False):
    n, pf, mean, stress = (metric[k] for k in ('trades','profit_factor','mean_r','stress_pf'))
    sample = 'DESCRIPTIVE_ONLY' if n < (20 if dimensions == 2 else 30) or missing else 'LOW_CONFIDENCE' if n < 50 else 'USABLE'
    edge = 'NEUTRAL_BUCKET'
    if not missing and n >= 30 and finite(pf):
        if pf > 1 and mean > 0 and metric['pnl_r'] > 0:
            edge = 'POSITIVE_EDGE_BUCKET'
        if pf < .85 and mean < 0:
            edge = 'NEGATIVE_EDGE_BUCKET'
        if n >= 50 and pf >= 1.10 and mean >= .05 and finite(stress) and stress >= .95:
            edge = 'ROBUST_POSITIVE_EDGE_BUCKET'
    return dict(sample_confidence=sample, edge_classification=edge)


def persistence(folds, rule):
    adequate = len(folds) == len(rule['required_folds']) and all(f['trades'] >= rule['minimum_each_fold'] and finite(f['profit_factor']) for f in folds)
    pfs = [f['profit_factor'] for f in folds if finite(f['profit_factor'])]
    means = [f['mean_r'] for f in folds if finite(f['mean_r'])]
    stable = adequate and min(pfs) >= rule['minimum_pf'] and sum(p > 1 for p in pfs) >= rule['positive_folds_required']
    weak = adequate and max(pfs) < rule['negative_maximum_pf'] and sum(p < rule['negative_weak_pf'] for p in pfs) >= rule['negative_weak_folds_required'] and all(m < 0 for m in means)
    state = 'STABLE' if stable else 'UNSTABLE' if not adequate or min(pfs) < rule['minimum_pf'] else 'MIXED'
    return dict(stability=state, persistent_weakness=weak, fold_median_pf=statistics.median(pfs) if pfs else None,
        worst_fold_pf=min(pfs) if pfs else None, fold_median_mean_r=statistics.median(means) if means else None,
        worst_fold_mean_r=min(means) if means else None)


def group_tables(rows, spec, folds, bivariate=False):
    dims = {d['name']:d for d in spec['dimensions']}
    combinations = spec['pairs'] if bivariate else [[name] for name in dims]
    output, fold_output = [], []
    for names in combinations:
        buckets = {labels:[] for labels in itertools.product(*(dims[n]['labels'] for n in names))}
        for row in rows:
            buckets[tuple(bucket(row, dims[n]) for n in names)].append(row)
        require(sum(map(len,buckets.values())) == len(rows), 'Partition lost trades')
        for labels, subset in buckets.items():
            identity = dict(group_id=' x '.join(names)+':'+ '|'.join(labels), dimensions=names, buckets=list(labels))
            local_folds = []
            for fold,start,end in folds:
                selected = [r for r in subset if start <= r['entry_time_api'] < end]
                fm = dict(identity, fold=fold, **metrics(selected,len(rows),(datetime.fromisoformat(end)-datetime.fromisoformat(start)).days))
                local_folds.append(fm)
            metric = metrics(subset,len(rows),reporting_days(folds))
            result = dict(identity, trade_ids=[r['trade_id'] for r in subset], **metric,
                          **classify(metric,len(names),'MISSING' in labels), **persistence(local_folds,spec['stability']))
            result['research_candidate'] = None
            pf, stress = result['profit_factor'], result['stress_pf']
            if 'MISSING' not in labels and result['trades'] >= 50 and finite(pf) and finite(stress):
                if pf >= 1.05 and result['mean_r'] > 0 and stress >= .95 and result['stability'] == 'STABLE':
                    result['research_candidate'] = 'ENTRY_FILTER_RESEARCH_CANDIDATE'
                if pf <= .75 and result['mean_r'] < 0 and stress <= .75 and result['persistent_weakness']:
                    result['research_candidate'] = 'ENTRY_EXCLUSION_RESEARCH_CANDIDATE'
            output.append(result)
            fold_output.extend(local_folds)
    return output, fold_output


def summaries(univariate, bivariate, spec):
    cells = univariate+bivariate
    positive = [r for r in cells if r['edge_classification'] in ('POSITIVE_EDGE_BUCKET','ROBUST_POSITIVE_EDGE_BUCKET')]
    negative = [r for r in cells if r['edge_classification']=='NEGATIVE_EDGE_BUCKET']
    def rank(r):
        return (-(2 if r['sample_confidence']=='USABLE' else 1), -(r['stress_pf'] if finite(r['stress_pf']) else -1),
                -r['profit_factor'], -r['mean_r'], {'STABLE':0,'MIXED':1,'UNSTABLE':2}[r['stability']], -r['pnl_r'],r['group_id'])
    positive.sort(key=rank)
    negative.sort(key=lambda r:(-r['trades'],r['stress_pf'] if finite(r['stress_pf']) else float('inf'),r['profit_factor'],r['group_id']))
    inventory = dict(number_of_dimensions=len(spec['dimensions']), number_of_univariate_buckets=len(univariate),
        number_of_bivariate_pairs=len(spec['pairs']), number_of_bivariate_cells=len(bivariate),
        number_of_inferential_cells=sum(r['sample_confidence']!='DESCRIPTIVE_ONLY' for r in cells),
        number_of_nonempty_cells=sum(r['trades']>0 for r in cells), higher_order_cells=0,
        bucket_counts_include_missing_and_empty=True, inference='Exploratory dependent overlapping tests; no confirmatory significance or production selection')
    stable = dict(counts={k:sum(r['stability']==k for r in cells) for k in ('STABLE','MIXED','UNSTABLE')},
        positive_edge_buckets=len(positive), robust_positive_buckets=sum(r['edge_classification']=='ROBUST_POSITIVE_EDGE_BUCKET' for r in cells),
        negative_edge_buckets=len(negative), research_candidates=[r['group_id'] for r in cells if r['research_candidate']],
        research_result='EDGE_BUCKETS_FOUND' if any(r['research_candidate']=='ENTRY_FILTER_RESEARCH_CANDIDATE' for r in cells) else 'NO_ROBUST_EDGE_BUCKETS_FOUND',
        production_filter=None, production_promoted=False)
    return positive, negative, inventory, stable


def csv_write(path, rows, fields=None):
    keys = fields or list(rows[0])
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def event(run, name, records):
    path = Path(run)/'research_events.jsonl'
    chain = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    require(len(chain) < len(EVENTS) and EVENTS[len(chain)] == name, 'Event order')
    payload = dict(sequence=len(chain), event=name, at_utc=datetime.now(timezone.utc).isoformat(),
                   previous_sha256=chain[-1]['event_sha256'] if chain else None, records=records)
    payload['event_sha256'] = digest(payload)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(payload,allow_nan=False)+'\n')


def artifact_hashes(run, names):
    return {name:sha(Path(run)/name) for name in names}
