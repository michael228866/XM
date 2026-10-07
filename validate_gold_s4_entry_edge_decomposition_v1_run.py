"""Independent arithmetic, grouping, joins and evidence validation; no optimizer imports."""
import csv
import itertools
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from gold_s4_entry_edge_decomposition_v1_support import ROOT, read, write, sha, digest, require, safe_path

TABLES = ['entry_edge_univariate.csv','entry_edge_bivariate.csv','entry_edge_fold_stability.csv',
          'entry_edge_score_calibration.csv','entry_edge_regime_summary.csv']
JSONS = ['entry_edge_top_positive_buckets.json','entry_edge_top_negative_buckets.json',
         'entry_edge_stability_summary.json','multiple_testing_inventory.json']
ORDER = ['reference_loaded','reference_reproduction_pass','feature_inventory_frozen','bucket_spec_frozen',
         'univariate_complete','bivariate_complete','stability_complete','research_freeze','validation_result','finalization']


def equal(actual,expected,path='root'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),'Keys: '+path)
        for k,v in expected.items():
            equal(actual[k],v,path+'.'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'Length: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):
            equal(a,b,path+str(i))
    elif isinstance(expected,float):
        require(isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12),'Number: '+path)
    else:
        require(actual==expected,'Value: '+path)


def accounting(rows,total,days):
    net = [r['net_r'] for r in rows]
    stress = [r['stress_r'] for r in rows]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in net+stress),'Nonfinite ledger')
    n=len(rows)
    w=sum(x>0 for x in net); l=sum(x<0 for x in net)
    gp=math.fsum(max(x,0) for x in net); gl=math.fsum(max(-x,0) for x in net)
    sp=math.fsum(max(x,0) for x in stress); sl=math.fsum(max(-x,0) for x in stress)
    aw=gp/w if w else 0; al=gl/l if l else 0
    be=al/(al+aw) if al+aw else None
    path=[0.]
    for v in net:
        path.append(path[-1]+v)
    dd=min((value-max(path[:i+1]) for i,value in enumerate(path)),default=0.)
    out=dict(trades=n,trade_share=n/total if total else 0,wins=w,losses=l,flat_or_breakeven=n-w-l,
        realized_win_rate=w/n if n else None,trades_per_day=n/days,profit_factor=gp/gl if gl else None,
        pf_status='FINITE' if gl else 'NO_LOSSES' if gp else 'NO_PROFIT_OR_LOSS',
        mean_r=math.fsum(net)/n if n else None,pnl_r=math.fsum(net),max_drawdown_r=dd,
        stress_pf=sp/sl if sl else None,stress_pf_status='FINITE' if sl else 'NO_LOSSES' if sp else 'NO_PROFIT_OR_LOSS',
        stress_mean_r=math.fsum(stress)/n if n else None,avg_win_r=aw,avg_loss_r=al,payoff_ratio=aw/al if al else None,
        break_even_wr=be,break_even_adjusted_wr_edge=w/n-be if n and be is not None else None,
        tp_exits=sum(r['exit_reason']=='take_profit' for r in rows),sl_exits=sum(r['exit_reason']=='stop_loss' for r in rows),
        timeout_exits=sum(r['exit_reason']=='timeout' for r in rows),tp_first_wr=sum(r['exit_reason']=='take_profit' for r in rows)/n if n else None)
    for field,label in [('mfe_completed_bar_r','mfe'),('mae_completed_bar_r','mae'),('holding_minutes','holding_time'),('model_probability','predicted_score')]:
        values=[r[field] for r in rows if type(r.get(field)) in (int,float) and math.isfinite(r[field])]
        out['avg_'+label]=statistics.mean(values) if values else None
        out['median_'+label]=statistics.median(values) if values else None
        out[label+'_available_count']=len(values)
    out['mfe_mae_ratio']=out['avg_mfe']/out['avg_mae'] if out['avg_mae'] else None
    return out


def membership(row,d):
    allowed={'entry_time_actual_utc','VOLA_RATIO','M5_TREND','H1_TREND','BIAS_20','ROC_5','M1_RSI','BODY_PCT','model_probability'}
    require(set(d['fields'])<=allowed,'Causality rejection')
    values=[row.get(k) for k in d['fields']]
    if any(x is None for x in values):
        return 'MISSING'
    if d['kind'] in ('hour','weekday'):
        time=datetime.fromisoformat(values[0])
        require(time.tzinfo is None or time.utcoffset().total_seconds()==0,'Non-UTC')
        return d['labels'][int(time.hour/4) if d['kind']=='hour' else time.weekday()]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
        return 'MISSING'
    if d['kind']=='alignment':
        require(all(x in (-1,1) for x in values),'Trend domain')
        return {(-1,-1):'OPPOSED',(1,1):'ALIGNED'}.get(tuple(values),'MIXED')
    if d['kind']=='sign':
        return d['labels'][int(values[0]>=d['pivot'])+int(values[0]>d['pivot'])]
    require(d['kind']=='bins','Unknown rule')
    return d['labels'][sum(values[0]>=edge for edge in d['edges'])]


def tables(rows,spec,folds):
    lookup={d['name']:d for d in spec['dimensions']}
    require(len(lookup)==len(spec['dimensions']) and len(spec['pairs'])<=10,'Search dimension count')
    days=sum((datetime.fromisoformat(b)-datetime.fromisoformat(a)).days for _,a,b in folds)
    single=[]; paired=[]; details=[]
    for names in [[name] for name in lookup]+spec['pairs']:
        require(1<=len(names)<=2 and len(set(names))==len(names),'Higher-order mining')
        seen=[]
        for labels in itertools.product(*[lookup[name]['labels'] for name in names]):
            subset=[r for r in rows if all(membership(r,lookup[name])==label for name,label in zip(names,labels))]
            seen.extend(r['trade_id'] for r in subset)
            identity=dict(group_id=' x '.join(names)+':'+ '|'.join(labels),dimensions=names,buckets=list(labels))
            fm=[]
            for fold,start,end in folds:
                part=[r for r in subset if start<=r['entry_time_api']<end]
                fm.append(dict(identity,fold=fold,**accounting(part,len(rows),(datetime.fromisoformat(end)-datetime.fromisoformat(start)).days)))
            result=dict(identity,trade_ids=[r['trade_id'] for r in subset],**accounting(subset,len(rows),days))
            n,p,m,s=(result[k] for k in ('trades','profit_factor','mean_r','stress_pf'))
            missing='MISSING' in labels
            result['sample_confidence']='DESCRIPTIVE_ONLY' if missing or n<(20 if len(names)==2 else 30) else 'LOW_CONFIDENCE' if n<50 else 'USABLE'
            edge='NEUTRAL_BUCKET'
            if not missing and n>=30 and p is not None:
                if p>1 and m>0 and result['pnl_r']>0: edge='POSITIVE_EDGE_BUCKET'
                if p<.85 and m<0: edge='NEGATIVE_EDGE_BUCKET'
                if n>=50 and p>=1.10 and m>=.05 and s is not None and s>=.95: edge='ROBUST_POSITIVE_EDGE_BUCKET'
            result['edge_classification']=edge
            pfs=[f['profit_factor'] for f in fm if f['profit_factor'] is not None]
            means=[f['mean_r'] for f in fm if f['mean_r'] is not None]
            enough=len(fm)==3 and len(pfs)==3 and all(f['trades']>=8 for f in fm)
            stable=enough and min(pfs)>=.70 and sum(pf>1 for pf in pfs)>=2
            weak=enough and max(pfs)<1 and sum(pf<.85 for pf in pfs)>=2 and all(value<0 for value in means)
            result.update(stability='STABLE' if stable else 'UNSTABLE' if not enough or min(pfs)<.70 else 'MIXED',
                persistent_weakness=weak,fold_median_pf=statistics.median(pfs) if pfs else None,
                worst_fold_pf=min(pfs) if pfs else None,fold_median_mean_r=statistics.median(means) if means else None,
                worst_fold_mean_r=min(means) if means else None,research_candidate=None)
            if not missing and n>=50 and p is not None and s is not None:
                if p>=1.05 and m>0 and s>=.95 and stable: result['research_candidate']='ENTRY_FILTER_RESEARCH_CANDIDATE'
                if p<=.75 and m<0 and s<=.75 and weak: result['research_candidate']='ENTRY_EXCLUSION_RESEARCH_CANDIDATE'
            (single if len(names)==1 else paired).append(result)
            details.extend(fm)
        require(sorted(seen)==sorted(r['trade_id'] for r in rows),'Group membership partition')
    return single,paired,details


def derived(single,paired,spec):
    rows=single+paired
    positive=[r for r in rows if r['edge_classification'] in ('POSITIVE_EDGE_BUCKET','ROBUST_POSITIVE_EDGE_BUCKET')]
    positive.sort(key=lambda r:(-({'USABLE':2,'LOW_CONFIDENCE':1}[r['sample_confidence']]),
        -(r['stress_pf'] if r['stress_pf'] is not None else -1),-r['profit_factor'],-r['mean_r'],
        {'STABLE':0,'MIXED':1,'UNSTABLE':2}[r['stability']],-r['pnl_r'],r['group_id']))
    negative=sorted([r for r in rows if r['edge_classification']=='NEGATIVE_EDGE_BUCKET'],
        key=lambda r:(-r['trades'],r['stress_pf'] if r['stress_pf'] is not None else float('inf'),r['profit_factor'],r['group_id']))
    inventory=dict(number_of_dimensions=len(spec['dimensions']),number_of_univariate_buckets=len(single),
        number_of_bivariate_pairs=len(spec['pairs']),number_of_bivariate_cells=len(paired),
        number_of_inferential_cells=sum(r['sample_confidence']!='DESCRIPTIVE_ONLY' for r in rows),
        number_of_nonempty_cells=sum(r['trades']>0 for r in rows),higher_order_cells=0,bucket_counts_include_missing_and_empty=True,
        inference='Exploratory dependent overlapping tests; no confirmatory significance or production selection')
    stability=dict(counts={k:sum(r['stability']==k for r in rows) for k in ('STABLE','MIXED','UNSTABLE')},
        positive_edge_buckets=len(positive),robust_positive_buckets=sum(r['edge_classification']=='ROBUST_POSITIVE_EDGE_BUCKET' for r in rows),
        negative_edge_buckets=len(negative),research_candidates=[r['group_id'] for r in rows if r['research_candidate']],
        research_result='EDGE_BUCKETS_FOUND' if any(r['research_candidate']=='ENTRY_FILTER_RESEARCH_CANDIDATE' for r in rows) else 'NO_ROBUST_EDGE_BUCKETS_FOUND',
        production_filter=None,production_promoted=False)
    return positive,negative,inventory,stability


def csv_equal(path,expected):
    with Path(path).open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); actual=list(reader)
        require(set(reader.fieldnames)==set(expected[0]),'CSV schema')
    require(len(actual)==len(expected),'CSV row count')
    for a,e in zip(actual,expected):
        converted={}
        for k,v in e.items():
            if v is None: converted[k]=None if a[k]=='' else a[k]
            elif isinstance(v,(list,dict)): converted[k]=json.loads(a[k])
            elif isinstance(v,bool): converted[k]={'True':True,'False':False}[a[k]]
            elif isinstance(v,int): converted[k]=int(a[k])
            elif isinstance(v,float): converted[k]=float(a[k])
            else: converted[k]=a[k]
        equal(converted,e)


def check_chain(run,complete=False):
    records=[json.loads(line) for line in (Path(run)/'research_events.jsonl').read_text(encoding='utf-8').splitlines()]
    expected=ORDER if complete else ORDER[:8]
    require([r['event'] for r in records]==expected,'Event order/count')
    previous=None; stamp=''
    for i,row in enumerate(records):
        raw=dict(row); claimed=raw.pop('event_sha256')
        require(digest(raw)==claimed and row['sequence']==i and row['previous_sha256']==previous,'Canonical raw-payload hash')
        require(row['at_utc']>=stamp,'Event clock')
        for name,expected_hash in row['records'].items():
            require(sha(safe_path(run,name))==expected_hash,'Frozen event artifact mutation')
        previous=claimed; stamp=row['at_utc']
    require(set(TABLES+JSONS+['entry_snapshots.json','reference_ledger.json','feature_inventory.json',
            'bucket_spec.json','reference_reproduction.json'])<=set(records[7]['records']),'Incomplete research freeze')
    return True


def verify_outputs(run,rows,spec,inventory,ref):
    require(len(rows)==ref['accepted_trade_count'] and len({r['trade_id'] for r in rows})==len(rows),'Trade count/identity')
    require([r['entry_time_api'] for r in rows]==sorted(r['entry_time_api'] for r in rows),'Chronology')
    approved={f['field_name'] for f in inventory['fields'] if f['allowed_for_decomposition'] and f['causal'] and f['decision_time_available']}
    for d in spec['dimensions']:
        require(set(d['fields'])<=approved,'Feature availability/causality')
    for r in rows:
        require(sum(start<=r['entry_time_api']<end for _,start,end in ref['folds'])==1,'Fold membership')
    require(read(run/'bucket_spec.json')==spec,'Frozen bucket definitions')
    expected_inventory=[]
    for field in inventory['fields']:
        item=dict(field)
        if item['allowed_for_decomposition']:
            item['missing_rate']=sum(r.get(item['field_name']) is None for r in rows)/len(rows)
            item['missing_rate_status']='MEASURED_ENTRY_SNAPSHOT'
        expected_inventory.append(item)
    equal(read(run/'feature_inventory.json'),dict(scope='USER_ENTRY_SNAPSHOT_AUDIT',fields=expected_inventory,trade_count=len(rows)))
    days=sum((datetime.fromisoformat(b)-datetime.fromisoformat(a)).days for _,a,b in ref['folds'])
    overall=accounting(rows,len(rows),days)
    equal({k:overall[k] for k in ref['metrics']},ref['metrics'])
    # Reproduction occurs before feature attachment, so optional score counts stay zero there.
    reproduction=accounting([{k:v for k,v in r.items() if k!='model_probability'} for r in rows],len(rows),days)
    equal(read(run/'reference_reproduction.json'),reproduction)
    for (name,start,end),expected in zip(ref['folds'],ref['fold_metrics']):
        part=accounting([r for r in rows if start<=r['entry_time_api']<end],len(rows),(datetime.fromisoformat(end)-datetime.fromisoformat(start)).days)
        equal({k:part[k] for k in expected if k!='fold'},{k:v for k,v in expected.items() if k!='fold'})
    single,paired,folds=tables(rows,spec,ref['folds'])
    for name,expected in zip(TABLES,[single,paired,folds,[r for r in single if r['dimensions']==['model_score']],
        [r for r in single if r['dimensions'][0] in ('volatility','short_term_trend','trend_alignment')]]):
        csv_equal(run/name,expected)
    pos,neg,multiple,stability=derived(single,paired,spec)
    for name,expected in zip(JSONS,[pos,neg,stability,multiple]):
        equal(read(run/name),expected)
    return True


def independent_rows(ref):
    """Independently timestamp-join archived scores, using no runner join logic."""
    from gold_s4_entry_edge_decomposition_v1_launcher import require_session
    require_session()
    import numpy as np
    def data(item):
        path=safe_path(ROOT,item['path'])
        seal=safe_path(ROOT,item['seal'])
        require(sha(seal)==item['seal_sha256'],'Source seal')
        require(read(seal)['file_sha256'][path.relative_to(seal.parent).as_posix()]==item['sha256']==sha(path),'Source identity')
        return path
    old=read(data(ref['original_ledger'])); control=read(data(ref['control_ledger']))
    require(len(old)==len(control)==787,'Reference count')
    result=[dict(r, **{k: d[k] for k in ('holding_minutes','mfe_completed_bar_r','mae_completed_bar_r')})
            for r,d in zip(old,control)]
    for a,b in zip(old,control):
        for k in set(a)&set(b): equal(a[k],b[k],k)
        exit_time=datetime.fromisoformat(a['exit_time_actual_utc'])
        require(exit_time.replace(tzinfo=timezone.utc).timestamp()==b['exit_epoch'],'Exit identity')
        require((exit_time-datetime.fromisoformat(a['entry_time_actual_utc'])).total_seconds()/60 == b['holding_minutes'], 'Holding time identity')
    offset=0
    for src,(fold,start,end) in zip(ref['fold_sources'],ref['folds']):
        with np.load(data(src['audit']),allow_pickle=False) as a: times=a['score_ns'].copy()
        with np.load(data(src['probability']),allow_pickle=False) as a: probs=a['probability'].copy()
        require(len(times)==len(probs) and np.all(np.diff(times)>0),'Probability chronology')
        selected={}
        for i,row in enumerate(old):
            if start<=row['entry_time_api']<end:
                target=int(datetime.fromisoformat(row['entry_time_actual_utc']).replace(tzinfo=timezone.utc).timestamp()*1e9)
                position=int(np.searchsorted(times,target))
                require(position<len(times) and int(times[position])==target and row['entry_index']==offset+position,'Independent entry join')
                selected[position]=i
                p=float(probs[position])
                result[i].update(fold=fold,feature_decision_time=row['entry_time_actual_utc'],model_probability=p if math.isfinite(p) else None)
        first=0
        for chunk in src['score_chunks']:
            with np.load(data(chunk),allow_pickle=False) as a: values=a['values']
            require(values.shape==(chunk['rows'],len(ref['feature_columns'])),'Independent feature schema')
            for position,i in selected.items():
                if first<=position<first+len(values):
                    snapshot=dict(zip(ref['feature_columns'],[float(x) if math.isfinite(float(x)) else None for x in values[position-first]]))
                    result[i]['feature_snapshot']=snapshot
                    for name in ('VOLA_RATIO','M5_TREND','H1_TREND','BIAS_20','ROC_5','M1_RSI','BODY_PCT'): result[i][name]=snapshot[name]
            first+=len(values)
        require(first==len(times),'Independent feature row count')
        offset+=len(times)
    return old,result


def run_authorized(run,token):
    from gold_s4_entry_edge_decomposition_v1_launcher import consume_validation
    consume_validation(token,run)
    require(not (run/'validator_attempt.json').exists(),'One-shot validation')
    write(run/'validator_attempt.json',dict(scope='USER_LAUNCHED_INDEPENDENT_VALIDATION'))
    try:
        config=read(ROOT/'gold_s4_entry_edge_decomposition_v1_config.json')
        ref=read(ROOT/config['reference_binding']); spec=read(ROOT/config['bucket_spec']); inventory=read(ROOT/config['feature_inventory'])
        for name,expected in config['source_bindings'].items(): require(sha(safe_path(ROOT,name))==expected,'Source binding')
        for name,expected in config['protected_sha256'].items(): require(sha(ROOT/name)==expected,'Production protection')
        equal(read(run/'approved_config.json'),config)
        equal(read(run/'reference_binding.json'),ref)
        old,rows=independent_rows(ref)
        require(read(run/'reference_ledger.json')==old,'Original trade mutation')
        require(read(run/'entry_snapshots.json')==rows,'Entry/exit/R/feature identity')
        verify_outputs(run,rows,spec,inventory,ref)
        check_chain(run)
        result=dict(overall='PASS',failed_checks=[],holdout_used=False,production_changed=False)
    except Exception as error:
        result=dict(overall='FAIL',failed_checks=[str(error)],holdout_used=False,production_changed=False)
    write(run/'validator.json',result)
    return result


if __name__=='__main__':
    raise SystemExit('Independent historical validation is USER-launched workflow only; use synthetic fixtures for certification')
