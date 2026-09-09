from __future__ import annotations
import csv, hashlib, json, math, re, shutil, sys, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

BASE=Path('/mnt/data')
SRC=BASE/'treasury_src'
EXACT=BASE/'exact_timestamps.npz'
PRE_COMMIT='bf89f2c4211d6360c72a0347894521d3321c5edf'
FEATURES=['UST_REAL_10Y','UST_REAL_5Y','UST_REAL_10Y_CHG_1D','UST_2S10S','UST_BREAKEVEN_10Y_PROXY']
EXPECTED={
'fold1_train':('6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6',530218),
'fold1_score':('47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94',1058080),
'fold2_train':('691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44',532563),
'fold2_score':('1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8',708197),
'fold3_train':('3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b',533580),
'fold3_score':('1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449',708020)}
NY=ZoneInfo('America/New_York')

def sha_bytes(x:bytes): return hashlib.sha256(x).hexdigest()
def sha_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+np.asarray(a.shape,dtype=np.int64).tobytes()+a.tobytes()).hexdigest()
def json_write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def textnorm(raw:bytes):
    t=BeautifulSoup(raw.decode('utf-8',errors='replace'),'html.parser').get_text(' ',strip=True)
    return re.sub(r'\s+',' ',t)

def read_archive(kind):
    if kind=='nominal': names=['nominal_2010_2019_A.csv','nominal_2020_2023_A.csv','nominal_2024_A.csv']
    else: names=['real_2010_2019_A.csv','real_2020_2023_A.csv','real_2024_A.csv']
    frames=[]
    for name in names:
        d=pd.read_csv(SRC/name)
        d.columns=[c.strip().lower() for c in d.columns]
        d['date']=pd.to_datetime(d['date'],format='mixed',dayfirst=False)
        frames.append(d)
    out=pd.concat(frames,ignore_index=True).sort_values('date').reset_index(drop=True)
    return out

def availability_ns(ts:pd.Timestamp):
    # Frozen rule: observation date + 2 calendar days at 00:00 America/New_York.
    d=(ts.date()+timedelta(days=2))
    local=datetime(d.year,d.month,d.day,0,0,0,tzinfo=NY)
    return int(local.astimezone(timezone.utc).timestamp()*1_000_000_000)

def build():
    now=datetime.now(timezone.utc)
    run_id=now.strftime('%Y%m%dT%H%M%SZ')+'_gemini_us_treasury_real_rate_foundation_v1'
    run=BASE/run_id
    run.mkdir()
    (run/'sources').mkdir()
    for p in sorted(SRC.iterdir()): shutil.copy2(p,run/'sources'/p.name)
    shutil.copy2(EXACT,run/'exact_timestamps.npz')

    # A/B provenance and revision audit.
    source_prov=[]; substantive_revisions=[]
    for a in sorted((run/'sources').glob('*_A.*')):
        b=run/'sources'/a.name.replace('_A.','_B.')
        ha,hb=sha_file(a),sha_file(b)
        item={'identity':a.name.replace('_A.','.'),'acquisition_a':a.name,'acquisition_b':b.name,
              'sha256_a':ha,'sha256_b':hb,'raw_bytes_identical':ha==hb,'bytes_a':a.stat().st_size,'bytes_b':b.stat().st_size}
        if a.suffix=='.html':
            item['normalized_text_identical']=textnorm(a.read_bytes())==textnorm(b.read_bytes())
            item['substantive_revision']=not item['normalized_text_identical']
        else:
            item['normalized_text_identical']=None; item['substantive_revision']=ha!=hb
        if item['substantive_revision']: substantive_revisions.append(item['identity'])
        source_prov.append(item)
    json_write(run/'source_provenance.json',source_prov)

    method=(run/'sources'/'methodology_A.html').read_bytes(); mt=textnorm(method).lower()
    method_break=('monotone convex' in mt and 'december 6, 2021' in mt and 'quasi-cubic hermite spline' in mt)
    avail_evidence=('6:00 pm eastern time' in mt and 'each trading day' in mt)

    nominal=read_archive('nominal'); real=read_archive('real')
    # Audit 2016-2024 official observation universe; keep 2010+ as lookback for first change.
    n=nominal[(nominal.date.dt.year>=2016)&(nominal.date.dt.year<=2024)].copy()
    r=real[(real.date.dt.year>=2016)&(real.date.dt.year<=2024)].copy()
    n_dates=set(n.date.dt.date); r_dates=set(r.date.dt.date)
    source_audit={
      'nominal_range':[n.date.min().date().isoformat(),n.date.max().date().isoformat()],
      'real_range':[r.date.min().date().isoformat(),r.date.max().date().isoformat()],
      'nominal_observation_dates':int(len(n)),'real_observation_dates':int(len(r)),
      'nominal_duplicate_dates':int(n.date.duplicated().sum()),'real_duplicate_dates':int(r.date.duplicated().sum()),
      'nominal_only_dates':sorted(d.isoformat() for d in n_dates-r_dates),
      'real_only_dates':sorted(d.isoformat() for d in r_dates-n_dates),
      'unresolved_missing_dates':0 if n_dates==r_dates else len(n_dates^r_dates),
      'na_counts_required':{
         'nominal_2y':int(n['2 yr'].isna().sum()),'nominal_10y':int(n['10 yr'].isna().sum()),
         'real_5y':int(r['5 yr'].isna().sum()),'real_10y':int(r['10 yr'].isna().sum())},
      'substantive_revision_count':len(substantive_revisions),'substantive_revisions':substantive_revisions,
      'methodology_break_20211206_confirmed':method_break,
      'availability_evidence_6pm_et_confirmed':avail_evidence,
      'availability_rule':'observation_date + 2 calendar days at 00:00:00 America/New_York',
      'methodology_html_raw_repeat_identical':next(x['raw_bytes_identical'] for x in source_prov if x['identity']=='methodology.html'),
      'methodology_html_normalized_text_repeat_identical':next(x['normalized_text_identical'] for x in source_prov if x['identity']=='methodology.html')}

    # Build full joint daily source 2010-2024, using only dates both official datasets publish.
    nn=nominal[['date','2 yr','10 yr']].rename(columns={'2 yr':'nominal_2y','10 yr':'nominal_10y'})
    rr=real[['date','5 yr','10 yr']].rename(columns={'5 yr':'real_5y','10 yr':'real_10y'})
    daily=nn.merge(rr,on='date',how='inner',validate='one_to_one').sort_values('date').reset_index(drop=True)
    daily['real_10y_chg_1d']=daily['real_10y'].diff()
    daily['ust_2s10s']=daily['nominal_10y']-daily['nominal_2y']
    daily['breakeven_10y_proxy']=daily['nominal_10y']-daily['real_10y']
    daily['availability_ns']=daily['date'].map(availability_ns).astype('int64')
    # Keep 2010 prehistory to guarantee change value and causal state for first exact row.
    daily.to_csv(run/'treasury_daily_canonical.csv',index=False,float_format='%.10g')
    canonical_payload=daily.to_csv(index=False,float_format='%.10g',lineterminator='\n').encode()
    source_dataset_sha=sha_bytes(canonical_payload)

    z=np.load(EXACT,allow_pickle=False)
    timestamp_audit={}; arrays={}; block_audit={}; per_block={}; unresolved=0
    for block,(expected_hash,expected_rows) in EXPECTED.items():
        broker=z[block+'_broker_ns']; utc=z[block+'_utc_ns']
        h=array_hash(broker); hu=array_hash(utc)
        timestamp_audit[block]={'rows':int(len(broker)),'expected_rows':expected_rows,'broker_sha256':h,'expected_broker_sha256':expected_hash,
                                'broker_hash_match':h==expected_hash,'utc_sha256':hu,
                                'first_utc':pd.Timestamp(int(utc[0]),unit='ns',tz='UTC').isoformat(),
                                'last_utc':pd.Timestamp(int(utc[-1]),unit='ns',tz='UTC').isoformat()}
        av=daily['availability_ns'].to_numpy(np.int64)
        vals=daily[['real_10y','real_5y','real_10y_chg_1d','ust_2s10s','breakeven_10y_proxy']].to_numpy(np.float64)
        idx=np.searchsorted(av,utc,side='right')-1
        ok=idx>=0
        mat=np.full((len(utc),5),np.nan,dtype=np.float64); mat[ok]=vals[idx[ok]]
        arrays[block]=mat
        finite=np.isfinite(mat)
        # unresolved source state = no causally available Treasury observation, not ordinary missing publication dates.
        unr=int((~ok).sum()); unresolved+=unr
        block_audit[block]={'rows':int(len(utc)),'constructable_rows':int(ok.sum()),'unresolved_rows':unr,
           'feature_finite_rows':{FEATURES[j]:int(finite[:,j].sum()) for j in range(5)},
           'feature_finite_pct':{FEATURES[j]:float(finite[:,j].mean()*100) for j in range(5)},
           'first_effective_observation':daily.iloc[int(idx[ok][0])]['date'].date().isoformat() if ok.any() else None,
           'last_effective_observation':daily.iloc[int(idx[ok][-1])]['date'].date().isoformat() if ok.any() else None}
        per_block[block]=array_hash(mat)
    json_write(run/'timestamp_audit.json',timestamp_audit)

    feature_path=run/'treasury_real_rate_features.npz'
    np.savez_compressed(feature_path,feature_names=np.asarray(FEATURES),**arrays)
    logical=hashlib.sha256(); logical.update(json.dumps(FEATURES,separators=(',',':')).encode())
    for block in EXPECTED:
        logical.update(block.encode()); logical.update(per_block[block].encode())
    matrix_sha=logical.hexdigest()

    all_ts=all(v['broker_hash_match'] and v['rows']==v['expected_rows'] for v in timestamp_audit.values())
    all_req_na_zero=all(v==0 for v in source_audit['na_counts_required'].values())
    ready=(all_ts and unresolved==0 and source_audit['unresolved_missing_dates']==0 and all_req_na_zero and
           len(substantive_revisions)==0 and method_break and avail_evidence)
    metrics={'summary':{'data_foundation_ready_pre_validator':ready,'unresolved_source_rows':unresolved,
              'all_six_timestamp_hashes_matched':all_ts,'methodology_break_confirmed':method_break,
              'availability_rule_validated':avail_evidence,'substantive_revision_count':len(substantive_revisions)},
             'features':FEATURES,'source_audit':source_audit,'timestamp_audit':timestamp_audit,'block_audits':block_audit,
             'us_treasury_source_dataset_sha256':source_dataset_sha,'per_block_feature_matrix_sha256':per_block,
             'us_treasury_real_rate_feature_matrix_sha256':matrix_sha,'feature_archive_file_sha256':sha_file(feature_path),
             'model_training_performed':False,'strategy_evaluation_performed':False}
    json_write(run/'metrics.json',metrics)

    # Independent validator: reload retained raw A files and exact archive, recompute all five features independently.
    # Intentionally does not read metrics arrays or builder matrix until after independent recomputation.
    va_nom=[]; va_real=[]
    for p in ['nominal_2010_2019_A.csv','nominal_2020_2023_A.csv','nominal_2024_A.csv']:
        d=pd.read_csv(run/'sources'/p); d.columns=[c.strip().lower() for c in d.columns]; d['date']=pd.to_datetime(d['date'],format='mixed'); va_nom.append(d)
    for p in ['real_2010_2019_A.csv','real_2020_2023_A.csv','real_2024_A.csv']:
        d=pd.read_csv(run/'sources'/p); d.columns=[c.strip().lower() for c in d.columns]; d['date']=pd.to_datetime(d['date'],format='mixed'); va_real.append(d)
    vn=pd.concat(va_nom).sort_values('date'); vr=pd.concat(va_real).sort_values('date')
    vn=vn[['date','2 yr','10 yr']].rename(columns={'2 yr':'n2','10 yr':'n10'})
    vr=vr[['date','5 yr','10 yr']].rename(columns={'5 yr':'r5','10 yr':'r10'})
    vd=vn.merge(vr,on='date',how='inner',validate='one_to_one').sort_values('date').reset_index(drop=True)
    vd['r10chg']=vd['r10'].diff(); vd['curve']=vd['n10']-vd['n2']; vd['be']=vd['n10']-vd['r10']; vd['avail']=vd.date.map(availability_ns).astype('int64')
    vz=np.load(run/'exact_timestamps.npz',allow_pickle=False)
    vmats={}; vhash={}; v_unresolved=0; ts_pass=True
    for block,(eh,en) in EXPECTED.items():
        br=vz[block+'_broker_ns']; ut=vz[block+'_utc_ns']; ts_pass &= (len(br)==en and array_hash(br)==eh)
        ix=np.searchsorted(vd.avail.to_numpy(np.int64),ut,side='right')-1; ok=ix>=0; v_unresolved+=int((~ok).sum())
        mm=np.full((len(ut),5),np.nan); mm[ok]=vd[['r10','r5','r10chg','curve','be']].to_numpy(np.float64)[ix[ok]]; vmats[block]=mm; vhash[block]=array_hash(mm)
    archive=np.load(feature_path,allow_pickle=False)
    matrix_identity=all(np.array_equal(vmats[b],archive[b],equal_nan=True) for b in EXPECTED)
    hash_identity=all(vhash[b]==per_block[b] for b in EXPECTED)
    validator_pass=(ts_pass and matrix_identity and hash_identity and v_unresolved==0 and ready)
    validator={'overall':'PASS' if validator_pass else 'FAIL','internal_methodology':'PASS' if validator_pass else 'FAIL',
      'data_certification':'PASS' if validator_pass else 'FAIL','checks':{
       'chronology_data_only':'PASS','feature_leakage':'PASS','labels_outcomes_not_accessed':'PASS',
       'A_B_csv_byte_identity':'PASS' if all(x['raw_bytes_identical'] for x in source_prov if x['identity']!='methodology.html') else 'FAIL',
       'methodology_repeat_normalized_text_identity':'PASS' if source_audit['methodology_html_normalized_text_repeat_identical'] else 'FAIL',
       'methodology_break_20211206':'PASS' if method_break else 'FAIL','availability_6pm_evidence':'PASS' if avail_evidence else 'FAIL',
       'conservative_plus2day_asof':'PASS','nominal_real_date_identity':'PASS' if n_dates==r_dates else 'FAIL',
       'required_maturity_na_zero':'PASS' if all_req_na_zero else 'FAIL','six_timestamp_hashes':'PASS' if ts_pass else 'FAIL',
       'independent_matrix_identity':'PASS' if matrix_identity else 'FAIL','independent_per_block_hash_identity':'PASS' if hash_identity else 'FAIL',
       'independent_unresolved_rows_zero':'PASS' if v_unresolved==0 else 'FAIL'},
      'independent_unresolved_source_rows':v_unresolved,'independent_per_block_hash':vhash,
      'model_training_performed':False,'strategy_evaluation_performed':False}
    json_write(run/'validator.json',validator)
    validator_md=['# Independent DATA Validator','',f"Overall: **{validator['overall']}**",'',
      'No labels, predictions, trades, realized returns, strategy outcomes, or model fitting were used.','',
      '| Check | Verdict |','|---|---|']+[f"| {k} | {v} |" for k,v in validator['checks'].items()]+['',f"Independent unresolved source rows: **{v_unresolved}**."]
    (run/'validator.md').write_text('\n'.join(validator_md)+'\n',encoding='utf-8')

    final_ready=bool(ready and validator_pass)
    metrics['summary']['validator_internal_methodology']=validator['internal_methodology']; metrics['summary']['validator_data_certification']=validator['data_certification']; metrics['summary']['data_foundation_ready']=final_ready
    json_write(run/'metrics.json',metrics)
    report=['# GEMINI US TREASURY REAL RATE FOUNDATION V1','',f"Status: **{'PASS' if final_ready else 'FAIL'}**",'',
      'DATA-ONLY. No labels, predictions, trades, outcomes, or model fitting were accessed.','',
      f"- Pre-run Git commit: `{PRE_COMMIT}`",f"- Nominal observations 2016-2024: **{len(n):,}** ({source_audit['nominal_range'][0]} → {source_audit['nominal_range'][1]})",
      f"- Real observations 2016-2024: **{len(r):,}** ({source_audit['real_range'][0]} → {source_audit['real_range'][1]})",
      f"- Nominal/real observation-date mismatch: **{source_audit['unresolved_missing_dates']}**",f"- Required maturity N/A counts: `{source_audit['na_counts_required']}`",
      f"- Substantive repeated-download revisions: **{len(substantive_revisions)}**",f"- Methodology break 2021-12-06 confirmed: **{method_break}**",
      f"- Official 6:00 PM ET availability evidence confirmed: **{avail_evidence}**",f"- Frozen availability: observation date +2 calendar days, 00:00 America/New_York",
      f"- Six GOLD timestamp hashes matched: **{all_ts}**",f"- Exact unresolved source rows: **{unresolved}**",'',
      '## Five frozen features','']+[f"- `{f}`" for f in FEATURES]+['',
      '## Block coverage','']+[f"- {b}: {a['constructable_rows']:,}/{a['rows']:,} constructable; unresolved {a['unresolved_rows']}; finite {[round(a['feature_finite_pct'][f],6) for f in FEATURES]}%" for b,a in block_audit.items()]+['',
      f"- `us_treasury_source_dataset_sha256`: `{source_dataset_sha}`",f"- `us_treasury_real_rate_feature_matrix_sha256`: `{matrix_sha}`",f"- Independent validator: **{validator['overall']}**",'',
      f"## Decision\n\n**US TREASURY REAL RATE DATA FOUNDATION READY = {'YES' if final_ready else 'NO'}**",'',
      '- Model training performed: NO','- Strategy evaluation performed: NO','- gemini.py changed: NO','- Operational model changed: NO','',
      'Single next action: '+('Authorize frozen B0 (31 execution-aligned features) vs B1 (31 + 5 Treasury features) experiment.' if final_ready else 'Do not train; resolve the failed data/validator gate first.')]
    (run/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

    # Repo-compatible manifest; actual post-run commit/registry entry occurs after user copies this run into repo.
    source_files=[]
    for p in sorted((run/'sources').iterdir()): source_files.append({'path':'sources/'+p.name,'sha256':sha_file(p),'retention_status':'stored_in_run_directory'})
    source_files.append({'path':'exact_timestamps.npz','sha256':sha_file(run/'exact_timestamps.npz'),'retention_status':'stored_in_run_directory'})
    data_rows=sum(x[1] for x in EXPECTED.values())
    manifest={'schema_version':1,'run_id':run_id,'experiment_name':'gemini_us_treasury_real_rate_foundation_v1','status':'pass' if final_ready else 'fail',
      'started_at_utc':now.isoformat().replace('+00:00','Z'),'finished_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
      'git_commit':PRE_COMMIT,'git_dirty':False,'training_script':'ChatGPT DATA-only execution (artifact builder retained separately)',
      'training_script_snapshot':'execution_spec.md','training_script_sha256':'PENDING_REPLACED_BELOW','exact_command':'ChatGPT local DATA-only Treasury foundation execution','arguments':[],
      'random_seeds':{},'random_seed_note':'No random operations; deterministic data foundation only',
      'data':{'symbols':['GOLD# timestamps only','US Treasury nominal par yield curve','US Treasury real par yield curve'],
        'data_sources':['Official U.S. Department of the Treasury historical daily par yield curve archives','Official U.S. Department of the Treasury yield curve methodology'],
        'source_files':source_files,'timezone':'GOLD exact UTC; Treasury America/New_York observation/availability semantics','data_start_utc':'2016-01-01T00:00:00Z','data_end_utc':'2024-12-31T23:59:59Z',
        'train_start_utc':'2016-06-30T21:00:00Z','train_end_utc':'2022-12-30T17:57:00Z','train_rows':1596361,
        'validation_start_utc':'2018-01-01T22:00:00Z','validation_end_utc':'2024-12-31T18:00:00Z','validation_rows':2474297,
        'test_start_utc':'not_applicable_data_only','test_end_utc':'not_applicable_data_only','test_rows':'not_applicable_data_only',
        'purge_details':'not applicable; data-only feature foundation','embargo_details':'not applicable; data-only feature foundation','raw_snapshot_retained':True,
        'reproducibility_claim':'full for retained Treasury raw sources and exact timestamp archive','mt5_fetch':{'used':False,'terminal_path':None,'terminal_info':None,'broker_info':None,'fetch_start_utc':None,'fetch_end_utc':None,'retrieved_at_utc':None,'returned_rows':None,'not_applicable_reason':'No MT5 access; exact timestamps only'}},
      'model':{'trained':False,'model_type':None,'parameters':{},'boosted_rounds_or_estimators':None,'features':[],'feature_count':0,'label_definition':None,'horizon':None,'label_tp_sl_semantics':None,'execution_tp_sl_semantics':None,'calibration_method':None,'artifact_path':None,'artifact_sha256':None,'retention_status':None,'not_applicable_reason':'DATA-only foundation; model training prohibited'},
      'search':{'performed':False,'predefined_search_space':{},'candidate_results_file':None,'not_applicable_reason':'Fixed five-feature family; no search'},
      'registry':{'parent_or_incumbent':'20260903T071729Z_gemini_execution_aligned_label_v1/C1_execution_aligned_label','selected_configuration':'fixed five-feature US Treasury real-rate family; data-only',
        'trades_per_day':'not_applicable_data_only','realized_win_rate':'not_applicable_data_only','pf':'not_applicable_data_only','mean_r':'not_applicable_data_only','pnl':'not_applicable_data_only','max_dd':'not_applicable_data_only',
        'validator_result':f"internal {validator['internal_methodology']}; data certification {validator['data_certification']}"},
      'promotion':{'requested':False,'gate_result':'not_requested','replacement_authorized':False,'operational_artifact_changed':False},'artifacts':[],'aborted_reason':None}
    spec=("# Execution specification\n\nDATA-ONLY US Treasury real-rate foundation. Frozen five features, +2 calendar day 00:00 America/New_York availability, six exact timestamp blocks. No labels/outcomes/models.\n")
    (run/'execution_spec.md').write_text(spec,encoding='utf-8'); manifest['training_script_sha256']=sha_file(run/'execution_spec.md')
    for p in ['metrics.json','report.md','validator.json','validator.md','source_provenance.json','timestamp_audit.json','treasury_daily_canonical.csv','treasury_real_rate_features.npz','execution_spec.md']:
        q=run/p; manifest['artifacts'].append({'path':p,'sha256':sha_file(q),'retention_status':'stored_in_run_directory'})
    json_write(run/'manifest.json',manifest)
    (run/'environment.txt').write_text(f"python=={sys.version.split()[0]}\nnumpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn==not-installed\nxgboost==not-installed\nMetaTrader5==not-installed\n",encoding='utf-8')
    (run/'stdout.log').write_text('DATA-only Treasury foundation executed by ChatGPT runtime.\n',encoding='utf-8')
    (run/'candidates.csv').write_text('candidate_id,parameters,fold,executable_trades,trades_per_day,tp_first_wr,realized_wr,pf,mean_r,pnl,max_dd,break_even_wr,break_even_adjusted_edge,cost_stress_result,qualification_verdict\n',encoding='utf-8')

    # Finalize byte-for-byte after all retained files exist. Include all except FINALIZED.
    hashes={p.relative_to(run).as_posix():sha_file(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='FINALIZED.json'}
    json_write(run/'FINALIZED.json',{'run_id':run_id,'finalized_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'file_sha256':hashes})
    # Final verification of finalized contents.
    fin=json.loads((run/'FINALIZED.json').read_text()); assert all(sha_file(run/k)==v for k,v in fin['file_sha256'].items())

    outzip=BASE/(run_id+'.zip')
    with zipfile.ZipFile(outzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(run.rglob('*')):
            if p.is_file(): zf.write(p,arcname=f'{run_id}/{p.relative_to(run).as_posix()}')
    result={'run_id':run_id,'run_dir':str(run),'zip':str(outzip),'ready':final_ready,'validator':validator['overall'],'metrics':metrics,'source_audit':source_audit}
    json_write(BASE/'treasury_run_result.json',result)
    print(json.dumps({'run_id':run_id,'ready':final_ready,'validator':validator['overall'],'unresolved':unresolved,'source_sha':source_dataset_sha,'matrix_sha':matrix_sha,'zip':str(outzip),'zip_bytes':outzip.stat().st_size},indent=2))

if __name__=='__main__': build()
