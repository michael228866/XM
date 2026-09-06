"""Data-only canonical release identity and exact eight-feature integration audit.

No model, price feature, label, prediction or outcome is loaded. Review records
are official-source annotations, not experimental choices. Existing archives
are read-only. acquire -> review -> build -> separate validator -> finalize.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import gold_fomc_canonical_foundation_v1 as io
import training_run_history as archive

ROOT = Path(__file__).resolve().parent
OLD = ROOT/'training_runs/20260903T154042Z_gemini_macro_event_timing_v1'
LABEL = ROOT/'training_runs/20260903T071729Z_gemini_execution_aligned_label_v1'
FOMC = [ROOT/'training_runs'/n for n in (
    '20260905T163617Z_gemini_fomc_2016_2017_foundation_v1',
    '20260905T153912Z_gemini_fomc_canonical_foundation_v1')]
FEATURES = ['EVENT_MINUTES_SINCE', 'EVENT_POST_0_15', 'EVENT_POST_15_60',
            'EVENT_POST_60_240', 'EVENT_TYPE_CPI', 'EVENT_TYPE_EMPLOYMENT',
            'EVENT_TYPE_PCE', 'EVENT_TYPE_FOMC']
FAMILIES = ['CPI', 'EMPLOYMENT', 'PCE', 'FOMC']
MINUTE = 60_000_000_000
CPI_URL = 'https://www.bls.gov/news.release/archives/cpi_06162016.htm'
MONTHS = 'January February March April May June July August September October November December'.split()
DATE_RE = r'('+'|'.join(MONTHS)+r')\s+(\d{1,2}),?\s+(20\d{2})'
TIME_RE = r'(\d{1,2}):(\d{2})\s*([ap])\.?m\.?\s*\(?\s*(EST|EDT|ET)\b'


def array_hash(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str(values.dtype).encode()+np.asarray(values.shape, dtype=np.int64).tobytes()+values.tobytes()).hexdigest()


def fetch(run, url):
    """Retain failures, including HTTP 200 empty bodies; never silently drop."""
    assert urllib.parse.urlparse(url).hostname in ('www.bea.gov', 'www.bls.gov')
    target = run/'sources'/(hashlib.sha256(url.encode()).hexdigest()[:24]+'.html')
    row = dict(url=url, path=target.relative_to(run).as_posix(), acquisition_utc=io.stamp(),
               status='error', http_status=0, bytes=0, sha256='', final_url='', error='')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (historical release provenance research)'})
        with urllib.request.urlopen(request, timeout=45) as response:
            row.update(http_status=response.status, final_url=response.url)
            assert urllib.parse.urlparse(response.url).hostname in ('www.bea.gov', 'www.bls.gov')
            payload = response.read()
        target.write_bytes(payload)
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), status='ok' if payload else 'empty_body')
    except Exception as exc:
        row['error'] = str(exc)
    return row


def source_text(run, source):
    return (run/source['path']).read_text(encoding='utf-8', errors='replace') if source['status']=='ok' else ''


def release_annotation(run, source):
    doc = source_text(run, source)
    text = io.text_of(doc)
    titles = re.findall(r'<h1\b[^>]*>(.*?)</h1>', doc, re.S|re.I)
    title = io.text_of(titles[-1]) if titles else io.title_of(doc)
    kind = 'unresolved'
    if re.search(r'Personal Income and Outlays', title, re.I) and re.search(r'personal consumption expenditures', text, re.I):
        kind = 'national_pce'
    elif re.search(r'\b(state|regional)\b', title, re.I):
        kind = 'state_or_regional_excluded'
    elif title:
        kind = 'other_noncanonical_excluded'
    snippet, utc, zone, local = '', '', '', ''
    for anchor in re.finditer(r'embargoed until|for release', text, re.I):
        value = text[anchor.start():anchor.start()+450]
        d, t = re.search(DATE_RE, value, re.I), re.search(TIME_RE, value, re.I)
        if not d or not t:
            continue
        month = next(i+1 for i, m in enumerate(MONTHS) if m.lower()==d[1].lower())
        hour = int(t[1]) % 12 + (12 if t[3].lower()=='p' else 0)
        dt = datetime(int(d[3]), month, int(d[2]), hour, int(t[2]), tzinfo=ZoneInfo('America/New_York'))
        zone = t[4].upper()
        if zone != 'ET' and zone != dt.tzname():
            continue
        local, utc, snippet = dt.isoformat(), dt.astimezone(timezone.utc).isoformat(), value
        break
    return dict(url=source['url'], classification=kind, official_title=title,
                reference_periods='', release_timestamp_utc=utc, release_local=local,
                original_timezone=zone, release_evidence=snippet, identity_evidence=title,
                source_sha256=source['sha256'], acquisition_utc=source['acquisition_utc'],
                source_path=source['path'], document_id=source['url'].rstrip('/').split('/')[-1],
                reviewed='no', review_notes='')


def acquire(run):
    m = archive.read_json(run/'manifest.json')
    assert not m['git_dirty']
    assert io.git('rev-parse','HEAD') == io.git('rev-parse','origin/main') == m['git_commit']
    assert archive.file_sha256(Path(__file__)) == m['training_script_sha256']
    m['pre_run_git'] = dict(pre_run_git_commit=m['git_commit'], pre_run_git_dirty=False,
                           head_sha=m['git_commit'], origin_main_sha=m['git_commit'])
    m['protected_runs_before'] = {p.parent.name:io.inventory(p.parent) for p in sorted((ROOT/'training_runs').glob('*/FINALIZED.json'))}
    m['operational_hashes_before'] = {n:archive.file_sha256(ROOT/n) for n in ('gemini.py','gold_long_recent_candidate_xgb.json')}
    m['dependencies'] = {n:archive.file_sha256(ROOT/n) for n in ('training_run_history.py','gold_fomc_canonical_foundation_v1.py','gold_macro_event_integration_foundation_v1_validator.py')}
    archive.write_json(run/'manifest.json',m)
    shutil.copyfile(ROOT/'gold_macro_event_integration_foundation_v1_validator.py',run/'validator_script.py')
    for p, name in [(OLD/'event_timestamp_dataset.csv','inherited_events.csv'),
                    (OLD/'event_source_provenance.csv','inherited_sources.csv'),
                    (LABEL/'fold_model_provenance.json','timestamp_reference.json')]:
        # JSON contains provenance hashes/config only; no outcome or prediction arrays.
        shutil.copyfile(p,run/name)
    (run/'sources').mkdir()
    root = fetch(run,'https://www.bea.gov/sitemap.xml')
    sources = [root]
    assert root['status']=='ok', 'Cannot enumerate official BEA universe'
    links = re.findall(r'<loc>(.*?)</loc>', source_text(run,root))
    index_urls = sorted(set(html.unescape(u) for u in links if 'sitemap' in u))
    assert 0 < len(index_urls) <= 30, 'Review changed BEA sitemap topology'
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources.extend(pool.map(lambda u:fetch(run,u),index_urls))
    io.write_csv(run/'source_provenance.csv',sources)
    assert all(s['status']=='ok' for s in sources), 'Incomplete official index acquisition'
    discovered = sorted({html.unescape(u) for s in sources for u in re.findall(r'<loc>(.*?)</loc>',source_text(run,s))
                         if re.search(r'/news/20(?:1[6-9]|2[0-4])/personal-income-and-outlays',u)})
    inherited = io.read_csv(run/'inherited_events.csv')
    urls = sorted(set(discovered) | {r['source_document'] for r in inherited if r['event_type']=='PCE'})
    io.write_csv(run/'official_pce_universe.csv',[dict(url=u, basis='official_BEА_sitemap_national_release_series') for u in discovered],['url','basis'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, source in enumerate(pool.map(lambda u:fetch(run,u),urls+[CPI_URL]),1):
            sources.append(source)
            io.write_csv(run/'source_provenance.csv',sources)
            if i%10==0:
                print(f'Official releases retained {i}/{len(urls)+1}',flush=True)
    reviews = [release_annotation(run,s) for s in sources if s['url'] in urls]
    io.write_csv(run/'pce_review.csv', reviews)
    cpi = release_annotation(run,sources[-1])
    archive.write_json(run/'cpi_recovery.json',cpi)
    print('ACQUIRED: official-source annotations must be reviewed before build',flush=True)


def exact_timestamps(run):
    folds = archive.read_json(run/'timestamp_reference.json')['folds']
    source = next(r for r in archive.read_json(LABEL/'manifest.json')['data']['source_files']
                  if Path(r['path']).name.startswith('GOLD#_M1_'))
    assert archive.file_sha256(Path(source['path']))==source['sha256']
    parts = []
    start, end = min(f['train_start'] for f in folds), max(f['score_end'] for f in folds)
    for frame in pd.read_csv(source['path'],sep='\t',usecols=lambda c:c.strip('<>').upper() in ('DATE','TIME'),chunksize=200000):
        frame.columns = [c.strip('<>').upper() for c in frame.columns]
        times = pd.to_datetime(frame['DATE']+' '+frame['TIME'])
        selected = times[(times>=start)&(times<=end)]
        if len(selected):
            parts.append(selected)
    all_times = pd.DatetimeIndex(pd.concat(parts).sort_values()).as_unit('ns')
    blocks, arrays = [], {}
    for i, fold in enumerate(folds,1):
        for kind in ('train','score'):
            first, last = fold[kind+'_start'],fold['train_feature_end' if kind=='train' else 'score_end']
            dt = all_times[(all_times>=first)&(all_times<=last)]
            raw = dt.asi8
            digest = array_hash(raw)
            assert len(raw)==fold[kind+'_rows'] and digest==fold[kind+'_timestamp_sha256'], (i,kind,'exact timestamp mismatch')
            utc = dt.tz_localize('Europe/Helsinki', ambiguous='raise', nonexistent='raise').tz_convert('UTC').asi8
            key = f'fold{i}_{kind}'
            arrays[key+'_broker_ns'],arrays[key+'_utc_ns'] = raw,utc
            blocks.append(dict(block=key,rows=len(raw),first_broker=str(dt[0]),last_broker=str(dt[-1]),
                               first_utc=pd.Timestamp(utc[0],tz='UTC').isoformat(),last_utc=pd.Timestamp(utc[-1],tz='UTC').isoformat(),
                               timestamp_sha256=digest,utc_timestamp_sha256=array_hash(utc)))
    arrays['unique_utc_ns'] = np.unique(np.concatenate([v for k,v in arrays.items() if k.endswith('_utc_ns')]))
    np.savez_compressed(run/'exact_timestamps.npz',**arrays)
    archive.write_json(run/'timestamp_universe.json',dict(blocks=blocks,unique_rows=len(arrays['unique_utc_ns']),source=source,
        timezone='Inherited audited XM EET/EEST broker-wall -> Europe/Helsinki -> UTC; no new calibration',
        raw_columns_loaded=['DATE','TIME'], timestamp_archive_sha256=archive.file_sha256(run/'exact_timestamps.npz')))
    return blocks, arrays


def feature_matrix(times, events):
    # Frozen prior implementation: at simultaneous releases alphabetical last
    # event_type wins. Retain all canonical records in the source dataset.
    events = sorted(events,key=lambda r:(pd.Timestamp(r['release_timestamp_utc']).value,r['event_type']))
    releases = np.array([pd.Timestamp(r['release_timestamp_utc']).value for r in events],dtype=np.int64)
    types = np.array([FAMILIES.index(r['event_type']) for r in events],dtype=np.int8)
    result = np.zeros((len(times),8),dtype=np.float64)
    result[:,0] = 1
    if not len(releases):
        return result
    ix = np.searchsorted(releases,times,side='right')-1
    valid = ix>=0
    age = np.full(len(times),np.inf)
    age[valid] = (times[valid]-releases[ix[valid]])/MINUTE
    assert np.all(age[valid]>=0)
    result[:,0] = np.minimum(age,1440)/1440
    result[:,1] = age<15
    result[:,2] = (age>=15)&(age<60)
    result[:,3] = (age>=60)&(age<240)
    for i in range(4):
        result[:,4+i] = valid&(age<240)&(types[np.maximum(ix,0)]==i)
    return result


def delta_counts(a,b,indices=None):
    changed = a!=b
    if indices is not None:
        changed = changed[indices]
    return dict(minutes_since_changed=int(changed[:,0].sum()),
                post_window_changed=int(changed[:,1:4].any(axis=1).sum()),
                pce_flag_changed=int(changed[:,6].sum()),total_distinct_rows_changed=int(changed.any(axis=1).sum()))


def build(run):
    reviews = io.read_csv(run/'pce_review.csv')
    assert all(r['reviewed']=='yes' for r in reviews), 'Complete official identity/reference/time review first'
    national = [r for r in reviews if r['classification']=='national_pce']
    assert all(r['release_timestamp_utc'] and r['reference_periods'] for r in national)
    assert len({r['release_timestamp_utc'] for r in national})==len(national), 'Duplicate representations require explicit classification'
    sequence = archive.read_json(run/'pce_sequence_review.json')
    assert sequence['basis']=='official_national_release_reference_period_sequence'
    blocks, arrays = exact_timestamps(run)
    times = arrays['unique_utc_ns']
    old = io.read_csv(run/'inherited_events.csv')
    old_sources = {r['requested_url']:r for r in io.read_csv(run/'inherited_sources.csv')}
    events = []
    for r in old:
        if r['event_type'] in ('CPI','EMPLOYMENT'):
            s = old_sources[r['source_document']]
            assert int(s['bytes'])>0 and s['content_sha256'], 'Additional inherited provenance defect requires exact review'
            events.append(dict(event_id=r['event_type']+'_'+r['source_release_identifier'],event_type=r['event_type'],
                release_timestamp_utc=r['release_timestamp_utc'],source_document=r['source_document'],source_sha256=s['content_sha256'],
                provenance_run=OLD.name,reference_period='',official_title='',original_timezone=r['original_timezone']))
    recovery = archive.read_json(run/'cpi_recovery.json')
    recovered = bool(recovery['release_timestamp_utc'] and recovery['source_sha256'] and recovery['release_evidence'])
    if recovered:
        events.append(dict(event_id='CPI_cpi_06162016',event_type='CPI',release_timestamp_utc=recovery['release_timestamp_utc'],
             source_document=CPI_URL,source_sha256=recovery['source_sha256'],provenance_run=run.name,
             reference_period='2016-05',official_title=recovery['official_title'],original_timezone=recovery['original_timezone']))
    for r in national:
        events.append(dict(event_id='PCE_'+r['document_id'],event_type='PCE',release_timestamp_utc=r['release_timestamp_utc'],
            source_document=r['url'],source_sha256=r['source_sha256'],provenance_run=run.name,
            reference_period=r['reference_periods'],official_title=r['official_title'],original_timezone=r['original_timezone']))
    for foundation in FOMC:
        for r in io.read_csv(foundation/'fomc_verified_events.csv'):
            events.append(dict(event_id=r['event_id'],event_type='FOMC',release_timestamp_utc=r['release_timestamp_utc'],
                source_document=r['official_statement_url'],source_sha256=r['content_sha256'],provenance_run=foundation.name,
                reference_period='',official_title=r['official_statement_title'],original_timezone=r['original_timezone']))
    events = sorted(events,key=lambda r:(pd.Timestamp(r['release_timestamp_utc']).value,r['event_type'],r['event_id']))
    assert len({r['event_id'] for r in events})==len(events)
    io.write_csv(run/'canonical_macro_events.csv',events)
    corrected = feature_matrix(times,events)
    inherited_pce = [dict(event_type='PCE',release_timestamp_utc=r['release_timestamp_utc'],source_document=r['source_document']) for r in old if r['event_type']=='PCE']
    fixed = [r for r in events if r['event_type']!='PCE']
    inherited_matrix = feature_matrix(times,fixed+inherited_pce)
    impacts = [dict(block='unique_union',**delta_counts(inherited_matrix,corrected))]
    for block in blocks:
        impacts.append(dict(block=block['block'],**delta_counts(inherited_matrix,corrected,np.searchsorted(times,arrays[block['block']+'_utc_ns']))))
    io.write_csv(run/'pce_exact_feature_impact.csv',impacts)
    excluded = [r for r in reviews if r['classification'] not in ('national_pce','unresolved')]
    individual = []
    for r in excluded:
        if not any(e['source_document']==r['url'] for e in inherited_pce):
            continue
        without = feature_matrix(times,fixed+[e for e in inherited_pce if e['source_document']!=r['url']])
        individual.append(dict(url=r['url'],official_title=r['official_title'],classification=r['classification'],
                               **delta_counts(inherited_matrix,without)))
    io.write_csv(run/'excluded_pce_event_impact.csv',individual,['url','official_title','classification','minutes_since_changed','post_window_changed','pce_flag_changed','total_distinct_rows_changed'])
    # Conservative envelope includes every June 16 US-local timestamp and more.
    # No pre-event features and all effects expire after 1440 minutes.
    lo,hi = pd.Timestamp('2016-06-16T00:00:00Z').value,pd.Timestamp('2016-06-18T00:00:00Z').value
    cpi_mask = (times>=lo)&(times<hi+1440*MINUTE)
    assert not cpi_mask.any(), 'CPI equivalence requires a more detailed proof'
    for t in (lo,hi):
        alternative = fixed+[r for r in events if r['event_type']=='PCE']+[dict(event_type='CPI',release_timestamp_utc=pd.Timestamp(t,tz='UTC').isoformat())]
        assert np.array_equal(feature_matrix(times,alternative),corrected)
    cpi_proof = dict(official_timestamp_recovered=recovered,source_retention_complete=recovered,
        cpi_20160616_exact_feature_rows_affected=int(cpi_mask.sum()),feature_impact_neutral=True,
        classification='proven_feature_impact_neutral',possible_release_utc_envelope=[str(pd.Timestamp(lo,tz='UTC')),str(pd.Timestamp(hi,tz='UTC'))],
        maximum_effect_end_utc=str(pd.Timestamp(hi+1440*MINUTE,tz='UTC')),earliest_required_row_utc=str(pd.Timestamp(times[0],tz='UTC')),
        proof='For every possible release in the conservative envelope, every required row is >1440 minutes later; clipped age=1 and all flags=0. A later verified event supersedes it. No pre-event features. Exact intersection empty; endpoint insertion regression identical.',
        inherited_zero_byte_defect_remains=True,original_source_retention_complete=False)
    archive.write_json(run/'cpi_feature_impact_proof.json',cpi_proof)
    # Missing canonical events must be explicitly bounded by source-reviewed
    # possible public-release intervals. Exact feature ambiguity is unioned per row.
    unresolved = sequence.get('unresolved_release_intervals',[])
    unknown = np.zeros(len(times),dtype=bool)
    for r in unresolved:
        low,high = pd.Timestamp(r['earliest_possible_release_utc']).value,pd.Timestamp(r['latest_possible_release_utc']).value
        # Exact ambiguity: known latest event before decision suppresses the
        # unknown iff it is strictly later than every possible unknown timestamp.
        known_ns = np.sort(np.array([pd.Timestamp(e['release_timestamp_utc']).value for e in events],dtype=np.int64))
        ix = np.searchsorted(known_ns,times,side='right')-1
        latest = np.where(ix>=0,known_ns[np.maximum(ix,0)],np.iinfo(np.int64).min)
        unknown |= (times>=low)&(times<high+1440*MINUTE)&(latest<=high)
    ambiguous = [r for r in reviews if r['classification']=='unresolved']
    if ambiguous and not unresolved:
        raise ValueError('Unbounded unresolved source identities cannot be reported as zero affected rows')
    np.savez_compressed(run/'event_features.npz',utc_ns=times,features=corrected,feature_names=np.asarray(FEATURES),incomplete_mask=unknown)
    readiness=[]
    for block in blocks:
        ix = np.searchsorted(times,arrays[block['block']+'_utc_ns'])
        block['constructable_rows'] = int((~unknown[ix]).sum())
        for family in FAMILIES:
            affected = int(unknown[ix].sum()) if family=='PCE' else 0
            readiness.append(dict(block=block['block'],family=family,canonical_stream_sufficient=affected==0,
                unresolved_events=len(unresolved) if family=='PCE' else (int(not recovered) if family=='CPI' else 0),exact_rows_affected=affected))
    io.write_csv(run/'family_block_readiness.csv',readiness)
    io.write_csv(run/'exact_block_constructability.csv',blocks)
    metrics = dict(run_id=run.name,run_status='pending_independent_validator',
        pce=dict(inherited_rows=len(inherited_pce),canonical_national_releases=len(national),
            canonical_national_in_required_history=sum(pd.Timestamp(r['release_timestamp_utc']).value>=times[0]-1440*MINUTE and pd.Timestamp(r['release_timestamp_utc']).value<=times[-1] for r in national),
            state_level_excluded=sum(r['classification']=='state_or_regional_excluded' for r in excluded),
            other_noncanonical_excluded=sum(r['classification']!='state_or_regional_excluded' for r in excluded),
            missing_canonical_national_releases=sequence['missing_canonical_national_releases'],ambiguous_canonical_releases=len(ambiguous),
            exact_affected_rows=impacts[0]['total_distinct_rows_changed'],history_sufficient=not unknown.any() and sequence['sequence_complete']),
        cpi=cpi_proof,employment_history_sufficient=True,fomc_history_sufficient=True,blocks=blocks,
        macro_event_dataset_sha256=archive.file_sha256(run/'canonical_macro_events.csv'),
        event_feature_matrix_sha256=array_hash(corrected),event_feature_archive_sha256=archive.file_sha256(run/'event_features.npz'),
        unique_required_rows=len(times),incomplete_event_history_rows_exact=int(unknown.sum()),causal_alignment='PASS',
        data_foundation_ready=False,model_training_performed=False,strategy_evaluation_performed=False,
        gemini_py_changed=False,operational_model_changed=False,prior_runs_unchanged=True,
        tie_policy='Alphabetical-last family on simultaneous release; frozen prior feature convention',
        next_action='Wait for explicit B0/B1 authorization only if independent data validator passes; otherwise resolve documented data defects without training.')
    archive.write_json(run/'metrics.json',metrics)
    m = archive.read_json(run/'manifest.json')
    assert all(io.inventory(ROOT/'training_runs'/n)==v for n,v in m['protected_runs_before'].items())
    assert all(archive.file_sha256(ROOT/n)==v for n,v in m['operational_hashes_before'].items())
    sources = io.read_csv(run/'source_provenance.csv')
    m['data'].update(symbols=['GOLD# timestamps only','CPI','EMPLOYMENT','national PCE','FOMC'],
        data_sources=['official BEA/BLS release documents','immutable retained canonical CPI/EMPLOYMENT/FOMC'],
        source_files=[dict(path=s['path'],sha256=s['sha256'],retention_status='raw bytes retained in this Git run') for s in sources if s['sha256']],
        timezone='America/New_York release -> UTC; inherited XM Europe/Helsinki broker-wall -> UTC',
        data_start_utc=str(pd.Timestamp(times[0]-1440*MINUTE,tz='UTC')),data_end_utc=str(pd.Timestamp(times[-1],tz='UTC')),
        train_start_utc='not_applicable_data_only',train_end_utc='not_applicable_data_only',train_rows=0,
        validation_start_utc=str(pd.Timestamp(times[0],tz='UTC')),validation_end_utc=str(pd.Timestamp(times[-1],tz='UTC')),validation_rows=len(times),
        test_start_utc='not_applicable_data_only',test_end_utc='not_applicable_data_only',test_rows=0,
        purge_details='no labels or fitting; frozen timestamp blocks only',embargo_details='no model evaluation',raw_snapshot_retained=True,
        reproducibility_claim='Exact timestamp-only cohorts, macro matrix and BEA/BLS bytes retained in Git. Inherited CPI/EMPLOYMENT are previously certified timestamp/excerpt+hash records, not complete raw snapshots. CPI defect separately proven neutral.',
        mt5_fetch=dict(used=False,not_applicable_reason='No MT5 connection'))
    m['model']['not_applicable_reason']='data-only; no model or labels loaded'
    m['search']['not_applicable_reason']='fixed eight-feature certification; zero candidate selection'
    m['registry'].update({k:'not_applicable_data_only' for k in archive.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent=OLD.name,selected_configuration='Canonical national PCE + exact all-fold eight-feature certification',validator_result='PENDING')
    m['promotion'].update(gemini_py_changed=False,operational_model_changed=False,operational_artifact_changed=False)
    archive.write_json(run/'manifest.json',m)
    (run/'report.md').write_text('# GEMINI MACRO EVENT INTEGRATION FOUNDATION V1\n\nData-only; no training, no strategy evaluation.\n\n'+(run/'source_review_notes.md').read_text(encoding='utf-8')+'\n\n## Results\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


def self_check():
    t = pd.Timestamp('2020-01-01T00:00:00Z').value
    offsets=np.array([-1,0,14,15,59,60,239,240,1439,1440,1441])
    a=feature_matrix(t+offsets*MINUTE,[dict(event_type='PCE',release_timestamp_utc='2020-01-01T00:00:00Z')])
    assert a[0,0]==1 and a[0,1:].sum()==0
    assert a[:,1].tolist()==[0,1,1,0,0,0,0,0,0,0,0]
    assert a[:,2].tolist()==[0,0,0,1,1,0,0,0,0,0,0]
    assert a[:,3].tolist()==[0,0,0,0,0,1,1,0,0,0,0]
    assert a[7,6]==0 and a[-1,0]==1
    assert np.array_equal(feature_matrix(t+offsets*MINUTE,[]),np.column_stack([np.ones(11),np.zeros((11,7))]))
    print('SELF_CHECK_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['acquire','build','self-check'])
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.action=='self-check':
        self_check()
    else:
        run=args.run_dir.resolve()
        assert run.parent==ROOT/'training_runs' and not (run/'FINALIZED.json').exists()
        {'acquire':acquire,'build':build}[args.action](run)
