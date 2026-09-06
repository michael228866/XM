"""Independent data validator: does not import the experiment or model modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import gold_fomc_canonical_foundation_v1 as io
import training_run_history as archive

ROOT=Path(__file__).resolve().parent


def digest(a):
    return hashlib.sha256(str(a.dtype).encode()+np.array(a.shape,dtype='int64').tobytes()+a.tobytes()).hexdigest()


def independent_features(times,events):
    # Independent pandas backward as-of join, not experiment's numpy indexing.
    right=pd.DataFrame(events)[['release_timestamp_utc','event_type']].copy()
    right['release']=pd.to_datetime(right.release_timestamp_utc,utc=True).astype('int64')
    right=right.sort_values(['release','event_type']).drop_duplicates('release',keep='last')
    left=pd.DataFrame({'decision':times})
    joined=pd.merge_asof(left,right,left_on='decision',right_on='release',direction='backward',allow_exact_matches=True)
    assert ((joined.release<=joined.decision)|joined.release.isna()).all()
    age=(joined.decision-joined.release)/60_000_000_000
    out=np.zeros((len(times),8),dtype='float64')
    out[:,0]=age.clip(upper=1440).fillna(1440)/1440
    for col,(low,high) in enumerate([(0,15),(15,60),(60,240)],1):
        out[:,col]=(age>=low)&(age<high)
    for col,family in enumerate(['CPI','EMPLOYMENT','PCE','FOMC'],4):
        out[:,col]=(age>=0)&(age<240)&(joined.event_type==family)
    return out


def validate(run):
    checks=[]
    def check(name,condition,evidence):
        checks.append(dict(check=name,verdict='PASS' if condition else 'FAIL',evidence=evidence))
    m=archive.read_json(run/'manifest.json')
    metrics=archive.read_json(run/'metrics.json')
    reviews=io.read_csv(run/'pce_review.csv')
    events=io.read_csv(run/'canonical_macro_events.csv')
    universe=archive.read_json(run/'timestamp_universe.json')
    refs=archive.read_json(run/'timestamp_reference.json')['folds']
    sequence=archive.read_json(run/'pce_sequence_review.json')
    sources={s['url']:s for s in io.read_csv(run/'source_provenance.csv')}
    canonical=[r for r in reviews if r['classification']=='national_pce']
    source_errors=[]
    for r in reviews:
        s=sources[r['url']]
        if s['status']!='ok':
            source_errors.append(r['url']+': unavailable identity');continue
        path=run/s['path']
        text=io.text_of(path.read_text(encoding='utf-8',errors='replace'))
        if archive.file_sha256(path)!=r['source_sha256'] or r['official_title'] not in text:
            source_errors.append(r['url']+': hash/title mismatch')
        if r['classification']=='national_pce':
            if not re.search('Personal Income and Outlays',r['official_title'],re.I) or re.search(r'\b(state|regional|industry|GDP)\b',r['official_title'],re.I):
                source_errors.append(r['url']+': not national release identity')
            if 'personal consumption expenditures' not in text.lower() or not r['reference_periods'] or r['release_evidence'] not in text:
                source_errors.append(r['url']+': missing content/reference/time evidence')
            utc=pd.Timestamp(r['release_timestamp_utc'])
            local=pd.Timestamp(r['release_local'])
            if utc!=local or utc!=local.tz_convert('America/New_York'):
                source_errors.append(r['url']+': timezone mismatch')
        if r['reviewed']!='yes':
            source_errors.append(r['url']+': unreviewed')
    check('national PCE identity and retained official source bytes',not source_errors,source_errors or f'{len(canonical)} canonical releases verified')
    discovered={r['url'] for r in io.read_csv(run/'official_pce_universe.csv')}
    reviewed={r['url'] for r in reviews}
    check('official universe and reference-period sequence, not release-month presence',
          discovered<=reviewed and sequence['basis']=='official_national_release_reference_period_sequence' and sequence['sequence_complete']
          and not sequence['missing_canonical_national_releases'] and bool(sequence['reference_period_evidence']),
          'Official sitemap universe exhaustively identity reviewed; reference periods and release timestamps separately retained; see pce_sequence_review.json')
    excluded={r['url'] for r in reviews if r['classification']!='national_pce'}
    check('state/regional PCE exclusion',not any(e['event_type']=='PCE' and e['source_document'] in excluded for e in events)
          and all(any(fragment in r['url'] and r['classification']=='state_or_regional_excluded' for r in reviews)
                  for fragment in ['personal-consumption-expenditures-state-1997-2014','personal-consumption-expenditures-state-2015','personal-consumption-expenditures-state-2016']),
          '2015, 2016 and 2017 state releases classified by official title; not excluded by date alone')
    loaded=np.load(run/'exact_timestamps.npz',allow_pickle=False)
    exact=True
    for i,fold in enumerate(refs,1):
        for kind in ['train','score']:
            raw=loaded[f'fold{i}_{kind}_broker_ns']
            utc=loaded[f'fold{i}_{kind}_utc_ns']
            exact &= len(raw)==fold[kind+'_rows'] and digest(raw)==fold[kind+'_timestamp_sha256']
            converted=pd.to_datetime(raw).tz_localize('Europe/Helsinki',ambiguous='raise',nonexistent='raise').tz_convert('UTC').asi8
            exact &= np.array_equal(utc,converted)
    times=loaded['unique_utc_ns']
    exact &= np.array_equal(times,np.unique(np.concatenate([loaded[f'fold{i}_{kind}_utc_ns'] for i in [1,2,3] for kind in ['train','score']])))
    check('six exact retained timestamp hashes and UTC conversion',bool(exact),'Compared all six to immutable C1 provenance hashes; exact deduplicated union')
    matrix=np.load(run/'event_features.npz',allow_pickle=False)
    independently=independent_features(times,events)
    check('eight-feature causal as-of construction',np.array_equal(times,matrix['utc_ns']) and np.array_equal(independently,matrix['features']),
          f'Independent backward as-of join: all {len(times)} rows; release <= decision; 15/60/240/1440 fixed boundaries')
    check('combined dataset and feature matrix hashes',archive.file_sha256(run/'canonical_macro_events.csv')==metrics['macro_event_dataset_sha256']
          and digest(matrix['features'])==metrics['event_feature_matrix_sha256'] and archive.file_sha256(run/'event_features.npz')==metrics['event_feature_archive_sha256'],
          'CSV bytes, logical matrix dtype/shape/bytes and NPZ bytes independently SHA-256 checked')
    proof=archive.read_json(run/'cpi_feature_impact_proof.json')
    endpoint=pd.Timestamp(proof['maximum_effect_end_utc']).value
    check('CPI retention-gap mathematical equivalence',times.min()>endpoint and proof['cpi_20160616_exact_feature_rows_affected']==0
          and proof['original_source_retention_complete'] is False,
          'Whole conservative release envelope expires before earliest required timestamp; no pre-event features; original source still incomplete')
    old=io.read_csv(run/'inherited_events.csv')
    inherited=[e for e in events if e['event_type']!='PCE']+[dict(event_type='PCE',release_timestamp_utc=r['release_timestamp_utc']) for r in old if r['event_type']=='PCE']
    original=independent_features(times,inherited)
    diff=original!=independently
    impacts=io.read_csv(run/'pce_exact_feature_impact.csv')
    correct=True
    for row in impacts:
        d=diff if row['block']=='unique_union' else diff[np.searchsorted(times,loaded[row['block']+'_utc_ns'])]
        correct &= [int(d[:,0].sum()),int(d[:,1:4].any(axis=1).sum()),int(d[:,6].sum()),int(d.any(axis=1).sum())]==[int(row[k]) for k in ['minutes_since_changed','post_window_changed','pce_flag_changed','total_distinct_rows_changed']]
    check('exact PCE changed-row accounting, not whole-stream withholding',bool(correct),'Independently compared all eight features on exact timestamps; per-block and deduplicated union')
    check('unique constructability and explicit unresolved-event accounting',not matrix['incomplete_mask'].any()
          and metrics['incomplete_event_history_rows_exact']==0 and not sequence.get('unresolved_release_intervals')
          and all(int(b['constructable_rows'])==int(b['rows']) for b in metrics['blocks']),
          'No blanket 530218 withholding count; CPI known neutral, canonical PCE sequence complete')
    fomc=[e for e in events if e['event_type']=='FOMC']
    inherited_fomc=[]
    for name in ['20260905T163617Z_gemini_fomc_2016_2017_foundation_v1','20260905T153912Z_gemini_fomc_canonical_foundation_v1']:
        inherited_fomc.extend(io.read_csv(ROOT/'training_runs'/name/'fomc_verified_events.csv'))
    check('FOMC immutable reuse including unscheduled statements',
          {(e['event_id'],e['release_timestamp_utc'],e['source_sha256']) for e in fomc}=={(e['event_id'],e['release_timestamp_utc'],e['content_sha256']) for e in inherited_fomc},
          f'{len(fomc)} unchanged FOMC records, including '+str(sum(r['scheduled_or_unscheduled']=='unscheduled' for r in inherited_fomc))+' unscheduled')
    check('all previous finalized runs byte-identical',all(io.inventory(ROOT/'training_runs'/n)==v for n,v in m['protected_runs_before'].items()),f"{len(m['protected_runs_before'])} complete archived file inventories")
    check('operational code/model unchanged',all(archive.file_sha256(ROOT/n)==v for n,v in m['operational_hashes_before'].items()),m['operational_hashes_before'])
    script=(run/m['script_snapshot_path']).read_text(encoding='utf-8') if 'script_snapshot_path' in m else (ROOT/'gold_macro_event_integration_foundation_v1.py').read_text(encoding='utf-8')
    check('data-only implementation, no fitting or outcome access',not any(term in script for term in ['import xgboost','import sklearn','.fit(','predict_proba(','load_model(','import MetaTrader5']),
          'Manual code-path review plus forbidden operation checks; only DATE/TIME raw GOLD columns; archived macro and timestamp arrays only')
    check('pre-run Git and immutable executed code',m['git_dirty'] is False and m['pre_run_git']['head_sha']==m['pre_run_git']['origin_main_sha']==m['git_commit']
          and archive.file_sha256(ROOT/'gold_macro_event_integration_foundation_v1.py')==m['training_script_sha256'],m['pre_run_git'])
    passed=all(r['verdict']=='PASS' for r in checks)
    result=dict(internal_methodology='PASS' if passed else 'FAIL',final_untouched_validity='FAIL',
                final_untouched_reason='Data foundation only; no untouched strategy evaluation or performance claim.',checks=checks)
    archive.write_json(run/'validator_results.json',result)
    lines=['# Independent DATA Validator','',f"Internal methodology: {result['internal_methodology']}",
           'Final untouched strategy validity: FAIL (not a performance study).','',
           '| Check | Verdict | Evidence |','|---|---|---|']
    lines.extend(f"| {r['check']} | {r['verdict']} | {str(r['evidence']).replace('|','/')} |" for r in checks)
    lines.extend(['','## Walk-forward-validator applicability','',
        '| Check | Verdict | Evidence / scope |','|---|---|---|'])
    for name in ['chronology','feature leakage','label maturity','OOF predictions','calibration','threshold selection','purge/embargo','holdout contamination','recent-period reuse','execution alignment','cost assumptions','multiple-testing risk']:
        fail=name in ['holdout contamination','multiple-testing risk']
        lines.append(f"| {name} | {'FAIL' if fail else ('PASS' if passed else 'FAIL')} | "+
                     ('No untouched strategy claim supported; historical development only.' if fail else 'Data-only scope; relevant data checks above. No fitting, labels, calibration, threshold selection, trades or cost changes.')+' |')
    lines.extend(['','No OOS trading metrics calculated. Submitted performance claim: none; final performance validity invalid/not established.',
                  'Correction required: '+('none for internal data certification; wait for explicit authorization.' if passed else 'resolve failed data checks; no training authorized.')])
    (run/'validator.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metrics.update(validator_internal_methodology=result['internal_methodology'],validator_final_untouched_validity='FAIL',
        data_foundation_ready=passed and metrics['pce']['history_sufficient'],run_status='pass' if passed else 'fail')
    archive.write_json(run/'metrics.json',metrics)
    m['registry']['validator_result']='internal '+result['internal_methodology']+'; untouched FAIL, data-only'
    archive.write_json(run/'manifest.json',m)
    with (run/'report.md').open('a',encoding='utf-8') as handle:
        handle.write('\n## Independent final certification\n\n'+json.dumps(result,indent=2)+'\n\nMACRO EVENT B0/B1 DATA FOUNDATION READY = '+('YES' if metrics['data_foundation_ready'] else 'NO')+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    assert run.parent==ROOT/'training_runs' and not (run/'FINALIZED.json').exists()
    validate(run)
