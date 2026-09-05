"""Independent source/timestamp/readiness audit; never imports experiment logic."""
import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import training_run_history as archive

ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def plain(path):
    return ' '.join(html.unescape(re.sub('<[^>]+>',' ',path.read_text(encoding='utf-8'))).split())


def validate(run):
    manifest=archive.read_json(run/'manifest.json')
    checks=[]
    def check(name,ok,evidence):
        checks.append(dict(check=name,verdict='PASS' if ok else 'FAIL',evidence=evidence))
    sources=read(run/'fomc_source_provenance.csv'); byurl={r['url']:r for r in sources}
    expected=read(run/'fomc_expected_events.csv'); verified=read(run/'fomc_verified_events.csv')
    reviewed=read(run/'document_review.csv'); reviewmap={r['url']:r for r in reviewed}
    check('official source and raw byte provenance',all(urlparse(r['url']).hostname=='www.federalreserve.gov' and (r['status']!='ok' or archive.file_sha256(run/r['path'])==r['sha256']) for r in sources),len(sources))
    original=archive.read_json(run/'inherited_canonical_definition.json'); current=archive.read_json(run/'fomc_canonical_definition.json')
    keys=set(original)-{'years','folds','created_at_utc'}
    check('unchanged canonical definition',all(original[k]==current[k] for k in keys) and current['years']==[2016,2017],sorted(keys))
    errors=[]
    for r in expected:
        text=plain(run/byurl[r['official_statement_url']]['path'])
        if not re.search('FOMC statement',r['official_statement_title'],re.I) or r['identity_quote'] not in text:
            errors.append(r['event_id']+': statement identity')
        if r['meeting_evidence_quote'] not in plain(run/byurl[r['meeting_detail_url']]['path']):
            errors.append(r['event_id']+': meeting')
        if r['scheduled_or_unscheduled']=='unscheduled' and not re.search('unscheduled|notation vote|conference call',r['meeting_evidence_quote'],re.I):
            errors.append(r['event_id']+': unscheduled evidence')
        if r['verification_status']=='verified':
            m=re.search(r'(\d+):(\d+) ([ap])\.m\. (EST|EDT|ET)',r['time_quote'])
            if not m or r['date_quote'] not in text or r['time_quote'] not in text:
                errors.append(r['event_id']+': time evidence');continue
            d=datetime.strptime(r['date_quote'],'%B %d, %Y')
            local=d.replace(hour=int(m[1])%12+(12 if m[3]=='p' else 0),minute=int(m[2]),tzinfo=ZoneInfo('America/New_York'))
            if local.astimezone(timezone.utc).isoformat()!=r['release_timestamp_utc'] or local.date().isoformat()!=r['statement_release_date']:
                errors.append(r['event_id']+': UTC mismatch')
    check('all-event identity / meeting / release-time / DST audit',not errors,errors or [r['event_id'] for r in expected])
    urls={r['official_statement_url'] for r in expected}
    links=read(run/'official_calendar_links.csv')
    statement_links={r['url'] for r in links if r['anchor'].lower()=='statement'}
    check('scheduled and unscheduled universe / exclusions',statement_links==urls and all(reviewmap[u]['classification']=='canonical_statement_retained' for u in urls) and all(r['url'] not in urls for r in read(run/'fomc_excluded_documents.csv')),{'official_statement_links':len(statement_links),'canonical':len(urls)})
    check('no duplicate identities or verified timestamps',len(expected)==len({r['event_id'] for r in expected}) and len(verified)==len({r['release_timestamp_utc'] for r in verified}),len(expected))
    coverage=read(run/'fomc_coverage.csv'); arithmetic=True; gates=[]
    for r in coverage:
        years=r['scope'].split('_'); n=sum(e['statement_release_date'][:4] in years for e in expected); k=sum(e['statement_release_date'][:4] in years for e in verified)
        arithmetic &= n==int(r['expected']) and k==int(r['verified']) and abs(100*k/n-float(r['coverage_pct']))<1e-8
        gates.append(k/n>=.95)
    check('95 percent coverage arithmetic',arithmetic,coverage)
    audit=archive.read_json(run/'integration_readiness.json')
    info=archive.read_json(run/'training_timestamp_provenance.json')
    with np.load(run/'training_timestamps.npz',allow_pickle=False) as data:
        broker=data['broker_ns']; utc=data['utc_ns']
    h=hashlib.sha256(str(broker.dtype).encode()+np.asarray(broker.shape,dtype=np.int64).tobytes()+broker.tobytes()).hexdigest()
    calculated=pd.DatetimeIndex(pd.to_datetime(broker)).tz_localize('Europe/Helsinki',ambiguous='raise',nonexistent='raise').tz_convert('UTC').as_unit('ns').asi8
    check('every exact required training timestamp / timezone',h==info['original_fold']['train_timestamp_sha256'] and np.array_equal(utc,calculated) and len(utc)==info['original_fold']['train_rows'],{'rows':len(utc),'hash':h})
    events=read(run/'integration_event_history.csv')
    required=pd.Timestamp(utc[0],tz='UTC')-pd.Timedelta(minutes=1440)
    months=pd.period_range(required.tz_localize(None).to_period('M'),pd.Timestamp(utc[-1]).to_period('M'),freq='M')
    expected_flags={}
    for family in ('CPI','EMPLOYMENT','PCE','FOMC'):
        items=[r for r in events if r['event_type']==family]
        dates=pd.DatetimeIndex(pd.to_datetime([r['release_timestamp_utc'] for r in items],utc=True))
        missing=[] if family=='FOMC' else [str(m) for m in months if not any(d.strftime('%Y-%m')==str(m) for d in dates)]
        suspect=[r['source_document'] for r in items if family=='PCE' and re.search('state|county|metropolitan|regional',r['source_document'],re.I)]
        ok=any(d<=required for d in dates) and not missing and not suspect and all(r['source_sha256'] for r in items)
        if family=='FOMC':
            ok=ok and len(expected)==len(verified) and bool(expected)
        expected_flags[family]=ok
        submitted=next(r for r in audit['families'] if r['event_type']==family)
        check(f'{family} readiness evidence accounting',submitted['sufficient']==ok and submitted['missing_months']==missing,{'sufficient':ok,'missing_months':missing,'ambiguous_sources':suspect})
    with np.load(run/'integration_row_audit.npz',allow_pickle=False) as rows:
        flagsok=np.array_equal(rows['utc_ns'],utc) and all(np.all(rows[k]==v) for k,v in expected_flags.items())
        complete=all(expected_flags.values()); incomplete=0 if complete else len(utc)
        flagsok &= np.all(rows['complete']==complete) and audit['incomplete_event_history_rows']==incomplete
    check('missing history remains uncertified / never zero event features',flagsok and audit['feature_values_generated'] is False,{'uncertified_rows':incomplete,'method':'conservative full-stream certificate, not exact missing-event impact attribution'})
    check('existing finalized runs unchanged',all(values=={p.relative_to(ROOT/'training_runs'/name).as_posix():archive.file_sha256(p) for p in (ROOT/'training_runs'/name).rglob('*') if p.is_file()} for name,values in manifest['protected_runs_before'].items()),list(manifest['protected_runs_before']))
    check('operational artifacts unchanged',all(archive.file_sha256(ROOT/n)==v for n,v in manifest['operational_hashes_before'].items()),manifest['operational_hashes_before'])
    script=(run/'training_script.py').read_text(encoding='utf-8')
    check('no models / labels / strategy performance',not manifest['model']['trained'] and not (run/'models').exists() and not re.search(r'\.fit\(|import xgboost|import gold_gemini|simulate\(',script),'DATE/TIME columns only; no label or model pipeline')
    internal=all(c['verdict']=='PASS' for c in checks)
    ready=internal and all(gates) and incomplete==0
    result=dict(internal_methodology='PASS' if internal else 'FAIL',overall='PASS' if ready else 'FAIL',
                every_required_row_constructable='PASS' if incomplete==0 else 'FAIL',data_foundation_ready=ready,checks=checks,
                strategy_validity='not_applicable_data_only',b0_b1_training_authorized=False)
    archive.write_json(run/'validator.json',result)
    (run/'validator.md').write_text('# Independent data validator\n\nOverall: '+result['overall']+'\n\nInternal methodology: '+result['internal_methodology']+'\n\n| Check | Verdict | Evidence |\n|---|---|---|\n'+'\n'.join('| '+c['check']+' | '+c['verdict']+' | '+str(c['evidence']).replace('|','/')+' |' for c in checks)+'\n\nEvery required row constructable: '+result['every_required_row_constructable']+'\nNo strategy-validity claim. Do not train until all data gaps are independently resolved.\n',encoding='utf-8')
    manifest['registry']['validator_result']=result['overall']; manifest['data_validator']=result
    archive.write_json(run/'manifest.json',manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run_dir',type=Path); args=p.parse_args()
    assert not (args.run_dir/'FINALIZED.json').exists()
    validate(args.run_dir.resolve())
