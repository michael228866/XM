"""Fixed-scope data acquisition and integration audit. No model/label imports."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import gold_fomc_canonical_foundation_v1 as base
import training_run_history as archive

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'training_runs/20260903T154042Z_gemini_macro_event_timing_v1'
CANON = ROOT / 'training_runs/20260905T153912Z_gemini_fomc_canonical_foundation_v1'
LABEL = ROOT / 'training_runs/20260903T071729Z_gemini_execution_aligned_label_v1'
FAMILIES = ('CPI', 'EMPLOYMENT', 'PCE', 'FOMC')


def acquire(run):
    manifest = archive.read_json(run / 'manifest.json')
    assert manifest['git_dirty'] is False
    assert base.git('rev-parse', 'HEAD') == base.git('rev-parse', 'origin/main') == manifest['git_commit']
    assert archive.file_sha256(Path(__file__)) == manifest['training_script_sha256']
    manifest['pre_run_git'] = dict(pre_run_git_commit=manifest['git_commit'], pre_run_git_dirty=False,
                                 head_sha=manifest['git_commit'], origin_main_sha=manifest['git_commit'])
    manifest['protected_runs_before'] = {p.name: base.inventory(p) for p in (OLD, CANON)}
    manifest['operational_hashes_before'] = {n: archive.file_sha256(ROOT/n) for n in ('gemini.py', 'gold_long_recent_candidate_xgb.json')}
    for p in (OLD, CANON, LABEL):
        assert not archive.validate_run(p), p
    manifest['dependencies'] = {n: archive.file_sha256(ROOT/n) for n in ('gold_fomc_canonical_foundation_v1.py', 'training_run_history.py')}
    archive.write_json(run/'manifest.json', manifest)
    shutil.copyfile(CANON/'fomc_canonical_definition.json', run/'inherited_canonical_definition.json')
    definition = archive.read_json(CANON/'fomc_canonical_definition.json')
    definition.update(years=[2016, 2017], folds={'2016': [2016, 2016], '2017': [2017, 2017], '2016_2017': [2016, 2017]}, created_at_utc=base.stamp())
    archive.write_json(run/'fomc_canonical_definition.json', definition)
    shutil.copyfile(ROOT/'gold_fomc_2016_2017_foundation_v1_validator.py', run/'validator_script.py')
    (run/'sources').mkdir()
    sources = [base.fetch_one(run, f'{base.BASE}/monetarypolicy/fomchistorical{y}.htm') for y in (2016, 2017)]
    links = []
    for source in sources:
        if source['status'] != 'ok':
            raise RuntimeError('Official historical index unavailable; cannot establish denominator')
        doc = (run/source['path']).read_text(encoding='utf-8')
        for match in re.finditer(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', doc, re.S|re.I):
            try:
                url = base.safe_url(match[1])
            except ValueError:
                continue
            if re.search('201[67]', url) and any(s in url.lower() for s in ('monetary', 'fomc')):
                links.append(dict(index_url=source['url'], url=url, anchor=base.text_of(match[2]),
                                  context=base.text_of(doc[max(0,match.start()-1100):match.end()+300])))
    base.write_csv(run/'official_calendar_links.csv', links)
    urls = sorted({r['url'] for r in links if '/pressreleases/monetary' in r['url'] or 'statement' in r['anchor'].lower()})
    for url in urls:
        sources.append(base.fetch_one(run,url))
    base.write_csv(run/'fomc_source_provenance.csv', sources)
    print(f'ACQUIRED {len(sources)} official sources; review annotations required', flush=True)


def timestamp_rows(run):
    """Read only DATE/TIME; prove exact equality to frozen training timestamp hash."""
    prior = archive.read_json(LABEL/'fold_model_provenance.json')['folds'][0]
    inputs = archive.read_json(LABEL/'manifest.json')['data']['source_files']
    source = next(r for r in inputs if Path(r['path']).name.startswith('GOLD#_M1_'))
    path = Path(source['path'])
    assert archive.file_sha256(path) == source['sha256'], 'Historical source identity changed'
    parts = []
    for frame in pd.read_csv(path, sep='\t', usecols=lambda c:c.strip('<>').upper() in ('DATE','TIME'), chunksize=200000):
        frame.columns = [c.strip('<>').upper() for c in frame.columns]
        times = pd.to_datetime(frame['DATE']+' '+frame['TIME'])
        part = times[(times>=prior['train_start']) & (times<=prior['train_feature_end'])]
        if len(part):
            parts.append(part)
    times = pd.DatetimeIndex(pd.concat(parts).sort_values()).as_unit('ns')
    values = times.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    h = hashlib.sha256(str(values.dtype).encode()+np.asarray(values.shape,dtype=np.int64).tobytes()+values.tobytes()).hexdigest()
    assert h == prior['train_timestamp_sha256'] and len(values)==prior['train_rows'], 'Exact training rows not reconstructable without further work; stop'
    utc = times.tz_localize('Europe/Helsinki', ambiguous='raise', nonexistent='raise').tz_convert('UTC')
    np.savez_compressed(run/'training_timestamps.npz', broker_ns=values, utc_ns=utc.as_unit('ns').asi8)
    archive.write_json(run/'training_timestamp_provenance.json', dict(source=source, original_fold=prior,
        timestamp_sha256=h, exact_timestamp_reproduction=True, raw_prices_loaded=False,
        timezone='Inherited XM EET/EEST broker-wall convention; Europe/Helsinki IANA conversion; not a new broker-time calibration'))
    return utc, prior


def integration(run, verified):
    times, prior = timestamp_rows(run)
    required = times[0]-pd.Timedelta(minutes=1440)
    old_events = base.read_csv(OLD/'event_timestamp_dataset.csv')
    provenance = base.read_csv(OLD/'event_source_provenance.csv')
    sources = {r['requested_url']:r for r in provenance}
    rows = []
    for r in old_events:
        if r['event_type'] in FAMILIES[:3]:
            rows.append(dict(event_type=r['event_type'], release_timestamp_utc=r['release_timestamp_utc'],
                source_document=r['source_document'], original_timezone=r['original_timezone'],
                provenance_run=OLD.name, source_sha256=sources.get(r['source_document'],{}).get('content_sha256','')))
    rows.extend(dict(event_type='FOMC', release_timestamp_utc=r['release_timestamp_utc'], source_document=r['official_statement_url'],
                     original_timezone=r['original_timezone'], provenance_run=run.name, source_sha256=r['content_sha256']) for r in verified)
    rows = sorted((r for r in rows if pd.Timestamp(r['release_timestamp_utc'])<=times[-1]), key=lambda r:(r['release_timestamp_utc'],r['event_type']))
    base.write_csv(run/'integration_event_history.csv', rows)
    months = pd.period_range(required.tz_localize(None).to_period('M'), times[-1].tz_localize(None).to_period('M'), freq='M')
    audits, flags = [], {}
    coverage = archive.read_json(run/'coverage_summary.json')['coverage']
    for family in FAMILIES:
        items = [r for r in rows if r['event_type']==family]
        dates = pd.DatetimeIndex(pd.to_datetime([r['release_timestamp_utc'] for r in items], utc=True))
        missing = [] if family=='FOMC' else [str(m) for m in months if not any(d.strftime('%Y-%m')==str(m) for d in dates)]
        unverified = [r['source_document'] for r in items if not r['source_sha256']]
        # A monthly presence screen is necessary, NOT proof of canonical completeness.
        suspect = [r['source_document'] for r in items if family=='PCE' and re.search('state|county|metropolitan|regional',r['source_document'],re.I)]
        before = dates[dates<=required]
        known = len(before)>0 and not missing and not unverified and not suspect
        if family=='FOMC':
            known = known and all(r['verified']==r['expected'] and r['expected']>0 for r in coverage)
        flags[family] = np.full(len(times), known, dtype=bool)
        audits.append(dict(event_type=family, earliest_available_event=dates.min().isoformat() if len(dates) else '',
            latest_available_event=dates.max().isoformat() if len(dates) else '', prior_anchor=before.max().isoformat() if len(before) else '',
            required_history_start=required.isoformat(), required_history_end=times[-1].isoformat(),
            missing_months=missing, unverified_sources=unverified, ambiguous_noncanonical_sources=suspect,
            sufficient=bool(known), fully_constructable_rows=int(flags[family].sum()), incomplete_event_history_rows=int((~flags[family]).sum()),
            counting_method='Conservative whole-stream certification: if completeness is unproven, all required rows remain uncertified; not a measured per-row missing-event impact count'))
    combined = np.logical_and.reduce(list(flags.values()))
    np.savez_compressed(run/'integration_row_audit.npz', utc_ns=times.as_unit('ns').asi8, **flags, complete=combined)
    result = dict(earliest_training_timestamp=prior['train_start'], earliest_training_timestamp_utc=times[0].isoformat(),
        earliest_canonical_macro_history_required=required.isoformat(), training_rows=len(times),
        fully_constructable_rows=int(combined.sum()), incomplete_event_history_rows=int((~combined).sum()),
        families=audits, ready=bool(combined.all()), monthly_screen_is_not_independent_universe_proof=True,
        inherited_official_timestamp_provenance=True, feature_values_generated=False)
    base.write_csv(run/'integration_readiness_audit.csv', audits)
    archive.write_json(run/'integration_readiness.json', result)
    return result


def build(run):
    sources = {r['url']:r for r in base.read_csv(run/'fomc_source_provenance.csv')}
    expected = base.read_csv(run/'reviewed_events.csv')
    reviews = base.read_csv(run/'document_review.csv')
    for row in expected:
        src = sources[row['official_statement_url']]
        text = base.text_of((run/src['path']).read_text(encoding='utf-8')) if src['status']=='ok' else ''
        assert row['statement_release_date'][:4] in ('2016','2017')
        for field in ('identity_quote','date_quote','time_quote'):
            assert not row[field] or row[field] in text, (field,row)
        meeting = sources[row['meeting_detail_url']]
        assert row['meeting_evidence_quote'] in base.text_of((run/meeting['path']).read_text(encoding='utf-8'))
        row.update(event_id='FOMC_'+row['statement_release_date']+'_'+row['scheduled_or_unscheduled'],
                   official_statement_title=src['title'], content_sha256=src['sha256'],
                   acquisition_timestamp_utc=src['acquisition_timestamp_utc'], source_document_id=Path(src['url']).stem,
                   source_snapshot_path=src['path'], verification_status='unverified', release_timestamp_utc='',
                   official_release_timestamp_verified=False)
        if row['time_quote']:
            dt=datetime.fromisoformat(row['statement_release_date']+'T'+row['statement_release_time_local']).replace(tzinfo=ZoneInfo('America/New_York'))
            assert row['original_timezone'] in ('ET',dt.tzname())
            row.update(verification_status='verified', official_release_timestamp_verified=True, release_timestamp_utc=dt.astimezone(timezone.utc).isoformat())
    verified=[r for r in expected if r['official_release_timestamp_verified']]
    base.write_csv(run/'fomc_expected_events.csv', expected)
    base.write_csv(run/'fomc_verified_events.csv', verified, list(expected[0]))
    base.write_csv(run/'fomc_excluded_documents.csv', [r for r in reviews if r['classification']!='canonical_statement_retained'])
    coverage=[]
    for scope, years in [('2016',('2016',)),('2017',('2017',)),('2016_2017',('2016','2017'))]:
        selected=[r for r in expected if r['statement_release_date'][:4] in years]
        n=len(selected); k=sum(r['official_release_timestamp_verified'] for r in selected)
        coverage.append(dict(scope=scope, expected=n, verified=k, coverage_pct=100*k/n if n else 0, ambiguous=n-k,
                             missing=sum(not r['content_sha256'] for r in selected), duplicates=n-len({r['event_id'] for r in selected}), gate_pass=n>0 and k/n>=.95))
    base.write_csv(run/'fomc_coverage.csv',coverage)
    archive.write_json(run/'coverage_summary.json',{'coverage':coverage})
    readiness=integration(run, verified)
    metrics=dict(coverage=coverage, integration=readiness, unscheduled_decisions=sum(r['scheduled_or_unscheduled']=='unscheduled' for r in expected),
                 data_foundation_ready=all(r['gate_pass'] for r in coverage) and readiness['ready'], model_training_performed=False,
                 strategy_evaluation_performed=False, b0_b1_training_authorized=False)
    archive.write_json(run/'metrics.json',metrics)
    manifest=archive.read_json(run/'manifest.json')
    assert all(base.inventory(ROOT/'training_runs'/name)==values for name,values in manifest['protected_runs_before'].items())
    assert all(archive.file_sha256(ROOT/name)==value for name,value in manifest['operational_hashes_before'].items())
    manifest['data'].update(symbols=['FOMC'], data_sources=[base.BASE,'retained official CPI/EMPLOYMENT/PCE timestamps'],
        source_files=[dict(path=s['path'],sha256=s['sha256'],retention_status='raw source retained in Git archive') for s in sources.values() if s['status']=='ok'],
        timezone='official ET/EST/EDT -> UTC; inherited XM EET/EEST for training timestamps',
        data_start_utc='2016-01-01T00:00:00Z',data_end_utc='2018-01-01T00:00:00Z',
        train_start_utc='not_applicable_data_only',train_end_utc='not_applicable_data_only',train_rows=0,
        validation_start_utc=readiness['earliest_training_timestamp_utc'],validation_end_utc=readiness['families'][0]['required_history_end'],validation_rows=readiness['training_rows'],
        test_start_utc='not_applicable_data_only',test_end_utc='not_applicable_data_only',test_rows=0,
        purge_details='no labels or model fitting',embargo_details='not_applicable_data_only',raw_snapshot_retained=True,
        reproducibility_claim='FOMC raw sources and exact timestamp-only training cohort retained; inherited other-family source hashes only, not raw release snapshots',
        mt5_fetch={'used':False,'not_applicable_reason':'no MT5 access'})
    manifest['model']['not_applicable_reason']='data-only; no model or labels loaded'
    manifest['search']['not_applicable_reason']='fixed data scope, no strategy search'
    manifest['registry'].update({k:'not_applicable_data_only' for k in archive.REGISTRY_FIELDS})
    manifest['registry'].update(parent_or_incumbent=CANON.name,selected_configuration='2016-2017 canonical FOMC + existing macro readiness',validator_result='PENDING')
    manifest['promotion'].update(gemini_py_changed=False,operational_model_changed=False,operational_artifact_changed=False)
    archive.write_json(run/'manifest.json',manifest)
    notes=(run/'source_review_notes.md').read_text(encoding='utf-8') if (run/'source_review_notes.md').exists() else ''
    (run/'report.md').write_text('# GEMINI FOMC 2016-2017 FOUNDATION V1\n\n'+notes+'\n\n## Results\n\n```json\n'+__import__('json').dumps(metrics,indent=2)+'\n```\n',encoding='utf-8')
    print(__import__('json').dumps(metrics,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['acquire','build','self-check'])
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.action=='self-check':
        assert datetime(2016,7,1,tzinfo=ZoneInfo('Europe/Helsinki')).astimezone(timezone.utc).hour==21
        assert datetime(2016,1,27,14,tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc).hour==19
        print('SELF_CHECK_PASS'); return
    run=args.run_dir.resolve()
    assert run.parent==ROOT/'training_runs' and not (run/'FINALIZED.json').exists()
    {'acquire':acquire,'build':build}[args.action](run)


if __name__=='__main__':
    main()
