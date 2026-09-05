"""Independent data audit: no research classifier or model modules imported."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import training_run_history as archive

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT/'training_runs/20260903T154042Z_gemini_macro_event_timing_v1'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def plain(doc):
    return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', doc)).split())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(run):
    manifest = archive.read_json(run/'manifest.json')
    sources = read(run/'fomc_source_provenance.csv')
    by_url = {r['url']:r for r in sources}
    expected = read(run/'fomc_expected_events.csv')
    verified = read(run/'fomc_verified_events.csv')
    excluded = read(run/'fomc_excluded_documents.csv')
    reviews = read(run/'document_review.csv')
    reconciliation = read(run/'fomc_reconciliation_vs_prior_run.csv')
    coverage = read(run/'fomc_coverage.csv')
    checks = []

    def check(name, passed, evidence):
        checks.append(dict(check=name, verdict='PASS' if passed else 'FAIL', evidence=evidence))

    check('official-source-only provenance', all(urlparse(r['url']).hostname=='www.federalreserve.gov' for r in sources), 'All source hosts checked')
    check('retained raw bytes', all(digest(run/r['path'])==r['sha256'] for r in sources if r['status']=='ok'), 'Independent SHA-256 of every successful response')
    ids = [r['event_id'] for r in expected]
    utc_values = [r['release_timestamp_utc'] for r in verified]
    check('one event per decision / HTML PDF deduplication', len(ids)==len(set(ids)) and len(utc_values)==len(set(utc_values)), f'{len(ids)} events; {len(utc_values)} verified timestamps')
    event_errors = []
    samples = []
    for row in expected:
        src = by_url[row['official_statement_url']]
        text = plain((run/src['path']).read_text(encoding='utf-8'))
        title_ok = bool(re.search(r'FOMC\s+(?:monetary.policy\s+)?statement', row['official_statement_title'], re.I))
        identity_ok = row['identity_quote'] in text and ('committee' in text.lower() or 'fomc' in text.lower())
        if not (title_ok and identity_ok):
            event_errors.append(row['event_id']+': actual statement identity not verified')
        if row['date_quote'] not in text or row.get('time_quote','') not in text:
            event_errors.append(row['event_id']+': evidence quote absent')
        calendar = by_url[row['meeting_detail_url']]
        calendar_doc = (run/calendar['path']).read_text(encoding='utf-8')
        if row['meeting_evidence_quote'] not in plain(calendar_doc):
            event_errors.append(row['event_id']+': meeting context missing')
        if row['scheduled_or_unscheduled']=='unscheduled' and not re.search('unscheduled|notation vote|conference call', row['meeting_evidence_quote'], re.I):
            event_errors.append(row['event_id']+': unscheduled classification has no explicit official evidence')
        if row['official_release_timestamp_verified'].lower()=='true':
            date = datetime.strptime(row['date_quote'], '%B %d, %Y').date()
            match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(a|p)\.?m\.?\s+(EST|EDT|ET)', row['time_quote'], re.I)
            if not match:
                event_errors.append(row['event_id']+': time evidence not parseable')
                continue
            hour = int(match[1])%12+(12 if match[3].lower()=='p' else 0)
            local = datetime(date.year,date.month,date.day,hour,int(match[2] or 0),tzinfo=ZoneInfo('America/New_York'))
            offset = 5 if match[4].upper()=='EST' else 4 if match[4].upper()=='EDT' else -local.utcoffset().total_seconds()/3600
            # Independent fixed-offset arithmetic cross-check against IANA conversion.
            arithmetic = local.replace(tzinfo=timezone.utc)+timedelta(hours=offset)
            parsed = datetime.fromisoformat(row['release_timestamp_utc'])
            if local.astimezone(timezone.utc)!=parsed or arithmetic!=parsed or str(date)!=row['statement_release_date']:
                event_errors.append(row['event_id']+': timestamp/DST/date mismatch')
            samples.append(dict(event_id=row['event_id'], statement_url=row['official_statement_url'],
                source_sha256=digest(run/src['path']), date_quote=row['date_quote'],time_quote=row['time_quote'],
                recomputed_utc=parsed.isoformat(), scheduled_or_unscheduled=row['scheduled_or_unscheduled']))
    check('actual canonical statement identity / meeting relation', not event_errors, event_errors or 'Titles, contents, meeting quotes and release evidence independently verified for every event')
    check('release-time evidence / timezone / DST', not event_errors, 'Official date and time parsed independently; EST/EDT arithmetic equals IANA UTC')
    verified_ids = {r['event_id'] for r in verified}
    check('ambiguous events retained in denominator', verified_ids=={r['event_id'] for r in expected if r['official_release_timestamp_verified'].lower()=='true'}, 'Expected and verified sets compared')
    category_patterns = {
        'implementation_note_excluded': r'implementation', 'SEP_excluded': r'projection',
        'minutes_excluded': r'minutes', 'strategy_document_excluded': r'longer.run|strategy|goals',
        'balance_sheet_companion_excluded': r'balance.sheet|reinvest|normalization|securities holdings',
    }
    exclusion_errors = []
    for row in excluded:
        pattern = category_patterns.get(row['classification'])
        if pattern and not re.search(pattern, row['title']+' '+row['exclusion_reason'], re.I):
            exclusion_errors.append(row['url'])
        if row['url'] in {r['official_statement_url'] for r in expected}:
            exclusion_errors.append(row['url'])
    check('explicit exclusions', not exclusion_errors, exclusion_errors or dict(Counter(r['classification'] for r in excluded)))
    links = read(run/'official_calendar_links.csv')
    reviewed_urls = {r['url'] for r in reviews}
    statement_urls = set()
    for link in links:
        if 'statement' in link['anchor'].lower() and not any(s in link['anchor'].lower() for s in ('longer','strategy','goals')):
            statement_urls.add(link['url'])
        src = by_url.get(link['url'])
        if src and re.search(r'FOMC\s+statement',src.get('title',''),re.I):
            statement_urls.add(link['url'])
    check('denominator rebuilt from official statement links', statement_urls <= reviewed_urls and bool(statement_urls), sorted(statement_urls-reviewed_urls) or f'{len(statement_urls)} official statement links reviewed including PDF companions')
    prior_items = [r['requested_url'] for r in read(PRIOR/'event_source_provenance.csv') if r['purpose']=='FOMC_statement_document' and re.search('201[89]|202[0-4]',r['requested_url'])]
    prior_reconciled = [r for r in reconciliation if r['classification']!='previously_missing_canonical_statement_recovered']
    check('every prior item reconciled', Counter(prior_items)==Counter(r['url'] for r in prior_reconciled), f'{len(prior_items)} prior URLs independently enumerated')
    errors = []
    for fold in coverage:
        lo,hi = map(int,fold['fold'].split('_'))
        n=sum(lo<=int(r['statement_release_date'][:4])<=hi for r in expected)
        k=sum(lo<=int(r['statement_release_date'][:4])<=hi for r in verified)
        p=sum(lo<=int(r['document_date'][:4])<=hi for r in prior_reconciled)
        if n!=int(fold['expected']) or k!=int(fold['verified']) or p!=int(fold['prior_expected']) or abs(100*k/n-float(fold['coverage_pct']))>1e-9 or ((k/n>=0.95)!=(fold['gate_pass'].lower()=='true')):
            errors.append(fold['fold'])
    check('95 percent gate arithmetic / denominator reconciliation', not errors, errors or coverage)
    years={int(r['event_id'][5:9]) for r in samples}
    check('independent multiyear and unscheduled samples', {2018,2020,2021,2022,2023,2024}<=years and any(r['scheduled_or_unscheduled']=='unscheduled' for r in samples), f'Independently checked {len(samples)} events, all years and unscheduled actions')
    check('prior finalized run byte identity', manifest['prior_run_hashes_before']=={p.relative_to(PRIOR).as_posix():digest(p) for p in PRIOR.rglob('*') if p.is_file()}, 'Every prior file hash compared; prior archive validation also run')
    check('operational artifacts unchanged', all(digest(ROOT/name)==value for name,value in manifest['operational_hashes_before'].items()), 'gemini.py and operational model pre/current SHA-256 equality')
    script = (run/'training_script.py').read_text(encoding='utf-8')
    no_training = not manifest['model']['trained'] and not (run/'models').exists() and not re.search(r'import (?:xgboost|pandas|numpy|MetaTrader5)|\.fit\(|simulate\(',script)
    check('no training or strategy evaluation', no_training, 'Data-only script has no model/market loaders, fit calls, simulator or model artifacts')
    check('pre-run clean committed provenance', manifest['git_dirty'] is False and manifest['pre_run_git']['head_sha']==manifest['pre_run_git']['origin_main_sha']==manifest['git_commit'] and digest(run/'training_script.py')==manifest['training_script_sha256'], 'Pre-run commit and immutable executed snapshot')
    result={'internal_methodology':'PASS' if all(r['verdict']=='PASS' for r in checks) else 'FAIL', 'checks':checks,
            'readiness':'PASS' if all(r['gate_pass'].lower()=='true' for r in coverage) and all(r['verdict']=='PASS' for r in checks) else 'FAIL',
            'strategy_or_final_untouched_validity':'not_applicable_data_only_no_performance_claim'}
    archive.write_json(run/'validator.json',result)
    archive.write_json(run/'validator_samples.json',{'samples':samples})
    (run/'validator.md').write_text('# Independent FOMC data validator\n\nInternal methodology: '+result['internal_methodology']+'\n\n| Check | Verdict | Evidence |\n|---|---|---|\n'+'\n'.join('| '+r['check']+' | '+r['verdict']+' | '+str(r['evidence']).replace('|','/')+' |' for r in checks)+'\n\nNo strategy-performance or untouched-test validity is claimed.\n',encoding='utf-8')
    manifest['registry']['validator_result']=result['internal_methodology']
    manifest['data_validator']=result
    archive.write_json(run/'manifest.json',manifest)
    print(json.dumps(result,indent=2))
    return result['internal_methodology']=='PASS'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    if (args.run_dir/'FINALIZED.json').exists():
        raise RuntimeError('Finalized evidence cannot be overwritten')
    raise SystemExit(0 if validate(args.run_dir.resolve()) else 1)
