"""Data-only official FOMC acquisition, reviewed canonicalization, and archival.

Lifecycle: acquire -> inspect retained official documents -> reviewed_events.csv
and document_review.csv -> build -> independent validator -> finalize.
Review CSVs are source-backed data annotations, never model/strategy choices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import training_run_history as archive

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / 'training_runs/20260903T154042Z_gemini_macro_event_timing_v1'
BASE = 'https://www.federalreserve.gov'
FOLDS = {'2018_2020': (2018, 2020), '2021_2022': (2021, 2022), '2023_2024': (2023, 2024)}
CATEGORIES = {
    'canonical_statement_retained', 'duplicate_representation',
    'implementation_note_excluded', 'SEP_excluded', 'minutes_excluded',
    'strategy_document_excluded', 'balance_sheet_companion_excluded',
    'other_non_statement_monetary_release_excluded', 'unresolved',
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError('CSV schema required')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def inventory(directory):
    return {p.relative_to(directory).as_posix(): archive.file_sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def safe_url(url):
    url = urllib.parse.urljoin(BASE, html.unescape(url)).split('#')[0]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'www.federalreserve.gov':
        raise ValueError(f'Non-official URL: {url}')
    return url


def text_of(document):
    document = re.sub(r'<(script|style)\b.*?</\1>', '', document, flags=re.S | re.I)
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', document)).split())


def title_of(document):
    titles = re.findall(r'<(?:h3|h2)[^>]*class=["\'][^"\']*title[^"\']*["\'][^>]*>(.*?)</(?:h3|h2)>', document, re.S | re.I)
    if not titles:
        titles = re.findall(r'<title[^>]*>(.*?)</title>', document, re.S | re.I)
    return text_of(titles[0]) if titles else ''


def link_rows(document, source):
    rows = []
    for match in re.finditer(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', document, re.S | re.I):
        try:
            url = safe_url(urllib.parse.urljoin(source, match[1]))
        except ValueError:
            continue
        if not re.search(r'(?:201[8-9]|202[0-4])', url):
            continue
        if not any(word in url.lower() for word in ('fomc', 'monetary', 'projection')):
            continue
        rows.append({'index_url': source, 'url': url, 'anchor': text_of(match[2]),
                     'context': text_of(document[max(0, match.start()-1000):match.end()+250])})
    return rows


def fetch_one(run, url):
    url = safe_url(url)
    ident = hashlib.sha256(url.encode()).hexdigest()[:20]
    suffix = '.pdf' if urllib.parse.urlparse(url).path.endswith('.pdf') else '.html'
    target = run / 'sources' / (ident + suffix)
    record = {'url': url, 'acquisition_timestamp_utc': stamp(), 'path': target.relative_to(run).as_posix()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (FOMC historical data audit)'})
        with urllib.request.urlopen(request, timeout=40) as response:
            safe_url(response.geturl())
            body = response.read()
            record['final_url'] = response.geturl()
        target.write_bytes(body)
        record.update(status='ok', sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), error='')
        if suffix == '.html':
            doc = body.decode('utf-8', errors='replace')
            record['title'] = title_of(doc)
            text = text_of(doc)
            start = re.search(r'(?:For release|For immediate release)', text, re.I)
            record['release_excerpt'] = text[start.start():start.start()+900] if start else ''
    except OSError as exc:
        record.update(status='error', sha256='', bytes=0, error=str(exc), title='')
    return record


def freeze(run):
    manifest = archive.read_json(run / 'manifest.json')
    assert manifest['status'] == 'in_progress' and manifest['git_dirty'] is False
    assert manifest['training_script_sha256'] == archive.file_sha256(Path(__file__))
    head, remote = git('rev-parse', 'HEAD'), git('rev-parse', 'origin/main')
    assert head == remote == manifest['git_commit']
    assert not archive.validate_run(PRIOR)
    manifest['pre_run_git'] = dict(pre_run_git_commit=head, pre_run_git_dirty=False,
                                   head_sha=head, origin_main_sha=remote, head_equals_origin_main=True)
    manifest['operational_hashes_before'] = {name: archive.file_sha256(ROOT / name)
        for name in ('gemini.py', 'gold_long_recent_candidate_xgb.json')}
    manifest['prior_run_hashes_before'] = inventory(PRIOR)
    manifest['data_only'] = True
    archive.write_json(run / 'manifest.json', manifest)
    definition = {
        'event': 'One official FOMC monetary-policy decision statement public release',
        'years': list(range(2018, 2025)), 'folds': FOLDS, 'minimum_coverage_each_fold': 0.95,
        'denominator': 'Statements enumerated from official meeting/policy records; no fixed annual count',
        'exclusions': sorted(CATEGORIES - {'canonical_statement_retained', 'unresolved'}),
        'classification': 'Official title/content and meeting relation, never URL suffix alone',
        'review_protocol': 'Source-backed CSV annotations, including event date/time quotes and calendar membership; all ambiguous events remain in denominator',
        'timezone': 'Explicit EST/EDT verified against America/New_York; ET uses IANA DST rules',
        'training_authorized': False, 'strategy_evaluation_authorized': False,
        'created_at_utc': stamp(),
    }
    archive.write_json(run / 'fomc_canonical_definition.json', definition)
    shutil.copyfile(ROOT / 'gold_fomc_canonical_foundation_v1_validator.py', run / 'validator_script.py')


def acquire(run):
    if (run / 'fomc_source_provenance.csv').exists():
        raise FileExistsError('Acquisition already retained; use supplement with explicit official URLs')
    freeze(run)
    (run / 'sources').mkdir()
    roots = [f'{BASE}/monetarypolicy/fomchistorical{year}.htm' for year in range(2018, 2025)]
    roots += [f'{BASE}/monetarypolicy/fomccalendars.htm']
    records = [fetch_one(run, url) for url in roots]
    links = []
    for row in records:
        if row['status'] == 'ok':
            links.extend(link_rows((run / row['path']).read_text(encoding='utf-8'), row['url']))
    # Meeting-detail pages establish scheduled/unscheduled context, not annual heuristics.
    detail_urls = {row['url'] for row in links if re.search(r'/fomc\d{8}(?:meeting)?\.htm$', row['url'])}
    with ThreadPoolExecutor(max_workers=4) as pool:
        details = list(pool.map(lambda url: fetch_one(run, url), sorted(detail_urls)))
    records.extend(details)
    for row in details:
        if row['status'] == 'ok':
            links.extend(link_rows((run / row['path']).read_text(encoding='utf-8'), row['url']))
    write_csv(run / 'official_calendar_links.csv', links, ['index_url', 'url', 'anchor', 'context'])
    prior_urls = {row['requested_url'] for row in read_csv(PRIOR / 'event_source_provenance.csv')
                  if row['purpose'] == 'FOMC_statement_document' and re.search('201[89]|202[0-4]', row['requested_url'])}
    linked_urls = {row['url'] for row in links if re.search(r'(pressreleases/monetary|fomcstatement|fomcminutes|fomcproj)', row['url'], re.I)}
    urls = sorted((prior_urls | linked_urls) - {r['url'] for r in records})
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ordinal, row in enumerate(pool.map(lambda url: fetch_one(run, url), urls), 1):
            records.append(row)
            if ordinal % 25 == 0:
                print(f'ACQUIRED {ordinal}/{len(urls)} documents', flush=True)
    write_csv(run / 'fomc_source_provenance.csv', records)
    print(f'ACQUISITION_COMPLETE sources={len(records)} links={len(links)}', flush=True)


def supplement(run, urls):
    records = read_csv(run / 'fomc_source_provenance.csv')
    existing = {row['url'] for row in records}
    for url in urls:
        url = safe_url(url)
        if url in existing:
            raise ValueError('Source already retained')
        records.append(fetch_one(run, url))
    write_csv(run / 'fomc_source_provenance.csv', records)


def build(run):
    """Build solely from reviewed official evidence; refuse unverifiable claims."""
    sources = {row['url']: row for row in read_csv(run / 'fomc_source_provenance.csv')}
    events = read_csv(run / 'reviewed_events.csv')
    reviews = read_csv(run / 'document_review.csv')
    expected, verified = [], []
    for item in events:
        source = sources[item['official_statement_url']]
        doc = text_of((run / source['path']).read_text(encoding='utf-8')) if source['status'] == 'ok' else ''
        for field in ('identity_quote', 'date_quote', 'time_quote'):
            if item.get(field) and item[field] not in doc:
                raise ValueError(f'{field} is not in official document: {item}')
        if item['scheduled_or_unscheduled'] not in ('scheduled', 'unscheduled'):
            raise ValueError('Schedule type requires official evidence')
        calendar_source = sources[item['meeting_detail_url']]
        context = text_of((run / calendar_source['path']).read_text(encoding='utf-8'))
        if not item.get('meeting_evidence_quote') or item['meeting_evidence_quote'] not in context:
            raise ValueError('Missing official meeting context')
        row = dict(item)
        row['event_id'] = 'FOMC_' + item['statement_release_date'] + '_' + item['scheduled_or_unscheduled']
        row['content_sha256'] = source['sha256']
        row['acquisition_timestamp_utc'] = source['acquisition_timestamp_utc']
        row['source_document_id'] = Path(urllib.parse.urlparse(source['url']).path).stem
        row['official_statement_title'] = source['title']
        row['source_snapshot_path'] = source['path']
        row['verification_status'] = 'unverified'
        row['official_release_timestamp_verified'] = False
        row['release_timestamp_utc'] = ''
        if item.get('statement_release_time_local') and item.get('time_quote'):
            local = datetime.fromisoformat(item['statement_release_date']+'T'+item['statement_release_time_local']).replace(tzinfo=ZoneInfo('America/New_York'))
            zone = item['original_timezone']
            if zone not in ('ET', 'EST', 'EDT') or (zone != 'ET' and local.tzname() != zone):
                raise ValueError(f'DST abbreviation mismatch {item}')
            row['release_timestamp_utc'] = local.astimezone(timezone.utc).isoformat()
            row['verification_status'] = 'verified'
            row['official_release_timestamp_verified'] = True
        expected.append(row)
        if row['official_release_timestamp_verified']:
            verified.append(row)
    if len({row['event_id'] for row in expected}) != len(expected):
        raise ValueError('Duplicate canonical event identities')
    if len({row['release_timestamp_utc'] for row in verified}) != len(verified):
        raise ValueError('Duplicate canonical event timestamps')
    expected.sort(key=lambda row: row['statement_release_date'])
    verified.sort(key=lambda row: row['release_timestamp_utc'])
    write_csv(run / 'fomc_expected_events.csv', expected)
    write_csv(run / 'fomc_verified_events.csv', verified, list(expected[0]))
    event_by_url = {row['official_statement_url']: row for row in expected}
    excluded = []
    review_by_url = {}
    for row in reviews:
        if row['classification'] not in CATEGORIES:
            raise ValueError('Invalid exclusion category')
        if row['url'] in review_by_url:
            raise ValueError('Duplicate reviewed URL')
        review_by_url[row['url']] = row
        if row['classification'] == 'canonical_statement_retained':
            if row['url'] not in event_by_url:
                raise ValueError('Retained review has no canonical event')
        else:
            excluded.append(row)
    write_csv(run / 'fomc_excluded_documents.csv', excluded)
    old = [row for row in read_csv(PRIOR/'event_source_provenance.csv')
           if row['purpose'] == 'FOMC_statement_document' and re.search('201[89]|202[0-4]', row['requested_url'])]
    old_verified_urls = {r['source_document'] for r in read_csv(PRIOR/'event_timestamp_dataset.csv') if r['event_type']=='FOMC'}
    reconciliation = []
    for item in old:
        url = item['requested_url']
        review = review_by_url.get(url)
        if review is None:
            raise ValueError(f'Unreviewed prior URL: {url}')
        row = dict(review)
        row['prior_expected'] = True
        row['prior_timestamp_retained'] = url in old_verified_urls
        row['canonical_event_id'] = event_by_url[url]['event_id'] if url in event_by_url else row.get('canonical_event_id','')
        reconciliation.append(row)
    old_urls = {row['requested_url'] for row in old}
    for row in expected:
        if row['official_statement_url'] not in old_verified_urls:
            reconciliation.append(dict(url=row['official_statement_url'], document_date=row['statement_release_date'],
                title=row['official_statement_title'], classification='previously_missing_canonical_statement_recovered',
                exclusion_reason='canonical public statement absent from prior retained verified timestamp dataset',
                canonical_event_id=row['event_id'], prior_expected=row['official_statement_url'] in old_urls,
                prior_timestamp_retained=False))
    write_csv(run / 'fomc_reconciliation_vs_prior_run.csv', reconciliation)
    coverage = []
    for fold, (lo, hi) in FOLDS.items():
        selected = [r for r in expected if lo <= int(r['statement_release_date'][:4]) <= hi]
        valid = [r for r in selected if r['official_release_timestamp_verified']]
        old_fold = [r for r in reconciliation if r['classification'] != 'previously_missing_canonical_statement_recovered'
                    and lo <= int(r['document_date'][:4]) <= hi]
        categories = {category: sum(r['classification']==category for r in old_fold) for category in sorted(CATEGORIES)}
        coverage.append(dict(fold=fold, expected=len(selected), verified=len(valid),
            coverage_pct=100*len(valid)/len(selected) if selected else 0,
            ambiguous=len(selected)-len(valid), missing=sum(not r['content_sha256'] for r in selected),
            duplicates=0, first_timestamp=min((r['release_timestamp_utc'] for r in valid), default=''),
            last_timestamp=max((r['release_timestamp_utc'] for r in valid), default=''),
            gate_pass=bool(selected) and len(valid)/len(selected)>=0.95,
            prior_expected=len(old_fold), denominator_difference=len(old_fold)-len(selected),
            prior_exclusions=json.dumps(categories, sort_keys=True)))
    write_csv(run / 'fomc_coverage.csv', coverage)
    manifest = archive.read_json(run/'manifest.json')
    assert inventory(PRIOR) == manifest['prior_run_hashes_before']
    after = {name: archive.file_sha256(ROOT/name) for name in manifest['operational_hashes_before']}
    assert after == manifest['operational_hashes_before']
    metrics = dict(annual_counts={str(y):sum(r['statement_release_date'].startswith(str(y)) for r in expected) for y in range(2018,2025)},
        folds=coverage, readiness='PASS' if all(r['gate_pass'] for r in coverage) else 'FAIL',
        prior_denominator_inflation_confirmed=all(r['denominator_difference']>0 for r in coverage),
        prior_excluded_categories={c:sum(r['classification']==c for r in reconciliation) for c in sorted(CATEGORIES)},
        recovered_previously_missing=sum(r['classification']=='previously_missing_canonical_statement_recovered' for r in reconciliation),
        unresolved_canonical=len(expected)-len(verified), model_training_performed=False,
        strategy_evaluation_performed=False, operational_artifacts_changed=False, prior_run_changed=False,
        macro_event_rerun_authorized=False)
    archive.write_json(run/'metrics.json', metrics)
    lines = ['# GEMINI FOMC CANONICAL TIMESTAMP FOUNDATION V1', '', 'Status: `'+metrics['readiness']+'` (data only)', '',
        'Expected universe is reviewed from official FOMC meeting/policy records. No fixed annual denominator is used.', '',
        '| Fold | Prior expected | Canonical | Verified | Coverage |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['fold']} | {r['prior_expected']} | {r['expected']} | {r['verified']} | {r['coverage_pct']:.2f}% |" for r in coverage]
    lines += ['', 'Annual counts: '+json.dumps(metrics['annual_counts']), '',
        'prior_denominator_inflation_confirmed = '+str(metrics['prior_denominator_inflation_confirmed']), '',
        'Reconciliation categories (prior items): '+json.dumps(metrics['prior_excluded_categories']), '',
        'See official_calendar_links.csv, document_review.csv, reviewed_events.csv and retained sources for each classification and timestamp quote.', '',
        'No GOLD bars, strategy labels, predictions, or performance were loaded. No new forward cutoff. No B0/B1 training authorization.', '']
    notes = run/'source_review_notes.md'
    if notes.exists():
        lines.append(notes.read_text(encoding='utf-8'))
    (run/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest['operational_hashes_after'] = after
    manifest['prior_run_unchanged'] = True
    manifest['data'].update(symbols=['FOMC policy statements'], data_sources=[BASE],
        source_files=[dict(path=r['path'], sha256=r['sha256'], retention_status='raw official document retained in this Git-archived run') for r in sources.values() if r['status']=='ok'],
        timezone='America/New_York official ET/EST/EDT converted to UTC; no GOLD alignment claimed',
        data_start_utc='2018-01-01T00:00:00Z', data_end_utc='2025-01-01T00:00:00Z',
        train_start_utc='not_applicable_data_only', train_end_utc='not_applicable_data_only', train_rows=0,
        validation_start_utc='2018-01-01T00:00:00Z', validation_end_utc='2025-01-01T00:00:00Z', validation_rows=len(expected),
        test_start_utc='not_applicable_data_only', test_end_utc='not_applicable_data_only', test_rows=0,
        purge_details='not_applicable_no_labels', embargo_details='not_applicable_no_model', raw_snapshot_retained=True,
        reproducibility_claim='retained raw official sources and reviewed canonical annotations; no strategy-validity claim',
        mt5_fetch={'used':False, 'not_applicable_reason':'data-only official Federal Reserve documents'})
    manifest['model']['not_applicable_reason'] = 'Data only; no model loaded, fitted or saved'
    manifest['search']['not_applicable_reason'] = 'No strategy, parameter, feature or threshold search'
    manifest['registry'].update({field:'not_applicable_data_only' for field in archive.REGISTRY_FIELDS})
    manifest['registry'].update(parent_or_incumbent=PRIOR.name, selected_configuration='fixed canonical FOMC definition; 95% each fold', validator_result='PENDING')
    manifest['promotion'].update(gemini_py_changed=False, operational_model_changed=False, operational_artifact_changed=False)
    manifest['artifacts'] = [dict(path=p.relative_to(run).as_posix(), sha256=archive.file_sha256(p), retention_status='stored in run directory for Git archival')
        for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('manifest.json','stdout.log','FINALIZED.json')]
    archive.write_json(run/'manifest.json', manifest)
    print(json.dumps(metrics, indent=2), flush=True)


def self_check():
    assert title_of('<title>Federal Reserve issues FOMC statement</title>') == 'Federal Reserve issues FOMC statement'
    assert text_of('<p>A &amp; B</p>') == 'A & B'
    assert safe_url('/monetarypolicy/fomchistorical2020.htm').startswith(BASE)
    try:
        safe_url('https://example.com/fomc')
    except ValueError:
        pass
    else:
        raise AssertionError('Non-official source accepted')
    assert datetime(2020,3,15,17,tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc).hour == 21
    print('SELF_CHECK_PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['acquire','supplement','build','self-check'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--url', action='append', default=[])
    args = parser.parse_args()
    if args.action == 'self-check':
        return self_check()
    run = args.run_dir.resolve()
    assert run.parent == (ROOT/'training_runs').resolve()
    assert not (run/'FINALIZED.json').exists()
    {'acquire':acquire, 'build':build, 'supplement':lambda p:supplement(p,args.url)}[args.action](run)


if __name__ == '__main__':
    main()
