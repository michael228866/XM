"""Adversarial audit of frozen raw JForex evidence; no acquisition or fitting."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import struct
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import training_run_history as archive
from gold_jforex_offline_completion import RUN
import gold_gemini_dukascopy_jforex_final_reconciliation_v1 as study


def main() -> None:
    for name in ('metrics.json', 'report.md'):
        saved = RUN / ('pre_audit_' + name)
        if not saved.exists():
            shutil.copy2(RUN / name, saved)
    shutil.copy2(Path(__file__), RUN / 'independent_validator_script.py')
    metrics = archive.read_json(RUN / 'pre_audit_metrics.json')
    manifest = archive.read_json(RUN / 'manifest.json')
    requests = pd.read_csv(RUN / 'jforex_request_manifest.tsv', sep='\t')
    audit = pd.read_csv(RUN / 'jforex_request_audit.tsv', sep='\t', keep_default_na=False)
    ticks = pd.read_csv(RUN / 'jforex_ticks.tsv.gz', sep='\t', float_precision='round_trip')
    bars = pd.read_csv(RUN / 'jforex_bars.tsv.gz', sep='\t', float_precision='round_trip')
    tick_groups = {key: g for key, g in ticks.groupby(['request_id', 'mechanism'], sort=False)}
    bar_groups = {key: g for key, g in bars.groupby('request_id', sort=False)}
    checks = []
    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({'check': name, 'verdict': 'PASS' if passed else 'FAIL', 'evidence': evidence})

    errors = pd.read_csv(study.PARENT / 'tick_request_audit.csv')
    errors = errors[errors.state.eq('retrieval_error')]
    expected = {(r.instrument, pd.Timestamp(r.hour_utc).value // 1_000_000)
                for r in errors.itertuples()}
    actual = {(r.pair, r.from_ms) for r in requests.itertuples() if r.kind == 'retrieval_error_hour'}
    check('783 error-hour identities', len(expected) == 783 and expected == actual, len(actual))
    check('Complete mechanism inventory', len(audit) == 2352 and all(
        set(g.mechanism) == {'getTicks', 'readTicks', 'getBars'} and len(g) == 3
        for _, g in audit.groupby('request_id')), len(audit))
    bad_hashes = []
    bad_counts = []
    for r in audit.itertuples():
        h = hashlib.sha256()
        if r.mechanism == 'getBars':
            group = bar_groups.get(r.request_id, bars.iloc[:0])
            for x in group.itertuples():
                h.update(struct.pack('>qdddd', x.bar_time_ms, x.open, x.close, x.low, x.high))
        else:
            group = tick_groups.get((r.request_id, r.mechanism), ticks.iloc[:0]).sort_values('sequence')
            for x in group.itertuples():
                h.update(struct.pack('>qdddd', x.time_ms, x.bid, x.ask, x.bid_volume, x.ask_volume))
        if h.hexdigest() != r.raw_sha256:
            bad_hashes.append([r.request_id, r.mechanism])
        if len(group) != int(r.count):
            bad_counts.append([r.request_id, r.mechanism])
    check('Raw serialized hashes', not bad_hashes, bad_hashes)
    check('Raw serialized counts', not bad_counts, bad_counts)

    tick_fields = ['time_ms', 'bid', 'ask', 'bid_volume', 'ask_volume']
    pair_bars = {p: {} for p in study.INSTRUMENTS}
    per_request = {}
    difference_ids = []
    bounds_bad = []
    for req in requests.itertuples():
        g = tick_groups.get((req.request_id, 'getTicks'), ticks.iloc[:0]).sort_values('sequence')
        r = tick_groups.get((req.request_id, 'readTicks'), ticks.iloc[:0]).sort_values('sequence')
        same = g[tick_fields].reset_index(drop=True).equals(r[tick_fields].reset_index(drop=True))
        if not same:
            difference_ids.append(req.request_id)
        if ((g.time_ms < req.from_ms) | (g.time_ms > req.to_ms)).any():
            bounds_bad.append(req.request_id)
        # Preserve provider sequence for equal millisecond instants; never sort by BID.
        g = g.drop_duplicates(tick_fields).sort_values('time_ms', kind='stable')
        reconstructed = {}
        for minute, group in g.groupby((g.time_ms.astype('int64') // 60_000) * 60_000):
            if minute < req.from_ms or minute >= req.to_ms:
                continue
            values = group.bid.to_numpy()
            reconstructed[int(minute) * 1_000_000] = dict(zip(study.FIELDS, (
                float(values[0]), float(values.max()), float(values.min()), float(values[-1]))))
        per_request[req.request_id] = reconstructed
        if same:
            pair_bars[req.pair].update(reconstructed)
    check('getTicks/readTicks sequence equality', not difference_ids, difference_ids)
    check('Inclusive retrieval boundaries', not bounds_bad, bounds_bad)

    classes = pd.read_csv(RUN / 'three_way_arbitration.csv.gz', keep_default_na=False)
    b_bad = []
    for r in classes.itertuples():
        b = per_request[r.request_id].get(pd.Timestamp(r.minute_open_utc).value)
        saved = json.loads(r.b_ohlc) if r.b_ohlc else None
        if saved != b:
            b_bad.append([r.pair, r.minute_open_utc])
    check('Half-open M1 and chronological tie handling', not b_bad, b_bad[:20])

    log = (RUN / 'stdout.log').read_text(encoding='utf-8')
    failed_urls = sorted(set(re.findall(r'https://[^\s]+/([A-Z]{6})/(\d{4})/(\d{2})/(\d{2})/(\d{2})h_ticks\.bi5', log)))
    failed_hours = set()
    for code, year, month, day, hour in failed_urls:
        stamp = pd.Timestamp(year=int(year), month=int(month) + 1, day=int(day), hour=int(hour), tz='UTC')
        failed_hours.add((code[:3] + '/' + code[3:], stamp.value // 1_000_000))
    implicated = sorted(r.request_id for r in requests.itertuples()
                        if (r.pair, r.from_ms) in failed_hours)
    check('API success versus internal retrieval errors', not implicated, {
        'requested_hours_with_internal_download_errors': len(implicated),
        'request_ids': implicated,
        'reason': 'Returned success/empty data does not independently prove absence after SDK internal HTTP errors.',
    })
    # Exact row reconstruction uses the independently preserved parent validator.
    spec = importlib.util.spec_from_file_location('parent_independent', study.PARENT / 'validator_script_corrected.py')
    independent = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(independent)
    with np.load(RUN / 'exact_timestamps.npz', allow_pickle=False) as z:
        times = np.unique(np.concatenate([z[b + '_utc_ns'] for b in study.EXPECTED_TIMESTAMP_HASHES]))
        hashes = {b: independent.digest(z[b + '_broker_ns'].astype(np.int64))
                  for b in study.EXPECTED_TIMESTAMP_HASHES}
    check('Six GOLD timestamp hashes', hashes == study.EXPECTED_TIMESTAMP_HASHES, hashes)
    native = study.load_native()
    old_classes = pd.read_csv(study.PARENT / 'gap_minute_classification.csv.gz')
    all_gaps = pd.read_csv(study.SOURCE_PARENT / 'source_gap_audit.csv')
    closures = independent.closures(all_gaps, old_classes)
    _, old_unknown, _ = independent.matrix(times, native, closures)
    old_mask = old_unknown.any(axis=1)
    with np.load(RUN / 'preserved_defect_universe.npz', allow_pickle=False) as z:
        check('Exact 6931-row identity', int(old_mask.sum()) == 6931 and
              np.array_equal(times[old_mask], z['unresolved_utc_ns']), int(old_mask.sum()))
    for r in classes[classes.classification.eq('verified_no_quote_minute')].itertuples():
        m = pd.Timestamp(r.minute_open_utc).value
        closures[r.pair].append((m + study.MINUTE_NS, m + 2 * study.MINUTE_NS))
    closures = {p: independent.merge_intervals(v) for p, v in closures.items()}
    repaired = {p: {k: v.copy() for k, v in s.items()} for p, s in native.items()}
    repair_rows = classes[classes.classification.eq('retrieval_error_resolved_same_provider')]
    for p in study.INSTRUMENTS:
        additions = []
        for r in repair_rows[repair_rows.pair.eq(p)].itertuples():
            minute = pd.Timestamp(r.minute_open_utc).value
            additions.append((minute, json.loads(r.c_ohlc)))
        if additions:
            repaired[p]['open_utc_ns'] = np.concatenate([native[p]['open_utc_ns'], np.array([a[0] for a in additions], dtype=np.int64)])
            for field in study.FIELDS:
                repaired[p][field] = np.concatenate([native[p][field], np.array([a[1][field] for a in additions])])
            order = np.argsort(repaired[p]['open_utc_ns'], kind='stable')
            repaired[p] = {k: v[order] for k, v in repaired[p].items()}
    _, residual, _ = independent.matrix(times, repaired, closures)
    remaining = old_mask & residual.any(axis=1)
    added = ~old_mask & residual.any(axis=1)
    np.savez_compressed(RUN / 'independent_row_reconciliation.npz',
                        previous_unresolved_utc_ns=times[old_mask],
                        resolved_utc_ns=times[old_mask & ~remaining],
                        remaining_unresolved_utc_ns=times[remaining], new_unresolved_utc_ns=times[added])
    check('Independent row reconciliation', int(remaining.sum()) == metrics['remaining_unresolved_rows']
          and int(added.sum()) == metrics['new_unresolved_rows'], {
              'resolved': int(old_mask.sum() - remaining.sum()), 'remaining': int(remaining.sum()), 'new': int(added.sum())})

    dispositions = pd.read_csv(RUN / 'retrieval_error_disposition.csv')
    unresolved_ids = set(dispositions.loc[dispositions.disposition.isin([
        'continued_api_failure', 'inconsistent_provider_evidence', 'jforex_bar_only_evidence']), 'request_id'])
    impact_rows = []
    for req in requests.itertuples():
        if req.kind != 'retrieval_error_hour':
            continue
        # Conservative upper bound: any minute in [entry-65m, entry) could alter a source anchor.
        lo = np.searchsorted(times, req.from_ms * 1_000_000, side='right')
        hi = np.searchsorted(times, req.to_ms * 1_000_000 + 65 * study.MINUTE_NS, side='left')
        impact_rows.append({'request_id': req.request_id, 'pair': req.pair,
                           'unresolved': req.request_id in unresolved_ids,
                           'gold_rows_in_conservative_65m_envelope': int(hi-lo)})
    impact = pd.DataFrame(impact_rows)
    impact.to_csv(RUN / 'independent_hour_impact.csv', index=False)
    potentially_affecting = int((impact.unresolved & impact.gold_rows_in_conservative_65m_envelope.gt(0)).sum())
    check('Reported hour impact is supported', metrics['retrieval_error_hours_affecting_required_rows_remaining'] == potentially_affecting,
          {'reported': metrics['retrieval_error_hours_affecting_required_rows_remaining'],
           'conservative_upper_bound': potentially_affecting,
           'reason': 'Original counter counted gap hours, not verified feature impact; bound is not an exact causal count.'})
    mismatch = archive.read_json(RUN / 'gbpusd_single_mismatch_root_cause.json')
    check('Original tolerance preserved', mismatch['original_tolerance'] == 1e-5, 1e-5)
    a, b, c = (mismatch[k] for k in ('a_native_endpoint_ohlc', 'b_previous_tick_derived_ohlc', 'c_jforex_native_ohlc'))
    jf = per_request['MISMATCH_GBPUSD_20231210T2259Z'].get(study.MISMATCH_NS)
    check('Mismatch A/C same-provider arbitration', a == c and a != b,
          {'A': a, 'B': b, 'C': c, 'JForex_ticks': jf})
    check('Full previous HTTP mismatch window retained', False,
          'Frozen collector extracts only 22h archive; prescribed +/-2 minute window crosses into 23h. No new acquisition authorized in offline completion.')
    check('Prior finalized archives byte-identical', all(
        study.tree_hash(study.ROOT / 'training_runs' / name) == digest
        for name, digest in manifest['protected_runs_before'].items()), manifest['protected_runs_before'])
    check('Production unchanged', all(archive.file_sha256(study.ROOT / name) == value
          for name, value in manifest['operational_hashes_before'].items()), manifest['operational_hashes_before'])
    result = {'run_id': RUN.name, 'validator_internal_methodology': 'FAIL' if any(c['verdict']=='FAIL' for c in checks) else 'PASS',
              'validator_data_certification': 'FAIL', 'final_untouched_test_validity': 'NOT_APPLICABLE_DATA_ONLY',
              'checks': checks, 'internal_http_error_hours': len(implicated),
              'potentially_affecting_unresolved_hours_upper_bound': potentially_affecting,
              'no_new_acquisition': True, 'no_training_or_strategy_evaluation': True}
    archive.write_json(RUN / 'validator.json', result)
    lines = ['Overall: FAIL', '', 'Independent audit of the frozen data-only acquisition.', '',
             '| Check | Verdict | Evidence / required correction |', '|---|---|---|']
    for x in checks:
        lines.append('| ' + x['check'] + ' | ' + x['verdict'] + ' | ' + json.dumps(x['evidence'], ensure_ascii=True).replace('|', '/') + ' |')
    lines += ['', 'No strategy performance claim exists. Final untouched-test validity: not applicable.',
              'Data certification FAIL. Stop this foundation path under the requested stopping rule.',
              'Evidence is insufficient for zero-quote certification and full mismatch root-cause closure; no new data family or model is selected.']
    (RUN / 'validator.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    metrics.update({'run_status': 'fail', 'validator_internal_methodology': result['validator_internal_methodology'],
                    'validator_data_certification': 'FAIL', 'data_foundation_ready': False,
                    'family_testable_to_provenance_standard': False,
                    'retrieval_error_hours_affecting_required_rows_remaining': None,
                    'potentially_affecting_unresolved_hours_upper_bound': potentially_affecting,
                    'exact_hour_impact_status': 'not_certified; original 721 counter invalid',
                    'mismatch_root_cause_complete': False,
                    'equivalence_gate': 'FAIL_INCOMPLETE_ROOT_CAUSE_AUDIT',
                    'observed_A_C_arbitration': 'PASS',
                    'row_reconciliation_status': 'recomputed under submitted rules; not certified due acquisition assurance failures',
                    'internal_http_error_hours': len(implicated)})
    archive.write_json(RUN / 'metrics.json', metrics)
    manifest['registry']['validator_result'] = 'internal FAIL; data certification FAIL'
    data = manifest['data']
    data.update({'data_start_utc': '2016-06-30T19:59:00Z', 'data_end_utc': '2024-12-31T18:02:00Z',
                 'train_start_utc': 'not_applicable_data_only', 'train_end_utc': 'not_applicable_data_only',
                 'validation_start_utc': '2016-06-30T21:00:00Z', 'validation_end_utc': '2024-12-31T18:00:00Z',
                 'test_start_utc': 'not_applicable_data_only', 'test_end_utc': 'not_applicable_data_only'})
    manifest['offline_validation'] = {'commit': study.git('rev-parse','HEAD'),
                                    'script_sha256': archive.file_sha256(Path(__file__)),
                                    'raw_data_reacquired': False}
    manifest['artifacts'] = []
    archive.write_json(RUN / 'manifest.json', manifest)
    report = ['# JForex Final Reconciliation V1', '', 'Status: FAIL. DATA FOUNDATION READY = NO.', '',
              'The API returned 784 requests through all three mechanisms. Internal SDK HTTP errors make return-status-only absence certification unsafe.', '',
              'Original GBP/USD 2023-12-10 22:59 UTC CLOSE: native 1.25433, HTTP tick 1.25428, JForex native 1.25433. Original tolerance 0.00001 preserved.',
              'A/C arbitration supports the native representation; the full prescribed old-tick window and causal failure audit are incomplete.', '',
              'Provisional row replay resolves 15 of 6931 rows, leaving 6916. This is not certified source completeness.',
              'No final source or feature matrix was certified or generated.', '',
              'See validator.md for independent recomputation and rejected claims; pre_audit_metrics.json retains the original submitted result.', '',
              '```json', json.dumps(metrics, indent=2), '```', '',
              'Single next action: stop USD FX PRESSURE foundation research. A different external information family requires separate authorization.']
    (RUN / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
