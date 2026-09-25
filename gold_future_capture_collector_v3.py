"""Empirical-time native capture fork; immutable v2 publication primitives retained."""
import argparse
import csv
import io
import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from gold_future_capture_collector_v2 import (encode, sha, load, utc, now, require,
    file_under, sealed_write, writer_lock, assess_tip, validate_schema, TIMEFRAMES)
from gold_future_empirical_time_v1 import identity, runtime_validate, normalize

ROOT = Path(__file__).resolve().parent
VERSION = 'gold_future_capture_collector_v3'
FIELDS = ['SOURCE_TIMESTAMP', 'ORIGINAL_SOURCE_TIMESTAMP', 'SOURCE_ID', 'SYMBOL',
          'TIMEFRAME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'TICK_VOLUME', 'REAL_VOLUME']


def binding(root):
    source, time_doc, activation = [load(file_under(root, n)) for n in
                                   ('source_attestation.json', 'time_attestation.json', 'activation.json')]
    identity(source)
    require(source['template'] is False and time_doc['template'] is False and activation['template'] is False, 'Incomplete binding')
    require(time_doc['status'] == 'EMPIRICALLY_CERTIFIED_SOURCE_TIME_RULE', 'Empirical time not certified')
    require(time_doc['policy'] == 'EMPIRICALLY_CERTIFIED_SOURCE_TIME_RULE' and time_doc['official_attestation_required'] is False, 'Time policy substitution')
    require(activation['status'] == 'ACTIVE_UNVERIFIED' and activation['collector_version'] == VERSION, 'Not activated')
    require(activation['higher_timeframe_policy'] == 'NATIVE_20TF_REQUIRED' and activation['prefix_policy'] == 'PREDECLARED_CAUSAL_REINITIALIZATION', 'Prefix/native policy')
    require(activation['source_attestation_sha256'] == sha(encode(source)) and activation['time_attestation_sha256'] == sha(encode(time_doc)), 'Binding hash')
    require(Path(activation['capture_root']).resolve() == root.resolve(), 'Capture root mismatch')
    require(activation['protocol_freeze_commit'] and activation['protocol_freeze_sha256'] == sha(file_under(root, 'protocol_freeze.json').read_bytes()), 'Freeze required')
    freeze = load(file_under(root, 'protocol_freeze.json'))
    require(freeze['prefix_approved'] is True and freeze['outcome_inspection_prohibited'] is True, 'Frozen gates')
    require(freeze['collector_sha256'] == sha(Path(__file__).read_bytes()) and freeze['runtime_validator_sha256'] == sha((ROOT/'gold_future_empirical_time_v1.py').read_bytes()), 'Frozen code binding')
    require(freeze['source_attestation_sha256'] == sha(encode(source)) and freeze['time_attestation_sha256'] == sha(encode(time_doc)), 'Freeze attestations')
    return source, time_doc, activation


def validate_payload(payload, prior):
    result = runtime_validate(payload, prior)
    rows = payload['rows']
    require(len(rows) == len(result['raw_epochs']), 'Runtime row count binding')
    for row, raw, stamp in zip(rows, result['raw_epochs'], result['normalized_timestamps']):
        require(set(row) == set(FIELDS), 'Raw schema; strategy columns rejected')
        require(row['ORIGINAL_SOURCE_TIMESTAMP'] == raw and row['SOURCE_TIMESTAMP'] == stamp, 'Raw/normalized binding')
        require(row['SOURCE_ID'] == payload['identity']['source_id'] and row['SYMBOL'] == 'GOLD#' and row['TIMEFRAME'] == payload['timeframe'], 'Row identity')
        prices = [row[k] for k in ('OPEN', 'HIGH', 'LOW', 'CLOSE')]
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in prices)
                and prices[2] <= min(prices) and prices[1] >= max(prices), 'Invalid raw OHLC')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ('SPREAD', 'TICK_VOLUME', 'REAL_VOLUME')), 'Raw numeric fields')
    return result


def recover_chain(root, schema):
    source, time_doc, activation = binding(root)
    entries = []; prior = {}; paths = set()
    for number, path in enumerate(sorted(file_under(root, 'manifests').iterdir())):
        require(path.name == f'{number:012d}.json', 'Manifest sequence gap')
        item = load(path); validate_schema(item, schema)
        require(item['previous_manifest_sha256'] == (sha(encode(entries[-1])) if entries else None), 'Previous hash')
        require(item['source_attestation_sha256'] == sha(encode(source)) and item['timezone_attestation_sha256'] == sha(encode(time_doc)) and item['activation_sha256'] == sha(encode(activation)), 'Manifest binding')
        raw = file_under(root, item['snapshot_path']).read_bytes()
        require(sha(raw) == item['raw_sha256'], 'Snapshot bytes changed')
        payload = json.loads(raw)
        result = validate_payload(payload, prior.get(item['timeframe']))
        require(item['runtime_validation_sha256'] == sha(encode(result)) and item['first_source_timestamp'] == result['normalized_timestamps'][0] and item['last_source_timestamp'] == result['normalized_timestamps'][-1], 'Runtime evidence binding')
        require(item['row_count'] == len(result['raw_epochs']) and item['timeframe'] == payload['timeframe'], 'Snapshot metadata')
        require(utc(item['first_source_timestamp']) > max(utc(activation['activation_effective_at_utc']), utc(activation['protocol_freeze_effective_at_utc'])), 'No backdated evidence')
        require(item['collector_version'] == VERSION and item['collector_commit'] == activation['collector_commit'], 'Collector identity')
        if entries:
            require(utc(item['capture_started_at_utc']) > utc(entries[-1]['capture_finished_at_utc']), 'Capture clock reversal')
        require(item['snapshot_path'] not in paths and item['sequence'] == number, 'Duplicate publication')
        paths.add(item['snapshot_path']); entries.append(item); prior[item['timeframe']] = result
    require(all(p.relative_to(root).as_posix() in paths for p in file_under(root, 'snapshots').iterdir()), 'Orphan snapshot; review required')
    tip_path = file_under(root, 'chain_tip.json')
    tip = load(tip_path) if tip_path.exists() else None
    return entries, prior, assess_tip(entries, tip)


def pause(root, payload, reason):
    receipt = {'status': 'PAUSED_TIME_RULE', 'created_at_utc': now(), 'reason': reason,
               'recovery': 'Re-certification and a new reviewed capture root required', 'rejected_payload': payload}
    sealed_write(root, file_under(root, 'quarantine')/(uuid.uuid4().hex+'.json'), encode(receipt))
    marker = file_under(root, 'PAUSED_TIME_RULE.json')
    if not marker.exists():
        sealed_write(root, marker, encode({'status': 'PAUSED_TIME_RULE', 'created_at_utc': now()}))


def publish_tip(root, schema):
    require(file_under(root, 'writer.lock').is_dir(), 'Writer lock required')
    entries, _, state = recover_chain(root, schema)
    require(entries, 'No chain')
    tip = {'manifest_sequence': len(entries)-1, 'manifest_sha256': sha(encode(entries[-1])),
           'updated_at_utc': entries[-1]['manifest_created_at_utc']}
    pending = file_under(root, 'pending')/(uuid.uuid4().hex+'.tip')
    with pending.open('xb') as f:
        f.write(encode(tip)); f.flush(); os.fsync(f.fileno())
    os.replace(pending, file_under(root, 'chain_tip.json'))


def append_snapshot(root, payload, schema, started, finished):
    with writer_lock(root):
        require(not file_under(root, 'PAUSED_TIME_RULE.json').exists(), 'Capture paused; no automatic resume')
        try:
            entries, prior, tip = recover_chain(root, schema)
            require(tip['readiness_pass'], 'Conflicting non-authoritative index')
            source, time_doc, activation = binding(root)
            result = validate_payload(payload, prior.get(payload['timeframe']))
            raw = encode(payload); number = len(entries)
            require(utc(result['normalized_timestamps'][0]) > max(utc(activation['activation_effective_at_utc']), utc(activation['protocol_freeze_effective_at_utc'])), 'Post-activation bar required')
            require(not entries or utc(started) > utc(entries[-1]['capture_finished_at_utc']), 'Capture clock reversal')
            item = {'manifest_version': 'gold_future_capture_manifest_v3', 'sequence': number, 'snapshot_sequence': number,
                    'previous_manifest_sha256': sha(encode(entries[-1])) if entries else None,
                    'manifest_created_at_utc': now(), 'snapshot_id': f'{number:012d}-'+uuid.uuid4().hex,
                    'snapshot_path': 'snapshots/'+f'{number:012d}.json', 'raw_sha256': sha(raw),
                    'schema_sha256': sha(encode({'fields': FIELDS, 'container': 'empirical_native_snapshot_v3'})),
                    'source_attestation_sha256': sha(encode(source)), 'timezone_attestation_sha256': sha(encode(time_doc)),
                    'activation_sha256': sha(encode(activation)), 'runtime_validation_sha256': sha(encode(result)),
                    'source_id': source['source_id'], 'symbol': 'GOLD#', 'timeframe': payload['timeframe'],
                    'capture_started_at_utc': started, 'capture_finished_at_utc': finished,
                    'first_source_timestamp': result['normalized_timestamps'][0], 'last_source_timestamp': result['normalized_timestamps'][-1],
                    'row_count': len(result['raw_epochs']), 'duplicate_timestamps': 0, 'non_monotonic_timestamps': 0,
                    'invalid_timestamp_rows': 0, 'invalid_price_rows': 0, 'spread_missing_rows': 0,
                    'spread_nonpositive_rows': sum(r['SPREAD'] == 0 for r in payload['rows']),
                    'collector_commit': activation['collector_commit'], 'collector_version': VERSION,
                    'revision_of_snapshot_id': None, 'partial_snapshot': False, 'sealed': True,
                    'notes': 'Actual native bars only; empirical runtime check; no outcomes'}
            validate_schema(item, schema)
        except (ValueError, KeyError, TypeError, OSError) as error:
            pause(root, payload, type(error).__name__)
            raise
        sealed_write(root, file_under(root, item['snapshot_path']), raw)
        sealed_write(root, file_under(root, 'manifests')/f'{number:012d}.json', encode(item))
        publish_tip(root, schema)
        recover_chain(root, schema)
        return item


def _capture_once(root, timeframe, count, certification):
    """Acquire actual source bars; callers cannot inject a fabricated identity envelope."""
    require(root.resolve().is_relative_to((ROOT/'future_holdout_capture').resolve()), 'Capture root restriction')
    protocol = load(ROOT/'gold_future_capture_protocol_v3.json')
    require(protocol['activation_permitted'] is True, 'Activation disabled')
    source, time_doc, activation = binding(root)
    report = load(certification)
    require(report['readiness']['status'] == 'READY_FOR_CAPTURE_ACTIVATION', 'Certification not ready')
    require(report['collector_static_review']['collector_sha256'] == sha(Path(__file__).read_bytes()), 'Collector hash')
    require(report['time_attestation_sha256'] == sha(encode(time_doc)), 'Time certificate binding')
    require(report['source_attestation_sha256'] == sha(encode(source)), 'Source certificate binding')
    require(report['freeze_sha256'] == activation['protocol_freeze_sha256'], 'Freeze recertification binding')
    require(timeframe in TIMEFRAMES and 1 <= count <= 1000, 'Bounded native request')
    blob = subprocess.check_output(['git', 'cat-file', 'blob', activation['collector_commit']+':gold_future_capture_collector_v3.py'], cwd=ROOT)
    require(Path(__file__).read_bytes().replace(b'\r\n', b'\n') == blob.replace(b'\r\n', b'\n'), 'Collector commit bytes')
    schema = load(ROOT/'gold_future_capture_manifest_schema_v3.json')
    require(sha(encode(schema)) == protocol['manifest_schema_sha256'], 'Schema hash')
    clock_raw = file_under(root, 'clock_certificate.json').read_bytes()
    clock = json.loads(clock_raw)
    require(sha(clock_raw) == activation['clock_certificate_sha256'], 'Clock certificate binding')
    require(clock['quality'] == 'INDEPENDENTLY_VERIFIED' and clock['max_measured_skew_seconds'] <= 5
            and utc(clock['valid_from']) <= utc(now()) < utc(clock['valid_until']), 'Clock quality/expiry')
    from gold_mt5_timestamp_semantics_diagnostic_v1 import session
    with session() as (mt5, observed):
        live_identity = {'source_id': source['source_id'], 'symbol': observed['symbol'], 'broker': observed['broker_company'],
                         'server': observed['broker_server'], 'environment': observed['source_account_environment']}
        identity(live_identity)
        started = now(); anchors = []
        for index in range(2):
            tick = mt5.symbol_info_tick('GOLD#')
            require(tick is not None, 'No current tick')
            anchors.append({'raw_epoch': int(tick.time), 'observed_utc_epoch': datetime.now(timezone.utc).timestamp()})
            del tick
            if index == 0:
                # At startup, cover at least one entire M1 interval. Longer native
                # bars outside this observed window remain blocked, never guessed.
                for _ in range(26):
                    time.sleep(5)
        rates = mt5.copy_rates_from_pos('GOLD#', TIMEFRAMES[timeframe], 0, count+1)
        require(rates is not None and len(rates) >= 2, 'No closed native bars')
        epochs = [int(r['time']) for r in rates]
        offset = activation['certified_offset_seconds']
        rows = []
        for r in rates[:-1]:
            rows.append(dict(zip(FIELDS, [normalize(int(r['time']), offset), int(r['time']), source['source_id'], 'GOLD#', timeframe,
                float(r['open']), float(r['high']), float(r['low']), float(r['close']), int(r['spread']), int(r['tick_volume']), int(r['real_volume'])])))
        del rates
        account = mt5.account_info()
        require(account is not None, 'Source identity unavailable after acquisition')
        final_identity = {'source_id': source['source_id'], 'symbol': 'GOLD#', 'broker': account.company,
                          'server': account.server, 'environment': 'demo' if account.trade_mode == 0 else 'live' if account.trade_mode == 2 else 'unknown'}
        del account
        identity(final_identity)
        payload = {'identity': live_identity, 'clock_quality': clock['quality'], 'timeframe': timeframe,
                   'native_timeframes': list(TIMEFRAMES), 'native_source': True, 'resampled': False,
                   'anchors': anchors, 'raw_epochs': epochs[:-1], 'successor_raw_epochs': epochs[1:],
                   'certified_offset_seconds': offset, 'regime_certification_id': activation['regime_certification_id'],
                   'transition_recertified': False, 'rows': rows}
        if utc(now()) >= utc(clock['valid_until']):
            pause(root, payload, 'Clock certificate expired during acquisition')
            raise ValueError('Clock expired')
        return append_snapshot(root, payload, schema, started, now())


def capture_once(root, timeframe, count, certification):
    require(root.resolve().is_relative_to((ROOT/'future_holdout_capture').resolve()), 'Capture root restriction')
    require(not file_under(root, 'PAUSED_TIME_RULE.json').exists(), 'Capture paused')
    try:
        return _capture_once(root, timeframe, count, certification)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        pause(root, {'acquisition_failed': True}, type(error).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--certification', type=Path, required=True)
    parser.add_argument('--timeframe', choices=TIMEFRAMES, default='M1')
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()
    capture_once(args.capture_root.resolve(), args.timeframe, args.count, args.certification)
