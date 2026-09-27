"""Immutable future M1 raw capture under the current-regime-only v4 policy."""
import math
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from gold_future_capture_collector_v2 import encode, sha, load, utc, now, require, sealed_write, writer_lock
from gold_future_current_regime_diagnostic_v1 import check_samples, observe
from gold_future_source_identity_v4 import session, verify_identity

ROOT = Path(__file__).resolve().parent
VERSION = 'gold_future_capture_collector_v4'
FIELDS = ['RAW_SOURCE_EPOCH', 'SOURCE_TIMESTAMP', 'SOURCE_ID', 'SYMBOL', 'TIMEFRAME',
          'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'TICK_VOLUME', 'REAL_VOLUME']


def binding(root):
    freeze = load(root/'protocol'/'freeze.json')
    activation = load(root/'attestations'/'activation.json')
    certificate = load(root/'attestations'/'recertification.json')
    require(activation['status'] == 'ACTIVE' and activation['collector_version'] == VERSION, 'Activation required')
    require(activation['protocol_freeze_commit'] and activation['protocol_freeze_sha256'] == sha(encode(freeze)), 'Freeze binding')
    blob = subprocess.check_output(['git', 'cat-file', 'blob', activation['protocol_freeze_commit']+':gold_future_locked_holdout_protocol_v4.json'], cwd=ROOT)
    require(blob == encode(freeze), 'Committed freeze bytes')
    require(utc(activation['activation_effective_at_utc']) > utc(freeze['protocol_freeze_effective_at_utc']), 'Activation must follow freeze')
    require(certificate['status'] == 'READY_FOR_CAPTURE_ACTIVATION' and certificate['freeze_sha256'] == sha(encode(freeze)), 'Frozen recertification required')
    require(freeze['certified_offsets_seconds'] == [10800] and freeze['uncertified_candidate_offsets_seconds'] == [7200]
            and freeze['runtime_fail_closed'] is True and freeze['prefix_approved'] is True, 'Frozen gates')
    require(freeze['outcome_inspection_prohibited'] is True, 'No-peeking gate')
    schema_path = ROOT/'gold_future_capture_manifest_schema_v4.json'
    require(sha(schema_path.read_bytes()) == freeze['manifest_schema_sha256'], 'Frozen schema changed')
    for name, expected in freeze['code_sha256'].items():
        require(Path(name).name == name and sha((ROOT/name).read_bytes()) == expected, 'Frozen code changed')
    inventory = load(root/'context_seed'/'gold_future_native_context_inventory_v1.json')
    require(inventory['inventory_root_sha256'] == freeze['context_inventory_root_sha256']
            == sha(encode(inventory['entries'])), 'Context binding')
    for entry in inventory['entries']:
        require(entry['raw_path'] == 'raw/'+entry['timeframe']+'.json', 'Context path')
        require(sha((root/'context_seed'/entry['raw_path']).read_bytes()) == entry['raw_sha256'], 'Context changed')
        require((root/'context_seed'/'manifests'/(entry['timeframe']+'.json')).read_bytes() == encode(entry), 'Context manifest changed')
    require(utc(inventory['context_fetch_completed_at_utc']) < utc(freeze['protocol_freeze_effective_at_utc']), 'Context precedes freeze')
    return freeze, activation


def validate_payload(payload, freeze, activation, previous=None):
    check_samples(payload['samples'], previous['samples'][-1] if previous else None)
    require(payload['native_source_confirmed'] is True and payload['resampled'] is False
            and payload['timeframe'] == 'M1' and payload['mt5_timeframe_constant'] == 1, 'Native M1 required')
    rows = payload['rows']
    require(rows, 'Empty evidence')
    epochs = [r['RAW_SOURCE_EPOCH'] for r in rows]
    require(all(type(e) is int and e % 60 == 0 for e in epochs) and all(b > a for a, b in zip(epochs, epochs[1:])), 'Raw epoch order')
    if previous:
        require(epochs[0] > previous['rows'][-1]['RAW_SOURCE_EPOCH'], 'Evidence replay')
    cutoff = max(utc(freeze['protocol_freeze_effective_at_utc']), utc(activation['activation_effective_at_utc']))
    returned = payload['returned_raw_epochs']
    require(all(type(e) is int and e % 60 == 0 for e in returned)
            and all(b > a for a, b in zip(returned, returned[1:])), 'Returned epoch order')
    for row in rows:
        require(set(row) == set(FIELDS), 'Raw-only schema')
        raw = row['RAW_SOURCE_EPOCH']
        stamp = datetime.fromtimestamp(raw-10800, timezone.utc)
        require(row['SOURCE_TIMESTAMP'] == stamp.isoformat() and stamp > cutoff, 'Strict future boundary')
        require(raw in returned[:-1] and raw+60 <= payload['samples'][-1]['raw_epoch'], 'Closed native evidence required')
        require(row['SOURCE_ID'] == 'XMGlobal-MT5-6_GOLD' and row['SYMBOL'] == 'GOLD#' and row['TIMEFRAME'] == 'M1', 'Row source')
        prices = [row[k] for k in ('OPEN', 'HIGH', 'LOW', 'CLOSE')]
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in prices)
                and prices[1] == max(prices) and prices[2] == min(prices), 'OHLC')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('SPREAD', 'TICK_VOLUME', 'REAL_VOLUME')), 'Raw quantities')
    return {'status': 'PASS', 'offset_seconds': 10800, 'row_count': len(rows),
            'first_source_timestamp': rows[0]['SOURCE_TIMESTAMP'], 'last_source_timestamp': rows[-1]['SOURCE_TIMESTAMP']}


def recover_chain(root):
    freeze, activation = binding(root)
    entries, previous = [], None
    paths = set()
    for number, path in enumerate(sorted((root/'manifests').iterdir())):
        require(path.name == f'{number:012d}.json' and not path.is_symlink(), 'Manifest sequence')
        item = load(path)
        schema = load(ROOT/'gold_future_capture_manifest_schema_v4.json')
        require(set(item) == set(schema['required']) and item['manifest_version'] == schema['version'], 'Manifest schema')
        require(item['sequence'] == number and item['previous_manifest_sha256'] == (sha(encode(entries[-1])) if entries else None), 'Chain hash')
        relative = f'snapshots/{number:012d}.json'
        require(item['snapshot_path'] == relative and item['activation_sha256'] == sha(encode(activation)), 'Manifest binding')
        raw = (root/relative).read_bytes()
        require(sha(raw) == item['snapshot_sha256'], 'Snapshot hash')
        payload = load(root/relative)
        result = validate_payload(payload, freeze, activation, previous)
        require(item['runtime_validation'] == result, 'Runtime verification')
        entries.append(item)
        previous = payload
        paths.add(relative)
    require({p.relative_to(root).as_posix() for p in (root/'snapshots').iterdir()} == paths, 'Orphan snapshot')
    return entries, previous


def pause(root, reason, payload=None):
    receipt = {'status': 'PAUSED_TIME_RULE', 'created_at_utc': now(), 'reason': reason,
               'recovery': 'Separate regime certification, independent validation and immutable amendment required; no auto-resume',
               'payload': payload}
    sealed_write(root, root/'quarantine'/(uuid.uuid4().hex+'.json'), encode(receipt))
    if not (root/'PAUSED_TIME_RULE.json').exists():
        sealed_write(root, root/'PAUSED_TIME_RULE.json', encode(receipt))


def append_snapshot(root, payload):
    with writer_lock(root):
        require(not (root/'PAUSED_TIME_RULE.json').exists(), 'Capture persistently paused')
        try:
            freeze, activation = binding(root)
            entries, previous = recover_chain(root)
            result = validate_payload(payload, freeze, activation, previous)
            number = len(entries)
            raw = encode(payload)
            item = {'manifest_version': 'gold_future_capture_manifest_v4', 'sequence': number,
                    'previous_manifest_sha256': sha(encode(entries[-1])) if entries else None,
                    'snapshot_path': f'snapshots/{number:012d}.json', 'snapshot_sha256': sha(raw),
                    'activation_sha256': sha(encode(activation)), 'runtime_validation': result,
                    'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#', 'sealed_at_utc': now()}
            sealed_write(root, root/item['snapshot_path'], raw)
            sealed_write(root, root/'manifests'/f'{number:012d}.json', encode(item))
            recover_chain(root)
            return item
        except (ValueError, KeyError, TypeError, OSError) as error:
            pause(root, type(error).__name__, payload)
            raise


def capture_once(root):
    root = Path(root).resolve()
    require(root == (ROOT/'future_holdout'/'gold_s4_v4').resolve(), 'Isolated v4 root required')
    require(not (root/'PAUSED_TIME_RULE.json').exists(), 'Capture persistently paused')
    payload = None
    try:
        freeze, activation = binding(root)
        entries, previous = recover_chain(root)
        with session() as (mt5, identity):
            samples = []
            for index in range(3):
                if index:
                    time.sleep(2)
                samples.append(observe(mt5))
            check_samples(samples, previous['samples'][-1] if previous else None)
            rates = mt5.copy_rates_from_pos('GOLD#', mt5.TIMEFRAME_M1, 0, 1000)
            require(rates is not None and len(rates) >= 2, 'Native M1 unavailable')
            require(verify_identity(mt5) == identity, 'Source changed during capture')
            cutoff = max(utc(freeze['protocol_freeze_effective_at_utc']).timestamp(),
                         utc(activation['activation_effective_at_utc']).timestamp())
            previous_raw = previous['rows'][-1]['RAW_SOURCE_EPOCH'] if previous else 0
            rows = []
            for row in rates[:-1]:
                raw = int(row['time'])
                if raw-10800 <= cutoff or raw <= previous_raw or raw+60 > samples[-1]['raw_epoch']:
                    continue
                stamp = datetime.fromtimestamp(raw-10800, timezone.utc).isoformat()
                rows.append(dict(zip(FIELDS, [raw, stamp, 'XMGlobal-MT5-6_GOLD', 'GOLD#', 'M1',
                    *[float(row[k]) for k in ('open', 'high', 'low', 'close')],
                    *[int(row[k]) for k in ('spread', 'tick_volume', 'real_volume')]])))
            payload = {'samples': samples, 'rows': rows, 'returned_raw_epochs': [int(r['time']) for r in rates],
                       'native_source_confirmed': True, 'resampled': False, 'timeframe': 'M1',
                       'mt5_timeframe_constant': 1, 'strategy_outcome_inspected': False}
            del rates
            if not rows:
                return None
            return append_snapshot(root, payload)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        if not (root/'PAUSED_TIME_RULE.json').exists():
            pause(root, str(error) if isinstance(error, ValueError) else type(error).__name__, payload)
        raise
