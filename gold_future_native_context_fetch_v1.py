"""Direct native context acquisition, before freeze; never a strategy dataset."""
import calendar
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from gold_future_capture_collector_v2 import TIMEFRAMES, encode, sha, sealed_write, now
from gold_future_source_identity_v4 import session, verify_identity

ROOT = Path(__file__).resolve().parent
FIELDS = ['RAW_SOURCE_EPOCH', 'SOURCE_ID', 'SYMBOL', 'TIMEFRAME', 'OPEN',
          'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'TICK_VOLUME', 'REAL_VOLUME']


def close_epoch(raw, timeframe):
    if timeframe.startswith('M') and timeframe != 'Monthly':
        return raw + int(timeframe[1:])*60
    if timeframe.startswith('H'):
        return raw + int(timeframe[1:])*3600
    if timeframe == 'Daily':
        return raw + 86400
    if timeframe == 'Weekly':
        return raw + 604800
    stamp = datetime.fromtimestamp(raw, timezone.utc)
    if timeframe != 'Monthly' or (stamp.day, stamp.hour, stamp.minute, stamp.second) != (1, 0, 0, 0):
        raise ValueError('AMBIGUOUS_NATIVE_CALENDAR_BAR')
    return raw + calendar.monthrange(stamp.year, stamp.month)[1]*86400


def select_closed(rates, timeframe, source_now):
    required = 4096 if timeframe == 'M1' else 21
    epochs = [int(row['time']) for row in rates]
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError('DUPLICATE_OR_REVERSED_RAW_EPOCH')
    rows = []
    # Always exclude the newest record; additionally require nominal closure
    # and a successor at or beyond closure. No resampling or historical UTC claim.
    for row, successor in zip(rates[:-1], epochs[1:]):
        raw = int(row['time'])
        if raw <= 0 or raw % 60:
            raise ValueError('INVALID_RAW_EPOCH')
        closure = close_epoch(raw, timeframe)
        if closure > source_now or successor < closure:
            continue
        prices = [float(row[k]) for k in ('open', 'high', 'low', 'close')]
        if not all(math.isfinite(p) and p > 0 for p in prices) or prices[2] != min(prices) or prices[1] != max(prices):
            raise ValueError('MALFORMED_OHLC')
        quantities = [int(row[k]) for k in ('spread', 'tick_volume', 'real_volume')]
        if min(quantities) < 0:
            raise ValueError('INVALID_RAW_QUANTITY')
        rows.append(dict(zip(FIELDS, [raw, 'XMGlobal-MT5-6_GOLD', 'GOLD#', timeframe, *prices, *quantities])))
    if len(rows) < required:
        raise ValueError('INSUFFICIENT_CLOSED_NATIVE_CONTEXT:' + timeframe)
    return rows[-required:]


def fetch(destination, source_commit):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('CONTEXT_DESTINATION_ALREADY_EXISTS')
    destination.mkdir(parents=True)
    for name in ('raw', 'manifests', 'attestations', 'quarantine', 'pending'):
        (destination/name).mkdir()
    script_sha = sha(Path(__file__).read_bytes())
    blob = subprocess.check_output(['git', 'cat-file', 'blob', source_commit+':'+Path(__file__).name], cwd=ROOT)
    if blob.replace(b'\r\n', b'\n') != Path(__file__).read_bytes().replace(b'\r\n', b'\n'):
        raise ValueError('FETCHER_SOURCE_COMMIT_MISMATCH')
    entries = []
    with session() as (mt5, initial):
        for timeframe, constant in TIMEFRAMES.items():
            started = now()
            identity = verify_identity(mt5)
            if identity != initial:
                raise ValueError('SOURCE_CHANGED_BEFORE_CONTEXT_FETCH')
            requested = 4200 if timeframe == 'M1' else 25
            rates = mt5.copy_rates_from_pos('GOLD#', constant, 0, requested)
            tick = mt5.symbol_info_tick('GOLD#')
            if rates is None or tick is None:
                raise ValueError('NATIVE_CONTEXT_UNAVAILABLE:' + timeframe)
            source_now = int(tick.time)
            del tick
            observed_utc_epoch = time.time()
            if abs(source_now-10800-observed_utc_epoch) > 5:
                raise ValueError('CONTEXT_CURRENT_REGIME_OR_CLOCK_MISMATCH')
            rows = select_closed(rates, timeframe, source_now)
            returned_epochs = [int(row['time']) for row in rates]
            del rates
            if verify_identity(mt5) != initial:
                raise ValueError('SOURCE_CHANGED_DURING_CONTEXT_FETCH')
            finished = now()
            raw = encode(rows)
            relative = 'raw/'+timeframe+'.json'
            sealed_write(destination, destination/relative, raw)
            entry = {**identity, 'timeframe': timeframe, 'native_source_confirmed': True,
                     'resampled': False, 'mt5_timeframe_constant': constant,
                     'request_method': 'copy_rates_from_pos', 'request_start_pos': 0,
                     'requested_count': requested, 'returned_raw_epochs': returned_epochs,
                     'source_now_raw_epoch': source_now, 'row_count': len(rows),
                     'source_now_observed_utc_epoch': observed_utc_epoch,
                     'first_raw_epoch': rows[0]['RAW_SOURCE_EPOCH'], 'last_raw_epoch': rows[-1]['RAW_SOURCE_EPOCH'],
                     'first_normalized_timestamp': None, 'last_normalized_timestamp': None,
                     'historical_normalization': 'NOT_CLAIMED; raw native order and observed closure only',
                     'raw_path': relative, 'raw_sha256': sha(raw), 'schema_sha256': sha(encode(FIELDS)),
                     'fetch_started_at_utc': started, 'fetch_finished_at_utc': finished,
                     'closed_bar_rule': 'BAR_OPEN; exclude newest; nominal calendar closure <= current raw tick and successor',
                     'time_rule_version': 'RAW_HISTORICAL_CONTEXT_NO_UTC_OFFSET_INFERENCE_V1',
                     'script_sha256': script_sha, 'source_commit': source_commit,
                     'context_role': 'CAUSAL_CONTEXT_ONLY', 'holdout_evidence': False}
            sealed_write(destination, destination/'manifests'/(timeframe+'.json'), encode(entry))
            entries.append(entry)
    inventory = {'entries': entries, 'inventory_root_sha256': sha(encode(entries)),
                 'context_fetch_completed_at_utc': now(), 'context_role': 'CAUSAL_CONTEXT_ONLY',
                 'holdout_evidence': False, 'historically_equivalent': False, 'outcome_tuning': False,
                 'latest_context_raw_epoch': max(e['last_raw_epoch'] for e in entries),
                 'latest_context_source_timestamp': datetime.fromtimestamp(entries[0]['last_raw_epoch']-10800, timezone.utc).isoformat(),
                 'latest_timestamp_scope': 'Latest M1 endpoint only, under contemporaneously checked 10800 regime; older context rows remain unnormalized',
                 'causal_upper_bound_utc': max(e['fetch_finished_at_utc'] for e in entries),
                 'causal_bound_basis': 'All retained native bars closed before acquisition; no historical regime extrapolation'}
    sealed_write(destination, destination/'gold_future_native_context_inventory_v1.json', encode(inventory))
    manifests = [{'path': 'manifests/'+e['timeframe']+'.json',
                  'sha256': sha(encode(e))} for e in entries]
    sealed_write(destination, destination/'gold_future_native_context_manifest_v1.json', encode({
        'entries': manifests, 'manifests_root_sha256': sha(encode(manifests))}))
    return inventory
