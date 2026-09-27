"""Conservative data primitives; no approved real-data requirements in v1.

Coverage needs independently certified expected bar timestamps. Endpoints alone
cannot prove completeness across market closures. No model libraries are imported.
"""
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from training_holdout_guard_v1 import check_path, check_interval

ROOT = Path(__file__).resolve().parent
COLUMNS = ('time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume')
SOURCE_TYPES = {'MT5_NATIVE_API', 'LEGACY_CERTIFIED_FILE',
                'USER_IMPORTED_CERTIFIED_FILE', 'DERIVED_APPROVED', 'UNKNOWN'}
TIMEFRAMES = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15',
              'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12',
              'D1', 'W1', 'MN1')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def requirement(req):
    if (req.get('symbol') not in {'GOLD#', 'GAUCNH#'}
            or req.get('timeframe') not in TIMEFRAMES
            or req.get('source_id') != 'XMGlobal-MT5-6_' + req['symbol'].rstrip('#')
            or req.get('native_or_derived') != 'NATIVE'
            or req.get('timestamp_semantics') != 'UTC'
            or req.get('coverage_certified') is not True):
        raise ValueError('Exact symbol/source/native timeframe/UTC coverage certification required')
    check_interval(req['start'], req['end'], req['cutoff'])
    stamps = req['expected_timestamps']
    if (not stamps or any(type(t) is not int for t in stamps)
            or stamps != sorted(set(stamps))
            or stamps[0] != req['start'] or stamps[-1] != req['end']):
        raise ValueError('Explicit complete expected timestamp inventory required')
    if not isinstance(req.get('api_timestamp_offset_seconds'), int):
        raise ValueError('Historical API timestamp mapping required')


def validate_rows(rows, req, metadata, complete=False):
    requirement(req)
    for key in ('symbol', 'source_id', 'timeframe', 'native_or_derived', 'timestamp_semantics'):
        if metadata.get(key) != req[key]:
            raise ValueError('Dataset identity mismatch: ' + key)
    if metadata.get('source_type') not in SOURCE_TYPES - {'UNKNOWN', 'DERIVED_APPROVED'}:
        raise ValueError('Uncertified source is not eligible')
    expected = set(req['expected_timestamps'])
    previous = None
    for row in rows:
        if set(row) != set(COLUMNS):
            raise ValueError('Malformed schema')
        t = row['time']
        if type(t) is not int or t not in expected or (previous is not None and t <= previous):
            raise ValueError('Duplicate, unordered, out-of-range or non-native timestamp')
        prices = [row[k] for k in ('open', 'high', 'low', 'close')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in prices):
            raise ValueError('Invalid OHLC')
        if not row['low'] <= min(row['open'], row['close']) <= max(row['open'], row['close']) <= row['high']:
            raise ValueError('Invalid OHLC ordering')
        if any(type(row[k]) is not int or row[k] < 0 for k in ('spread', 'tick_volume', 'real_volume')):
            raise ValueError('Invalid volume/spread')
        previous = t
    missing = sorted(expected - {row['time'] for row in rows})
    if complete and missing:
        raise ValueError('Incomplete historical data')
    return missing


def intervals(missing, expected):
    """Group adjacent missing slots in the certified inventory, not wall time."""
    result = []
    missing = set(missing)
    start = end = None
    for stamp in expected:
        if stamp in missing:
            if start is None:
                start = stamp
            end = stamp
        elif start is not None:
            result.append((start, end))
            start = end = None
    if start is not None:
        result.append((start, end))
    return result


def read_csv(path, req, metadata, root=ROOT):
    path = check_path(path, root)
    if metadata.get('sha256') != sha(path.read_bytes()):
        raise ValueError('CSV hash does not match certified sidecar')
    # Canonical schema only. MT5 exports need explicit reviewed column/time mapping.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError('Unapproved CSV schema/encoding; no inferred column mapping')
        rows = []
        for item in reader:
            if set(item) != set(COLUMNS) or None in item.values():
                raise ValueError('Malformed CSV row')
            rows.append({k: float(item[k]) if k in COLUMNS[1:5] else int(item[k]) for k in COLUMNS})
    validate_rows(rows, req, metadata)
    return rows


def mt5_fetch(req, start, end):
    """Read-only native fetch; caller must have reviewed historical clock mapping."""
    requirement(req)
    check_interval(start, end, req['cutoff'])
    import MetaTrader5 as mt5
    if not mt5.initialize(path=r'D:\XM2\terminal64.exe', timeout=10000):
        raise ValueError('MT5 連線失敗')
    try:
        account = mt5.account_info()
        identity = {k: getattr(account, k, None) for k in ('company', 'server', 'trade_mode')}
        if identity != req['broker_identity']:
            raise ValueError('MT5 source identity mismatch')
        if mt5.symbol_info(req['symbol']) is None or not mt5.symbol_select(req['symbol'], True):
            raise ValueError('Exact MT5 symbol unavailable: ' + req['symbol'])
        offset = req['api_timestamp_offset_seconds']
        raw = mt5.copy_rates_range(req['symbol'], getattr(mt5, 'TIMEFRAME_' + req['timeframe']),
                                   datetime.fromtimestamp(start + offset, timezone.utc),
                                   datetime.fromtimestamp(end + offset, timezone.utc))
        if raw is None or len(raw) == 0:
            raise ValueError('MT5 無法取得這段資料')
        rows = [{k: float(row[k]) if k in COLUMNS[1:5] else int(row[k]) for k in COLUMNS} for row in raw]
        for row in rows:
            row['time'] -= offset
        return rows
    finally:
        mt5.shutdown()


def ensure_dataset(req, rows, metadata, cache, fetch=mt5_fetch, root=ROOT):
    """Validate before/after fetch and seal new immutable cache files exclusively."""
    missing = validate_rows(rows, req, metadata)
    if not missing:
        return rows, {'status': 'READY', 'fetch_attempted': False}
    combined = list(rows)
    fetched = []
    for start, end in intervals(missing, req['expected_timestamps']):
        print('缺少歷史資料，正在從 MT5 自動補抓...')
        print(f"Symbol: {req['symbol']}\nTimeframe: {req['timeframe']}\nRange: {start} ~ {end}")
        try:
            part = fetch(req, start, end)
            if not part:
                raise ValueError('Empty historical range')
            validate_rows(part, req, {**metadata, 'source_type': 'MT5_NATIVE_API'})
            if any(row['time'] < start or row['time'] > end for row in part):
                raise ValueError('MT5 returned bars outside requested range')
            combined.extend(part)
            fetched.append({'start': start, 'end': end, 'row_count': len(part)})
        except Exception as exc:
            raise ValueError(f"訓練無法開始：缺少歷史資料\nSymbol: {req['symbol']}\n"
                             f"Timeframe: {req['timeframe']}\n缺少區間: {start} ~ {end}\n"
                             f"MT5 無法取得這段資料。\n如有已認證 CSV，可放入：{Path(root)/'historical_training_data/import'}\n"
                             '完成後再次雙擊 RUN_TRAINING.bat') from exc
    combined.sort(key=lambda row: row['time'])
    validate_rows(combined, req, metadata, complete=True)
    cache = check_path(cache, root)
    if not cache.is_relative_to((Path(root)/'historical_training_data').resolve()):
        raise ValueError('Cache must stay in dedicated training directory')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(combined)
    raw = stream.getvalue().encode('utf-8')
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache/(sha(raw) + '.csv')
    with dest.open('xb') as out:
        out.write(raw)
    sealed = {**metadata, 'path': str(dest), 'sha256': sha(raw),
              'first_timestamp': combined[0]['time'], 'last_timestamp': combined[-1]['time'],
              'row_count': len(combined), 'schema_sha256': sha(','.join(COLUMNS).encode()),
              'fetch_method': 'copy_rates_range', 'fetches': fetched,
              'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_server': req['broker_identity']['server'],
              'training_eligible': True, 'blockers': [], 'status': 'READY',
              'fetch_attempted': True, 'input_source_type': metadata['source_type'],
              'source_type': 'MT5_NATIVE_API' if not rows else metadata['source_type']}
    with dest.with_suffix('.manifest.json').open('x', encoding='utf-8') as out:
        json.dump(sealed, out, indent=2)
    return combined, sealed
