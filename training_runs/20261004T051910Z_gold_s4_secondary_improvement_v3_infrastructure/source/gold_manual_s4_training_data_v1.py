"""Exact S4 export restoration; newly fetched nonidentical data is never eligible."""
import csv
import hashlib
import io
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from training_holdout_guard_v1 import check_path

ROOT = Path(__file__).resolve().parent
HEADER = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
TF = {**{x: x for x in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30',
                         'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12')},
      'Daily': 'D1', 'Weekly': 'W1', 'Monthly': 'MN1'}


def columns(item):
    return [x for x in HEADER if x != '<TIME>'] if item['timeframe'] in {'Daily', 'Weekly', 'Monthly'} else HEADER


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def validate_file(path, item, root=ROOT, scan=False):
    path = check_path(path, root)
    if (item['symbol'] != 'GOLD#' or item['timeframe'] not in TF
            or path.name != item['filename'] or sha(path) != item['sha256']):
        raise ValueError('Exact GOLD export identity mismatch: '+path.name)
    if item['last_timestamp'] > '2026-05-08T23:57:00':
        raise PermissionError('Post-cutoff historical source rejected')
    with path.open(encoding='utf-8', newline='') as stream:
        rows = csv.reader(stream, delimiter='\t')
        header = columns(item)
        if next(rows, None) != header:
            raise ValueError('Unapproved legacy schema')
        if not scan:
            return {'path': str(path), 'sha256': item['sha256'], 'training_eligible': True}
        first = previous = None
        count = 0
        for row in rows:
            if len(row) != len(header):
                raise ValueError('Malformed export row')
            if len(row) == 8:
                row = [row[0], '00:00:00', *row[1:]]
            stamp = row[0].replace('.', '-')+'T'+row[1]
            if previous is not None and stamp <= previous:
                raise ValueError('Duplicate or non-monotonic export timestamp')
            prices = [float(x) for x in row[2:6]]
            o, h, low, c = prices
            if (not all(math.isfinite(x) and x > 0 for x in prices)
                    or not low <= min(o, c) <= max(o, c) <= h
                    or any(int(x) < 0 for x in row[6:])):
                raise ValueError('Invalid OHLC/volume/spread')
            first = first or stamp
            previous = stamp
            count += 1
    if not count or first != item['first_timestamp'] or previous != item['last_timestamp']:
        raise ValueError('Export coverage mismatch')
    return {'path': str(path), 'sha256': item['sha256'], 'row_count': count,
            'first_timestamp': first, 'last_timestamp': previous, 'schema': header,
            'encoding': 'UTF-8 / ASCII', 'timestamp_semantics': 'LEGACY_BROKER_EXPORT_NAIVE',
            'classification': 'EXACT_PROVENANCE_TRAINING_SOURCE', 'training_eligible': True}


def fetch_original_export(item):
    """Attempt byte-exact repair, not UTC certification or a new training dataset.

    Original wall-clock strings are API query coordinates only. The original
    SHA256 is the acceptance criterion; no current offset is applied to history.
    """
    if item['symbol'] != 'GOLD#' or item['timeframe'] not in TF:
        raise ValueError('Exact symbol/native timeframe required')
    if item['last_timestamp'] > '2026-05-08T23:57:00':
        raise PermissionError('Historical cutoff exceeded')
    import MetaTrader5 as mt5
    if not mt5.initialize(path=r'D:\XM2\terminal64.exe', timeout=10000):
        raise ValueError('MT5 連線失敗')
    try:
        account = mt5.account_info()
        identity = {k: getattr(account, k, None) for k in ('company', 'server', 'trade_mode')}
        if identity != {'company': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'trade_mode': 0}:
            raise ValueError('MT5 source identity mismatch; no substitution')
        if mt5.symbol_info('GOLD#') is None or not mt5.symbol_select('GOLD#', True):
            raise ValueError('GOLD# unavailable; no symbol substitution')
        start = datetime.fromisoformat(item['first_timestamp']).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(item['last_timestamp']).replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_range('GOLD#', getattr(mt5, 'TIMEFRAME_'+TF[item['timeframe']]), start, end)
        if rates is None or len(rates) == 0:
            raise ValueError('MT5 無法取得指定歷史區間')
        out = io.StringIO(newline='')
        writer = csv.writer(out, delimiter='\t', lineterminator='\r\n')
        writer.writerow(columns(item))
        for row in rates:
            stamp = datetime.fromtimestamp(int(row['time']), timezone.utc)
            date_fields = [stamp.strftime('%Y.%m.%d')]
            if '<TIME>' in columns(item):
                date_fields.append(stamp.strftime('%H:%M:%S'))
            writer.writerow([*date_fields,
                             *[f"{float(row[k]):.2f}" for k in ('open', 'high', 'low', 'close')],
                             int(row['tick_volume']), int(row['real_volume']), int(row['spread'])])
        return out.getvalue().encode('utf-8')
    finally:
        mt5.shutdown()


def prepare(config, root=ROOT, fetch=fetch_original_export):
    root = Path(root).resolve()
    started = datetime.now(timezone.utc).isoformat()
    fetches = []
    if config['dataset_mode'] != 'REPRODUCTION_DATASET':
        raise ValueError('NEW_RETRAIN_DATASET requires a new approved version')
    items = config['required_datasets']
    identity = hashlib.sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    base = check_path(root/'historical_training_data'/'s4_exact'/identity, root)
    if not base.is_relative_to(root/'historical_training_data'):
        raise PermissionError('Cache escape')
    base.mkdir(parents=True, exist_ok=True)
    records = []
    for item in items:
        target = base/item['filename']
        if Path(item['filename']).name != item['filename']:
            raise PermissionError('Invalid dataset filename')
        candidates = [target, root/item['filename'], root/'historical_training_data/import'/item['filename']]
        existing = next((p for p in candidates if p.is_file() and sha(check_path(p, root)) == item['sha256']), None)
        if target.exists() and existing != target:
            raise ValueError('Existing sealed cache was modified: '+str(target))
        method = 'EXACT_LOCAL_OR_IMPORTED_BYTES'
        if existing is None:
            print('缺少歷史資料，正在從 MT5 嘗試精確還原...')
            print(f"Symbol: GOLD#\nTimeframe: {item['timeframe']}\nRange: {item['first_timestamp']} ~ {item['last_timestamp']}")
            try:
                raw = fetch(item)
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('Fetched NEW_RETRAIN_DATASET differs from original S4 export; not training eligible')
                with target.open('xb') as stream:
                    stream.write(raw)
                method = 'MT5_BYTE_EXACT_RESTORATION'
                fetches.append({'symbol': item['symbol'], 'timeframe': item['timeframe'],
                                'start': item['first_timestamp'], 'end': item['last_timestamp'],
                                'returned_rows': len(raw.splitlines())-1, 'sha256': item['sha256']})
            except Exception as error:
                raise ValueError('歷史資料無法精確還原：'+item['filename']+'\n'+str(error)
                                 +'\n請將原始且 hash 相符的 CSV 放入 '+str(root/'historical_training_data/import')
                                 +'\n不得以不同 MT5 資料冒稱重現。') from error
        elif existing != target:
            # Validate before/after copying; never modify an original export.
            validate_file(existing, item, root)
            with existing.open('rb') as src, target.open('xb') as dest:
                shutil.copyfileobj(src, dest)
        checked = validate_file(target, item, root)
        records.append({**item, **checked, 'fetch_method': method,
                        'retention_status': 'preserved_local_content_addressed_cache; not a remote raw-data backup'})
    if {p.name for p in base.glob('*.csv')} != {i['filename'] for i in items}:
        raise ValueError('Unexpected dataset in frozen input directory')
    return base, {'dataset_mode': 'REPRODUCTION_DATASET', 'source_id': config['training_source_id'],
                  'symbol': 'GOLD#', 'training_eligible': True, 'datasets': records,
                  'cutoff': config['raw_data_cutoff'], 'timestamp_semantics': 'LEGACY_BROKER_EXPORT_NAIVE',
                  'holdout_input': False, 'mt5_fetches': fetches,
                  'preparation_started_at_utc': started, 'preparation_finished_at_utc': datetime.now(timezone.utc).isoformat()}
