"""Independent context reader: no fetcher or MT5 import."""
import calendar
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

TIMEFRAMES = {'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5, 'M6': 6,
              'M10': 10, 'M12': 12, 'M15': 15, 'M20': 20, 'M30': 30,
              'H1': 16385, 'H2': 16386, 'H3': 16387, 'H4': 16388,
              'H6': 16390, 'H8': 16392, 'H12': 16396,
              'Daily': 16408, 'Weekly': 32769, 'Monthly': 49153}
FIELDS = ['RAW_SOURCE_EPOCH', 'SOURCE_ID', 'SYMBOL', 'TIMEFRAME', 'OPEN',
          'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'TICK_VOLUME', 'REAL_VOLUME']
IDENTITY = {'broker_company': 'XM Global Limited', 'broker_server': 'XMGlobal-MT5 6',
            'source_account_environment': 'demo', 'source_id': 'XMGlobal-MT5-6_GOLD',
            'symbol': 'GOLD#'}
SENSITIVE = {'login', 'name', 'account_number', 'balance', 'equity', 'margin',
             'margin_free', 'margin_level', 'profit', 'credit', 'leverage',
             'currency', 'password', 'token', 'credentials', 'account_info'}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sensitive_values(value):
    if isinstance(value, dict):
        return any(str(k).lower() in SENSITIVE or sensitive_values(v) for k, v in value.items())
    return isinstance(value, list) and any(sensitive_values(v) for v in value)


def validate(root, source_commit, script_sha, boundary=None):
    root = Path(root)
    inventory = load(root/'gold_future_native_context_inventory_v1.json')
    entries = inventory['entries']
    require([e['timeframe'] for e in entries] == list(TIMEFRAMES), 'Exactly 21 canonical native timeframes required')
    require(inventory['inventory_root_sha256'] == digest(encode(entries)), 'Inventory root hash')
    require(not sensitive_values(inventory), 'Sensitive account fields')
    require(inventory['context_role'] == 'CAUSAL_CONTEXT_ONLY' and inventory['holdout_evidence'] is False
            and inventory['historically_equivalent'] is False and inventory['outcome_tuning'] is False, 'Context role')
    manifest = load(root/'gold_future_native_context_manifest_v1.json')
    expected_manifests = []
    for entry in entries:
        tf = entry['timeframe']
        require(all(entry.get(k) == v for k, v in IDENTITY.items()), 'Source identity')
        require(entry['identity_check_passed'] is True and entry['native_source_confirmed'] is True
                and entry['resampled'] is False and entry['transport'] == 'MetaTrader5 Python API', 'Native source')
        require(entry['source_commit'] == source_commit and entry['script_sha256'] == script_sha, 'Source binding')
        require(entry['mt5_timeframe_constant'] == TIMEFRAMES[tf] and entry['request_method'] == 'copy_rates_from_pos'
                and entry['request_start_pos'] == 0 and entry['requested_count'] == (4200 if tf == 'M1' else 25), 'Native API request')
        require(entry['context_role'] == 'CAUSAL_CONTEXT_ONLY' and entry['holdout_evidence'] is False, 'Entry context role')
        require(entry['raw_path'] == 'raw/'+tf+'.json', 'Raw path')
        path = root/entry['raw_path']
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Raw path escape')
        raw = path.read_bytes()
        rows = load(path)
        require(raw == encode(rows) and digest(raw) == entry['raw_sha256'], 'Raw bytes/hash')
        require(entry['schema_sha256'] == digest(encode(FIELDS)), 'Raw schema hash')
        require(len(rows) == entry['row_count'] == (4096 if tf == 'M1' else 21), 'Retained row count')
        returned = entry['returned_raw_epochs']
        require(abs(entry['source_now_raw_epoch']-10800-entry['source_now_observed_utc_epoch']) <= 5, 'Current acquisition clock')
        require(2 <= len(returned) <= entry['requested_count'] and all(type(e) is int and e % 60 == 0 for e in returned)
                and all(b > a for a, b in zip(returned, returned[1:])), 'Returned epochs')
        accepted = []
        for raw_epoch, successor in zip(returned, returned[1:]):
            if tf == 'Monthly':
                dt = datetime.fromtimestamp(raw_epoch, timezone.utc)
                require((dt.day, dt.hour, dt.minute, dt.second) == (1, 0, 0, 0), 'Monthly boundary')
                duration = calendar.monthrange(dt.year, dt.month)[1]*86400
            elif tf in ('Daily', 'Weekly'):
                duration = 86400 if tf == 'Daily' else 604800
            else:
                duration = int(tf[1:])*(60 if tf[0] == 'M' else 3600)
            if raw_epoch+duration <= entry['source_now_raw_epoch'] and successor >= raw_epoch+duration:
                accepted.append(raw_epoch)
        epochs = [row['RAW_SOURCE_EPOCH'] for row in rows]
        require(epochs == accepted[-len(rows):] and epochs[0] == entry['first_raw_epoch'] and epochs[-1] == entry['last_raw_epoch'], 'Closed native selection')
        for row in rows:
            require(set(row) == set(FIELDS), 'Unexpected raw fields')
            require(row['SOURCE_ID'] == IDENTITY['source_id'] and row['SYMBOL'] == 'GOLD#' and row['TIMEFRAME'] == tf, 'Raw identity')
            prices = [row[k] for k in ('OPEN', 'HIGH', 'LOW', 'CLOSE')]
            require(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in prices)
                    and prices[1] == max(prices) and prices[2] == min(prices), 'OHLC invalid')
            require(all(type(row[k]) is int and row[k] >= 0 for k in ('SPREAD', 'TICK_VOLUME', 'REAL_VOLUME')), 'Quantity invalid')
        started, finished = (datetime.fromisoformat(entry[k]) for k in ('fetch_started_at_utc', 'fetch_finished_at_utc'))
        require(started.utcoffset().total_seconds() == 0 and finished.utcoffset().total_seconds() == 0 and started <= finished, 'Fetch UTC order')
        if boundary is not None:
            require(finished < datetime.fromisoformat(boundary), 'Post-boundary context')
        relative = 'manifests/'+tf+'.json'
        require((root/relative).read_bytes() == encode(entry), 'Per-timeframe manifest')
        expected_manifests.append({'path': relative, 'sha256': digest(encode(entry))})
    require(manifest['entries'] == expected_manifests and manifest['manifests_root_sha256'] == digest(encode(expected_manifests)), 'Manifest root')
    require(inventory['causal_upper_bound_utc'] == max(e['fetch_finished_at_utc'] for e in entries), 'Causal bound')
    require(inventory['latest_context_source_timestamp'] == datetime.fromtimestamp(entries[0]['last_raw_epoch']-10800, timezone.utc).isoformat(), 'Latest context endpoint')
    return {'status': 'PASS', 'timeframe_count': 21, 'native_timeframe_count': 21,
            'm1_context_bars': 4096, 'htf_required_bars_each': 21,
            'inventory_root_sha256': inventory['inventory_root_sha256'],
            'manifests_root_sha256': manifest['manifests_root_sha256'],
            'sensitive_account_values_present': False}
