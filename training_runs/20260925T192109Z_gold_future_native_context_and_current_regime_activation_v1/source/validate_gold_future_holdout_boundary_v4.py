"""Independent raw-only v4 boundary verifier; no collector or strategy imports."""
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_future_native_context_v1 import load, digest, encode, validate, require, IDENTITY, sensitive_values

ROOT = Path(__file__).resolve().parent


def verify():
    root = ROOT/'future_holdout'/'gold_s4_v4'
    boundary = load(ROOT/'gold_future_holdout_boundary_v4.json')
    freeze = load(root/'protocol'/'freeze.json')
    activation = load(root/'attestations'/'activation.json')
    certificate = load(root/'attestations'/'recertification.json')
    require(not (root/'PAUSED_TIME_RULE.json').exists(), 'Runtime paused')
    for commit, path, expected in [(boundary['protocol_commit'], 'gold_future_locked_holdout_protocol_v4.json', encode(freeze)),
                                    (boundary['activation_commit'], 'gold_future_capture_activation_record_v4.json', encode(activation))]:
        blob = subprocess.check_output(['git', 'cat-file', 'blob', commit+':'+path], cwd=ROOT)
        require(blob == expected, 'Commit binding')
    require(boundary['protocol_commit'] == activation['protocol_freeze_commit'], 'Protocol commit')
    require(activation['status'] == 'ACTIVE' and activation['protocol_freeze_sha256'] == digest(encode(freeze)), 'Activation binding')
    require(certificate['status'] == 'READY_FOR_CAPTURE_ACTIVATION' and certificate['freeze_sha256'] == digest(encode(freeze)), 'Frozen certificate')
    require(freeze['certified_offsets_seconds'] == [10800] and freeze['uncertified_candidate_offsets_seconds'] == [7200], 'Current regime only')
    require(all(digest((ROOT/n).read_bytes()) == h for n, h in freeze['code_sha256'].items()), 'Code immutable')
    native = validate(root/'context_seed', freeze['source_commit'], digest((ROOT/'gold_future_native_context_fetch_v1.py').read_bytes()), boundary['holdout_start'])
    require(native['inventory_root_sha256'] == boundary['context_inventory_root_sha256'] == freeze['context_inventory_root_sha256'], 'Context root')
    inventory = load(root/'context_seed'/'gold_future_native_context_inventory_v1.json')
    stamp = datetime.fromisoformat(boundary['holdout_start'])
    require(datetime.fromisoformat(inventory['context_fetch_completed_at_utc']) < datetime.fromisoformat(freeze['protocol_freeze_effective_at_utc']) < datetime.fromisoformat(activation['activation_effective_at_utc']) < stamp, 'Strict temporal order')
    require(datetime.fromisoformat(inventory['latest_context_source_timestamp']) < stamp, 'Latest context precedes boundary')
    require(boundary['raw_epoch']-10800 == stamp.timestamp() and boundary['normalized_source_timestamp'] == boundary['holdout_start'], 'Boundary arithmetic')
    prefix = load(root/'protocol'/'prefix.json')
    require(digest(encode(prefix)) == boundary['prefix_protocol_sha256'] == freeze['prefix_protocol_sha256'] and prefix['approved'] is True, 'Prefix binding')
    previous_hash, last_raw, prior_tick = None, None, None
    first_row = None
    paths = set()
    manifests = sorted((root/'manifests').iterdir())
    require(bool(manifests), 'Missing evidence')
    for seq, path in enumerate(manifests):
        item = load(path)
        require(path.name == f'{seq:012d}.json' and item['sequence'] == seq and item['previous_manifest_sha256'] == previous_hash, 'Chain sequence/hash')
        require(item['activation_sha256'] == digest(encode(activation)), 'Chain activation binding')
        relative = f'snapshots/{seq:012d}.json'
        require(item['snapshot_path'] == relative, 'Snapshot path')
        raw = (root/relative).read_bytes()
        require(digest(raw) == item['snapshot_sha256'], 'Snapshot hash')
        payload = load(root/relative)
        require(not sensitive_values(payload) and payload['strategy_outcome_inspected'] is False, 'Raw-only capture')
        require(payload['native_source_confirmed'] is True and payload['resampled'] is False and payload['timeframe'] == 'M1' and payload['mt5_timeframe_constant'] == 1, 'Native capture')
        require(len(payload['samples']) == 3, 'Runtime observations')
        for sample in payload['samples']:
            require(all(sample['source_identity'].get(k) == v for k, v in IDENTITY.items()), 'Runtime identity')
            tick = sample['raw_epoch']; observed = sample['observed_utc_epoch']
            require(abs(tick-observed-10800) <= 5 and (prior_tick is None or tick > prior_tick), 'Runtime regime/skew/monotonicity')
            require(sample['observed_offset_seconds'] == 10800 and sample['normalized_clock_error_seconds'] == tick-10800-observed, 'Runtime arithmetic')
            epochs = sample['m1_raw_epochs']
            require(len(epochs) == 3 and all(e % 60 == 0 for e in epochs) and all(b-a == 60 for a, b in zip(epochs, epochs[1:])), 'Runtime M1 spacing')
            prior_tick = tick
        for row in payload['rows']:
            require(set(row) == {'RAW_SOURCE_EPOCH', 'SOURCE_TIMESTAMP', 'SOURCE_ID', 'SYMBOL', 'TIMEFRAME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'TICK_VOLUME', 'REAL_VOLUME'}, 'No strategy columns')
            epoch = row['RAW_SOURCE_EPOCH']
            require(epoch % 60 == 0 and (last_raw is None or epoch > last_raw), 'Raw ordering')
            require(epoch in payload['returned_raw_epochs'][:-1] and epoch+60 <= payload['samples'][-1]['raw_epoch'], 'Closed native bar')
            require(row['SOURCE_TIMESTAMP'] == datetime.fromtimestamp(epoch-10800, timezone.utc).isoformat() and datetime.fromisoformat(row['SOURCE_TIMESTAMP']) >= stamp, 'Normalized evidence')
            require(row['SYMBOL'] == 'GOLD#' and row['SOURCE_ID'] == IDENTITY['source_id'] and row['TIMEFRAME'] == 'M1', 'Row identity')
            prices = [row[k] for k in ('OPEN', 'HIGH', 'LOW', 'CLOSE')]
            require(all(math.isfinite(p) and p > 0 for p in prices) and prices[1] == max(prices) and prices[2] == min(prices), 'OHLC')
            first_row = first_row or row
            last_raw = epoch
        expected = {'status': 'PASS', 'offset_seconds': 10800, 'row_count': len(payload['rows']), 'first_source_timestamp': payload['rows'][0]['SOURCE_TIMESTAMP'], 'last_source_timestamp': payload['rows'][-1]['SOURCE_TIMESTAMP']}
        require(item['runtime_validation'] == expected, 'Runtime result binding')
        if seq == 0:
            require(boundary['first_manifest_sequence'] == 0 and boundary['first_manifest_sha256'] == digest(path.read_bytes()) and boundary['snapshot_sha256'] == digest(raw), 'First boundary binding')
        previous_hash = digest(path.read_bytes())
        paths.add(relative)
    require(paths == {p.relative_to(root).as_posix() for p in (root/'snapshots').iterdir()}, 'Orphan snapshot')
    require(first_row['RAW_SOURCE_EPOCH'] == boundary['raw_epoch'], 'First eligible evidence')
    require(boundary['holdout_evaluation_start'] is None, 'No unverified feature-readiness claim')
    require(boundary['historical_context_only'] is True and boundary['historical_rows_holdout_evidence'] is False and boundary['strategy_outcome_inspected'] is False, 'No leakage/no peeking')
    return {'overall': 'PASS', 'holdout_start': boundary['holdout_start'], 'holdout_evaluation_start': None,
            'feature_readiness': 'NOT_CLAIMED', 'strategy_outcome_inspected': False}


if __name__ == '__main__':
    path = ROOT/'future_holdout'/'gold_s4_v4'/'attestations'/'boundary_validator.json'
    with path.open('xb') as output:
        try:
            result = verify()
        except Exception as error:
            result = {'overall': 'FAIL', 'error_type': type(error).__name__, 'reason': str(error) if isinstance(error, ValueError) else 'Unexpected verifier error'}
        output.write(encode(result))
    print(json.dumps(result))
    raise SystemExit(0 if result['overall'] == 'PASS' else 1)
