"""Independent eligibility verifier; no eligibility-runner import."""
import ast
import calendar
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout'/'gold_s4_v4'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def verify(binding, result):
    freeze = load(CAPTURE/'protocol'/'freeze.json')
    boundary = load(ROOT/'gold_future_holdout_boundary_v4.json')
    for key, name in [('protocol_commit', 'gold_future_locked_holdout_protocol_v4.json'),
                      ('activation_commit', 'gold_future_capture_activation_record_v4.json'),
                      ('boundary_commit', 'gold_future_holdout_boundary_v4.json')]:
        blob = subprocess.check_output(['git', 'cat-file', 'blob', result[key]+':'+name], cwd=ROOT)
        if blob != (ROOT/name).read_bytes():
            raise ValueError('Frozen commit bytes changed')
    raw = (ROOT/'drl_trading_v2.py').read_bytes()
    tree = ast.parse(raw)
    indicator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators')
    assert binding['feature_pipeline_sha256'] == sha(raw) == freeze['prefix_protocol']['pipeline_sha256']
    assert binding['feature_function_sha256'] == sha(ast.dump(indicator, include_attributes=False).encode()) == freeze['feature_implementation_sha256']
    assert binding['feature_list'] == freeze['feature_names'] and len(binding['feature_list']) == 31
    assert binding['feature_list_sha256'] == sha(encode(freeze['feature_names'])) == result['feature_list_sha256']
    assert binding['initialization_semantics'] == freeze['prefix_protocol']['initialization_semantics']
    pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_and_prepare_data')
    first_statement = next(i for i, n in enumerate(pipeline.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'm1')
    label_boundary = next(i for i, n in enumerate(pipeline.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prices')
    statements = pipeline.body[first_statement:label_boundary]
    assert binding['feature_construction_ast_sha256'] == sha(ast.dump(ast.Module(body=statements, type_ignores=[]), include_attributes=False).encode())
    assert binding['htf_implementation_sha256'] == sha(ast.dump(next(n for n in statements if isinstance(n, ast.For)), include_attributes=False).encode())
    inventory = load(CAPTURE/'context_seed'/'gold_future_native_context_inventory_v1.json')
    assert inventory['inventory_root_sha256'] == result['context_inventory_root_sha256'] == sha(encode(inventory['entries']))
    seed = {}
    for entry in inventory['entries']:
        raw = (CAPTURE/'context_seed'/entry['raw_path']).read_bytes()
        assert sha(raw) == entry['raw_sha256'] and entry['native_source_confirmed'] is True and entry['resampled'] is False
        assert datetime.fromisoformat(entry['fetch_finished_at_utc']) < datetime.fromisoformat(boundary['holdout_start'])
        seed[entry['timeframe']] = json.loads(raw)
    rows = []
    previous = None
    for number, entry in enumerate(result['chain_view']):
        manifest_path = CAPTURE/'manifests'/f'{number:012d}.json'
        manifest = load(manifest_path)
        assert sha(manifest_path.read_bytes()) == entry['manifest_sha256']
        assert manifest['previous_manifest_sha256'] == previous and manifest['sequence'] == entry['sequence'] == number
        snapshot = (CAPTURE/entry['snapshot_path']).read_bytes()
        assert sha(snapshot) == entry['snapshot_sha256'] == manifest['snapshot_sha256']
        payload = json.loads(snapshot)
        assert payload['native_source_confirmed'] is True and payload['resampled'] is False
        rows.extend(payload['rows'])
        previous = entry['manifest_sha256']
    assert len(result['candidate_checks']) == len(rows)
    earliest = None
    for index, (candidate, check) in enumerate(zip(rows, result['candidate_checks'])):
        candidate_raw = candidate['RAW_SOURCE_EPOCH']
        candidate_time = candidate['SOURCE_TIMESTAMP']
        assert candidate_time == datetime.fromtimestamp(candidate_raw-10800, timezone.utc).isoformat()
        assert candidate_time == check['candidate_timestamp'] and candidate_time >= result['holdout_start']
        epochs = [seed['M1'][-1]['RAW_SOURCE_EPOCH'], *[r['RAW_SOURCE_EPOCH'] for r in rows[:index+1]]]
        contiguous = all(b-a == 60 for a, b in zip(epochs, epochs[1:]))
        missing_htf = []
        for entry in inventory['entries'][1:]:
            tf = entry['timeframe']
            for epoch in entry['returned_raw_epochs']:
                if tf == 'Monthly':
                    stamp = datetime.fromtimestamp(epoch, timezone.utc)
                    duration = calendar.monthrange(stamp.year, stamp.month)[1]*86400
                else:
                    duration = 86400 if tf == 'Daily' else 604800 if tf == 'Weekly' else int(tf[1:])*(60 if tf[0] == 'M' else 3600)
                if epoch > entry['last_raw_epoch'] and epoch+duration <= candidate_raw:
                    missing_htf.append(tf)
                    break
        assert check['m1_state_ready'] == contiguous and check['recursive_state_ready'] == contiguous
        assert check['missing_closed_native_timeframes'] == missing_htf and check['htf_state_ready'] == (not missing_htf)
        assert check['max_input_raw_epoch'] == candidate_raw and check['lookahead_detected'] is False
        if not contiguous or missing_htf:
            assert check['eligibility'] == 'NOT_ELIGIBLE_INSUFFICIENT_CONTEXT'
            assert check['missing_features'] == binding['feature_list'] and check['feature_count_available'] == 0
            assert check['feature_values_computed'] is False and check['nan_count'] == check['inf_count'] == 0
        else:
            import numpy as np
            import pandas as pd
            assert pd.__version__ == binding['initialization_semantics']['pandas_version']
            frames = {}
            for tf, seed_rows in seed.items():
                input_rows = seed_rows + rows[:index+1] if tf == 'M1' else seed_rows
                frame = pd.DataFrame(input_rows)
                frame['TIME_DT'] = pd.to_datetime(frame['RAW_SOURCE_EPOCH'], unit='s')
                frames[tf] = frame.sort_values('TIME_DT')
            base = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'BASE_FEATURES')
            namespace = {'np': np, 'pd': pd, 'BASE_FEATURES': base, 'data_dict': frames}
            verified = ast.fix_missing_locations(ast.Module(body=[indicator, *statements], type_ignores=[]))
            exec(compile(verified, '<independent-frozen-pre-label-verification>', 'exec'), namespace)
            vector = namespace['df'][binding['feature_list']].iloc[-1].to_numpy(dtype=np.float64)
            missing = [n for n, v in zip(binding['feature_list'], vector) if not np.isfinite(v)]
            assert check['missing_features'] == missing and check['feature_count_available'] == 31-len(missing)
            assert check['nan_count'] == int(np.isnan(vector).sum()) and check['inf_count'] == int(np.isinf(vector).sum())
            assert check['eligibility'] == ('NOT_ELIGIBLE_NAN' if missing else 'ELIGIBLE')
            if not missing:
                assert check['feature_vector_sha256'] == sha(vector.astype('<f8').tobytes())
                earliest = earliest or candidate_time
    assert result['holdout_evaluation_start'] == result['earliest_feature_complete_timestamp'] == earliest
    assert result['holdout_start'] == boundary['holdout_start'] == '2026-09-25T19:24:00+00:00'
    assert result['prefix_protocol_sha256'] == boundary['prefix_protocol_sha256']
    assert all(result[k] is False for k in ('model_loaded', 'strategy_evaluated', 'strategy_outcome_inspected'))
    return {'overall': 'PASS', 'feature_binding_status': 'PASS', 'feature_eligibility_status': 'PASS' if earliest else 'PARTIAL',
            'candidate_bars_checked': len(rows), 'holdout_evaluation_start': earliest,
            'lookahead_detected': False, 'reason': 'Verified missing seed-to-evidence inputs and unsealed closed HTF rows; no fabricated feature vector'}
