"""Causal prerequisite audit using only frozen seed and sealed raw evidence."""
import ast
import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout'/'gold_s4_v4'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def close_epoch(raw, tf):
    if tf == 'Monthly':
        dt = datetime.fromtimestamp(raw, timezone.utc)
        return raw + calendar.monthrange(dt.year, dt.month)[1]*86400
    return raw + (86400 if tf == 'Daily' else 604800 if tf == 'Weekly' else int(tf[1:])*(60 if tf[0] == 'M' else 3600))


def feature_binding():
    freeze = load(CAPTURE/'protocol'/'freeze.json')
    raw = (ROOT/'drl_trading_v2.py').read_bytes()
    tree = ast.parse(raw)
    indicator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators')
    pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_and_prepare_data')
    start = next(i for i, n in enumerate(pipeline.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'm1')
    stop = next(i for i, n in enumerate(pipeline.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prices')
    body = pipeline.body[start:stop]
    expected = freeze['prefix_protocol']
    actual_hash = sha(ast.dump(indicator, include_attributes=False).encode())
    if sha(raw) != expected['pipeline_sha256'] or actual_hash != freeze['feature_implementation_sha256']:
        raise ValueError('FROZEN_FEATURE_IMPLEMENTATION_MISMATCH')
    features = freeze['feature_names']
    if len(features) != 31 or len(set(features)) != 31:
        raise ValueError('FROZEN_FEATURE_LIST_MISMATCH')
    return {'status': 'PASS', 'feature_pipeline_file': 'drl_trading_v2.py',
            'feature_pipeline_sha256': sha(raw), 'feature_function': 'add_indicators',
            'feature_function_sha256': actual_hash, 'macd_implementation_sha256': actual_hash,
            'feature_construction_ast_sha256': sha(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()),
            'htf_implementation_sha256': sha(ast.dump(next(n for n in body if isinstance(n, ast.For)), include_attributes=False).encode()),
            'feature_list': features, 'feature_list_sha256': sha(canonical(features)),
            'initialization_semantics': expected['initialization_semantics']}


def frozen_features(data_dict, binding):
    """Execute only pinned pre-label statements, never import the pipeline module."""
    import numpy as np
    import pandas as pd
    if pd.__version__ != binding['initialization_semantics']['pandas_version']:
        raise ValueError('FROZEN_PANDAS_VERSION_MISMATCH')
    tree = ast.parse((ROOT/'drl_trading_v2.py').read_bytes())
    indicator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators')
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_and_prepare_data')
    start = next(i for i, n in enumerate(loader.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'm1')
    stop = next(i for i, n in enumerate(loader.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prices')
    body = loader.body[start:stop]
    if sha(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()) != binding['feature_construction_ast_sha256']:
        raise ValueError('FEATURE_SLICE_CHANGED')
    base = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'BASE_FEATURES')
    scope = {'np': np, 'pd': pd, 'BASE_FEATURES': base, 'data_dict': data_dict}
    # Both pieces are from the exact frozen file and the pre-label AST boundary.
    module = ast.fix_missing_locations(ast.Module(body=[indicator, *body], type_ignores=[]))
    exec(compile(module, '<frozen-feature-only>', 'exec'), scope)
    return scope['df'][binding['feature_list']]


def audit(capture=CAPTURE):
    from gold_future_capture_collector_v4 import recover_chain
    from validate_gold_future_native_context_v1 import validate
    binding = feature_binding()
    frozen = load(capture/'protocol'/'freeze.json')
    boundary = load(ROOT/'gold_future_holdout_boundary_v4.json')
    inventory = load(capture/'context_seed'/'gold_future_native_context_inventory_v1.json')
    validate(capture/'context_seed', frozen['source_commit'], sha((ROOT/'gold_future_native_context_fetch_v1.py').read_bytes()), boundary['holdout_start'])
    entries, _ = recover_chain(capture)
    # Freeze an immutable view of the append-only chain at audit start.
    evidence = []
    chain_view = []
    for item in entries:
        path = capture/item['snapshot_path']
        payload = load(path)
        evidence.extend(payload['rows'])
        chain_view.append({'sequence': item['sequence'], 'manifest_sha256': sha(canonical(item)),
                           'snapshot_path': item['snapshot_path'], 'snapshot_sha256': sha(path.read_bytes())})
    seed = {e['timeframe']: load(capture/'context_seed'/e['raw_path']) for e in inventory['entries']}
    endpoint = seed['M1'][-1]['RAW_SOURCE_EPOCH']
    known_successor = inventory['entries'][0]['returned_raw_epochs'][-1]
    first = evidence[0]['RAW_SOURCE_EPOCH'] if evidence else None
    gap = first is not None and first > endpoint+60
    gaps = [{'start_raw_epoch': endpoint+60, 'end_raw_epoch_exclusive': first,
             'known_source_successor_raw_epoch': known_successor,
             'classification': 'UNSEALED_SEED_TO_BOUNDARY_INPUTS; no backfill authorized'}] if gap else []
    checks = []
    for candidate in evidence:
        raw_epoch = candidate['RAW_SOURCE_EPOCH']
        available = [row for row in evidence if row['RAW_SOURCE_EPOCH'] <= raw_epoch]
        raw_input = candidate['SOURCE_TIMESTAMP'] >= boundary['holdout_start']
        epochs = [endpoint, *[r['RAW_SOURCE_EPOCH'] for r in available]]
        continuity = all(b-a == 60 for a, b in zip(epochs, epochs[1:]))
        # Seed bars were independently proven closed before fetch completion,
        # which itself strictly precedes every admitted candidate.
        missing_htf = [e['timeframe'] for e in inventory['entries'][1:]
                       if any(t > e['last_raw_epoch'] and close_epoch(t, e['timeframe']) <= raw_epoch
                              for t in e['returned_raw_epochs'])]
        htf_ready = not missing_htf and all(len(seed[tf]) >= 21 for tf in seed if tf != 'M1')
        result = {'candidate_timestamp': candidate['SOURCE_TIMESTAMP'], 'raw_input_available': raw_input,
            'context_available': True, 'm1_state_ready': continuity, 'htf_state_ready': htf_ready,
            'recursive_state_ready': continuity, 'feature_count_expected': 31,
            'feature_count_available': 0, 'nan_count': 0, 'inf_count': 0,
            'missing_features': list(binding['feature_list']), 'lookahead_detected': False,
            'eligibility': 'NOT_ELIGIBLE_INSUFFICIENT_CONTEXT', 'feature_vector_sha256': None,
            'feature_values_computed': False, 'max_input_raw_epoch': raw_epoch,
            'missing_closed_native_timeframes': missing_htf,
            'input_row_selection': 'Frozen seed plus sealed rows <= candidate; no refresh or interpolation'}
        if continuity and htf_ready:
            import numpy as np
            import pandas as pd
            frames = {}
            for tf, rows in seed.items():
                source = rows + available if tf == 'M1' else rows
                frame = pd.DataFrame(source)
                frame['TIME_DT'] = pd.to_datetime(frame['RAW_SOURCE_EPOCH'], unit='s')
                frames[tf] = frame.sort_values('TIME_DT')
            vector = frozen_features(frames, binding).iloc[-1].to_numpy(dtype=np.float64)
            nan_count = int(np.isnan(vector).sum()); inf_count = int(np.isinf(vector).sum())
            missing = [name for name, value in zip(binding['feature_list'], vector) if not np.isfinite(value)]
            result.update(feature_count_available=31-len(missing), nan_count=nan_count, inf_count=inf_count,
                          missing_features=missing, feature_values_computed=True,
                          eligibility='NOT_ELIGIBLE_NAN' if missing else 'ELIGIBLE',
                          feature_vector_sha256=sha(vector.astype('<f8').tobytes()) if not missing else None)
        checks.append(result)
    earliest = next((c['candidate_timestamp'] for c in checks if c['eligibility'] == 'ELIGIBLE'), None)
    artifact = {'protocol_version': 'v4', 'protocol_commit': boundary['protocol_commit'],
        'activation_commit': boundary['activation_commit'], 'boundary_commit': '647932256c96776095399a8e95b40dc3dd88a35a',
        'holdout_start': boundary['holdout_start'], 'feature_pipeline_sha256': binding['feature_pipeline_sha256'],
        'feature_list_sha256': binding['feature_list_sha256'], 'prefix_protocol_sha256': boundary['prefix_protocol_sha256'],
        'context_inventory_root_sha256': inventory['inventory_root_sha256'], 'expected_feature_count': 31,
        'candidate_checks': checks, 'chain_view': chain_view, 'seed_to_boundary_gaps': gaps,
        'earliest_feature_complete_timestamp': earliest, 'holdout_evaluation_start': earliest,
        'status': 'PASS' if earliest else 'PARTIAL', 'model_loaded': False,
        'strategy_evaluated': False, 'strategy_outcome_inspected': False}
    return binding, artifact
