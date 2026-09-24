"""Independent stdlib verifier. Does not import the inventory builder."""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

FRAMES = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30',
          'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'Daily', 'Weekly', 'Monthly']
START, END = datetime(2024, 1, 1), datetime(2026, 9, 24)
SLUG = 'gold_future_causal_context_inventory_v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_header(path):
    with path.open('rb') as f:
        encoding = 'utf-16' if f.read(2) in {b'\xff\xfe', b'\xfe\xff'} else 'utf-8-sig'
    with path.open(encoding=encoding, newline='') as f:
        line = f.readline(16385)
    if len(line) > 16384:
        raise ValueError('Oversized schema')
    sep = max(('\t', ',', ';'), key=line.count)
    cols = next(csv.reader([line], delimiter=sep))
    norm = [c.strip().strip('<>').upper() for c in cols]
    return {'columns': cols, 'normalized_columns': norm, 'encoding': encoding, 'delimiter': sep,
            'timestamp_fields': [c for c, n in zip(cols, norm) if n in {'DATE', 'TIME'}],
            'timestamp_dtype': 'naive text; YYYY.MM.DD[ HH:MM:SS]',
            'ohlc_fields': {n.lower(): cols[norm.index(n)] for n in ('OPEN', 'HIGH', 'LOW', 'CLOSE') if n in norm},
            'spread_field': cols[norm.index('SPREAD')] if 'SPREAD' in norm else None,
            'tick_volume_field': cols[norm.index('TICKVOL')] if 'TICKVOL' in norm else None,
            'real_volume_field': cols[norm.index('VOL')] if 'VOL' in norm else None,
            'source_metadata_fields': [c for c, n in zip(cols, norm) if n in {'SYMBOL', 'SOURCE_ID', 'BROKER', 'SERVER'}]}


def scan(path, schema, frame):
    """Independently stream timestamps; never parse numeric market/strategy values."""
    cols = schema['normalized_columns']
    di = cols.index('DATE'); ti = cols.index('TIME') if 'TIME' in cols else None
    nominal = int(frame[1:]) * (60 if frame[0] == 'M' else 3600) if frame[1:].isdigit() else {'Daily': 86400, 'Weekly': 604800, 'Monthly': None}[frame]
    need = 4096 if frame == 'M1' else 21
    seen = set(); previous = first = last = None
    count = duplicate = reversal = missing = gaps = available = 0
    spacing = Counter(); selected = deque(maxlen=need); largest = None
    with path.open(encoding=schema['encoding'], newline='') as f:
        reader = csv.reader(f, delimiter=schema['delimiter'])
        next(reader)
        for raw in reader:
            if not raw:
                continue
            if len(raw) != len(cols):
                raise ValueError('Malformed row width')
            text = raw[di].replace('.', '-') + ('T'+raw[ti] if ti is not None else '')
            stamp = datetime.fromisoformat(text)
            if stamp.tzinfo is not None:
                raise ValueError('Unexpected timezone-aware source')
            duplicate += stamp in seen
            seen.add(stamp)
            if first is None:
                first = stamp
            if previous is not None:
                delta = int((stamp - previous).total_seconds())
                reversal += delta < 0
                largest = delta if largest is None else max(largest, delta)
                if delta > 0:
                    spacing[delta] += 1
                slots = (stamp.year-previous.year)*12 + stamp.month-previous.month if nominal is None else delta//nominal
                missing += max(0, slots-1)
                gaps += slots > 1 if nominal is None else delta > nominal
                if START <= previous < END and previous < stamp <= END:
                    available += 1
                    selected.append({'row_index': count-1, 'timestamp': previous.isoformat(), 'successor_timestamp': stamp.isoformat()})
            count += 1; previous = last = stamp
    rows = list(selected)
    return {'row_count': count, 'first_timestamp': first.isoformat() if first else None,
            'last_timestamp': last.isoformat() if last else None, 'duplicate_count': duplicate,
            'non_monotonic_count': reversal, 'missing_interval_count': missing, 'largest_gap_seconds': largest,
            'gap_count': gaps, 'gap_distribution_summary': [{'seconds': s, 'count': c} for s, c in sorted(spacing.items())],
            'available_preboundary_bars': available, 'selected_context_bars': len(rows), 'selected_rows': rows,
            'selection_sha256': canonical(rows), 'warmup_requirement_satisfied': len(rows) >= need and not duplicate and not reversal}


def validate(run):
    root = run.parent.parent
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root)
    m = load(run/'manifest.json'); spec = load(run/'execution_spec.json'); s = run/'source'
    commit = m['source_commit']
    checks = {'source_commit': commit == m['git_commit'] == git('rev-parse', 'HEAD').decode().strip(),
              'remote_commit': commit == m['pre_run_remote_commit'] == git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0],
              'clean_provenance': m['git_dirty'] is False and m['pre_run_clean'] is True and m['pre_run_git_status'] == '',
              'source_inventory': set(spec['source_files']) == {e['source_path'] for e in m['input_snapshots']},
              'prior_runs_unchanged': not git('diff', spec['base_commit'], '--', 'training_runs'),
              'prior_seals_unchanged': all(sha(root/n) == h for n, h in spec['prior_seals'].items()),
              'preserved_inputs': all(sha(root/n) == h for n, h in spec['preserved_sha256'].items())}
    allowed = '?? '+run.relative_to(root).as_posix()+'/'
    checks['only_run_untracked'] = all(e.startswith(allowed) for status in (m['git_status_after_execution'], git('status', '--porcelain', '-z').decode('utf-8')) for e in status.split('\0') if e)
    for item in m['input_snapshots']:
        p = (run/item['path']).resolve()
        if not p.is_relative_to(s.resolve()):
            raise ValueError('Snapshot escape')
        checks['source:'+item['source_path']] = sha(p) == item['sha256'] == sha(root/item['source_path']) and p.read_bytes().replace(b'\r\n', b'\n') == git('cat-file', 'blob', commit+':'+item['source_path']).replace(b'\r\n', b'\n')
    checks['validator_snapshot'] = Path(__file__).read_bytes() == (run/'validator_script.py').read_bytes() == (s/('validate_'+SLUG+'.py')).read_bytes()
    checks['runner_snapshot'] = sha(run/'training_script.py') == m['training_script_sha256'] == sha(s/('run_'+SLUG+'.py'))
    checks['protected_production'] = m['protected_sha256_before'] == m['protected_sha256_after'] == spec['protected_sha256'] == {n: sha(root/n) for n in spec['protected_sha256']}
    candidates = load(run/'candidate_sources.json'); inventory = load(run/(SLUG+'.json'))
    fm_path = run/'gold_future_causal_context_file_manifest_v1.json'; fm = load(fm_path)
    decision = load(run/'gold_future_causal_context_compatibility_v1.json')
    binding = load(run/'gold_recursive_prefix_causal_context_inventory_binding_v1.json')
    metrics = load(run/'metrics.json')
    checks['all_candidates_accounted'] = {e['path'] for e in candidates} == set(spec['candidate_paths']) == {e['path'] for e in fm['entries']} and len(candidates) == len(spec['candidate_paths']) == len(fm['entries'])
    discovered = []
    skip = set(spec['discovery_excluded_directories'])
    for base, dirs, names in os.walk(root.parent):
        dirs[:] = [d for d in dirs if d not in skip and not (Path(base)/d).is_symlink()]
        for name in names:
            p = Path(base)/name
            if re.search('gold|xauusd', name, re.I) and p.suffix.lower() in {'.csv', '.gz', '.npz', '.npy', '.parquet', '.pkl'}:
                discovered.append(str(p.resolve()))
    checks['discovery_complete'] = set(discovered) == set(spec['candidate_paths'])
    actual = {}
    forbidden = re.compile(r'target|label|score|probab|predict|profit|pnl|reward|signal|trade|entry|exit|result|model|training', re.I)
    for e in candidates:
        path = Path(e['path'])
        print('Validate identity: '+path.name, flush=True)
        schema = read_header(path) if path.suffix.lower() == '.csv' else {}
        checks['file:'+str(path)] = sha(path) == e['sha256'] and path.stat().st_size == e['file_size']
        checks['schema:'+str(path)] = schema == e['schema'] and canonical(schema) == e['schema_sha256']
        names = schema.get('normalized_columns', [])
        if forbidden.search(path.name) or any(forbidden.search(n) for n in names) or path.suffix == '.pkl':
            classification = 'REJECTED_STRATEGY_OUTPUT'
        elif re.search(r'resampl|derived|aggregat', str(path), re.I):
            classification = 'DERIVED_OR_RESAMPLED'
        elif {'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SPREAD', 'VOL', 'TICKVOL'} <= set(names):
            classification = 'POTENTIALLY_COMPATIBLE' if re.match(r'GOLD#_(M\d+|H\d+|Daily|Weekly|Monthly)_', path.name) else 'INCOMPATIBLE_SOURCE'
        else:
            classification = 'UNKNOWN_PROVENANCE'
        checks['classification:'+str(path)] = classification == e['source_classification'] and e['native_source_confirmed'] is False and e['resampled'] == (classification == 'DERIVED_OR_RESAMPLED')
        manifest_entry = next(x for x in fm['entries'] if x['path'] == str(path))
        checks['manifest_entry:'+str(path)] = all(manifest_entry[k] == e[k] for k in ('path', 'sha256', 'timeframe', 'row_count', 'first_timestamp', 'last_timestamp', 'schema_sha256')) and manifest_entry['size'] == e['file_size'] and manifest_entry['tracked_in_git'] == (e['relative_path'] in spec['tracked_paths'])
        if classification == 'POTENTIALLY_COMPATIBLE' and e['timeframe'] in FRAMES:
            result = scan(path, schema, e['timeframe'])
            actual[str(path)] = result
            checks['timestamps:'+e['timeframe']] = all(e[k] == v for k, v in result.items())
            checks['stable_after_scan:'+e['timeframe']] = sha(path) == e['sha256']
        else:
            checks['no_rejected_body:'+str(path)] = e['row_count'] is None and e['first_timestamp'] is None and e['last_timestamp'] is None
    frames = inventory['timeframes']
    checks['exact_21_unique_frames'] = len(frames) == 21 and {e['timeframe'] for e in frames} == set(FRAMES)
    for e in frames:
        c = next(c for c in candidates if c['path'] == e['source_path'])
        checks['selected_source:'+e['timeframe']] = Path(e['source_path']).parent == root and e['source_file_sha256'] == c['sha256'] and e['source_file_size'] == c['file_size'] and all(e[k] == v for k, v in actual[c['path']].items()) and e['schema'] == c['schema']
        checks['conditional_context:'+e['timeframe']] = e['timezone_compatibility'] == 'CONDITIONAL' and e['timestamp_semantics'] == {'raw': 'NAIVE_TEXT_CLOCK', 'normalized': 'UNRESOLVED', 'bar_open': 'UNCONFIRMED_FOR_FILE'} and e['native_source_confirmed'] is False and e['certified_usable_causal_bars'] == 0 and e['gap_classification'] == 'UNKNOWN'
        checks['no_postboundary_rows:'+e['timeframe']] = all(START <= datetime.fromisoformat(r['timestamp']) < datetime.fromisoformat(r['successor_timestamp']) <= END for r in e['selected_rows'])
        need = 4096 if e['timeframe'] == 'M1' else 21
        # A missing requirement must produce research FAIL; archive validation still verifies an honest negative result.
        checks['warmup_accounting:'+e['timeframe']] = e['required_warmup_bars'] == need and e['warmup_requirement_satisfied'] == (e['selected_context_bars'] >= need and not e['duplicate_count'] and not e['non_monotonic_count'])
    checks['context_identity'] = inventory['source_id'] == 'XMGlobal-MT5-6_GOLD' and inventory['symbol'] == 'GOLD#' and inventory['broker'] == 'XM Global Limited' and inventory['server'] == 'XMGlobal-MT5 6' and inventory['environment'] == 'demo' and 'unconfirmed' in inventory['identity_scope']
    checks['context_honesty'] = inventory['context_role'] == metrics['context_role'] == 'CAUSAL_CONTEXT_ONLY' and all(inventory[k] is False and metrics[k] is False for k in ('historically_equivalent', 'outcome_tuning', 'holdout_evidence'))
    checks['frozen_envelope'] = inventory['proposed_context_start'] == metrics['context_start'] == binding['context_start'] == '2024-01-01T00:00:00Z' and inventory['proposed_context_end'] == metrics['context_end'] == binding['context_end'] == '2026-09-24T00:00:00Z'
    ordered = sorted(fm['entries'], key=lambda e: (e['timeframe'] or '', e['path'].replace('\\', '/').casefold()))
    root_hash = hashlib.sha256(''.join(canonical(e) for e in ordered).encode('ascii')).hexdigest()
    checks['inventory_root'] = ordered == fm['entries'] and root_hash == fm['inventory_root_sha256'] == inventory['inventory_root_sha256'] == metrics['inventory_root_sha256'] == binding['inventory_root_sha256']
    checks['file_manifest_hash'] = sha(fm_path) == binding['file_manifest_sha256'] == metrics['file_manifest_sha256']
    checks['binding'] = binding['inventory_file_sha256'] == sha(run/(SLUG+'.json')) and binding['prefix_proposal_sha256'] == sha(s/'gold_recursive_prefix_reinitialization_proposal_v1.json') and binding['causal_context_protocol_sha256'] == sha(s/'gold_recursive_prefix_causal_context_protocol_v1.json') and binding['pipeline_sha256'] == sha(s/'drl_trading_v2.py') and binding['timeframes'] == FRAMES and binding['required_warmup'] == {f: 4096 if f == 'M1' else 21 for f in FRAMES}
    fn = next(n for n in ast.parse((s/'drl_trading_v2.py').read_text(encoding='utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators')
    checks['feature_binding'] = binding['recursive_feature'] == 'MACD_HIST' and binding['feature_implementation_sha256'] == hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    checks['timezone_honesty'] = inventory['timezone_certification_status'] == 'UNRESOLVED' and metrics['timezone_compatibility'] == decision['timezone_compatibility'] == 'CONDITIONAL' and binding['approved'] is False and decision['approved'] is False and metrics['prefix_approval_ready'] is False and metrics['prefix_approval_ready_after_timezone'] is False and decision['ready_for_approval_after_timezone_attestation'] is False and binding['ready_for_approval_after_timezone_attestation'] is False
    good = all(e['warmup_requirement_satisfied'] and not e['resampled'] and e['source_classification'] == 'POTENTIALLY_COMPATIBLE' for e in frames)
    expected = 'PARTIAL' if good else 'FAIL'
    checks['honest_verdict'] = m['formal_run_status'] == metrics['formal_run_status'] == decision['formal_run_status'] == expected and metrics['inventory_status'] == decision['status'] == ('PARTIAL_TIMEZONE_PENDING' if good else 'FAIL')
    m1 = next(e for e in frames if e['timeframe'] == 'M1')
    checks['metrics_counts'] = metrics['timeframe_count'] == 21 and metrics['native_timeframe_count'] == 0 and metrics['m1_context_bars'] == m1['selected_context_bars'] and metrics['m1_available_preboundary_bars'] == m1['available_preboundary_bars'] and metrics['m1_requirement_satisfied'] == m1['warmup_requirement_satisfied'] and metrics['htf_requirements_satisfied'] == all(e['warmup_requirement_satisfied'] for e in frames if e['timeframe'] != 'M1')
    for name in ('schema_inventory.json', 'timestamp_inventory.json', 'continuity_inventory.json', 'coverage_inventory.json'):
        subset = load(run/name)
        checks['projection:'+name] = len(subset) == 21 and all(all(e[k] == next(f for f in frames if f['timeframe'] == e['timeframe'])[k] for k in e) for e in subset)
    checks['no_activation_or_production'] = all(metrics[k] is False for k in ('strategy_outcome_inspected', 'model_loaded', 'model_trained', 'protocol_frozen', 'capture_activated', 'holdout_started', 'production_changed', 'production_promoted'))
    for name in (SLUG+'.py', 'run_'+SLUG+'.py', 'test_'+SLUG+'.py'):
        tree = ast.parse((s/name).read_text(encoding='utf-8'))
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        checks['no_model_or_strategy:'+name] = not imports.intersection({'xgboost', 'torch', 'sklearn', 'MetaTrader5', 'gemini', 'drl_trading_v2'}) and not calls.intersection({'fit', 'predict', 'predict_proba', 'load_model', 'order_send'})
    tests = load(run/'self_test_stdout.txt')
    checks['self_tests'] = tests['overall'] == 'PASS' and tests['synthetic_only'] is True and len(tests['checks']) >= 32
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run/'FINALIZED.json').exists():
        raise FileExistsError('Finalized run immutable')
    with (run/'validator_attempt.json').open('x', encoding='utf-8') as f:
        json.dump({'one_shot': True, 'started_at_utc': datetime.now(timezone.utc).isoformat()}, f)
    try:
        checks = validate(run)
    except Exception as error:
        checks = {'exception:'+type(error).__name__: False}
    failed = [k for k, v in checks.items() if not v]
    result = {'overall': 'FAIL' if failed else 'PASS', 'failed_check_names': failed, 'checks': checks}
    with (run/'validator.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
    with (run/'validator.md').open('x', encoding='utf-8') as f:
        f.write('# Independent causal context inventory validation\n\n'+result['overall']+'\nResearch compatibility remains separate from archive validation.\n')
    print('overall='+result['overall']); print('failed_check_names='+json.dumps(failed))
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
