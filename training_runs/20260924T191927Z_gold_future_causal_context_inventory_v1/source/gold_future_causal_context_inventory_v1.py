"""Historical file identity and timestamp structure only; no market computations."""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_future_causal_context_inventory_v1'
FRAMES = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20',
          'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'Daily', 'Weekly', 'Monthly']
START, END = '2024-01-01T00:00:00', '2026-09-24T00:00:00'
SOURCE = 'XMGlobal-MT5-6_GOLD'
SKIP = {'.git', '.venv', 'venv', '__pycache__', 'training_runs', 'untouched_forward',
        '.private_research_provenance', '.research_tools', '.agents', '.codex'}
OUTCOME = re.compile(r'target|label|score|probab|predict|profit|pnl|reward|signal|trade|entry|exit|result|model|training', re.I)
RAW = {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def discover(base):
    found = []
    for directory, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory)/d).is_symlink())
        for name in names:
            p = Path(directory)/name
            if re.search(r'gold|xauusd', name, re.I) and p.suffix.lower() in {'.csv', '.gz', '.npz', '.npy', '.parquet', '.pkl'}:
                if p.is_symlink():
                    raise ValueError('Symlink candidate requires explicit review')
                found.append(p.resolve())
    return sorted(found, key=lambda p: str(p).casefold())


def header(path):
    with Path(path).open('rb') as f:
        encoding = 'utf-16' if f.read(2) in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    with Path(path).open(encoding=encoding, newline='') as f:
        line = f.readline(16385)
    if len(line) > 16384:
        raise ValueError('Oversized header')
    delimiter = max(('\t', ',', ';'), key=line.count)
    columns = next(csv.reader([line], delimiter=delimiter))
    normalized = [c.strip().strip('<>').upper() for c in columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError('Duplicate schema columns')
    return {'columns': columns, 'normalized_columns': normalized, 'encoding': encoding,
            'delimiter': delimiter, 'timestamp_fields': [columns[i] for i, n in enumerate(normalized) if n in {'DATE', 'TIME'}],
            'timestamp_dtype': 'naive text; YYYY.MM.DD[ HH:MM:SS]',
            'ohlc_fields': {n.lower(): columns[normalized.index(n)] for n in ('OPEN', 'HIGH', 'LOW', 'CLOSE') if n in normalized},
            'spread_field': columns[normalized.index('SPREAD')] if 'SPREAD' in normalized else None,
            'tick_volume_field': columns[normalized.index('TICKVOL')] if 'TICKVOL' in normalized else None,
            'real_volume_field': columns[normalized.index('VOL')] if 'VOL' in normalized else None,
            'source_metadata_fields': [c for c, n in zip(columns, normalized) if n in {'SYMBOL', 'SOURCE_ID', 'BROKER', 'SERVER'}]}


def classify(path, schema):
    if OUTCOME.search(Path(path).name) or any(OUTCOME.search(c) for c in schema.get('normalized_columns', [])) or Path(path).suffix == '.pkl':
        return 'REJECTED_STRATEGY_OUTPUT'
    if re.search(r'resampl|derived|aggregat', str(path), re.I):
        return 'DERIVED_OR_RESAMPLED'
    cols = set(schema.get('normalized_columns', []))
    if RAW <= cols and 'DATE' in cols:
        if re.match(r'GOLD#_(M\d+|H\d+|Daily|Weekly|Monthly)_', Path(path).name):
            return 'POTENTIALLY_COMPATIBLE'
        return 'INCOMPATIBLE_SOURCE'
    return 'UNKNOWN_PROVENANCE'


def interval(frame):
    if frame.startswith('M') and frame[1:].isdigit():
        return int(frame[1:]) * 60
    if frame.startswith('H'):
        return int(frame[1:]) * 3600
    return {'Daily': 86400, 'Weekly': 604800, 'Monthly': None}[frame]


def analyze(times, frame):
    """Naive-clock candidate coverage, never a certified UTC conversion."""
    times = pd.DatetimeIndex(times)
    invalid = int(times.isna().sum())
    if invalid:
        raise ValueError('Invalid timestamps; no silent dropping')
    seconds = times.to_numpy(dtype='datetime64[s]').astype('int64')
    deltas = np.diff(seconds)
    nominal = interval(frame)
    units = np.diff(times.year * 12 + times.month) if nominal is None else deltas // nominal
    values, counts = np.unique(deltas[deltas > 0], return_counts=True)
    gap_mask = units > 1 if nominal is None else deltas > nominal
    # Successor is a conservative closure proxy, conditional on BAR_OPEN semantics.
    eligible = np.flatnonzero((times[:-1] >= pd.Timestamp(START)) &
                             (times[1:] <= pd.Timestamp(END)) &
                             (times[:-1] < pd.Timestamp(END)) & (deltas > 0))
    need = 4096 if frame == 'M1' else 21
    selected = eligible[-need:]
    projection = [{'row_index': int(i), 'timestamp': times[i].isoformat(),
                   'successor_timestamp': times[i+1].isoformat()} for i in selected]
    result = {'row_count': len(times), 'first_timestamp': times[0].isoformat() if len(times) else None,
              'last_timestamp': times[-1].isoformat() if len(times) else None,
              'duplicate_count': int(times.duplicated().sum()), 'non_monotonic_count': int((deltas < 0).sum()),
              'invalid_timestamp_count': invalid,
              'missing_interval_count': int(np.maximum(units - 1, 0).sum()),
              'missing_interval_semantics': 'Nominal calendar slots only; not confirmed missing tradable bars',
              'largest_gap_seconds': int(deltas.max()) if len(deltas) else None,
              'gap_distribution_summary': [{'seconds': int(v), 'count': int(c)} for v, c in zip(values, counts)],
              'gap_classification': 'UNKNOWN', 'gap_count': int(gap_mask.sum()),
              'expected_interval_seconds': nominal, 'monthly_spacing_basis': 'Calendar months' if nominal is None else None,
              'available_preboundary_bars': int(len(eligible)), 'required_warmup_bars': need,
              'selected_context_bars': len(selected), 'selected_rows': projection,
              'selection_sha256': digest(projection),
              'certified_usable_causal_bars': 0,
              'warmup_requirement_satisfied': len(selected) >= need and not times.duplicated().any() and not (deltas < 0).any(),
              'coverage_basis': 'Conditional naive-clock envelope and successor proxy; not certified UTC/closure',
              'final_source_row_excluded': True}
    return result


def inspect(path, repo):
    before = path.stat()
    h = sha(path)
    schema = header(path) if path.suffix.lower() == '.csv' else {}
    classification = classify(path, schema)
    match = re.match(r'(GOLD#|XAUUSD[#]?)_(M\d+|H\d+|Daily|Weekly|Monthly)_', path.name)
    frame = match[2] if match else None
    item = {'path': str(path), 'relative_path': path.relative_to(repo).as_posix() if path.is_relative_to(repo) else None,
            'file_size': before.st_size, 'mtime_ns': before.st_mtime_ns, 'sha256': h,
            'format': path.suffix.lower(), 'timeframe': frame,
            'source_symbol': match[1] if match else None, 'symbol_basis': 'Filename claim only; no embedded source certificate',
            'source_id_if_present': None, 'source_classification': classification, 'schema': schema,
            'schema_sha256': digest(schema), 'first_timestamp': None, 'last_timestamp': None, 'row_count': None,
            'native_source_confirmed': False, 'resampled': classification == 'DERIVED_OR_RESAMPLED',
            'resampling_provenance': 'Derived flag is not proof of native origin when false',
            'inspection': 'Identity/header only; no body deserialization'}
    if classification == 'POTENTIALLY_COMPATIBLE' and frame in FRAMES:
        cols = schema['columns']; names = schema['normalized_columns']
        date = cols[names.index('DATE')]
        time = cols[names.index('TIME')] if 'TIME' in names else None
        if frame not in {'Daily', 'Weekly', 'Monthly'} and time is None:
            item['parse_error'] = 'Intraday TIME field missing'
        else:
            data = pd.read_csv(path, sep=schema['delimiter'], encoding=schema['encoding'],
                               usecols=schema['timestamp_fields'], dtype=str, keep_default_na=False)
            text = data[date] + (' ' + data[time] if time else '')
            parsed = pd.to_datetime(text, format='%Y.%m.%d %H:%M:%S' if time else '%Y.%m.%d', errors='raise')
            item.update(analyze(parsed, frame))
            item['inspection'] = 'Timestamp columns only; OHLC/volume/spread values not interpreted'
    if before.st_size != path.stat().st_size or before.st_mtime_ns != path.stat().st_mtime_ns or sha(path) != h:
        raise ValueError('Source mutated during inventory')
    return item


def manifest_root(entries):
    ordered = sorted(entries, key=lambda e: (e['timeframe'] or '', e['path'].replace('\\', '/').casefold()))
    return hashlib.sha256(''.join(digest(e) for e in ordered).encode('ascii')).hexdigest()


def validate_context(doc):
    frames = doc['timeframes']
    if len(frames) != 21 or {e['timeframe'] for e in frames} != set(FRAMES):
        raise ValueError('Exactly 21 unique frames required')
    if doc['context_role'] != 'CAUSAL_CONTEXT_ONLY' or any(doc[k] is not False for k in ('historically_equivalent', 'outcome_tuning', 'holdout_evidence')):
        raise ValueError('Context honesty')
    if doc['source_id'] != SOURCE or doc['symbol'] != 'GOLD#' or doc['broker'] != 'XM Global Limited' or doc['server'] != 'XMGlobal-MT5 6':
        raise ValueError('Context identity')
    if doc['proposed_context_start'] != START+'Z' or doc['proposed_context_end'] != END+'Z':
        raise ValueError('Frozen envelope')
    for e in frames:
        if e['source_symbol'] != 'GOLD#' or e.get('source_id_if_present') not in (None, SOURCE):
            raise ValueError('Per-file identity mismatch')
        if e['resampled'] or e['source_classification'] in {'DERIVED_OR_RESAMPLED', 'INCOMPATIBLE_SOURCE', 'REJECTED_STRATEGY_OUTPUT'}:
            raise ValueError('Unsuitable context')
        if not e['warmup_requirement_satisfied'] or e['selected_context_bars'] < (4096 if e['timeframe'] == 'M1' else 21):
            raise ValueError('Insufficient context')
        if e['duplicate_count'] or e['non_monotonic_count']:
            raise ValueError('Timestamp ordering')
        for row in e['selected_rows']:
            if not START <= row['timestamp'] < row['successor_timestamp'] <= END:
                raise ValueError('Post-boundary or unordered context')


def build(run, spec):
    paths = discover(ROOT.parent)
    if [str(p) for p in paths] != spec['candidate_paths']:
        raise ValueError('Discovery changed since preregistration')
    candidates = []
    for path in paths:
        print('Inventory: '+path.name, flush=True)
        candidates.append(inspect(path, ROOT))
    entries = []
    for frame in FRAMES:
        eligible = [e for e in candidates if e['timeframe'] == frame and Path(e['path']).parent == ROOT]
        if len(eligible) != 1 or 'selected_rows' not in eligible[0]:
            raise ValueError('Required timeframe missing or ambiguous: '+frame)
        e = dict(eligible[0])
        e.update(source_path=e['path'], source_file_sha256=e['sha256'], source_file_size=e['file_size'],
                 timestamp_field=e['schema']['timestamp_fields'], timestamp_semantics={
                     'raw': 'NAIVE_TEXT_CLOCK', 'normalized': 'UNRESOLVED', 'bar_open': 'UNCONFIRMED_FOR_FILE'},
                 timezone_compatibility='CONDITIONAL', coverage_status='STRUCTURAL_CANDIDATE_ONLY',
                 context_status='SEALED_STRUCTURAL_CONTEXT',
                 blockers=['No exact file-to-broker/server/native export attestation',
                           'Timezone/DST/BAR_OPEN/closure authority unresolved',
                           'Unknown session gaps and missing continuity bridge to future activation'])
        entries.append(e)
    doc = {'inventory_version': 1, 'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
           'source_id': SOURCE, 'symbol': 'GOLD#', 'broker': 'XM Global Limited', 'server': 'XMGlobal-MT5 6',
           'environment': 'demo', 'identity_scope': 'Target identity; historical file origin remains unconfirmed',
           'context_role': 'CAUSAL_CONTEXT_ONLY', 'historically_equivalent': False, 'outcome_tuning': False,
           'holdout_evidence': False, 'proposed_context_start': START+'Z', 'proposed_context_end': END+'Z',
           'higher_timeframe_policy': 'NATIVE_20TF_REQUIRED', 'timezone_certification_status': 'UNRESOLVED',
           'timeframes': entries}
    structural_errors = []
    try:
        validate_context(doc)
    except ValueError as error:
        structural_errors.append(str(error))
    for e in entries:
        if not e['warmup_requirement_satisfied']:
            e['context_status'] = 'REJECTED'
    fm_entries = [{k: e[k] for k in ('path', 'timeframe', 'first_timestamp', 'last_timestamp', 'row_count', 'schema_sha256')} |
                  {'size': e['file_size'], 'sha256': e['sha256'], 'tracked_in_git': e['relative_path'] in spec['tracked_paths'],
                   'retention': 'External local file; identity sealed in Git, data bytes are not Git-immutable'} for e in candidates]
    fm_entries.sort(key=lambda e: (e['timeframe'] or '', e['path'].replace('\\', '/').casefold()))
    fm = {'version': 1, 'entries': fm_entries, 'inventory_root_sha256': manifest_root(fm_entries)}
    doc['inventory_root_sha256'] = fm['inventory_root_sha256']
    status = 'FAIL' if structural_errors else 'PARTIAL_TIMEZONE_PENDING'
    decision = {'status': status, 'formal_run_status': 'FAIL' if structural_errors else 'PARTIAL',
                'source_identity_compatible': False, 'native_origin_certified': False,
                'structural_requirements_satisfied': not structural_errors, 'structural_errors': structural_errors,
                'timezone_compatibility': 'CONDITIONAL', 'approved': False,
                'ready_for_approval_after_timezone_attestation': False,
                'blockers': ['File-level broker/server/native export provenance unconfirmed',
                             'Timezone and closed-bar interpretation remain conditional',
                             'Historical files stop before proposed end; causal continuation bridge not established']}
    write(run/(SLUG+'.json'), doc)
    fm_path = run/'gold_future_causal_context_file_manifest_v1.json'
    write(fm_path, fm)
    write(run/'gold_future_causal_context_compatibility_v1.json', decision)
    binding = {'prefix_proposal_sha256': sha(ROOT/'gold_recursive_prefix_reinitialization_proposal_v1.json'),
               'causal_context_protocol_sha256': sha(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json'),
               'inventory_root_sha256': fm['inventory_root_sha256'], 'file_manifest_sha256': sha(fm_path),
               'inventory_file_sha256': sha(run/(SLUG+'.json')), 'source_id': SOURCE, 'symbol': 'GOLD#',
               'context_start': START+'Z', 'context_end': END+'Z', 'timeframes': FRAMES,
               'required_warmup': {f: 4096 if f == 'M1' else 21 for f in FRAMES},
               'recursive_feature': 'MACD_HIST', 'approved': False, 'ready_for_approval_after_timezone_attestation': False,
               'pipeline_sha256': sha(ROOT/'drl_trading_v2.py'),
               'feature_implementation_sha256': load(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json')['feature_implementation_sha256'],
               'feature_mechanics': 'Frozen rolling20 plus lag1; M1 recursive4096 proposal unchanged; no feature computation',
               'continuation_required': True}
    write(run/'gold_recursive_prefix_causal_context_inventory_binding_v1.json', binding)
    write(run/'candidate_sources.json', candidates)
    sections = {'schema_inventory.json': ['timeframe', 'schema', 'schema_sha256'],
                'timestamp_inventory.json': ['timeframe', 'timestamp_semantics', 'timezone_compatibility', 'first_timestamp', 'last_timestamp'],
                'continuity_inventory.json': ['timeframe', 'row_count', 'duplicate_count', 'non_monotonic_count', 'missing_interval_count', 'largest_gap_seconds', 'gap_count', 'gap_distribution_summary', 'gap_classification'],
                'coverage_inventory.json': ['timeframe', 'available_preboundary_bars', 'selected_context_bars', 'required_warmup_bars', 'warmup_requirement_satisfied', 'coverage_basis', 'certified_usable_causal_bars', 'selected_rows', 'selection_sha256']}
    for name, fields in sections.items():
        write(run/name, [{k: e[k] for k in fields} for e in entries])
    return doc, decision, binding
