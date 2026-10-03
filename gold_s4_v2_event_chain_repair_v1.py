"""Exact-run event-chain adjudication; no training, search, or target writes."""
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from validate_gold_s4_secondary_improvement_run_v1 import read, sha, digest, require, path_in

ROOT = Path(__file__).resolve().parent
TARGET_ID = '20261003T172302Z_gold_s4_secondary_improvement_v2'
TARGET = ROOT/'training_runs'/TARGET_ID
ARCHIVE_COMMIT = '70acd1780a005aa15285af7e3a6ad7a64ae858a1'
EXECUTION_COMMIT = 'dd60320199d3abaec12e2ab39f88331fdfd19de2'
ORIGINAL_SEAL = 'f7bf22f3f8faf82941e0153b3fc71a68b853f7ce1d144282dae81b37d666edcd'
ORIGINAL_TREE = '8b9ff3ace10ff482fff8f9ce5b63f909444baaa9efbdb7fa00c356961e7f7e48'
ORIGINAL_VALIDATOR = '845c4124b3a994b54eb6d933469fcfe8eaf284cfd90a85a668082a82d7a91d31'
POLICY_FILE = 'gold_s4_v2_event_chain_policy_v1.json'
CAUSE = ('REFERENCE_PASS_PAYLOAD_WRAPPER_MISMATCH: the archived trainer hashes the unwrapped '
         'control record, but the archived validator hashes dict(status="PASS", **control). '
         'The status wrapper belongs only to reference_control.json; all original event '
         'hashes, links, sequence, timestamps and candidate transitions are valid.')


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree(root):
    hashes = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink in evidence tree')
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha(path)
    return {'tree_sha256': digest(hashes), 'file_sha256': hashes}


def unchanged(root, expected):
    current = tree(root)
    require(current == expected, 'Original tree/file inventory changed')
    return current


def original_inventory():
    require(sha(TARGET/'FINALIZED.json') == ORIGINAL_SEAL, 'Original seal changed')
    snapshot = tree(TARGET)
    require(snapshot['tree_sha256'] == ORIGINAL_TREE, 'Original tree changed')
    seal = read(TARGET/'FINALIZED.json')['file_sha256']
    require(set(snapshot['file_sha256']) == set(seal) | {'FINALIZED.json'}, 'Original file inventory')
    require(all(snapshot['file_sha256'][name] == value for name, value in seal.items()), 'Archived file integrity')
    require(read(TARGET/'validator.json') == {'overall': 'FAIL', 'failed_checks': ['ValueError: Frozen chronological event chain']}, 'Original FAIL evidence')
    return snapshot


def expected_events(rows, selection, old=False):
    control = dict(status='PASS', **rows[0]) if old else rows[0]
    expected = [('space_frozen', None, None), ('reference_pass', 'F5_CONTROL', control), ('candidate_result', 'F5_CONTROL', rows[0])]
    for row in rows[1:]:
        expected += [('candidate_begin', row['candidate_id'], None), ('candidate_result', row['candidate_id'], row)]
    return expected + [('research_complete', None, selection)]


def inspect_chain(events, rows, selection, space, old=False):
    expected = expected_events(rows, selection, old)
    failures, previous, previous_time, previous_text = [], None, None, ''
    keys = {'sequence', 'event', 'at_utc', 'space_sha256', 'candidate_id', 'stage',
            'record_sha256', 'previous_sha256', 'event_sha256'}
    for i, event in enumerate(events):
        exp = expected[i] if i < len(expected) else None
        timestamp = event.get('at_utc')
        try:
            parsed = datetime.fromisoformat(timestamp)
            utc = parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
        except (TypeError, ValueError):
            parsed, utc = None, False
        predicates = {
            'schema': set(event) == keys and event.get('stage') is None,
            'sequence': type(event.get('sequence')) is int and event['sequence'] == i,
            'previous_hash': event.get('previous_sha256') == previous,
            'event_hash': digest({k: v for k, v in event.items() if k != 'event_sha256'}) == event.get('event_sha256'),
            'search_space_hash': event.get('space_sha256') == digest(space),
            'expected_transition': exp is not None and (event.get('event'), event.get('candidate_id')) == exp[:2],
            'record_hash': exp is not None and event.get('record_sha256') == (digest(exp[2]) if exp[2] else None),
            'timestamp': (isinstance(timestamp, str) and timestamp >= previous_text) if old else
                         bool(utc and (previous_time is None or parsed >= previous_time)),
        }
        if not all(predicates.values()):
            failures.append(dict(event_index=i, event_type=event.get('event'), timestamp=timestamp,
                sequence=event.get('sequence'), expected_previous_hash=previous, actual_previous_hash=event.get('previous_sha256'),
                expected_record_hash=digest(exp[2]) if exp and exp[2] else None, actual_record_hash=event.get('record_sha256'),
                freeze_state='FROZEN' if i >= len(expected) else 'RESEARCH_COMPLETE' if i == len(expected)-1 else 'OPEN',
                failed_predicates=[k for k,v in predicates.items() if not v]))
        previous = event.get('event_sha256')
        previous_text = timestamp if isinstance(timestamp, str) else previous_text
        if utc:
            previous_time = parsed
    if len(events) != len(expected):
        failures.append(dict(event_index=min(len(events),len(expected)), reason='Complete event inventory', expected_count=len(expected), actual_count=len(events)))
    return dict(overall='FAIL' if failures else 'PASS', failures=failures, event_count=len(events),
                chain_root=events[0]['event_sha256'] if events else None,
                chain_tip=events[-1]['event_sha256'] if events else None)


def check_chain(events, rows, selection, space):
    result = inspect_chain(events, rows, selection, space)
    require(result['overall'] == 'PASS', 'Frozen chronological event chain: '+json.dumps(result['failures']))
    return result


def check_lifecycle(research_events, phase, event, validation_complete=False):
    """Validation/archive records are separate artifacts, not research-chain members."""
    require(research_events and research_events[-1]['event'] == 'research_complete', 'Research freeze required')
    allowed = {'VALIDATION_ARTIFACT': {'validator_start', 'validator_result'},
               'ARCHIVE_ARTIFACT': {'finalization', 'git_commit', 'validation_amendment'}}
    require(event in allowed.get(phase, set()), 'Invalid post-freeze research event')
    if phase == 'ARCHIVE_ARTIFACT':
        require(validation_complete is True, 'Archive requires completed validation, including a preserved FAIL')


def source_provenance():
    manifest = read(TARGET/'manifest.json')
    require(manifest['git_commit'] == EXECUTION_COMMIT, 'Execution commit identity')
    require(subprocess.check_output(['git','rev-parse',ARCHIVE_COMMIT],cwd=ROOT).decode().strip() == ARCHIVE_COMMIT, 'Full archive commit')
    require(subprocess.check_output(['git','show',ARCHIVE_COMMIT+':training_runs/'+TARGET_ID+'/FINALIZED.json'],cwd=ROOT) == (TARGET/'FINALIZED.json').read_bytes(), 'Git archive seal')
    records = []
    for source, archived in [('gold_s4_secondary_improvement_v2.py','training_script.py'),
                              ('validate_gold_s4_secondary_improvement_v2_run.py','validator_script.py')]:
        raw = subprocess.check_output(['git','show',EXECUTION_COMMIT+':'+source],cwd=ROOT)
        require(raw == (TARGET/archived).read_bytes() == (ROOT/source).read_bytes(), 'Archived/current execution source:'+source)
        records.append(dict(role='ARCHIVED_EXECUTION_SOURCE',path='training_runs/'+TARGET_ID+'/'+archived,
                            sha256=sha(TARGET/archived),git_commit=EXECUTION_COMMIT))
    for name, expected in manifest['source_bindings'].items():
        path = path_in(ROOT,name)
        raw = path.read_bytes()
        blob = subprocess.check_output(['git','show',EXECUTION_COMMIT+':'+name],cwd=ROOT)
        require(sha(path) == expected and (raw == blob or raw.replace(b'\r\n',b'\n') == blob), 'Git-bound dependency:'+name)
        records.append(dict(role='CURRENT_REPOSITORY_SOURCE',path=name,sha256=expected,git_commit=EXECUTION_COMMIT,
                            git_blob_sha256=hashlib.sha256(blob).hexdigest(),representation='EXACT_BYTES' if raw==blob else 'CRLF_TO_LF'))
    return records


def diagnosis():
    rows = [json.loads(line) for line in (TARGET/'candidate_results.jsonl').read_text(encoding='utf-8').splitlines()]
    events = [json.loads(line) for line in (TARGET/'research_events.jsonl').read_text(encoding='utf-8').splitlines()]
    selection = read(TARGET/'selection.json')
    space = read(TARGET/'predeclared_search_space.json')
    old = inspect_chain(events,rows,selection,space,old=True)
    new = inspect_chain(events,rows,selection,space)
    require(len(old['failures']) == 1 and old['failures'][0]['event_index'] == 1
            and old['failures'][0]['failed_predicates'] == ['record_hash'], 'Exact original failure reproduction')
    require(read(TARGET/'reference_control.json') == dict(status='PASS',**rows[0]), 'Status wrapper remains verified separately')
    selected = read(TARGET/'selected_candidate.json')['candidate']
    require(selection['selected_candidate'] == (selected['candidate_id'] if selected else None)
            and read(TARGET/'training_result.json')['candidate'] == selected, 'Existing selection agreement')
    return dict(target_run_id=TARGET_ID,target_archive_commit=ARCHIVE_COMMIT,root_cause=CAUSE,
        original_predicate_result=old,corrected_chain_result=new,research_result=selection['research_result'],
        selected_candidate=selected['candidate_id'] if selected else None,selected_metrics=selected['pooled'] if selected else None,
        candidate_gate=selected['gate']['gate'] if selected else 'NONE',
        selected_economic_status=selected['economic_status'] if selected else None,
        research_chain_scope=read(ROOT/POLICY_FILE), duplicate_timestamp_count=len(events)-len({e['at_utc'] for e in events}),
        freeze_timestamp=events[-1]['at_utc'],finalization_timestamp=read(TARGET/'FINALIZED.json')['finalized_at_utc'],
        hypotheses=dict(A='No time regression',B='No duplicate timestamps; nondecreasing permits ties',
            C='All 56 sequence numbers valid',D='Every event and previous hash valid',E='No post-freeze append',
            F='Both sources freeze at research_complete before validation',G='Finalization followed validator failure',
            H='Validator attempt is not a research event',I='No Git/archive events in research chain',
            J='Same canonical JSON; wrong control payload wrapper in validator',K='All event timestamps UTC aware',
            L='Line endings outside canonical JSON hash; no event-byte mutation',M='No three-chain conflation',
            N='No mutable operational events validated as research'),
        metadata_limitations=['combined_result research NOT_RUN reflects error handler, not completed candidate loops',
                              'candidates.csv is the preserved header-only placeholder; authoritative completed rows are candidate_results.jsonl'])


def amendment(result, repair_path, seal_sha):
    require(result.get('overall') == 'PASS' and result.get('full_validation_completed') is True
            and result.get('target_run_id') == TARGET_ID and result.get('original_tree_after') == ORIGINAL_TREE
            and result.get('event_chain_integrity_status') == 'PASS', 'Full successful revalidation required')
    return dict(target_run_id=TARGET_ID, original_validation_status='FAIL', original_failure='Frozen chronological event chain',
        effective_validation_status='PASS',effective_final_status='PASS',repair_run=repair_path,repair_finalized_sha256=seal_sha,
        original_run_tree_sha256=ORIGINAL_TREE,original_finalized_sha256=ORIGINAL_SEAL,
        research_result=result['research_result'],selected_candidate=result['selected_candidate'],candidate_gate=result['candidate_gate'],
        model_training_already_executed_by_user=True,model_training_executed_by_repair=False,research_search_executed_by_repair=False,
        holdout_used=False,production_changed=False,production_promoted=False,
        evidence_scope='Historical development only; not untouched promotion evidence')
