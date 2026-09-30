"""Additive, exact-run revalidation; retain the complete v1 independent audit."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import validate_gold_s4_secondary_improvement_run_v1 as audit

ROOT = Path(__file__).resolve().parent
TARGET_ID = '20260929T131155Z_gold_s4_secondary_improvement_v1'
TARGET_SEAL = '2693777cd8d7ff8b2c6c56beb78933cc03e4f88af2ba0492678ff515028e8856'
MODEL_SHA = '617e5724e2405e7423ec61b4104c4272d9094ed1fe29d6da95735326baf18753'
OLD_SHA = 'ae353354cdf7a7301ca4dbe849a5568e9fd8aaf74387709761979d50eeac130d'
HISTORICAL = 'gold_independent_secondary_classifier_v1.py'
HISTORICAL_BLOB_SHA = '191ab2d90533927651fb22eaa06a0ef494bafb2389775c415e942f8d13f1c0b8'
DISCOVERY = 'training_runs/20260915T151723Z_gold_independent_secondary_classifier_v1'
FAILURE = 'source_binding:' + HISTORICAL
CAUSE = ('The manifest binds raw working-tree bytes, identical to the sealed discovery '
         'snapshot; v1 also incorrectly requires that raw SHA to equal the LF-normalized '
         'Git blob SHA. The discovery file has 472 CRLF and 23 LF line endings. '
         'Only CRLF-to-LF normalization differs; no obsolete source or role substitution.')
ROLES = {
    HISTORICAL: 'HISTORICAL_DISCOVERY_PROVENANCE',
    'gold_manual_s4_secondary_retrain_v1.py': 'REFERENCE_REPRODUCTION',
    'gold_s4_secondary_improvement_v1.py': 'CURRENT_IMPROVEMENT_EXECUTION',
    'validate_gold_s4_secondary_improvement_run_v1.py': 'INDEPENDENT_VALIDATOR',
}
PROTECTED = {
    'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07',
    'gold_long_recent_candidate_xgb.json': '2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2',
}
EXPECTED_METRICS = dict(zip(audit.KEYS, (768, 438, 330, .5703125, .30035197497066873,
    .8436457240683531, -.06800170013341929, -52.22530570246601,
    -56.4297416185602, .8016509741596182)))
require, read, sha = audit.require, audit.read, audit.sha


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def git_blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def verify_binding(binding, path, expected, commit, raw, blob):
    """Raw hash never normalized; Git equivalence has a separate, explicit binding."""
    require(binding.get('path') == path and binding.get('role') == ROLES.get(path, 'WORKFLOW_DEPENDENCY')
            and binding.get('git_commit') == commit and binding.get('sha256') == expected,
            'binding_schema:' + path)
    require(byte_sha(raw) == expected, 'raw_source:' + path)
    require(binding.get('git_blob_sha256') == byte_sha(blob), 'git_blob:' + path)
    if raw != blob:
        require(path == HISTORICAL and byte_sha(blob) == HISTORICAL_BLOB_SHA
                and binding.get('git_representation') == 'CRLF_TO_LF'
                and raw.replace(b'\r\n', b'\n') == blob, 'git_representation:' + path)
    else:
        require(binding.get('git_representation') == 'EXACT_BYTES', 'git_representation:' + path)


def verify_inventory(root, hashes):
    for name, expected in hashes.items():
        require(sha(audit.path_in(root, name)) == expected, 'immutable:' + name)


def check_original(run):
    require(run.resolve() == ROOT/'training_runs'/TARGET_ID, 'same_original_run')
    require(sha(run/'FINALIZED.json') == TARGET_SEAL, 'original_seal')
    seal = read(run/'FINALIZED.json')
    verify_inventory(run, seal['file_sha256'])
    require(read(run/'validator.json') == {'overall': 'FAIL', 'run_id': TARGET_ID,
            'failed_checks': ['ValueError: ' + FAILURE]}, 'original_FAIL_preserved')
    candidate = read(run/'selected_candidate.json')['candidate']
    require(candidate['candidate_id'] == 'F5_NO_REDUNDANT_HTF', 'selected_candidate')
    require(candidate['pooled'] == EXPECTED_METRICS, 'exact_existing_metrics')
    require(sha(run/'models/F5_NO_REDUNDANT_HTF/fold3.json') == MODEL_SHA, 'candidate_model')
    verify_inventory(ROOT, PROTECTED)
    return seal['file_sha256']


def source_diagnosis(run):
    manifest = read(run/'manifest.json')
    require(set(ROLES).issubset(manifest['source_bindings']), 'missing_role_binding')
    require(manifest['manual_start'] is True and manifest['started_by'] == 'USER_LAUNCHER'
            and manifest['git_dirty'] is False, 'manual_source')
    require(sha(ROOT/'validate_gold_s4_secondary_improvement_run_v1.py') == OLD_SHA,
            'unchanged_full_validator')
    bindings, failures = [], []
    for path, expected in manifest['source_bindings'].items():
        raw = audit.path_in(ROOT, path).read_bytes()
        blob = git_blob(manifest['git_commit'], path)
        if not byte_sha(raw) == expected == byte_sha(blob):
            failures.append('source_binding:' + path)
        binding = dict(path=path, sha256=expected, git_commit=manifest['git_commit'],
                       role=ROLES.get(path, 'WORKFLOW_DEPENDENCY'), git_blob_sha256=byte_sha(blob),
                       git_representation='EXACT_BYTES' if raw == blob else 'CRLF_TO_LF')
        verify_binding(binding, path, expected, manifest['git_commit'], raw, blob)
        bindings.append(binding)
    require(failures == [FAILURE], 'exact_failure_reproduction')
    for source, snapshot in [('gold_s4_secondary_improvement_v1.py', 'training_script.py'),
                             ('gold_s4_secondary_improvement_config_v1.json', 'approved_config.json'),
                             ('validate_gold_s4_secondary_improvement_run_v1.py', 'validator_script.py')]:
        require(git_blob(manifest['git_commit'], source) == (run/snapshot).read_bytes(),
                'committed_snapshot:' + source)
    discovery = ROOT/DISCOVERY
    require(sha(discovery/'FINALIZED.json') == '522b9a8f477e23a79a9200905667b386747bb4fc9e7e40c66a2bdadd4ba19f4f',
            'discovery_seal')
    expected = manifest['source_bindings'][HISTORICAL]
    require(sha(discovery/'training_script.py') == expected
            == read(discovery/'FINALIZED.json')['file_sha256']['training_script.py'], 'discovery_snapshot')
    config = read(run/'approved_config.json')
    require(read(run/'research_plan.json') == config and
            read(run/'predeclared_search_space.json') == config['candidates'], 'plan_space')
    reference = ROOT/config['reference_run']
    require(sha(reference/'training_script.py') == manifest['source_bindings']['gold_manual_s4_secondary_retrain_v1.py'],
            'reference_training_snapshot')
    new_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    new_path = Path(__file__).name
    require(git_blob(new_commit, new_path) == Path(__file__).read_bytes(), 'committed_repair_validator')
    return {'schema_version': 'SOURCE_BINDING_V1_1', 'target_run_id': TARGET_ID,
            'failure_reproduced': True, 'original_failure': FAILURE, 'root_cause': CAUSE,
            'source_bindings': bindings, 'reference_binding_file_present': (run/'reference_binding.json').exists(),
            'repaired_validator_binding': {'path': new_path, 'sha256': sha(Path(__file__)),
                'git_commit': new_commit, 'role': 'INDEPENDENT_VALIDATOR'},
            'reference_evidence': ['reference_control.json', 'approved_config.json', 'manifest.json'],
            'historical_snapshot': DISCOVERY+'/training_script.py', 'source_binding_schema_status': 'PASS'}


def reject_training(frame, event, arg):
    """Reject trainer/search module calls and native model-fitting entry points."""
    if event == 'call':
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        if (module in {'gold_s4_secondary_improvement_v1', 'gold_manual_s4_secondary_retrain_v1',
                       'gold_s4_secondary_improvement_logic_v1'}
                or module.startswith(('xgboost', 'sklearn')) and name in {'fit', 'train', 'update', 'boost'}):
            raise PermissionError('Training/search prohibited in revalidation: ' + module + '.' + name)


def amendment_payload(result, repair_run, result_sha):
    require(result.get('overall') == 'PASS' and result.get('full_validation_completed') is True
            and result.get('run_id') == TARGET_ID and result.get('original_inventory_unchanged') is True
            and result.get('source_binding_schema_status') == 'PASS', 'full_validation_required')
    return {'target_run_id': TARGET_ID, 'original_validation_status': 'FAIL',
            'original_failure': FAILURE, 'repair_adjudication_status': 'PASS',
            'revalidation_status': 'PASS', 'effective_validation_status': 'PASS',
            'effective_final_status': 'PASS', 'research_result': 'IMPROVEMENT_FOUND',
            'candidate_gate': 'INTERESTING', 'candidate': 'F5_NO_REDUNDANT_HTF',
            'candidate_model_sha256': MODEL_SHA, 'repair_run': repair_run,
            'revalidation_sha256': result_sha, 'original_finalized_sha256': TARGET_SEAL,
            'holdout_used': False, 'production_changed': False, 'production_promoted': False,
            'evidence_scope': 'Historical development only; not untouched promotion evidence'}


def main():
    run, output = (Path(value).resolve() for value in sys.argv[1:3])
    require(run == ROOT/'training_runs'/TARGET_ID and output.parent == ROOT/'training_runs'
            and output.name.endswith('_gold_s4_source_binding_validator_repair_v1')
            and not (output/'FINALIZED.json').exists(), 'exact_target_separate_repair_run')
    write_new(output/'validator_attempt.json', {'target_run_id': TARGET_ID,
              'rule': 'One additive revalidation explicitly authorized; preserve original attempt',
              'at_utc': datetime.now(timezone.utc).isoformat()})
    release = None
    result = {'overall': 'FAIL', 'run_id': TARGET_ID, 'full_validation_completed': False,
              'model_training_executed': False, 'research_search_executed': False,
              'holdout_used': False, 'production_changed': False, 'production_promoted': False}
    try:
        print('Verifying original seal and source bindings', flush=True)
        inventory = check_original(run)
        diagnosis = source_diagnosis(run)
        write_new(output/'source_binding_diagnosis.json', diagnosis)
        write_new(output/'failure_reproduction.json', {'failure_reproduced': True,
                  'original_failure': FAILURE, 'method': 'Exact v1 source predicate, no old main invocation'})
        import numpy
        import pandas
        import xgboost
        import gold_gemini_execution_semantics_v1
        from training_holdout_guard_v1 import install
        release = install(ROOT, write_root=output)
        sys.setprofile(reject_training)
        print('Running unchanged complete independent validation against existing evidence', flush=True)
        checked = audit.validate(run)
        sys.setprofile(None)
        print('Rechecking all 212 original artifact hashes', flush=True)
        require(check_original(run) == inventory, 'inventory_unchanged')
        result.update(checked, overall='PASS', failed_checks=[], full_validation_completed=True,
                      original_inventory_unchanged=True, source_binding_schema_status='PASS')
    except Exception as error:
        result['failed_checks'] = [type(error).__name__ + ': ' + str(error)]
    finally:
        sys.setprofile(None)
        if release:
            release()
    write_new(output/'target_run_revalidation.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['overall'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
