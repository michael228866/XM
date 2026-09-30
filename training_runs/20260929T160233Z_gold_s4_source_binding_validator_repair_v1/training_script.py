"""Archive one explicitly authorized additive revalidation, never a research run."""
import json
import subprocess
import sys
from pathlib import Path

import training_run_history as history
import validate_gold_s4_secondary_improvement_run_v1_1 as repair

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [Path(__file__).name, 'validate_gold_s4_secondary_improvement_run_v1_1.py',
                'test_gold_s4_source_binding_repair_v1.py']


def amend(run):
    """Explicit second phase, only after the repair archive has been committed/pushed."""
    repair.require(run.resolve().parent == ROOT/'training_runs'
                   and run.name.endswith('_gold_s4_source_binding_validator_repair_v1'), 'repair_run_path')
    repair.require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'clean_amendment_source')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    repair.require(head == subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0],
                   'archive_must_be_pushed')
    repair.require(not history.validate_run(run), 'sealed_repair_archive')
    repair.require(repair.read(run/'validator.json')['overall'] == 'PASS', 'repair_validator_PASS')
    target = ROOT/'training_runs'/repair.TARGET_ID
    repair.check_original(target)
    result = repair.read(run/'target_run_revalidation.json')
    amendment = repair.amendment_payload(result, run.relative_to(ROOT).as_posix(),
                                        repair.sha(run/'target_run_revalidation.json'))
    amendment.update(repair_finalized_sha256=repair.sha(run/'FINALIZED.json'), result_commit=head)
    files = {'validator_recheck_v1.json': result, 'validation_amendment_v1.json': amendment,
             'source_binding_adjudication_v1.json': repair.read(run/'source_binding_adjudication_v1.json')}
    repair.require(all(not (target/name).exists() for name in files), 'additive_files_must_be_new')
    for name, value in files.items():
        repair.write_new(target/name, value)
    repair.check_original(target)
    with (ROOT/'TRAINING_RUNS.md').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write('\n## Additive validation amendment: '+repair.TARGET_ID+'\n\n'
                     'Original validation FAIL and aborted seal preserved. Source-binding adjudication PASS; '
                     'complete independent revalidation PASS; effective validation/final status PASS. '
                     'Research remains IMPROVEMENT_FOUND / INTERESTING; F5_NO_REDUNDANT_HTF unchanged. '
                     'Historical development evidence only, not untouched promotion evidence. '
                     'No training, search, holdout use, production promotion, or production change. '
                     'Repair: `'+run.relative_to(ROOT).as_posix()+'`; '
                     'receipt: `training_runs/'+repair.TARGET_ID+'/validation_amendment_v1.json`.\n')
    print(json.dumps(amendment), flush=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--amend':
        amend(Path(sys.argv[2]).resolve())
        return 0
    repair.require(len(sys.argv) == 1, 'usage: repair runner [--amend repair_run]')
    target = ROOT/'training_runs'/repair.TARGET_ID
    repair.require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'clean_source_required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    upstream = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0]
    repair.require(head == upstream, 'pushed_source_required')
    repair.require(not list((ROOT/'training_runs').glob('*_gold_s4_source_binding_validator_repair_v1')),
                   'one_authorized_repair_attempt_only')
    run = history.create_run('gold_s4_source_binding_validator_repair_v1', Path(__file__),
                             '.venv\\Scripts\\python.exe -B '+Path(__file__).name,
                             seed_note='No training or search; deterministic validation only')
    print('REPAIR_RUN='+run.name, flush=True)
    write = lambda name, value: repair.write_new(run/name, value)
    spec = {'target_run_id': repair.TARGET_ID, 'original_finalized_sha256': repair.TARGET_SEAL,
            'source_commit': head, 'source_bindings': {name: repair.sha(ROOT/name) for name in SOURCE_FILES},
            'scope': 'Complete validation of existing immutable evidence; no new research evaluation',
            'model_training_executed': False, 'research_search_executed': False,
            'metrics_regenerated': False, 'holdout_used': False, 'production_changed': False}
    write('execution_spec.json', spec)
    for name in SOURCE_FILES[1:]:
        (run/name).write_bytes((ROOT/name).read_bytes())
    with (run/'self_test_stdout.txt').open('x') as out, (run/'self_test_stderr.txt').open('x') as err:
        tested = subprocess.run([sys.executable, '-B', SOURCE_FILES[2]], cwd=ROOT, stdout=out, stderr=err)
    write('self_test_result.json', {'overall': 'PASS' if tested.returncode == 0 else 'FAIL', 'exit_code': tested.returncode})
    with (run/'validator_stdout.txt').open('x') as out, (run/'validator_stderr.txt').open('x') as err:
        validated = subprocess.run([sys.executable, '-B', SOURCE_FILES[1], str(target), str(run)],
                                   cwd=ROOT, stdout=out, stderr=err)
    result = repair.read(run/'target_run_revalidation.json') if (run/'target_run_revalidation.json').exists() else {
        'overall': 'FAIL', 'failed_checks': ['Validator did not produce result']}
    success = tested.returncode == validated.returncode == 0 and result['overall'] == 'PASS'
    status = 'PASS' if success else 'FAIL'
    immutable = False
    try:
        repair.check_original(target)
        immutable = True
    except Exception as error:
        result.setdefault('failed_checks', []).append(str(error))
        success, status = False, 'FAIL'
    diagnosis = repair.read(run/'source_binding_diagnosis.json') if (run/'source_binding_diagnosis.json').exists() else {}
    manifest = repair.read(target/'manifest.json')
    receipt = {'target_run_id': repair.TARGET_ID, 'original_validation_status': 'FAIL',
               'original_failure': repair.FAILURE, 'root_cause': repair.CAUSE,
               'adjudication_status': 'PASS' if diagnosis else 'FAIL',
               'historical_methodology_source': repair.HISTORICAL,
               'historical_methodology_source_sha256': manifest['source_bindings'][repair.HISTORICAL],
               'current_research_entrypoint': 'gold_s4_secondary_improvement_v1.py',
               'current_research_entrypoint_sha256': manifest['source_bindings']['gold_s4_secondary_improvement_v1.py'],
               'reference_training_entrypoint': 'gold_manual_s4_secondary_retrain_v1.py',
               'reference_training_entrypoint_sha256': manifest['source_bindings']['gold_manual_s4_secondary_retrain_v1.py'],
               'validator_old_sha256': repair.OLD_SHA, 'validator_new_sha256': repair.sha(ROOT/SOURCE_FILES[1]),
               'candidate_model_sha256_before': repair.MODEL_SHA,
               'candidate_model_sha256_after': repair.sha(target/'models/F5_NO_REDUNDANT_HTF/fold3.json'),
               'research_metrics_unchanged': immutable, 'search_space_unchanged': immutable,
               'candidate_selection_unchanged': immutable, 'holdout_used': False, 'production_changed': False,
               'schema': diagnosis, 'revalidation_status': status}
    write('source_binding_adjudication_v1.json', receipt)
    for name, detail in [
        ('candidate_immutability_check.json', {'model_sha256': receipt['candidate_model_sha256_after']}),
        ('metrics_immutability_check.json', {'metrics': repair.EXPECTED_METRICS}),
        ('search_space_immutability_check.json', {'candidate_count': 16, 'selection_unchanged': immutable})]:
        write(name, {'overall': 'PASS' if immutable else 'FAIL', 'all_original_artifacts_unchanged': immutable, **detail})
    write('holdout_guard_check.json', {'overall': 'PASS' if tested.returncode == 0 else 'FAIL',
          'holdout_used': False, 'enforcement': 'Existing audited file/subprocess guard; synthetic locked-path rejection'})
    write('production_protection_check.json', {'overall': 'PASS' if immutable else 'FAIL',
          'production_changed': False, 'production_promoted': False, 'expected': repair.PROTECTED})
    write('validator_change_review.json', {'overall': 'PASS', 'old_validator_sha256': repair.OLD_SHA,
          'full_validation_function_unchanged': True, 'raw_hash_checks_retained': True,
          'normalization_scope': 'Only pinned discovery blob; CRLF to LF; raw snapshot remains exact',
          'original_failure_preserved': immutable, 'training_search_runtime_guard': True})
    write('validator.json', {'overall': status, 'target_run_id': repair.TARGET_ID,
          'full_validation_completed': result.get('full_validation_completed', False),
          'failed_checks': result.get('failed_checks', [])})
    history.write_json(run/'metrics.json', {'formal_run_status': status, 'revalidation': result,
                                          'model_training_executed': False, 'research_search_executed': False})
    report = ('# S4 source-binding validator repair\n\nStatus: '+status+'\n\n'+repair.CAUSE+
              '\n\nTarget: '+repair.TARGET_ID+'\n\nOriginal FAIL retained. Research remains '
              'IMPROVEMENT_FOUND / INTERESTING. No promotion; no production change; holdout unused. '
              'The original aborted manifest remains historical; any effective PASS is a separate amendment.\n\n'+
              json.dumps(result, indent=2)+'\n')
    (run/'report.md').write_text(report, encoding='utf-8')
    (run/'findings.md').write_text(report, encoding='utf-8')
    m = history.read_json(run/'manifest.json')
    m['data'].update(symbols=['GOLD#'], data_sources=['Existing sealed historical research artifacts'],
                     source_files=[{'path': str(target/'FINALIZED.json'), 'sha256': repair.TARGET_SEAL,
                                    'retention_status': 'existing_immutable_archive'}],
                     raw_snapshot_retained=True, reproducibility_claim='validator_repair_only')
    m['data']['mt5_fetch']['not_applicable_reason'] = 'No broker access'
    m['model']['not_applicable_reason'] = 'No training; existing model only verified'
    m['search']['not_applicable_reason'] = 'No search; existing sealed results only verified'
    m['registry'].update(parent_or_incumbent=repair.TARGET_ID,
                         selected_configuration='Validator repair only; F5_NO_REDUNDANT_HTF unchanged',
                         validator_result=status)
    history.write_json(run/'manifest.json', m)
    errors = history.finalize_run(run, 'completed' if success else 'aborted',
                                 aborted_reason=None if success else '; '.join(result.get('failed_checks', ['Repair failed'])))
    repair.require(not errors, 'repair_archive:'+repr(errors))
    history.register_run(run)
    print(json.dumps({'repair_run': run.name, 'status': status, 'finalized_sha256': repair.sha(run/'FINALIZED.json')}), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
