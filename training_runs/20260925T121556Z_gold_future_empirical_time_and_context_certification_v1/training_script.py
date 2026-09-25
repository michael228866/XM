"""Execute empirical diagnostics and immutable context review from pushed source."""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
import gold_future_capture_certification_v3 as cert
from gold_future_empirical_time_v1 import decision

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_future_empirical_time_and_context_certification_v1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def protection(spec):
    actual = {n: cert.sha(ROOT/n) for n in spec['protected_sha256']}
    if actual != spec['protected_sha256']:
        raise RuntimeError('PRODUCTION_CHANGED=true; STOP; no silent restore')
    return actual


def child(run, label, args, spec):
    protection(spec)
    with (run/(label+'_stdout.txt')).open('xb') as out, (run/(label+'_stderr.txt')).open('xb') as err:
        result = subprocess.run([sys.executable, '-B', *args], cwd=ROOT, stdout=out, stderr=err, timeout=60, check=False)
    protection(spec)
    if result.returncode:
        raise RuntimeError('Preserve failed child; no retry: '+label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    spec = cert.load(ROOT/('execution_spec_'+SLUG+'.json'))
    before = protection(spec)
    commit = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    if git('status', '--porcelain', '-z') or commit != remote or git('branch', '--show-current').decode().strip() != 'main':
        raise RuntimeError('Clean pushed main required')
    for n, h in (spec['preserved_sha256'] | spec['prior_seals']).items():
        if cert.sha(ROOT/n) != h:
            raise RuntimeError('Preserved input changed: '+n)
    for n in spec['source_files']:
        if (ROOT/n).read_bytes().replace(b'\r\n', b'\n') != git('cat-file', 'blob', commit+':'+n).replace(b'\r\n', b'\n'):
            raise RuntimeError('Uncommitted source')
    run = history.create_run(SLUG, Path(__file__), subprocess.list2cmdline([sys.executable, '-B', str(Path(__file__)), '--execute']), arguments=['--execute'], seed_note='Deterministic empirical arithmetic and synthetic checks; no models')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = history.read_json(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=remote, pre_run_clean=True, pre_run_git_status='', git_branch='main', git_upstream='origin/main', input_snapshots=[], protected_sha256_before=before)
    for n in spec['source_files']:
        dest = run/'source'/n; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/n, dest)
        m['input_snapshots'].append({'source_path': n, 'path': dest.relative_to(run).as_posix(), 'sha256': cert.sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/('execution_spec_'+SLUG+'.json'), run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'), run/'validator_script.py')
    history.write_json(run/'manifest.json', m)
    child(run, 'self_test', ['test_'+SLUG+'.py'], spec)
    child(run, 'diagnostic', ['gold_future_empirical_time_diagnostic_v1.py', '--output', str(run/'empirical_time_segments.json')], spec)
    empirical = cert.load(run/'empirical_time_segments.json')
    time_result = decision(empirical['segments'], empirical['clock'])
    history.write_json(run/'empirical_time_decision.json', time_result)
    shutil.copyfile(ROOT/'gold_future_empirical_time_protocol_v1.json', run/'runtime_time_validation_spec.json')
    attestation = cert.load(ROOT/'gold_future_capture_time_attestation_v3.json')
    attestation.update(status=time_result['policy'], template=time_result['status'] != 'PASS',
                       empirical_coverage_start=time_result['empirical_coverage_start'], empirical_coverage_end=time_result['empirical_coverage_end'],
                       empirical_segments=empirical['segments'], empirical_evidence=[{'path': 'empirical_time_segments.json', 'sha256': cert.sha(run/'empirical_time_segments.json')}],
                       attested_by='Empirical review; scope limited to retained observations', attested_at_utc=datetime.now(timezone.utc).isoformat())
    history.write_json(run/'gold_future_capture_time_attestation_v3.json', attestation)
    source = cert.load(ROOT/'gold_future_capture_source_attestation_v3.json')
    source.update(template=not empirical['identity_verified'], identity_verified=empirical['identity_verified'], historical_file_origin_certified=False)
    history.write_json(run/'source_attestation_v3.json', source)
    prior = ROOT/spec['inventory_run']
    prior_seal = cert.load(prior/'FINALIZED.json')
    for n in ('gold_future_causal_context_inventory_v1.json', 'gold_future_causal_context_file_manifest_v1.json'):
        if cert.sha(prior/n) != prior_seal['file_sha256'][n]:
            raise RuntimeError('Prior inventory evidence changed')
        shutil.copyfile(prior/n, run/n)
    shutil.copyfile(run/'gold_future_causal_context_inventory_v1.json', run/'causal_context_inventory.json')
    shutil.copyfile(run/'gold_future_causal_context_file_manifest_v1.json', run/'causal_context_file_manifest.json')
    inventory = cert.load(run/'causal_context_inventory.json')
    files = cert.load(run/'causal_context_file_manifest.json')
    for e in files['entries']:
        if cert.sha(Path(e['path'])) != e['sha256']:
            raise RuntimeError('Context file identity changed')
    compatibility = {'status': 'PASS_STRUCTURAL_ONLY', 'formal_status': 'PARTIAL', 'official_attestation_required': False,
                     'time_rule_compatibility': 'UNRESOLVED', 'native_origin_certified': False,
                     'blockers': ['Empirical offset coverage does not cover historical context',
                                  'Original CSV-to-native-feed provenance and continuation bridge unconfirmed'],
                     'timeframes': [{'timeframe': e['timeframe'], 'gap_summary': e['gap_distribution_summary'],
                                     'time_rule_compatibility': 'UNRESOLVED', 'required_bars': e['required_warmup_bars'],
                                     'requirement_satisfied': e['warmup_requirement_satisfied']} for e in inventory['timeframes']]}
    history.write_json(run/'causal_context_compatibility.json', compatibility)
    prefix = cert.load(ROOT/'gold_recursive_prefix_protocol_v3.json')
    history.write_json(run/'prefix_protocol.json', prefix)
    history.write_json(run/'prefix_decision.json', {'status': 'PARTIAL', 'policy': 'UNRESOLVED', 'proposed_policy': prefix['policy'], 'approved': False, 'historically_equivalent': False, 'outcome_tuning': False, 'context_role': 'CAUSAL_CONTEXT_ONLY', 'holdout_evidence': False, 'blockers': compatibility['blockers']})
    report = cert.certify(run)
    history.write_json(run/'capture_certification_report.json', report)
    history.write_json(run/'collector_static_review.json', report['collector_static_review'])
    m1 = next(e for e in inventory['timeframes'] if e['timeframe'] == 'M1')
    metrics = {'source_status': 'PASS', 'self_test_status': 'PASS', 'empirical_time_status': time_result['status'],
               'time_policy': time_result['policy'], 'empirical_allowed_offsets_seconds': time_result['allowed_offsets_seconds'],
               'empirical_coverage_start': time_result['empirical_coverage_start'], 'empirical_coverage_end': time_result['empirical_coverage_end'],
               'runtime_time_validation': 'PASS', 'runtime_validation_scope': 'Implementation and synthetic tests; not activated live evidence', 'runtime_fail_closed': True,
               'context_inventory_status': 'PARTIAL', 'timeframe_count': len(inventory['timeframes']),
               'native_timeframe_count': sum(e['native_source_confirmed'] for e in inventory['timeframes']),
               'm1_context_bars': m1['selected_context_bars'], 'm1_required_bars': 4096, 'm1_requirement_satisfied': m1['warmup_requirement_satisfied'],
               'htf_required_bars_each': 21, 'htf_requirements_satisfied': all(e['warmup_requirement_satisfied'] for e in inventory['timeframes'] if e['timeframe'] != 'M1'),
               'context_role': 'CAUSAL_CONTEXT_ONLY', 'historically_equivalent': False, 'outcome_tuning': False, 'holdout_evidence': False,
               'prefix_policy': 'UNRESOLVED', 'prefix_certification_status': 'PARTIAL', 'prefix_approved': False,
               'higher_timeframe_policy': 'NATIVE_20TF_REQUIRED', 'collector_static_status': report['collector_static_review']['static_status'],
               'capture_certification_status': report['readiness']['status'], 'formal_run_status': report['formal_run_status'],
               'protocol_frozen': False, 'capture_activated': False, 'capture_start': None, 'holdout_started': False, 'holdout_start': None,
               'holdout_evaluation_start': None, 'strategy_outcome_inspected': False, 'model_loaded_for_holdout': False,
               'model_trained_for_holdout': False, 'production_changed': False, 'production_promoted': False}
    history.write_json(run/'metrics.json', metrics)
    m.update(formal_run_status=metrics['formal_run_status'], protected_sha256_after=protection(spec), git_status_after_execution=git('status', '--porcelain', '-z').decode('utf-8'), git_dirty_reason='Clean pushed source; only new formal run artifacts generated')
    data = m['data']; data.update(symbols=['GOLD#'], data_sources=['Timestamp-only bounded MT5 observations', 'Previously sealed historical context inventory'], source_files=m['input_snapshots'], timezone='Empirically supported only; no certified UTC coverage', raw_snapshot_retained=False, reproducibility_claim='Timestamp projections and external file hashes retained; no raw OHLC copied', purge_details='No strategy dataset', embargo_details='No outcomes inspected')
    for k in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[k] = 'NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for k in ('train_rows', 'validation_rows', 'test_rows'):
        data[k] = 0
    data['mt5_fetch'].update(used=True, terminal_path=r'D:\XM2\terminal64.exe', terminal_info='Read-only timestamp observations; no terminal object serialized', broker_info=empirical.get('identity', 'UNRESOLVED'), fetch_start_utc=min((s['sample_start'] for s in empirical['segments']), default='NO_SAMPLES'), fetch_end_utc=max((s['sample_end'] for s in empirical['segments']), default='NO_SAMPLES'), retrieved_at_utc=empirical['generated_at_utc'], returned_rows=sum(s['sample_count'] for s in empirical['segments']))
    m['model']['not_applicable_reason'] = 'No models loaded or trained'; m['search']['not_applicable_reason'] = 'No strategy search'
    m['registry'].update({k: 'N/A: empirical time/context infrastructure' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Causal context inventory v1 PARTIAL', selected_configuration='Empirical time v3 '+metrics['formal_run_status']+'; no freeze/activation/production change', validator_result='PENDING')
    history.write_json(run/'manifest.json', m)
    text = '# GOLD empirical time and context certification\n\nFormal result: '+metrics['formal_run_status']+'\n\nOfficial broker attestation is no longer mandatory. Holiday calendar is explanatory metadata.\n\n'+json.dumps(time_result, indent=2)+'\n\nRuntime PASS means synthetic implementation checks only. Capture was not activated.\n\nHistorical context inventory retained by exact bytes and external hashes; 21 structurally sufficient candidate timeframes, native file origin unconfirmed. Causal context only, never holdout evidence.\n\nNo model/strategy execution or production change. Prefix unapproved. No freeze, activation or boundary.\n'
    for n in ('report.md', 'findings.md'):
        (run/n).write_text(text, encoding='utf-8')
    shutil.copyfile(run/'diagnostic_stdout.txt', run/'stdout.txt'); shutil.copyfile(run/'diagnostic_stderr.txt', run/'stderr.txt'); shutil.copyfile(run/'stdout.txt', run/'stdout.log')
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'])


if __name__ == '__main__':
    main()
