"""One formal v4 native-context/current-regime run from clean pushed source."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
from gold_future_capture_collector_v2 import load, encode, sha, now
from gold_future_current_regime_diagnostic_v1 import diagnose
from gold_future_native_context_fetch_v1 import fetch
from gold_future_capture_certification_v4 import certify, prefix_decision
from validate_gold_future_native_context_v1 import validate

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_future_native_context_and_current_regime_activation_v1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def protected(spec):
    if any(sha((ROOT/p).read_bytes()) != expected for p, expected in spec['protected_sha256'].items()):
        raise RuntimeError('PRODUCTION_CHANGED; STOP')


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Explicit --execute required')
    spec = load(ROOT/('execution_spec_'+SLUG+'.json'))
    protected(spec)
    commit = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    if git('status', '--porcelain', '-z') or commit != remote or git('branch', '--show-current').decode().strip() != 'main':
        raise ValueError('CLEAN_PUSHED_MAIN_REQUIRED')
    for p, h in (spec['preserved_sha256'] | spec['prior_seals'] | spec['strategy_binding_sha256']).items():
        if sha((ROOT/p).read_bytes()) != h:
            raise ValueError('PRIOR_ARTIFACT_CHANGED:' + p)
    run = history.create_run(SLUG, Path(__file__), subprocess.list2cmdline([sys.executable, '-B', str(Path(__file__)), '--execute']), arguments=['--execute'], seed_note='Deterministic native-context certification; no model or strategy')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = load(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=remote, pre_run_clean=True,
             pre_run_git_status='', git_branch='main', git_upstream='origin/main', input_snapshots=[])
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        m['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
                                    'sha256': sha(dest.read_bytes()), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/('execution_spec_'+SLUG+'.json'), run/'execution_spec.json')
    shutil.copyfile(ROOT/('validate_'+SLUG+'.py'), run/'validator_script.py')
    history.write_json(run/'manifest.json', m)
    with (run/'self_test_stdout.txt').open('xb') as out, (run/'self_test_stderr.txt').open('xb') as err:
        result = subprocess.run([sys.executable, '-B', str(ROOT/('test_'+SLUG+'.py'))], stdout=out, stderr=err, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError('SYNTHETIC_TEST_FAILED_PRESERVE_RUN_NO_RETRY')
    started = now()
    diagnostic = {'samples': [], 'decision': {'status': 'PARTIAL', 'classification': 'INSUFFICIENT'}}
    try:
        diagnostic = diagnose()
    except Exception as error:
        diagnostic['error'] = str(error) if isinstance(error, ValueError) else type(error).__name__
    history.write_json(run/'current_regime_diagnostic.json', diagnostic)
    history.write_json(run/'current_regime_decision.json', diagnostic['decision'])
    inventory = {'entries': [], 'inventory_root_sha256': None}
    native = {'status': 'PARTIAL', 'timeframe_count': 0, 'native_timeframe_count': 0, 'm1_context_bars': None}
    # Identity mismatch blocks further source actions, rather than trying again.
    identity_failed = 'SOURCE_' in (diagnostic.get('error') or '')
    if not identity_failed:
        try:
            inventory = fetch(run/'context_seed', commit)
            native = validate(run/'context_seed', commit, sha((ROOT/'gold_future_native_context_fetch_v1.py').read_bytes()))
        except Exception as error:
            native['error'] = str(error) if isinstance(error, ValueError) else type(error).__name__
    else:
        native['error'] = 'SOURCE_IDENTITY_BLOCKED_CONTEXT_FETCH'
    history.write_json(run/'native_context_inventory.json', inventory)
    manifest_path = run/'context_seed'/'gold_future_native_context_manifest_v1.json'
    history.write_json(run/'native_context_file_manifest.json', load(manifest_path) if manifest_path.exists() else {'entries': [], 'status': 'PARTIAL'})
    history.write_json(run/'native_context_validation.json', native)
    prefix = prefix_decision(native, inventory)
    history.write_json(run/'prefix_protocol.json', prefix)
    history.write_json(run/'prefix_decision.json', prefix)
    policy = load(ROOT/'gold_future_capture_time_attestation_v4.json')
    policy.update(template=diagnostic['decision']['status'] != 'PASS', effective_from_utc=diagnostic.get('effective_from_utc'),
                  status=diagnostic['decision']['status'])
    history.write_json(run/'runtime_time_validation_spec.json', policy)
    certificate = certify(run)
    history.write_json(run/'collector_static_review.json', certificate['collector_static_review'])
    history.write_json(run/'capture_certification_report.json', certificate)
    metrics = {'formal_run_status': certificate['formal_run_status'], 'native_context_status': native['status'],
               'current_regime_status': diagnostic['decision']['status'], 'prefix_approved': prefix['approved'],
               'capture_certification_status': certificate['status'], 'strategy_outcome_inspected': False,
               'model_loaded_for_holdout': False, 'model_trained_for_holdout': False,
               'production_changed': False, 'production_promoted': False, 'validator_status': 'PENDING'}
    history.write_json(run/'metrics.json', metrics)
    protected(spec)
    finished = now()
    data = m['data']
    data.update(symbols=['GOLD#'], data_sources=['Direct native MetaTrader5 API; sanitized account identity'],
                source_files=m['input_snapshots'], timezone='Raw source epochs; current regime only +10800 if certified',
                raw_snapshot_retained=bool(inventory['entries']), reproducibility_claim='Exact retained context bytes and bounded timestamp observations',
                purge_details='No strategy dataset', embargo_details='No strategy outcomes inspected')
    for k in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[k] = 'NOT_APPLICABLE_NO_STRATEGY_DATASET'
    for k in ('train_rows', 'validation_rows', 'test_rows'):
        data[k] = 0
    data['mt5_fetch'].update(used=True, terminal_path=r'D:\XM2\terminal64.exe',
        terminal_info='Only sanitized identity and terminal build retained', broker_info=diagnostic.get('source_identity', 'IDENTITY_UNRESOLVED'),
        fetch_start_utc=started, fetch_end_utc=finished, retrieved_at_utc=finished,
        returned_rows=sum(e['row_count'] for e in inventory['entries']))
    m['model']['not_applicable_reason'] = 'No model loaded or trained'
    m['search']['not_applicable_reason'] = 'No strategy search'
    m['registry'].update({k: 'N/A: native context/current-regime infrastructure' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='Empirical v3 PARTIAL', selected_configuration='v4 '+certificate['formal_run_status']+'; no production change; no strategy evaluation', validator_result='PENDING')
    m.update(formal_run_status=certificate['formal_run_status'], git_status_after_execution=git('status', '--porcelain', '-z').decode('utf-8'))
    history.write_json(run/'manifest.json', m)
    report = '# Native context and current regime v4\n\nFormal result: '+certificate['formal_run_status']+'\n\n'+json.dumps(metrics, indent=2)+'\n\nBlockers: '+json.dumps(certificate['blockers'])+'\n\nHistorical context is causal context only, never holdout evidence. No historical UTC regime extrapolation. No model or strategy execution. Production unchanged. Freeze/activation are separate later commits conditional on independent PASS.\n'
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    for name in ('stdout.txt', 'stdout.log'):
        (run/name).write_text('FORMAL_RUN_STATUS='+certificate['formal_run_status']+'\n', encoding='utf-8')
    (run/'stderr.txt').write_text('', encoding='utf-8')
    print('FORMAL_RUN_STATUS='+certificate['formal_run_status'])


if __name__ == '__main__':
    main()
