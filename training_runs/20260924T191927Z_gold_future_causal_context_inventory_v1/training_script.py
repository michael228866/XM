"""Execute the preregistered file inventory once from clean pushed main."""
import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
import gold_future_causal_context_inventory_v1 as inventory

ROOT = Path(__file__).resolve().parent
SPEC = 'execution_spec_gold_future_causal_context_inventory_v1.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def protection(spec):
    current = {n: inventory.sha(ROOT/n) for n in spec['protected_sha256']}
    if current != spec['protected_sha256']:
        raise RuntimeError('PRODUCTION_CHANGED=true; STOP; do not restore silently')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    spec = inventory.load(ROOT/SPEC)
    protected = protection(spec)
    commit = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    if git('status', '--porcelain', '-z') or commit != remote or git('branch', '--show-current').decode().strip() != 'main':
        raise RuntimeError('Clean pushed main required before create')
    for name, expected in spec['preserved_sha256'].items():
        if inventory.sha(ROOT/name) != expected:
            raise RuntimeError('Preserved input changed: '+name)
    for name, expected in spec['prior_seals'].items():
        if inventory.sha(ROOT/name) != expected:
            raise RuntimeError('Prior finalized seal changed')
    for name in spec['source_files']:
        if (ROOT/name).read_bytes().replace(b'\r\n', b'\n') != git('cat-file', 'blob', commit+':'+name).replace(b'\r\n', b'\n'):
            raise RuntimeError('Uncommitted source')
    run = history.create_run(spec['experiment_name'], Path(__file__),
                             subprocess.list2cmdline([sys.executable, '-B', str(Path(__file__)), '--execute']),
                             arguments=['--execute'], seed_note='Deterministic structural inventory; no models or random search')
    print('RUN_ID='+run.relative_to(ROOT).as_posix(), flush=True)
    m = history.read_json(run/'manifest.json')
    m.update(source_commit=commit, pre_run_remote_commit=remote, pre_run_clean=True, pre_run_git_status='',
             git_branch='main', git_upstream='origin/main', input_snapshots=[], protected_sha256_before=protected)
    for name in spec['source_files']:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        m['input_snapshots'].append({'source_path': name, 'path': dest.relative_to(run).as_posix(),
                                     'sha256': inventory.sha(dest), 'retention_status': 'stored_in_run_directory_and_git'})
    shutil.copyfile(ROOT/SPEC, run/'execution_spec.json')
    shutil.copyfile(ROOT/'validate_gold_future_causal_context_inventory_v1.py', run/'validator_script.py')
    history.write_json(run/'manifest.json', m)
    with (run/'self_test_stdout.txt').open('xb') as out, (run/'self_test_stderr.txt').open('xb') as err:
        result = subprocess.run([sys.executable, '-B', 'test_gold_future_causal_context_inventory_v1.py'],
                                cwd=ROOT, stdout=out, stderr=err, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError('Preserve failed self-test; no retry')
    with (run/'stdout.txt').open('x', encoding='utf-8') as out, (run/'stderr.txt').open('x', encoding='utf-8') as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            doc, decision, binding = inventory.build(run, spec)
    after = protection(spec)
    entries = doc['timeframes']
    m1 = next(e for e in entries if e['timeframe'] == 'M1')
    flags = {k: False for k in ('strategy_outcome_inspected', 'model_loaded', 'model_trained', 'protocol_frozen',
                                'capture_activated', 'holdout_started', 'production_changed', 'production_promoted')}
    metrics = dict(source_status='PASS', self_test_status='PASS', inventory_status=decision['status'],
                   timeframe_count=len(entries), native_timeframe_count=sum(e['native_source_confirmed'] for e in entries),
                   m1_context_bars=m1['selected_context_bars'], m1_available_preboundary_bars=m1['available_preboundary_bars'],
                   m1_required_bars=4096, m1_requirement_satisfied=m1['warmup_requirement_satisfied'],
                   htf_required_bars_each=21, htf_requirements_satisfied=all(e['warmup_requirement_satisfied'] for e in entries if e['timeframe'] != 'M1'),
                   context_start=inventory.START+'Z', context_end=inventory.END+'Z', context_role='CAUSAL_CONTEXT_ONLY',
                   historically_equivalent=False, outcome_tuning=False, holdout_evidence=False,
                   timezone_compatibility='CONDITIONAL', prefix_policy_proposed='PREDECLARED_CAUSAL_REINITIALIZATION',
                   prefix_approval_ready=False, prefix_approval_ready_after_timezone=False,
                   inventory_root_sha256=binding['inventory_root_sha256'], file_manifest_sha256=binding['file_manifest_sha256'],
                   formal_run_status=decision['formal_run_status'], **flags)
    history.write_json(run/'metrics.json', metrics)
    m.update(formal_run_status=metrics['formal_run_status'], protected_sha256_after=after,
             git_status_after_execution=git('status', '--porcelain', '-z').decode('utf-8'),
             git_dirty_reason='Clean pushed source; only new run artifacts created',
             result_commit_binding='Commit containing immutable FINALIZED.json')
    data = m['data']
    data.update(symbols=['GOLD#'], data_sources=['External historical timestamp-column inventory'], source_files=m['input_snapshots'],
                timezone='UNRESOLVED; naive-clock candidate envelope only', raw_snapshot_retained=False,
                reproducibility_claim='External data bytes not in Git; sealed path/size/hash manifest detects subsequent change',
                purge_details='No strategy dataset', embargo_details='No outcomes accessed')
    for k in ('data_start_utc', 'data_end_utc', 'train_start_utc', 'train_end_utc', 'validation_start_utc', 'validation_end_utc', 'test_start_utc', 'test_end_utc'):
        data[k] = 'NOT_APPLICABLE_NO_CERTIFIED_UTC_STRATEGY_DATASET'
    for k in ('train_rows', 'validation_rows', 'test_rows'):
        data[k] = 0
    data['mt5_fetch']['used'] = False
    m['model']['not_applicable_reason'] = 'No model loaded or trained'
    m['search']['not_applicable_reason'] = 'No strategy search or warmup tuning'
    m['registry'].update({k: 'N/A: context inventory only' for k in history.REGISTRY_FIELDS})
    m['registry'].update(parent_or_incumbent='External attestation and prefix readiness v1 PARTIAL',
                         selected_configuration='Causal context inventory '+metrics['formal_run_status']+'; no approval/freeze/activation/production change', validator_result='PENDING')
    history.write_json(run/'manifest.json', m)
    text = ('# GOLD future causal context inventory v1\n\nFormal result: '+metrics['formal_run_status']+
            '\n\nAll coverage counts are conditional naive-clock structural candidates, not certified causal UTC bars. '
            'No exact file-to-server native export provenance was found. Filename and loader references do not prove native origin. '
            'native_timeframe_count counts independently confirmed origins, not matching filenames.\n\n'
            'The latest required rows with a successor inside the frozen envelope are bound by row index and timestamp. '
            'Successor is only a closure proxy pending BAR_OPEN authority; the final source row is excluded. '
            'Monthly historical context is permitted; no 21-month wait after activation is imposed. '
            'Raw files end before the envelope end; a continuous bridge to future activation remains unestablished.\n\n'
            'Gaps are UNKNOWN. Missing intervals mean nominal calendar slots, not confirmed missing market bars. '
            'External files remain local, outside Git. Their identity manifests, not their data bytes, are sealed.\n\n'
            'context_role=CAUSAL_CONTEXT_ONLY; historically_equivalent=false; outcome_tuning=false; holdout_evidence=false. '
            'Prefix approved=false; ready_after_timezone=false because file provenance and continuation also remain unresolved. '
            'No model, strategy computation, capture, holdout boundary or production change.\n')
    for name in ('report.md', 'findings.md'):
        (run/name).write_text(text, encoding='utf-8')
    shutil.copyfile(run/'stdout.txt', run/'stdout.log')
    print('FORMAL_RUN_STATUS='+metrics['formal_run_status'])


if __name__ == '__main__':
    main()
