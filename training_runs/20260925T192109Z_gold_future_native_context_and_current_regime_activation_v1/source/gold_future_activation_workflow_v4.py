"""Explicit staged freeze, activation and first-boundary preparation; no Git writes."""
import argparse
import shutil
import subprocess
from pathlib import Path

from gold_future_capture_collector_v2 import encode, load, sha, now, utc
from gold_future_capture_certification_v4 import certify
from gold_future_capture_collector_v4 import capture_once
from gold_future_source_identity_v4 import session
from validate_gold_future_native_context_v1 import validate

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout'/'gold_s4_v4'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def clean_pushed():
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or commit != git('ls-remote', 'origin', 'refs/heads/main').split()[0]:
        raise ValueError('CLEAN_PUSHED_SOURCE_REQUIRED')
    return commit


def write_new(path, value):
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(encode(value)); f.flush(); os.fsync(f.fileno())


def archived_pass(run):
    seal = load(run/'FINALIZED.json')
    for path, expected in seal['file_sha256'].items():
        if sha((run/path).read_bytes()) != expected:
            raise ValueError('SEALED_RUN_CHANGED')
    if load(run/'validator.json')['overall'] != 'PASS' or load(run/'metrics.json')['formal_run_status'] != 'PASS':
        raise ValueError('FORMAL_AND_VALIDATOR_PASS_REQUIRED')
    if certify(run)['status'] != 'READY_FOR_CAPTURE_ACTIVATION':
        raise ValueError('CAPTURE_NOT_READY')
    spec = load(run/'execution_spec.json')
    for name, expected in (spec['protected_sha256'] | spec['strategy_binding_sha256']).items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('PROTECTED_OR_FROZEN_STRATEGY_SOURCE_CHANGED')


def freeze(run):
    result_commit = clean_pushed()
    archived_pass(run)
    with session():
        pass  # Fresh exact source identity required before the freeze action.
    inventory = load(run/'native_context_inventory.json')
    prefix = load(run/'prefix_protocol.json')
    protocol = load(ROOT/'gold_future_capture_protocol_v4.json')
    spec = load(run/'execution_spec.json')
    effective = now()
    if utc(inventory['context_fetch_completed_at_utc']) >= utc(effective):
        raise ValueError('CONTEXT_MUST_PRECEDE_FREEZE')
    document = {**protocol, 'template': False, 'status': 'FROZEN',
        'protocol_freeze_effective_at_utc': effective, 'freeze_timestamp_basis': 'Actual protocol declaration creation UTC; containing commit recorded by subsequent activation',
        'result_commit': result_commit, 'run_id': run.relative_to(ROOT).as_posix(),
        'finalized_sha256': sha((run/'FINALIZED.json').read_bytes()),
        'source_commit': load(run/'manifest.json')['source_commit'],
        'context_inventory_root_sha256': inventory['inventory_root_sha256'],
        'context_manifests_root_sha256': load(run/'native_context_file_manifest.json')['manifests_root_sha256'],
        'context_fetch_source_commit': inventory['entries'][0]['source_commit'],
        'prefix_approved': True, 'prefix_protocol_sha256': sha(encode(prefix)),
        'prefix_protocol': prefix, 'historically_equivalent': False, 'context_role': 'CAUSAL_CONTEXT_ONLY',
        'code_sha256': protocol['runtime_code_sha256'],
        'feature_names': load(ROOT/'execution_spec_gold_future_holdout_data_foundation_v1.json')['required_features'],
        'feature_implementation_sha256': prefix['feature_implementation_sha256'],
        'candidate': {'id': 'S4', 'primary': 'B0 >= 0.75', 'secondary': 'B0 < 0.75 AND secondary >= 0.75'},
        'execution': {'semantics': 'S5', 'source_sha256': spec['strategy_binding_sha256']},
        'unlock_rule': 'One-shot evaluation only after 365 calendar days from recorded HOLDOUT_START, complete independently validated capture and separately authorized unlock; no preview or interim strategy metrics',
        'context_bridge_policy': 'No context refresh; evaluation_start remains null unless every frozen feature prerequisite independently verified without models',
        'production_promotion': False}
    write_new(ROOT/'gold_future_locked_holdout_protocol_v4.json', document)
    with (ROOT/'GOLD_FUTURE_LOCKED_HOLDOUT_PROTOCOL_V4.md').open('x', encoding='utf-8') as f:
        f.write('# GOLD future locked holdout protocol v4\n\nFrozen declaration: '+effective+'\n\nCanonical binding: gold_future_locked_holdout_protocol_v4.json.\n\nS4: primary B0 >= 0.75; secondary B0 < 0.75 and specialist >= 0.75. Exact 31 features and S5 source hashes frozen. No model or strategy execution before one-shot unlock after 365 calendar days and independent data validation.\n\nCurrent offset 10800 only. Offset 7200 pauses admission pending a separate certified immutable amendment. Unknown offset, stale clocks, source changes or integrity failures pause and quarantine. No automatic resume.\n\nNative context is causal context only, not historical equivalence or holdout evidence. No context refresh after the evidence boundary. No production promotion or production change.\n')
    print('PROTOCOL_PREPARED='+effective)


def activate(run):
    protocol_commit = clean_pushed()
    archived_pass(run)
    frozen_path = ROOT/'gold_future_locked_holdout_protocol_v4.json'
    frozen = load(frozen_path)
    committed = subprocess.check_output(['git', 'cat-file', 'blob', protocol_commit+':'+frozen_path.name], cwd=ROOT)
    if committed != frozen_path.read_bytes():
        raise ValueError('FREEZE_COMMIT_BYTES_MISMATCH')
    with session() as (_, identity):
        pass
    inventory = load(run/'native_context_inventory.json')
    native = validate(run/'context_seed', frozen['source_commit'], sha((ROOT/'gold_future_native_context_fetch_v1.py').read_bytes()))
    if native['inventory_root_sha256'] != frozen['context_inventory_root_sha256']:
        raise ValueError('FROZEN_CONTEXT_MISMATCH')
    for name, expected in frozen['code_sha256'].items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('FROZEN_CODE_CHANGED')
    certificate = {'status': 'READY_FOR_CAPTURE_ACTIVATION', 'freeze_sha256': sha(encode(frozen)),
                   'protocol_commit': protocol_commit, 'verified_at_utc': now(),
                   'source_identity': identity, 'native_context_validation': native}
    if CAPTURE.exists():
        raise FileExistsError('CAPTURE_ROOT_ALREADY_EXISTS')
    CAPTURE.mkdir(parents=True)
    for name in ('attestations', 'protocol', 'snapshots', 'manifests', 'pending', 'quarantine'):
        (CAPTURE/name).mkdir()
    shutil.copytree(run/'context_seed', CAPTURE/'context_seed')
    write_new(CAPTURE/'protocol'/'freeze.json', frozen)
    write_new(CAPTURE/'attestations'/'recertification.json', certificate)
    write_new(CAPTURE/'attestations'/'time.json', load(run/'runtime_time_validation_spec.json'))
    write_new(CAPTURE/'attestations'/'source.json', identity)
    write_new(CAPTURE/'protocol'/'prefix.json', load(run/'prefix_protocol.json'))
    effective = now()
    if utc(effective) <= utc(frozen['protocol_freeze_effective_at_utc']):
        raise ValueError('ACTIVATION_BEFORE_FREEZE')
    activation = {'status': 'ACTIVE', 'template': False, 'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#',
        'source_identity': identity, 'collector_version': 'gold_future_capture_collector_v4',
        'collector_commit': frozen['source_commit'], 'protocol_freeze_commit': protocol_commit,
        'protocol_freeze_sha256': sha(encode(frozen)),
        'protocol_freeze_effective_at_utc': frozen['protocol_freeze_effective_at_utc'],
        'activation_requested_at_utc': effective, 'activation_effective_at_utc': effective,
        'current_certified_offset_seconds': 10800, 'runtime_fail_closed': True,
        'prefix_policy': 'PREDECLARED_CAUSAL_REINITIALIZATION',
        'context_inventory_root_sha256': inventory['inventory_root_sha256'],
        'capture_root': str(CAPTURE), 'capture_mode': 'BOUNDED_RAW_CAPTURE_UNTIL_FIRST_LEGAL_BOUNDARY; no strategy execution'}
    write_new(CAPTURE/'attestations'/'activation.json', activation)
    # The source template remains immutable; actual activation lives in its isolated root.
    write_new(ROOT/'gold_future_capture_activation_record_v4.json', activation)
    print('ACTIVATION_PREPARED='+effective)


def boundary():
    activation_commit = clean_pushed()
    item = capture_once(CAPTURE)
    if item is None:
        print('WAITING_FOR_FIRST_CLOSED_FUTURE_M1')
        return
    frozen = load(CAPTURE/'protocol'/'freeze.json')
    activation = load(CAPTURE/'attestations'/'activation.json')
    payload = load(CAPTURE/item['snapshot_path'])
    row = payload['rows'][0]
    document = {'protocol_commit': activation['protocol_freeze_commit'], 'activation_commit': activation_commit,
        'raw_epoch': row['RAW_SOURCE_EPOCH'], 'normalized_source_timestamp': row['SOURCE_TIMESTAMP'],
        'observed_offset_seconds': 10800, 'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#',
        'context_inventory_root_sha256': frozen['context_inventory_root_sha256'],
        'prefix_protocol_sha256': frozen['prefix_protocol_sha256'],
        'first_manifest_sequence': item['sequence'], 'first_manifest_sha256': sha(encode(item)),
        'snapshot_sha256': item['snapshot_sha256'], 'runtime_time_validation': 'PASS',
        'historical_context_only': True, 'historical_rows_holdout_evidence': False,
        'strategy_outcome_inspected': False, 'holdout_start': row['SOURCE_TIMESTAMP'],
        'holdout_evaluation_start': None,
        'feature_prerequisites_status': 'NOT_YET_INDEPENDENTLY_VERIFIED; no feature or model execution; no claim of continuous seed-to-boundary inputs',
        'created_at_utc': now()}
    write_new(ROOT/'gold_future_holdout_boundary_v4.json', document)
    print('HOLDOUT_START='+row['SOURCE_TIMESTAMP'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'activate', 'boundary'])
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.stage == 'boundary':
        boundary()
    elif args.run is None:
        parser.error('--run required')
    else:
        (freeze if args.stage == 'freeze' else activate)(args.run.resolve())
