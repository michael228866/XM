"""Approval and status metadata only; no research execution on import/dry-run."""
import json
import subprocess
from pathlib import Path

from gold_manual_s4_training_data_v1 import ROOT, read, sha


def verify_approval(root=ROOT):
    path = Path(root)/'gold_s4_improvement_approval_v1.json'
    value = read(path)
    if value.get('approved') is not True or value.get('training_execution_owner') != 'USER':
        raise PermissionError('Improvement USER approval required')
    if subprocess.check_output(['git', 'show', 'HEAD:'+path.name], cwd=root) != path.read_bytes():
        raise PermissionError('Improvement approval must be committed')
    for name, expected in value['bindings'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target) != expected:
            raise ValueError('Improvement approved binding changed: '+name)
    run = (root/value['certification_run']).resolve()
    if run.parent != root/'training_runs' or sha(run/'FINALIZED.json') != value['finalized_sha256']:
        raise ValueError('Improvement certification identity')
    seal = read(run/'FINALIZED.json')['file_sha256']
    for name in ('validator.json', 'metrics.json'):
        if sha(run/name) != seal[name]:
            raise ValueError('Improvement certificate altered')
    if read(run/'validator.json')['overall'] != 'PASS' or read(run/'metrics.json')['formal_run_status'] != 'PASS':
        raise PermissionError('Independent improvement certification PASS required')
    return value


def binding_status(root=ROOT):
    verify_approval(root)
    runs = sorted((root/'training_runs').glob('*_gold_s4_secondary_improvement_v1/FINALIZED.json'))
    last = {'run_id': 'NEVER', 'reference_status': 'NOT_RUN', 'execution_status': 'NOT_RUN',
            'research_result': 'NEVER', 'candidate_gate': 'NONE'}
    if runs:
        run = runs[-1].parent
        seal = read(run/'FINALIZED.json')['file_sha256']
        if sha(run/'combined_result.json') != seal['combined_result.json']:
            raise ValueError('Last improvement run changed')
        last = read(run/'combined_result.json')
    return {'workflow': 'READY', 'owner': 'USER', 'last': last}
