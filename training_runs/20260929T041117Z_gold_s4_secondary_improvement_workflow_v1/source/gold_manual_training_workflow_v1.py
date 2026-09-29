"""Manual receipt and independently certified S4 binding gate; import never trains."""
import hashlib
import json
import os
import secrets
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_RECEIPTS = {}


def issue_receipt():
    from manual_training_launcher_v1 import user_double_click
    if not user_double_click():
        raise PermissionError('只允許使用者雙擊啟動')
    token = secrets.token_urlsafe(32)
    _RECEIPTS[token] = (os.getpid(), time.monotonic() + 30)
    return token


def consume_receipt(token):
    receipt = _RECEIPTS.pop(token, None)
    if receipt is None or receipt[0] != os.getpid() or time.monotonic() > receipt[1]:
        raise PermissionError('手動啟動憑證缺少、已過期或已使用')


def binding_status(root=ROOT):
    root = Path(root)
    approval = root/'gold_manual_s4_approval_v1.json'
    if approval.exists():
        verify_s4_approval(root)
        config = json.loads((root/'gold_manual_s4_secondary_retrain_config_v1.json').read_text(encoding='utf-8'))
        runs = sorted((root/'training_runs').glob('*_gold_manual_s4_secondary_retrain_v1/FINALIZED.json'))
        from training_holdout_guard_v1 import check_path
        inventory_hash = hashlib.sha256(json.dumps(config['required_datasets'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        ready = True
        for item in config['required_datasets']:
            found = False
            for p in (root/item['filename'], root/'historical_training_data/s4_exact'/inventory_hash/item['filename']):
                p = check_path(p, root)
                if p.is_file():
                    with p.open('rb') as stream:
                        found = hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
                    if found:
                        break
            ready = ready and found
        return {'workflow': 'READY', 'owner': 'USER', 'auto_fetch': 'ENABLED',
                'symbol': config['training_symbol'], 'historical_data': 'READY' if ready else 'INCOMPLETE',
                'last_training': runs[-1].parent.name if runs else 'NEVER', 'provenance_status': 'PASS', 'blockers': []}
    config = json.loads((root/'gold_manual_training_config_v1.json').read_text(encoding='utf-8'))
    spec_path = root/config['spec']['path']
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != config['spec']['sha256']:
        raise ValueError('訓練 spec hash 不符')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    trace = spec['provenance_trace']
    if hashlib.sha256((root/trace['path']).read_bytes()).hexdigest() != trace['sha256']:
        raise ValueError('訓練 provenance hash 不符')
    # Editing a JSON approval flag cannot authorize an unimplemented adapter.
    return {'workflow': 'NOT_READY', 'owner': 'USER', 'auto_fetch': 'DISABLED',
            'symbol': spec['training_symbol'], 'historical_data': 'INCOMPLETE',
            'last_training': config['last_training'] or 'NEVER',
            'provenance_status': spec['status'], 'blockers': spec['blockers']}


def summary(run_id, status, output):
    return f'========================================\nRUN_ID:\n{run_id}\nSTATUS:\n{status}\n輸出位置:\n{output}\n========================================'


def run_manual(token):
    from gold_manual_s4_secondary_retrain_v1 import run_manual as retrain
    return retrain(token)


def verify_s4_approval(root=ROOT):
    root = Path(root)
    path = root/'gold_manual_s4_approval_v1.json'
    if not path.is_file():
        raise ValueError('S4 手動訓練尚未完成獨立認證與 approval commit；未啟動訓練')
    approval = json.loads(path.read_text(encoding='utf-8'))
    if approval.get('approved') is not True or approval.get('training_execution_owner') != 'USER':
        raise PermissionError('Invalid S4 approval')
    committed = subprocess.check_output(['git', 'show', 'HEAD:'+path.name], cwd=root)
    if committed != path.read_bytes():
        raise PermissionError('S4 approval must be committed')
    for name, expected in approval['bindings'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('S4 approved binding changed: '+name)
    run = (root/approval['certification_run']).resolve()
    if run.parent != (root/'training_runs').resolve():
        raise PermissionError('Invalid certification path')
    seal = json.loads((run/'FINALIZED.json').read_text(encoding='utf-8'))
    if hashlib.sha256((run/'FINALIZED.json').read_bytes()).hexdigest() != approval['finalized_sha256']:
        raise ValueError('Certification seal changed')
    for name in ('validator.json', 'metrics.json'):
        if hashlib.sha256((run/name).read_bytes()).hexdigest() != seal['file_sha256'][name]:
            raise ValueError('Certification result changed')
    validator = json.loads((run/'validator.json').read_text(encoding='utf-8'))
    metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    if validator['overall'] != 'PASS' or metrics['formal_run_status'] != 'PASS':
        raise PermissionError('S4 certification must PASS')
    return approval


if __name__ == '__main__':
    raise SystemExit('請由使用者雙擊 RUN_TRAINING.bat；本版本尚未核准實際訓練')
