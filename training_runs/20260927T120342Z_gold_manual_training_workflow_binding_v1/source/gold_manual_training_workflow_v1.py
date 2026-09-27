"""Manual binding gate. No actual training entrypoint is approved in this version."""
import hashlib
import json
import os
import secrets
import time
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
    consume_receipt(token)
    from manual_training_launcher_v1 import CONFIG, load, verify_environment
    verify_environment(load(CONFIG))
    state = binding_status()
    raise ValueError('訓練流程尚未核准（PARTIAL）：production 重現、B0 與 S4 尚無唯一可重複執行預設；'
                     '歷史 CSV／MT5 時間語義與資料替換亦未認證。請查看 README_GOLD_MANUAL_TRAINING_V1.md。'
                     '\nTRAINING WORKFLOW: ' + state['workflow'] + '\n未下載資料、未啟動訓練。')


if __name__ == '__main__':
    raise SystemExit('請由使用者雙擊 RUN_TRAINING.bat；本版本尚未核准實際訓練')
