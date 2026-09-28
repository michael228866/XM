"""User-only training entrypoint. Import and --dry-run never start training."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT/'training_launcher_config_v1.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_environment(config):
    expected = (ROOT/config['selected_interpreter']).resolve()
    if not expected.is_file() or Path(sys.executable).resolve() != expected:
        raise ValueError('找不到核准的 Python 環境')
    if list(sys.version_info[:2]) != config['python_version']:
        raise ValueError('Python 版本不符')
    for name, expected_hash in config['protected_sha256'].items():
        if digest(ROOT/name) != expected_hash:
            raise ValueError('Production hash 不符')
    policy = load(ROOT/'manual_training_policy_v1.json')
    if (policy['training_execution_owner'] != 'USER' or not policy['manual_launcher_required']
            or any(policy[k] for k in ('automatic_training_by_codex','automatic_training_by_scheduler',
                                      'automatic_training_on_startup','holdout_peeking_allowed','production_promotion_automatic'))):
        raise ValueError('手動訓練政策不符')


def verify_workflow(config):
    from training_holdout_guard_v1 import check_path
    if config['approval_status'] != 'APPROVED' or config['workflow'] is None:
        raise ValueError('尚未指定核准的訓練腳本、設定與資料集；未啟動訓練')
    workflow = config['workflow']
    for entry in [workflow['script'],workflow['validator'],workflow['configuration'],*workflow['datasets']]:
        path = check_path(ROOT/entry['path'], ROOT)
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError('訓練資料或核准設定不合法')
    if not workflow['datasets'] or workflow.get('native_io_review_pass') is not True:
        raise ValueError('缺少資料來源或原生檔案存取審核')
    return workflow


def manual_parent_chain(chain, interpreter):
    """Windows venv may insert one redirector process ahead of CMD."""
    if chain and str(chain[0].get('ExecutablePath','')).casefold() == str(interpreter).casefold():
        chain = chain[1:]
    return (len(chain) >= 2 and chain[0].get('Name','').lower() == 'cmd.exe'
            and chain[1].get('Name','').lower() == 'explorer.exe'
            and chain[0].get('SessionId',0) > 0
            and chain[0].get('ParentProcessId') == chain[1].get('ProcessId'))


def user_double_click():
    """Require CMD launched directly by Explorer, not a scheduler/agent shell."""
    if os.name != 'nt' or os.environ.get('XM_USER_TRAINING_BAT') != 'RUN_TRAINING_V1':
        return False
    command = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+"+str(os.getpid())+"); $chain=@(); for($i=0;$i -lt 3;$i++){ $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$p.ParentProcessId); if($null -eq $p){break}; $chain+=($p | Select-Object Name,ExecutablePath,SessionId,ProcessId,ParentProcessId) }; ConvertTo-Json -InputObject $chain -Compress"
    p = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],capture_output=True,timeout=15,check=False)
    if p.returncode != 0:
        return False
    return manual_parent_chain(json.loads(p.stdout.decode('utf-8-sig')), (ROOT/load(CONFIG)['selected_interpreter']).resolve())


def run_training(config):
    """Only the reviewed manual wrapper can decide whether training is available."""
    from gold_manual_training_workflow_v1 import issue_receipt, run_manual
    verify_environment(config)
    return run_manual(issue_receipt())


def main():
    try:
        print("========================================\nXM GOLD S4 Train + Validate\n========================================")
        print("[1/6] 檢查環境...")
        config = load(CONFIG)
        verify_environment(config)
        if sys.argv[1:] == ['--dry-run']:
            from gold_manual_training_workflow_v1 import binding_status
            state = binding_status()
            print(json.dumps({'infrastructure':'PASS','training_enabled':state['workflow']=='READY',
                'training_executed':False,'approval_status':config['approval_status']},ensure_ascii=False))
            return 0
        if sys.argv[1:] or not user_double_click():
            raise ValueError('只能由使用者雙擊 RUN_TRAINING.bat 啟動；排程器與自動程序禁止訓練')
        print("正在確認訓練設定...")
        from gold_manual_s4_train_validate_v1 import format_result
        result = run_training(config)
        print(format_result(result))
        return {'PASS': 0, 'PARTIAL': 2, 'FAIL': 1}[result['final_status']]
    except Exception as error:
        from gold_manual_s4_train_validate_v1 import combined_result, format_result
        result = getattr(error, 'result', None)
        print(format_result(result or combined_result(None, 'NOT_STARTED', 'NOT_RUN', errors=[str(error)])))
        print('[失敗] '+str(error),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
