"""User-only training entrypoint. Import and --dry-run never start training."""
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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


def user_double_click():
    """Require CMD launched directly by Explorer, not a scheduler/agent shell."""
    if os.name != 'nt' or os.environ.get('XM_USER_TRAINING_BAT') != 'RUN_TRAINING_V1':
        return False
    command = "$p=Get-CimInstance Win32_Process -Filter ('ProcessId='+"+str(os.getpid())+"); $c=Get-CimInstance Win32_Process -Filter ('ProcessId='+$p.ParentProcessId); $e=Get-CimInstance Win32_Process -Filter ('ProcessId='+$c.ParentProcessId); [bool](($c.Name -ieq 'cmd.exe') -and ($e.Name -ieq 'explorer.exe') -and ($c.SessionId -gt 0))"
    p = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],capture_output=True,timeout=15,check=False)
    return p.returncode == 0 and p.stdout.decode('utf-8-sig').strip().lower() == 'true'


def import_approved(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training(config):
    """Called only after explicit Explorer/BAT ownership, policy and Git checks."""
    import training_run_history as history
    from training_holdout_guard_v1 import install
    workflow = verify_workflow(config)
    git = lambda *a: subprocess.check_output(['git',*a],cwd=ROOT).decode().strip()
    source_commit = git('rev-parse','HEAD')
    if git('status','--porcelain') or git('ls-remote','origin','refs/heads/main').split()[0] != source_commit:
        raise ValueError('訓練來源必須先提交並推送，且工作樹必須乾淨')
    run = history.create_run(workflow['experiment_name'], ROOT/workflow['script']['path'],
        'RUN_TRAINING.bat (user double-click)', arguments=[], seed_note='See approved versioned configuration')
    manifest = load(run/'manifest.json')
    manifest.update(manual_start=True,started_by='USER_LAUNCHER',launcher_version='manual_training_launcher_v1',
        launcher_sha256=digest(Path(__file__)),config_sha256=digest(CONFIG),source_commit=source_commit)
    history.write_json(run/'manifest.json',manifest)
    # Open logs before the irreversible child-process-local audit hook.
    with (run/'stdout.txt').open('w',encoding='utf-8') as out, (run/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            install(ROOT)
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                module = import_approved(ROOT/workflow['script']['path'],'approved_manual_training')
                module.run_training(run, load(ROOT/workflow['configuration']['path']))
                with (run/'validator_attempt.json').open('x',encoding='utf-8') as marker:
                    json.dump({'rule':'Do not retry validation','started_at_utc':datetime.now(timezone.utc).isoformat()},marker)
                validator = import_approved(ROOT/workflow['validator']['path'],'approved_manual_validator')
                validation = validator.validate(run)
                history.write_json(run/'validator.json',validation)
                if validation['overall'] != 'PASS':
                    raise ValueError('獨立驗證未通過')
            out.flush(); err.flush()
            (run/'stdout.log').write_text((run/'stdout.txt').read_text(encoding='utf-8'),encoding='utf-8')
            errors = history.finalize_run(run,status='research_only')
            if errors:
                raise ValueError('封存驗證失敗: '+'; '.join(errors))
            history.register_run(run)
            print('========================================\n訓練完成\n========================================')
            print('RUN_ID:\n'+run.name+'\nSTATUS:\n'+load(run/'metrics.json').get('formal_run_status','PARTIAL')+'\n輸出位置:\n'+str(run))
        except BaseException as error:
            out.flush(); err.flush()
            current = load(run/'manifest.json')
            current['aborted_reason']='User cancellation or training failure: '+type(error).__name__
            history.write_json(run/'manifest.json',current)
            if not (run/'FINALIZED.json').exists():
                errors = history.finalize_run(run,status='aborted',aborted_reason=current['aborted_reason'])
                if not errors:
                    history.register_run(run)
            raise


def main():
    try:
        config = load(CONFIG)
        verify_environment(config)
        if sys.argv[1:] == ['--dry-run']:
            print(json.dumps({'infrastructure':'PASS','training_enabled':config['approval_status']=='APPROVED',
                'training_executed':False,'approval_status':config['approval_status']},ensure_ascii=False))
            return 0
        if sys.argv[1:] or not user_double_click():
            raise ValueError('只能由使用者雙擊 RUN_TRAINING.bat 啟動；排程器與自動程序禁止訓練')
        run_training(config)
        return 0
    except Exception as error:
        print('[失敗] '+str(error),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
