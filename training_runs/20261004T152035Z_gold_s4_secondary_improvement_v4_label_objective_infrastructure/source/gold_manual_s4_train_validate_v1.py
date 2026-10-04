"""One-session handoff and display; no model fitting or holdout evaluation."""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from gold_manual_s4_training_data_v1 import ROOT, read, sha, write

METRICS = ('trades', 'wins', 'losses', 'realized_win_rate', 'trades_per_day',
           'profit_factor', 'mean_r', 'pnl_r', 'max_drawdown_r', 'stress_pf')


class WorkflowError(RuntimeError):
    def __init__(self, result, error):
        super().__init__(str(error))
        self.result = {**result, 'final_status': 'FAIL',
                       'failed_checks': [*result['failed_checks'], 'Archive: '+str(error)]}


def final_status(train, validator):
    if train == 'PASS' and validator in ('PASS', 'PARTIAL'):
        return validator
    return 'FAIL'


def training_handoff(run, candidate):
    path = (run/candidate['model_path']).resolve()
    if not path.is_relative_to((run/'models').resolve()) or sha(path) != candidate['model_sha256']:
        raise ValueError('Candidate path/hash mismatch')
    result = {'run_id': run.name, 'train_status': 'PASS',
              'candidate_model_path': candidate['model_path'],
              'candidate_model_sha256': candidate['model_sha256'],
              'metrics_path': 'metrics.json',
              'training_completed_at_utc': datetime.now(timezone.utc).isoformat()}
    write(run/'training_result.json', result)
    return result


def validate_exact_run(run):
    """Pass the created path directly; never discover a latest run or retry."""
    if (run/'validator_attempt.json').exists() or (run/'validator.json').exists():
        raise FileExistsError('Do not retry validation')
    handoff = read(run/'training_result.json')
    if handoff['run_id'] != run.name or handoff['train_status'] != 'PASS':
        raise ValueError('Training handoff run/status mismatch')
    with (run/'validator_stdout.txt').open('x', encoding='utf-8') as out, (run/'validator_stderr.txt').open('x', encoding='utf-8') as err:
        process = subprocess.run([sys.executable, '-B', str(ROOT/'validate_gold_manual_s4_training_run_v1.py'), str(run)],
                                 cwd=ROOT, stdout=out, stderr=err)
    result = read(run/'validator.json')
    if (result['run_id'] != run.name
            or result['validator_status'] not in ('PASS', 'PARTIAL', 'FAIL')
            or result['overall'] != result['validator_status']):
        raise ValueError('Validator handoff identity/status mismatch')
    if result['candidate_model_sha256'] != handoff['candidate_model_sha256']:
        raise ValueError('; '.join([*result['failed_checks'], 'Validator candidate SHA256 mismatch']))
    if process.returncode != {'PASS': 0, 'PARTIAL': 2, 'FAIL': 1}[result['validator_status']]:
        raise ValueError('Validator exit code/status mismatch')
    return result


def combined_result(run, train, validator, candidate=None, metrics=None, errors=(), production_changed=False):
    candidate, metrics = candidate or {}, metrics or {}
    result = {'run_id': run.name if run else None, 'train_status': train,
              'validator_status': validator, 'final_status': final_status(train, validator),
              'candidate_model_path': candidate.get('model_path'),
              'candidate_model_sha256': candidate.get('model_sha256'),
              **{key: metrics.get(key) for key in METRICS},
              'production_changed': production_changed, 'production_promoted': False,
              'holdout_used': False, 'failed_checks': list(errors)}
    if production_changed:
        result['final_status'] = 'FAIL'
    if run:
        write(run/'combined_result.json', result)
    return result


def format_result(result):
    lines = ['='*40, 'XM GOLD S4 Train + Validate'+(' 完成' if result['final_status'] == 'PASS' else ''), '='*40]
    for label, key in [('RUN_ID', 'run_id'), ('訓練', 'train_status'), ('驗證', 'validator_status'),
                       ('最終結果', 'final_status'), ('候選模型', 'candidate_model_path'), ('Model SHA256', 'candidate_model_sha256')]:
        value = result.get(key)
        if key == 'candidate_model_path' and value and result.get('run_id'):
            value = ROOT/'training_runs'/result['run_id']/value
        lines.extend([label+':', str(value or 'N/A')])
    if result.get('failed_checks'):
        lines.extend(['原因:', *result['failed_checks']])
    if result['final_status'] == 'PARTIAL':
        lines.append('候選模型已保留，但未核准 promotion。')
    lines.extend(['-'*40, 'Historical Validation Metrics'])
    for label, key in zip(('Trades', 'Wins', 'Losses', 'Win Rate', 'Trades/Day', 'PF', 'Mean-R', 'PnL', 'Max DD', 'Stress PF'), METRICS):
        value = result.get(key)
        lines.extend([label+':', (str(value)+(' R' if key in ('pnl_r', 'max_drawdown_r') else '')) if value is not None else 'N/A'])
    lines.extend(['Production:', '已變更：檢查失敗' if result['production_changed'] else '未變更',
                  'Locked Future Holdout:', '未使用', '='*40])
    return '\n'.join(lines)


def last_result(root):
    """Status display only; never used to select a run for validation."""
    runs = sorted((root/'training_runs').glob('*_gold_manual_s4_secondary_retrain_v1/FINALIZED.json'))
    if not runs:
        return {'run_id': 'NEVER', 'train_status': 'NOT_STARTED', 'validator_status': 'NOT_RUN',
                'final_status': 'NOT_RUN', 'candidate_model_path': None}
    run = runs[-1].parent
    seal = read(run/'FINALIZED.json')['file_sha256']
    def sealed(name):
        if sha(run/name) != seal[name]:
            raise ValueError('Last run sealed artifact changed: '+name)
        return read(run/name)
    if 'combined_result.json' in seal:
        return sealed('combined_result.json')
    # Earlier sealed runs predate combined_result; do not edit or revalidate them.
    manifest = sealed('manifest.json')
    validator = sealed('validator.json') if 'validator.json' in seal else {}
    candidate = sealed('candidate_model_manifest.json') if 'candidate_model_manifest.json' in seal else {}
    train = 'PASS' if candidate and manifest['status'] != 'aborted' else 'FAIL'
    validation = validator.get('overall', 'NOT_RUN')
    return {'run_id': run.name, 'train_status': train, 'validator_status': validation,
            'final_status': final_status(train, validation), 'candidate_model_path': candidate.get('model_path')}


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        self.console.write(text)
        return len(text)

    def flush(self):
        self.log.flush()
        self.console.flush()
