"""USER-only exact S4 secondary reproduction. Import never trains or predicts."""
import contextlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from gold_manual_s4_training_data_v1 import ROOT, prepare, read, sha, write
from gold_manual_s4_training_output_adapter_v1 import model_path, publish
from gold_manual_s4_train_validate_v1 import (
    Tee, WorkflowError, combined_result, training_handoff, validate_exact_run,
)

CONFIG = ROOT/'gold_manual_s4_secondary_retrain_config_v1.json'
EXPERIMENT = 'gold_manual_s4_secondary_retrain_v1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def verify_config(config):
    if (config['training_symbol'] != 'GOLD#' or config['dataset_mode'] != 'REPRODUCTION_DATASET'
            or config['b0_primary_threshold'] != .75 or config['secondary_threshold'] != .75
            or config['production_overwrite_allowed'] is not False
            or config['output_policy'] != 'RUN_LOCAL_CANDIDATE_ONLY'):
        raise ValueError('Frozen manual S4 contract drift')
    if sha(ROOT/'gold_manual_s4_training_data_policy_v1.json') != config['data_policy_sha256']:
        raise ValueError('Historical data policy drift')
    for name, expected in config['source_bindings'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Source identity mismatch: '+name)
    for name, expected in config['archive_seals'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen archive seal mismatch: '+name)
    for package, version in config['environment'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Approved package version mismatch: '+package)
    frozen = read(ROOT/config['discovery_spec'])
    for key, reference in [('feature_list', 'features'), ('hyperparameters', 'parameters'),
                           ('fold_definition', 'folds'), ('random_seed', 'seed')]:
        if config[key] != frozen[reference]:
            raise ValueError('Frozen S4 configuration mismatch: '+key)
    return frozen


def train_folds(run, config, discovery, helper, old, archive, data_dir):
    """Reuses frozen reconstruction, conditioning, fit, gate and S5 evaluation."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    spec = discovery.specification()
    old.drl_trading_v2.DATA_DIR = str(data_dir)
    print('讀取資料與建立 frozen features / labels...', flush=True)
    frame, target, folds, identities = helper.reconstruct(old)
    discovery.write_json(run/'identity_audit.json', identities)
    ns = frame.TIME_DT.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    evidence, inventory, frames, b0_parts, secondary_parts, fold_parts = {}, [], [], [], [], []
    with np.load(archive/'paired_oof_predictions.npz', allow_pickle=False) as original:
        for number, name, train, score in folds:
            print(f'建立 Fold {number}/3...', flush=True)
            key = f'fold{number}'
            item = next(x for x in spec['b0_models'] if x['fold'] == name)
            if sha(archive/item['path']) != item['sha256']:
                raise ValueError('Archived B0 identity mismatch')
            b0_model = xgb.XGBClassifier()
            b0_model.load_model(archive/item['path'])
            if b0_model.get_booster().feature_names != spec['features']:
                raise ValueError('Archived B0 feature ordering mismatch')
            train_prob = b0_model.predict_proba(frame.loc[train, spec['features']].astype(np.float32))[:, 1]
            score_prob = b0_model.predict_proba(frame.loc[score, spec['features']].astype(np.float32))[:, 1]
            if (not np.array_equal(score, original[key+'_indices'])
                    or not np.array_equal(score_prob, original['B0_31_technical_'+key])):
                raise ValueError('B0 OOF reconstruction mismatch')
            subset = discovery.training_subset(train, score, ns[train], ns[score], train_prob)
            if discovery.array_hash(target[train]) != spec['c1_target_sha256'][key+'_train']:
                raise ValueError('C1 target identity mismatch')
            x = frame.loc[subset, spec['features']].astype(np.float32)
            print(f'訓練 Fold {number}/3...', flush=True)
            model, weights = discovery.fit_secondary(x, target[subset], spec)
            path = model_path(run, name)
            model.save_model(path)
            values = discovery.probabilities(model.predict_proba(frame.loc[score, spec['features']].astype(np.float32))[:, 1])
            evidence.update({key+'_train_indices': train, key+'_score_indices': score,
                             key+'_subset_indices': subset, key+'_b0_train': train_prob,
                             key+'_b0_score': score_prob, key+'_subset_target': target[subset],
                             key+'_train_ns': ns[train], key+'_score_ns': ns[score],
                             key+'_secondary_score': values, key+'_sample_weight': weights})
            inventory.append({'fold': name, 'path': path.relative_to(run).as_posix(), 'sha256': sha(path),
                              'parameters': spec['parameters'], 'features': spec['features'], 'seed': 42,
                              'rounds': model.get_booster().num_boosted_rounds(), 'train_rows': len(subset),
                              'train_indices_sha256': discovery.array_hash(subset),
                              'x_sha256': discovery.array_hash(x.to_numpy()),
                              'target_sha256': discovery.array_hash(target[subset]),
                              'weights_sha256': discovery.array_hash(weights)})
            part = frame.loc[score, helper.PRICE_COLUMNS].copy()
            part['buy_prob'], part['sell_prob'] = score_prob, np.float32(0)
            frames.append(part)
            b0_parts.append(score_prob)
            secondary_parts.append(values)
            fold_parts.append(np.full(len(score), number, dtype=np.int8))
    # Only the fixed S4 gate; no threshold sweep, baseline fitting or confirmation rerun.
    result, trades, gates = discovery.evaluate(old, helper, pd.concat(frames, ignore_index=True),
        np.concatenate(b0_parts), np.concatenate(secondary_parts), np.concatenate(fold_parts),
        ('S4_SECONDARY_P075', .75), spec)
    for suffix, values in zip(('primary', 'secondary', 'union'), gates):
        evidence['S4_SECONDARY_P075_'+suffix] = values
    np.savez_compressed(run/'secondary_evidence.npz', **evidence)
    discovery.write_csv(run/'trade_ledger.csv', trades)
    discovery.write_json(run/'historical_metrics.json', result)
    discovery.write_json(run/'model_inventory.json', inventory)
    return result, inventory


def run_manual(token):
    from gold_manual_training_workflow_v1 import consume_receipt, verify_s4_approval
    from manual_training_launcher_v1 import verify_environment, load, CONFIG as launcher_config
    consume_receipt(token)
    verify_environment(load(launcher_config))
    verify_s4_approval()
    config = read(CONFIG)
    spec = verify_config(config)
    commit = git('rev-parse', 'HEAD')
    if (git('status', '--porcelain') or git('branch', '--show-current') != 'main'
            or git('rev-parse', '@{u}') != commit or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit):
        raise ValueError('訓練前必須為乾淨且已推送的 main')
    print('[2/6] 檢查歷史資料...', flush=True)
    data_dir, dataset = prepare(config)
    print('資料檢查完成')
    import training_run_history as history
    print('[3/6] 建立訓練 Run...', flush=True)
    run = history.create_run(EXPERIMENT, Path(__file__), 'RUN_TRAINING.bat (USER double-click)',
                             arguments=[], seeds={'secondary': 42}, seed_note='Three frozen CPU single-thread secondary folds')
    print('RUN_ID: '+run.name)
    m = read(run/'manifest.json')
    m.update(manual_start=True, started_by='USER_LAUNCHER', source_commit=commit, pre_run_clean=True,
             launch_attestation='process-local single-use 30-second receipt consumed before preflight')
    write(run/'manifest.json', m)
    before = {n: sha(ROOT/n) for n in config['protected_sha256']}
    write(run/'production_before.json', before)
    write(run/'training_dataset_manifest.json', dataset)
    shutil.copyfile(CONFIG, run/'approved_training_config.json')
    shutil.copyfile(ROOT/'gold_manual_s4_training_data_policy_v1.json', run/'data_policy.json')
    shutil.copyfile(ROOT/'validate_gold_manual_s4_training_run_v1.py', run/'validator_script.py')
    write(run/'execution_spec.json', spec)
    train_status, validator_status = 'FAIL', 'NOT_RUN'
    candidate, metrics = None, {}
    combined = None
    try:
        print('[4/6] 訓練模型...', flush=True)
        # Import the reviewed frozen dependency graph before native-library loading is locked.
        import gold_independent_secondary_classifier_v1 as discovery
        archive, helper, old = discovery.frozen_inputs(spec)
        from training_holdout_guard_v1 import install
        release = install(ROOT, write_root=run)
        try:
            with (run/'training_stdout.txt').open('x', encoding='utf-8') as out, (run/'training_stderr.txt').open('x', encoding='utf-8') as err, contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
                result, inventory = train_folds(run, config, discovery, helper, old, archive, data_dir)
        finally:
            release()
        after = {n: sha(ROOT/n) for n in config['protected_sha256']}
        if before != after or after != config['protected_sha256']:
            raise ValueError('Protected production identity changed')
        write(run/'production_after.json', after)
        candidate = publish(run, config, inventory, sha(Path(__file__)), sha(CONFIG), sha(run/'training_dataset_manifest.json'))
        row = result['metrics'][-1]
        metrics = {'formal_run_status': 'PENDING', 'research_verdict': result['assessment'],
                   'classification': 'historical_development_reproduction_only',
                   'trades': row['trades'], 'wins': row['wins'], 'losses': row['losses'],
                   'realized_win_rate': row['realized_wr'], 'trades_per_day': row['trades_per_day'],
                   'profit_factor': row['pf'], 'mean_r': row['mean_r'], 'pnl_r': row['pnl_r'],
                   'max_drawdown_r': row['max_dd_r'], 'stress_pf': row['cost_stress_pf'],
                   'production_promoted': False, 'validator_status': 'PENDING'}
        write(run/'metrics.json', metrics)
        m = read(run/'manifest.json')
        m['data'] = read(archive/'manifest.json')['data']
        m['data'].update(source_files=dataset['datasets'], data_sources=['Exact S4 legacy reproduction dataset'],
                         raw_snapshot_retained=True, reproducibility_claim='Exact SHA256 local content-addressed cache; not remote raw backup',
                         train_rows=sum(i['train_rows'] for i in inventory),
                         secondary_training_rows_by_fold={i['fold']: i['train_rows'] for i in inventory})
        if dataset['mt5_fetches']:
            m['data']['mt5_fetch'] = {'used': True, 'terminal_path': r'D:\XM2\terminal64.exe',
                'terminal_info': {'requested_path': r'D:\XM2\terminal64.exe'},
                'broker_info': {'company': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'trade_mode': 0},
                'fetch_start_utc': dataset['preparation_started_at_utc'],
                'fetch_end_utc': dataset['preparation_finished_at_utc'],
                'retrieved_at_utc': dataset['preparation_finished_at_utc'],
                'returned_rows': sum(x['returned_rows'] for x in dataset['mt5_fetches']),
                'requests': dataset['mt5_fetches'], 'acceptance': 'Original export SHA256 exact restoration'}
        m['model'] = {**read(ROOT/'training_runs'/spec['c1_run']/'manifest.json')['model'],
                      'trained': True, 'model_type': config['model_type'], 'parameters': config['hyperparameters'],
                      'features': config['feature_list'], 'feature_count': 31,
                      'artifact_path': candidate['model_path'], 'artifact_sha256': candidate['model_sha256'],
                      'retention_status': 'stored_in_run_directory', 'boosted_rounds_or_estimators': 220}
        m['search'] = {'performed': False, 'not_applicable_reason': 'Only frozen S4 .75; no tuning'}
        m['registry'].update(parent_or_incumbent=config['discovery_run'], selected_configuration='S4 .75 exact historical secondary reproduction; no promotion',
                              validator_result='PENDING', trades_per_day=row['trades_per_day'], realized_win_rate=row['realized_wr'],
                              pf=row['pf'], mean_r=row['mean_r'], pnl=row['pnl_r'], max_dd=row['max_dd_r'])
        (run/'model.sha256').write_text(candidate['model_sha256']+'  '+candidate['model_path']+'\n', encoding='utf-8')
        (run/'report.md').write_text('# Manual S4 secondary retraining\n\nHistorical reproduction only; no production promotion.\n\n'+json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
        write(run/'manifest.json', m)
        training_handoff(run, candidate)
        train_status, validator_status = 'PASS', 'FAIL'
        print('[5/6] 驗證訓練結果...', flush=True)
        validation = validate_exact_run(run)
        validator_status = validation['validator_status']
        combined = combined_result(run, train_status, validator_status, candidate, metrics, validation['failed_checks'])
        if combined['final_status'] == 'FAIL':
            raise ValueError('; '.join(validation['failed_checks']) or 'Independent validator failed')
        metrics['validator_status'] = validator_status
        metrics['formal_run_status'] = combined['final_status']
        write(run/'metrics.json', metrics)
        (run/'report.md').write_text('# Manual S4 Train + Validate\n\n'+json.dumps(combined, indent=2)+'\n', encoding='utf-8')
        m['registry']['validator_result'] = validator_status
        m['artifacts'] = [{'path': p.relative_to(run).as_posix(), 'sha256': sha(p),
                           'retention_status': 'stored_in_run_directory_and_git'}
                          for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('manifest.json', 'stdout.log')]
        write(run/'manifest.json', m)
    except BaseException as error:
        import traceback
        with (run/'training_stderr.txt').open('a', encoding='utf-8') as err:
            traceback.print_exc(file=err)
        changed = any(not (ROOT/n).is_file() or sha(ROOT/n) != h for n, h in config['protected_sha256'].items())
        combined = combined_result(run, train_status, validator_status, candidate, metrics,
                                   [type(error).__name__+': '+str(error)], changed)
        write(run/'metrics.json', {**metrics, 'formal_run_status': 'FAIL', 'validator_status': validator_status})
        (run/'report.md').write_text('# Manual S4 Train + Validate failure\n\n'+json.dumps(combined, indent=2)+'\n', encoding='utf-8')
        for name in ('training_stdout.txt', 'validator_stdout.txt', 'validator_stderr.txt'):
            (run/name).touch(exist_ok=True)
        if not (run/'training_result.json').exists():
            write(run/'training_result.json', {'run_id': run.name, 'train_status': train_status,
                  'candidate_model_path': None, 'candidate_model_sha256': None,
                  'metrics_path': None, 'training_completed_at_utc': None})
        write(run/'failure.json', {'error': type(error).__name__+': '+str(error), 'production_promoted': False})
        current = read(run/'manifest.json')
        current['model_training_attempted'] = True
        current['completed_fold_models'] = [p.relative_to(run).as_posix() for p in (run/'models').glob('*.json')]
        current['registry']['validator_result'] = validator_status
        current['registry']['selected_configuration'] = 'S4 train='+train_status+'; validation='+validator_status+'; final=FAIL; no promotion'
        write(run/'manifest.json', current)
        print('[6/6] 封存結果...', flush=True)
        try:
            errors = history.finalize_run(run, 'aborted', aborted_reason=str(error))
            if errors:
                raise ValueError('; '.join(errors))
            history.register_run(run)
            archive_git(run)
        except Exception as archive_error:
            raise WorkflowError(combined, archive_error) from archive_error
        return combined
    print('[6/6] 封存結果...', flush=True)
    try:
        errors = history.finalize_run(run, 'research_only')
        if errors:
            raise ValueError('Archive validation failed: '+'; '.join(errors))
        history.register_run(run)
        archive_git(run)
    except Exception as error:
        raise WorkflowError(combined, error) from error
    return combined


def archive_git(run):
    # Normal immutable archive commit/push; no candidate promotion.
    prefix = run.relative_to(ROOT).as_posix()+'/'
    staged = git('diff', '--cached', '--name-only').splitlines()
    if any(p != 'TRAINING_RUNS.md' and not p.startswith(prefix) for p in staged):
        raise ValueError('Unrelated staged changes; archive preserved but not committed')
    paths = [p.relative_to(ROOT).as_posix() for p in run.rglob('*') if p.is_file()]
    subprocess.run(['git', 'add', '-f', '--', *paths, 'TRAINING_RUNS.md'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'Archive user-started GOLD S4 secondary retraining '+run.name], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    if git('ls-remote', 'origin', 'refs/heads/main').split()[0] != git('rev-parse', 'HEAD'):
        raise ValueError('Archive remote verification failed')
    if git('status', '--porcelain'):
        raise ValueError('Archive pushed; unrelated worktree changes remain')


if __name__ == '__main__':
    raise SystemExit('請由使用者雙擊 RUN_TRAINING.bat；禁止直接或自動執行訓練')
