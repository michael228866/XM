"""Deploy the launcher binding only after an immutable independent PASS."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def approve(run):
    run = Path(run).resolve()
    if run.parent != ROOT/'training_runs' or not run.name.endswith('_gold_manual_s4_secondary_retrain_workflow_v1'):
        raise ValueError('Wrong certification run')
    if (ROOT/'gold_manual_s4_approval_v1.json').exists():
        raise FileExistsError('Approval already exists; a new version is required')
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    if git('status', '--porcelain') or git('ls-remote', 'origin', 'refs/heads/main').split()[0] != git('rev-parse', 'HEAD'):
        raise ValueError('Clean pushed archival commit required')
    seal = read(run/'FINALIZED.json')
    for name, expected in seal['file_sha256'].items():
        path = (run/name).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError('Certification archive modified')
    metrics, validator = read(run/'metrics.json'), read(run/'validator.json')
    if metrics['formal_run_status'] != 'PASS' or validator['overall'] != 'PASS' or not validator['manual_s4_binding_approved']:
        raise ValueError('Certification must PASS')
    spec = read(run/'execution_spec.json')
    for name in spec['source_files']:
        if sha(ROOT/name) != sha(run/'source'/name):
            raise ValueError('Certification source changed: '+name)
    config_name = 'gold_manual_s4_secondary_retrain_config_v1.json'
    config = read(ROOT/config_name)
    main = read(ROOT/'gold_manual_training_spec_v1.json')
    main.update(status='PASS', approval_status='APPROVED', approved=True, workflow='GOLD_MANUAL_S4_SECONDARY_RETRAIN_V1',
                workflow_class='S4_SECONDARY_RESEARCH', training_symbol='GOLD#', training_source_id=config['training_source_id'],
                training_script='gold_manual_s4_secondary_retrain_v1.py', training_script_sha256=sha(ROOT/'gold_manual_s4_secondary_retrain_v1.py'),
                training_entrypoint='run_manual (process-local receipt required)', training_config=config_name, training_config_sha256=sha(ROOT/config_name),
                feature_pipeline=config['feature_pipeline'], feature_pipeline_sha256=config['feature_pipeline_sha256'],
                ordered_feature_list=config['feature_list'], feature_list_sha256=config['feature_list_sha256'],
                label_pipeline=config['label_pipeline'], label_pipeline_sha256=config['label_pipeline_sha256'], execution_semantics='S5',
                model_type=config['model_type'], model_output_extension='.json', historical_training_start=config['training_start'],
                historical_training_end_rule='Three frozen 18-month folds; last train interval ends before 2023-01-01, strict label maturity purge; raw export cutoff 2026-05-08T23:57:00 broker clock',
                required_timeframes=config['required_timeframes'], required_symbols=['GOLD#'], random_seed=42,
                fold_definition=config['fold_definition'], hyperparameters=config['hyperparameters'], blockers=[],
                provenance_trace={'path': 'gold_manual_s4_provenance_v1.json', 'sha256': sha(ROOT/'gold_manual_s4_provenance_v1.json')},
                dataset_policy=read(ROOT/'gold_manual_s4_training_data_policy_v1.json'))
    write(ROOT/'gold_manual_training_spec_v1.json', main)
    write(ROOT/'gold_manual_training_config_v1.json', {
        'experiment_name': 'gold_manual_s4_secondary_retrain_v1', 'approval_status': 'APPROVED',
        'spec': {'path': 'gold_manual_training_spec_v1.json', 'sha256': sha(ROOT/'gold_manual_training_spec_v1.json')},
        'symbol': 'GOLD#', 'historical_date_range': [config['training_start'], config['training_end']],
        'input_datasets': config['required_datasets'], 'features': config['feature_list'], 'labels': config['secondary_target_rule'],
        'execution_semantics': 'S5', 'folds': config['fold_definition'], 'model_hyperparameters': config['hyperparameters'],
        'seed': 42, 'output_directory_policy': 'RUN_LOCAL_CANDIDATE_ONLY',
        'validator': 'validate_gold_manual_s4_training_run_v1.py', 'training_workflow_ready': True,
        'auto_fetch_enabled': True, 'last_training': None})
    launcher = read(ROOT/'training_launcher_config_v1.json')
    entry = lambda name: {'path': name, 'sha256': sha(ROOT/name)}
    launcher.update(approval_status='APPROVED', workflow={
        'experiment_name': 'gold_manual_s4_secondary_retrain_v1', 'script': entry('gold_manual_s4_secondary_retrain_v1.py'),
        'validator': entry('validate_gold_manual_s4_training_run_v1.py'), 'configuration': entry(config_name),
        'datasets': [{'path': x['filename'], 'sha256': x['sha256']} for x in config['required_datasets']],
        'native_io_review_pass': True})
    write(ROOT/'training_launcher_config_v1.json', launcher)
    inventory = read(ROOT/'gold_manual_s4_dataset_decision_v1.json')
    write(ROOT/'gold_historical_training_data_manifest_v1.json', {**inventory, 'status': 'READY', 'training_eligible': True,
          'blockers': [], 'cache_policy': 'Created on USER launch; original exact files already present'})
    mutable = ['gold_manual_training_spec_v1.json', 'gold_manual_training_config_v1.json',
               'training_launcher_config_v1.json', 'gold_historical_training_data_manifest_v1.json']
    bound = [name for name in spec['source_files'] if name not in mutable]+mutable
    receipt = {'approved': True, 'workflow': 'GOLD_MANUAL_S4_SECONDARY_RETRAIN_V1', 'training_execution_owner': 'USER',
               'certification_run': run.relative_to(ROOT).as_posix(), 'finalized_sha256': sha(run/'FINALIZED.json'),
               'source_commit': read(run/'manifest.json')['git_commit'], 'result_commit': git('rev-parse', 'HEAD'),
               'bindings': {name: sha(ROOT/name) for name in bound}, 'production_promoted': False}
    receipt['bindings'].update(config['protected_sha256'])
    write(ROOT/'gold_manual_s4_approval_v1.json', receipt)
    print('APPROVAL_PREPARED; commit and push required; NO TRAINING EXECUTED')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Expected finalized certification run path')
    approve(sys.argv[1])
