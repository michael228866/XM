"""Run-local candidate manifest, with three frozen fold models, never production."""
from pathlib import Path

from gold_manual_s4_training_data_v1 import ROOT, read, sha, write


def checked_run(run, root=ROOT):
    run = Path(run).resolve()
    if run.parent != (Path(root)/'training_runs').resolve() or (run/'FINALIZED.json').exists():
        raise PermissionError('New training run directory required')
    manifest = read(run/'manifest.json')
    if manifest['run_id'] != run.name or manifest['status'] != 'in_progress':
        raise ValueError('Invalid run manifest')
    return run


def model_path(run, fold, root=ROOT):
    run = checked_run(run, root)
    if fold not in {'2018_2020', '2021_2022', '2023_2024'} or (run/'FINALIZED.json').exists():
        raise ValueError('Invalid fold or finalized run')
    path = (run/'models'/('secondary_'+fold+'.json')).resolve()
    if not path.is_relative_to(run/'models') or path.exists():
        raise PermissionError('Candidate output must be new and run-local')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_manifest(run, config, inventory, script_hash, config_hash, dataset_hash, root=ROOT):
    run = checked_run(run, root)
    if len(inventory) != 3 or [x['fold'] for x in inventory] != [f[0] for f in config['fold_definition']]:
        raise ValueError('Exactly three frozen fold models required')
    for item in inventory:
        p = (run/item['path']).resolve()
        expected = run/'models'/('secondary_'+item['fold']+'.json')
        if p != expected or not p.is_relative_to(run/'models') or sha(p) != item['sha256']:
            raise PermissionError('Invalid candidate model location/hash')
    return {'workflow_version': config['workflow_version'], 'run_id': run.name,
            'model_path': inventory[-1]['path'], 'model_sha256': inventory[-1]['sha256'],
            'model_role': 'last historical fold identity only; not a latest-date production model',
            'fold_models': inventory, 'training_script_sha256': script_hash, 'config_sha256': config_hash,
            'training_dataset_manifest_sha256': dataset_hash,
            'feature_list_sha256': config['feature_list_sha256'],
            'feature_pipeline_sha256': config['feature_pipeline_sha256'],
            'label_pipeline_sha256': config['label_pipeline_sha256'],
            'training_symbol': config['training_symbol'],
            'training_range': [config['training_start'], config['training_end']],
            'model_type': config['model_type'], 'hyperparameters': config['hyperparameters'],
            'seed': config['random_seed'], 'folds': config['fold_definition'],
            'b0_conditioning_rule': config['b0_conditioning_rule'],
            'historical_metrics_path': 'historical_metrics.json',
            'production_promoted': False, 'production_overwrite_allowed': False}


def publish(run, config, inventory, script_hash, config_hash, dataset_hash):
    value = build_manifest(run, config, inventory, script_hash, config_hash, dataset_hash)
    path = Path(run)/'candidate_model_manifest.json'
    if path.exists():
        raise FileExistsError(path)
    write(path, value)
    return value
