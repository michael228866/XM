"""Verify source audit; regenerate continuous labels only inside USER session."""
from pathlib import Path
from gold_manual_s4_training_data_v1 import read, sha

ROOT = Path(__file__).resolve().parent
AUDIT = 'gold_s4_v4_label_pipeline_audit.json'


def verify_audit():
    audit = read(ROOT/AUDIT)
    if audit['overall'] != 'PASS':
        raise ValueError('Label pipeline source audit required')
    for name, expected in audit['source_bindings'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen label dependency changed: '+name)
    return audit


def label_source():
    from gold_s4_secondary_improvement_v4_label_objective_launcher import require_session
    require_session()
    from gold_independent_secondary_classifier_v1 import load_module
    audit = verify_audit()
    return load_module(ROOT/audit['original_label_source'], 'v4_frozen_c1')


def add_label_context(state, config, ref, old):
    from gold_s4_secondary_improvement_v4_label_objective_launcher import require_session
    require_session()
    import numpy as np
    from gold_s4_secondary_improvement_v4_label_objective_support import require, array_hash, safe_path
    audit = verify_audit()
    manifest = read(ROOT/audit['raw_dataset_manifest'])
    files = manifest['datasets']
    # Use only the already-certified cache, never fetch or substitute market data.
    directory = safe_path(ROOT, Path(files[0]['path']).parent)
    require(directory.is_relative_to(ROOT/'historical_training_data'), 'Exact historical cache only')
    require({p.name for p in directory.glob('*.csv')} == {x['filename'] for x in files}, 'No extra raw input files')
    for item in files:
        path = safe_path(directory,item['filename'])
        require(sha(path) == item['sha256'], 'Certified raw source missing/changed: '+item['filename'])
    old.drl_trading_v2.DATA_DIR = str(directory)
    frame, names = old.prepare_barrier_data()
    require(names == config['input_features'], 'Frozen original feature pipeline')
    frame = frame.reset_index(drop=True)
    labels = old.build_execution_aligned_labels(frame)
    ns = frame.TIME_DT.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    for n, data in state.items():
        train, score = data['train_indices'], data['score_indices']
        for part, indices in [('train',train),('score',score)]:
            require(np.array_equal(ns[indices],data[part+'_ns']) and
                    array_hash(frame.loc[indices,names].to_numpy(dtype=np.float32)) == array_hash(data[part+'_x']),
                    'Regenerated frame differs from certified evidence')
        require(np.array_equal(labels.C1_TARGET.to_numpy(dtype=np.int8)[train],data['target'])
                and np.array_equal(labels.C1_MATURITY_NS.to_numpy(dtype=np.int64)[train],data['maturity'])
                and np.all(labels.C1_MATURE.to_numpy()[train]), 'Exact C1 label/maturity reproduction')
        data['realized_r'] = labels.C1_NET_R.to_numpy(dtype=np.float64)[train]
        # Conservative, source-audited inclusive bound; not a measured bar-open time.
        data['feature_cutoff_time'] = data['train_ns'].copy()
    return state
