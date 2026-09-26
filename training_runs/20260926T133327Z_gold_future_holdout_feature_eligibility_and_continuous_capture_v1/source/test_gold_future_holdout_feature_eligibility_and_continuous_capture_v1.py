"""Synthetic-only feature causality, supervisor lock and static-safety checks."""
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import gold_future_holdout_feature_eligibility_v1 as features
import gold_future_capture_supervisor_v1 as supervisor


def main():
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    binding = features.feature_binding()
    check('exact frozen 31 features', len(binding['feature_list']) == 31)
    check('frozen pandas semantics', pd.__version__ == binding['initialization_semantics']['pandas_version'])
    epochs = np.arange(5000, dtype=np.int64)*60 + 1700000040
    close = 100 + np.sin(np.arange(5000)/9) + np.arange(5000)*.001
    frame = pd.DataFrame({'TIME_DT': pd.to_datetime(epochs, unit='s'), 'OPEN': close-.1,
                          'HIGH': close+1, 'LOW': close-1, 'CLOSE': close})
    frames = {'M1': frame.iloc[:4097].copy()}
    for name in binding['feature_list']:
        if name.endswith('_TREND'):
            frames[name[:-6]] = frame.iloc[:100].copy()
    vector = features.frozen_features(frames, binding)
    check('31 finite synthetic features', np.isfinite(vector.iloc[-1].to_numpy()).all() and vector.shape[1] == 31)
    extended = dict(frames); extended['M1'] = frame.copy()
    extended_vector = features.frozen_features(extended, binding)
    check('future suffix cannot change past vector', np.array_equal(vector.iloc[-1].to_numpy(), extended_vector.iloc[4096].to_numpy()))
    changed = dict(frames); changed['M1'] = frames['M1'].copy()
    changed['M1'].loc[4096, 'CLOSE'] += 1000
    check('final M1 lag excludes current close', np.array_equal(vector.iloc[-1].to_numpy(), features.frozen_features(changed, binding).iloc[-1].to_numpy()))
    check('M2 close rule', features.close_epoch(1700000040, 'M2') == 1700000160)
    check('H4 close rule', features.close_epoch(1700000040, 'H4') == 1700014440)
    check('daily close rule', features.close_epoch(1700000040, 'Daily') == 1700086440)
    check('weekly close rule', features.close_epoch(1700000040, 'Weekly') == 1700604840)
    raw = int(pd.Timestamp('2024-02-01', tz='UTC').timestamp())
    check('calendar monthly leap closure', features.close_epoch(raw, 'Monthly') == int(pd.Timestamp('2024-03-01', tz='UTC').timestamp()))
    with tempfile.TemporaryDirectory(prefix='gold-supervisor-測試-') as name:
        root = Path(name); lock = root/'collector.lock'
        check('exclusive PID lock', supervisor.acquire_lock(lock))
        with patch.object(supervisor, 'process_exists', return_value=True):
            check('duplicate refused', not supervisor.acquire_lock(lock))
        with patch.object(supervisor, 'process_exists', return_value=False):
            check('verified absent stale PID reclaimed', supervisor.acquire_lock(lock))
        supervisor.atomic_json(root/'heartbeat.json', {'collector_status': 'RUNNING'})
        supervisor.atomic_json(root/'heartbeat.json', {'collector_status': 'PAUSED'})
        check('operational heartbeat replace', supervisor.load(root/'heartbeat.json')['collector_status'] == 'PAUSED')
        check('no temporary heartbeat residue', not list(root.glob('*.tmp')))
    check('bounded backoff', supervisor.BACKOFF == (5, 15, 30, 60, 120, 300))
    check('static runtime safety', supervisor.static_review()['status'] == 'PASS')
    for name in ['start', 'stop', 'status']:
        check(name+' script exists', (features.ROOT/(name+'_gold_future_capture_v4.ps1')).is_file())
    install = (features.ROOT/'install_gold_future_capture_task_v1.ps1').read_text()
    check('user-level task only', '-RunLevel Limited' in install and '-LogonType Interactive' in install)
    check('no credentials stored', '-Password' not in install and '-Credential' not in install)
    check('task duplicate prevention', '-MultipleInstances IgnoreNew' in install)
    check('logon persistence', '-AtLogOn' in install)
    check('hidden windows', '-WindowStyle Hidden' in install)
    start = (features.ROOT/'start_gold_future_capture_v4.ps1').read_text()
    check('start respects persistent policy pause', 'PAUSED*.json' in start and 'explicit recertification required' in start)
    check('no frozen collector mutation', supervisor.digest(features.ROOT/'gold_future_capture_collector_v4.py') == features.load(features.CAPTURE/'protocol'/'freeze.json')['code_sha256']['gold_future_capture_collector_v4.py'])
    print(json.dumps({'status': 'PASS', 'check_count': len(checks), 'checks': checks}))


if __name__ == '__main__':
    main()
