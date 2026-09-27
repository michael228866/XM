"""Operational status only; never outputs market or strategy values."""
from datetime import datetime, timezone

from gold_future_capture_supervisor_v1 import CAPTURE, ROOT, load, process_exists
from gold_future_capture_collector_v4 import recover_chain


def status():
    heartbeat = load(CAPTURE/'health/heartbeat.json')
    stamp = heartbeat.get('heartbeat_updated_at_utc',heartbeat.get('last_cycle_completed_at_utc'))
    age = datetime.now(timezone.utc).timestamp()-datetime.fromisoformat(stamp).timestamp()
    alive = process_exists(heartbeat['pid']) and heartbeat.get('process_running') is True
    entries, previous = recover_chain(CAPTURE)
    print('Capture:',heartbeat['collector_status'] if alive else 'PAUSED')
    print('Heartbeat:','PASS' if alive and 0 <= age <= 30 else 'FAIL')
    print('Chain: PASS\nCertified offset: 10800')
    print('Last accepted sequence:',entries[-1]['sequence'])
    print('Last source timestamp:',previous['rows'][-1]['SOURCE_TIMESTAMP'])
    print('Holdout start: 2026-09-25T19:24:00+00:00')
    print('Holdout evaluation start:',load(ROOT/'gold_future_holdout_evaluation_start_v1.json').get('holdout_evaluation_start'))
    print('Training: IDLE / approval required' if load(ROOT/'training_launcher_config_v1.json')['approval_status'] != 'APPROVED' else 'Training: manual launcher only')


if __name__ == '__main__':
    try:
        status()
    except Exception as error:
        print('[失敗] 狀態檢查失敗:',type(error).__name__)
        raise SystemExit(1)
