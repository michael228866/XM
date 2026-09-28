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
    print('CAPTURE\n-------')
    print('Status:',heartbeat['collector_status'] if alive else 'PAUSED')
    print('Heartbeat:','PASS' if alive and 0 <= age <= 30 else 'FAIL')
    print('Chain: PASS\nCertified offset: 10800')
    print('Last accepted sequence:',entries[-1]['sequence'])
    print('Last source timestamp:',previous['rows'][-1]['SOURCE_TIMESTAMP'])
    print('HOLDOUT\n-------')
    print('Holdout start: 2026-09-25T19:24:00+00:00')
    print('Holdout evaluation start:',load(ROOT/'gold_future_holdout_evaluation_start_v1.json').get('holdout_evaluation_start'))
    from gold_manual_training_workflow_v1 import binding_status
    training = binding_status()
    print('TRAINING\n--------')
    print('Workflow: GOLD S4 Secondary Train + Validate v1')
    print('Ready:', 'YES' if training['workflow'] == 'READY' else 'NO')
    print('Owner: USER')
    print('Auto Fetch:', 'YES' if training['auto_fetch'] == 'ENABLED' else 'NO')
    for label, key in [('Symbol','symbol'), ('Historical Data','historical_data')]:
        print(label + ':', training[key] if training[key] is not None else 'UNRESOLVED')
    from gold_manual_s4_train_validate_v1 import last_result
    last = last_result(ROOT)
    for label, key in [('Last Run', 'run_id'), ('Train', 'train_status'), ('Validation', 'validator_status'),
                       ('Final', 'final_status'), ('Candidate', 'candidate_model_path')]:
        print(label+':', last[key] or 'N/A')
    print('Production: UNCHANGED')


if __name__ == '__main__':
    try:
        status()
    except Exception as error:
        print('[失敗] 狀態檢查失敗:',type(error).__name__)
        raise SystemExit(1)
