"""Automatic raw capture with continuously refreshed healthy-idle heartbeat."""
import os
import signal
import sys
import time

from gold_future_capture_supervisor_v1 import (
    CAPTURE, HEALTH, now, load, atomic_json, process_mutex, acquire_lock,
)
from gold_future_capture_collector_v4 import recover_chain
from gold_future_capture_collector_v4_1 import capture_once, migration_binding

STOP = False


def publish(heartbeat, health=HEALTH):
    heartbeat['heartbeat_updated_at_utc'] = now()
    atomic_json(health/'heartbeat.json', heartbeat)


def apply_cycle(heartbeat, decision, sequence, timestamp):
    heartbeat.update(collector_status=decision['collector_status'], runtime_time_status=decision['runtime_time_status'],
        chain_status='PASS', source_identity_status='PASS', last_successful_snapshot_sequence=sequence,
        last_successful_source_timestamp=timestamp, last_cycle_completed_at_utc=now(), process_running=True,
        evidence_admitted_last_cycle=decision['evidence_admitted'])
    if decision.get('last_observation') is not None:
        heartbeat['last_observation'] = decision['last_observation']


def main():
    global STOP
    if sys.argv[1:] != ['--run']:
        raise ValueError('--run required')
    HEALTH.mkdir(parents=True, exist_ok=True)
    mutex, kernel = process_mutex()
    if mutex is None:
        return
    lock = HEALTH/'collector.lock'
    if not acquire_lock(lock):
        kernel.ReleaseMutex(mutex); kernel.CloseHandle(mutex)
        return
    heartbeat = {'pid':os.getpid(),'supervisor_started_at_utc':now(),'collector_status':'STARTING',
        'runtime_time_status':'UNVERIFIED','chain_status':'UNVERIFIED','process_running':True,
        'process_lock_status':'PASS','mutable_operational_metadata':True,'holdout_evidence':False,
        'runtime_version':'gold_future_capture_supervisor_v1_1'}
    if (HEALTH/'heartbeat.json').exists():
        prior = load(HEALTH/'heartbeat.json')
        if prior.get('last_observation') is not None:
            heartbeat['last_observation'] = prior['last_observation']
    def stopping(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stopping); signal.signal(signal.SIGTERM, stopping)
    try:
        migration_binding()
        while not STOP and not (HEALTH/'stop.request').exists():
            heartbeat['last_cycle_started_at_utc'] = now()
            publish(heartbeat)
            decision = capture_once(heartbeat.get('last_observation'))
            entries, previous = recover_chain(CAPTURE)
            apply_cycle(heartbeat, decision, entries[-1]['sequence'], previous['rows'][-1]['SOURCE_TIMESTAMP'])
            publish(heartbeat)
            # One-minute acquisition; heartbeat remains current during idle.
            for _ in range(6):
                if STOP or (HEALTH/'stop.request').exists():
                    break
                time.sleep(10)
                migration_binding()
                publish(heartbeat)
    except Exception as error:
        heartbeat.update(collector_status='PAUSED',runtime_time_status='FAIL',last_error_class=type(error).__name__,
                         last_error_at_utc=now())
        if 'CHAIN' in str(error): heartbeat['chain_status']='FAIL'
        if not list(HEALTH.glob('PAUSED*.json')):
            atomic_json(HEALTH/'PAUSED_PROTOCOL_MISMATCH.json',{'created_at_utc':now(),
                'error_class':type(error).__name__,'automatic_resume':False})
    finally:
        heartbeat['process_running'] = False
        if heartbeat['collector_status'] != 'PAUSED': heartbeat['collector_status']='STOPPED'
        publish(heartbeat)
        if lock.exists() and load(lock).get('pid') == os.getpid(): lock.unlink()
        kernel.ReleaseMutex(mutex); kernel.CloseHandle(mutex)


if __name__ == '__main__':
    main()
