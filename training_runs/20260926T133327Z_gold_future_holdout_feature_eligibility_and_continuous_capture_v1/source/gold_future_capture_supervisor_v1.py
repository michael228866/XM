"""Thin, raw-only supervisor; frozen collector owns every admission decision."""
import argparse
import ast
import ctypes
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'future_holdout'/'gold_s4_v4'
HEALTH = CAPTURE/'health'
STOP = False
BACKOFF = (5, 15, 30, 60, 120, 300)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def static_review():
    spec = load(ROOT/'execution_spec_gold_future_holdout_feature_eligibility_and_continuous_capture_v1.json')
    forbidden = {'xgboost', 'lightgbm', 'catboost', 'sklearn', 'joblib', 'pickle', 'gemini'}
    pending = ['gold_future_capture_supervisor_v1.py', 'gold_future_capture_collector_v4.py']
    scanned = {}
    while pending:
        name = pending.pop()
        if name in scanned:
            continue
        path = ROOT/name
        tree = ast.parse(path.read_bytes())
        scanned[name] = digest(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            else:
                modules = []
            for module in modules:
                if module.split('.')[0] in forbidden or 'secondary_classifier' in module:
                    raise ValueError('PAUSED_PROTOCOL_MISMATCH: forbidden runtime import')
                local = ROOT/(module+'.py')
                if local.is_file():
                    pending.append(local.name)
            if isinstance(node, ast.Call):
                called = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                if called in {'predict', 'predict_proba', 'fit', 'load_model', 'order_send', 'order_check'}:
                    raise ValueError('PAUSED_PROTOCOL_MISMATCH: forbidden runtime call')
    if scanned != spec['runtime_source_sha256']:
        raise ValueError('PAUSED_PROTOCOL_MISMATCH: runtime source changed')
    for name, expected in spec['protected_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('PRODUCTION_CHANGED')
    return {'status': 'PASS', 'scanned_source_sha256': scanned, 'model_imports': False, 'strategy_imports': False}


def process_exists(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid PID; lock cannot be reclaimed')
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        return True  # Access denied/unknown is never proof of absence.
    try:
        code = wintypes.DWORD()
        return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)


def acquire_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with path.open('x', encoding='utf-8') as f:
                json.dump({'pid': os.getpid(), 'started_at_utc': now()}, f)
                f.flush(); os.fsync(f.fileno())
            return True
        except FileExistsError:
            prior = load(path)
            if process_exists(prior['pid']):
                return False
            # Compare exact bytes again; do not reclaim a lock changed by a peer.
            if load(path) != prior:
                return False
            path.unlink()
    return False


def process_mutex():
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, 'Local\\GOLD-Future-Holdout-Capture-v4')
    if not handle:
        raise OSError('Supervisor mutex unavailable')
    if kernel.WaitForSingleObject(handle, 0) not in (0, 128):
        kernel.CloseHandle(handle)
        return None, kernel
    return handle, kernel


def pause_markers():
    return sorted(CAPTURE.glob('PAUSED*.json')) + sorted(HEALTH.glob('PAUSED*.json'))


def coverage():
    from gold_future_capture_collector_v4 import recover_chain
    entries, _ = recover_chain(CAPTURE)
    stamps = []
    for entry in entries:
        stamps.extend(row['SOURCE_TIMESTAMP'] for row in load(CAPTURE/entry['snapshot_path'])['rows'])
    intervals, gaps = [], []
    for stamp in stamps:
        epoch = datetime.fromisoformat(stamp).timestamp()
        if intervals and epoch == intervals[-1]['end_exclusive_epoch']:
            intervals[-1]['end_exclusive_epoch'] = epoch+60
        else:
            if intervals:
                gaps.append({'start_epoch': intervals[-1]['end_exclusive_epoch'], 'end_exclusive_epoch': epoch,
                             'classification': 'UNCLASSIFIED_SOURCE_OR_CAPTURE_GAP'})
            intervals.append({'start_epoch': epoch, 'end_exclusive_epoch': epoch+60})
    paused = [{'start': load(p).get('created_at_utc'), 'end': None, 'marker': p.name} for p in pause_markers()]
    return {'coverage_start': stamps[0] if stamps else None, 'latest_certified_bar': stamps[-1] if stamps else None,
            'sealed_bar_count': len(stamps), 'capture_gap_intervals': gaps, 'paused_intervals': paused,
            'runtime_certified_intervals': intervals, 'last_sequence': entries[-1]['sequence'] if entries else None,
            'status': 'PARTIAL' if gaps or paused else 'PASS',
            'duration_rule': '365 certified calendar days; raw timestamp coverage only, not a duration-completion assertion'}


def event(kind):
    with (HEALTH/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'event': kind, 'at_utc': now()})+'\n')


def wait(seconds):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline and not STOP and not (HEALTH/'stop.request').exists():
        time.sleep(min(1, max(0, deadline-time.monotonic())))


def main():
    global STOP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    HEALTH.mkdir(parents=True, exist_ok=True)
    lock = HEALTH/'collector.lock'
    mutex, kernel = process_mutex()
    if mutex is None:
        return
    if not acquire_lock(lock):
        kernel.ReleaseMutex(mutex); kernel.CloseHandle(mutex)
        return
    heartbeat = {'supervisor_started_at_utc': now(), 'pid': os.getpid(),
        'last_cycle_started_at_utc': None, 'last_cycle_completed_at_utc': None,
        'last_successful_snapshot_sequence': None, 'last_successful_source_timestamp': None,
        'collector_status': 'STARTING', 'runtime_time_status': 'UNVERIFIED',
        'source_identity_status': 'UNVERIFIED', 'chain_status': 'UNVERIFIED',
        'consecutive_failures': 0, 'last_error_class': None, 'last_error_at_utc': None,
        'process_lock_status': 'PASS', 'process_running': True,
        'mutable_operational_metadata': True, 'holdout_evidence': False}
    def stopping(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        static_review()
        from gold_future_capture_collector_v4 import capture_once
        event('STARTED')
        while not STOP and not (HEALTH/'stop.request').exists():
            if pause_markers():
                heartbeat['collector_status'] = 'PAUSED'
                break
            heartbeat['last_cycle_started_at_utc'] = now()
            try:
                static_review()
                before = coverage()
                heartbeat.update(last_successful_snapshot_sequence=before['last_sequence'],
                                 last_successful_source_timestamp=before['latest_certified_bar'], chain_status='PASS')
                capture_once(CAPTURE)
                current = coverage()
                atomic_json(HEALTH/'gold_future_capture_coverage_v1.json', current)
                heartbeat.update(last_cycle_completed_at_utc=now(), collector_status='RUNNING',
                    runtime_time_status='PASS', source_identity_status='PASS', chain_status='PASS',
                    last_successful_snapshot_sequence=current['last_sequence'],
                    last_successful_source_timestamp=current['latest_certified_bar'], consecutive_failures=0,
                    last_error_class=None, last_error_at_utc=None)
                event('SNAPSHOT_SEALED' if current['last_sequence'] != before['last_sequence'] else 'RECOVERED_TRANSIENT_FAILURE')
                atomic_json(HEALTH/'heartbeat.json', heartbeat)
                wait(60 - time.time()%60 + 3)
            except Exception as error:
                heartbeat.update(last_cycle_completed_at_utc=now(), last_error_class=type(error).__name__,
                                 last_error_at_utc=now(), consecutive_failures=heartbeat['consecutive_failures']+1)
                if pause_markers() or isinstance(error, (ValueError, KeyError, TypeError)):
                    heartbeat.update(collector_status='PAUSED', runtime_time_status='FAIL')
                    if not pause_markers():
                        atomic_json(HEALTH/'PAUSED_PROTOCOL_MISMATCH.json', {'created_at_utc': now(), 'error_class': type(error).__name__})
                    event('PAUSED_TIME_RULE' if (CAPTURE/'PAUSED_TIME_RULE.json').exists() else 'PAUSED_CHAIN_INTEGRITY')
                    try:
                        atomic_json(HEALTH/'gold_future_capture_coverage_v1.json', coverage())
                    except Exception:
                        heartbeat['chain_status'] = 'FAIL'
                    break
                heartbeat['collector_status'] = 'RETRY_WAIT'
                atomic_json(HEALTH/'heartbeat.json', heartbeat)
                event('MT5_RETRY')
                wait(BACKOFF[min(heartbeat['consecutive_failures']-1, len(BACKOFF)-1)])
    except Exception as error:
        heartbeat.update(collector_status='PAUSED', last_error_class=type(error).__name__, last_error_at_utc=now())
        if not pause_markers():
            atomic_json(HEALTH/'PAUSED_PROTOCOL_MISMATCH.json', {'created_at_utc': now(), 'error_class': type(error).__name__})
    finally:
        if heartbeat['collector_status'] != 'PAUSED':
            heartbeat['collector_status'] = 'STOPPED'
        heartbeat['process_running'] = False
        atomic_json(HEALTH/'heartbeat.json', heartbeat)
        event('STOPPED')
        if lock.exists() and load(lock).get('pid') == os.getpid():
            lock.unlink()
        kernel.ReleaseMutex(mutex); kernel.CloseHandle(mutex)


if __name__ == '__main__':
    main()
