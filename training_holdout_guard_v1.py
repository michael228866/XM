"""Fail-closed audited file access for reviewed, hash-pinned training workflows."""
import os
import sys
from pathlib import Path
from datetime import datetime

HOLDOUT_START = int(datetime.fromisoformat('2026-09-25T19:24:00+00:00').timestamp())


def check_interval(start, end, cutoff):
    if (any(type(value) is not int for value in (start, end, cutoff))
            or not 0 <= start <= end <= cutoff < HOLDOUT_START):
        raise PermissionError('歷史資料不得超過已核准 cutoff 或進入 locked holdout')


def check_path(path, root):
    if isinstance(path, int):
        raise PermissionError('不允許未驗證的檔案描述元')
    target = Path(os.fsdecode(path)).resolve()
    locked = (Path(root)/'future_holdout/gold_s4_v4').resolve()
    if target == locked or target.is_relative_to(locked):
        raise PermissionError('偵測到 locked future holdout 存取')
    return target


def install(root, write_root=None):
    root = Path(root).resolve()
    write_root = Path(write_root).resolve() if write_root is not None else None
    active = True
    def writable(path):
        target = check_path(path, root)
        if write_root is not None and not target.is_relative_to(write_root):
            raise PermissionError('訓練只能寫入本次 run directory')
    def hook(event, args):
        if not active:
            return
        if event == 'open':
            check_path(args[0], root)
            mode = args[1] or ''
            flags = args[2] if len(args) > 2 else 0
            if any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                writable(args[0])
        elif event in {'os.listdir', 'os.scandir'}:
            check_path(args[0] if args and args[0] is not None else '.', root)
        elif event in {'os.mkdir', 'os.remove', 'os.rmdir'}:
            writable(args[0])
        elif event in {'os.rename', 'os.link', 'os.symlink'}:
            writable(args[0])
            writable(args[1])
        elif event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'ctypes.dlopen'}:
            raise PermissionError('訓練工作不可建立外部程序或載入未審核原生程式庫')
    sys.addaudithook(hook)
    def release():
        nonlocal active
        active = False
    return release


# CPython audit hooks are an enforcement layer for reviewed Python workflows,
# not an OS sandbox against hostile native code. Approval must reject native
# direct-file readers or escape paths; unapproved workflows stay disabled.
