"""Fail-closed audited file access for reviewed, hash-pinned training workflows."""
import os
import sys
from pathlib import Path


def check_path(path, root):
    if isinstance(path, int):
        raise PermissionError('不允許未驗證的檔案描述元')
    target = Path(os.fsdecode(path)).resolve()
    locked = (Path(root)/'future_holdout/gold_s4_v4').resolve()
    if target == locked or target.is_relative_to(locked):
        raise PermissionError('偵測到 locked future holdout 存取')
    return target


def install(root):
    root = Path(root).resolve()
    def hook(event, args):
        if event == 'open':
            check_path(args[0], root)
        elif event in {'os.listdir', 'os.scandir'}:
            check_path(args[0] if args and args[0] is not None else '.', root)
        elif event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'ctypes.dlopen'}:
            raise PermissionError('訓練工作不可建立外部程序或載入未審核原生程式庫')
    sys.addaudithook(hook)


# CPython audit hooks are an enforcement layer for reviewed Python workflows,
# not an OS sandbox against hostile native code. Approval must reject native
# direct-file readers or escape paths; unapproved workflows stay disabled.
