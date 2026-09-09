"""Resume the frozen JForex run without making any network/data requests."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import gold_gemini_dukascopy_jforex_final_reconciliation_v1 as study
import training_run_history as archive

RUN = study.ROOT / 'training_runs/20260908T140809Z_gemini_dukascopy_jforex_final_reconciliation_v1'


def sanitize_log() -> None:
    """Retain the original privately, publish only redacted SDK diagnostics."""
    private = study.ROOT / '.private_research_provenance' / RUN.name
    private.mkdir(parents=True, exist_ok=True)
    original = private / 'stdout.original.log'
    if not original.exists():
        shutil.copy2(RUN / 'stdout.log', original)
    raw = original.read_bytes()
    encoding = 'utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8'
    lines = raw.decode(encoding, errors='replace').splitlines()
    sensitive = re.compile(
        r'AuthorizationClient|ActivityLogger|DCClientImpl.*@|client-ip|'
        r'[?&](?:login|check|sermo|appello|publicus_pendo|testimonium_nuntius)='
    )
    safe = [
        '[REDACTED: SDK authentication/account diagnostic]'
        if sensitive.search(line) else line for line in lines
    ]
    (RUN / 'stdout.log').write_text('\n'.join(safe) + '\n', encoding='utf-8')
    archive.write_json(RUN / 'log_redaction_manifest.json', {
        'original_sha256': hashlib.sha256(raw).hexdigest(),
        'original_bytes': len(raw),
        'storage_path': str(original),
        'retention_status': 'permanent_private_repository_storage_not_git_backed_up',
        'remote_preservation_of_original': False,
        'public_log_sha256': archive.file_sha256(RUN / 'stdout.log'),
        'redacted_lines': sum(bool(sensitive.search(line)) for line in lines),
        'reason': 'SDK emitted authentication protocol and account identifiers despite WARN configuration.',
    })


def reconcile() -> None:
    manifest = archive.read_json(RUN / 'manifest.json')
    if archive.file_sha256(study.ROOT / 'gold_gemini_dukascopy_jforex_final_reconciliation_v1.py') != manifest['training_script_sha256']:
        raise RuntimeError('Executed reconciliation source changed since acquisition')
    audit, _, _ = study.read_jforex(RUN)
    if len(audit) != 2352:
        raise RuntimeError('Collection incomplete; refusing restart or reacquisition')
    blocks = {}
    with np.load(RUN / 'exact_timestamps.npz', allow_pickle=False) as source:
        for block, expected in study.EXPECTED_TIMESTAMP_HASHES.items():
            assert study.digest(source[f'{block}_broker_ns']) == expected
            blocks[block] = source[f'{block}_utc_ns'].copy()
    times = np.unique(np.concatenate(list(blocks.values())))
    _, old_unknown, _ = study.parent_state(study.load_native(), times)
    assert int(old_unknown.any(axis=1).sum()) == 6931
    snapshot = RUN / 'offline_completion_script.py'
    shutil.copy2(Path(__file__), snapshot)
    archive.write_json(RUN / 'offline_completion_provenance.json', {
        'commit': study.git('rev-parse', 'HEAD'),
        'origin_main': study.git('rev-parse', 'origin/main'),
        'script_sha256': archive.file_sha256(snapshot),
        'acquisition_commit': manifest['git_commit'],
        'new_acquisition': False,
        'reason': 'Collector completed 784 requests but JVM background threads did not exit; offline continuation calls frozen reconcile function.',
        'raw_evidence_before': {name: archive.file_sha256(RUN / name) for name in (
            'jforex_ticks.tsv.gz', 'jforex_bars.tsv.gz', 'jforex_request_audit.tsv')},
    })
    result = study.reconcile(RUN, manifest, blocks, times, old_unknown)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--sanitize-log']:
        sanitize_log()
    elif sys.argv[1:] == ['--reconcile']:
        reconcile()
    else:
        raise SystemExit('Use --sanitize-log or --reconcile')
