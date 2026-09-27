"""Explicit, validated safe-idle migration. Never deletes the historical pause."""
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from gold_future_capture_collector_v2 import encode, sha, load

ROOT=Path(__file__).resolve().parent


def main():
    if len(sys.argv)!=3 or sys.argv[1]!='--activate':
        raise ValueError('--activate finalized_run required')
    run=Path(sys.argv[2]).resolve()
    if not run.is_relative_to(ROOT/'training_runs'):
        raise ValueError('Run outside registry')
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode().strip()
    if git('status','--porcelain') or git('ls-remote','origin','refs/heads/main').split()[0]!=git('rev-parse','HEAD'):
        raise ValueError('Committed pushed clean result required')
    seal=load(run/'FINALIZED.json'); validator=load(run/'validator.json'); spec=load(run/'execution_spec.json')
    if validator['overall']!='PASS' or validator.get('runtime_migration_approved') is not True:
        raise ValueError('Independent migration certification required')
    for name,expected in seal['file_sha256'].items():
        if sha((run/name).read_bytes())!=expected: raise ValueError('Archive changed')
    for name,expected in {**spec['preserved_sha256'],**spec['protected_sha256']}.items():
        if sha((ROOT/name).read_bytes())!=expected: raise ValueError('Preserved file changed')
    policy=ROOT/'gold_future_capture_runtime_freshness_policy_v1.json'
    if policy.read_bytes()!=(run/'runtime_idle_policy.json').read_bytes(): raise ValueError('Policy changed')
    original=ROOT/'future_holdout/gold_s4_v4/PAUSED_TIME_RULE.json'
    migration=load(run/'runtime_idle_migration.json')
    if sha(original.read_bytes())!=migration['original_pause_sha256']: raise ValueError('Original pause changed')
    stamp=datetime.now(timezone.utc)
    migration.update(activated_at_utc=stamp.isoformat(),runtime_policy_sha256=sha(policy.read_bytes()),
        validation_run=run.relative_to(ROOT).as_posix(),validation_finalized_sha256=sha((run/'FINALIZED.json').read_bytes()),
        validator_sha256=sha((run/'validator.json').read_bytes()),
        capture_not_before_utc=datetime.fromtimestamp((math.floor(stamp.timestamp()/60)+1)*60,timezone.utc).isoformat(),
        result_commit=git('rev-parse','HEAD'))
    with (ROOT/'gold_future_runtime_idle_migration_v1.json').open('xb') as out:
        out.write(encode(migration)); out.flush()
    print('MIGRATION_CREATED; commit and push before updating the single existing capture task')


if __name__=='__main__': main()
