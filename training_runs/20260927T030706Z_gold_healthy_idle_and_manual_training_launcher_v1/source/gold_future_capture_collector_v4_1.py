"""Additive freshness-first acquisition; frozen v4 validates every admitted row."""
import subprocess
import time
import uuid
from datetime import datetime, timezone

from gold_future_capture_collector_v4 import (
    ROOT, FIELDS, binding, recover_chain, validate_payload,
    encode, sha, load, now, require, sealed_write, writer_lock,
)
from gold_future_current_regime_diagnostic_v1 import check_samples
from gold_future_fresh_market_diagnostic_v1 import assess, observe
from gold_future_source_identity_v4 import session, verify_identity

CAPTURE = ROOT/'future_holdout/gold_s4_v4'
POLICY = ROOT/'gold_future_capture_runtime_freshness_policy_v1.json'
MIGRATION = ROOT/'gold_future_runtime_idle_migration_v1.json'


def migration_binding():
    policy, migration = load(POLICY), load(MIGRATION)
    require(migration['resume_mode'] == 'SAFE_IDLE_MIGRATION_WITH_NO_EVIDENCE_ADMISSION'
            and migration['historical_original_pause_root_cause_proven'] is False,
            'PAUSED_PROTOCOL_MISMATCH')
    require(sha(POLICY.read_bytes()) == migration['runtime_policy_sha256'], 'PAUSED_PROTOCOL_MISMATCH')
    committed = subprocess.check_output(['git','cat-file','blob','HEAD:'+MIGRATION.name], cwd=ROOT)
    require(committed == MIGRATION.read_bytes(), 'PAUSED_PROTOCOL_MISMATCH')
    require(sha((CAPTURE/'PAUSED_TIME_RULE.json').read_bytes()) == migration['original_pause_sha256'],
            'PAUSED_PROTOCOL_MISMATCH')
    for name, expected in policy['runtime_code_sha256'].items():
        require(sha((ROOT/name).read_bytes()) == expected, 'PAUSED_PROTOCOL_MISMATCH')
    run = ROOT/migration['validation_run']
    require(run.resolve().is_relative_to((ROOT/'training_runs').resolve()), 'PAUSED_PROTOCOL_MISMATCH')
    require(sha((run/'FINALIZED.json').read_bytes()) == migration['validation_finalized_sha256']
            and sha((run/'validator.json').read_bytes()) == migration['validator_sha256'], 'PAUSED_PROTOCOL_MISMATCH')
    validator = load(run/'validator.json')
    require(validator['overall'] == 'PASS' and validator['runtime_migration_approved'] is True, 'PAUSED_PROTOCOL_MISMATCH')
    for p in [*CAPTURE.glob('PAUSED*.json'), *(CAPTURE/'health').glob('PAUSED*.json')]:
        require(p == CAPTURE/'PAUSED_TIME_RULE.json', 'ACTIVE_POLICY_PAUSE')
    return migration


def operational_decision(samples, previous_tick, last_bar):
    result = assess(samples, previous_tick, last_bar)
    if result['source_identity_status'] != 'PASS':
        raise ValueError('PAUSED_SOURCE_IDENTITY')
    if result['classification'] in {'TIME_RULE_CONFLICT','UNCERTIFIED_CANDIDATE_REGIME'}:
        raise ValueError('PAUSED_TIME_RULE')
    if (result['classification'] == 'NO_FRESH_MARKET_DATA'
            or len({s['raw_tick_epoch'] for s in samples}) < 3
            or result['new_closed_m1_count'] == 0):
        return {'collector_status':'HEALTHY_IDLE', 'runtime_time_status':'NO_FRESH_MARKET_DATA',
                'evidence_admitted':False, 'diagnostic':result}
    return {'collector_status':'RUNNING', 'runtime_time_status':'PASS',
            'evidence_admitted':False, 'diagnostic':result}


def quarantine(reason, samples):
    receipt = {'created_at_utc':now(), 'status':reason, 'samples':samples,
               'historical_pause_preserved':True, 'automatic_resume':False}
    sealed_write(CAPTURE, CAPTURE/'quarantine'/(uuid.uuid4().hex+'.json'), encode(receipt))
    marker = CAPTURE/'health'/(reason+'.json')
    if not marker.exists():
        sealed_write(CAPTURE, marker, encode(receipt))


def capture_once(previous_observation=None):
    samples = []
    try:
        migration = migration_binding()
        try:
            freeze, activation = binding(CAPTURE)
        except (ValueError, KeyError, TypeError) as error:
            message = str(error).lower()
            reason = 'PAUSED_CONTEXT_INTEGRITY' if 'context' in message else 'PAUSED_SCHEMA_MISMATCH' if 'schema' in message else 'PAUSED_PROTOCOL_MISMATCH'
            raise ValueError(reason) from None
        try:
            entries, previous = recover_chain(CAPTURE)
        except (ValueError, KeyError, TypeError):
            raise ValueError('PAUSED_CHAIN_INTEGRITY') from None
        previous_tick = previous['samples'][-1]['raw_epoch']
        last_bar = previous['rows'][-1]['RAW_SOURCE_EPOCH']
        with session() as (mt5, identity):
            for index in range(3):
                if index:
                    time.sleep(2)
                try:
                    samples.append(observe(mt5))
                except ValueError as error:
                    if str(error) in {'NO_CURRENT_TICK','NO_CURRENT_M1'}:
                        return {'collector_status':'HEALTHY_IDLE','runtime_time_status':'NO_FRESH_MARKET_DATA',
                                'evidence_admitted':False,'last_observation':previous_observation}
                    raise
            if previous_observation is not None:
                require(samples[0]['raw_tick_epoch'] >= previous_observation['raw_tick_epoch']
                    and samples[0]['system_utc_epoch'] > previous_observation['system_utc_epoch']
                    and samples[0]['m1_raw_epochs'][-1] >= previous_observation['m1_raw_epochs'][-1], 'PAUSED_TIME_RULE')
            decision = operational_decision(samples, previous_tick, last_bar)
            decision['last_observation'] = samples[-1]
            if decision['collector_status'] == 'HEALTHY_IDLE':
                return decision
            frozen_samples = [{'source_identity':s['source_identity'], 'raw_epoch':s['raw_tick_epoch'],
                'observed_utc_epoch':s['system_utc_epoch'], 'm1_raw_epochs':s['m1_raw_epochs']}
                for s in samples]
            check_samples(frozen_samples, previous['samples'][-1])
            rates = mt5.copy_rates_from_pos('GOLD#', mt5.TIMEFRAME_M1, 0, 3)
            require(rates is not None and len(rates) == 3, 'PAUSED_SCHEMA_MISMATCH')
            require(verify_identity(mt5) == identity, 'PAUSED_SOURCE_IDENTITY')
            # No retrospective paused-period catch-up: only newly observed native
            # closed bars at or after the explicit migration boundary are eligible.
            floor = datetime.fromisoformat(migration['capture_not_before_utc']).timestamp()
            rows = []
            for row in rates[:-1]:
                raw = int(row['time'])
                if raw <= last_bar or raw-10800 < floor or raw+60 > samples[-1]['raw_tick_epoch']:
                    continue
                stamp = datetime.fromtimestamp(raw-10800, timezone.utc).isoformat()
                rows.append(dict(zip(FIELDS,[raw,stamp,'XMGlobal-MT5-6_GOLD','GOLD#','M1',
                    *[float(row[k]) for k in ('open','high','low','close')],
                    *[int(row[k]) for k in ('spread','tick_volume','real_volume')]])))
            payload = {'samples':frozen_samples,'rows':rows,'returned_raw_epochs':[int(r['time']) for r in rates],
                'native_source_confirmed':True,'resampled':False,'timeframe':'M1','mt5_timeframe_constant':1,
                'strategy_outcome_inspected':False}
            del rates
        if not rows:
            return dict(decision, collector_status='HEALTHY_IDLE', runtime_time_status='NO_FRESH_MARKET_DATA')
        with writer_lock(CAPTURE):
            migration_binding()
            entries, previous = recover_chain(CAPTURE)
            validation = validate_payload(payload, freeze, activation, previous)
            number = len(entries)
            raw = encode(payload)
            item = {'manifest_version':'gold_future_capture_manifest_v4','sequence':number,
                'previous_manifest_sha256':sha(encode(entries[-1])),'snapshot_path':f'snapshots/{number:012d}.json',
                'snapshot_sha256':sha(raw),'activation_sha256':sha(encode(activation)),
                'runtime_validation':validation,'source_id':'XMGlobal-MT5-6_GOLD','symbol':'GOLD#','sealed_at_utc':now()}
            sealed_write(CAPTURE,CAPTURE/item['snapshot_path'],raw)
            sealed_write(CAPTURE,CAPTURE/'manifests'/f'{number:012d}.json',encode(item))
            recover_chain(CAPTURE)
        return dict(decision, evidence_admitted=True)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        text = str(error)
        known = {'PAUSED_SOURCE_IDENTITY','PAUSED_CHAIN_INTEGRITY','PAUSED_SCHEMA_MISMATCH','PAUSED_PROTOCOL_MISMATCH','PAUSED_CONTEXT_INTEGRITY'}
        reason = text if text in known else 'PAUSED_SOURCE_IDENTITY' if text.startswith('SOURCE_') else 'PAUSED_SCHEMA_MISMATCH' if isinstance(error,(KeyError,TypeError)) else 'PAUSED_TIME_RULE'
        quarantine(reason, samples)
        raise ValueError(reason) from None
