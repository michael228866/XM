"""Causal-context and external-attestation review; no strategy execution."""
import contextlib
import io
import json
from copy import deepcopy
from pathlib import Path

import gold_future_capture_certification_v2 as cert
import gold_future_capture_collector_v2 as storage
from gold_future_capture_final_readiness_v1 import boundary

ROOT = Path(__file__).resolve().parent
SLUG = 'gold_future_capture_external_attestation_and_prefix_freeze_v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evidence_review():
    items = load(ROOT / 'research_evidence' / SLUG / 'evidence_index.json')
    if not isinstance(items, list) or not items:
        raise ValueError('Evidence list required')
    cert.check_evidence(items)
    for item in items:
        cert.require(all(item.get(k) for k in ('publisher', 'title', 'unsupported_claims')), 'Review scope missing')
    return items


def context_rules(doc, rows, evidence_boundary, require_approval=True):
    """Validate already-certified context inventory metadata, never compute signals."""
    cert.require(doc['policy'] == 'PREDECLARED_CAUSAL_REINITIALIZATION', 'Explicit reinitialization')
    cert.require(doc['source_id'] == cert.SOURCE_ID and doc['symbol'] == cert.SYMBOL, 'Context identity')
    cert.require(doc['recursive_features'] == ['MACD_HIST'], 'Recursive feature list')
    cert.require(doc['historically_equivalent'] is False and doc['outcome_tuning'] is False, 'Reinit honesty')
    cert.require(doc['context_role'] == 'CAUSAL_CONTEXT_ONLY' and doc['context_only'] is True
                 and doc['holdout_evidence'] is False, 'Context is never evidence')
    cert.require(doc['pipeline_sha256'] == storage.sha((ROOT/'drl_trading_v2.py').read_bytes()), 'Pipeline hash')
    import gold_recursive_prefix_final_readiness_v1 as feature_source
    cert.require(doc['feature_implementation_sha256'] == feature_source.feature_definition(), 'Feature hash')
    frames = set(cert.load('gold_future_capture_manifest_schema_v2.json')['properties']['timeframe']['enum'])
    cert.require(set(rows) == frames == set(doc['warmup_bars_by_timeframe']), 'Native 21 TF completeness')
    start, end, bound = map(storage.utc, (doc['context_start'], doc['context_end_exclusive'], evidence_boundary))
    cert.require(start < end <= bound, 'Context envelope must precede boundary')
    for frame, entries in rows.items():
        cert.require(len(entries) >= doc['warmup_bars_by_timeframe'][frame], 'Context warmup missing')
        previous = None
        for entry in entries:
            stamp, closed = storage.utc(entry['timestamp']), storage.utc(entry['closed_at'])
            cert.require(start <= stamp < closed < bound and stamp < end and closed <= end, 'Future context leakage')
            cert.require(previous is None or previous < stamp, 'Context order/duplicate')
            cert.require(entry['source_id'] == cert.SOURCE_ID and entry['symbol'] == cert.SYMBOL, 'Row identity')
            previous = stamp
    if require_approval:
        cert.require(doc['approved'] is True and doc['source_compatibility_certified'] is True, 'Context not certified/approved')
        cert.require(doc['sealed_context_sha256'] == cert.canonical(rows), 'Sealed context hash mismatch')
    return True


def authority_scope(evidence_scope):
    cert.require(evidence_scope['exact_server'] == 'XMGlobal-MT5 6' and evidence_scope['symbol'] == 'GOLD#', 'Exact source authority')
    cert.require(evidence_scope['authority'] in {'BROKER_DOCUMENT', 'SIGNED_ATTESTATION'}, 'Diagnostic-only authority rejected')
    cert.require(evidence_scope['holiday_authority'] and evidence_scope['epoch_encoding_confirmed'] is True,
                 'Holiday authority and epoch confirmation required')


def authority_gate(source, tz, evidence_scope):
    authority_scope(evidence_scope)
    cert.complete_source(source)
    cert.timezone_check(tz, source)


def sealed_boundary(artifacts, bar, freeze, activation, coverage, context_complete):
    cert.require(artifacts['certification_status'] == 'READY_FOR_CAPTURE_ACTIVATION', 'Activation certification required')
    cert.require(artifacts['sealed'] is True, 'Sealed snapshot required')
    cert.require(storage.sha(artifacts['snapshot_bytes']) == artifacts['snapshot_sha256'], 'Snapshot hash mismatch')
    cert.require(cert.canonical(artifacts['manifest']) == artifacts['manifest_sha256'], 'Manifest hash mismatch')
    cert.require(artifacts['manifest']['first_source_timestamp'] == bar, 'Manifest boundary mismatch')
    cert.require(artifacts['full_chain_valid'] is True, 'Full chain required')
    return boundary(bar, freeze, activation, coverage, context_complete)


def decisions(report, diagnostic):
    doc = load(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json')
    # Methodology authorization is distinct from evidence of source compatibility.
    unresolved = not doc['approved'] or not doc['source_compatibility_certified'] or not doc['sealed_context_sha256']
    result = {'policy': 'UNRESOLVED' if unresolved else doc['policy'],
        'status': 'PARTIAL' if unresolved else 'PASS', 'approved': not unresolved,
        'proposed_policy': doc['policy'], 'context_role': doc['context_role'],
        'historically_equivalent': False, 'outcome_tuning': False,
        'historical_context_allowed_in_principle': True,
        'requires_post_activation_monthly_accumulation': False,
        'context_protocol_sha256': cert.canonical(doc), 'blockers': doc['approval_blockers']}
    formal = 'PARTIAL' if report['readiness']['status'] != 'READY_FOR_CAPTURE_ACTIVATION' or unresolved else 'PASS'
    if report['formal_run_status'] == 'FAIL': formal = 'FAIL'
    return result, formal


def self_test():
    import gold_future_capture_final_readiness_v1 as prior
    import test_gold_future_capture_protocol_v2 as storage_tests
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        prior.self_test()
        storage_tests.self_test()
    inherited = [json.loads(line) for line in output.getvalue().splitlines()]
    checks = ['inherited:'+name for item in inherited for name in item['checks']]
    cert.require(all(item['overall']=='PASS' for item in inherited), 'Inherited tests')
    def rejected(name, call):
        try: call()
        except (ValueError, KeyError, TypeError): checks.append(name); return
        raise AssertionError('Not rejected: '+name)
    evidence_review(); checks.append('external_evidence_schema_and_hashes')
    import gold_mt5_timezone_transition_diagnostic_v2 as diagnostic
    assert len(diagnostic.WINDOWS) == 10
    assert {'October_before', 'October_after', 'US_autumn_before', 'US_autumn_after'} <= {name for name, _ in diagnostic.WINDOWS}
    checks.append('bounded_october_sampling_windows')
    sample = diagnostic.summarize([1761307200, 1761307260])
    assert sample['spacing_seconds'] == [60] and sample['monotonic']
    assert sample['classification'] == 'INSUFFICIENT_DATA' and sample['offset_inference'] is None
    assert all(row['difference_from_independent_anchor_if_available'] is None for row in sample['timestamp_rows'])
    assert not diagnostic.summarize([1761307260, 1761307200])['monotonic']
    checks.append('timestamp_projection_without_false_certification')
    result, verdict = decisions({'readiness': {'status': 'NOT_READY_MULTIPLE_BLOCKERS'}, 'formal_run_status': 'PARTIAL'}, {})
    assert verdict == 'PARTIAL' and result['policy'] == 'UNRESOLVED'
    checks.append('unresolved_readiness_remains_partial')
    doc = load(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json')
    from datetime import datetime, timedelta, timezone
    start = datetime(2024,1,1,tzinfo=timezone.utc)
    rows = {frame:[{'timestamp':(start+timedelta(minutes=i)).isoformat(),
        'closed_at':(start+timedelta(minutes=i,seconds=59)).isoformat(),
        'source_id':cert.SOURCE_ID,'symbol':cert.SYMBOL} for i in range(count)]
        for frame,count in doc['warmup_bars_by_timeframe'].items()}
    # Synthetic metadata tests only; no claim that Monthly bars close in a minute.
    bound = '2026-09-26T00:00:00+00:00'
    fixture = {**doc,'approved':True,'source_compatibility_certified':True,'sealed_context_sha256':cert.canonical(rows)}
    assert context_rules(fixture,rows,bound)
    checks.append('causal_preboundary_context_permitted')
    for key,value,name in [('recursive_features',['ATR'],'recursive_features_exact'),
        ('historically_equivalent',True,'not_historically_equivalent'),('outcome_tuning',True,'no_outcome_tuning'),
        ('holdout_evidence',True,'context_not_evidence'),('pipeline_sha256','0'*64,'prefix_pipeline_hash'),
        ('feature_implementation_sha256','0'*64,'feature_implementation_hash'),
        ('sealed_context_sha256','0'*64,'context_hash'),('approved',False,'unapproved_context'),
        ('source_compatibility_certified',False,'uncertified_context')]:
        rejected(name,lambda key=key,value=value:context_rules({**fixture,key:value},rows,bound))
    bad=deepcopy(rows);bad['M1'][-1]['timestamp']=bound
    rejected('context_after_boundary',lambda:context_rules(fixture,bad,bound))
    bad=deepcopy(rows);bad['Monthly'][-1]['closed_at']=bound
    rejected('future_native_closure',lambda:context_rules(fixture,bad,bound))
    rejected('native21_missing',lambda:context_rules(fixture,{k:v for k,v in rows.items() if k!='Monthly'},bound))
    source=cert.load('gold_future_capture_source_attestation_v2.json')
    rejected('diagnostic_cannot_complete_timezone',lambda:authority_gate(source,{}, {'authority':'INDEPENDENT_REVIEW'}))
    scope={'exact_server':'XMGlobal-MT5 6','symbol':'GOLD#','authority':'BROKER_DOCUMENT',
           'holiday_authority':'SYNTHETIC','epoch_encoding_confirmed':True}
    for key,value,name in [('authority','','missing_authority'),('authority','INDEPENDENT_REVIEW','diagnostic_only_authority'),
        ('holiday_authority',None,'holiday_authority_missing')]:
        rejected(name,lambda key=key,value=value:authority_scope({**scope,key:value}))
    artifacts={'certification_status':'READY_FOR_CAPTURE_ACTIVATION','sealed':True,'snapshot_bytes':b'synthetic',
        'snapshot_sha256':storage.sha(b'synthetic'),'manifest':{'first_source_timestamp':bound},'full_chain_valid':True}
    artifacts['manifest_sha256']=cert.canonical(artifacts['manifest'])
    freeze='2026-09-25T00:00:00+00:00';active='2026-09-25T01:00:00+00:00';coverage=(freeze,'2027-10-01T00:00:00+00:00')
    assert sealed_boundary(artifacts,bound,freeze,active,coverage,True)['holdout_evaluation_start']==bound
    checks.append('sealed_boundary_valid')
    for key,value,name in [('sealed',False,'sealed_required'),('snapshot_sha256','f'*64,'snapshot_hash_required'),
        ('manifest_sha256','f'*64,'manifest_hash_required'),('certification_status','NOT_READY','activation_certification'),
        ('full_chain_valid',False,'full_chain_required')]:
        rejected(name,lambda key=key,value=value:sealed_boundary({**artifacts,key:value},bound,freeze,active,coverage,True))
    aborted=ROOT/'training_runs/20260923T143230Z_gold_future_capture_final_readiness_v1'
    assert (aborted/'failure.json').exists() and load(aborted/'manifest.json')['status']=='aborted'
    assert all(storage.sha((aborted/name).read_bytes())==h for name,h in load(aborted/'FINALIZED.json')['file_sha256'].items())
    checks.append('aborted_attempt_preserved')
    assert 'NOT_SENT' in (ROOT/'XM_TIMEZONE_ATTESTATION_REQUEST_V1.md').read_text(encoding='utf-8')
    checks.append('support_request_not_sent')
    print(json.dumps({'overall':'PASS','checks':checks,'synthetic_only':True}))


if __name__ == '__main__': self_test()
