"""Additional final-readiness guards and synthetic checks; v2 rules unchanged."""
import ast
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import gold_future_capture_certification_v2 as cert
import gold_future_capture_collector_v2 as storage
import gold_recursive_prefix_final_readiness_v1 as prefix
from gold_mt5_timezone_transition_diagnostic_v1 import current_classification

ROOT = Path(__file__).resolve().parent


def identity(source):
    cert.source_identity(source)
    cert.require(source['source_account_environment'] == 'demo', 'Frozen demo environment')
    cert.require(source['source_timestamp_semantics'] == 'BAR_OPEN', 'Bar-open required')


def prefix_evidence(doc, expected_feature_hash, expected_range):
    cert.require(doc['feature_hash'] == expected_feature_hash, 'Feature hash mismatch')
    cert.require(doc['source_range'] == expected_range and expected_range is not None, 'Source range mismatch')
    cert.require(doc['approved'] is True, 'Explicit freeze approval required')
    if doc['policy'] == 'EXACT_CONTINUOUS_CERTIFIED_PREFIX':
        cert.require(doc['recursive_state_equal'] is True and doc['certified_history'] is True, 'Exact recursive state unproven')
    else:
        cert.require(doc['policy'] == 'PREDECLARED_CAUSAL_REINITIALIZATION'
            and doc['historically_equivalent'] is False and doc['outcome_tuning'] is False, 'Reinit honesty')


def boundary(bar, freeze, activation, coverage, warmup_complete):
    cert.require(freeze is not None and activation is not None, 'Freeze and activation required')
    stamp = storage.utc(bar)
    cert.require(stamp > max(storage.utc(freeze), storage.utc(activation)), 'No backdating or equal boundary')
    cert.require(storage.utc(coverage[0]) <= stamp < storage.utc(coverage[1]), 'Uncovered boundary')
    return {'capture_start': bar, 'holdout_evaluation_start': bar if warmup_complete else None}


def readiness(report, diagnostic):
    cert.require(diagnostic['classification'] != 'CONTRADICTS_BROKER_SERVER_RULE', 'Diagnostic contradiction')
    return report['readiness']['status'] == 'READY_FOR_CAPTURE_ACTIVATION'


def self_test():
    checks = []
    from run_gold_future_capture_final_readiness_v1 import evidence_index
    assert len(evidence_index()) == 6
    checks.append('evidence_index_list_loading')
    def passed(name, call):
        call()
        checks.append(name)
    def rejected(name, call):
        try:
            call()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
            return
        raise AssertionError('Not rejected: ' + name)
    def require(value):
        cert.require(value, 'Assertion')
    source = cert.load('gold_future_capture_source_attestation_v2.json')
    identity(source)
    for key, value, name in [('broker_server_name','other','wrong_server'), ('symbol','GOLD','wrong_symbol'),
        ('source_account_environment','live','wrong_environment'), ('source_timestamp_semantics','BAR_CLOSE','bar_open')]:
        rejected(name, lambda key=key,value=value: identity({**source,key:value}))
    tz = cert.load('gold_future_capture_timezone_attestation_template_v2.json')
    a,b,c = '2030-01-01T00:00:00+00:00','2030-06-01T00:00:00+00:00','2031-01-01T00:00:00+00:00'
    evidence = cert.load('research_evidence/gold_future_capture_final_readiness_v1/evidence_index.json')
    intervals = [{'start_utc':a,'end_utc':b,'offset_minutes':120,'evidence':evidence},
                 {'start_utc':b,'end_utc':c,'offset_minutes':180,'evidence':evidence}]
    tz.update(broker_offset_intervals=intervals, certified_coverage_start_utc=a, certified_coverage_end_utc=c,
              timezone_status='CERTIFIED_BROKER_SERVER_RULE',mt5_epoch_semantics='LOCAL_EPOCH_REQUIRES_RULE',
              timestamp_encoding='EPOCH_SECONDS')
    passed('interval_ordering',lambda: storage.validate_intervals(tz,a,c))
    for name,start in [('overlap',a),('gap','2030-06-02T00:00:00+00:00')]:
        bad=deepcopy(tz);bad['broker_offset_intervals'][1]['start_utc']=start
        rejected(name,lambda bad=bad:storage.validate_intervals(bad,a,c))
    for name,stamp,offset in [('start_inclusion',a,7200),('boundary_new_interval',b,10800)]:
        epoch=int(storage.utc(stamp).timestamp())+offset
        passed(name,lambda epoch=epoch,stamp=stamp:require(storage.normalize_epoch(epoch,tz)==stamp))
    rejected('end_exclusion',lambda:storage.normalize_epoch(int(storage.utc(c).timestamp())+10800,tz))
    rejected('uncovered_timestamp',lambda:storage.normalize_epoch(0,tz))
    rejected('missing_evidence',lambda:cert.check_evidence([]))
    complete=deepcopy(source)
    complete.update(template=False,source_timezone='SYNTHETIC',timezone_authority='TEST',dst_policy='TABLE',
        session_rollover='TEST',weekend_policy='TEST',holiday_policy_source='TEST',attested_by='TEST',attested_at_utc=a,evidence=evidence)
    tz.update(template=False,source_id=source['source_id'],source_attestation_sha256=cert.canonical(complete),
        timezone='SYNTHETIC',timezone_authority='TEST',dst_policy='TABLE',server_display_clock_rule='TABLE',
        session_rollover='TEST',weekend_policy='TEST',holiday_calendar_authority='TEST',attested_by='TEST',attested_at_utc=a,
        bar_timestamp_semantics='BAR_OPEN',timestamp_encoding='EPOCH_SECONDS',evidence=evidence,broker_transition_rule='EXPLICIT TEST TABLE')
    # Structural fixtures do not assert evidence authority for a real source.
    passed('timezone_structural',lambda:cert.timezone_check(tz,complete,evidence_check=lambda _:None))
    rejected('hash_mismatch',lambda:cert.timezone_check({**tz,'source_attestation_sha256':'0'*64},complete))
    rejected('unsupported_recurrence',lambda:cert.timezone_check({**tz,'recurrence_rule':'yearly'},complete,evidence_check=lambda _:None))
    rejected('diagnostic_contradiction',lambda:readiness({}, {'classification':'CONTRADICTS_BROKER_SERVER_RULE'}))
    now=storage.utc('2030-07-01T12:00:00+00:00').timestamp()
    passed('diagnostic_support',lambda:require(current_classification(int(now+10800),now)=='SUPPORTS_BROKER_SERVER_RULE'))
    passed('diagnostic_direct_utc',lambda:require(current_classification(int(now),now)=='CONTRADICTS_BROKER_SERVER_RULE'))
    protocol=cert.load('gold_future_capture_protocol_v2.json')
    passed('native_20tf',lambda:require(protocol['higher_timeframe_policy']=='NATIVE_20TF_REQUIRED'))
    rejected('silent_resample',lambda:require('RESAMPLE'==protocol['higher_timeframe_policy']))
    doc={'feature_hash':'x','source_range':{'start':a,'end':b},'approved':True,
         'policy':'EXACT_CONTINUOUS_CERTIFIED_PREFIX','recursive_state_equal':True,'certified_history':True}
    passed('prefix_exact_structural',lambda:prefix_evidence(doc,'x',doc['source_range']))
    reinit={**doc,'policy':'PREDECLARED_CAUSAL_REINITIALIZATION','historically_equivalent':False,'outcome_tuning':False}
    passed('prefix_reinit_structural',lambda:prefix_evidence(reinit,'x',doc['source_range']))
    for name,bad,expected,source_range in [('unapproved_reinit',{**reinit,'approved':False},'x',doc['source_range']),
        ('recursive_mismatch',{**doc,'recursive_state_equal':False},'x',doc['source_range']),
        ('feature_hash',doc,'wrong',doc['source_range']),('source_range',doc,'x',{'start':b,'end':c})]:
        rejected(name,lambda bad=bad,expected=expected,source_range=source_range:prefix_evidence(bad,expected,source_range))
    collector=(ROOT/'gold_future_capture_collector_v2.py').read_text(encoding='utf-8')
    passed('collector_static',lambda:require(cert.inspect_collector(collector,protocol)['static_status']=='STATICALLY_CONFORMANT'))
    passed('non_authoritative_tip',lambda:require(protocol['chain_tip_policy']=='NON_AUTHORITATIVE_REBUILDABLE_INDEX'))
    passed('production_hashes',lambda:require(all(prefix.digest(ROOT/name)==h for name,h in protocol['protected_sha256'].items())))
    for name in ('gold_mt5_timezone_transition_diagnostic_v1.py','gold_recursive_prefix_final_readiness_v1.py'):
        tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        passed('no_model_imports:'+name,lambda:require(not imports.intersection({'torch','xgboost','gemini','drl_trading_v2','sklearn'})))
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        passed('no_strategy_execution:'+name,lambda:require(not calls.intersection({'predict','predict_proba','fit','load_model','order_send','account_info'})))
    for name,frozen,active,bar in [('freeze_required',None,a,b),('activation_required',a,None,b),
        ('strict_boundary',a,b,b),('no_backdating',b,b,a),('covered_bar',a,a,c)]:
        rejected(name,lambda frozen=frozen,active=active,bar=bar:boundary(bar,frozen,active,(a,c),False))
    passed('capture_vs_evaluation',lambda:require(boundary(b,a,a,(a,c),False)['holdout_evaluation_start'] is None))
    passed('eligible_boundary',lambda:require(boundary(b,a,a,(a,c),True)['holdout_evaluation_start']==b))
    tests=prefix.equivalence_tests()
    passed('no_outcome_metrics',lambda:require(all(not {'wr','pf','pnl','trades_per_day'}.intersection(row) for row in tests['tests'])))
    passed('prior_immutable',lambda:require(all(prefix.digest(ROOT/name)==h for name,h in protocol['v1_preserved_sha256'].items())))
    with tempfile.TemporaryDirectory(prefix='前綴_') as directory:
        path=Path(directory)/'證據.json';path.write_text('{}',encoding='utf-8')
        passed('unicode_paths',lambda:require(json.loads(path.read_text(encoding='utf-8'))=={}))
    print(json.dumps({'overall':'PASS','checks':checks,'synthetic_only':True}))


if __name__ == '__main__':
    self_test()
