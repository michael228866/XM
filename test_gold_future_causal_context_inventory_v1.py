"""Synthetic-only inventory guard tests. Never reads market rows."""
import ast
import contextlib
import io
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd
import gold_future_causal_context_inventory_v1 as b


def self_test():
    checks = []
    def rejected(name, fn):
        try:
            fn()
        except (ValueError, AssertionError):
            checks.append(name)
            return
        raise AssertionError('Not rejected: '+name)
    base = {'context_role': 'CAUSAL_CONTEXT_ONLY', 'historically_equivalent': False,
            'outcome_tuning': False, 'holdout_evidence': False, 'source_id': b.SOURCE,
            'symbol': 'GOLD#', 'broker': 'XM Global Limited', 'server': 'XMGlobal-MT5 6',
            'proposed_context_start': b.START+'Z', 'proposed_context_end': b.END+'Z', 'timeframes': []}
    for frame in b.FRAMES:
        times = pd.date_range('2024-01-01', periods=4097 if frame == 'M1' else 22,
                              freq='min' if frame == 'M1' else 'MS' if frame == 'Monthly' else 'D')
        entry = b.analyze(times, frame)
        entry.update(timeframe=frame, resampled=False, source_classification='POTENTIALLY_COMPATIBLE', source_symbol='GOLD#')
        base['timeframes'].append(entry)
    b.validate_context(base); checks.append('all_21_timeframes_required')
    rejected('missing_timeframe', lambda: b.validate_context({**base, 'timeframes': base['timeframes'][:-1]}))
    rejected('duplicate_timeframe', lambda: b.validate_context({**base, 'timeframes': base['timeframes'][:-1]+[base['timeframes'][0]]}))
    for key, value, name in [('symbol', 'GOLD', 'wrong_symbol'), ('source_id', 'OTHER', 'wrong_source_id'),
                             ('broker', 'OTHER', 'wrong_broker'), ('server', 'OTHER', 'wrong_server'),
                             ('context_role', 'HOLDOUT', 'context_role_fixed'), ('historically_equivalent', True, 'not_equivalent'),
                             ('outcome_tuning', True, 'no_outcome_tuning'), ('holdout_evidence', True, 'never_holdout'),
                             ('proposed_context_end', '2027-01-01T00:00:00Z', 'fixed_end_before_boundary')]:
        rejected(name, lambda key=key, value=value: b.validate_context({**base, key: value}))
    for key, value, name in [('resampled', True, 'resampled_rejected'),
                             ('source_classification', 'DERIVED_OR_RESAMPLED', 'derived_blocks_native'),
                             ('source_symbol', 'XAUUSD', 'wrong_file_symbol'),
                             ('source_id_if_present', 'OTHER', 'wrong_file_source'),
                             ('selected_context_bars', 4095, 'insufficient_m1')]:
        bad = deepcopy(base); bad['timeframes'][0][key] = value
        rejected(name, lambda bad=bad: b.validate_context(bad))
    bad = deepcopy(base); bad['timeframes'][1]['selected_context_bars'] = 20
    rejected('insufficient_htf', lambda: b.validate_context(bad))
    assert base['timeframes'][-1]['selected_context_bars'] == 21
    checks.append('historical_monthly_allowed')
    duplicate = b.analyze(pd.to_datetime(['2024-01-01', '2024-01-01']), 'M1')
    assert duplicate['duplicate_count'] == 1
    checks.append('duplicate_timestamp_detected')
    reverse = b.analyze(pd.to_datetime(['2024-01-02', '2024-01-01']), 'M1')
    assert reverse['non_monotonic_count'] == 1
    checks.append('timestamp_reversal_detected')
    tail = b.analyze(pd.date_range('2026-09-23', periods=4, freq='D'), 'Daily')
    assert all(r['timestamp'] < b.END and r['successor_timestamp'] <= b.END for r in tail['selected_rows'])
    checks.append('postboundary_rows_excluded')
    bad = deepcopy(base); bad['timeframes'][0]['selected_rows'][-1]['timestamp'] = b.END
    rejected('postboundary_selected_row_rejected', lambda: b.validate_context(bad))
    with tempfile.TemporaryDirectory(prefix='context_') as tmp:
        p = Path(tmp)/'資料_GOLD#_M1.csv'
        p.write_text('<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n2024.01.01\t00:00:00\t1\t1\t1\t1\t1\t0\t1\n', encoding='utf-8')
        schema = b.header(p); initial = b.sha(p)
        assert schema['timestamp_fields'] == ['<DATE>', '<TIME>']
        checks.append('unicode_paths')
        entry = {'path': str(p), 'timeframe': 'M1', 'sha256': initial, 'schema_sha256': b.digest(schema)}
        rejected('source_hash_mismatch', lambda: assert_equal(b.sha(p), '0'*64))
        rejected('schema_hash_mismatch', lambda: assert_equal(b.digest({**schema, 'columns': []}), entry['schema_sha256']))
        another = {**entry, 'timeframe': 'H1'}
        assert b.manifest_root([entry, another]) == b.manifest_root([another, entry])
        checks.extend(['manifest_deterministic_ordering', 'inventory_root_deterministic'])
        with p.open('a', encoding='utf-8') as f:
            f.write('\n')
        rejected('external_mutation_invalidates_manifest', lambda: assert_equal(b.sha(p), initial))
        import validate_gold_future_causal_context_inventory_v1 as independent
        for frame in ('M1', 'Monthly'):
            p = Path(tmp)/('GOLD#_'+frame+'_synthetic.csv')
            times = pd.date_range('2024-01-01', periods=4098 if frame == 'M1' else 24,
                                  freq='min' if frame == 'M1' else 'MS')
            with p.open('w', encoding='utf-8', newline='') as f:
                f.write('<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n')
                for t in times:
                    f.write(t.strftime('%Y.%m.%d\t%H:%M:%S')+'\t1\t1\t1\t1\t1\t0\t1\n')
            entry = b.inspect(p, Path(tmp))
            result = independent.scan(p, independent.read_header(p), frame)
            assert all(entry[k] == v for k, v in result.items())
            checks.append('independent_synthetic_csv_roundtrip:'+frame)
    assert b.classify('GOLD#_M1.csv', {'normalized_columns': ['DATE', 'TARGET', 'CLOSE']}) == 'REJECTED_STRATEGY_OUTPUT'
    checks.append('strategy_schema_rejected_before_body')
    assert b.classify('GOLD#_M1_resampled.csv', {'normalized_columns': list(b.RAW)+['DATE']}) == 'DERIVED_OR_RESAMPLED'
    checks.append('resampling_classification')
    assert all(e['certified_usable_causal_bars'] == 0 for e in base['timeframes'])
    checks.append('unresolved_timezone_never_certified')
    for name in (b.SLUG+'.py', 'run_'+b.SLUG+'.py', 'validate_'+b.SLUG+'.py'):
        tree = ast.parse((b.ROOT/name).read_text(encoding='utf-8'))
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not imports & {'xgboost', 'sklearn', 'torch', 'MetaTrader5', 'gemini', 'drl_trading_v2'}
        assert not calls & {'fit', 'predict', 'predict_proba', 'load_model', 'order_send'}
    checks.extend(['no_model_imports', 'no_strategy_execution', 'no_outcome_metrics'])
    spec = b.load(b.ROOT/('execution_spec_'+b.SLUG+'.json'))
    assert all(b.sha(b.ROOT/n) == h for n, h in spec['protected_sha256'].items())
    checks.append('protected_production_hashes')
    assert all(b.sha(b.ROOT/n) == h for n, h in spec['prior_seals'].items())
    checks.append('prior_finalized_seals_unchanged')
    print(json.dumps({'overall': 'PASS', 'checks': checks, 'synthetic_only': True}))


def assert_equal(a, b):
    if a != b:
        raise ValueError('Hash mismatch')


if __name__ == '__main__':
    self_test()
