"""V4 readiness and prefix certification; no model or strategy execution."""
import ast
from pathlib import Path

from gold_future_capture_collector_v2 import load, encode, sha
from validate_gold_future_native_context_v1 import validate

ROOT = Path(__file__).resolve().parent


def static_review(protocol):
    risks = []
    for name, expected in protocol['runtime_code_sha256'].items():
        raw = (ROOT/name).read_bytes()
        if sha(raw) != expected:
            risks.append('CODE_HASH:' + name)
        if not name.endswith('.py'):
            continue
        tree = ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or '']
                if any(any(x in m.lower() for x in ('xgboost', 'sklearn', 'drl_trading', 'gemini', 'torch')) for m in modules):
                    risks.append('STRATEGY_MODEL_IMPORT:' + name)
            if isinstance(node, ast.Call):
                called = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                if called in {'order_send', 'order_check', 'predict', 'predict_proba', 'fit', 'load_model', '_asdict'}:
                    risks.append('PROHIBITED_CALL:' + name)
    return {'status': 'STATICALLY_CONFORMANT' if not risks else 'FAIL', 'blockers': risks,
            'runtime_code_sha256': protocol['runtime_code_sha256'],
            'scope': 'Pinned source and synthetic evidence; actual runtime validation remains mandatory'}


def prefix_decision(native, inventory):
    prior = load(ROOT/'gold_recursive_prefix_causal_context_protocol_v1.json')
    pipeline = (ROOT/'drl_trading_v2.py').read_bytes()
    functions = [n for n in ast.walk(ast.parse(pipeline)) if isinstance(n, ast.FunctionDef) and n.name == 'add_indicators']
    implementation = sha(ast.dump(functions[0], include_attributes=False).encode()) if len(functions) == 1 else None
    approved = (native['status'] == 'PASS' and sha(pipeline) == prior['pipeline_sha256']
                and implementation == prior['feature_implementation_sha256'])
    return {'policy': 'PREDECLARED_CAUSAL_REINITIALIZATION', 'approved': approved,
            'status': 'PASS' if approved else 'PARTIAL', 'historically_equivalent': False,
            'outcome_tuning': False, 'context_role': 'CAUSAL_CONTEXT_ONLY', 'holdout_evidence': False,
            'pipeline_sha256': sha(pipeline), 'feature_implementation_sha256': implementation,
            'initialization_semantics': prior['initialization_semantics'],
            'inventory_root_sha256': inventory.get('inventory_root_sha256'),
            'context_fetch_completed_at_utc': inventory.get('context_fetch_completed_at_utc'),
            'causal_upper_bound_utc': inventory.get('causal_upper_bound_utc'),
            'approval_scope': 'Frozen native sequence initialization; future boundary must follow acquisition; feature eligibility independently checked at boundary',
            'historical_normalized_utc_claim': False}


def certify(run):
    protocol = load(ROOT/'gold_future_capture_protocol_v4.json')
    native = load(run/'native_context_validation.json')
    regime = load(run/'current_regime_decision.json')
    prefix = load(run/'prefix_protocol.json')
    static = static_review(protocol)
    blockers = []
    for label, good in [('NATIVE_CONTEXT', native['status'] == 'PASS'),
                        ('CURRENT_REGIME', regime['status'] == 'PASS'),
                        ('PREFIX', prefix['approved'] is True),
                        ('COLLECTOR_STATIC', static['status'] == 'STATICALLY_CONFORMANT')]:
        if not good:
            blockers.append(label)
    if not blockers:
        inventory = load(run/'context_seed'/'gold_future_native_context_inventory_v1.json')
        validate(run/'context_seed', inventory['entries'][0]['source_commit'], inventory['entries'][0]['script_sha256'])
    result = 'FAIL' if 'FAIL' in (native['status'], regime['status']) or static['status'] == 'FAIL' else 'PARTIAL' if blockers else 'PASS'
    return {'status': 'READY_FOR_CAPTURE_ACTIVATION' if not blockers else 'NOT_READY_MULTIPLE_BLOCKERS',
            'formal_run_status': result, 'blockers': blockers,
            'collector_static_review': static, 'prefix_approved': prefix['approved'],
            'runtime_fail_closed': True, 'strategy_outcome_inspected': False,
            'protocol_frozen': False, 'capture_activated': False}
