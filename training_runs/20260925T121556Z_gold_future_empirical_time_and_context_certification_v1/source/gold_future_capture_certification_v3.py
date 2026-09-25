"""Empirical-time readiness fork; official authority is supplementary only."""
import ast
import hashlib
import json
from pathlib import Path

from gold_future_empirical_time_v1 import decision, identity

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def ast_hash(text):
    return hashlib.sha256(ast.dump(ast.parse(text), include_attributes=False).encode()).hexdigest()


def inspect_collector(text, protocol):
    tree = ast.parse(text); risks = []
    if ast_hash(text) != protocol['approved_collector_ast_sha256']:
        risks.append('Unreviewed collector AST')
    for name, expected in protocol['dependency_sha256'].items():
        if sha(ROOT/name) != expected:
            risks.append('Dependency changed: '+name)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {'fit', 'predict', 'predict_proba', 'load_model', 'order_send', 'eval', 'exec', 'unlink', 'remove', 'rename', 'truncate'}
    if any((n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else '') in forbidden for n in calls):
        risks.append('Forbidden execution or mutation')
    replacements = [n for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'os' and n.func.attr == 'replace']
    expected = ast.dump(ast.parse("file_under(root, 'chain_tip.json')", mode='eval').body, include_attributes=False)
    if len(replacements) != 1 or ast.dump(replacements[0].args[1], include_attributes=False) != expected:
        risks.append('Replacement outside non-authoritative tip')
    return {'static_status': 'NOT_STATICALLY_CONFORMANT' if risks else 'STATICALLY_CONFORMANT',
            'collector_sha256': sha(ROOT/'gold_future_capture_collector_v3.py'),
            'collector_ast_sha256': ast_hash(text), 'runtime_validator_sha256': sha(ROOT/'gold_future_empirical_time_v1.py'),
            'chain_tip_policy': 'NON_AUTHORITATIVE_REBUILDABLE_INDEX', 'blockers': risks,
            'operational_status': 'NOT_ACTIVATED',
            'limitations': 'Pinned implementation plus synthetic checks; runtime requires independently checked clock and anchors covering every admitted bar; long native spans fail closed until covered'}


def certify(run):
    protocol = load(ROOT/'gold_future_capture_protocol_v3.json')
    empirical = load(run/'empirical_time_segments.json')
    time_decision = decision(empirical['segments'], empirical['clock'])
    context = load(run/'causal_context_inventory.json')
    compatibility = load(run/'causal_context_compatibility.json')
    prefix = load(run/'prefix_protocol.json')
    source = load(run/'source_attestation_v3.json')
    attestation = load(run/'gold_future_capture_time_attestation_v3.json')
    static = inspect_collector((ROOT/'gold_future_capture_collector_v3.py').read_text(encoding='utf-8'), protocol)
    blockers = []
    try:
        identity(source)
        if not empirical['identity_verified']:
            blockers.append('SOURCE_IDENTITY_UNRESOLVED')
    except (ValueError, KeyError):
        blockers.append('SOURCE_IDENTITY_CONFLICT')
    if time_decision['status'] != 'PASS':
        blockers.append('EMPIRICAL_TIME_COVERAGE_INSUFFICIENT')
    if compatibility['status'] != 'PASS_EMPIRICAL_TIME_COMPATIBLE':
        blockers.append('CONTEXT_TIME_OR_NATIVE_PROVENANCE_UNRESOLVED')
    if not prefix['approved']:
        blockers.append('PREFIX_UNAPPROVED')
    if static['static_status'] != 'STATICALLY_CONFORMANT':
        blockers.append('COLLECTOR_STATIC_FAILURE')
    return {'formal_run_status': 'FAIL' if time_decision['status'] == 'FAIL' or static['blockers'] else 'PARTIAL' if blockers else 'PASS',
            'readiness': {'status': 'NOT_READY_MULTIPLE_BLOCKERS' if blockers else 'READY_FOR_CAPTURE_ACTIVATION', 'blockers': blockers},
            'official_broker_attestation_required': False, 'holiday_calendar_role': 'EXPLANATORY_METADATA',
            'higher_timeframe_policy': 'NATIVE_20TF_REQUIRED', 'collector_static_review': static,
            'empirical_time_status': time_decision['status'], 'context_inventory_status': compatibility['formal_status'],
            'prefix_certification_status': 'PASS' if prefix['approved'] else 'PARTIAL',
            'source_attestation_sha256': canonical(source), 'time_attestation_sha256': canonical(attestation),
            'prefix_protocol_sha256': canonical(prefix), 'inventory_root_sha256': context['inventory_root_sha256'],
            'freeze_sha256': None, 'protocol_frozen': False, 'capture_activated': False}
