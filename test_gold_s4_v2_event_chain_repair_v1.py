"""Synthetic event-chain and guard regressions, plus read-only source/target checks."""
import ast
import copy
import json
import tempfile
from pathlib import Path

import gold_s4_v2_event_chain_repair_v1 as repair
import validate_gold_s4_secondary_improvement_v2_run_v1_1 as fixed


def rejected(call):
    try:
        call()
    except (ValueError,PermissionError,FileExistsError):
        return True
    return False


def fixture():
    rows = [dict(candidate_id='F5_CONTROL',value=1),dict(candidate_id='A',value=2)]
    selection = dict(selected_candidate='A')
    space = dict(candidates=['F5_CONTROL','A'])
    events = []
    for i,(kind,cid,payload) in enumerate(repair.expected_events(rows,selection)):
        event = dict(sequence=i,event=kind,at_utc=f'2026-01-01T00:00:{i:02d}+00:00',
                     space_sha256=repair.digest(space),candidate_id=cid,stage=None,
                     record_sha256=repair.digest(payload) if payload else None,
                     previous_sha256=events[-1]['event_sha256'] if events else None)
        event['event_sha256'] = repair.digest(event)
        events.append(event)
    return events,rows,selection,space


def rehash(events):
    previous = None
    for event in events:
        event['previous_sha256'] = previous
        event['event_sha256'] = repair.digest({k:v for k,v in event.items() if k!='event_sha256'})
        previous=event['event_sha256']


def source_review():
    """AST equality proves all non-chain independent validation checks survived."""
    old = ast.parse((repair.TARGET/'validator_script.py').read_text(encoding='utf-8'))
    new = ast.parse(Path(fixed.__file__).read_text(encoding='utf-8'))
    funcs = lambda module:{n.name:n for n in module.body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(old),funcs(new)
    equal = lambda x,y:ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False)
    for name in ('classify','economics','decision','validate_records','isotonic_values','check_fit_positions','verify_model_file'):
        repair.require(equal(a[name],b[name]),'Unchanged independent helper:'+name)
    before,after = a['validate'].body,b['validate'].body
    # Compare the original full-validation suffix after the chain, including the
    # separate status-bearing reference_control check, through metrics/selection.
    marker="require(read(run / 'reference_control.json')"
    ai=next(i for i,n in enumerate(before) if ast.unparse(n).startswith(marker))
    bi=next(i for i,n in enumerate(after) if ast.unparse(n).startswith(marker))
    repair.require(all(equal(x,y) for x,y in zip(before[ai:],after[bi:])) and len(before[ai:])==len(after[bi:]),'Unchanged complete verification suffix')
    # The pre-chain checks are unchanged except exact-run capability and the pinned
    # archived validator hash (which must not equal the new repair source hash).
    ast_a=copy.deepcopy(a['validate']); ast_b=copy.deepcopy(b['validate'])
    ast_a.body=before[2:ai]; ast_b.body=after[2:bi]
    old_chain=next(i for i,n in enumerate(ast_a.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='expected_events')
    new_chain=next(i for i,n in enumerate(ast_b.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='check_chain')
    ast_a.body=ast_a.body[:old_chain];ast_b.body=ast_b.body[:new_chain]
    class Snapshot(ast.NodeTransformer):
        def visit_Compare(self,node):
            if "sha(Path(__file__))" in ast.unparse(node):
                node.comparators=[ast.Name(id='ORIGINAL_VALIDATOR',ctx=ast.Load())]
            return self.generic_visit(node)
    repair.require(equal(Snapshot().visit(ast_a),ast_b),'Unchanged pre-chain checks')
    loader=funcs(ast.parse((repair.ROOT/'gold_s4_secondary_improvement_v2_support.py').read_text(encoding='utf-8')))['load_evidence']
    repair.require(len(loader.body[3:])==len(b['load_evidence'].body[2:]) and
                   all(equal(x,y) for x,y in zip(loader.body[3:],b['load_evidence'].body[2:])), 'Unchanged sealed-input loader checks')
    return dict(overall='PASS',original_validator_sha256=repair.ORIGINAL_VALIDATOR,
                repair_validator_sha256=repair.sha(Path(fixed.__file__)), independent_checks_preserved=True,
                changed_checks=['reference_pass uses unwrapped record', 'UTC-aware nondecreasing timestamps',
                                'exact-run read-only capability; archived validator identity remains pinned'])


def run_tests():
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        repair.require(value,'Test:'+name)
    events,rows,selection,space=fixture()
    check('valid_chain',repair.check_chain(events,rows,selection,space)['overall']=='PASS')
    check('original_failure_synthetic',repair.inspect_chain(events,rows,selection,space,old=True)['failures'][0]['event_index']==1)
    for name,key,value in [('sequence','sequence',9),('timestamp','at_utc','2025-01-01T00:00:00+00:00'),
                           ('previous_hash','previous_sha256','0'*64),('payload','record_sha256','0'*64)]:
        bad=copy.deepcopy(events);bad[2][key]=value
        if name=='timestamp':rehash(bad)
        check(name+'_mutation',rejected(lambda:repair.check_chain(bad,rows,selection,space)))
    for kind in ('candidate_begin','candidate_result','validator_start','finalization'):
        bad=copy.deepcopy(events);bad.append({**bad[-1],'event':kind,'sequence':len(bad)})
        rehash(bad)
        check('post_freeze_'+kind,rejected(lambda:repair.check_chain(bad,rows,selection,space)))
    for phase,kind in [('VALIDATION_ARTIFACT','validator_start'),('VALIDATION_ARTIFACT','validator_result'),
                       ('ARCHIVE_ARTIFACT','finalization'),('ARCHIVE_ARTIFACT','git_commit'),('ARCHIVE_ARTIFACT','validation_amendment')]:
        repair.check_lifecycle(events,phase,kind,validation_complete=phase=='ARCHIVE_ARTIFACT')
        check('permitted_'+kind,True)
    check('late_candidate_rejected',rejected(lambda:repair.check_lifecycle(events,'ARCHIVE_ARTIFACT','candidate_result')))
    check('premature_archive_rejected',rejected(lambda:repair.check_lifecycle(events,'ARCHIVE_ARTIFACT','finalization')))
    tied=copy.deepcopy(events);tied[2]['at_utc']=tied[1]['at_utc'];rehash(tied)
    check('duplicate_time_allowed',repair.check_chain(tied,rows,selection,space)['overall']=='PASS')
    bad=copy.deepcopy(events);bad[2]['at_utc']='2026-01-01T00:00:02';rehash(bad)
    check('naive_time_rejected',rejected(lambda:repair.check_chain(bad,rows,selection,space)))
    check('jsonl_CRLF_not_hash_input',repair.check_chain([json.loads(json.dumps(e)+'\r\n') for e in events],rows,selection,space)['overall']=='PASS')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for name in ('candidate_results.jsonl','model.json','evidence.npz','predeclared_search_space.json','selection.json','validator.json','FINALIZED.json','gemini.py'):
            p=root/name;p.write_bytes(b'original')
            baseline=repair.tree(root);p.write_bytes(b'mutated')
            check('immutable_'+name,rejected(lambda:repair.unchanged(root,baseline)))
            p.write_bytes(b'original')
        p=root/'FAIL.json';repair.write_new(p,{'overall':'FAIL'})
        check('FAIL_not_overwritten',rejected(lambda:repair.write_new(p,{'overall':'PASS'})))
        from training_holdout_guard_v1 import install
        output=root/'repair';output.mkdir()
        release=install(root,write_root=output)
        try:
            check('holdout_access_rejected',rejected(lambda:(root/'future_holdout/gold_s4_v4/fake').read_bytes()))
            check('target_write_rejected',rejected(lambda:(root/'model.json').write_bytes(b'overwrite')))
        finally:release()
    good=dict(overall='PASS',full_validation_completed=True,target_run_id=repair.TARGET_ID,
              original_tree_after=repair.ORIGINAL_TREE,event_chain_integrity_status='PASS',
              research_result='IMPROVEMENT_FOUND',selected_candidate='synthetic',candidate_gate='INTERESTING')
    check('additive_amendment',repair.amendment(good,'external','seal')['original_validation_status']=='FAIL')
    for key,value in [('overall','FAIL'),('full_validation_completed',False),('target_run_id','other'),
                      ('original_tree_after','changed'),('event_chain_integrity_status','FAIL')]:
        check('amendment_reject_'+key,rejected(lambda:repair.amendment({**good,key:value},'external','seal')))
    check('direct_validation_denied',rejected(fixed.require_read_only_session))
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    release=fixed.prohibit_model_work()
    try:
        for name,call in [('xgb_train',lambda:xgb.train()),('xgb_fit',lambda:xgb.XGBClassifier().fit(None,None)),
                          ('platt_fit',lambda:LogisticRegression().fit(None,None)),('isotonic_fit',lambda:IsotonicRegression().fit(None,None)),
                          ('candidate_search_import',lambda:__import__('gold_s4_secondary_improvement_v2'))]:
            check('blocked_'+name,rejected(call))
    finally:release()
    baseline=repair.original_inventory()
    diagnosed=repair.diagnosis()
    check('actual_failure_reproduced',diagnosed['original_predicate_result']['failures'][0]['event_index']==1)
    check('actual_chain_integrity',diagnosed['corrected_chain_result']['overall']=='PASS')
    check('source_check_preservation',source_review()['overall']=='PASS')
    check('target_tree_unchanged',repair.unchanged(repair.TARGET,baseline)==baseline)
    return dict(overall='PASS',checks=checks,model_training_executed_by_repair=False,research_search_executed_by_repair=False)


if __name__=='__main__':
    print(json.dumps(run_tests(),sort_keys=True))
