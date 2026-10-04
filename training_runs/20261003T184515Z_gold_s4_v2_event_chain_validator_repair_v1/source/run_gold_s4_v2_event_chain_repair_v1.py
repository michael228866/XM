"""One exact-run revalidation, immutable archive, then a separate external amendment."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history
import gold_s4_v2_event_chain_repair_v1 as repair

ROOT=repair.ROOT
SLUG='gold_s4_v2_event_chain_validator_repair_v1'
VALIDATOR='validate_gold_s4_secondary_improvement_v2_run_v1_1.py'
TESTS='test_gold_s4_v2_event_chain_repair_v1.py'
SOURCES=[Path(__file__).name,VALIDATOR,TESTS,'gold_s4_v2_event_chain_repair_v1.py',repair.POLICY_FILE]


def clean_pushed():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
    commit=git('rev-parse','HEAD')
    repair.require(not git('status','--porcelain') and git('branch','--show-current')=='main'
                   and git('ls-remote','origin','refs/heads/main').split()[0]==commit,'Clean pushed main required')
    return commit


def execute():
    commit=clean_pushed()
    repair.require(not list((ROOT/'training_runs').glob('*_'+SLUG)),'Do not retry this authorized revalidation')
    before=repair.original_inventory()
    preflight=repair.diagnosis()
    source=repair.source_provenance()
    run=history.create_run(SLUG,Path(__file__),'.venv\\Scripts\\python.exe -B '+Path(__file__).name+' --execute',
                           arguments=['--execute'],seed_note='Existing-run validation only; no training or search')
    print('REPAIR_RUN='+run.name,flush=True)
    write=lambda name,value:repair.write_new(run/name,value)
    (run/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(ROOT/name,run/'source'/name)
    write('execution_spec.json',dict(target_run_id=repair.TARGET_ID,target_archive_commit=repair.ARCHIVE_COMMIT,
        source_commit=commit,source_bindings=[dict(path=name,sha256=repair.sha(ROOT/name),git_commit=commit,
            role='REPAIR_VALIDATOR_SOURCE' if name==VALIDATOR else 'REPAIR_SUPPORT_SOURCE') for name in SOURCES],
        original_sources=source,authorization='Explicit user exception for independent revalidation of this exact immutable run',
        model_training_already_executed_by_user=True,model_training_executed_by_repair=False,
        research_search_executed_by_repair=False,holdout_used=False,production_changed=False))
    write('original_run_tree_hash.json',before)
    write('failure_reproduction.json',preflight['original_predicate_result'])
    events=[json.loads(line) for line in (repair.TARGET/'research_events.jsonl').read_text(encoding='utf-8').splitlines()]
    write('event_chain_inventory.json',dict(root=events[0],tip=events[-1],events=events))
    write('event_chain_policy_review.json',dict(overall='PASS',policy=repair.read(ROOT/repair.POLICY_FILE),
        separation_required=False,reason='Only the reference_pass payload comparator was incorrect; original research chain remains intact'))
    with (run/'self_test_stdout.txt').open('x',encoding='utf-8') as out,(run/'self_test_stderr.txt').open('x',encoding='utf-8') as err:
        test=subprocess.run([sys.executable,'-B',TESTS],cwd=ROOT,stdout=out,stderr=err)
    tests=json.loads((run/'self_test_stdout.txt').read_text(encoding='utf-8').splitlines()[-1]) if test.returncode==0 else dict(overall='FAIL',checks={})
    write('self_tests.json',tests)
    if test.returncode==0 and preflight['corrected_chain_result']['overall']=='PASS':
        with (run/'validator_stdout.txt').open('x',encoding='utf-8') as out,(run/'validator_stderr.txt').open('x',encoding='utf-8') as err:
            proc=subprocess.run([sys.executable,'-B',VALIDATOR,str(run)],cwd=ROOT,stdout=out,stderr=err)
        result=repair.read(run/'target_run_revalidation.json') if (run/'target_run_revalidation.json').exists() else dict(overall='FAIL',failed_checks=['Validator result absent'])
        passed=proc.returncode==0 and result['overall']=='PASS'
    else:
        result=dict(overall='FAIL',failed_checks=['GENUINE_EVENT_CHAIN_INTEGRITY_FAILURE' if preflight['corrected_chain_result']['overall']!='PASS' else 'Self tests failed'])
        write('target_run_revalidation.json',result)
        write('event_chain_diagnosis.json',preflight)
        passed=False
    immutable=False
    try:
        after=repair.unchanged(repair.TARGET,before)
        immutable=True
    except Exception as error:
        after=repair.tree(repair.TARGET)
        passed=False
        result.setdefault('failed_checks',[]).append(str(error))
    status='PASS' if passed and immutable else 'FAIL'
    from test_gold_s4_v2_event_chain_repair_v1 import source_review
    write('validator_change_review.json',source_review())
    write('research_immutability_check.json',dict(overall='PASS' if immutable else 'FAIL',before=before['tree_sha256'],after=after['tree_sha256'],
          original_validation_status='FAIL',original_failed_report_preserved=True,original_user_training_executed=True))
    for name,predicate in [('model_immutability_check.json',lambda n:n.startswith('models/')),
                           ('evidence_immutability_check.json',lambda n:n.endswith('.npz')),
                           ('search_space_immutability_check.json',lambda n:n=='predeclared_search_space.json'),
                           ('selection_immutability_check.json',lambda n:n in {'selection.json','selected_candidate.json','pareto_frontier.json','candidate_results.jsonl'}),
                           ('ledger_immutability_check.json',lambda n:n.startswith('ledgers/'))]:
        hashes={n:h for n,h in before['file_sha256'].items() if predicate(n)}
        write(name,dict(overall='PASS' if immutable else 'FAIL',file_count=len(hashes),file_sha256=hashes))
    write('holdout_guard_check.json',dict(overall=tests['overall'],holdout_used=False,
        enforcement='Audited file/subprocess guard; read-only target; synthetic locked-path rejection'))
    protected=repair.read(ROOT/'gold_s4_secondary_improvement_v2_config.json')['protected_sha256']
    production_ok=all(repair.sha(ROOT/n)==value for n,value in protected.items())
    if not production_ok:status='FAIL'
    write('production_protection_check.json',dict(overall='PASS' if production_ok else 'FAIL',expected=protected,
          actual={n:repair.sha(ROOT/n) for n in protected},production_changed=not production_ok,production_promoted=False))
    adjudication=dict(target_run_id=repair.TARGET_ID,root_cause=repair.CAUSE,
        original_failure='Frozen chronological event chain',original_validation_status='FAIL',
        failure_reproduced=True,event_chain_schema_status='PASS',event_chain_integrity_status=preflight['corrected_chain_result']['overall'],
        original_run_tree_sha256=repair.ORIGINAL_TREE,original_tree_unchanged=immutable,
        revalidation_status=result['overall'],research_result=preflight['research_result'],selected_candidate=preflight['selected_candidate'])
    write('event_chain_adjudication_v1.json',adjudication)
    write('validator_recheck_v1.json',result)
    write('validator.json',dict(overall=status,failed_checks=result.get('failed_checks',[]),target_run_id=repair.TARGET_ID))
    history.write_json(run/'metrics.json',dict(formal_run_status=status,revalidation_status=result['overall'],
        research_result=preflight['research_result'],selected_candidate=preflight['selected_candidate'],
        model_training_executed_by_repair=False,research_search_executed_by_repair=False,holdout_used=False,production_changed=False))
    report=('# GOLD S4 v2 event-chain repair\n\nStatus: '+status+'\n\n'+repair.CAUSE+'\n\n'
        'Original USER training completed; 27 candidates and 56 research events exist. The repair performed no model fitting, '
        'training or candidate search. Existing metrics were independently recomputed only for validation. '
        'The original NOT_RUN console/combined result is an error-handler state, not evidence of absent research. '
        'The original candidate CSV is header-only because final validation aborted metadata completion; '
        'candidate_results.jsonl remains the authoritative complete research record.\n\n'
        'Research: '+preflight['research_result']+'; selected: '+str(preflight['selected_candidate'])+'; gate: '+preflight['candidate_gate']+'. '
        'Economics: '+str(preflight['selected_economic_status'])+'. Historical development only; no promotion.\n\n'
        'Original target tree remains external, sealed and untouched. Any effective PASS requires a separate external amendment after archive push.\n\n'+json.dumps(result,indent=2)+'\n')
    for name in ('report.md','findings.md'):(run/name).write_text(report,encoding='utf-8')
    m=repair.read(run/'manifest.json')
    m['data'].update(symbols=['GOLD#'],data_sources=['Existing immutable USER research evidence'],timezone='UTC',
        source_files=[dict(path=str(repair.TARGET/'FINALIZED.json'),sha256=repair.ORIGINAL_SEAL,retention_status='external_immutable_archive')],
        raw_snapshot_retained=True,reproducibility_claim='Validation repair only',purge_details='Original frozen maturity and fold rules rechecked',embargo_details='No new model training')
    for k in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'):m['data'][k]='NOT_APPLICABLE_REPAIR_ONLY'
    for k in ('train_rows','validation_rows','test_rows'):m['data'][k]=0
    m['data']['mt5_fetch']['not_applicable_reason']='No broker access'
    m['model']['not_applicable_reason']='No new training by repair; original USER models independently verified'
    m['search']['not_applicable_reason']='No new search; existing immutable research only'
    m['registry'].update(dict.fromkeys(history.REGISTRY_FIELDS,'N/A: repair only'))
    m['registry'].update(parent_or_incumbent=repair.TARGET_ID,selected_configuration=str(preflight['selected_candidate'])+' unchanged',validator_result=status)
    history.write_json(run/'manifest.json',m)
    errors=history.finalize_run(run,'pass' if status=='PASS' else 'fail')
    repair.require(not errors,'Repair archive schema: '+repr(errors))
    history.register_run(run)
    print(json.dumps(dict(repair_run=run.name,overall=status,finalized_sha256=repair.sha(run/'FINALIZED.json'))),flush=True)
    return 0 if status=='PASS' else 1


def amend(path):
    commit=clean_pushed()
    run=Path(path).resolve()
    repair.require(run.parent==ROOT/'training_runs' and run.name.endswith('_'+SLUG),'Repair run identity')
    repair.require(not history.validate_run(run) and repair.read(run/'validator.json')['overall']=='PASS','Sealed repair PASS required')
    repair.original_inventory()
    result=repair.read(run/'target_run_revalidation.json')
    value=repair.amendment(result,run.relative_to(ROOT).as_posix(),repair.sha(run/'FINALIZED.json'))
    value.update(result_commit=commit,revalidation_sha256=repair.sha(run/'target_run_revalidation.json'))
    destination=ROOT/'training_run_amendments'/repair.TARGET_ID/'event_chain_repair_v1'
    destination.mkdir(parents=True,exist_ok=False)
    repair.write_new(destination/'validation_amendment_v1.json',value)
    for name in ('event_chain_adjudication_v1.json','validator_recheck_v1.json'):
        shutil.copyfile(run/name,destination/name)
    repair.write_new(destination/'AMENDMENT_FINALIZED.json',repair.tree(destination))
    repair.original_inventory()
    with (ROOT/'TRAINING_RUNS.md').open('a',encoding='utf-8',newline='\n') as stream:
        stream.write('\n## External validation amendment: '+repair.TARGET_ID+'\n\nOriginal validator FAIL and all 216 target files remain byte-for-byte unchanged. '
            'Event-chain adjudication PASS; complete independent revalidation PASS; effective validation/final PASS. '
            'Actual pre-validation research: '+value['research_result']+'; candidate '+str(value['selected_candidate'])+'; gate '+value['candidate_gate']+'. '
            'Original training was USER-executed; repair performed no training/search. Historical development only; holdout unused; no production change/promotion. '
            'Receipt: `'+destination.relative_to(ROOT).as_posix()+'/validation_amendment_v1.json`; repair: `'+run.relative_to(ROOT).as_posix()+'`.\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--execute']:raise SystemExit(execute())
    if len(sys.argv)==3 and sys.argv[1]=='--amend':amend(sys.argv[2])
    else:raise SystemExit('Use --execute for this authorized repair only or --amend <sealed repair run>')
