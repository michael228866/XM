"""Static/synthetic certification and launcher approval; never runs real research."""
import ast
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import training_run_history as history
from gold_s4_trade_economics_v1_support import ROOT, CONFIG, configuration, read, sha, require

SLUG = 'gold_s4_trade_economics_v1_infrastructure'
SOURCES = [
    'gold_s4_trade_economics_v1.py', CONFIG,
    'gold_s4_trade_economics_v1_search_space.json', 'gold_s4_a_no_long_htf_signal_reference_v1.json',
    'gold_s4_trade_economics_v1_support.py', 'gold_s4_trade_economics_v1_launcher.py',
    'validate_gold_s4_trade_economics_v1_run.py', 'manual_trade_economics_policy_v1.json',
    'test_gold_s4_trade_economics_v1.py', 'README_GOLD_S4_TRADE_ECONOMICS_V1.md',
    'certify_gold_s4_trade_economics_v1.py', 'RUN_TRAINING.bat', 'CHECK_STATUS.bat',
    'manual_training_launcher_v1.py', 'manual_training_policy_v1.json',
    'gold_manual_s4_training_data_v1.py', 'gold_manual_s4_train_validate_v1.py',
]
FLAGS = dict(MODEL_TRAINING_EXECUTED=False, REAL_VALIDATION_EXECUTED=False,
             REAL_RESEARCH_EXECUTED=False, REAL_ECONOMICS_RESEARCH_EXECUTED=False, HISTORICAL_DATA_USED=False)


def source_files():
    return sorted(set(SOURCES) | set(read(ROOT/CONFIG)['source_bindings']))


def clean_pushed():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    head = git('rev-parse', 'HEAD')
    require(not git('status', '--porcelain') and git('branch', '--show-current') == 'main'
            and git('ls-remote', 'origin', 'refs/heads/main').split()[0] == head, 'Clean pushed main required')
    return head


def execute():
    commit = clean_pushed()
    config, ref, space = configuration()
    require(all((ROOT/name).is_file() for name in source_files()),'Complete infrastructure source inventory required')
    run = history.create_run(SLUG, Path(__file__), '.venv\\Scripts\\python.exe -B '+Path(__file__).name+' --execute',
                             arguments=['--execute'], seed_note='Synthetic seed 42; no real execution')
    print('INFRASTRUCTURE_RUN='+run.name, flush=True)
    sources = source_files()
    (run/'source').mkdir()
    provenance = []
    for name in sources:
        raw = (ROOT/name).read_bytes()
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
        require(raw == blob or raw.replace(b'\r\n', b'\n') == blob, 'Source Git representation:'+name)
        shutil.copyfile(ROOT/name, run/'source'/name)
        import hashlib
        provenance.append(dict(path=name, raw_file_sha256=sha(ROOT/name), git_commit=commit,
                               git_blob_sha256=hashlib.sha256(blob).hexdigest(),
                               line_ending_mode='EXACT_BYTES' if raw == blob else 'CRLF_TO_LF', equivalence_status='PASS'))
    history.write_json(run/'execution_spec.json', dict(kind='INFRASTRUCTURE_ONLY', source_commit=commit,
                       source_files=sources, source_bindings=provenance, real_workflow_invoked=False, **FLAGS))
    with tempfile.TemporaryDirectory() as directory:
        for name in sources:
            if name.endswith('.py'):
                py_compile.compile(str(ROOT/name), cfile=str(Path(directory)/(name+'c')), doraise=True)
    with (run/'self_test_stdout.txt').open('x', encoding='utf-8') as out, (run/'self_test_stderr.txt').open('x', encoding='utf-8') as err:
        tested = subprocess.run([sys.executable, '-B', str(ROOT/'test_gold_s4_trade_economics_v1.py')], cwd=ROOT, stdout=out, stderr=err)
    try:
        result = json.loads((run/'self_test_stdout.txt').read_text(encoding='utf-8').splitlines()[-1])
    except (ValueError, IndexError):
        result = dict(overall='FAIL', checks={}, **FLAGS)
    groups = ['signal_reference_review','path_semantics_review','search_space_review','accounting_test',
              'same_bar_conflict_test','gap_test','time_stop_test','break_even_test','trailing_test',
              'pareto_logic_test','selection_logic_test','event_chain_test','validator_fixture_test',
              'manual_execution_guard_test','holdout_guard_test','production_protection_test','bat_static_test',
              'entry_compatibility_test','cooldown_compatibility_test']
    passed = (tested.returncode == 0 and result['overall'] == 'PASS' and all(result['checks'].values())
              and all(result.get(k) is False for k in FLAGS) and all(result['checks'].get(k) for k in groups))
    for group in groups:
        history.write_json(run/(group+'.json'),dict(overall='PASS' if result['checks'].get(group) else 'FAIL', synthetic_only=True,
                                                  evidence='self_tests.json', **FLAGS))
    history.write_json(run/'self_tests.json',result)
    status = 'PASS' if passed else 'FAIL'
    history.write_json(run/'validator.json',dict(overall=status, scope='STATIC_SYNTHETIC_FIXTURE_ONLY',
        checks=result['checks'], failed_checks=[k for k,v in result['checks'].items() if not v], **FLAGS))
    history.write_json(run/'metrics.json',dict(formal_run_status=status, validator_status=status,
                       total_predeclared_candidates=len(space['candidates']), **FLAGS))
    report = ('# GOLD S4 Trade Economics v1 infrastructure\n\nStatus: '+status+'\n\n'
        'INFRASTRUCTURE_ONLY. Static source and synthetic fixture certification. '
        '22 frozen configurations; 787 immutable accepted reference entries. '
        'SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY. Entire candidate fails on overlap, cooldown or daily-loss conflict. '
        'No overlap-allowed diagnostic. No fitting, inference, historical replay, real economics evaluation or holdout access. '
        'Original S5 barrier gap-fill convention retained; this is not evidence of live execution realism. '
        'Independent full real validation implemented but not executed. '
        'Only USER may double-click RUN_TRAINING.bat after committed approval. '
        'Production unchanged, no promotion. Real data availability and reference reproduction are USER runtime checks.\n')
    for name in ('report.md','findings.md'):
        (run/name).write_text(report,encoding='utf-8')
    m = read(run/'manifest.json')
    m.update(infrastructure_only=True, manual_start=False, execution_flags=FLAGS)
    m['data'].update(symbols=['GOLD#'],data_sources=['Synthetic fixtures and frozen reference metadata'],
        source_files=[dict(path='source/'+CONFIG,sha256=sha(run/'source'/CONFIG),retention_status='stored_in_run_directory')],
        timezone='UTC',raw_snapshot_retained=True,reproducibility_claim='synthetic fixtures only',
        purge_details='Synthetic isolation tests',embargo_details='No real folds executed')
    for key in ('data_start_utc','data_end_utc','train_start_utc','train_end_utc','validation_start_utc','validation_end_utc','test_start_utc','test_end_utc'):
        m['data'][key]='NOT_APPLICABLE_INFRASTRUCTURE_ONLY'
    for key in ('train_rows','validation_rows','test_rows'):
        m['data'][key]=0
    m['data']['mt5_fetch']['not_applicable_reason']='No broker access'
    m['model']['not_applicable_reason']='No candidate model training'
    m['search']['not_applicable_reason']='No real strategy search; fake candidate metrics only'
    m['registry'].update(dict.fromkeys(history.REGISTRY_FIELDS,'N/A: infrastructure only'))
    m['registry'].update(parent_or_incumbent=ref['source_run_id'],selected_configuration='22 frozen exit-economics configurations; USER-only; approval pending',validator_result=status)
    history.write_json(run/'manifest.json',m)
    require(not history.finalize_run(run,'pass' if passed else 'fail'),'Infrastructure archive validation')
    history.register_run(run)
    print(json.dumps(dict(run_id=run.name,status=status,finalized_sha256=sha(run/'FINALIZED.json'),checks=len(result['checks']),**FLAGS)))
    return 0 if passed else 1


def approve(path):
    commit = clean_pushed()
    run = Path(path).resolve()
    require(run.parent == ROOT/'training_runs' and run.name.endswith('_'+SLUG),'Infrastructure run only')
    require(not history.validate_run(run),'Sealed infrastructure')
    require(read(run/'validator.json')['overall'] == 'PASS' and read(run/'metrics.json')['formal_run_status'] == 'PASS','Certification PASS')
    require(all(read(run/'metrics.json').get(k) is False for k in FLAGS),'No real execution certification')
    for name in read(run/'execution_spec.json')['source_files']:
        require(sha(ROOT/name) == sha(run/'source'/name),'Certified source changed:'+name)
    from gold_s4_trade_economics_v1_launcher import APPROVAL
    require(not (ROOT/APPROVAL).exists(),'No overwrite of approval')
    launcher_path = ROOT/'training_launcher_config_v1.json'
    launcher = read(launcher_path)
    entry = lambda name:dict(path=name,sha256=sha(ROOT/name))
    launcher.update(approval_status='APPROVED',workflow=dict(experiment_name='gold_s4_trade_economics_v1',
        script=entry('gold_s4_trade_economics_v1.py'),validator=entry('validate_gold_s4_trade_economics_v1_run.py'),
        configuration=entry(CONFIG),datasets=[],historical_input_mode='SEALED_ACCEPTED_ENTRIES_AND_M1_PATH_USER_ONLY',native_io_review_pass=True))
    history.write_json(launcher_path,launcher)
    bindings = {name:sha(ROOT/name) for name in source_files()+['training_launcher_config_v1.json']}
    history.write_json(ROOT/APPROVAL,dict(approved=True,workflow='GOLD_S4_TRADE_ECONOMICS_V1',
        execution_owner='USER',validation_execution_owner='USER_LAUNCHED_WORKFLOW',
        certification_run=run.relative_to(ROOT).as_posix(),finalized_sha256=sha(run/'FINALIZED.json'),
        source_commit=read(run/'manifest.json')['git_commit'],result_commit=commit,bindings=bindings,**FLAGS))
    print('Approval prepared. Commit/push, then STOP. USER alone may double-click RUN_TRAINING.bat.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--execute']:
        raise SystemExit(execute())
    if len(sys.argv) == 3 and sys.argv[1] == '--approve':
        approve(sys.argv[2])
    else:
        raise SystemExit('Use --execute for synthetic certification or --approve <sealed infrastructure run>')
