"""USER-only frozen-entry exit economics; never fits or predicts a model."""
import contextlib
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from gold_s4_trade_economics_v1_support import (
    ROOT, CONFIG, EXPERIMENT, CONTROL, configuration, read, write, sha, digest,
    require, load_inputs, entry_projection, summarize, gates, select, event, check_production,
)
from gold_s4_trade_economics_path_audit import replay_entries


def evaluate(run, entries, bars, candidate, config, ref):
    from gold_s4_trade_economics_v1_launcher import require_session
    require_session()
    ledger, failure, conflict = replay_entries(entries, bars, candidate)
    path = run/'ledgers'/(candidate['candidate_id']+'.json')
    path.parent.mkdir(exist_ok=True)
    write(path, ledger)
    compatible = failure is None
    if compatible:
        require(entry_projection(ledger) == ref['entries'], 'Frozen entry identity')
        pooled, folds = summarize(ledger, config['folds'])
        gate = gates(pooled, folds)
    else:
        pooled, folds = None, []
        gate = dict(robustness_pass=False, economic_status='NOT_EVALUATED_INCOMPATIBLE', economic_gate='NONE')
    return dict(candidate_id=candidate['candidate_id'], config=candidate,
                execution_compatibility='PASS' if compatible else 'FAIL', fail_reason=failure,
                EXECUTION_COMPATIBILITY='PASS' if compatible else 'FAIL', FAIL_REASON=failure,
                conflict_entry_id=conflict, frozen_entry_count=len(entries), completed_trades=len(ledger),
                entry_stream_sha256=ref['entry_stream_sha256'], pooled=pooled, fold_metrics=folds,
                gate=gate, ledger_path=path.relative_to(run).as_posix(), ledger_sha256=sha(path),
                source_of_truth='SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY', diagnostic=False)


def check_control(control, run, ref):
    from validate_gold_s4_trade_economics_v1_run import equal
    try:
        require(control['execution_compatibility'] == 'PASS', 'Control compatibility')
        equal({k:control['pooled'][k] for k in ref['metrics']}, ref['metrics'])
        for actual, expected in zip(control['fold_metrics'], ref['fold_metrics']):
            equal({k:actual[k] for k in expected}, expected)
        original = read(ROOT/ref['ledger_path'])
        ledger = read(run/control['ledger_path'])
        require(len(ledger) == len(original) == ref['accepted_trade_count'], 'Control count')
        for a, b in zip(ledger, original):
            equal({k:a[k] for k in ('exit_price','exit_reason','net_r','stress_r','same_bar_both_hit','risk_mult','risk_budget','account_pnl')},
                  {k:b[k] for k in ('exit_price','exit_reason','net_r','stress_r','same_bar_both_hit','risk_mult','risk_budget','account_pnl')})
            from datetime import datetime, timezone
            require(a['exit_epoch'] == datetime.fromisoformat(b['exit_time_actual_utc']).replace(tzinfo=timezone.utc).timestamp(), 'Control exit time')
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('REFERENCE_EXECUTION_MISMATCH') from exc


def research(run, config, ref, space):
    from gold_s4_trade_economics_v1_launcher import require_session
    require_session()
    print('[3/10] Load Frozen A_NO_LONG_HTF Signal Stream', flush=True)
    entries, bars = load_inputs(ref)
    receipt = dict(candidate_id='A_NO_LONG_HTF', signal_count=len(entries), accepted_trade_count=len(entries),
                   entry_stream_sha256=ref['entry_stream_sha256'], direction='LONG', threshold=.75)
    write(run/'reference_signal_loaded.json', receipt)
    space_hash = digest(space)
    event(run, 'reference_signal_loaded', space_hash, record=receipt)
    print('[4/10] Reproduce Reference Execution', flush=True)
    control = evaluate(run, entries, bars, space['candidates'][0], config, ref)
    write(run/'reference_execution.json', control)
    check_control(control, run, ref)
    event(run, 'reference_execution_pass', space_hash, CONTROL, control)
    print('[5/10] Audit Path / Intrabar Semantics', flush=True)
    audit = read(ROOT/'gold_s4_trade_economics_path_audit.json')
    event(run, 'path_semantics_frozen', space_hash, record=audit)
    event(run, 'search_space_frozen', space_hash, record=space)
    records = []
    for candidate in space['candidates']:
        print('[6/10] Execute Frozen Candidate: '+candidate['candidate_id'], flush=True)
        event(run, 'candidate_start', space_hash, candidate['candidate_id'])
        row = control if candidate['candidate_id'] == CONTROL else evaluate(run, entries, bars, candidate, config, ref)
        with (run/'candidate_results.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        records.append(row)
        event(run, 'candidate_end', space_hash, candidate['candidate_id'], row)
    require(read(run/'predeclared_search_space.json') == space, 'Search space changed')
    print('[7/10] Robustness / Economics Gates', flush=True)
    event(run, 'economics_gate_complete', space_hash, record=[r['gate'] for r in records])
    print('[8/10] Pareto / Selection', flush=True)
    decision = select(records)
    write(run/'selection.json', decision)
    write(run/'pareto_frontier.json', [r for r in records if r['candidate_id'] in decision['pareto_frontier']])
    event(run, 'pareto_complete', space_hash, record=decision['pareto_frontier'])
    event(run, 'selection_complete', space_hash, record=decision)
    event(run, 'research_freeze', space_hash, record=decision)
    return decision, records


def fill_manifest(manifest, run, ref, space, records, history):
    manifest['data'].update(symbols=['GOLD#'], data_sources=['Sealed original 2018-2024 M1 path and frozen accepted entries'],
        source_files=[dict(item, retention_status='existing_immutable_archive') for item in ref['price_files']],
        timezone='UTC', data_start_utc='2018-01-01', data_end_utc='2025-01-01',
        train_start_utc='NOT_APPLICABLE_NO_TRAINING', train_end_utc='NOT_APPLICABLE_NO_TRAINING', train_rows=0,
        validation_start_utc='2018-01-01', validation_end_utc='2025-01-01', validation_rows=787,
        test_start_utc='NOT_APPLICABLE_NO_UNTOUCHED_TEST', test_end_utc='NOT_APPLICABLE_NO_UNTOUCHED_TEST', test_rows=0,
        purge_details='Original accepted-entry chronology unchanged', embargo_details='No fitting or new data',
        raw_snapshot_retained=True, reproducibility_claim='Sealed reference path evidence; historical development only')
    manifest['data']['mt5_fetch']['not_applicable_reason'] = 'No broker access or fetch'
    manifest['model'].update(trained=False, not_applicable_reason='Frozen accepted entry stream; no fitting, inference or model loading')
    manifest['search'].update(performed=True, predefined_search_space=space, candidate_results_file='candidates.csv')
    with (run/'candidates.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=history.CANDIDATE_COLUMNS)
        writer.writeheader()
        for row in records:
            item = dict.fromkeys(history.CANDIDATE_COLUMNS, 'N/A: see candidate_results.jsonl')
            item.update(candidate_id=row['candidate_id'], parameters=json.dumps(row['config']), fold='pooled_three_folds',
                        qualification_verdict=row['fail_reason'] or row['gate']['economic_gate'])
            if row['pooled']:
                for key, source in [('executable_trades','trades'),('trades_per_day','trades_per_day'),('realized_wr','realized_win_rate'),
                                    ('pf','profit_factor'),('mean_r','mean_r'),('pnl','pnl_r'),('max_dd','max_drawdown_r'),('cost_stress_result','stress_pf')]:
                    item[key] = row['pooled'][source]
            writer.writerow(item)
    manifest['registry'].update(dict.fromkeys(history.REGISTRY_FIELDS, 'See candidate_results.jsonl'))
    decision = select(records)
    chosen = next((r for r in records if r['candidate_id'] == decision['selected_candidate']), records[0])
    manifest['registry'].update(parent_or_incumbent=ref['source_run_id'], selected_configuration=decision['selected_candidate'] or decision['research_result'], validator_result='PENDING')
    p = chosen['pooled']
    manifest['registry'].update(trades_per_day=p['trades_per_day'], realized_win_rate=p['realized_win_rate'], pf=p['profit_factor'],
                                mean_r=p['mean_r'], pnl=p['pnl_r'], max_dd=p['max_drawdown_r'])


def run_manual(token):
    from gold_s4_trade_economics_v1_launcher import begin_session, end_session
    begin_session(token)
    try:
        return _run()
    finally:
        end_session()


def _run():
    from gold_s4_trade_economics_v1_launcher import require_session, verify_approval, validation_permit
    require_session()
    import numpy  # Reviewed dependency loaded before audited I/O.
    import training_run_history as history
    from gold_manual_s4_train_validate_v1 import Tee
    from training_holdout_guard_v1 import install
    from validate_gold_s4_trade_economics_v1_run import run_authorized
    from certify_gold_s4_trade_economics_v1 import source_files, clean_pushed
    approval = verify_approval()
    config, ref, space = configuration()
    commit = clean_pushed()
    print('[2/10] Historical Data / Provenance', flush=True)
    run = history.create_run(EXPERIMENT, Path(__file__), 'RUN_TRAINING.bat (USER double-click)', seed_note='No training; deterministic frozen economics')
    manifest = read(run/'manifest.json')
    manifest.update(manual_start=True, started_by='USER_LAUNCHER', infrastructure_only=False, source_bindings=approval['bindings'],
                    source_commit=commit, MODEL_TRAINING_EXECUTED=False, REAL_ECONOMICS_RESEARCH_EXECUTED=True, HISTORICAL_DATA_USED=True)
    (run/'source').mkdir()
    for name in source_files():
        shutil.copyfile(ROOT/name, run/'source'/name)
    for source, destination in [(CONFIG,'approved_config.json'),(config['search_space'],'predeclared_search_space.json'),
                                 (config['reference_binding'],'signal_reference.json'),('gold_s4_trade_economics_path_audit.json','path_audit.json')]:
        shutil.copyfile(ROOT/source, run/destination)
    status = dict(run_id=run.name, execution_status='FAIL', research_result='NOT_RUN', economic_gate='NONE', selected_candidate=None,
                  MODEL_TRAINING_EXECUTED=False, REAL_ECONOMICS_RESEARCH_EXECUTED=True, HISTORICAL_DATA_USED=True,
                  production_changed=False, production_promoted=False, holdout_used=False, failed_checks=[])
    release = None
    try:
        release = install(ROOT, write_root=run)
        with (run/'research_stdout.txt').open('x', encoding='utf-8') as out, (run/'research_stderr.txt').open('x', encoding='utf-8') as err, contextlib.redirect_stdout(Tee(sys.stdout,out)), contextlib.redirect_stderr(Tee(sys.stderr,err)):
            decision, rows = research(run, config, ref, space)
            status.update(decision)
            fill_manifest(manifest, run, ref, space, rows, history)
            write(run/'manifest.json', manifest)
            write(run/'research_result.json', status)
            print('[9/10] Independent Validation', flush=True)
            result = run_authorized(run, validation_permit(run))
            require(result['overall'] == 'PASS', 'Independent validator FAIL: '+repr(result['failed_checks']))
            status['execution_status'] = 'PASS'
            manifest['registry']['validator_result'] = 'PASS'
    except BaseException as error:
        import traceback
        (run/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status['failed_checks'] = [str(error)]
        manifest['registry']['validator_result'] = 'FAIL'
        write(run/'failure.json', status)
    finally:
        if release:
            release()
    check_production(config)
    write(run/'manifest.json', manifest)
    write(run/'combined_result.json', status)
    write(run/'metrics.json', status)
    report = '# GOLD S4 Trade Economics v1\n\n'+json.dumps(status, indent=2)+'\n\nHistorical development only; no production promotion.\n'
    for name in ('report.md','findings.md'):
        (run/name).write_text(report, encoding='utf-8')
    from datetime import datetime, timezone
    chain = run/'research_events.jsonl'
    tip = json.loads(chain.read_text(encoding='utf-8').splitlines()[-1])['event_sha256'] if chain.exists() else None
    write(run/'finalization.json',dict(event='finalization',at_utc=datetime.now(timezone.utc).isoformat(),research_tip=tip,
          execution_status=status['execution_status'],combined_result_sha256=sha(run/'combined_result.json')))
    print('[10/10] Finalize', flush=True)
    errors = history.finalize_run(run, 'research_only' if status['execution_status'] == 'PASS' else 'aborted', aborted_reason='; '.join(status['failed_checks']) or None)
    require(not errors, 'Archive metadata: '+repr(errors))
    history.register_run(run)
    subprocess.run(['git','add','-f','--',run.relative_to(ROOT).as_posix(),'TRAINING_RUNS.md'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Archive user-started GOLD S4 Trade Economics v1 '+run.name],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT).decode().split()[0]
            == subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(), 'Remote archive')
    return status


if __name__ == '__main__':
    raise SystemExit('USER double-click RUN_TRAINING.bat required; direct real execution prohibited')
