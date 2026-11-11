"""Run one preregistered research experiment from the ledger inside a formal training_runs/ directory.

    python -B research_run_v1.py E-001

Refuses if the experiment has no preregistration, was withdrawn, lacks an approving mechanical and an
approving independent critic review, already has a result, or any of its code files changed since
preregistration. The experiment module exposes run(run_dir, spec, trials) -> dict with keys:
outcome, evidence_class, summary, failure_cause, lessons, next_questions, metrics (dict), report (markdown),
manifest_data (dict merged into manifest["data"]). The run is finalized as research_only and registered;
the result is appended to the ledger and RESEARCH_MEMORY.md is re-rendered.
"""
import argparse
import contextlib
import importlib
import io
import json
import subprocess
import sys
import traceback

import research_memory_v1 as rm
import training_run_history as trh


def mt5_manifest_data(mt5, symbols, fetch_start, fetch_end, rows, test_windows, note):
    """manifest['data'] for an evaluation that reads MT5 bars and trains nothing (no train/validation split)."""
    terminal, account = mt5.terminal_info(), mt5.account_info()
    return {
        "symbols": symbols, "data_sources": ["MT5 copy_rates_range (XM)"], "timezone": "MT5 server time (UTC+2/+3)",
        "data_start_utc": str(fetch_start), "data_end_utc": str(fetch_end),
        "train_start_utc": "not_applicable_no_training", "train_end_utc": "not_applicable_no_training",
        "train_rows": "not_applicable_no_training", "validation_start_utc": "not_applicable_no_training",
        "validation_end_utc": "not_applicable_no_training", "validation_rows": "not_applicable_no_training",
        "test_start_utc": test_windows[0][0], "test_end_utc": test_windows[-1][1], "test_rows": rows,
        "purge_details": "not_applicable_no_training", "embargo_details": "not_applicable_no_training",
        "raw_snapshot_retained": False, "reproducibility_claim": "code_and_parameters_only; " + note,
        "mt5_fetch": {"used": True, "terminal_path": getattr(terminal, "path", "unknown"),
                      "terminal_info": {"build": getattr(terminal, "build", None), "name": getattr(terminal, "name", None)},
                      "broker_info": {"server": getattr(account, "server", None), "company": getattr(account, "company", None)},
                      "fetch_start_utc": str(fetch_start), "fetch_end_utc": str(fetch_end),
                      "retrieved_at_utc": trh.utc_text(trh.now_utc()), "returned_rows": rows,
                      "not_applicable_reason": None},
    }


def trials_in_family(events, hypothesis_id):
    """Experiments already run (with a result) on hypotheses of the same family, plus this one."""
    hyps = {e["id"]: e for e in events if e["kind"] == "hypothesis"}
    family = hyps[hypothesis_id]["family"]
    prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
    done = [e for e in events if e["kind"] == "result" and hyps[prereg[e["experiment"]]["hypothesis"]]["family"] == family]
    return len(done) + 1


def finalize(run_dir, out, hypothesis_id, experiment_id):
    """Write report/metrics, complete the manifest and finalize as research_only; returns validator errors."""
    (run_dir / "report.md").write_text(out["report"], encoding="utf-8")
    trh.write_json(run_dir / "metrics.json", {"summary": out["metrics"], "folds": []})
    manifest = trh.read_json(run_dir / "manifest.json")
    manifest["data"].update(out["manifest_data"])
    manifest["model"]["not_applicable_reason"] = out.get("model_note", "no model trained; fixed rules or frozen model")
    manifest["search"]["not_applicable_reason"] = "preregistered single specification; no search"
    manifest["registry"].update({"parent_or_incumbent": hypothesis_id, "selected_configuration": "none (research only)",
                                 "validator_result": f"research_memory {experiment_id}: {out['outcome']}",
                                 **{k: out["metrics"].get(k, "n/a") for k in
                                    ("trades_per_day", "realized_win_rate", "pf", "mean_r", "pnl", "max_dd")}})
    trh.write_json(run_dir / "manifest.json", manifest)
    return trh.finalize_run(run_dir, "research_only")


def git_problems(paths):
    """AGENTS.md commit/push discipline: the code about to run must be committed and pushed."""
    root = str(rm.ROOT)
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        ahead = subprocess.run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"git check failed: {exc}"]
    problems = [f"uncommitted: {line}" for line in dirty.splitlines()]
    if ahead != "0":
        problems.append(f"{ahead} local commit(s) not pushed")
    return problems


def run(experiment_id):
    events = rm.read()
    problems = rm.verify(events)
    if problems:
        raise SystemExit(f"ledger damaged: {problems[:3]}")
    _, prereg, results = rm.state(events)
    if experiment_id not in prereg:
        raise SystemExit(f"{experiment_id} is not preregistered")
    if experiment_id in results:
        raise SystemExit(f"{experiment_id} already has a result; preregister a new experiment instead")
    if experiment_id in rm.withdrawn(events):
        raise SystemExit(f"{experiment_id} was withdrawn; see the decision log in RESEARCH_MEMORY.md")
    gate = rm.review_gate(events, experiment_id)
    if gate:
        raise SystemExit(f"{experiment_id} has not passed review: {gate}")
    p = prereg[experiment_id]
    for name, digest in p["code_files"].items():
        if rm.file_sha256(rm.ROOT / name) != digest:
            raise SystemExit(f"{name} changed since preregistration; preregister a new experiment")
    tracked = [*p["code_files"], "research_run_v1.py", "research_eval_v1.py", "research_memory_v1.py",
               "research_memory/ledger.jsonl"]
    dirty = git_problems(tracked)
    if dirty:
        raise SystemExit(f"commit and push before running (AGENTS.md): {dirty}")
    module = importlib.import_module(p["module"])
    trials = trials_in_family(events, p["hypothesis"])
    run_dir = trh.create_run(f"research_{experiment_id.lower().replace('-', '')}_{p['module']}",
                             rm.ROOT / f"{p['module']}.py", f"python -B research_run_v1.py {experiment_id}",
                             arguments=[experiment_id], seed_note="fixed seeds inside the experiment module")
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            out = module.run(run_dir, p["spec"], trials)
    except Exception:
        (run_dir / "stdout.log").write_text(log.getvalue() + traceback.format_exc(), encoding="utf-8")
        errors = trh.finalize_run(run_dir, "aborted", aborted_reason="exception; see stdout.log")
        rm.append([{"kind": "result", "experiment": experiment_id, "outcome": "aborted",
                    "evidence_class": "development", "summary": "experiment raised an exception",
                    "failure_cause": "method", "lessons": "see stdout.log", "next_questions": "fix and preregister again",
                    "run_dir": run_dir.name}])
        rm.render(rm.read())
        raise SystemExit(f"aborted; run {run_dir.name}; finalize errors {errors}")
    (run_dir / "stdout.log").write_text(log.getvalue(), encoding="utf-8")
    errors = finalize(run_dir, out, p["hypothesis"], experiment_id)
    if errors:
        raise SystemExit(f"run {run_dir.name} failed validation, not finalized: {errors}")
    trh.register_run(run_dir)
    rm.append([{"kind": "result", "experiment": experiment_id, "run_dir": f"training_runs/{run_dir.name}",
                **{k: out[k] for k in ("outcome", "evidence_class", "summary", "failure_cause", "lessons",
                                       "next_questions")}}])
    rm.render(rm.read())
    print(json.dumps({k: out[k] for k in ("outcome", "evidence_class", "summary")}, ensure_ascii=False, indent=2))
    print(f"run: training_runs/{run_dir.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    run(parser.parse_args(argv).experiment)


if __name__ == "__main__":
    main()
