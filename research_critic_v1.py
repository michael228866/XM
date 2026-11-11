"""Mechanical critic: tries to reject a preregistered experiment before it runs.

It is one of two required reviews (research_memory_v1.review_gate); the other must come from an independent
critic, i.e. any reviewer other than "mechanical" and other than the preregistration's author. A critic is
told to break the design, not to improve it: hidden leakage, unfair controls, multiple comparisons, market
direction exposure, understated costs, small samples and conclusions wider than the evidence.

Blocking checks read these preregistration fields:
  training_window_note, claim_scope, decision_rule (with numbers; Sidak/Bonferroni when trials > 1)
  spec: controls, directional_benchmark, trials_declared, cost_multipliers (max >= 2), cost_note (mentions swap),
        metrics (net_mean, sharpe, calmar, max_dd), power_note, optional model_train_end

    python -B research_critic_v1.py E-005            # print findings
    python -B research_critic_v1.py E-005 --record   # append them to the ledger as a review
"""
import argparse
import re
import sys

import research_memory_v1 as rm

REQUIRED_METRICS = {"net_mean", "sharpe", "calmar", "max_dd"}


def critique(p, events, root=rm.ROOT):
    """Return (blocking, notes) for preregistration p."""
    spec, blocking, notes = p["spec"], [], []
    if not p.get("training_window_note"):
        blocking.append("leakage: no training_window_note (which data any fitted part has seen)")
    train_end = spec.get("model_train_end")
    if train_end and any(w[0] < train_end for w in p["evaluation_windows"]):
        blocking.append(f"leakage: an evaluation window starts before model_train_end {train_end}")
    if not spec.get("controls"):
        blocking.append("controls: none declared")
    if not spec.get("directional_benchmark"):
        blocking.append("direction exposure: no directional_benchmark to separate drift from the claimed effect")
    trials = spec.get("trials_declared")
    if not isinstance(trials, int) or trials < 1:
        blocking.append("multiple comparisons: trials_declared missing")
    elif trials > 1 and not re.search(r"sidak|bonferroni", p["decision_rule"], re.I):
        blocking.append(f"multiple comparisons: {trials} trials but the decision rule has no Sidak/Bonferroni adjustment")
    mults = spec.get("cost_multipliers") or []
    if not mults or max(mults) < 2:
        blocking.append("costs: no stress at >= 2x the assumed spread (cost_multipliers)")
    if "swap" not in str(spec.get("cost_note", "")).lower():
        blocking.append("costs: cost_note does not state how overnight swap is charged")
    missing = REQUIRED_METRICS - set(spec.get("metrics", []))
    if missing:
        blocking.append(f"metrics: missing {sorted(missing)}")
    if not spec.get("power_note"):
        blocking.append("sample size: no power_note (expected observations and the effect size they can detect)")
    if not p.get("claim_scope"):
        blocking.append("over-generalisation: no claim_scope (what a result can and cannot conclude)")
    if not re.search(r"\d", p["decision_rule"]):
        blocking.append("decision rule has no numeric threshold")
    for name, digest in p["code_files"].items():
        if not (root / name).exists() or rm.file_sha256(root / name) != digest:
            blocking.append(f"code: {name} differs from the preregistered hash")

    before = events[:events.index(p)] if p in events else events
    reuse = sum(1 for w in p["evaluation_windows"] for u, _ in rm.used_windows(before) if rm.overlaps(w, u))
    if reuse:
        notes.append(f"data reuse: evaluation windows overlap {reuse} earlier uses; evidence is development only")
    hyps, _, _ = rm.state(events)
    prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
    family = hyps[p["hypothesis"]]["family"]
    done = sum(1 for e in events if e["kind"] == "result" and hyps[prereg[e["experiment"]]["hypothesis"]]["family"] == family)
    if done:
        notes.append(f"family '{family}' already has {done} result(s); the runner widens the Sidak count")
    return blocking, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    events = rm.read()
    _, prereg, _ = rm.state(events)
    if args.experiment not in prereg:
        raise SystemExit(f"{args.experiment} is not preregistered")
    blocking, notes = critique(prereg[args.experiment], events)
    for line in blocking:
        print("BLOCK", line)
    for line in notes:
        print("note ", line)
    verdict = "revise" if blocking else "approve"
    print(verdict)
    if args.record:
        rm.append([{"kind": "review", "experiment": args.experiment, "reviewer": "mechanical", "verdict": verdict,
                    "findings": blocking + notes or ["all mechanical checks passed"]}])
        rm.render(rm.read())
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
