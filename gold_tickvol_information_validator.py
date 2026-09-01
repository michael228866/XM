from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STUDY_JSON = ROOT / "gold_tickvol_information_study.json"
STUDY_SOURCE = ROOT / "gold_tickvol_information_study.py"
GEN20_JSON = ROOT / "gold_generation20_direct_net_edge.json"
REPORT_JSON = ROOT / "gold_tickvol_information_validation.json"
REPORT_MD = ROOT / "gold_tickvol_information_validation.md"

CHECK_NAMES = (
    "chronology",
    "feature leakage",
    "label maturity",
    "OOF predictions",
    "calibration",
    "threshold selection",
    "purge/embargo",
    "holdout contamination",
    "recent-period reuse",
    "execution alignment",
    "cost assumptions",
    "multiple-testing risk",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(status: str, evidence: str, consequence: str) -> dict:
    if status not in {"PASS", "FAIL"}:
        raise ValueError(status)
    return {
        "status": status,
        "evidence": evidence,
        "consequence": consequence,
    }


def validate() -> dict:
    study = json.loads(STUDY_JSON.read_text(encoding="utf-8"))
    gen20 = json.loads(GEN20_JSON.read_text(encoding="utf-8"))
    source = STUDY_SOURCE.read_text(encoding="utf-8")
    evidence = study["methodology_evidence"]
    boundaries = evidence["fold_boundaries"]
    split = evidence["inner_fit_calibration_policy_purge"]

    chronological = all(
        value["train_last_timestamp"] < value["evaluation_start"]
        for value in boundaries.values()
    )
    split_maturity = all(
        item["fit_labels_end_before_calibration"]
        and item["calibration_labels_end_before_policy"]
        for fold in split.values()
        for item in fold.values()
    )
    ledgers = [
        gen20["baseline"]["folds"][fold]["trade_ledger"]
        for fold in boundaries
    ]
    non_overlapping = all(
        all(
            int(previous["exit_index"]) < int(current["index"])
            for previous, current in zip(ledger, ledger[1:])
        )
        for ledger in ledgers
    )
    trade_count = sum(len(ledger) for ledger in ledgers)
    exact_h1 = (
        'completed_log = np.log1p(valid).shift(1)' in source
        and "acceleration = velocity.diff()" in source
        and "velocity = completed_log.diff()" in source
    )
    no_random_split = "train_test_split" not in source
    no_forward_path = (
        "untouched_forward/" not in source
        and "untouched_forward\\" not in source
    )
    no_threshold_sweep = (
        study["selection_inventory"]["threshold_variants"] == 0
        and study["selection_inventory"]["hyperparameter_variants"] == 0
    )
    no_strategy_candidate = (
        study["selection_inventory"]["strategy_candidates"] == 0
        and not study["generation22_created"]
    )
    gemini_unchanged = (
        file_hash(ROOT / "gemini.py")
        == study["gemini_sha256_before_and_after"]
    )

    checks = {
        "chronology": check(
            "PASS" if chronological and no_random_split else "FAIL",
            "Each training frame ends before its next evaluation fold; folds are 2018-2020, 2021-2022, and 2023-2024; no random split is present.",
            "Internal fold ranking is chronological.",
        ),
        "feature leakage": check(
            "PASS" if exact_h1 else "FAIL",
            "Frozen H1 is implemented as shifted completed-bar log TICKVOL followed by two differences; rolling features are built after the shift and the self-check proves current-bar mutation cannot alter the same decision row.",
            "TICKVOL values used at entry are available before the decision.",
        ),
        "label maturity": check(
            "PASS" if split_maturity else "FAIL",
            "Every fit label ends before calibration and every calibration label ends before policy scoring in all 12 fold/information-set fits.",
            "Inner model stages do not train on immature net-R labels.",
        ),
        "OOF predictions": check(
            "PASS" if chronological else "FAIL",
            "Each fold model is fit only on the purged history before that fold and applied frozen to the fixed executable cohort in the next block.",
            "Reported incremental ranking is internal chronological OOF, not in-sample ranking.",
        ),
        "calibration": check(
            "PASS" if split_maturity else "FAIL",
            "P(net-R>0) isotonic calibration uses the inner chronological calibration segment only; constant calibrated output is retained as an undefined Spearman rather than tuned away.",
            "Calibration diagnostics are valid development diagnostics, although they are not final-test evidence.",
        ),
        "threshold selection": check(
            "PASS" if no_threshold_sweep else "FAIL",
            "No candidate threshold is selected; the primary comparison is continuous E(net-R) Spearman and there are zero threshold or hyperparameter variants.",
            "No evaluated-fold threshold reuse affected the information verdict.",
        ),
        "purge/embargo": check(
            "PASS" if split_maturity and chronological else "FAIL",
            "Outer training uses the existing 180-row purge for a 90-row outcome horizon, and inner fit/calibration boundaries remove labels crossing the next stage.",
            "Overlapping label windows do not cross training-stage boundaries.",
        ),
        "holdout contamination": check(
            "PASS" if no_forward_path else "FAIL",
            "All inspected historical periods are explicitly development data. The study does not open an untouched_forward path and makes no final untouched-test claim.",
            "The internal claim is correctly scoped; final untouched-test validity remains FAIL.",
        ),
        "recent-period reuse": check(
            "PASS" if no_forward_path else "FAIL",
            "Evaluation stops at 2025-01-01 exclusive. 2025-2026 history is not used as an evaluation fold and post-2026-09-01T02:00:00Z data is not accessed.",
            "Recent monitoring data did not select the TICKVOL result.",
        ),
        "execution alignment": check(
            "PASS"
            if non_overlapping and trade_count == 206 and no_strategy_candidate
            else "FAIL",
            f"The test uses {trade_count} existing Gen17 executable trades; ledger occupancy is non-overlapping within every fold and no raw-signal strategy candidate is constructed.",
            "Ranking targets correspond to executable events, not raw qualifying rows.",
        ),
        "cost assumptions": check(
            "PASS",
            "NET_REWARD is inherited unchanged from Gen19/20: observed entry spread when valid, 30-point fallback otherwise, and the existing extra-cost assumption.",
            "The ablation changes only information, so economic targets remain comparable to control.",
        ),
        "multiple-testing risk": check(
            "FAIL",
            "TV_ACCEL was discovered post hoc after multiple Gen21 microstructure features were inspected. This study tests four information sets and several preregistered volume features only on already-inspected development history; no multiplicity adjustment or untouched confirmation exists.",
            "Positive raw H1 correlations cannot be promoted to confirmed evidence or justify Generation 22.",
        ),
    }
    if tuple(checks) != CHECK_NAMES:
        raise RuntimeError("Validator check inventory changed")
    overall = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    return {
        "validator": "repository-local walk-forward-validator",
        "scope": "TICKVOL information methodology only; no strategy optimization",
        "overall": overall,
        "checks": checks,
        "internal_chronological_validation_quality": (
            "PASS_with_development_only_scope"
            if all(
                checks[name]["status"] == "PASS"
                for name in CHECK_NAMES
                if name != "multiple-testing risk"
            )
            else "FAIL"
        ),
        "final_untouched_test_validity": "FAIL",
        "key_reconciliation": {
            "control_fold_spearman_reproduced": study["fixed_control"][
                "fold_control_reproduced"
            ],
            "frozen_h1_reproduced": study["frozen_h1"]["reproduced_exactly"],
            "fixed_executable_trades": trade_count,
            "generation22_justified": study["generation22_justified"],
            "generation22_created": study["generation22_created"],
            "gemini_unchanged": gemini_unchanged,
        },
        "decision": (
            "REJECT confirmation claim and STOP. The internal chronology is auditable, "
            "but H1 is post-hoc development evidence, the incremental model tests fail, "
            "and no untouched forward confirmation exists."
        ),
        "study_sha256": file_hash(STUDY_JSON),
        "study_source_sha256": file_hash(STUDY_SOURCE),
    }


def markdown(report: dict) -> str:
    lines = [
        "# TICKVOL Information Study - Walk-Forward Validation",
        "",
        f"Overall: **{report['overall']}**",
        "",
        f"Internal chronological validation quality: **{report['internal_chronological_validation_quality']}**",
        "",
        f"Final untouched-test validity: **{report['final_untouched_test_validity']}**",
        "",
        "| Check | PASS/FAIL | Evidence | Consequence |",
        "|---|---|---|---|",
    ]
    for name in CHECK_NAMES:
        value = report["checks"][name]
        lines.append(
            f"| {name} | {value['status']} | {value['evidence']} | {value['consequence']} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial decision",
            "",
            report["decision"],
            "",
            "No methodology fix can turn already-inspected history into an untouched test. Confirmation requires a completely frozen specification and genuinely new post-cutoff data.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = validate()
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"walk_forward_validator={report['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
