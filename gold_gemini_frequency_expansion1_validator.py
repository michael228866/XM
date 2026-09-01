from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_JSON = ROOT / "gold_gemini_frequency_expansion1.json"
MODEL_ORIGIN_JSON = ROOT / "gold_long_recent_walk_forward.json"
GEMINI_FILE = ROOT / "gemini.py"
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
REPORT_JSON = ROOT / "gold_gemini_frequency_expansion1_validation.json"
REPORT_MD = ROOT / "gold_gemini_frequency_expansion1_validation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial validator for GEMINI Frequency Expansion 1"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path.name}")
    return value


def reconcile_metrics(research: dict) -> list[str]:
    errors: list[str] = []
    for policy, candidate in research["portfolios"].items():
        periods = {"pooled": candidate["pooled"], **candidate["folds"]}
        for period, values in periods.items():
            production = values["production"]
            expansion = values["expansion"]
            combined = values["combined"]
            if combined["trades"] != production["trades"] + expansion["trades"]:
                errors.append(f"{policy}/{period}: trade count does not reconcile")
            if abs(
                combined["pnl_r"]
                - production["pnl_r"]
                - expansion["pnl_r"]
            ) > 1e-8:
                errors.append(f"{policy}/{period}: PnL-R does not reconcile")
            expected_uplift = (
                0.0
                if production["trades"] == 0
                else expansion["trades"] / production["trades"]
            )
            if abs(values["frequency_uplift"] - expected_uplift) > 1e-12:
                errors.append(f"{policy}/{period}: frequency uplift is inconsistent")
            if values["unique_added_trades"] != expansion["trades"]:
                errors.append(f"{policy}/{period}: unique added count is inconsistent")
    return errors


def check(result: str, evidence: str, impact: str) -> dict:
    if result not in {"PASS", "FAIL"}:
        raise ValueError("Validator checks must be PASS or FAIL")
    return {"result": result, "evidence": evidence, "impact": impact}


def markdown_report(report: dict) -> str:
    lines = [
        "# GEMINI FREQUENCY EXPANSION 1 — adversarial validation",
        "",
        f"Overall: **{report['overall']}**",
        "",
        f"Internal chronological validation quality: **{report['internal_chronological_validation_quality']}**",
        f"Final untouched-test validity: **{report['final_untouched_test_validity']}**",
        "",
        "| Check | Result | Evidence | Impact |",
        "|---|---|---|---|",
    ]
    for name, value in report["checks"].items():
        lines.append(
            f"| {name} | {value['result']} | {value['evidence']} | {value['impact']} |"
        )
    baseline = report["metric_reconciliation"]["production"]
    reference = report["metric_reconciliation"]["diagnostic_reference_combined"]
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "| Layer | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Production | {baseline['trades']} | {baseline['trades_per_day']:.4f} | "
            f"{baseline['realized_win_rate']:.2%} | {baseline['profit_factor']:.2f} | "
            f"{baseline['mean_r']:.4f} | {baseline['pnl_r']:.2f} | "
            f"{baseline['max_drawdown_r']:.2f} |",
            f"| Diagnostic combined | {reference['trades']} | {reference['trades_per_day']:.4f} | "
            f"{reference['realized_win_rate']:.2%} | {reference['profit_factor']:.2f} | "
            f"{reference['mean_r']:.4f} | {reference['pnl_r']:.2f} | "
            f"{reference['max_drawdown_r']:.2f} |",
            "",
            f"Arithmetic reconciliation: **{report['metric_reconciliation']['result']}**.",
            "",
            "## Verdict",
            "",
            report["verdict"],
            "",
            "No validation-only correction was applied, because the failures are "
            "evidence-design limitations rather than arithmetic/reporting defects.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate() -> dict:
    research = load_json(RESEARCH_JSON)
    origin = load_json(MODEL_ORIGIN_JSON)
    reconciliation_errors = reconcile_metrics(research)
    hashes_match = (
        research["immutable_hashes"]["gemini_py_sha256"] == file_hash(GEMINI_FILE)
        and research["immutable_hashes"]["production_model_sha256"]
        == file_hash(MODEL_FILE)
    )
    forward_protected = bool(
        research.get("untouched_forward_cutoff") == "2026-09-01T02:00:00Z"
        and research.get("untouched_forward_inspected") is False
        and research["data"]["end"] < "2025-01-01"
    )
    checks = {
        "chronology": check(
            "FAIL",
            "The current model was trained/selected on 2025-2026 data, then scored 2018-2024.",
            "The backward replay cannot be interpreted as chronological OOS evidence.",
        ),
        "feature leakage": check(
            "PASS",
            "Expansion rules use frozen production scores and completed-bar technical/MTF fields only; outcomes are not rule inputs.",
            "No direct future outcome feature was identified in the expansion rules.",
        ),
        "label maturity": check(
            "PASS",
            "Entries require mature first-touch outcomes and a positive exit offset; labels are used only for offline scoring.",
            "The reported trade outcomes are mature within the available historical data.",
        ),
        "OOF predictions": check(
            "FAIL",
            "The production score is not OOF for any 2018-2024 fold; model origin starts in 2025.",
            "Winner/loser discrimination and expansion ranking are not independently estimated.",
        ),
        "calibration": check(
            "PASS",
            "No new probability calibration is fitted or claimed in this study.",
            "There is no calibration reuse layer, although the frozen production score remains non-OOF.",
        ),
        "threshold selection": check(
            "PASS",
            "Production 0.75 is immutable; 0.65-0.75 and the small contextual rules were frozen before outcome calculation.",
            "No post-result threshold rescue or broad sweep was performed.",
        ),
        "purge/embargo": check(
            "PASS",
            "No expansion model is trained; executable positions are non-overlapping and production-preempted.",
            "Overlapping event labels do not enter a fitted expansion model in this study.",
        ),
        "holdout contamination": check(
            "FAIL",
            "All 2018-2024 folds were previously inspected development history; no untouched historical holdout remains.",
            "The study cannot support a final promotion or shadow-performance claim.",
        ),
        "recent-period reuse": check(
            "FAIL",
            f"Model origin reports validation {origin['data']['validation_start']} and test {origin['data']['test_start']}; both were already inspected.",
            "The production score embeds reused recent-development decisions.",
        ),
        "execution alignment": check(
            "FAIL",
            "Non-overlap and production priority are causal, but exact broker account state, fills, slippage and export-time UTC mapping are unavailable.",
            "The reconstructed ledger is comparable research execution, not exact MT5 execution.",
        ),
        "cost assumptions": check(
            "FAIL",
            "Observed entry spread is used when valid and 30 points otherwise; exact commission/slippage and missing-spread costs are unobserved.",
            "PF and Mean-R remain assumption-sensitive despite the 10-point extra-cost stress.",
        ),
        "multiple-testing risk": check(
            "FAIL",
            "Six related portfolios are compared on repeatedly inspected development folds, and this repository contains many prior generations.",
            "Selecting any apparent winner on these folds would have material research-overfitting risk.",
        ),
    }
    required_failures = [
        name for name, value in checks.items() if value["result"] == "FAIL"
    ]
    reference_name = research["decision"]["diagnostic_reference_policy"]
    reference = research["portfolios"][reference_name]["pooled"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": research["study"],
        "validator_role": "adversarial_validation_only",
        "research_source_sha256": file_hash(RESEARCH_JSON),
        "production_hashes_match_frozen_report": hashes_match,
        "untouched_forward_protocol_preserved_in_report": forward_protected,
        "checks": checks,
        "failed_checks": required_failures,
        "overall": "PASS" if not required_failures else "FAIL",
        "internal_chronological_validation_quality": "FAIL",
        "final_untouched_test_validity": "FAIL",
        "metric_reconciliation": {
            "result": "PASS" if not reconciliation_errors else "FAIL",
            "errors": reconciliation_errors,
            "diagnostic_reference_policy": reference_name,
            "production": reference["production"],
            "diagnostic_reference_combined": reference["combined"],
        },
        "validation_only_corrections": [],
        "verdict": (
            "FAIL. No expansion candidate passed positive-expectancy discovery, "
            "and the reverse-time production scoring prevents an OOS claim. "
            "Do not create a sidecar, change production, or promote this research. "
            "The only valid next test is a fully frozen paper-shadow protocol on "
            "new post-cutoff data, without inspecting outcomes during collection."
        ),
    }
    if not hashes_match:
        report["overall"] = "FAIL"
        report["failed_checks"].append("production immutability hash")
    if not forward_protected:
        report["overall"] = "FAIL"
        report["failed_checks"].append("untouched forward protocol")
    if reconciliation_errors:
        report["overall"] = "FAIL"
        report["failed_checks"].append("metric reconciliation")
    return report


def self_check() -> None:
    sample = {
        "portfolios": {
            "x": {
                "pooled": {
                    "production": {"trades": 2, "pnl_r": 1.0},
                    "expansion": {"trades": 1, "pnl_r": -0.25},
                    "combined": {"trades": 3, "pnl_r": 0.75},
                    "frequency_uplift": 0.5,
                    "unique_added_trades": 1,
                },
                "folds": {},
            }
        }
    }
    assert reconcile_metrics(sample) == []
    sample["portfolios"]["x"]["pooled"]["combined"]["trades"] = 4
    assert reconcile_metrics(sample)
    print("SELF_CHECK_OK")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report = validate()
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {REPORT_JSON.name} and {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
