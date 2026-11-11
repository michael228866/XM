"""One-time seed of the research ledger (2026-11-12): pre-ledger knowledge, hypotheses, decisions and the
first two preregistrations. Kept for provenance; running it again fails because the ids already exist.

    python -B research_memory/seed_2026_11_12.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402


def H(**k):
    return {"kind": "hypothesis", "status": "open", **k}


def pri(info, test, cheap, value):
    """Each 1-5 (cheap 5 = low engineering cost). Information gain counts double."""
    return {"information_gain": info, "testability": test, "cheapness": cheap, "trading_value": value,
            "score": info * 2 + test + cheap + value}


EVENTS = [
    {"kind": "data_usage", "window": ["2012-01-01", "2026-11-12"], "by": "pre-ledger research",
     "note": "Gen1-21, S4 family, TICKVOL, frequency expansion, macro/Treasury/COT/USD-FX families and the 2026-11 "
             "labs all inspected GOLD# history up to 2026-11-11; nothing before this date can be confirmatory"},
    {"kind": "observation", "id": "O-001", "status": "verified",
     "statement": "XM GOLD# costs: live spread median 31 points ($0.31, p90 33), commission 0, long swap -78.59 "
                  "points/night (short +16.81), triple on Wednesday; swap_mode reports 0 but swap is charged",
     "evidence": "gemini_signal_log.csv order fills (114) since 2026-06; MT5 deal history 252 rows; symbol_info; user confirmation"},
    {"kind": "observation", "id": "O-002", "status": "verified",
     "statement": "Shadow replay of gemini matches live fills: same 16 trades +0.133R replay vs +0.131R live; score "
                  "parity 99.9%; replay spread about 6.5 points optimistic",
     "evidence": "forward_reports/20261108_shadow_replay_v1_oct_nov"},
    {"kind": "observation", "id": "O-003", "status": "verified",
     "statement": "gemini M1 (0.65) has about zero-to-negative net expectancy: 321 replay trades Jun-Nov 2026, -0.031R, "
                  "PF 0.93; break-even win rate about 55% with 1.3/1.6 ATR exits; spread is 6-10% of the TP distance",
     "evidence": "forward_reports/20261103_shadow_replay_v1_thr065_jun_nov; STRATEGY_RESEARCH_2026_11.md section 3"},
    {"kind": "observation", "id": "O-004", "status": "provisional",
     "statement": "Every positive price-only H1 candidate (H1 C, combo A, EMA breakout) beats always-long mainly in "
                  "2024+, and is roughly equal to always-long in 2018-2023",
     "evidence": "STRATEGY_RESEARCH_2026_11.md section 4 by-year tables; quantified by E-001"},
    {"kind": "observation", "id": "O-005", "status": "verified",
     "statement": "Combo A does not transfer: unchanged on 12 other XM markets it is negative or about 0 on 11; on gold "
                  "with swap its 2014-2023 mean is -0.086R",
     "evidence": "forward_reports/20261111_cross_asset_combo_v1"},
    {"kind": "observation", "id": "O-006", "status": "provisional",
     "statement": "2014-2017 GOLD# H1 bars give implausible two-sided profits for breakout rules (+0.6..+1.2R per trade "
                  "long AND short); no spread before 2018-10 and more bars per year; treat pre-2018 as low-reliability",
     "evidence": "EMA breakout by-year table and H1 bar counts per year (2026-11-11 session)"},
    {"kind": "observation", "id": "O-007", "status": "verified",
     "statement": "All earlier 'new information' families (macro event timing, US Treasury real rates, CFTC COT, USD FX "
                  "pressure, microstructure, TICKVOL) were evaluated only as extra features of the short-horizon "
                  "(M1, about 90 min) long classifier; their closure says nothing about their native horizon",
     "evidence": "TRAINING_RUNS.md rows 20260906T060148Z, 20260909T140906Z, 20260913T141817Z; "
                 "gold_generation21_new_information.md; gold_tickvol_information_study.md"},
    {"kind": "observation", "id": "O-008", "status": "verified",
     "statement": "Public rule families on gold with XM costs (20 classic rules, 4 web strategies incl. news straddle "
                  "with slippage, gold/silver ratio, daily 16-asset trend): none positive in both 2014-2023 and 2024+ "
                  "except the EMA20/50 breakout long",
     "evidence": "forward_reports/20261108_rules_lab_v1, 20261111_web_strategies_v1, 20261111_ratio_lab_v1, 20261105_trend_daily_v1"},
    H(id="H-001", title="Candidate H1 edges are drift exposure, not timing skill", family="timing_vs_drift",
      mechanism="gold rose strongly in 2024-2026; any long-biased entry filter earns that drift; removing each month's "
                "drift should erase the edge",
      novelty="first decomposition of R into monthly drift vs timing residual with month-matched random nulls; earlier "
              "work compared only with whole-period always-in",
      information_needed="GOLD# H1 bars; frozen candidate definitions",
      falsified_if="a candidate's 2018-2023 timing-residual month-bootstrap CI is above 0 and its 2024+ residual is positive",
      data_mining_risk="low: no fitting, fixed candidates, controls included", priority=pri(5, 5, 5, 3)),
    H(id="H-002", title="Exogenous macro data work at their native weekly horizon", family="exogenous_horizon",
      mechanism="real yields are gold's opportunity cost and managed-money positioning mean-reverts over weeks; both act "
                "over days to weeks, not 90 minutes",
      novelty="reuses certified Treasury/COT data that were closed only in the M1 classifier frame (O-007); tests each "
              "signal alone on non-overlapping weekly returns",
      information_needed="certified treasury_daily_canonical.csv, cftc_gold_cot_features.csv, GOLD# D1",
      falsified_if="discovery (2016-2022) IC in the wrong direction or |t| < 1",
      data_mining_risk="low-medium: two pre-declared signals with fixed signs; 2023-2024 prices were inspected before",
      priority=pri(5, 4, 5, 3)),
    H(id="H-003", title="US macro surprises (actual vs forecast) cause post-release gold drift", family="event_surprise",
      mechanism="surprises reprice the USD and real-rate path; slow repositioning can extend the move for hours",
      novelty="earlier macro work used event timing only; surprise sign/size and MT5 impact flags are new information",
      information_needed="usd_calendar_values.csv from ExportUsdCalendarValues (pending user export); GOLD# M1",
      falsified_if="post-release returns in the surprise direction are not positive beyond spread at the pre-declared horizon in 2014-2022",
      data_mining_risk="medium: several horizons; one primary horizon must be pre-declared", priority=pri(4, 3, 4, 3)),
    H(id="H-004", title="Gold's drift concentrates in specific sessions; intraday-only longs capture it without swap",
      family="session_structure",
      mechanism="Asian physical demand and the London fix create session asymmetries; swap is charged only across the "
                "rollover, so intraday holding avoids about 7%/yr",
      novelty="changes the cost structure instead of the entry signal; no earlier session-return decomposition",
      information_needed="GOLD# H1 2018+ (reliable bars)",
      falsified_if="no session's mean return exceeds spread consistently across 2018-2022 years",
      data_mining_risk="medium-high: session choice can be mined; fix 4 sessions in advance and confirm only forward",
      priority=pri(3, 5, 5, 3)),
    H(id="H-005", title="Scaling long exposure by inverse forecast volatility improves risk-adjusted drift harvesting",
      family="second_moment",
      mechanism="volatility clustering and the leverage effect: high-vol periods have worse return per unit of risk",
      novelty="predicts the second moment instead of direction; never targeted in XM",
      information_needed="GOLD# D1/H1", falsified_if="vol-managed Sharpe <= constant exposure in 2018-2023 after costs",
      data_mining_risk="low: one fixed rule", priority=pri(3, 5, 5, 2)),
    H(id="H-006", title="Strategy timing skill persists over months (premise of any regime switch or bandit)",
      family="adaptivity",
      mechanism="if edges come from slow regimes, recent realized timing residual should predict the next period",
      novelty="tests the premise of adaptive/meta/bandit layers before building them",
      information_needed="monthly timing-residual streams from E-001",
      falsified_if="rank correlation of consecutive 3-month residuals <= 0", data_mining_risk="low",
      priority=pri(4, 4, 4, 2)),
    H(id="H-007", title="For M1 trading, execution cost (hour-of-day spread, entry method) matters more than the model",
      family="execution",
      mechanism="spread is 6-10% of the M1 TP distance (O-003) and varies by hour; lower effective cost may flip expectancy",
      novelty="cost-side rather than prediction-side improvement",
      information_needed="gemini_signal_log.csv spreads by hour; M1 bars",
      falsified_if="restricting to the cheapest hours does not raise replay mean R by more than 0.03R",
      data_mining_risk="medium", priority=pri(3, 4, 4, 3)),
    {"kind": "decision", "id": "D-001",
     "statement": "Stop new price-only short-horizon timing variants (new indicators, thresholds, TP/SL, model swaps on the same features)",
     "rationale": "Gen1-21, the S4 family and the 2026-11 labs (O-008) cover this space; every positive result so far "
                  "looks like drift (O-004). New work must add information, change the cost structure, or test a premise"},
    {"kind": "decision", "id": "D-002", "statement": "Only data after 2026-11-12 or newly acquired sources can be confirmatory",
     "rationale": "all GOLD# history up to 2026-11-11 was inspected; the ledger refuses confirmatory preregistrations on used windows"},
]

SPECS = {
    "E-001": ("H-001", "exp_edge_decomposition_v1",
              ["exp_edge_decomposition_v1.py", "research_eval_v1.py", "h1_runner_v1.py", "combo_lab_v1.py",
               "rules_lab_v1.py", "strategy_lab_v1.py"],
              {"windows": [["2018-01-01", "2024-01-01"], ["2024-01-01", "2026-11-12"]], "tp": 3.0, "sl": 1.5,
               "hold": 48, "swap_charges_per_h1_bar": 7 / 115},
              "H-001 supported if no non-control candidate has 2018-2023 timing-residual CI > 0 AND 2024+ residual > 0; "
              "refuted otherwise; inconclusive if the random control shows |residual| >= 0.1"),
    "E-002": ("H-002", "exp_information_horizon_v1", ["exp_information_horizon_v1.py"],
              {"windows": [["2016-01-01", "2023-01-01"], ["2023-01-01", "2025-01-01"]], "spread_frac": 0.00008},
              "a signal is supported if discovery IC has the declared sign with |t| >= 2, the check window has the same "
              "sign, and weekly long/flat Sharpe beats always-long in both windows net of costs; refuted if discovery "
              "IC has the wrong sign or |t| < 1"),
}


def main():
    events = list(EVENTS)
    for eid, (hyp, module, files, spec, rule) in SPECS.items():
        events.append({"kind": "preregistration", "id": eid, "hypothesis": hyp, "role": "exploratory", "module": module,
                       "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
                       "evaluation_windows": spec["windows"], "decision_rule": rule})
    events.append({"kind": "decision", "id": "D-003", "statement": "Start with E-001 and E-002",
                   "set_status": {"H-001": "testing", "H-002": "testing"},
                   "rationale": "highest information gain per cost: E-001 tells whether price-only timing has any skill "
                                "beyond drift; E-002 whether the closed exogenous families were closed for the wrong reason"})
    rm.append(events)
    rm.render(rm.read())
    print(f"ledger events: {len(rm.read())}; integrity: {rm.verify(rm.read()) or 'OK'}")


if __name__ == "__main__":
    main()
