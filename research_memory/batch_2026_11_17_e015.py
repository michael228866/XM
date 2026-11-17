"""Ledger batch: H-023 (conservative cost upper bound), D-021 (Priority Engine v2 adopted; H-013 design-only approval,
prerequisites and open cost-input condition; H-023 formal approval) and the E-015 preregistration.

    python -B research_memory/batch_2026_11_17_e015.py           # dry run
    python -B research_memory/batch_2026_11_17_e015.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

H023 = {
    "kind": "hypothesis", "id": "H-023", "family": "cost_bound", "status": "testing",
    "title": "以交易時點之前的歷史點差，可以建立在前向資料上維持覆蓋率、又不至於過度保守的單筆報價成本上界",
    "mechanism": "點差的水準會漂移，但分布形狀（尤其換日時段的放大）相對穩定；用近期分位數而非平均值，可以給交易決策一個保守的成本輸入",
    "novelty": "目標從預測平均點差（E-014 Q1）改為上界的覆蓋率與漏判成本；以 tick 第一筆報價為成本，不再用 M1 欄位（分鐘最小值）",
    "information_needed": "GOLD# M1 SPREAD 欄位 2019+、tick 報價 2023-08+（每隔一個交易日）、demo 成交紀錄",
    "falsified_if": "預先登記的所有進場候選上界在 2025+ 前向期都未同時達到：覆蓋率 >= 93%（Sidak CI 下界 >= 90%）、每年 >= 90%、"
                    "換日 >= 85%、其他時段 >= 90%、整點決策時刻 >= 90%、未覆蓋成本占比 <= 3%、上界／實際 <= 1.75"
                    "（出場上界也要通過才算成立；高成本尖峰的漏判率是強制標籤，不是門檻）",
    "data_mining_risk": "medium-low：候選上界、分位數、門檻與選擇順序全部在看前向資料前固定",
    "priority_inputs": {"info_gain": 4, "novelty": 3, "falsifiability": 5, "data_quality": 4, "economic_value": 4,
                        "simplicity": 4, "mining_risk": 2},
    "tags": {"information": ["tick_spread"], "target": ["cost"]},
}

H013_PREREQS = [
    {"id": "cost_upper_bound", "need": "成本輸入通過可靠性門檻：E-015（H-023）依預先登記判定 supported，並指定進場與出場上界；"
                                       "若進場上界 spikes_covered=false，H-013 的回測與實盤都必須跳過即時報價高於上界的進場；"
                                       "若出場上界 spikes_covered=false，出場成本一律乘 2（未經驗證的壓力值）；不涵蓋週末持倉",
     "unblock": "執行 E-015；若 refuted 或 inconclusive，需要新的成本資料來源（更密的 tick、實盤成交）並另行預先登記",
     "met_by": {"experiment": "E-015", "outcomes": ["supported"]}, "evidence": ["E-014", "O-020", "D-020"]},
    {"id": "prereg_v2_reviewed", "need": "H-013 新版預先登記（問題 A 事後診斷與問題 B 決策時點預測分開；成本用 E-015 上界；"
                                         "持有時間與 gemini 實際持倉一致）通過機械與獨立 Critic 審查",
     "unblock": "依 H013_PREREG_DRAFT_V2.md 寫成實驗程式並預先登記 → Critic → 修正 → 鎖定"},
    {"id": "trade_sample", "need": "gemini 交易樣本足以檢驗第二條款（低可交易性組回放 R 至少低 0.05R）",
     "unblock": "目前 demo 只有 157 筆來回（2026-05 起，全部多單，中位持倉 4.9 分鐘），差 0.05R 的標準誤約 0.17-0.18R，"
                "要約 7,000 筆才到 2 個標準誤；需改用預先登記的歷史回放（並標示模型樣本內期間）或改寫第二條款（需新版預先登記與核准）"},
]

D021 = {
    "kind": "decision", "id": "D-021",
    "statement": "使用者授權（2026-11-17，第五階段）：(1) Priority Engine v2 生效：生命週期、證據品質、前提、資料品質、執行資格、"
                 "使用者核准分開計算；樣本不足改為不隨狀態消失的證據品質條件（修正 D-019 使 H-018 +2 的缺陷，v1 歷史凍結於 "
                 "research_memory/rerank_history_engine_v1.md）；同一根因每個維度只計一次；研究執行器在任何前置條件未成立時拒絕正式執行。"
                 "(2) H-013 只核准研究設計與可行性檢查（design_only），不核准正式驗證；成本輸入未通過可靠性門檻（E-014 Q1 not_reliable）"
                 "列為開放的證據品質條件；正式驗證的前置條件：成本上界（E-015 supported）、新版預先登記通過審查、交易樣本足夠。"
                 "H-013 原有門檻不變，若要修改需新版預先登記與核准。(3) 新增 H-023（保守成本上界），核准在審查通過後正式執行 E-015。",
    "rationale": "排名不等於執行資格；負面證據不能因狀態替換而提高優先分數；H-013 排名第一，但成本輸入尚未可靠",
    "set_approval": {"H-013": "design_only", "H-023": "formal"},
    "set_prerequisites": {"H-013": H013_PREREQS},
    "quality": {"H-013": {"cost_input_unreliable": {"state": "open", "evidence": ["E-014", "O-020", "D-020"],
                                                    "note": "E-014 Q1 not_reliable；D-020 只允許保守水準"}}},
    "set_depends": {"H-013": [{"on": "E-014", "type": "evidence", "note": "成本輸入：近 90 天水準 × 1.21；週內小時結構樣本外不穩"},
                              {"on": "H-023", "type": "evidence", "note": "成本上界（E-015）"}],
                    "H-023": [{"on": "E-014", "type": "evidence", "note": "bar 欄位是分鐘最小值；點差水準逐年漂移"}]},
}


def e015():
    module = "exp_cost_bound_v1"
    files = [f"{module}.py", "exp_spread_cost_v1.py"]
    criteria = {"coverage_min": 0.93, "coverage_ci_lower_min": 0.90, "year_coverage_min": 0.90,
                "rollover_coverage_min": 0.85, "other_coverage_min": 0.90, "hour_start_coverage_min": 0.90,
                "uncovered_cost_share_max": 0.03, "bound_over_cost_max": 1.75, "high_cost_miss_max": 0.5}
    spec = {
        "bars_start": "2019-01-01", "tick_start": "2023-08-07", "tick_every_nth_day": 2,
        "calibration": ["2023-08-07", "2025-01-01"], "forward": ["2025-01-01", "2099-01-01"],
        "windows": [["2025-01-01", "2099-01-01"]], "quantile": 0.95, "adaptive_days": 90, "level_days": 5,
        "min_trailing_tick_days": 20, "min_cell_minutes": 200, "a_factor_d020": 1.21, "horizon_minutes": 60,
        "exit_lookahead_hours": 3, "max_exit_gap_minutes": 180, "high_cost_multiple": 2.0, "min_group_minutes": 500, "bootstrap_draws": 2000, "seed": 20261117,
        "rollover_check_start": "2019-01-01", "rollover_flag_share_min": 0.9, "factor_year_tol": 0.10,
        "factor_group_tol": 0.15, "criteria": criteria, "trials_declared": 11,
        "candidates": ["A_fixed", "A_cal", "A_x1.5", "A_x2", "B95", "C95", "D95", "B95(exit)", "C95x", "D95x", "E95x"],
        "preference": ["A_fixed", "A_cal", "A_x1.5", "A_x2", "B95", "C95", "D95"],
        "exit_preference": ["B95", "C95x", "D95x", "E95x"],
        "selection_rule": "recommended entry bound = first passing candidate in `preference` (simplest first); exit bound = "
                          "first in `exit_preference` passing the same criteria on the +60-minute target; nothing is "
                          "chosen by forward performance beyond pass/fail",
        "cost_target": "quoted spread of the first tick quote inside a minute (entry) and of the first quote at or after "
                       "60 minutes later (exit; across the daily break to the reopen quote, not across a weekend), points",
        "cost_accounting": "points (0.01 USD/oz), the gemini.py unit (ask - bid) / point. On bid-priced bars a round trip "
                           "pays ONE spread (long at entry, short at exit); charging it at both ends double counts. "
                           "Slippage (fill vs quote side), commission, fee and swap are reported separately from demo "
                           "deals only; demo deal profit is checked to equal the fill-price difference (spread inside "
                           "the prices). Stress x1.5 / x2 are scenarios, not validated costs.",
        "involves_trading": False,
        "not_applicable_reason": "no positions are simulated: candidates are scored on quoted-spread coverage; direction "
                                 "and swap cannot enter any decision statistic (demo deals are only decomposed)",
        "data_windows": [
            {"window": ["2019-01-01", "2099-01-01"], "uses": ["spread_field"],
             "purpose": "rollover spike hour per month (DST check) and trailing means of the bar SPREAD field (A rules)"},
            {"window": ["2023-08-07", "2099-01-01"], "uses": ["spread_field", "h1_or_finer_price"],
             "purpose": "tick quotes: minute first-quote spreads, trailing quantiles, the 1.21 factor, exit targets"}],
        "controls": ["mean rule A_fixed (D-020) and its calibration-only factor A_cal as non-quantile baselines",
                     "stress multipliers x1.5 and x2 scored, not assumed",
                     "single quantile (B95) vs rollover-split (C95) vs level-adaptive (D95)",
                     "calibration-window scores reported beside forward scores",
                     "coverage by year, session, rollover vs other; month-block bootstrap CI"],
        "metrics": ["coverage", "coverage_ci", "coverage_by_year", "coverage_rollover", "coverage_other",
                    "coverage_by_session", "uncovered_cost_share", "bound_over_cost", "high_cost_miss_rate", "mae",
                    "bias_share", "factor_by_year_session_level", "rollover_spike_hour_by_month",
                    "profit_equals_price_difference_share", "slippage_points", "coverage_hour_start",
                    "coverage_spike_window_2355_0105", "coverage_exit_crossing_break", "worst_month", "worst_quarter",
                    "coverage_ci_3month_blocks", "spikes_covered", "high_cost_miss_rate_rollover"],
        "power_note": "forward 2025-01..2026-11 is ~23 months; every 2nd trading day gives ~240 tick days x ~1300 minutes "
                      "~ 300k minutes, but minutes within a day are dependent, so month blocks (~23) carry the CI. A true "
                      "coverage of 0.95 gives a month-block CI of roughly +-0.01-0.02; the 0.93 point / 0.90 lower bound "
                      "rule separates a working 95% bound from one at <= 0.90. High-cost events (> 2x level) are rare "
                      "(E-014: extreme runs 0.29 per 100 hours), so their miss rate has a wide interval; it sets the binding spikes_covered label, not the pass.",
        "feasibility_probe": "before preregistration only formats and counts were read: tick history from 2023-08-04, "
                             "point 0.01, gemini_trade_history 157 round trips (2026-05..11, long only, commission/fee/swap "
                             "all 0), signal log columns; no spread quantile, coverage or bound was computed",
    }
    rule = ("Sidak over trials_declared=11 scored candidates (entry: A_fixed, A_cal, A_x1.5, A_x2, B95, C95, D95; exit: "
            "B95, C95x, D95x, E95x) times the family count from the runner: CI level 0.95^(1/k), month-block bootstrap "
            "(2000 draws). Every candidate for day d uses only bars and tick days before d (previous 90 days, >= 20 tick "
            "days). A candidate PASSES on the forward window (2025-01-01+) only if all hold: coverage (cost <= bound) >= 0.93 "
            "and its CI lower bound >= 0.90; every forward year >= 0.90; rollover minutes (server 23/0/1; for exits the "
            "exit time's hour) >= 0.85 and other minutes >= 0.90; hour-start minutes (HH:00, H-013's decision times) "
            ">= 0.90; uncovered cost share sum(max(0, cost - bound)) / sum(cost) <= 0.03; mean bound / mean cost <= 1.75; "
            "A_* rules (constant factor) also need the factor to be "
            "stable. Entry candidates are scored on the entry target, exit candidates on the exit target (first quote at "
            "or after +60 minutes, across the daily break to the reopen, not across a weekend). H-023 supported only if an "
            "entry AND an exit candidate pass (recommended = first passing in each preference order; a short pays the exit "
            "spread); inconclusive if only one side passes or a candidate fails only the CI lower bound; else refuted. "
            "Binding label (not a pass gate, since a nominal-95% bound cannot cover most > 2x events under any "
            "continuous spread distribution): spikes_covered = high-cost events (NON-rollover minutes above 2x the "
            "non-rollover level known that morning) missed <= 50%; if the recommended entry bound has spikes_covered "
            "false, H-013 must skip entries whose live quote exceeds the bound, and if the exit bound has it false, "
            "H-013 must multiply exit costs by 2 (a fixed, unvalidated stress value); rollover high-cost events reported separately. "
            "Reported, not decisive: 3-month-block CI, worst month and quarter, coverage in the 23:55-01:05 spike window and "
            "on break-spanning exits, coverage on demo-deal minutes, calibration scores, MAE and bias. "
            "Factor: the tick-mean / bar-mean ratio (the 1.21) is stable only if every forward year is within 10% and every "
            "forward session and calibration-tercile level within 15% of the calibration value; otherwise it must not be "
            "applied to future data as a constant. Rollover flag: the fixed server-hour flag holds if the month's spike "
            "hour is in 23/0/1 in >= 90% of months 2019+. H-013's cost prerequisite is met only by supported.")
    scope = ("XM GOLD# demo-account quotes. Quoted spread is not the full execution cost: slippage, commission and swap "
             "are measured only on demo deals and do not transfer to a live account; a passing bound covers quoted "
             "spreads, not live fills. Forward 2025+ spreads were partly seen in E-014 (and the 1.21 factor was estimated "
             "on data including them), so evidence is development; A_cal and all quantile rules use no forward data. "
             "The exit bound does not cover positions held over a weekend (exits more than 180 minutes after the "
             "+60-minute target are excluded). No strategy edge is tested; H1 C, gemini.py and the Demo Runner are untouched.")
    return {"kind": "preregistration", "id": "E-015", "hypothesis": "H-023", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "k_cal and the factor baseline use the calibration window 2023-08-07..2024-12-31 only; "
                                    "all bounds use only the 90 days before each day; nothing is fitted on 2025+",
            "claim_scope": scope}


REVIEWS = [
    {"kind": "review", "experiment": "E-015", "reviewer": "critic_agent_7", "verdict": "revise", "findings": [
        "B1 high-cost miss diluted by routine rollover minutes: C95 passed while missing 100% of non-rollover news spikes",
        "B2 exit target split rollover by the entry hour; C95/D95 exit bounds keyed on the entry hour",
        "B3 exits crossing the daily break dropped (the most expensive reopen fills)",
        "B4 supported could unblock H-013 with no exit bound (shorts pay the exit spread)",
        "B5 coverage pooled over all minutes, not H-013's HH:00 decision times",
        "notes: A_* need factor stability; worst month/quarter and 3-month blocks; exit trials outside Sidak; spike window; "
        "quote_before after gaps"]},
    {"kind": "review", "experiment": "E-015", "reviewer": "critic_agent_7", "verdict": "approve", "findings": [
        "B1 accepted as a binding label: a high_cost_miss <= 0.5 gate cannot pass for a nominal-95% bound (it would need "
        ">= 2x level, breaking bound/cost <= 1.75); spikes now measured against the non-rollover level; cost capped by "
        "uncovered_cost_share; live-quote entry gate is observable",
        "B2 fixed (C95x/D95x and score keyed on the exit time); B3 fixed (reopen quote across the break, weekend excluded, "
        "flagged); B4 fixed (entry AND exit must pass); B5 fixed (HH:00 >= 0.90 gate)",
        "round-2 notes applied before locking: exit ratios use only exits completed before d (latent look-ahead with "
        "every-day sampling), exit stress fixed at x2, spikes_covered consequences written into the H-013 prerequisite, "
        "metrics/power_note updated, weekend holds out of scope"]},
]


def main():
    events = [H023, D021, e015(), *REVIEWS]
    if "--append" in sys.argv:
        rm.append(events)
        rm.render(rm.read())
        print(f"appended {len(events)} events")
        return
    current = rm.read()
    for e in events:
        rm.check(e, current)
        current = current + [{**e, "seq": len(current)}]
    print(f"dry run OK: {len(events)} events")


if __name__ == "__main__":
    main()
