"""Ledger batch: D-015 (research plan: H-021 -> H-009 -> H-013, priority-engine upgrade) and E-013 preregistration
(H-021 historical data quality diagnostic).

    python -B research_memory/batch_2026_11_12_e013.py           # dry run
    python -B research_memory/batch_2026_11_12_e013.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

D015 = {
    "kind": "decision", "id": "D-015",
    "statement": "使用者決定的順序：H-021（歷史資料可信度）→ H-009（點差異常與可執行性）→ H-013（交易經濟優勢判斷，主要方向），"
                 "並改進 Priority Engine（前提有效性、資料可用性、假說依賴、已有反證與範圍、研究多樣性、資料品質對下游的影響；"
                 "失效假說自動轉入適當狀態並保留原因與證據）。E-013 是資料診斷：不訓練模型、不評估策略、"
                 "不得用來重新宣稱任何過去策略具有獨立樣本外優勢。外部獨立黃金資料（Stooq／Dukascopy／Yahoo）"
                 "在 2026-11-12 從本機無法取得，改以 GOLD# 的 M1／H1／D1 互相核對與 GOLD#／EURUSD／XAUEUR# 交叉匯率恆等式核對；"
                 "能只靠外部資料判定的部分歸為無法判定",
    "rationale": "後續所有研究都依賴歷史資料；先確認哪些缺陷已證實、哪些是真實市場差異、哪些無法判定",
    "set_status": {"H-021": "testing"},
}


def e013():
    module = "exp_data_quality_v1"
    files = [f"{module}.py", "exp_vol_forecast_v1.py"]
    spec = {
        "windows": [["2014-01-01", "2099-01-01"]], "periods": {"P0": ["2014-01-01", "2018-01-01"],
                                                               "P1": ["2018-01-01", "2024-01-01"],
                                                               "P2": ["2024-01-01", "2099-01-01"]},
        "involves_trading": False, "trials_declared": 1, "seed": 20261112, "bootstrap_draws": 2000,
        "min_m1_bars": 600, "m1_must_start_by": "2014-01-06", "ohlc_defect_rate": 0.001, "mismatch_defect_rate": 0.01,
        "vr_inflation_delta": 0.10, "vr_equivalence_band": 0.15, "vr_week_tolerance": 0.25,
        "not_applicable_reason": "a data diagnostic: no positions, no forecasts and no fitted model, so spread, swap "
                                 "and market direction cannot enter any statistic (spread is itself audited as data)",
        "controls": ["variance ratio across time scales (day and week sums of H1 returns vs squared H1 returns)",
                     "hour-of-day decomposition incl. hours present only before 2018",
                     "change-month detection (spread field, H1 bars/day, M1 zero-return share)",
                     "cross-rate identity GOLD# - EURUSD - XAUEUR# (from 2017-08-28; XM's own gold, not external)",
                     "M1/H1/D1 aggregation and RV_H1/RV_15m (low discrimination, descriptive)",
                     "P1 2018-2023 as the reference period"],
        "metrics": ["vr_day", "vr_week", "delta_vr_day_ci95", "delta_vr_week_ci95", "rv_level_ratio_vr_adjusted",
                    "hour_share", "change_months", "m1_zero_return_share", "spikes_per_1000_days",
                    "roll_noise_share_h1", "spread_zero_share", "ohlc_inconsistent", "aggregation_mismatch",
                    "h1_bars_per_day", "weekly_open_hour", "z_kurtosis", "atr14_pct", "cross_residual_var_ratio"],
        "power_note": "about 1000 P0 and 1500 P1 days in 48 and 72 month blocks. On synthetic data with 288 days per "
                      "period the 95% CI of the VR_day difference is about +-0.2 wide; with the real sample it should "
                      "be about +-0.1 to +-0.15 (heavier tails widen it). So only inflation that lowers VR_day by "
                      "about 0.15 or more is reliably detected, and 'not explained' needs the whole CI inside +-0.15; "
                      "otherwise the verdict is inconclusive. Aug-Dec 2017 gives only ~2500 common H1 bars",
        "external_sources_tried": {"stooq": "browser JavaScript challenge (not bypassed)", "dukascopy": "HTTP 429",
                                   "yahoo": "HTTP 429"},
    }
    rule = ("Primary: VR_day = sum over days of (sum of the day's H1 log returns)^2 / sum of squared H1 log returns "
            "(consecutive H1 bars, overnight included), per period; VR_week the same with weekly sums. delta = "
            "VR_day(P1) - VR_day(P0), 95% month-block bootstrap CI (2000 draws, months resampled within each period). "
            "supported (pre-2018 H1 returns show excess intraday reversal consistent with quote noise that inflated "
            "H1 RV; real intraday mean reversion cannot be excluded) only if all hold: delta >= 0.10, the CI lower "
            "bound > 0, delta_week = VR_week(P1) - VR_week(P0) > 0, the 95% CI of VR_day(P0) lies below 1, and the "
            "95% CI of VR_day(P1) contains 1 (so a post-2018 market change cannot be booked as a pre-2018 defect). "
            "not_explained_by_quote_noise (ledger outcome refuted: "
            "quote noise reverting within about a week does not explain the RV level change; a feed-wide difference "
            "is not excluded and it is not called a market difference) if the CI lies inside (-0.15, 0.15) and "
            "|delta_week| < 0.25. Otherwise inconclusive. Independently of the verdict, findings are classed by fixed "
            "rules: confirmed_defect for SPREAD = 0 on > 50% of a year's bars (a missing cost field, not a price "
            "defect), OHLC inconsistency > 0.1% of bars, or aggregation mismatch > 1% before 2018 and > 5x the "
            "2018-2023 rate; confirmed_collection_change if the median H1 bars per day before 2018 and in 2018-2023 "
            "share no value; the cross-symbol check is undetermined if any change month is on or before 2017-08; the "
            "external cross-check is always undetermined. M1 must start by 2014-01-06 or the run aborts. All other "
            "tables (Roll noise share, spikes, RV_H1/RV_15m, aggregation, cross residual) are descriptive.")
    scope = ("Quality of XM GOLD# MT5 history (M1/H1/D1) before 2018 relative to 2018+, judged by internal "
             "time-scale consistency and XM's own cross rates only. Not explained by quote noise does not mean a market "
             "difference: a feed-wide difference that is internally consistent needs an external source, and noise "
             "contributing less than about 15% of P0 RV is not excluded. A supported result cannot be told apart "
             "from real intraday mean reversion before 2018. No model is "
             "trained, no strategy is evaluated, and nothing here re-establishes any past out-of-sample edge. "
             "Excluding or re-weighting any period in later work needs its own preregistration; past results stay as "
             "recorded, and rerunning old strategies on other periods is not confirmatory evidence.")
    return {"kind": "preregistration", "id": "E-013", "hypothesis": "H-021", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "nothing is fitted; thresholds above are fixed before the run and the bootstrap "
                                    "uses only month resampling",
            "claim_scope": scope}


def main():
    events = [D015, e013()]
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
