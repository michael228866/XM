"""Ledger batch: D-019 (user-approved statuses H-017/H-018/H-020, E-001..E-007 dependency audit) and E-014
preregistration (H-009 spread data, cost model and spread anomalies).

    python -B research_memory/batch_2026_11_16_e014.py           # dry run
    python -B research_memory/batch_2026_11_16_e014.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402


def dep(on, kind, note=""):
    return {"on": on, "type": kind, **({"note": note} if note else {})}


H1_2018 = "data:gold_h1_2018"

D019 = {
    "kind": "decision", "id": "D-019",
    "statement": "核准（使用者 2026-11-16）：H-017 設為 premise_invalid，範圍僅限已確認的本 demo 帳戶條件（swap_mode=DISABLED，O-016），"
                 "不擴及其他 XM 帳戶；H-018、H-020 設為 needs_revalidation（依賴受影響資料，不是被推翻）。H-009 設為 testing（E-014）。"
                 "E-001～E-007 依賴稽核：E-001、E-005、E-006 的開發期 2018-2022/2023 以 H1 計算，含過渡年 2018，宣告依賴 "
                 "data:gold_h1_2018（E-001 另依賴 model:h1_c）；E-002、E-007 只用 D1，不受影響，宣告依賴 data:gold_d1_all；"
                 "E-003、E-004 已作廢不宣告。H-001、H-004 的結論分別以 E-001、E-005 為證據。歷史結論與觀察一律不改寫",
    "rationale": "使用者核准系統建議；補齊依賴圖讓受 2018 過渡年影響的證據被標註",
    "set_status": {"H-017": "premise_invalid", "H-018": "needs_revalidation", "H-020": "needs_revalidation",
                   "H-009": "testing"},
    "set_depends": {
        "data:gold_d1_all": [],
        "E-001": [dep("model:h1_c", "evidence"), dep(H1_2018, "data", "開發期 2018-2023 的 H1 進場與報酬")],
        "E-005": [dep(H1_2018, "data", "時段報酬以 H1 計算，開發期 2018-2022")],
        "E-006": [dep(H1_2018, "data", "盤中／持有比較以 H1 計算，開發期 2018-2022")],
        "E-002": [dep("data:gold_d1_all", "data", "週頻，D1 收盤")],
        "E-007": [dep("data:gold_d1_all", "data", "D1 波動與報酬")],
        "H-001": [dep("E-001", "evidence")], "H-004": [dep("E-005", "evidence")],
    },
}


def e014():
    module = "exp_spread_cost_v1"
    files = [f"{module}.py"]
    spec = {
        "spread_start": "2018-10-01", "train": ["2019-01-01", "2023-01-01"], "test": ["2023-01-01", "2099-01-01"],
        "windows": [["2023-01-01", "2099-01-01"]], "tick_start": "2023-08-07", "tick_every_nth_day": 10,
        "bootstrap_draws": 2000, "seed": 20261116, "min_cell_bars": 100, "min_cell_bars_adaptive": 100,
        "adaptive_days": 90, "mae_ratio_max": 0.95, "mean_bias_max": 0.10, "horizon_minutes": 30,
        "max_span_minutes": 40, "anomaly_quantile": 0.90, "vol_ratio_threshold": 1.2, "min_anomalies": 2000,
        "min_months": 24, "min_fills": 100, "trials_declared": 4,
        "year_share_min": 0.75, "anomaly_share_band": [0.05, 0.20], "involves_trading": False,
        "not_applicable_reason": "no positions are simulated: Q1 scores spread forecasts, Q2 compares realized variance "
                                 "after spread anomalies, Q3 compares quotes and demo fills; swap and direction cannot "
                                 "enter any statistic (the H-009 replay clause is out of scope)",
        "data_windows": [
            {"window": ["2018-10-01", "2099-01-01"], "uses": ["spread_field"],
             "purpose": "descriptive spread tables only (no prices): missing share, sessions, percentiles, spike hour"},
            {"window": ["2019-01-01", "2023-01-01"], "uses": ["spread_field", "h1_or_finer_price"],
             "purpose": "Q1 train: hour-of-week spread medians/means (static cost model)"},
            {"window": ["2023-01-01", "2099-01-01"], "uses": ["spread_field", "h1_or_finer_price"],
             "purpose": "test: Q1 and Q2 decisions; Q2 norms and the adaptive model use the 90 days before each test month (from 2022-10)"}],
        "controls": ["Q1: train global median C0 for the static hour-of-week model C1",
                     "Q1: the same 90-day window's global median C0A for the monthly re-estimated model C2",
                     "Q1: MAE ratios excluding server hours 23/0/1 and per test year",
                     "Q2: trailing 90-day hour-of-week norms for spread z and volatility; within-month ratios",
                     "Q2: past 30-minute RV with hour-of-week x month demeaning (increment, decisive); past 4-hour RV "
                     "added as a descriptive control",
                     "Q2: rollover hours excluded, per-year ratios, anomaly share band, median-based and mid-price ratios",
                     "Q3: tick spreads on sampled days for the bar field; tick quote within 60 s before each demo fill"],
        "metrics": ["mae_ratio_C1", "mae_ratio_C2", "mae_ratio_C1_excl_rollover", "mae_ratio_C2_excl_rollover",
                    "mean_bias_by_year", "vol_ratio", "vol_ratio_ci", "vol_ratio_excl_rollover", "vol_ratio_by_year",
                    "vol_ratio_median_based", "vol_ratio_mid_price", "increment_log_rv", "increment_ci",
                    "increment_with_4h_control_descriptive", "anomaly_share_test", "extreme_runs",
                    "missing_share_by_year", "session_percentiles", "match_share_varied", "fill_adverse_points",
                    "quote_lag_ms"],
        "power_note": "test 2023-01..2026-11 is about 47 months and ~1.3M M1 minutes, so ~130k top-decile minutes; the "
                      "30-minute windows overlap, so month blocks carry the inference. A ratio near 1.2 is resolved only "
                      "if month-level ratios are stable; synthetic data with 6 test months gives a CI of about +-0.01-0.02. "
                      "Q2 needs >= 2000 anomalous minutes in >= 24 months; the month-block CI is somewhat narrow with few months "
                      "(critic note 3), mitigated by 47 months and the 1.2 threshold. Tick days: "
                      "every 10th trading day from 2023-08-07 (~80 days). Demo fills: ~1000 deals since 2023-08",
        "feasibility_probe": "before preregistration only formats were read: point 0.01, digits 2, spread_float, tick "
                             "history from 2023-08-04, M1 SPREAD 0 on 2018-10-01, 1178 deals in 500 days; no spread "
                             "statistic, anomaly or volatility relation was computed",
    }
    rule = ("Sidak over trials_declared=4 confidence intervals that enter decisions (C1, C2, volatility ratio, "
            "increment) times the family count passed by the runner: CI level 0.95^(1/k), month-block bootstrap "
            "(2000 draws). Q2 (the H-009 volatility clause, its own threshold 1.2 on volatility = sqrt of realized "
            "variance): for each test month, using only the 90 days ending 40 minutes before the month, z = (spread - "
            "hour-of-week mean) / max(hour-of-week sd, 1 point) for cells with >= 100 bars; anomaly = z above that "
            "window's 90th percentile (strict). Normalized volatility = sqrt(next-30-minute RV) / that window's "
            "hour-of-week median sqrt(RV). vol_ratio = anomaly-weighted geometric mean over months of the within-month "
            "ratio of mean normalized volatility after anomalous vs other minutes. Supported as "
            "vol_clause_not_refuted (ledger outcome supported, meaning only that the volatility clause survived; the "
            "replay clause is untested) only if all hold: vol_ratio >= 1.2, CI lower bound > 1, vol_ratio excluding "
            "server hours 23/0/1 >= 1.2, at least 75% of test years >= 1.2, and the test anomaly share within "
            "[5%, 20%]; labelled vol_clause_not_refuted_but_redundant if the OLS coefficient of the anomaly flag on "
            "log next RV, controlling log past-30-minute RV, demeaned within hour-of-week x month, flat windows "
            "dropped, has a CI lower bound <= 0. refuted if the CI upper bound < 1.2. inconclusive_sample if < 2000 "
            "anomalous test minutes or < 24 months with both groups; otherwise inconclusive. Q1 (cost model): "
            "static_generalizes if the hour-of-week median C1 (train 2019-2022) has test MAE / train-global-median "
            "MAE <= 0.95 with CI upper < 1, below 1 excluding rollover hours and in every test year, and its "
            "hour-of-week mean has |mean cost bias| <= 10% in every test year; adaptive_only if only C2 (monthly, "
            "previous 90 days) passes the same rules against the same window's global median; else not_reliable. "
            "Q3: the bar SPREAD field is labelled min/max/first/last/mean of the minute's tick spreads if, on minutes "
            "whose tick spread changed, one candidate matches >= 90% and leads the next by >= 0.2; else undetermined. "
            "H-013 cost input ready only if Q1 is not not_reliable, Q3 is determined and >= 100 demo fills were priced "
            "against the last tick within 60 s before the fill; if the field is not the minute mean, H-013 must scale "
            "it by the reported tick-mean / bar-mean factor. Spread 0 is missing in every statistic, never zero cost. "
            "Robustness reported, not decisive: median-based ratio, mid-price RV, increment with a past 4-hour RV "
            "control, quote lag by fill reason. A not-redundant increment does not show information beyond a "
            "multi-day volatility state.")
    scope = ("XM GOLD# MT5 demo-account quotes. The bar SPREAD field and tick spreads are quoted spreads; demo fills "
             "priced against the tick quote are demo execution, not live. Q2 tests only whether realized variance "
             "rises after spread anomalies; it does not test filtered entries' net R (the H-009 replay clause), "
             "slippage at scale, or any trading edge, and a correlation here is not a trading advantage. Test 2023+ "
             "spreads were partly seen in E-013 (spread zero share by year), so evidence is development. No model "
             "is trained beyond hour-of-week medians; H1 C, gemini.py and the Demo Runner are untouched.")
    return {"kind": "preregistration", "id": "E-014", "hypothesis": "H-009", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "Q1 static hour-of-week statistics use 2019-01-01 to 2022-12-31 only; Q2 z, "
                                    "threshold and volatility norms and the adaptive cost model use only the 90 days "
                                    "ending 40 minutes before each test month",
            "claim_scope": scope}


def main():
    events = [D019, e014()]
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
