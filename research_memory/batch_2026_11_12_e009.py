"""Ledger batch: H-020, D-012 and preregistrations E-009 (H-018 sizing value) and E-011 (H-020 drift diagnostic).

    python -B research_memory/batch_2026_11_12_e009.py           # dry run
    python -B research_memory/batch_2026_11_12_e009.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

H020 = {
    "kind": "hypothesis", "id": "H-020", "status": "open", "family": "fixed_parameter_drift",
    "title": "E-008／E-010 固定參數模型的退化可分解為可量測的來源（目標分布、共變量、概念、校準、資料品質），"
             "且主要來源可在前瞻資料上單獨檢驗",
    "mechanism": "2014-2017 擬合的水準對應在 2018 年後失準；可能是波動水準與分布改變、輸入落在訓練範圍外、"
                 "條件關係改變、單純的水準偏差，或 2018 年前 MT5 資料品質不同（SPREAD=0）",
    "novelty": "把兩次失敗變成可檢驗的診斷，而不是再換一個模型",
    "information_needed": "GOLD# H1（含原始 spread、tick volume）、GVZ 快照；後續檢驗只用 2026-11-12 之後的資料",
    "falsified_if": "描述性診斷本身不判定；後續每個解釋各自預先登記前瞻檢驗",
    "data_mining_risk": "high：全部使用已看過的資料；只能產生假說與觀察，不能作為成功證據",
    "tags": {"information": ["price_ohlc", "implied_vol"], "target": ["volatility"]},
    "priority_inputs": {"info_gain": 3, "novelty": 3, "falsifiability": 2, "data_quality": 4, "economic_value": 2,
                        "simplicity": 4, "mining_risk": 4},
}

D012 = {
    "kind": "decision", "id": "D-012",
    "statement": "暫停尋找新的波動預測模型。E-009（H-018）：固定 H1 C 凍結訊號與出場（TP 3／SL 1.5 ATR／48h，一次一筆），"
                 "只改變進場時固定的部位大小。主要比較為巢狀設計：EWMA_RISK = 固定風險倉位 × (σ_ref/σ_EWMA) vs 固定風險"
                 "（固定風險本身已以 H1 ATR 調整；審查發現直接以 EWMA 取代 ATR 主要只是按進場小時重新加權，"
                 "所以改成保留同一個 ATR 停損、只加入日線 EWMA 縮放）。純目標波動（EWMA_TARGET）與固定名目"
                 "（CONSTANT_NOTIONAL，runner 固定手數的對應）只作描述，各自附信賴區間。不測不交易濾網，因為它會改變交易樣本。判定只用 2024+（H1 C 在 2014-2022 擬合、"
                 "2023 選門檻）；2018-2023 為樣本內，只作描述。H-018 原 falsified_if 的點估計寫法由 E-009 預先登記的"
                 "檢定式規則取代。E-011（H-020）為獨立的描述性診斷，不判定任何假設，不混入 E-009。"
                 "若 E-009 顯示部位管理沒有額外價值，轉向不同經濟機制的問題（H-008／H-013／H-009）",
    "rationale": "訊號固定才能把部位管理的價值和方向預測分開；同一套交易、同一成本、同一槓桿上限，並以事前預算對齊與事後波動對齊兩種尺度報告",
    "set_status": {"H-018": "testing"},
}


def e009():
    module = "exp_vol_sizing_v1"
    files = [f"{module}.py", "exp_vol_forecast_v1.py", "exp_vol_managed_v2.py", "strategy_lab_v1.py", "h1_runner_v1.py"]
    spec = {
        "windows": [["2024-01-01", "2099-01-01"]], "model_train_end": "2024-01-01",
        "labelled_windows": {"in_sample_2018_2023": ["2018-01-01", "2024-01-01"], "oos_2024_on": ["2024-01-01", "2099-01-01"]},
        "calibration_window": ["2018-01-01", "2024-01-01"],
        "tp": 3.0, "sl": 1.5, "hold": 48, "risk_per_trade": 0.005, "leverage_cap": 5.0,
        "seed": 20261112, "bootstrap_draws": 2000, "min_trades": 150, "min_months": 24, "min_regime_days": 60,
        "materiality_sharpe": 0.2, "materiality_mdd_cut": 0.10, "trials_declared": 1,
        "controls": ["FIXED_RISK (baseline, b = 0.5% of equity at the stop)",
                     "EWMA_RISK = FIXED_RISK x sigma_ref / sigma_EWMA (primary arm, nested on the same ATR stop)",
                     "EWMA_TARGET pure target vol and CONSTANT_NOTIONAL (no vol information), descriptive with CIs",
                     "ex-ante R^2 of log weight ratios on entry-hour dummies (what each comparison re-weights)",
                     "identical trades, exits, costs and leverage cap 5x in every arm",
                     "ex-ante budget match: sigma_ref / sigma_target set on 2018-2023 entry information only",
                     "ex-post vol-matched drawdown and CVaR"],
        "directional_benchmark": "both arms are long the same trades, so drift is shared; EWMA_RISK daily returns are "
                                 "regressed on FIXED_RISK and on GOLD# daily close-to-close return (alpha, NW lag 5)",
        "cost_multipliers": [1.0, 2.0],
        "cost_note": "spread = lab bar spread floored at 0.008% of price, paid at entry (x2 scenario); swap = today's "
                     "swap_long as a fraction of price per night, charged on notional at every rollover crossed, "
                     "Wednesday triple (x1.5 scenario); larger notional pays proportionally more",
        "metrics": ["net_mean", "sharpe", "calmar", "max_dd", "vol_matched_max_dd", "vol_matched_cvar5", "delta_sharpe_ci",
                    "alpha_nw_t", "by_regime_sharpe", "by_year_sharpe", "return_per_unit_notional", "cost",
                    "top10pct_trades_share", "month_pnl_herfindahl", "leverage_cap_binding", "hour_r2_log_weight_ratio"],
        "power_note": "2024-01 to 2026-11: about 270 H1 C trades and 34 months. The two arms trade the same positions, "
                      "so their daily returns are highly correlated; even so a 34-month block bootstrap gives a delta-"
                      "Sharpe 95% CI roughly +-0.3 wide (synthetic check with 486 trades/36 months: +-0.22). Only "
                      "sizing gains of about 0.3 Sharpe or more are detectable; the likely verdict is underpowered, "
                      "which needs forward H1 C trades to resolve. The percentile bootstrap over ~34 month blocks is "
                      "slightly narrow (critic simulation: ~5% per tail instead of 2.5%); the CI width also grows with "
                      "how much the arms' weights differ. min_months counts months with at least one trade",
    }
    rule = ("Primary: window 2024+, base costs, delta Sharpe = Sharpe(EWMA_RISK) - Sharpe(FIXED_RISK) of daily net "
            "returns, 95% month-block bootstrap CI (2000 draws), a single test. inconclusive_sample if < 150 trades or "
            "< 24 months with a trade. refuted if the CI upper bound < 0. If the CI lower bound > 0: supported only if "
            "all hold - delta Sharpe > 0 under spread x2 and under swap x1.5; vol-matched max drawdown and vol-matched "
            "daily CVaR 5% of EWMA_RISK no worse than FIXED_RISK; EWMA_RISK Sharpe >= FIXED_RISK in all but at most "
            "one volatility regime with >= 60 days (if only one such regime, in that one; if none, supported is "
            "impossible; regime label from the previous day); and in >= 2 of the calendar years 2024-2026 - otherwise significant_not_robust. If the CI "
            "contains 0: not_significant if CI upper < 0.2 and the vol-matched drawdown cut < 10%, else underpowered. "
            "Ledger outcome: supported / refuted / inconclusive (all other verdicts). Alpha t, raw metrics, "
            "EWMA_TARGET, CONSTANT_NOTIONAL, Calmar, cost, concentration and the 2018-2023 window are descriptive.")
    scope = ("Value of scaling 1/ATR fixed-risk positions by daily EWMA_094 volatility (beyond the ATR) for the frozen "
             "H1 C long trades on GOLD# at XM costs, 2024+. Not a test of entry timing, of no-trade filters, of other "
             "signals or of other volatility models. 2024+ is not a clean holdout: H1 C was chosen among candidates "
             "after its 2024+ test was seen, and E-007 (H-005) already studied inverse-vol sizing of always-long, so "
             "trials_declared = 1 is optimistic (development evidence). Gap fills at the stop understate tails. "
             "A supported result does not change any runner without forward confirmation and the user's decision.")
    return {"kind": "preregistration", "id": "E-009", "hypothesis": "H-018", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "H1 C model fit 2014-2022 and threshold chosen on 2023 (frozen, not refit here); "
                                    "sizing fits nothing except sigma_ref, sigma_target and the constant notional, medians "
                                    "of entry-time quantities on 2018-2023 signal bars, no outcomes",
            "claim_scope": scope}


def e011():
    module = "exp_fixed_param_drift_v1"
    files = [f"{module}.py", "exp_gvz_increment_v1.py", "exp_vol_forecast_v1.py", "research_data_gvz_v1.py",
             "research_data/gvz/GVZCLS.csv"]
    spec = {"windows": [["2018-01-01", "2099-01-01"]], "periods": {"P0": ["2014-01-01", "2018-01-01"],
            "P1": ["2018-01-01", "2024-01-01"], "P2": ["2024-01-01", "2099-01-01"]},
            "involves_trading": False, "trials_declared": 1, "descriptive_only": True,
            "not_applicable_reason": "no positions and no decision: a description of forecast drift",
            "controls": ["frozen forecasts rebuilt exactly as E-008/E-010", "per-year optimal scalar (calibration-only)",
                         "in-year refit (descriptive floor)"],
            "metrics": ["ks_vs_P0", "outside_P0_1_99", "mean_y_over_f", "L_frozen", "L_scaled", "L_refit_in_year",
                        "mincer_zarnowitz_log", "data_quality_by_year"],
            "power_note": "descriptive; about 250 days per year and 3 periods; no test is run, so no power claim"}
    rule = ("Descriptive only: 0 hypothesis tests; outcome is always inconclusive whatever the numbers (no threshold "
            "decides anything). Interpretations are recorded as provisional observations; each candidate "
            "explanation needs its own preregistered test on data after 2026-11-12.")
    scope = ("Describes why the E-008/E-010 frozen forecasts degraded after 2017 using already-seen data; refits are "
             "in-sample floors. It cannot validate any model or explanation and is not part of E-009.")
    return {"kind": "preregistration", "id": "E-011", "hypothesis": "H-020", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "frozen forecasts use the E-008/E-010 2014-2017 estimation exactly; in-year refits "
                                    "and scalars use the evaluated year itself and are labelled descriptive",
            "claim_scope": scope}


def main():
    events = [H020, D012, e009(), e011()]
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
