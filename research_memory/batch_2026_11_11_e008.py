"""Ledger batch: E-008 (H-011 layer 1, forecast accuracy), H-018 (layer 2, decision value), D-008 scoping.

    python -B research_memory/batch_2026_11_11_e008.py           # dry run
    python -B research_memory/batch_2026_11_11_e008.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

H018 = {
    "kind": "hypothesis", "id": "H-018", "status": "open", "family": "second_moment_decision_value",
    "title": "波動預測能改善交易決策：倉位風險控制、實際回撤、辨識不值得交易的環境，且改善足以抵銷額外交易成本",
    "mechanism": "若未來波動可預測，固定風險預算下的部位可隨之調整，並在預期波動相對點差過小時不進場",
    "novelty": "第二層：把 E-008 的統計預測能力和經濟價值分開檢驗；預測更準不等於可交易",
    "information_needed": "E-008 的凍結預測（HAR 若成立，否則 EWMA）、既有策略的回放交易、XM 點差與 swap",
    "falsified_if": "2018-2023 以凍結預測做風險目標倉位與不交易濾網，在相同平均曝險下，扣除額外換手成本後 "
                    "Sharpe 不高於固定風險版本，且最大回撤降低 < 10%",
    "data_mining_risk": "medium：應用規則要在看 E-008 結果之前定義類別，參數在 E-009 預先登記時固定",
    "tags": {"information": ["price_ohlc"], "target": ["volatility", "tradeability"]},
    "priority_inputs": {"info_gain": 4, "novelty": 4, "falsifiability": 4, "data_quality": 5, "economic_value": 5,
                        "simplicity": 3, "mining_risk": 3},
}

D008 = {
    "kind": "decision", "id": "D-008",
    "statement": "H-011 分兩層：E-008 只檢驗統計預測能力（H-011 的預測子句）；條件報酬分布子句延後；交易決策價值另立 H-018，"
                 "於 E-008 有結果後才預先登記 E-009。E-008 結果如何對應下一步：supported → E-009 用 HAR 凍結預測；"
                 "refuted／not_significant → E-009 改用 EWMA，問題變成『任何波動預測是否改善決策』；"
                 "無論哪個結果，H-014（GVZ）以 E-008 中表現最好的『FROZEN 2014-2017 參數』模型為基準，不得改用看過 2024+ 後選的模型",
    "rationale": "避免把『預測誤差較小』和『可以賺錢』混在同一個實驗；兩層都要能各自被推翻",
    "set_status": {"H-011": "testing"},
}


def prereg():
    module = "exp_vol_forecast_v1"
    spec = {
        "estimation_window": ["2014-01-01", "2018-01-01"], "model_train_end": "2018-01-01",
        "windows": [["2018-01-01", "2024-01-01"], ["2024-01-01", "2099-01-01"]],
        "horizons": [1, 5, 22], "min_years_better": 4, "involves_trading": False, "trials_declared": 1,
        "not_applicable_reason": "no positions are taken: forecasts of variance are scored against realised "
                                 "variance, so spread, swap and market direction cannot enter the losses; economic "
                                 "value is H-018",
        "controls": ["RW (last h days)", "HIST (expanding mean)", "EWMA_094 (fixed)", "EWMA_FIT (lambda fit like HAR)"],
        "metrics": ["qlike", "mse", "rmse", "mae", "dm_t", "dm_t_lag22", "qlike_by_year", "qlike_by_regime",
                    "qlike_by_weekday", "qlike_parkinson_scaled", "har_at_floor"],
        "power_note": "about 1500 evaluation days 2018-2023; QLIKE differences are heavy-tailed and serially "
                      "correlated, so with a Newey-West DM test only QLIKE reductions of roughly 4-5% or more are "
                      "likely detectable (rough guess, not simulated); smaller real gains end as not_significant; "
                      "lag 5 may understate long-memory autocorrelation, hence the lag-22 sensitivity",
    }
    rule = ("Primary (single test): FROZEN parameters, h=1, 2018-2023, HAR vs EWMA_FIT on QLIKE. refuted only if "
            "HAR is significantly worse: one-sided Diebold-Mariano p < 0.05 in that direction (Newey-West lag 5). "
            "supported only if all hold: QLIKE ratio < 1 with one-sided DM p < 0.05 for HAR better; HAR QLIKE also "
            "below EWMA_094 and RW; HAR better in >= 4 of 6 calendar years; HAR better in >= 2 of 3 volatility "
            "regimes. Otherwise not_significant (ledger outcome inconclusive). h=5/22, RECURSIVE, MSE/RMSE/MAE, "
            "lag-22 DM, weekday table, scaled-Parkinson check and 2024+ are descriptive and cannot change the outcome.")
    scope = ("Statistical accuracy of daily realised-variance forecasts for GOLD# built from XM H1 bars, with "
             "parameters fixed on 2014-2017. Says nothing about economic value (H-018). A supported result names "
             "HAR as specified with FROZEN parameters; picking RECURSIVE, another horizon or a model that looks best "
             "in 2024+ is not independently validated. Does not test H-011's conditional-distribution clause.")
    return {"kind": "preregistration", "id": "E-008", "hypothesis": "H-011", "role": "exploratory", "module": module,
            "code_files": {f"{module}.py": rm.file_sha256(rm.ROOT / f"{module}.py")}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "HAR coefficients and EWMA_FIT lambda are estimated only on pairs whose whole "
                                    "target lies in 2014-01-01..2017-12-31 (FROZEN); RECURSIVE refits monthly on "
                                    "pairs whose target ended before each month's first origin",
            "claim_scope": scope}


def main():
    events = [H018, D008, prereg()]
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
