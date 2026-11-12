"""Ledger batch: D-010 scoping and E-010 preregistration (H-014 volatility clause, GVZ information increment).

    python -B research_memory/batch_2026_11_12_e010.py           # dry run
    python -B research_memory/batch_2026_11_12_e010.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

D010 = {
    "kind": "decision", "id": "D-010",
    "statement": "H-014 先只檢驗波動子句，問題是『資訊增量』：同一種擬合、同一批資料列下，加入 GVZ 的模型是否優於只用價格歷史"
                 "（EWMA_094）的模型；不是單純追求比 EWMA 更低的誤差。分三個預先指定的問題：一般波動（h=5）、極端波動"
                 "（未來 5 日超過過去 90 百分位）、狀態轉變（未來 22 日波動升到過去 22 日的 1.5 倍以上），Sidak 校正 3 次。"
                 "溢酬預測報酬的子句延後。GVZ 時間規則：起點 t 只用美國日期 <= t-1 日的值（歷史發布時間無法驗證，"
                 "採保守延遲），另以再晚一筆的 lag 2 作為必要條件。實驗編號 E-010；E-009 保留給 H-018。"
                 "H-018 保留為下一個獨立方向：在固定交易訊號下，檢驗 EWMA_094 波動預測能否改善風險調整後報酬、最大回撤或尾部風險，"
                 "核心不是預測準確度；E-010 結果不改變 H-018 的波動輸入（仍為 EWMA_094），除非某個問題 supported 且經前瞻資料確認。"
                 "GVZ 與任何未驗證模型都不接入 Runner",
    "rationale": "把重新校準的效果和新資訊的效果分開；預先固定問題、期間、參數與狀態，避免事後挑選",
    "set_status": {"H-014": "testing"},
}


def prereg():
    module = "exp_gvz_increment_v1"
    files = [f"{module}.py", "exp_vol_forecast_v1.py", "research_data_gvz_v1.py", "research_data/gvz/GVZCLS.csv"]
    spec = {
        "estimation_window": ["2014-01-01", "2018-01-01"], "model_train_end": "2018-01-01",
        "windows": [["2018-01-01", "2024-01-01"], ["2024-01-01", "2099-01-01"]],
        "questions": {"general": "h=5 QLIKE", "extreme": "h=5 P(5-day mean RV > expanding q90) log loss",
                      "regime_up": "h=22 P(22-day mean RV >= 1.5 x trailing 22-day RV) log loss"},
        "gvz_timing": "latest GVZCLS dated <= t-1 calendar day at the close of server day t; stale > 5 days = missing; "
                      "lag 2 = one observation older",
        "min_years_better": 4, "materiality_pct": 2, "involves_trading": False, "trials_declared": 3,
        "not_applicable_reason": "no positions are taken: variance forecasts and event probabilities are scored "
                                 "against realised variance, so spread, swap and market direction cannot enter the "
                                 "losses; economic value is H-018",
        "controls": ["R: nested model without GVZ (general [1, log EWMA_094]; extreme [1, log EWMA_094, log q90]; "
                     "regime_up [1, log EWMA_094, log RV22]; all coefficients free), fit identically on identical rows; "
                     "A = R + log IV adds exactly one column",
                     "P: R plus one extra price variable (log RV22, log RV22, log RV5; one-parameter flexibility control)",
                     "A_lag2: GVZ one observation older (publication-time robustness)",
                     "levels, descriptive: EWMA_094, RW5, HIST, IV_SCALED"],
        "metrics": ["qlike", "log_loss", "brier", "dm_t", "dm_t_lag22", "sidak_p", "gain_pct", "gain_upper95_pct",
                    "years_better", "by_regime_ratio", "event_rate"],
        "power_note": "about 1500 evaluation days 2018-2023. general: overlapping 5-day targets, NW lag 5; a QLIKE "
                      "gain of roughly 4-5% or more is likely detectable at Sidak alpha 0.017 (rough guess from E-008, "
                      "not simulated). extreme: about 150 event days in clusters, regime_up: 22-day overlapping "
                      "targets (about 70 independent windows); both have low power, so small gains end as "
                      "underpowered rather than not_significant, which the 2% upper-bound rule makes explicit",
        "data": {"source": "FRED GVZCLS", "snapshot": "research_data/gvz/GVZCLS.csv",
                 "sha256": rm.file_sha256(rm.ROOT / "research_data/gvz/GVZCLS.csv"),
                 "provenance": "research_data/gvz/provenance.json"},
    }
    rule = ("Three primary tests (general h=5, extreme h=5, regime_up h=22), FROZEN coefficients fit on 2014-2017, "
            "evaluated 2018-2023; each compares A (with GVZ) to R (without) on the question's own loss, one-sided "
            "Diebold-Mariano with Newey-West lag 5, 5 and 22, p Sidak-adjusted for 3 tests (times earlier trials in "
            "the family). R is nested in A (A adds only log IV). Per question: underpowered if any of R/A/P/A_lag2 "
            "fails to converge within 100 Newton steps or (logistic questions) the estimation rows hold < 30 events; "
            "else refuted if A is significantly worse (adjusted p < 0.05). supported only if all "
            "hold: A/R loss ratio < 1 with adjusted p < 0.05; A_lag2 loss below R; A loss below P; A better in >= 4 "
            "of 6 calendar years. Otherwise not_significant if the one-sided 95% upper bound of A's loss reduction "
            "is < 2% of R's loss, else underpowered. Experiment outcome: supported if any question is supported "
            "(the claim is limited to that question), else refuted if any is refuted, else inconclusive. general h=1 "
            "and h=22, level controls, regime splits, lag-22 DM and 2024+ are descriptive and cannot change verdicts.")
    scope = ("Whether GVZ, available one day late, adds information about GOLD# realised variance (from XM H1 bars) "
             "beyond EWMA_094 under fixed 2014-2017 coefficients, separately for typical variance, extreme 5-day "
             "variance and upward regime shifts. Says nothing about economic value (H-018) or about the H-014 "
             "return clause. A supported question does not license other horizons, thresholds or regimes, and "
             "2018-2023 GOLD# prices were seen before, so evidence is development until confirmed on forward data. "
             "A refuted question means the frozen GVZ model did worse out of sample (e.g. the implied-realised premium "
             "drifted), not that GVZ carries no information; when questions disagree, all verdicts are reported "
             "side by side.")
    return {"kind": "preregistration", "id": "E-010", "hypothesis": "H-014", "role": "exploratory", "module": module,
            "code_files": {f: rm.file_sha256(rm.ROOT / f) for f in files}, "spec": spec,
            "evaluation_windows": spec["windows"], "decision_rule": rule, "author": "researcher",
            "training_window_note": "all coefficients (R, A, P, A_lag2, IV scale) are estimated only on origins from "
                                    "2014-01-01 whose whole target lies before 2018-01-01; the q90 threshold uses "
                                    "only 5-day windows that ended by the origin; GVZ enters with lag >= 1 day",
            "claim_scope": scope}


def main():
    events = [D010, prereg()]
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
