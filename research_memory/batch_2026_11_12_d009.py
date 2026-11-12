"""Ledger batch: E-008 reading (O-013), D-009 next steps, H-019 (post-hoc, forward data only).

    python -B research_memory/batch_2026_11_12_d009.py           # dry run
    python -B research_memory/batch_2026_11_12_d009.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261112T020611Z_research_e008_exp_vol_forecast_v1"

O013 = {
    "kind": "observation", "id": "O-013", "status": "provisional",
    "statement": "E-008 失敗型態（事後描述，不是證據）：FROZEN h=1 HAR 係數 b0=5.2e-5、bd=0.028、bw=0.114、bm=0.559，"
                 "斜率和 0.70，預測被截距拉向 2014-2017 的波動水準；HAR 在低／中波動狀態較差（1.18／1.12），高波動較好（0.74），"
                 "週一最差（1.23）；未觸及下限（0 天）。每月重估的 RECURSIVE HAR 與 EWMA_FIT 打平（0.985，DM t -0.65），"
                 "且未調參的 EWMA_094 優於 2014-2017 擬合的 EWMA_FIT（λ=0.96）。在 MSE 下 HAR 並未較差（DM t -0.38）。"
                 "推測：失敗主因是凍結參數對波動水準變化不穩健，而非缺少多尺度結構；此推測是看過結果後提出的，不能用 2018 年後資料驗證",
    "evidence": f"{RUN}/report.md",
}

H019 = {
    "kind": "hypothesis", "id": "H-019", "status": "open", "family": "second_moment_forecast",
    "title": "對波動水準不敏感的 HAR（log-HAR 或無截距比例形式）凍結參數後不輸 EWMA",
    "mechanism": "E-008 的凍結 HAR 截距錨定舊波動水準；尺度不變的設定應移除這個失敗來源",
    "novelty": "由 E-008 失敗型態導出（O-013），屬事後假設",
    "information_needed": "GOLD# H1，僅 E-008 執行日（2026-11-12）之後的前瞻資料；2018-2026 已被看過",
    "falsified_if": "前瞻至少 250 個完整交易日，凍結 log-HAR 的 QLIKE 不低於 EWMA_094（DM 單尾 p >= 0.05）",
    "data_mining_risk": "high：設定是看過 E-008 後才選的；歷史資料只能當開發證據，判定必須用前瞻資料",
    "tags": {"information": ["price_ohlc"], "target": ["volatility"]},
    "priority_inputs": {"info_gain": 2, "novelty": 2, "falsifiability": 4, "data_quality": 5, "economic_value": 2,
                        "simplicity": 4, "mining_risk": 4},
}

D009 = {
    "kind": "decision", "id": "D-009",
    "statement": "E-008 依預先登記規則判為 refuted：凍結 2014-2017 參數的 HAR 在 2018-2023 h=1 QLIKE 顯著差於 EWMA_FIT"
                 "（比值 1.139，DM t 2.94，lag 22 t 2.26，6 年中僅 1 年較好）。H-011 預測子句設為 refuted，範圍限於"
                 "『此設定、凍結參數』；不代表 HAR 類模型無用，也不涵蓋條件報酬分布子句。依 D-008：E-009（H-018）"
                 "改用 EWMA_094 作波動輸入，問題是『任何波動預測是否改善決策』；H-014（GVZ）以 EWMA_094 為基準。"
                 "EWMA_094 是 E-008 主要窗口中最佳的 FROZEN 模型，且沒有任何擬合參數。不得改用 RECURSIVE HAR 或 2024+ 表現較好的模型。"
                 "H-019 只能用前瞻資料判定",
    "rationale": "照預先登記執行，不在看到結果後改寫判定；EWMA_094 原本就是預先宣告的對照，選它不引入新的擬合；"
                 "但它是依 2018-2023 結果選出，所以 E-009／H-014 在 2018-2023 的結果須標示此選擇",
    "set_status": {"H-011": "refuted"},
}


def main():
    events = [O013, H019, D009]
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
