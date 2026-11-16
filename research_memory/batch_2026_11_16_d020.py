"""Ledger batch: E-014 reading (O-020) and D-020 (H-009 refuted on its volatility clause; cost-input rule for H-013).

    python -B research_memory/batch_2026_11_16_d020.py           # dry run
    python -B research_memory/batch_2026_11_16_d020.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261116T153137Z_research_e014_exp_spread_cost_v1"

O020 = {
    "kind": "observation", "id": "O-020", "status": "verified",
    "statement": "E-014（2023-01 至 2026-11，47 個月）：(1) 點差異常（滾動 90 天同週內小時 z 前 10%，實際占 16.3%）之後 30 分鐘的"
                 "波動比 0.965（Sidak CI 0.88-1.05），排除換日時段 0.966，逐年 0.91-1.08，中位數比 1.016，中間價 0.965；控制前 30 分鐘"
                 "RV 後異常旗標係數 -0.005（CI -0.047~0.044），再控制前 4 小時 -0.013（描述）。異常時的點差平均 22.1 點、其他 18.1 點，"
                 "點差占典型 30 分鐘波動的中位數 6.8% 對 5.4%：異常點差只帶來直接成本，不預告波動。(2) 成本模型：週內小時中位數（2019-2022 擬合）"
                 "MAE 比全域中位數 0.945，但 2026 年 1.038，平均成本偏差 2023 +68%、2024 +33%、2025 +37%（點差水準逐年漂移：2023 平均 14.6、"
                 "2026 25.0 點）；每月以前 90 天重估的版本平均偏差每年都在 ±6.3% 內，但排除換日時段後 MAE 比 1.008、2024 年 1.005：成本主要由"
                 "近期水準決定，換日時段以外的週內小時結構樣本外不穩。(3) bar SPREAD 欄位是該分鐘 tick 點差的最小值（在點差有變動的 115,173 分鐘中"
                 "吻合 99.8%），tick 平均／bar 平均 1.21，以 bar 欄位估成本會低估約 21%。demo 帳戶 1,162 筆成交對成交前最後一筆 tick（中位延遲 "
                 "24 ms）幾乎沒有滑價（平均 -0.04 點，僅 0.17% 不利），成交時報價點差中位數 29、p90 33 點；這是 demo 成交，不代表實盤。"
                 "(4) 描述：2018-10 欄位缺失 11.9%（第一筆有效 2018-10-10 16:40），2019 年起幾乎無缺失；2020 年點差 p99 257 點；"
                 "時段中位數換日 24、亞洲 20、倫敦 19、紐約 18 點；超過週內小時 p99 的連續段落罕見（每 100 交易小時 0.29 段，中位 2 分鐘，最長 11）；"
                 "換日點差尖峰在伺服器 0／1 時，2021-2022 冬季在 23 時（季節性時區差異，原因未判定）；所有 bar 對齊整分",
    "evidence": f"{RUN}/report.md",
}

D020 = {
    "kind": "decision", "id": "D-020",
    "statement": "H-009 設為 refuted（範圍：XM GOLD# 2023-2026 以 M1 bar 點差欄位定義的同時段異常擴大，之後 30 分鐘的實現波動不高於其他時刻；"
                 "依 H-009 的推翻條件『實現波動比 < 1.2』成立，回放淨 R 條款未測，tick 層級滑價未測）。H-013 成本輸入規則：E-014 依預先登記判定"
                 "成本輸入未就緒（Q1 not_reliable），因此 H-013 不得使用週內小時成本模型；可用的是近 90 天的平均點差水準（各年偏差 ±6.3% 內）"
                 "乘以 tick／bar 係數 1.21，另加換日時段旗標，並以成本 ×1.5、×2 壓力測試；任何更精細的成本模型需另行預先登記。"
                 "依賴：H-013、H-007、H-012 以 E-014 為成本證據；H-013 data_quality 由 5 降為 4（成本輸入只能用保守水準）",
    "rationale": "研究結果自動進入排序；點差異常不預告波動，H-009 停止推進；H-013 的成本假設必須反映 bar 欄位低估與水準漂移",
    "set_status": {"H-009": "refuted"},
    "set_depends": {
        "H-013": [{"on": "E-014", "type": "evidence", "note": "成本輸入：近 90 天水準 × 1.21；週內小時結構樣本外不穩"}],
        "H-007": [{"on": "E-014", "type": "evidence", "note": "時段點差結構在換日以外不穩"}],
        "H-012": [{"on": "E-014", "type": "evidence", "note": "bar 點差欄位是分鐘最小值，需 × 1.21"}],
    },
    "set_priority_inputs": {"H-013": {"info_gain": 4, "novelty": 5, "falsifiability": 4, "data_quality": 4,
                                      "economic_value": 4, "simplicity": 3, "mining_risk": 3}},
}


def main():
    events = [O020, D020]
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
