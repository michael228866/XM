"""Ledger batch: E-012 reading (O-017) and D-014 next steps.

    python -B research_memory/batch_2026_11_12_d014.py           # dry run
    python -B research_memory/batch_2026_11_12_d014.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261112T142415Z_research_e012_exp_vol_sizing_v1"

O017 = {
    "kind": "observation", "id": "O-017", "status": "provisional",
    "statement": "E-012（2024+，274 筆 H1 C 交易，33 個有交易的月份）：EWMA_RISK Sharpe 1.291 vs FIXED_RISK 1.087，"
                 "差 +0.204，95% 月區塊 CI [-0.29, 0.62]，判定 underpowered。方向一致但都不顯著：點差×2 +0.202、報價 swap×1.5 +0.207、"
                 "三個波動狀態都略高、2024／2025 較好但 2026 較差；對 FIXED 與市場的 alpha 0.6 bp／日（NW t 1.41）。"
                 "風險面沒有改善：同波動下最大回撤較深（-5.40% vs -4.88%），CVaR5 略好（-91.0 vs -93.6 bp）；"
                 "原始回撤較淺（-4.14% vs -4.88%）只是因為 2024+ 波動高、平均部位較小（0.57 vs 0.75），正是不能誤認的槓桿效果。"
                 "損益更集中（前 10% 交易佔 23.4% vs 14.5%，月份 Herfindahl 0.083 vs 0.062）。進場小時只解釋權重比的 7%。"
                 "描述性：純目標波動對固定風險 +0.03、對固定名目 +0.11，CI 都跨 0；ATR 固定風險在 2018-2023 樣本內明顯優於固定名目"
                 "（2.63 vs 1.88），2024+ 差距小（1.09 vs 1.01）",
    "evidence": f"{RUN}/report.md",
}

D014 = {
    "kind": "decision", "id": "D-014",
    "statement": "H-018 設為 underpowered：E-012 既不能排除 0.2 Sharpe 量級的改善，也不能排除同量級的惡化；風險面"
                 "（同波動回撤、集中度）沒有改善跡象。以 H1 C 每年約 100 筆的頻率，要把 CI 半寬從約 0.45 縮到 0.2 需要約 5 倍資料"
                 "（十年以上），所以 H-018 以 H1 C 繼續累積的資訊增益很低，調降優先序；不更動任何 Runner。"
                 "依使用者指示，部位管理沒有可證實的額外價值，研究轉向不同經濟機制：優先順序依 Priority Engine，"
                 "其中 H-008 必須先以帳戶實際 swap（O-016）重新設計。",
    "rationale": "照預先登記執行；underpowered 不是推翻，但在可預見的資料量下也無法判定，繼續投入的價值低",
    "set_status": {"H-018": "underpowered"},
    "set_priority_inputs": {"H-018": {"info_gain": 2, "novelty": 4, "falsifiability": 2, "data_quality": 5,
                                      "economic_value": 4, "simplicity": 3, "mining_risk": 3}},
}


def main():
    events = [O017, D014]
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
