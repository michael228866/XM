"""Ledger batch: E-013 reading (O-018) and D-016 (data-usage rule and affected research).

    python -B research_memory/batch_2026_11_12_d016.py           # dry run
    python -B research_memory/batch_2026_11_12_d016.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261112T154049Z_research_e013_exp_data_quality_v1"

O018 = {
    "kind": "observation", "id": "O-018", "status": "verified",
    "statement": "E-013（依預先登記判為 supported）：2014-2017 的 VR_day 0.434（CI 0.39-0.49），2018-2023 為 0.932（CI 0.85-1.02），"
                 "差 0.498（CI 0.41-0.60），週頻差 0.458。逐年 VR_day：2014 0.67、2015 0.45、2016 0.35、2017 0.36、2018 0.69（過渡）、"
                 "2019-2023 0.88-1.02、2024-2026 1.02-1.16。2018 年前伺服器 0 時與 1 時兩根 H1 合占全部 H1 報酬平方的 58%"
                 "（各約 29%），2018 年後約 9%；0 時那根只在 2018 年前存在（每日 24 根→23 根，變更月 2018-04）；"
                 "0 時與 1 時份額相等，呈現換日時段報價先跳開再回歸的型態。RV_H1／Parkinson 2014-2017 為 2.30，之後約 1.0。"
                 "經變異比調整後，2018-2023 對 2014-2017 的 RV 水準比由 0.465 變為 0.999：所謂 2018 年後波動減半幾乎全部來自這個型態。"
                 "SPREAD 欄位在 2013-2018 有一半以上為 0（變更月 2018-10），屬成本欄位缺失。XM 自家交叉匯率殘差 2017 下半年 1.49，"
                 "2018 與 2020 約 0.55，2023 年後約 0.002，品質逐年改善。按預先登記的措辭，這是『與報價雜訊相符的日內反轉過多』，"
                 "形式上不能排除真實的日內均值回歸；但集中在換日一小時並在下一小時回歸、規模達日變異的一半以上，解讀上指向換日時段的報價瑕疵。"
                 "外部資料源無法取得，資料源層級差異未判定",
    "evidence": f"{RUN}/report.md",
}

D016 = {
    "kind": "decision", "id": "D-016",
    "statement": "H-021 設為 supported（範圍：2018 年前 H1 層級的價格統計受換日時段型態嚴重放大）。資料使用規則：任何以 H1（或更細）"
                 "價格統計擬合、校準或估計成本的研究，預設只用 2019-01-01 之後的資料；2018 為過渡年；2019 之前的期間只能在"
                 "預先登記中說明理由並把它當成獨立的資料品質條件。點差研究只能用 2018-10 之後（之前欄位缺失）。"
                 "受影響的既有結論（保留原紀錄，不改寫，只註記）：E-008／E-010 的凍結模型以 2014-2017 擬合，主要學到被放大的波動水準；"
                 "O-013 至 O-015 中『固定參數退化／目標分布改變／校準漂移』大部分應歸因於資料而非市場；H-019／H-022 的前提需重新審視；"
                 "H1 C 的訓練期 2014-2022 包含受影響期間（其 2024+ 評估不因此失效，但模型可能學到換日瑕疵），Demo Runner 不更動，"
                 "是否重新訓練由使用者決定並另行預先登記。不得以 E-013 重新宣稱任何過去策略的樣本外優勢",
    "rationale": "後續研究的資料基礎必須先可靠；把資料缺陷、收集方式改變與未判定部分分開記錄",
    "set_status": {"H-021": "supported"},
}


def main():
    events = [O018, D016]
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
