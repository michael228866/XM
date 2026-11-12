"""Ledger batch: E-010 reading (O-014) and D-011 next steps.

    python -B research_memory/batch_2026_11_12_d011.py           # dry run
    python -B research_memory/batch_2026_11_12_d011.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261112T130455Z_research_e010_exp_gvz_increment_v1"

O014 = {
    "kind": "observation", "id": "O-014", "status": "provisional",
    "statement": "E-008 與 E-010 的共同失敗型態（事後描述，不是證據）：2014-2017 擬合的水準對應在 2018-2023 都退化。"
                 "E-010 中連不含 GVZ 的 R（重新校準的 EWMA_094，斜率 0.69）都比原始 EWMA_094 差（general QLIKE 0.203 vs 0.182）；"
                 "GVZ 在估計期係數為正且大（log IV 0.47），樣本外卻在中／高波動狀態最差（A/R 1.26／1.82）；"
                 "IV_SCALED 的 RV/隱含變異數比例 k=1.62 在 2018-2023 明顯失準（QLIKE 0.374）。extreme 的事件率估計期 15%、"
                 "評估期 2.8%（擴張 q90 含 2013 年高波動），基準率漂移放大了 logistic 的失準。擬合皆收斂，事件數 127／206 足夠。"
                 "推測：GVZ 與 GOLD# 實現波動的水準關係（變異數風險溢酬）不穩定，凍結係數無法承接；這不排除 GVZ 的相對變化含有資訊",
    "evidence": f"{RUN}/report.md",
}

D011 = {
    "kind": "decision", "id": "D-011",
    "statement": "E-010 依預先登記規則三題皆 refuted（general A/R 1.125 DM t 3.09；extreme 1.559 t 5.40；regime_up 1.110 t 2.47；"
                 "lag 2 與對照 P 方向一致）。H-014 波動子句設為 refuted，範圍限於『GVZ 以 log 水準、lag 1、2014-2017 凍結係數加入"
                 "EWMA_094 模型』；依審查意見，這表示凍結的 GVZ 模型樣本外較差，不代表 GVZ 不含資訊；報酬子句未檢驗。"
                 "2024+ 的 general A/R 0.98、h22 0.92 只是描述，不得用來挽回結論。H-018 不變：下一個獨立研究方向，"
                 "以 EWMA_094（無擬合參數）作波動輸入，在固定交易訊號下檢驗風險調整後報酬、最大回撤與尾部風險，編號 E-009。"
                 "GVZ 的尺度不變形式（相對自身過去的變化）屬事後想法，若要研究須另立假設並只用前瞻資料判定；目前不登記",
    "rationale": "照預先登記執行；兩次失敗都指向凍結的水準校準，但這是看過結果後的歸納，不能在同一段資料上再驗證",
    "set_status": {"H-014": "refuted"},
}


def main():
    events = [O014, D011]
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
