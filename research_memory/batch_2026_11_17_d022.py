"""Ledger batch: E-015 reading (O-021) and D-022 (binding cost rules for H-013; H-023 status left for user approval).

    python -B research_memory/batch_2026_11_17_d022.py           # dry run
    python -B research_memory/batch_2026_11_17_d022.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

RUN = "training_runs/20261117T021737Z_research_e015_exp_cost_bound_v1"

O021 = {
    "kind": "observation", "id": "O-021", "status": "verified",
    "statement": "E-015（前向 2025-01 至 2026-11，242 個抽樣交易日，33 萬分鐘，23 個月）：(1) 進場上界 C95（近 90 天第一筆報價點差的"
                 "95 分位，換日與其他時段分開）覆蓋率 0.957（Sidak CI 0.923-0.979；3 個月區塊 0.917-0.979），2025 0.966、2026 0.947，"
                 "換日 0.956、整點 0.932，未覆蓋成本占比 2.75%，上界／實際 1.27，通過；但最差月份 2026-03 只有 0.763、最差季 2026Q1 0.870。"
                 "D95（水準調整）也通過（覆蓋 0.959，最差月 0.832）。B95 不分時段時換日只覆蓋 0.736，未通過。(2) 出場（+60 分鐘）上界 E95x"
                 "（進場前最後報價 × 近 90 天出場／報價比值的 95 分位）覆蓋 0.955（CI 0.946-0.963），最差月 0.922，跨過每日休市的出場 0.926，"
                 "未覆蓋成本 1.52%，上界／實際 1.43，通過；C95x、D95x 未覆蓋成本 3.9% 超過 3% 門檻。(3) 平均數規則不是上界：近 90 天平均 × 1.21"
                 "覆蓋只有 0.683（換日 0.434），×1.5 為 0.973 但換日 0.798，×2 為 0.984 但上界／實際 1.98 過於保守。(4) 非換日時段超過當日水準 2 倍的"
                 "尖峰（1,819 分鐘）進場上界全部漏判，出場上界漏判 66%：spikes_covered 皆為 false。(5) tick 平均／bar 欄位比值：校準期 1.224、"
                 "前向 1.193，逐年 2023 1.253、2024 1.213、2025 1.211、2026 1.178（逐年下降但在 ±10% 內），換日時段 1.319；依規則判定穩定，"
                 "但趨勢向下。換日尖峰小時 95 個月中 98.9% 落在伺服器 23／0／1 時（例外 2025-07 為 2 時），固定旗標成立。(6) demo 581 筆來回"
                 "損益 100% 等於成交價差（點差已含在成交價中，另扣點差會重複計算），佣金、手續費、隔夜利息皆為 0；1,162 筆成交對報價的不利滑價"
                 "平均 -0.04 點，成交時報價點差中位數 29、p95 33 點；gemini 成交所在分鐘的進場覆蓋率只有 0.910（低於整體），出場 0.940。"
                 "demo 成交不代表實盤；2025+ 曾在 E-014 被看過，證據等級 development",
    "evidence": f"{RUN}/report.md",
}

D022 = {
    "kind": "decision", "id": "D-022",
    "statement": "E-015 依預先登記判定 supported（推薦進場上界 C95、出場上界 E95x）。H-023 的狀態變更屬於研究結論，留待使用者核准（系統建議 "
                 "supported），本決策不設定。H-013 的 cost_upper_bound 前置條件依 D-021 的 met_by 規則機械性成立；H-013 仍因核准層級 "
                 "design_only、新版預先登記未審查、交易樣本不足而不可正式執行，cost_input_unreliable 條件維持開放，待使用者核准 E-015 結論後"
                 "再以附證據的決策關閉。H-013 成本規則（綁定，依 E-015 規則）：進場成本 = C95；因 spikes_covered=false，回測與實盤都必須跳過"
                 "即時報價高於 C95 的進場；出場成本 = E95x × 2（出場尖峰未覆蓋，2 為未驗證的壓力值）；不持倉過週末；一筆來回只計一次點差。"
                 "限制：C95 在 2026-03 單月覆蓋只有 0.76，gemini 成交分鐘的覆蓋 0.91，代表成本上界在部分月份與實際交易時刻偏低，H-013 必須"
                 "另報 ×1.5、×2 壓力結果；1.21 係數逐年下降（2026 1.178），不得當作固定常數延用到未來，A 規則不得作為上界",
    "rationale": "研究結果自動進入排序與前置條件；研究結論與正式核准分開",
    "set_depends": {"H-013": [{"on": "E-014", "type": "evidence", "note": "成本水準漂移、bar 欄位為分鐘最小值"},
                              {"on": "H-023", "type": "evidence", "note": "成本上界"},
                              {"on": "E-015", "type": "evidence", "note": "進場 C95、出場 E95x × 2、即時報價閘門"}],
                    "H-023": [{"on": "E-014", "type": "evidence", "note": "bar 欄位是分鐘最小值；點差水準逐年漂移"}]},
}


def main():
    events = [O021, D022]
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
