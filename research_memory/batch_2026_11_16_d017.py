"""Ledger batch: O-019 (training-data audit of H1 C and gemini), D-017 (dependency graph and upstream flags for the
Priority Engine) and D-018 (statuses the user approved on 2026-11-16).

    python -B research_memory/batch_2026_11_16_d017.py           # dry run
    python -B research_memory/batch_2026_11_16_d017.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

O019 = {
    "kind": "observation", "id": "O-019", "status": "verified",
    "statement": "訓練資料稽核（讀程式與紀錄，未重跑）：H1 C（frozen/h1_C_long_v1）以 GOLD# H1 擬合 2014-01-01 至 2022-12-31"
                 "（依持有期 purge），2023 挑門檻，2024-01-01 起測試。16 個特徵中 ATR_PCT、BB_WIDTH、VOLA_RATIO、BODY_PCT、"
                 "ROC_5、HOUR_SIN/COS 直接取自 H1 K 棒，標籤是以 H1 ATR 為單位（TP 3.0／SL 1.5）在 H1 高低價上的先觸及，"
                 "兩者都暴露在 E-013 的換日時段型態（2014-2017 的 0、1 時 K 棒）之下；2018 年前 SPREAD=0，因此以價格的 0.008% "
                 "作成本下限。結論：H1 C 的擬合期受影響（約 5 年中有 4 年在 2014-2017、1 年在過渡年 2018），2023 門檻期與 2024+ "
                 "測試期不受影響。gemini.py 載入的 gold_long_recent_candidate_xgb.json 由 gold_long_recent_walk_forward.py 以 "
                 "M1 訓練，資料起點 2025-01-01（驗證 2026-04-01、測試 2026-06-01），高階週期暖機最多回推 760 天（約 2022-12），"
                 "全部在 2019 年之後，不受 E-013 揭露的問題影響",
    "evidence": "frozen/h1_C_long_v1/h1_lab_v1.py（docstring 與 FEATURES）、frozen/h1_C_long_v1/config.json、"
                "gold_long_recent_walk_forward.json（data.start／validation_start／test_start）、gold_recent_walk_forward.py",
}


def dep(on, kind, note=""):
    return {"on": on, "type": kind, **({"note": note} if note else {})}


H1_OLD, H1_2018, SPREAD_OLD, SWAP = ("data:gold_h1_2014_2017", "data:gold_h1_2018", "data:gold_spread_field_pre_2018_10",
                                    "premise:overnight_swap_charged")

D017 = {
    "kind": "decision", "id": "D-017",
    "statement": "Priority Engine 依賴圖（第一版）與上游標記。資料：2014-2017 H1 與 2018 H1（過渡年）標為 defective，"
                 "2018-10 前的 SPREAD 欄位標為 defective，2019+ H1 與 2025+ M1 為乾淨葉節點；前提：本 demo 帳戶隔夜收 swap "
                 "標為 invalid。依賴：E-008／E-010 以 2014-2017 擬合；E-011 重建兩者的凍結參數；E-012 用 H1 C 訊號並以 "
                 "2018-2023 校準；E-001 用 H1 C；model:h1_c 以 2014-2022 擬合；model:gemini 以 2025+ M1；O-013／O-014／O-015／"
                 "O-017 分別以 E-008、E-008+E-010、E-011、E-012 為證據；H-019、H-020、H-022 以這些觀察為前提；H-018 以 E-012／O-017 "
                 "為證據；H-008、H-017 以隔夜收 swap 為前提；H-011、H-014 的結論以 E-008、E-010 為證據。只宣告依賴，不改變任何"
                 "實驗結果或已結案假說的狀態",
    "rationale": "E-013 證明上游資料出問題時，下游假說仍可能被當成有效證據；讓引擎沿依賴路徑追蹤受影響的證據並據以重排",
    "set_depends": {
        H1_OLD: [], H1_2018: [], SPREAD_OLD: [], SWAP: [], "data:gold_h1_2019_on": [], "data:gold_m1_2025_on": [],
        "E-008": [dep(H1_OLD, "data", "凍結 HAR 以 2014-2017 擬合")],
        "E-010": [dep(H1_OLD, "data", "凍結 EWMA／GVZ 模型以 2014-2017 擬合")],
        "E-011": [dep("E-008", "evidence"), dep("E-010", "evidence"), dep(H1_OLD, "data", "與 2014-2017 分布比較")],
        "E-012": [dep("model:h1_c", "evidence", "固定的 H1 C 進場訊號"), dep(H1_2018, "data", "校準窗 2018-2023")],
        "E-001": [dep("model:h1_c", "evidence")],
        "model:h1_c": [dep(H1_OLD, "data", "擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留"),
                       dep(H1_2018, "data"), dep(SPREAD_OLD, "data", "以 0.008% 成本下限代替")],
        "model:gemini": [dep("data:gold_m1_2025_on", "data", "O-019")],
        "O-013": [dep("E-008", "evidence")], "O-014": [dep("E-008", "evidence"), dep("E-010", "evidence")],
        "O-015": [dep("E-011", "evidence")], "O-017": [dep("E-012", "evidence")],
        "H-019": [dep("O-013", "premise", "截距把預測拉向 2014-2017 水準")],
        "H-022": [dep("O-014", "premise", "水準校準漂移"), dep("O-015", "premise")],
        "H-020": [dep("O-013", "premise"), dep("O-014", "premise"), dep("O-015", "premise")],
        "H-018": [dep("E-012", "evidence"), dep("O-017", "evidence")],
        "H-008": [dep(SWAP, "premise", "隔夜持有多單要付 swap")],
        "H-017": [dep(SWAP, "premise"), dep("model:h1_c", "evidence")],
        "H-011": [dep("E-008", "evidence")], "H-014": [dep("E-010", "evidence")],
    },
    "flag": {
        H1_OLD: {"state": "defective", "evidence": ["O-018", "E-013"],
                 "note": "換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%）"},
        H1_2018: {"state": "defective", "evidence": ["O-018"],
                  "note": "過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起"},
        SPREAD_OLD: {"state": "defective", "evidence": ["O-018"], "note": "SPREAD 欄位一半以上為 0"},
        SWAP: {"state": "invalid", "evidence": ["O-016"],
               "note": "本 demo 帳戶 swap_mode=DISABLED，跨換日部位未被收 swap（真實帳戶另需確認）"},
    },
}

D018 = {
    "kind": "decision", "id": "D-018",
    "statement": "核准（使用者 2026-11-16 指示）：H-008 設為 premise_invalid（隔夜收 swap 的前提在本帳戶不成立，停止依原假設推進）；"
                 "H-019、H-022 設為 needs_revalidation（前提來自受 2014-2017 資料影響的觀察；不等於被推翻）。model:h1_c 經 D-017 "
                 "標為訓練資料可能受污染，2024+ 評估保留，Demo Runner 不變，是否重訓另行預先登記。引擎另建議 H-017 premise_invalid、"
                 "H-018 與 H-020 needs_revalidation，屬系統建議，未在本決策核准",
    "rationale": "區分已核准的研究決策與系統自動建議；needs_revalidation 只代表證據基礎需要以 2019+ 資料重新驗證",
    "set_status": {"H-008": "premise_invalid", "H-019": "needs_revalidation", "H-022": "needs_revalidation"},
}


def main():
    events = [O019, D017, D018]
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
