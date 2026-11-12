"""Ledger batch: E-009 aborted before evaluation (swap_mode guard), O-015 (E-011 reading), O-016 (no swap on the
demo account), H-021, H-022, D-013, and E-012 = E-009 design with the account's actual swap.

    python -B research_memory/batch_2026_11_12_e012.py           # dry run
    python -B research_memory/batch_2026_11_12_e012.py --append  # also finalizes the orphan E-009 run as aborted
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402

E009_RUN = "20261112T135141Z_research_e009_exp_vol_sizing_v1"
E011_RUN = "training_runs/20261112T135155Z_research_e011_exp_fixed_param_drift_v1"

E009_ABORT = {
    "kind": "result", "experiment": "E-009", "outcome": "aborted", "evidence_class": "development",
    "summary": "aborted before any evaluation: the pre-run guard found GOLD# swap_mode 0 (DISABLED) where the design "
               "assumed swap in points; bars were fetched but no signal, trade or P&L was computed",
    "failure_cause": "method", "lessons": "verify broker cost settings against real deals before preregistering costs; "
                                          "the runner did not catch SystemExit from a module (fixed)",
    "next_questions": "E-012: same design with the account's actual swap (0) and the quoted swap as stress",
    "run_dir": f"training_runs/{E009_RUN}",
}

O015 = {
    "kind": "observation", "id": "O-015", "status": "provisional",
    "statement": "E-011（描述性，已看過的資料）：重建的凍結參數與 E-008／E-010 紀錄完全一致（資料未被修訂）。"
                 "2018-2023 的 GOLD# 實現波動比 2014-2017 低約一半（log y5 平均 -9.68 vs -8.99，KS 0.39），2024+ 回到接近 2014-2017；"
                 "輸入同步位移（log EWMA 有 31% 落在 2014-2017 的 1-99% 範圍外）。凍結預測在 2018-2023 每年高估（HAR mean y/f 0.56-0.78），"
                 "只乘上當年最佳尺度就補回大部分誤差（例：R 2018 QLIKE 0.198→0.116，同年重配 0.110），條件關係改變的部分相對小；"
                 "A（含 GVZ）重配後仍有較大差距（2021、2025、2026）。GVZ 對 EWMA 的 log 溢酬由 -0.41 翻到 +0.28（KS 0.67），"
                 "即 E-010 失敗來源。資料品質斷點：2018 年前 SPREAD 全為 0、每日 23.7-23.8 根 H1（之後 22.8-22.9），"
                 "且 2014-2017 隱含波動低於 H1 實現波動（溢酬為負，與常見的正溢酬相反），提示 2018 年前 H1 資料可能放大實現波動",
    "evidence": f"{E011_RUN}/report.md",
}

O016 = {
    "kind": "observation", "id": "O-016", "status": "verified",
    "statement": "XM demo 帳戶的 GOLD# swap_mode = 0（SYMBOL_SWAP_MODE_DISABLED），雖然 swap_long 報 -78.59。"
                 "唯讀檢查 400 天內 575 筆已平倉部位：9 筆跨過伺服器換日（多空皆有，含跨週三三倍夜的多單 2026-04-14→04-16），"
                 "swap 全為 0。E-001、E-006、E-007 以每年約 7% 的 swap 計成本，對這個帳戶不成立；真實帳戶可能不同，需另行確認",
    "evidence": "read-only MT5 symbol_info and history_deals_get on 2026-11-12 (no orders sent)",
}

H021 = {
    "kind": "hypothesis", "id": "H-021", "status": "open", "family": "data_quality",
    "title": "2018 年前的 XM GOLD# H1 資料放大了日實現波動（資料源或報價斷點），使 2014-2017 擬合的水準不能代表之後的市場",
    "mechanism": "2018 年前 SPREAD=0、每日 K 棒數不同、隱含波動低於實現波動，指向不同的報價來源或雜訊",
    "novelty": "把『固定參數退化』的一部分歸因於資料，而不是市場；若成立，所有以 2014-2017 擬合的研究都要重新界定",
    "information_needed": "XM H1 與 D1、外部獨立的黃金日價格（如 GLD 日線或其他公開來源）",
    "falsified_if": "以不涉及模型擬合的一致性比較：H1 加總的日實現波動相對 D1 收盤對收盤變異數（或外部來源）的比例，"
                    "在 2018 年前後差異 < 10%",
    "data_mining_risk": "medium：使用已看過的期間，但檢驗的是資料一致性，不是預測或交易績效",
    "tags": {"information": ["price_ohlc"], "target": ["data_quality"]},
    "priority_inputs": {"info_gain": 4, "novelty": 3, "falsifiability": 4, "data_quality": 3, "economic_value": 3,
                        "simplicity": 5, "mining_risk": 2},
}

H022 = {
    "kind": "hypothesis", "id": "H-022", "status": "open", "family": "fixed_parameter_drift",
    "title": "只做水準重新校準（每月一個尺度參數，用過去資料）的 EWMA_094 預測，在前瞻資料上優於凍結與未校準版本",
    "mechanism": "E-011 顯示退化主要來自水準偏移（校準漂移），條件關係改變較小",
    "novelty": "針對已診斷出的主要退化來源的最小修正；不增加模型複雜度",
    "information_needed": "GOLD# H1，只用 2026-11-12 之後的資料判定",
    "falsified_if": "前瞻至少 250 個完整交易日，校準版 QLIKE 不低於原始 EWMA_094（DM 單尾 p >= 0.05）",
    "data_mining_risk": "high：由 E-011 事後導出，只能用前瞻資料判定",
    "tags": {"information": ["price_ohlc"], "target": ["volatility"]},
    "priority_inputs": {"info_gain": 3, "novelty": 2, "falsifiability": 4, "data_quality": 5, "economic_value": 2,
                        "simplicity": 5, "mining_risk": 4},
}

D013 = {
    "kind": "decision", "id": "D-013",
    "statement": "E-009 在任何計算前被 swap 單位檢查中止（GOLD# swap_mode 為 DISABLED），記為 aborted；Runner 修正為也能接住"
                 "模組拋出的 SystemExit。E-012 沿用 E-009 已審查的設計，只改 swap：基準情境用帳戶實際 swap（0），"
                 "報價 swap_long×1.5 改為壓力情境。O-016 推翻了 E-001／E-006／E-007 的 swap 成本前提："
                 "H-008（E-006 的『盤中優於持有』主要來自 swap）需以帳戶實際成本重新檢驗，狀態維持 inconclusive；"
                 "H-017（換日前平倉以省 swap）的前提不成立，設為 parked。E-011 的判讀記為 O-015，"
                 "可檢驗的後續方向登記為 H-021（資料品質，可用一致性比較檢驗）與 H-022（只用前瞻資料）",
    "rationale": "成本前提必須以真實成交紀錄驗證；中止發生在看到任何結果之前，所以沿用同一設計不構成事後調整。"
                 "揭露：E-009 與 E-012 之間已看過 E-011 的描述結果（O-015，波動水準，沒有部位管理的損益）；"
                 "trials_in_family 會把這次中止算成 H-018 家族的一次嘗試",
    "set_status": {"H-017": "parked"},
}


def e012(events):
    old = next(e for e in events if e["kind"] == "preregistration" and e["id"] == "E-009")
    p = {k: copy.deepcopy(v) for k, v in old.items() if k not in ("seq", "sha256", "prev_sha256", "time_utc")}
    p["id"] = "E-012"
    p["code_files"] = {f: rm.file_sha256(rm.ROOT / f) for f in old["code_files"]}
    p["spec"]["cost_note"] = ("spread = lab bar spread floored at 0.008% of price, paid at entry (x2 scenario); swap: "
                              "base = the account's actual swap (GOLD# swap_mode DISABLED, 0 charged on every "
                              "rollover in 400 days of deals, O-016); stress = quoted swap_long as points, as a "
                              "fraction of price per night, charged on notional at every rollover crossed, Wednesday "
                              "triple, x1.5; larger notional pays proportionally more")
    rule = old["decision_rule"]
    assert rule.count("under swap x1.5") == 1
    p["decision_rule"] = rule.replace("under swap x1.5", "under quoted swap x1.5 (base swap is the account's 0)")
    p["claim_scope"] = old["claim_scope"] + (" Costs are this XM demo account's (no swap); a live account that "
                                             "charges swap is covered only by the quoted-swap stress scenario.")
    return p


def main():
    events = rm.read()
    batch = [E009_ABORT, O015, O016, H021, H022, D013]
    if "--append" in sys.argv:
        import training_run_history as trh
        run_dir = rm.ROOT / "training_runs" / E009_RUN
        log = run_dir / "stdout.log"
        if not log.read_text(encoding="utf-8"):
            log.write_text("SystemExit: swap_mode 0: swap_frac assumes swap in points\n"
                           "(raised inside the module before any evaluation; the runner did not catch SystemExit, "
                           "so the run was finalized afterwards by research_memory/batch_2026_11_12_e012.py)\n",
                           encoding="utf-8")
        errors = trh.finalize_run(run_dir, "aborted", aborted_reason="swap_mode guard before evaluation; see stdout.log")
        print("finalize errors:", errors)
        rm.append(batch)
        rm.append([e012(rm.read())])
        rm.render(rm.read())
        print(f"appended {len(batch) + 1} events")
        return
    current = events
    for e in batch:
        rm.check(e, current)
        current = current + [{**e, "seq": len(current)}]
    p = e012(current)
    rm.check(p, current)
    print(f"dry run OK: {len(batch) + 1} events")


if __name__ == "__main__":
    main()
