"""Ledger batch 2026-11-11: evidence-status taxonomy (D-006), withdrawal of E-003/E-004 before they ran,
new hypotheses H-008..H-014, priority inputs and tags, and preregistration of E-005/E-006/E-007.

    python -B research_memory/batch_2026_11_11_governance.py           # dry run: check every event, append nothing
    python -B research_memory/batch_2026_11_11_governance.py --append
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_memory_v1 as rm  # noqa: E402


def pi(info_gain, novelty, falsifiability, data_quality, economic_value, simplicity, mining_risk):
    return dict(info_gain=info_gain, novelty=novelty, falsifiability=falsifiability, data_quality=data_quality,
                economic_value=economic_value, simplicity=simplicity, mining_risk=mining_risk)


def tags(information, target):
    return {"information": information, "target": target}


def H(**kw):
    return {"kind": "hypothesis", "status": "open", **kw}


HYPOTHESES = [
    H(id="H-008", title="收盤前平倉、只做盤中多單，比一直持有多單（付過夜利息）的風險調整後報酬更好",
      family="cost_structure",
      mechanism="多單每晚付 swap（今日約 -78.6 點，週三三倍），盤中持有每天多付一次點差但不付 swap",
      novelty="改變的是持有方式的成本結構，不預測方向；從 H-004 拆出，避免和時段方向優勢混在一起",
      information_needed="GOLD# H1 2018+、今日 swap 與點差", tags=tags(["price_ohlc"], ["cost"]),
      falsified_if="2018-2022 成本 x1 下 INTRADAY_LONG 的 Sharpe 不高於 ALWAYS_LONG",
      data_mining_risk="low：單一固定規則", priority_inputs=pi(4, 3, 5, 5, 4, 5, 1)),
    H(id="H-009", title="點差相對同時段常態異常擴大時，接下來的波動與滑價較大，是不值得進場的時刻",
      family="microstructure",
      mechanism="點差擴大反映流動性提供者撤單或資訊不對稱，之後價格跳動與滑價成本上升",
      novelty="以 tick 層級點差動態當成『不交易』條件，不再把微結構當 M1 方向分類器的特徵",
      information_needed="MT5 copy_ticks_range（僅數個月歷史；需要先累積）",
      tags=tags(["tick_spread"], ["tradeability", "volatility"]),
      falsified_if="點差 z 值最高 10% 時段之後 30 分鐘的實現波動比 < 1.2，或排除這些時段進場使既有策略回放淨 R 改善 < 0.02R",
      data_mining_risk="medium：門檻要事先固定", priority_inputs=pi(4, 5, 4, 3, 4, 3, 3)),
    H(id="H-010", title="重要美國數據公布前後的波動放大可由事件類型和預期差大小預測；公布前後停止進場能改善 M1 策略",
      family="event_volatility",
      mechanism="排程事件前流動性收縮、公布後重新定價；預期差越大，波動越大，方向不必可預測",
      novelty="預測事件前後的波動與可交易性，而不是方向；過去只用事件時間當方向特徵",
      information_needed="ExportUsdCalendarValues 匯出的實際值／預期值（待使用者在 MT5 執行）",
      tags=tags(["economic_calendar"], ["volatility", "tradeability"]),
      falsified_if="公布後／公布前實現波動比與 |預期差| 的等級相關 <= 0，或 ±30 分鐘不交易使回放淨 R 改善 < 0.02R",
      data_mining_risk="medium：事件清單與窗口要事先固定", priority_inputs=pi(4, 4, 4, 2, 4, 3, 3)),
    H(id="H-011", title="HAR 實現波動模型對次日波動的預測優於 EWMA，且高波動狀態下的報酬分布（偏態、尾部）顯著不同",
      family="second_moment_forecast",
      mechanism="波動率具有長記憶（日、週、月成分），HAR 能捕捉 EWMA 忽略的多尺度結構",
      novelty="直接評估波動預測品質與條件分布，是倉位、停損距離和不交易規則的共同基礎",
      information_needed="GOLD# M5 → 日實現波動 2018+",
      tags=tags(["price_ohlc"], ["volatility", "tail", "regime"]),
      falsified_if="2018-2023 HAR 的 QLIKE 不低於 EWMA（Diebold-Mariano |t| < 2）",
      data_mining_risk="low：標準模型、無調參", priority_inputs=pi(4, 4, 5, 5, 3, 4, 2)),
    H(id="H-012", title="低流動時段的單根急動（> 3 倍 ATR）在 15-60 分鐘內部分回補，扣除當下實際點差後期望為正",
      family="transient_anomaly",
      mechanism="流動性稀薄時大單造成暫時性價格衝擊，之後由流動性提供者回補",
      novelty="以流動性衝擊事件為條件，不是 RSI 類的一般均值回歸；成本以急動當下擴大的點差計",
      information_needed="GOLD# M1 2018+（點差欄位）；tick 資料用於檢驗急動當下的實際點差",
      tags=tags(["price_ohlc", "tick_spread"], ["direction", "cost"]),
      falsified_if="2018-2022 以實際點差計的平均淨回補 <= 0；事件數 < 200 則判定樣本不足",
      data_mining_risk="medium-high：急動門檻與時段要事先固定", priority_inputs=pi(4, 5, 4, 4, 4, 3, 4)),
    H(id="H-013", title="可以預測未來 1 小時的波動是否大於 k 倍點差（不預測方向），而 gemini 的虧損集中在預測『不值得交易』的時段",
      family="tradeability",
      mechanism="波動太小時點差佔停利比例過高，任何方向判斷都難以覆蓋成本",
      novelty="預測目標改成『值不值得交易』；價格型模型換成新的預測目標，不是參數變體",
      information_needed="GOLD# H1/M1 2018+、gemini 回放交易",
      tags=tags(["price_ohlc"], ["tradeability"]),
      falsified_if="樣本外波動大小分類 AUC <= 0.55，或低可交易性組的 gemini 回放 R 沒有比其他組低 0.05R 以上",
      data_mining_risk="medium", priority_inputs=pi(4, 5, 4, 5, 4, 3, 3)),
    H(id="H-014", title="黃金隱含波動率（CBOE GVZ）與實現波動的差（變異數風險溢酬）預測未來 5-20 日的實現波動與報酬",
      family="new_information_implied_vol",
      mechanism="選擇權市場反映前瞻的風險定價；溢酬高代表避險需求與未來波動",
      novelty="全新資訊來源（XM 價格以外），免費可取得（FRED GVZCLS，2008 起）",
      information_needed="FRED GVZCLS 日資料，需新增資料匯入與可得時間戳",
      tags=tags(["implied_vol"], ["volatility", "direction"]),
      falsified_if="把 GVZ 加進 HAR 後 QLIKE 沒有下降（DM |t| < 2），且溢酬對 20 日報酬的 IC |t| < 1",
      data_mining_risk="low-medium", priority_inputs=pi(4, 5, 5, 4, 3, 4, 2)),
]

DECISION = {
    "kind": "decision", "id": "D-006",
    "statement": "依證據強度重新分類並縮小結論範圍：H-001 改為『未找到顯著優勢』、H-002 改為『樣本不足』；"
                 "D-001 的範圍縮小為『同一組價格特徵的方向預測參數／模型變體已過度研究，需要新資料』；"
                 "作廢 E-003、E-004（執行前審查發現設計缺陷，從未產生結果）；H-004 只保留時段方向問題，成本問題移到 H-008",
    "rationale": "H-001：只檢驗了 3 個固定候選（TP3/SL1.5/48h），H1_C 樣本外只有 274 筆，無法偵測 < 0.15R 的時機能力；"
                 "這不代表價格型策略或機器學習沒有研究價值。H-002：實質利率部分符合預先登記的推翻條件，"
                 "COT 方向在兩段一致但 |t| < 1.4，資料只到 2024。D-001：更多同類變體在同一段歷史資料上無法產生獨立證據，"
                 "只能用 2026-11-12 之後的資料或新資訊、新預測目標（H-011、H-013）判定。E-003 缺：成本與方向未分開、"
                 "無方向曝險對照、缺換日段、無成本壓力測試與 Sharpe/Calmar/回撤；E-004 缺：與 w=1 比較的風險預算不同、"
                 "無權重打亂對照、成本與 swap 只有單一假設。",
    "set_status": {"H-001": "not_significant", "H-002": "underpowered"},
    "withdraw": ["E-003", "E-004"],
    "set_tags": {"H-001": tags(["price_ohlc"], ["direction"]),
                 "H-002": tags(["rates_macro", "positioning_cot"], ["direction"]),
                 "H-003": tags(["economic_calendar"], ["direction"]), "H-004": tags(["price_ohlc"], ["direction"]),
                 "H-005": tags(["price_ohlc"], ["volatility"]), "H-006": tags(["price_ohlc"], ["regime"]),
                 "H-007": tags(["tick_spread"], ["cost"])},
    "set_priority_inputs": {"H-001": pi(2, 2, 4, 5, 2, 4, 3), "H-002": pi(3, 3, 4, 3, 3, 4, 2),
                            "H-003": pi(4, 4, 4, 2, 4, 3, 3), "H-004": pi(3, 3, 4, 5, 3, 5, 4),
                            "H-005": pi(3, 4, 5, 5, 3, 5, 1), "H-006": pi(2, 3, 4, 3, 2, 4, 2),
                            "H-007": pi(3, 3, 4, 4, 4, 3, 3)},
}

METRICS = ["net_mean", "sharpe", "calmar", "max_dd"]
SESSION_COMMON = {
    "windows": [["2018-01-01", "2023-01-01"], ["2023-01-01", "2099-01-01"]], "spread_floor_frac": 0.00008,
    "cost_multipliers": [1.0, 1.5, 2.0], "swap_multipliers": [0.5, 1.0], "min_positive_years": 4, "metrics": METRICS,
    "cost_note": "round trip max(first/last bar spread of the segment, 0.00008 x price) x multiplier, plus a "
                 "live-31-point floor scenario; swap as a rate = today's swap_long / today's price per night x nights "
                 "(Thursday triple) x {0.5, 1}, charged only on positions held across rollover",
}


def code(module):
    return {f"{module}.py": rm.file_sha256(rm.ROOT / f"{module}.py")}


def prereg(eid, hyp, module, spec, rule, scope):
    return {"kind": "preregistration", "id": eid, "hypothesis": hyp, "role": "exploratory", "module": module,
            "code_files": code(module), "spec": spec, "evaluation_windows": spec["windows"], "decision_rule": rule,
            "author": "researcher", "training_window_note": "no model is fitted; fixed definitions only",
            "claim_scope": scope}


PREREGS = [
    prereg("E-005", "H-004", "exp_session_structure_v2",
           {**SESSION_COMMON, "question": "direction", "trials_declared": 8,
            "controls": ["ALWAYS_LONG", "proportional drift share per session"],
            "directional_benchmark": "session hours / 23 x the month's mean 01-24 intraday return (leave-one-out: the day itself excluded); ALWAYS_LONG",
            "power_note": "about 1250 days per session in 2018-2022; daily session SD about 0.4-0.6% gives SE about "
                          "1.5 bp, so with Sidak over 8 trials (t about 2.7) only excess of about 4 bp/day "
                          "(about 10%/yr) is detectable; smaller effects end as not_significant"},
           "At cost x1 a session x side passes if its excess over proportional drift is positive in >= 4 of 5 years "
           "2018-2022 with Sidak p < 0.05 over 8 trials (times family trials), its 2018-2022 excess stays > 0 at "
           "cost x2, its 2023+ excess is > 0, and its raw traded net mean is > 0 in both windows. supported if any "
           "passes; refuted if the best 2018-2022 excess t has Sidak p >= 0.5; otherwise not_significant "
           "(ledger outcome inconclusive)",
           "Only these 4 fixed server-hour sessions on GOLD# at XM costs. refuted/not_significant only excludes "
           "session effects of about 4 bp/day or more (smaller ones are below one spread anyway); it does not rule out "
           "other session definitions or intraday signals conditioned on information; a pass is development "
           "evidence and needs data after 2026-11-12."),
    prereg("E-006", "H-008", "exp_session_structure_v2",
           {**SESSION_COMMON, "question": "cost", "trials_declared": 1,
            "controls": ["ALWAYS_LONG (net swap)", "OVERNIGHT_LONG (what intraday-only gives up)"],
            "directional_benchmark": "ALWAYS_LONG: both candidates are long; only the holding window differs",
            "power_note": "about 1250 days in 2018-2022 and about 950 after; the Sharpe comparison is descriptive, "
                          "with a cost x1.5 stress required for support"},
           "refuted if INTRADAY_LONG Sharpe <= ALWAYS_LONG Sharpe in 2018-2022 at cost x1 swap x1; supported if "
           "INTRADAY_LONG Sharpe > ALWAYS_LONG Sharpe in both windows in every one of: cost x1.5, cost live31, "
           "swap x0.5; otherwise inconclusive",
           "Whether flat-at-rollover beats always-long as a way to hold gold long exposure at XM's spread and today's "
           "swap. Says nothing about timing skill or other symbols."),
    prereg("E-007", "H-005", "exp_vol_managed_v2",
           {"windows": [["2018-01-01", "2024-01-01"], ["2024-01-01", "2099-01-01"]], "lambda": 0.94, "cap": 2.0,
            "spread_frac": 0.00008, "cost_multipliers": [1.0, 2.0, 3.0], "swap_multipliers": [0.5, 1.0],
            "shuffle_draws": 200, "trials_declared": 1, "metrics": METRICS,
            "controls": ["CONST_EQUAL_EXPOSURE", "CONST_VOL_MATCHED", "SHUFFLED month-block weights"],
            "directional_benchmark": "CONST_EQUAL_EXPOSURE: same average long exposure and swap bill",
            "cost_note": "spread_frac x multiplier x |weight change| per rebalance (constants once); swap as a rate "
                         "= today's swap_long / today's price per night x nights (Thursday triple) x weight x {0.5, 1}",
            "power_note": "about 1500 days 2018-2023; MANAGED and the constant correlate about 0.9, so the paired "
                          "month-bootstrap SE of the Sharpe difference is about 0.15-0.2 and differences above about "
                          "0.35 are detectable; smaller true gains end as inconclusive"},
           "refuted if Sharpe(MANAGED) - Sharpe(CONST_EQUAL_EXPOSURE) <= 0 in 2018-2023 at cost x1; supported if the "
           "difference is > 0 in 2024+ at cost x1, its 2018-2023 month-bootstrap 95% CI is above 0, it stays > 0 in "
           "2018-2023 at cost x3, and <= 5% of month-shuffled weight paths reach MANAGED's 2018-2023 Sharpe; "
           "otherwise inconclusive",
           "One fixed EWMA rule for sizing a gold long. A failure does not rule out better volatility forecasts "
           "(H-011) or sizing other strategies; daily rebalancing ignores 0.01-lot granularity."),
]

EVENTS = HYPOTHESES + [DECISION] + PREREGS


def main():
    if "--append" in sys.argv:
        rm.append(EVENTS)
        rm.render(rm.read())
        print(f"appended {len(EVENTS)} events")
        return
    events = rm.read()
    for e in EVENTS:  # dry run: same checks as append, on a growing in-memory copy
        rm.check(e, events)
        events = events + [{**e, "seq": len(events)}]
    print(f"dry run OK: {len(EVENTS)} events")


if __name__ == "__main__":
    main()
