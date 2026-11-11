"""Research Priority Engine and question generator over the research ledger.

rank(): orders unresolved hypotheses by expected research value, not by expected backtest profit. A test that
can only rule out an important possibility still scores through info_gain and falsifiability.

    score = 3*info_gain + 2*novelty + 2*falsifiability + data_quality + economic_value + simplicity
            - 2*mining_risk - 1.5*min(results already in the family, 4) - status penalty

Inputs are 1-5 per hypothesis (priority_inputs on the hypothesis, or set_priority_inputs in a later decision).
data_quality covers quality and acquisition cost (5 = clean and already in hand).

coverage_markdown(): crosses information sources with prediction targets, counts how often each cell was
explored (pre-ledger history + tagged hypotheses), and proposes unexplored cells whose data can be obtained.
Proposals are questions, not hypotheses: each still needs a mechanism, a falsification rule and a critic pass.

    python -B research_planner_v1.py        # print ranking and proposals
"""
import research_memory_v1 as rm

RANKED_STATUS = {"open": 0, "testing": 0, "inconclusive": 0, "underpowered": 2, "parked": 5,
                 "exhausted_needs_new_data": 8}

INFORMATION = {  # source -> (label, availability)
    "price_ohlc": ("價格 K 線", "in_hand"),
    "tick_spread": ("Tick／點差動態", "in_hand (MT5 ticks, months of history)"),
    "economic_calendar": ("經濟事件實際值／預期值", "export_needed (ExportUsdCalendarValues)"),
    "rates_macro": ("利率／總經", "in_hand (to 2024-12)"),
    "positioning_cot": ("COT 部位", "in_hand (to 2024-12)"),
    "cross_asset": ("跨市場（美元、白銀、指數）", "in_hand"),
    "implied_vol": ("隱含波動率（CBOE GVZ）", "new_source (FRED GVZCLS, free)"),
    "news_text": ("新聞文字", "new_source (no feed yet)"),
}
TARGETS = {"direction": "漲跌方向", "volatility": "未來波動", "tail": "報酬分布／尾部", "tradeability": "值不值得交易",
           "cost": "成本與執行", "regime": "市況轉換"}
# Pre-ledger research (README generation table, TRAINING_RUNS.md, 2026-11 labs), counted per cell.
HISTORICAL = {("price_ohlc", "direction"): 25, ("rates_macro", "direction"): 2, ("positioning_cot", "direction"): 1,
              ("economic_calendar", "direction"): 2, ("tick_spread", "direction"): 1, ("cross_asset", "direction"): 2}


def family_trials(events, hyps):
    prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
    counts = {}
    for e in events:
        if e["kind"] == "result" and e["outcome"] != "aborted":
            fam = hyps[prereg[e["experiment"]]["hypothesis"]]["family"]
            counts[fam] = counts.get(fam, 0) + 1
    return counts


def score(h, trials):
    p = h.get("priority_inputs")
    if not p:
        return None
    raw = (3 * p["info_gain"] + 2 * p["novelty"] + 2 * p["falsifiability"] + p["data_quality"] + p["economic_value"]
           + p["simplicity"] - 2 * p["mining_risk"] - 1.5 * min(trials.get(h["family"], 0), 4)
           - RANKED_STATUS[h["status"]])
    return round(raw, 1)


def rank(events):
    hyps, _, _ = rm.state(events)
    trials = family_trials(events, hyps)
    live = [h for h in hyps.values() if h["status"] in RANKED_STATUS]
    scored = [(h, score(h, trials)) for h in live]
    return sorted(scored, key=lambda x: -(x[1] if x[1] is not None else -1e9))


def coverage(events):
    hyps, _, _ = rm.state(events)
    cells = dict(HISTORICAL)
    for h in hyps.values():
        tags = h.get("tags", {})
        for info in tags.get("information", []):
            for target in tags.get("target", []):
                cells[(info, target)] = cells.get((info, target), 0) + 1
    return cells


SLOW_SOURCES = {"rates_macro", "positioning_cot", "implied_vol"}  # daily/weekly data cannot speak to intraday costs


def proposals(events, limit=8, per_source=2):
    cells = coverage(events)
    open_cells = [(i, t) for i in INFORMATION for t in TARGETS
                  if cells.get((i, t), 0) == 0 and not (i in SLOW_SOURCES and t == "cost")]
    # Prefer data already in hand, then sources whose other targets are least explored (most novel).
    def key(c):
        avail = INFORMATION[c[0]][1]
        tier = 0 if avail.startswith("in_hand") else 1 if avail.startswith("export") else 2
        return tier, sum(v for (i, _), v in cells.items() if i == c[0])
    picked, used = [], {}
    for c in sorted(open_cells, key=key):
        if used.get(c[0], 0) < per_source:
            picked.append(c)
            used[c[0]] = used.get(c[0], 0) + 1
    return picked[:limit]


def coverage_markdown(events):
    cells = coverage(events)
    out = ["探索次數（歷史研究＋帳本中已標註的假說）。0 代表從未研究過。", "",
           "| 資訊來源 | 取得狀態 | " + " | ".join(TARGETS.values()) + " |",
           "|---|---|" + "---|" * len(TARGETS)]
    for i, (label, avail) in INFORMATION.items():
        out.append(f"| {label} | {avail} | " + " | ".join(str(cells.get((i, t), 0)) for t in TARGETS) + " |")
    out += ["", "**自動產生的候選研究問題**（尚未成為假說；要成為假說需補上機制、推翻條件，並通過審查）：", ""]
    for i, t in proposals(events):
        out.append(f"- {INFORMATION[i][0]} 是否含有預測 GOLD# **{TARGETS[t]}** 的資訊？（資料：{INFORMATION[i][1]}）")
    return out


def main():
    events = rm.read()
    for h, s in rank(events):
        print(f"{s!s:>6}  {h['id']:6} {h['status']:24} {h['title']}")
    print()
    print("\n".join(coverage_markdown(events)))


if __name__ == "__main__":
    main()
