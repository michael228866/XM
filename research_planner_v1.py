"""Research Priority Engine and question generator over the research ledger.

rank(): orders unresolved hypotheses by expected research value, not by expected backtest profit. A test that
can only rule out an important possibility still scores through info_gain and falsifiability.

    score = 3*info_gain + 2*novelty + 2*falsifiability + data_quality + economic_value + simplicity
            - 2*mining_risk - 1.5*min(results already in the family, 4) - status penalty

Inputs are 1-5 per hypothesis (priority_inputs on the hypothesis, or set_priority_inputs in a later decision).
data_quality covers quality and acquisition cost (5 = clean and already in hand).

assess(): evidence-failure propagation over the dependency graph that decisions declare (set_depends: node -> what it
rests on, typed data / premise / evidence; flag: node -> defective / invalid / ok with evidence ids). A node's state
comes from its latest flag, else from research outcomes (refuted hypothesis or retracted observation = invalid). Walking
upstream through any depth: an invalid node reached through premise edges only -> premise_invalid; any other problem,
a missing dependency or a cycle -> needs_revalidation. An ok flag on an intermediate node stops propagation (repair).

ranking() applies the assessment automatically, citing the decision and evidence for every adjustment, and keeps the
original score: premise_invalid leaves the ranking whatever its inputs; needs_revalidation drops data_quality to 1 when
a defective data node is upstream and loses CREDIBILITY_PENALTY; each defective upstream node becomes a repair item
worth its best affected hypothesis + 0.5 per further affected item. suggestions() lists where the engine disagrees with
the approved status; only a decision (set_status) approves. Terminal statuses (refuted, supported, ...) and results are
never changed or suggested away: they get an annotation that their evidence is affected.

coverage_markdown(): crosses information sources with prediction targets, counts how often each cell was
explored (pre-ledger history + tagged hypotheses), and proposes unexplored cells whose data can be obtained.
Proposals are questions, not hypotheses: each still needs a mechanism, a falsification rule and a critic pass.

    python -B research_planner_v1.py        # print ranking and proposals
"""
import research_memory_v1 as rm

RANKED_STATUS = {"open": 0, "testing": 0, "inconclusive": 0, "underpowered": 2, "parked": 5,
                 "exhausted_needs_new_data": 8, "needs_revalidation": 0}
ENGINE_STATUS = {"premise_invalid", "needs_revalidation"}
TERMINAL = rm.HYPOTHESIS_STATUS - set(RANKED_STATUS) - ENGINE_STATUS
CREDIBILITY_PENALTY = 5  # testing a claim whose grounds are suspect tells less than its inputs say

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
           - RANKED_STATUS.get(h["status"], 0))
    return round(raw, 1)


def graph(events):
    deps, flags = {}, {}
    for e in events:
        if e["kind"] == "decision":
            deps.update(e.get("set_depends", {}))
            flags.update({n: {**f, "by": e["id"]} for n, f in e.get("flag", {}).items()})
    return deps, flags


def node_state(n, flags, hyps, obs):
    if n in flags:
        return flags[n]["state"], flags[n]["by"], flags[n]["evidence"], flags[n].get("note", "")
    status = hyps.get(n, {}).get("status")
    if status in ("refuted", "premise_invalid") or obs.get(n) == "retracted":
        return "invalid", n, [n], f"{n} 狀態 {status or 'retracted'}"
    if status == "needs_revalidation":
        return "defective", n, [n], f"{n} 已核准為需要重新驗證"
    return "ok", None, [], ""


def assess(events):
    hyps, _, _ = rm.state(events)
    obs = {e["id"]: e["status"] for e in events if e["kind"] == "observation"}
    deps, flags = graph(events)
    known = {e["id"] for e in events if e.get("id")} | set(deps) | set(flags)
    out = {}
    for n in set(deps) | set(hyps):
        reasons, missing, cycles = [], [], []

        def walk(m, path, premise_chain):
            # shortcut: enumerates paths (exponential on dense diamonds); fine for ledger-sized graphs, memoize if slow
            for d in deps.get(m, []):
                nxt, chain = d["on"], premise_chain and d["type"] == "premise"
                if nxt in path:
                    cycles.append(path + [nxt])
                elif nxt not in known:
                    missing.append(path + [nxt])
                else:
                    state, by, evidence, note = node_state(nxt, flags, hyps, obs)
                    if state != "ok":
                        reasons.append({"node": nxt, "state": state, "path": path + [nxt], "premise_chain": chain,
                                        "decision": by, "evidence": evidence,
                                        "note": " / ".join(x for x in (note, d.get("note")) if x)})
                    if not (nxt in flags and state == "ok"):  # a repaired node shields what rests on it
                        walk(nxt, path + [nxt], chain)

        walk(n, [n], True)
        if any(r["state"] == "invalid" and r["premise_chain"] for r in reasons):
            state = "premise_invalid"
        else:
            state = "needs_revalidation" if reasons or missing or cycles else "ok"
        out[n] = {"state": state, "reasons": reasons, "missing": missing, "cycles": cycles, "deps": deps.get(n, [])}
    return out


def cites(x):
    return sorted({c for r in x["reasons"] for c in [r["decision"], *r["evidence"]] if c})


def why(x, limit=3):
    return "；".join([" → ".join(r["path"]) + f"（{r['state']}：{r['note']}）" for r in x["reasons"][:limit]]
                    + [f"缺失依賴 {' → '.join(m)}" for m in x["missing"]] + [f"循環 {' → '.join(c)}" for c in x["cycles"]])


def ranking(events):
    """Rows {id, kind, title, status, base, adjust: [(delta, why, evidence)], score}, best first."""
    hyps, _, _ = rm.state(events)
    trials = family_trials(events, hyps)
    a = assess(events)
    _, flags = graph(events)
    rows = []
    for h in hyps.values():
        x = a[h["id"]]
        if h["status"] == "premise_invalid":
            if x["state"] == "premise_invalid" or not x["deps"]:
                continue  # back in the ranking only once its premises are repaired
        elif h["status"] not in RANKED_STATUS or x["state"] == "premise_invalid":
            continue  # a premise shown false keeps it out whatever its expected value
        base, adjust = score(h, trials), []
        if x["state"] == "needs_revalidation" and base is not None:
            data = [r for r in x["reasons"] if r["node"].startswith("data:") and r["state"] == "defective"]
            if data and h["priority_inputs"]["data_quality"] > 1:
                adjust.append((1 - h["priority_inputs"]["data_quality"], "資料品質：上游資料有缺陷，data_quality 降為 1",
                               sorted({c for r in data for c in [r["decision"], *r["evidence"], r["node"]]})))
            adjust.append((-CREDIBILITY_PENALTY, "依賴證據可信度：" + why(x), cites(x)))
        rows.append({"id": h["id"], "kind": "hypothesis", "title": h["title"], "status": h["status"], "base": base,
                     "adjust": adjust, "score": None if base is None else round(base + sum(d for d, _, _ in adjust), 1),
                     "falsified_if": h["falsified_if"]})
    ranked = {r["id"]: r["base"] for r in rows}  # a repair is worth what it unblocks, before penalties
    affected = {}
    for n, x in a.items():
        if n in ranked or n.startswith("model:"):
            for r in x["reasons"]:
                if r["state"] == "defective" and r["node"] in flags:
                    affected.setdefault(r["node"], set()).add(n)
    for node, items in affected.items():
        best = max((ranked[i] for i in items if ranked.get(i) is not None), default=0)
        rows.append({"id": node, "kind": "repair", "status": "repair",
                     "title": f"上游修復：處理 {node} 並重新驗證 {'、'.join(sorted(items))}", "base": None,
                     "adjust": [(0, flags[node].get("note", ""), [flags[node]["by"], *flags[node]["evidence"]])],
                     "score": round(best + 0.5 * (len(items) - 1), 1), "falsified_if": "—"})
    return sorted(rows, key=lambda r: -(r["score"] if r["score"] is not None else -1e9))


def rank(events):
    hyps, _, _ = rm.state(events)
    return [(hyps[r["id"]], r["score"]) for r in ranking(events) if r["kind"] == "hypothesis"]


def suggestions(events):
    """(node, approved status, engine suggestion, reason, evidence). Suggestions only: a decision approves them."""
    hyps, _, _ = rm.state(events)
    out = []
    for n, x in sorted(assess(events).items()):
        status = hyps.get(n, {}).get("status")
        if status is None or status in TERMINAL:
            if x["state"] != "ok":  # results, observations, models and closed hypotheses are annotated, never changed
                out.append((n, status or "—", "annotate_only", why(x, 4), cites(x)))
        elif x["state"] != "ok" and status != x["state"]:
            out.append((n, status, x["state"], why(x, 4), cites(x)))
        elif x["state"] == "ok" and status in ENGINE_STATUS and x["deps"]:
            out.append((n, status, "open", "依賴已修復或改以有效資料為基礎", []))
    return out


def rerank_log(events):
    """For every decision that changed engine inputs: what moved in the ranking (cause = that decision)."""
    keys = ("set_status", "set_priority_inputs", "set_depends", "flag")
    log = []
    for i, e in enumerate(events):
        if e["kind"] != "decision" or not any(k in e for k in keys):
            continue
        before = {r["id"]: (k, r["score"]) for k, r in enumerate(ranking(events[:i]), 1)}
        after = {r["id"]: (k, r["score"]) for k, r in enumerate(ranking(events[:i + 1]), 1)}
        gone = ("—", "未列入")
        moves = [f"{n} {before.get(n, gone)[1]} → {after.get(n, gone)[1]}"
                 f"（名次 {before.get(n, gone)[0]} → {after.get(n, gone)[0]}）"
                 for n in sorted(set(before) | set(after))
                 if before.get(n, gone)[1] != after.get(n, gone)[1]]  # rank shifts caused by others are omitted
        if moves:
            log.append((e["id"], moves))
    return log


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
    for r in ranking(events):
        print(f"{r['score']!s:>6} ({r['base']!s:>5})  {r['id']:28} {r['status']:20} {r['title'][:70]}")
    print()
    for n, status, suggestion, reason, _ in suggestions(events):
        print(f"{n:28} {status:20} -> {suggestion:18} {reason[:110]}")
    print()
    print("\n".join(coverage_markdown(events)))


if __name__ == "__main__":
    main()
