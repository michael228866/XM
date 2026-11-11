"""Research memory: an append-only, hash-chained ledger of hypotheses, preregistrations, results,
observations, data usage and decisions, rendered to RESEARCH_MEMORY.md.

Why: past research lives in ~100 reports with no link from hypothesis -> experiment -> result -> next
question, and no count of how often each data window has been looked at. The ledger makes both explicit,
and refuses a "confirmatory" preregistration whose evaluation window was already used.

    python -B research_memory_v1.py verify
    python -B research_memory_v1.py append event.json      # one JSON object or a list
    python -B research_memory_v1.py render                 # -> RESEARCH_MEMORY.md
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "research_memory" / "ledger.jsonl"
RENDERED = ROOT / "RESEARCH_MEMORY.md"

REQUIRED = {
    "hypothesis": ["id", "title", "family", "mechanism", "novelty", "information_needed", "falsified_if",
                   "data_mining_risk", "status"],
    "preregistration": ["id", "hypothesis", "role", "module", "code_files", "spec", "evaluation_windows",
                        "decision_rule"],
    "result": ["experiment", "outcome", "evidence_class", "summary", "failure_cause", "lessons", "next_questions"],
    "observation": ["id", "statement", "evidence", "status"],
    "data_usage": ["window", "by", "note"],
    "decision": ["id", "statement", "rationale"],
    "review": ["experiment", "reviewer", "verdict", "findings"],
}
# refuted = evidence contradicts it; not_significant = tested, no significant effect found (absence not proven);
# underpowered = sample too small to decide; exhausted_needs_new_data = over-explored, only fresh or new data can decide.
HYPOTHESIS_STATUS = {"open", "testing", "supported", "refuted", "not_significant", "underpowered",
                     "exhausted_needs_new_data", "inconclusive", "parked", "closed"}
STATUS_LABEL = {"refuted": "證據推翻", "not_significant": "未找到顯著優勢（不代表不存在）",
                "underpowered": "樣本不足，無法判定", "exhausted_needs_new_data": "已過度研究，需要新資料才能判定"}
VERDICTS = {"approve", "revise", "reject"}
PRIORITY_KEYS = {"info_gain", "novelty", "falsifiability", "data_quality", "economic_value", "simplicity",
                 "mining_risk"}
OUTCOMES = {"supported", "refuted", "inconclusive", "aborted"}
EVIDENCE = {"clean_forward", "development", "reuse_contaminated"}
FAILURE = {"none", "information", "cost", "model", "method", "data", "regime_dependence"}
OBS_STATUS = {"verified", "provisional", "retracted"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def read(ledger=LEDGER):
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(events):
    """Chain and content check; returns a list of problems (empty = intact)."""
    problems, prev = [], "0" * 64
    for i, e in enumerate(events):
        body = {k: v for k, v in e.items() if k != "sha256"}
        if e.get("seq") != i:
            problems.append(f"event {i}: seq {e.get('seq')}")
        if e.get("prev_sha256") != prev:
            problems.append(f"event {i}: broken chain")
        if sha256_bytes(canonical(body)) != e.get("sha256"):
            problems.append(f"event {i}: content changed after append")
        prev = e.get("sha256")
    return problems


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]  # ISO dates compare as strings


def used_windows(events):
    out = []
    for e in events:
        if e["kind"] == "data_usage":
            out.append((e["window"], e["by"]))
        elif e["kind"] == "preregistration":
            out.extend((w, e["id"]) for w in e["evaluation_windows"])
    return out


def check(event, events, root=ROOT):
    """Raise ValueError if the event is malformed or breaks a research rule."""
    kind = event.get("kind")
    if kind not in REQUIRED:
        raise ValueError(f"unknown kind {kind!r}")
    missing = [f for f in REQUIRED[kind] if event.get(f) in (None, "", [])]
    if missing:
        raise ValueError(f"{kind} missing {missing}")
    ids = {e.get("id") for e in events if e.get("id")}
    if event.get("id") in ids:
        raise ValueError(f"id {event['id']} already used")
    if kind == "hypothesis" and event["status"] not in HYPOTHESIS_STATUS:
        raise ValueError(f"status must be one of {sorted(HYPOTHESIS_STATUS)}")
    if kind == "observation" and event["status"] not in OBS_STATUS:
        raise ValueError(f"status must be one of {sorted(OBS_STATUS)}")
    if kind == "preregistration":
        hyps = {e["id"] for e in events if e["kind"] == "hypothesis"}
        if event["hypothesis"] not in hyps:
            raise ValueError(f"unknown hypothesis {event['hypothesis']}")
        if event["role"] not in ("exploratory", "confirmatory"):
            raise ValueError("role must be exploratory or confirmatory")
        for name, digest in event["code_files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"{name} hash does not match the file on disk")
        if event["role"] == "confirmatory":
            hits = [(w, by) for w in event["evaluation_windows"] for (u, by) in used_windows(events) if overlaps(w, u)]
            if hits:
                raise ValueError(f"confirmatory window already used: {hits[:3]}; use new (forward) data")
    if kind == "result":
        prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
        if event["experiment"] not in prereg:
            raise ValueError("result must reference a preregistered experiment")
        if any(e["kind"] == "result" and e["experiment"] == event["experiment"] for e in events):
            raise ValueError("experiment already has a result; a new analysis needs a new preregistration")
        if event["outcome"] not in OUTCOMES or event["evidence_class"] not in EVIDENCE:
            raise ValueError("bad outcome or evidence_class")
        if event["failure_cause"] not in FAILURE:
            raise ValueError(f"failure_cause must be one of {sorted(FAILURE)}")
        p = prereg[event["experiment"]]
        prior = [u for u in used_windows(events[:events.index(p)]) for w in p["evaluation_windows"] if overlaps(w, u[0])]
        if prior and event["evidence_class"] == "clean_forward":
            raise ValueError("evaluation window was used before; evidence cannot be clean_forward")
    if kind == "decision":
        statuses = event.get("set_status", {})
        hyps = {e["id"] for e in events if e["kind"] == "hypothesis"}
        for h, s in statuses.items():
            if h not in hyps or s not in HYPOTHESIS_STATUS:
                raise ValueError(f"bad set_status {h}: {s}")
        for h in event.get("set_tags", {}):
            if h not in hyps:
                raise ValueError(f"unknown hypothesis {h}")
        for h, inputs in event.get("set_priority_inputs", {}).items():
            if h not in hyps:
                raise ValueError(f"unknown hypothesis {h}")
            check_priority_inputs(inputs)
        prereg = {e["id"] for e in events if e["kind"] == "preregistration"}
        done = {e["experiment"] for e in events if e["kind"] == "result"}
        for x in event.get("withdraw", []):
            if x not in prereg or x in done:
                raise ValueError(f"can only withdraw a preregistered experiment without a result: {x}")
    if kind == "review":
        prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
        if event["experiment"] not in prereg:
            raise ValueError("review must reference a preregistered experiment")
        if event["verdict"] not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        if event["reviewer"] == prereg[event["experiment"]].get("author"):
            raise ValueError("the author of a preregistration cannot review it")
        if any(e["kind"] == "result" and e["experiment"] == event["experiment"] for e in events):
            raise ValueError("review must come before the result")
    if kind == "hypothesis" and "priority_inputs" in event:
        check_priority_inputs(event["priority_inputs"])


def check_priority_inputs(inputs):
    if set(inputs) != PRIORITY_KEYS or not all(1 <= v <= 5 for v in inputs.values()):
        raise ValueError(f"priority inputs need exactly {sorted(PRIORITY_KEYS)} scored 1-5")


def append(new_events, ledger=LEDGER, root=ROOT):
    events = read(ledger)
    problems = verify(events)
    if problems:
        raise ValueError(f"ledger damaged, refusing to append: {problems[:3]}")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for event in new_events:
        check(event, events, root)
        body = {**event, "seq": len(events), "time_utc": event.get("time_utc") or
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prev_sha256": events[-1]["sha256"] if events else "0" * 64}
        body["sha256"] = sha256_bytes(canonical(body))
        events.append(body)
        lines.append(json.dumps(body, ensure_ascii=False, sort_keys=True))
    with ledger.open("a", encoding="utf-8", newline="\n") as f:
        f.write("".join(line + "\n" for line in lines))
    return events


def state(events):
    """Current hypothesis status and priority inputs: the last decision wins over what was declared."""
    hyps = {e["id"]: dict(e) for e in events if e["kind"] == "hypothesis"}
    prereg = {e["id"]: e for e in events if e["kind"] == "preregistration"}
    results = {e["experiment"]: e for e in events if e["kind"] == "result"}
    for e in events:
        if e["kind"] == "decision":
            for h, s in e.get("set_status", {}).items():
                hyps[h]["status"] = s
            for h, inputs in e.get("set_priority_inputs", {}).items():
                hyps[h]["priority_inputs"] = inputs
            for h, tags in e.get("set_tags", {}).items():
                hyps[h]["tags"] = tags
    return hyps, prereg, results


def withdrawn(events):
    return {x for e in events if e["kind"] == "decision" for x in e.get("withdraw", [])}


def reviews(events, experiment):
    return [e for e in events if e["kind"] == "review" and e["experiment"] == experiment]


def review_gate(events, experiment):
    """Problems blocking a run: it needs an approving mechanical review AND an approving independent critic,
    and no reviewer's latest verdict may be revise/reject."""
    latest = {}
    for e in reviews(events, experiment):
        latest[e["reviewer"]] = e["verdict"]
    problems = [f"{r} says {v}" for r, v in latest.items() if v != "approve"]
    if latest.get("mechanical") != "approve":
        problems.append("no approving mechanical review (python -B research_critic_v1.py E-xxx --record)")
    if not any(v == "approve" for r, v in latest.items() if r != "mechanical"):
        problems.append("no approving independent critic review")
    return problems


def render(events, path=RENDERED):
    hyps, prereg, results = state(events)
    windows = used_windows(events)
    problems = verify(events)

    def reuse_count(p):
        before = events[:events.index(p)]
        return sum(1 for w in p["evaluation_windows"] for u, _ in used_windows(before) if overlaps(w, u))

    out = ["# XM 研究記憶（自動產生，請勿手改）", "",
           f"由 `research_memory_v1.py render` 從 `research_memory/ledger.jsonl` 產生。事件數 {len(events)}；"
           f"帳本完整性：{'OK' if not problems else 'DAMAGED ' + '; '.join(problems[:3])}。", "",
           "分類規則：**已驗證觀察**＝有證據且未撤回；**假說**＝尚未有結論；**失敗**＝被推翻或中止；"
           "**資料重複使用**＝評估區間在登錄前已被看過，只能當開發證據。", ""]

    out += ["## 1. 已驗證的觀察", ""]
    for e in events:
        if e["kind"] == "observation" and e["status"] == "verified":
            out.append(f"- **{e['id']}** {e['statement']}  \n  證據：{e['evidence']}")
    import research_planner_v1 as planner  # late import: the planner uses this module's state()
    out += ["", "## 2. 尚未驗證的假說（依 Research Priority Engine 排序）", "",
            "分數綜合資訊增益、與過去研究的差異、可反證性、資料品質與取得成本、經濟價值、工程簡單度，"
            "扣掉資料探勘風險與同族群已試次數。失敗但能排除重要可能性的實驗同樣得分。公式見 `research_planner_v1.py`。", "",
            "| ID | 假說 | 族群 | 狀態 | 優先分 | 推翻條件 |", "|---|---|---|---|---|---|"]
    for h, score in planner.rank(events):
        out.append(f"| {h['id']} | {h['title']} | {h['family']} | {h['status']} | {score} | "
                   f"{h['falsified_if'].replace('|', '&#124;')} |")
    out += ["", "## 2b. 證據狀態分類（結論不得超出這個範圍）", ""]
    for status, label in STATUS_LABEL.items():
        names = [f"{h['id']} {h['title']}" for h in hyps.values() if h["status"] == status]
        out.append(f"- **{label}**：" + ("；".join(names) if names else "（無）"))
    out += ["", "## 3. 失敗的實驗與被推翻的假說", ""]
    for h in hyps.values():
        if h["status"] in ("refuted", "closed"):
            out.append(f"- **{h['id']}** {h['title']}（{h['status']}）")
    for exp, r in results.items():
        if r["outcome"] in ("refuted", "aborted"):
            out.append(f"- 實驗 **{exp}**（{prereg[exp]['hypothesis']}）：{r['summary']}  \n"
                       f"  失敗原因：`{r['failure_cause']}`；學到：{r['lessons']}")
    out += ["", "## 4. 受資料重複使用影響的結論", ""]
    for exp, r in results.items():
        n = reuse_count(prereg[exp])
        if r["evidence_class"] == "reuse_contaminated" or n:
            out.append(f"- 實驗 **{exp}**：{r['outcome']}（證據等級 `{r['evidence_class']}`，登錄前已有 {n} 次使用重疊）"
                       f"：{r['summary']}")
    out += ["", "## 5. 已登錄、尚未有結果的實驗", ""]
    gone = withdrawn(events)
    for exp, p in prereg.items():
        if exp in results:
            continue
        if exp in gone:
            out.append(f"- ~~{exp}~~ 已作廢（執行前發現設計缺陷，見決策紀錄）")
            continue
        gate = review_gate(events, exp)
        out.append(f"- **{exp}** → {p['hypothesis']}（{p['role']}）：`{p['module']}`；"
                   f"審查：{'通過，可執行' if not gate else '未通過：' + '；'.join(gate)}  \n  判定規則：{p['decision_rule']}")
        for r in reviews(events, exp):
            out.append(f"  - 審查 `{r['reviewer']}` → **{r['verdict']}**：" + "；".join(r["findings"]))
    out += ["", "## 6. 所有實驗結果與下一個問題", ""]
    for exp, r in results.items():
        out.append(f"- **{exp}**（{prereg[exp]['hypothesis']}）`{r['outcome']}` / `{r['evidence_class']}`："
                   f"{r['summary']}  \n  下一個問題：{r['next_questions']}"
                   + (f"  \n  紀錄：`{r['run_dir']}`" if r.get("run_dir") else ""))
    out += ["", "## 7. 決策紀錄", ""]
    for e in events:
        if e["kind"] == "decision":
            out.append(f"- **{e['id']}** {e['statement']}：{e['rationale']}")
    out += ["", "## 8. 資料區間使用帳（越多次越不能當獨立驗證）", "", "| 區間 | 使用者 |", "|---|---|"]
    for w, by in windows:
        out.append(f"| {w[0]} → {w[1]} | {by} |")
    out += ["", "## 9. 研究空間覆蓋與自動產生的候選問題", "", *planner.coverage_markdown(events)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("render")
    a = sub.add_parser("append")
    a.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "verify":
        problems = verify(read())
        print("OK" if not problems else "\n".join(problems))
        return 1 if problems else 0
    if args.cmd == "append":
        data = json.loads(args.file.read_text(encoding="utf-8"))
        append(data if isinstance(data, list) else [data])
    render(read())
    print(f"rendered {RENDERED.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
