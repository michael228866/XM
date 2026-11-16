import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import research_eval_v1 as ev
import research_memory_v1 as rm


def hyp(i="H-1", family="f"):
    return {"kind": "hypothesis", "id": i, "title": "t", "family": family, "mechanism": "m", "novelty": "n",
            "information_needed": "i", "falsified_if": "x", "data_mining_risk": "low", "status": "open"}


def prereg(root, i="E-1", role="exploratory", window=("2020-01-01", "2021-01-01")):
    code = root / "exp.py"
    if not code.exists():
        code.write_text("x = 1\n", encoding="utf-8")
    return {"kind": "preregistration", "id": i, "hypothesis": "H-1", "role": role, "module": "exp",
            "code_files": {"exp.py": rm.file_sha256(code)}, "spec": {"a": 1},
            "evaluation_windows": [list(window)], "decision_rule": "r"}


def result(exp="E-1", evidence="development"):
    return {"kind": "result", "experiment": exp, "outcome": "refuted", "evidence_class": evidence, "summary": "s",
            "failure_cause": "information", "lessons": "l", "next_questions": "q"}


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ledger = self.dir / "ledger.jsonl"

    def add(self, *events):
        return rm.append(list(events), ledger=self.ledger, root=self.dir)

    def test_chain_detects_edits(self):
        self.add(hyp(), prereg(self.dir), result())
        self.assertEqual(rm.verify(rm.read(self.ledger)), [])
        lines = self.ledger.read_text(encoding="utf-8").splitlines()
        edited = json.loads(lines[2])
        edited["outcome"] = "supported"
        lines[2] = json.dumps(edited)
        self.ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertTrue(rm.verify(rm.read(self.ledger)))
        with self.assertRaises(ValueError):
            self.add(hyp("H-2"))

    def test_research_rules(self):
        self.add(hyp(), {"kind": "data_usage", "window": ["2019-01-01", "2024-01-01"], "by": "old", "note": "n"})
        with self.assertRaises(ValueError):  # confirmatory on a window already used
            self.add(prereg(self.dir, role="confirmatory"))
        self.add(prereg(self.dir, "E-1"), prereg(self.dir, "E-2", "confirmatory", ("2026-12-01", "2027-06-01")))
        with self.assertRaises(ValueError):  # result without preregistration
            self.add(result("E-9"))
        with self.assertRaises(ValueError):  # reused window cannot be clean forward evidence
            self.add(result("E-1", "clean_forward"))
        self.add(result("E-1"))
        with self.assertRaises(ValueError):  # one result per experiment
            self.add(result("E-1"))
        stale = prereg(self.dir, "E-3")
        (self.dir / "exp.py").write_text("x = 2\n", encoding="utf-8")
        with self.assertRaises(ValueError):  # code changed after the hash was taken
            self.add(stale)

    def test_render_sections(self):
        self.add(hyp(), prereg(self.dir), result(),
                 {"kind": "observation", "id": "O-1", "statement": "s", "evidence": "e", "status": "verified"})
        out = self.dir / "M.md"
        rm.render(rm.read(self.ledger), out)
        text = out.read_text(encoding="utf-8")
        for section in ("已驗證的觀察", "失敗的實驗", "受資料重複使用影響", "資料區間使用帳"):
            self.assertIn(section, text)


def frame(n=4000, drift=0.02, seed=0):
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(drift, 1, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"TIME_DT": pd.date_range("2020-01-01", periods=n, freq="h"), "OPEN": open_, "CLOSE": close,
                         "HIGH": np.maximum(open_, close) + 0.3, "LOW": np.minimum(open_, close) - 0.3,
                         "SPREAD": 0.0, "ATR": 1.0})


class EvalTests(unittest.TestCase):
    def test_random_entries_show_no_timing_but_perfect_foresight_does(self):
        f = frame()
        rng = np.random.default_rng(1)
        rand = ev.evaluate(f, rng.random(len(f)) < 0.05, "long", 3.0, 1.5, 24)
        self.assertLess(abs(rand["timing_residual_mean_r"]), 0.15)
        r, _ = ev.trade_returns(f, "long", 3.0, 1.5, 24)
        cheat = ev.evaluate(f, np.nan_to_num(r) > 0, "long", 3.0, 1.5, 24)
        self.assertGreater(cheat["timing_residual_mean_r"], 0.5)
        self.assertGreater(cheat["timing_residual_ci_month_bootstrap"][0], 0)

    def test_sidak_grows_with_trials(self):
        self.assertLess(ev.sidak_p(2.0, 1), ev.sidak_p(2.0, 20))


class InformationHorizonTests(unittest.TestCase):
    def test_weekly_panel_respects_availability_and_long_flat_costs(self):
        import exp_information_horizon_v1 as ih
        days = pd.bdate_range("2020-01-06", periods=200)
        d1 = pd.DataFrame({"TIME_DT": days, "OPEN": np.linspace(100, 120, 200), "CLOSE": np.linspace(100, 120, 200)})
        ns = (pd.DatetimeIndex(days) + pd.Timedelta(hours=21)).asi8  # each rate is known the evening of its date
        treasury = pd.DataFrame({"real_10y": np.arange(200.0), "availability_ns": ns})
        cot = pd.DataFrame({"availability_time_utc": [str(d + pd.Timedelta(days=4)) + "+00:00" for d in days[::5]],
                            "COT_MM_NET_PCT_OI": np.arange(len(days[::5]), dtype=float)})
        panel = ih.weekly_panel(d1, treasury, cot)
        self.assertTrue((panel["TIME_DT"].dt.dayofweek == 0).all())
        # Monday 00:00 decision sees Friday's rate (index i-1), never Monday's own.
        first = panel.iloc[0]
        monday_index = list(days).index(first["TIME_DT"])
        self.assertEqual(first["REAL10Y_CHG_20D"], 20.0)
        self.assertGreaterEqual(monday_index - 1, 20)
        net, always, share = ih.long_flat(panel, "REAL10Y_CHG_20D", -1, cost_frac=0.001, swap_frac_per_night=-0.0001)
        self.assertTrue(0 <= share <= 1)
        self.assertEqual(net.iloc[0], 0.0)  # first week has no past median, so it stays flat and pays nothing
        self.assertLess(always.iloc[0], panel["ret"].iloc[0])  # always-long pays swap


class SessionAndVolTests(unittest.TestCase):
    def test_session_returns_cover_only_their_hours(self):
        import exp_session_structure_v1 as ss
        times = pd.date_range("2021-03-01", periods=48, freq="h")
        h1 = pd.DataFrame({"TIME_DT": times, "OPEN": 100.0 + np.arange(48), "CLOSE": 101.0 + np.arange(48),
                           "SPREAD": 0})
        s = ss.session_returns(h1, 0.0001, 0.01)
        asia = s[s["session"] == "ASIA"].iloc[0]
        self.assertAlmostEqual(asia["ret"], 109 / 101 - 1)  # bars 01..08: open of 01, close of 08
        self.assertEqual(asia["cost"], 0.0001)
        self.assertEqual(len(s), 8)  # 2 days x 4 sessions; the 00 bar is in none

    def test_vol_managed_uses_yesterdays_sigma_and_pays_costs(self):
        import exp_vol_managed_v1 as vm
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
        d1 = pd.DataFrame({"TIME_DT": pd.bdate_range("2020-01-01", periods=300), "CLOSE": close})
        out = vm.vol_managed(d1, 0.94, 2.0, 0.0, 0.0)
        bumped = d1.copy()
        bumped.loc[250, "CLOSE"] *= 1.2  # a shock at day 250 cannot change day 250's own weight
        self.assertAlmostEqual(vm.vol_managed(bumped, 0.94, 2.0, 0.0, 0.0)["w"][250], out["w"][250])
        self.assertTrue((out["w"].dropna() <= 2.0).all())
        swap = vm.vol_managed(d1, 0.94, 2.0, 0.0, -0.05)
        self.assertTrue((swap["constant"].iloc[1:] < out["constant"].iloc[1:]).all())


class V2DesignTests(unittest.TestCase):
    def test_session_excess_removes_proportional_drift(self):
        import exp_session_structure_v2 as ss
        times = pd.date_range("2021-03-01", periods=24 * 40, freq="h")
        close = 100 * np.exp(np.arange(len(times)) * 0.001)  # steady uptrend: no session is special
        h1 = pd.DataFrame({"TIME_DT": times, "OPEN": np.r_[close[0], close[:-1]], "CLOSE": close, "SPREAD": 0})
        t = ss.day_table(h1, 0.01, 0.0)
        s = ss.strategies(t, 0.01, 0.0)
        self.assertTrue((s["ASIA_long"] > 0).all())  # raw long looks good in a rising market
        for name in ss.SESSIONS:
            self.assertLess(abs(s[f"{name}_long_excess"].mean()), 2e-4)  # but carries no excess over its share
        swap = ss.strategies(t, 0.01, -0.05)
        self.assertTrue((swap["ALWAYS_LONG"] < s["ALWAYS_LONG"]).all())
        self.assertTrue((swap["INTRADAY_LONG"] == s["INTRADAY_LONG"]).all())  # intraday pays no swap

    def test_vol_managed_v2_controls(self):
        import exp_vol_managed_v2 as vm
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)))
        w = vm.managed_weights(close, 0.94, 2.0)
        bumped = close.copy()
        bumped[300] *= 1.2
        self.assertAlmostEqual(vm.managed_weights(bumped, 0.94, 2.0)[300], w[300])
        months = pd.Series(pd.bdate_range("2020-01-01", periods=400)).dt.to_period("M")
        sh = vm.month_shuffle(w.fillna(1.0), months, np.random.default_rng(1))
        self.assertEqual(sorted(np.round(sh, 9)), sorted(np.round(w.fillna(1.0).to_numpy(), 9)))
        ret = close.pct_change().fillna(0)
        net = vm.net_returns(ret, pd.Series(0.0, index=ret.index), np.full(400, 0.5), 0.001)
        self.assertAlmostEqual(net[0], -0.0005)  # constant pays the spread once, on entry
        self.assertAlmostEqual(net[5], 0.5 * ret[5])


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ledger = self.dir / "ledger.jsonl"

    def add(self, *events):
        return rm.append(list(events), ledger=self.ledger, root=self.dir)

    def test_review_gate_and_withdraw(self):
        p = {**prereg(self.dir), "author": "researcher"}
        self.add(hyp(), p)
        events = rm.read(self.ledger)
        self.assertEqual(len(rm.review_gate(events, "E-1")), 2)
        with self.assertRaises(ValueError):  # the author cannot approve their own design
            self.add({"kind": "review", "experiment": "E-1", "reviewer": "researcher", "verdict": "approve",
                      "findings": ["ok"]})
        self.add({"kind": "review", "experiment": "E-1", "reviewer": "mechanical", "verdict": "approve", "findings": ["ok"]},
                 {"kind": "review", "experiment": "E-1", "reviewer": "critic", "verdict": "revise", "findings": ["leak"]})
        self.assertIn("critic says revise", rm.review_gate(rm.read(self.ledger), "E-1"))
        self.add({"kind": "review", "experiment": "E-1", "reviewer": "critic", "verdict": "approve", "findings": ["fixed"]})
        self.assertEqual(rm.review_gate(rm.read(self.ledger), "E-1"), [])
        self.add({"kind": "decision", "id": "D-1", "statement": "s", "rationale": "r", "withdraw": ["E-1"],
                  "set_status": {"H-1": "underpowered"}})
        events = rm.read(self.ledger)
        self.assertEqual(rm.withdrawn(events), {"E-1"})
        self.assertEqual(rm.state(events)[0]["H-1"]["status"], "underpowered")
        with self.assertRaises(ValueError):
            self.add({"kind": "decision", "id": "D-2", "statement": "s", "rationale": "r", "withdraw": ["E-9"]})

    def test_mechanical_critic_blocks_incomplete_design(self):
        import research_critic_v1 as rc
        p = prereg(self.dir)
        self.add(hyp(), p)
        blocking, _ = rc.critique(rm.read(self.ledger)[1], rm.read(self.ledger), root=self.dir)
        self.assertTrue(any(b.startswith("controls") for b in blocking))
        good = {**prereg(self.dir, "E-2"), "training_window_note": "no model", "claim_scope": "only X",
                "decision_rule": "pass if Sidak p < 0.05 over 8 trials",
                "spec": {"controls": ["a"], "directional_benchmark": "always long", "trials_declared": 8,
                         "cost_multipliers": [1, 2], "cost_note": "spread and swap", "power_note": "n=1000",
                         "metrics": ["net_mean", "sharpe", "calmar", "max_dd"],
                         "data_windows": [{"window": ["2020-01-01", "2021-01-01"], "uses": ["h1_or_finer_price"]}]}}
        events = self.add(good)
        self.assertEqual(rc.critique(events[-1], events, root=self.dir)[0], [])

    def test_data_boundaries_are_checked_mechanically(self):
        import research_critic_v1 as rc
        p = lambda windows, ev=("2019-01-01", "2020-01-01"), **extra: {
            **prereg(self.dir, window=ev), "spec": {"data_windows": windows, **extra}}
        spread_only = [{"window": ["2018-10-01", "2019-01-01"], "uses": ["spread_field"]}]
        joint = [{"window": ["2018-10-01", "2020-01-01"], "uses": ["spread_field", "h1_or_finer_price"]}]
        self.assertEqual(rc.boundary_problems(p(spread_only + [{"window": ["2019-01-01", "2020-01-01"],
                                                                "uses": ["h1_or_finer_price"]}])), [])
        self.assertTrue(rc.boundary_problems(p(joint)))  # joint use defaults to the later 2019-01-01 boundary
        self.assertTrue(rc.boundary_problems(p(spread_only, ev=("2018-10-01", "2020-01-01"),)) == [])
        self.assertTrue(rc.boundary_problems(p([{"window": ["2018-12-01", "2020-01-01"], "uses": ["h1_or_finer_price"]}])))
        self.assertTrue(rc.boundary_problems(p([{"window": ["2019-01-01", "2020-01-01"], "uses": ["tick_magic"]}])))
        self.assertTrue(rc.boundary_problems(p(None)))
        self.assertEqual(rc.boundary_problems(p(joint, boundary_exception="why")), [])
        self.assertEqual(rc.boundary_problems(p([{"window": ["2014-01-01", "2020-01-01"], "uses": ["daily_or_coarser_price"]}],
                                                ev=("2014-01-01", "2020-01-01"))), [])

    def test_planner_ranks_by_research_value(self):
        import research_planner_v1 as rp
        inputs = dict.fromkeys(rm.PRIORITY_KEYS, 3)
        self.add({**hyp("H-1", "a"), "priority_inputs": {**inputs, "info_gain": 5}},
                 {**hyp("H-2", "b"), "priority_inputs": {**inputs, "mining_risk": 5},
                  "tags": {"information": ["implied_vol"], "target": ["volatility"]}})
        ranked = rp.rank(rm.read(self.ledger))
        self.assertEqual([h["id"] for h, _ in ranked], ["H-1", "H-2"])
        self.assertNotIn(("implied_vol", "volatility"), rp.proposals(rm.read(self.ledger), limit=100))
        with self.assertRaises(ValueError):
            self.add({**hyp("H-3"), "priority_inputs": {"info_gain": 9}})


class PriorityEngineTests(unittest.TestCase):
    """Evidence-failure propagation: upstream data/premise problems reach every dependent, repairs bring them back."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ledger = self.dir / "ledger.jsonl"
        inputs = dict.fromkeys(rm.PRIORITY_KEYS, 3)
        self.add({**hyp("H-1", "a"), "priority_inputs": {**inputs, "info_gain": 5, "economic_value": 5}},
                 {**hyp("H-2", "b"), "priority_inputs": inputs}, {**hyp("H-3", "c"), "priority_inputs": inputs},
                 {**hyp("H-9", "d"), "status": "refuted"}, prereg(self.dir),
                 result(), {"kind": "observation", "id": "O-1", "statement": "s", "evidence": "E-1", "status": "verified"})

    def add(self, *events):
        return rm.append(list(events), ledger=self.ledger, root=self.dir)

    def decide(self, i, **fields):
        return self.add({"kind": "decision", "id": i, "statement": "s", "rationale": "r", **fields})

    def graph(self):
        # H-1 -premise-> O-1 -evidence-> E-1 -data-> data:old (three hops); H-2 -premise-> premise:p; H-9 refuted on E-1
        self.decide("D-1", set_depends={
            "data:old": [], "premise:p": [], "E-1": [{"on": "data:old", "type": "data"}],
            "O-1": [{"on": "E-1", "type": "evidence"}], "H-1": [{"on": "O-1", "type": "premise"}],
            "H-2": [{"on": "premise:p", "type": "premise"}], "H-9": [{"on": "E-1", "type": "evidence"}],
            "model:m": [{"on": "data:old", "type": "data"}]})

    def ids(self, events):
        import research_planner_v1 as rp
        return [r["id"] for r in rp.ranking(events)]

    def test_upstream_failure_reaches_all_dependents_without_touching_results(self):
        import research_planner_v1 as rp
        self.graph()
        before = self.ledger.read_bytes()
        events = self.decide("D-2", flag={"data:old": {"state": "defective", "evidence": ["O-1"], "note": "bad"},
                                          "premise:p": {"state": "invalid", "evidence": ["O-1"]}})
        a = rp.assess(events)
        self.assertEqual(a["H-1"]["state"], "needs_revalidation")  # through three hops, not refuted
        self.assertEqual(a["H-1"]["reasons"][-1]["path"], ["H-1", "O-1", "E-1", "data:old"])
        self.assertEqual(a["H-2"]["state"], "premise_invalid")
        rows = {r["id"]: r for r in rp.ranking(events)}
        self.assertNotIn("H-2", rows)  # top-tier inputs cannot rank a hypothesis whose premise is false
        self.assertEqual(rows["H-1"]["base"], 30.5)  # original score kept beside the adjusted one
        self.assertEqual(rows["H-1"]["score"], 30.5 - 2 - rp.CREDIBILITY_PENALTY)  # data_quality 3 -> 1, credibility
        self.assertTrue(all(c for _, _, c in rows["H-1"]["adjust"]))  # every adjustment cites evidence
        self.assertEqual(rows["data:old"]["kind"], "repair")
        self.assertIn("model:m", rows["data:old"]["title"])
        sugg = {n: s for n, _, s, _, _ in rp.suggestions(events)}
        self.assertEqual((sugg["H-1"], sugg["H-2"]), ("needs_revalidation", "premise_invalid"))
        self.assertEqual((sugg["H-9"], sugg["E-1"], sugg["model:m"]), ("annotate_only",) * 3)
        hyps, _, results = rm.state(events)
        self.assertEqual(hyps["H-9"]["status"], "refuted")  # nothing approved, nothing rewritten
        self.assertEqual(hyps["H-1"]["status"], "open")
        self.assertEqual(results["E-1"]["outcome"], "refuted")
        self.assertTrue(self.ledger.read_bytes().startswith(before))
        rm.render(events, path=self.dir / "m.md")
        self.assertEqual(rm.verify(rm.read(self.ledger)), [])
        log = dict(rp.rerank_log(events))
        self.assertTrue(any(m.startswith("H-2 ") and "未列入" in m for m in log["D-2"]))

    def test_repair_brings_hypotheses_back(self):
        import research_planner_v1 as rp
        self.graph()
        self.decide("D-2", flag={"data:old": {"state": "defective", "evidence": ["O-1"]},
                                 "premise:p": {"state": "invalid", "evidence": ["O-1"]}},
                    set_status={"H-2": "premise_invalid", "H-1": "needs_revalidation"})
        events = rm.read(self.ledger)
        self.assertNotIn("H-2", self.ids(events))
        # E-1 re-checked on clean data shields what rests on it; the premise is restored
        events = self.decide("D-3", flag={"E-1": {"state": "ok", "evidence": ["O-1"]},
                                          "premise:p": {"state": "ok", "evidence": ["O-1"]}})
        rows = {r["id"]: r for r in rp.ranking(events)}
        self.assertEqual(rows["H-1"]["score"], rows["H-1"]["base"])
        self.assertIn("H-2", rows)
        self.assertTrue(rows["data:old"]["title"].endswith("model:m"))  # only the model still rests on it
        sugg = {n: s for n, _, s, _, _ in rp.suggestions(events)}
        self.assertEqual((sugg["H-1"], sugg["H-2"]), ("open", "open"))  # suggested, not applied
        self.assertEqual(rm.state(events)[0]["H-2"]["status"], "premise_invalid")
        events = self.decide("D-4", set_depends={"H-3": [{"on": "data:old", "type": "data"}]})
        self.assertEqual(rp.assess(events)["H-3"]["state"], "needs_revalidation")
        events = self.decide("D-5", set_depends={"H-3": [{"on": "data:new", "type": "data"}], "data:new": []})
        self.assertEqual(rp.assess(events)["H-3"]["state"], "ok")  # rebased onto clean data

    def test_cycles_missing_and_several_failed_premises(self):
        import research_planner_v1 as rp
        events = self.decide("D-1", set_depends={
            "H-1": [{"on": "H-2", "type": "premise"}], "H-2": [{"on": "H-1", "type": "premise"}],
            "H-3": [{"on": "premise:a", "type": "premise"}, {"on": "premise:b", "type": "premise"},
                    {"on": "data:never_declared", "type": "data"}],
            "premise:a": [], "premise:b": []})
        a = rp.assess(events)  # terminates on the cycle
        self.assertEqual(a["H-1"]["state"], "needs_revalidation")
        self.assertIn(["H-1", "H-2", "H-1"], a["H-1"]["cycles"])
        self.assertEqual(a["H-3"]["missing"], [["H-3", "data:never_declared"]])
        self.assertEqual(a["H-3"]["state"], "needs_revalidation")
        events = self.decide("D-2", flag={"premise:a": {"state": "invalid", "evidence": ["O-1"]},
                                          "premise:b": {"state": "invalid", "evidence": ["E-1"]}})
        a = rp.assess(events)
        self.assertEqual(a["H-3"]["state"], "premise_invalid")
        self.assertEqual({r["node"] for r in a["H-3"]["reasons"]}, {"premise:a", "premise:b"})
        # repairing one of two failed premises is not enough
        events = self.decide("D-3", flag={"premise:a": {"state": "ok", "evidence": ["O-1"]}})
        self.assertEqual(rp.assess(events)["H-3"]["state"], "premise_invalid")
        # a premise on a refuted hypothesis fails through research outcomes alone, no flag needed
        events = self.decide("D-4", set_depends={"H-2": [{"on": "H-9", "type": "premise"}]})
        self.assertEqual(rp.assess(events)["H-2"]["state"], "premise_invalid")
        self.assertEqual(rp.assess(events)["H-1"]["state"], "premise_invalid")  # H-1 -> H-2 -> H-9, all premises

    def test_bad_graph_inputs_are_rejected(self):
        for fields in ({"set_depends": {"H-1": [{"on": "O-1", "type": "hunch"}]}},
                       {"set_depends": {"nonsense": []}},
                       {"flag": {"data:x": {"state": "broken", "evidence": ["O-1"]}}},
                       {"flag": {"data:x": {"state": "defective", "evidence": ["O-404"]}}},
                       {"flag": {"data:x": {"state": "defective"}}}):
            with self.assertRaises(ValueError):
                self.decide("D-9", **fields)


def synthetic_h1(years=6, seed=0):
    """GOLD-like H1 bars with persistent (GARCH-ish) volatility, 24 bars per weekday."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-01", periods=260 * years)
    vol, rows, price = 0.002, [], 1300.0
    for d in days:
        vol = np.sqrt(0.00000005 + 0.9 * vol ** 2 + 0.08 * (vol * rng.normal()) ** 2)
        for hr in range(24):
            r = vol * rng.normal()
            o, price = price, price * np.exp(r)
            rows.append((d + pd.Timedelta(hours=hr), o, max(o, price) * 1.0001, min(o, price) * 0.9999, price))
    return pd.DataFrame(rows, columns=["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE"])


class VolForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import exp_vol_forecast_v1 as vf
        cls.vf, cls.h1 = vf, synthetic_h1()
        cls.spec = {"estimation_window": ["2014-01-01", "2016-01-01"],
                    "windows": [["2016-01-01", "2019-01-01"], ["2019-01-01", "2099-01-01"]], "min_years_better": 2}

    def test_partial_days_dropped_and_targets_aligned(self):
        vf = self.vf
        h1 = self.h1[~((self.h1["TIME_DT"].dt.normalize() == pd.Timestamp("2014-03-05"))
                       & (self.h1["TIME_DT"].dt.hour >= 10))]
        days = vf.daily_rv(h1)
        self.assertNotIn(pd.Timestamp("2014-03-05"), days.index)
        x = vf.design(days["rv"])
        i = 30
        self.assertAlmostEqual(x["y5"].iloc[i], days["rv"].iloc[i + 1:i + 6].mean())
        self.assertAlmostEqual(x["rv_w"].iloc[i], days["rv"].iloc[i - 4:i + 1].mean())

    def test_forecasts_never_see_the_future(self):
        vf = self.vf
        rv = vf.daily_rv(self.h1)["rv"]
        k = 700  # shock a day after the estimation window
        bumped = rv.copy()
        bumped.iloc[k] *= 50
        est = (pd.Timestamp("2014-01-01"), pd.Timestamp("2016-01-01"))
        out = []
        for series in (rv, bumped):
            x = vf.design(series)
            paths = {lam: vf.ewma_path(series, lam) for lam in vf.LAMBDAS}
            fz, _ = vf.frozen(x, series, paths, 5, est)
            rc = vf.recursive(x, series, paths, 5, pd.Timestamp("2016-01-01"), est[0])
            out.append((fz, rc))
        np.testing.assert_allclose(out[0][0].iloc[:k].to_numpy(), out[1][0].iloc[:k].to_numpy())
        np.testing.assert_allclose(out[0][1].iloc[:k].dropna().to_numpy(), out[1][1].iloc[:k].dropna().to_numpy())
        self.assertFalse(np.allclose(out[0][0].iloc[k:].to_numpy(), out[1][0].iloc[k:].to_numpy()))

    def test_end_to_end_on_persistent_volatility(self):
        results, params, _ = self.vf.evaluate(self.h1, self.spec)
        r = results["FROZEN|h1|eval_2016_2018"] if "FROZEN|h1|eval_2016_2018" in results else results["FROZEN|h1|eval_2018_2023"]
        self.assertLess(r["HAR_vs_HIST"]["qlike_ratio"], 1.0)  # persistent vol: conditional beats unconditional
        self.assertIn(self.vf.decide(results, self.spec), ("supported", "refuted", "not_significant"))
        self.assertTrue(0.8 <= params[1]["lambda"] <= 0.99)


class GvzIncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import exp_gvz_increment_v1 as gi
        import exp_vol_forecast_v1 as vf
        import research_data_gvz_v1 as gd
        cls.gi, cls.gd, cls.h1 = gi, gd, synthetic_h1()
        cls.rv = vf.daily_rv(cls.h1)["rv"]
        cls.x = vf.design(cls.rv)
        cls.spec = {"estimation_window": ["2014-01-01", "2016-01-01"],
                    "windows": [["2016-01-01", "2019-01-01"], ["2019-01-01", "2099-01-01"]],
                    "min_years_better": 2, "materiality_pct": 2}

    def gvz_from(self, values):
        """GVZ dated the calendar day before each server day, i.e. the newest value lag 1 allows."""
        return pd.Series(values, index=self.rv.index - pd.Timedelta(days=1)).dropna()

    def test_align_uses_only_earlier_dates_and_drops_stale(self):
        gvz = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06",
                                                                          "2020-01-20"]))
        days = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-14"])
        a, a2 = self.gd.align(gvz, days), self.gd.align(gvz, days, extra_lag=1)
        self.assertEqual(a["gvz"].tolist()[:3], [10.0, 11.0, 12.0])  # Friday's value for Monday, never same day
        self.assertTrue(np.isnan(a["gvz"].iloc[3]))  # 8 days old -> missing
        self.assertEqual(a2["gvz"].tolist()[1:3], [10.0, 11.0])

    def test_fit_recovers_coefficients(self):
        rng = np.random.default_rng(1)
        X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
        e = (rng.random(4000) < 1 / (1 + np.exp(-(X @ [-1.0, 2.0])))).astype(float)
        np.testing.assert_allclose(self.gi.fit("logit", X, e)[0], [-1.0, 2.0], atol=0.15)
        y = np.exp(X @ [0.5, 0.3]) * rng.exponential(size=4000)  # E[y | x] = exp(Xb): QLIKE target
        np.testing.assert_allclose(self.gi.fit("qlike", X, y)[0], [0.5, 0.3], atol=0.08)

    def test_informative_gvz_is_credited_and_noise_is_not(self):
        rng = np.random.default_rng(2)
        oracle = self.gvz_from(100 * np.sqrt(252 * self.x["y5"]) * np.exp(0.1 * rng.normal(size=len(self.rv))))
        res, _, _ = self.gi.evaluate(self.h1, oracle, self.spec)
        self.assertLess(res["general|eval_2018_2023"]["A_vs_R"]["ratio"], 0.9)
        self.assertEqual(self.gi.decide(res, self.spec)[1]["general"], "supported")
        # synthetic burn-in leaves no extreme events before 2016: separation must end as underpowered, not refuted
        self.assertEqual(self.gi.decide(res, self.spec)[1]["extreme"], "underpowered")
        for seed in range(3):  # noise must not be credited in any question (critic B1 caught extreme/regime_up)
            noise = self.gvz_from(np.exp(np.random.default_rng(seed + 10).normal(3, 0.3, size=len(self.rv))))
            res, _, _ = self.gi.evaluate(self.h1, noise, self.spec)
            overall, verdicts = self.gi.decide(res, self.spec)
            self.assertNotIn("supported", verdicts.values(), (seed, verdicts))
            self.assertIn(overall, ("refuted", "inconclusive"))

    def test_gvz_on_or_after_origin_does_not_change_inputs(self):
        gvz = self.gvz_from(np.full(len(self.rv), 20.0))
        k = 700
        bumped = gvz.copy()
        bumped[bumped.index >= self.rv.index[k] - pd.Timedelta(days=1)] *= 3  # values dated t-1 onwards
        qa = self.gi.questions(self.rv, self.x, gvz)[0]["general"]["regs"]["A"]
        qb = self.gi.questions(self.rv, self.x, bumped)[0]["general"]["regs"]["A"]
        np.testing.assert_allclose(qa.iloc[:k].dropna().to_numpy(), qb.iloc[:k].dropna().to_numpy())
        self.assertFalse(np.allclose(qa.iloc[k:].dropna().to_numpy(), qb.iloc[k:].dropna().to_numpy()))


class VolSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import exp_vol_sizing_v1 as vs
        h1 = synthetic_h1(years=6)
        h1["SPREAD"] = 30.0
        h1["ATR"] = (h1["HIGH"] - h1["LOW"]).rolling(14).mean().bfill()
        cls.vs, cls.frame = vs, h1.reset_index(drop=True)
        cls.signal = np.random.default_rng(3).random(len(h1)) < 0.03
        cls.spec = {"tp": 3.0, "sl": 1.5, "hold": 48, "risk_per_trade": 0.005, "leverage_cap": 5.0,
                    "calibration_window": ["2014-03-01", "2017-01-01"],
                    "labelled_windows": {"in_sample_2018_2023": ["2014-03-01", "2017-01-01"],
                                         "oos_2024_on": ["2017-01-01", "2099-01-01"]},
                    "seed": 0, "bootstrap_draws": 200, "min_trades": 150, "min_months": 24, "min_regime_days": 60,
                    "materiality_sharpe": 0.2, "materiality_mdd_cut": 0.10}

    def test_fixed_risk_trade_pnl_is_risk_times_r(self):
        vs = self.vs
        t = vs.trades(self.frame, self.signal, self.spec, 1.0)
        sigma, _ = vs.entry_sigma(self.frame)
        calib = vs.calibrate(self.frame, self.signal, self.spec, sigma, pd.Timestamp("2014-03-01"), pd.Timestamp("2017-01-01"))
        w, _ = vs.weights(t, sigma, self.spec, calib)
        _, per_trade, _ = vs.daily_pnl(self.frame, t, w["FIXED_RISK"], 0.0, 1.0)
        capped = w["FIXED_RISK"].to_numpy() >= self.spec["leverage_cap"]
        np.testing.assert_allclose(per_trade[~capped], 0.005 * t["r"].to_numpy()[~capped], rtol=1e-9)
        self.assertGreater(len(t), 100)

    def test_entry_sigma_uses_only_earlier_days(self):
        vs = self.vs
        base, _ = vs.entry_sigma(self.frame)
        day = self.frame["TIME_DT"].dt.normalize()
        k = day.unique()[400]
        bumped = self.frame.copy()
        rows = np.flatnonzero((day == k).to_numpy())
        bumped.loc[rows[10]:rows[12], "CLOSE"] *= 1.05  # a shock inside day k
        new, _ = vs.entry_sigma(bumped)
        through_k = (day <= k).to_numpy()
        np.testing.assert_allclose(base[through_k], new[through_k], equal_nan=True)
        self.assertFalse(np.allclose(base[(day > k).to_numpy()][:24], new[(day > k).to_numpy()][:24]))

    def test_vol_matched_metrics_are_scale_free(self):
        vs = self.vs
        ret = pd.Series(np.random.default_rng(4).normal(1e-4, 1e-3, 600), index=pd.bdate_range("2020-01-01", periods=600))
        a = vs.arm_metrics(ret, 1e-3, np.ones(5), np.ones(5), 0.0)
        b = vs.arm_metrics(ret * 2, 1e-3, np.ones(5), np.ones(5), 0.0)
        self.assertAlmostEqual(a["sharpe"], b["sharpe"], places=3)
        self.assertAlmostEqual(a["vol_matched_max_dd_pct"], b["vol_matched_max_dd_pct"], places=6)

    def test_end_to_end_same_trades_and_budget(self):
        vs = self.vs
        results, calib = vs.evaluate(self.frame, self.signal, self.spec, {"account": 0.0, "quoted": -1e-4})
        base = results["oos_2024_on"]["base"]
        self.assertEqual(len({base["arms"][a]["days"] for a in vs.ARMS}), 1)
        self.assertAlmostEqual(base["arms"]["CONSTANT_NOTIONAL"]["avg_w"], calib["constant_w"], places=3)
        self.assertIn(vs.decide(results, self.spec),
                      ("supported", "refuted", "not_significant", "underpowered", "inconclusive_sample"))
        self.assertLess(results["oos_2024_on"]["spread_x2"]["arms"]["FIXED_RISK"]["cost_bp_total"],
                        base["arms"]["FIXED_RISK"]["cost_bp_total"])
        # nested EWMA_RISK only rescales by a day-level factor, so entry hour explains little of its weight ratio
        self.assertLess(base["hour_r2_log_weight_ratio"]["EWMA_RISK"], 0.2)

    def test_decide_positive_ci_is_never_not_significant(self):
        arm = {"vol_matched_max_dd_pct": -5.0, "vol_matched_cvar5_bp": -50.0}
        base = {"trades": 300, "months_with_trades": 30, "delta_sharpe_ci": [0.05, 0.18],
                "arms": {"EWMA_RISK": arm, "FIXED_RISK": arm},
                "by_regime_sharpe": {"low": {"days": 100, "EWMA_RISK": 1.0, "FIXED_RISK": 0.9},
                                     "high": {"days": 100, "EWMA_RISK": 1.0, "FIXED_RISK": 0.9}},
                "by_year_sharpe": {2024: {"EWMA_RISK": 1, "FIXED_RISK": 0}, 2025: {"EWMA_RISK": 1, "FIXED_RISK": 0}}}
        results = {"oos_2024_on": {"base": base, "spread_x2": {"delta_sharpe": -0.01}, "swap_quoted_x1.5": {"delta_sharpe": 0.1}}}
        self.assertEqual(self.vs.decide(results, self.spec), "significant_not_robust")
        results["oos_2024_on"]["spread_x2"]["delta_sharpe"] = 0.01
        self.assertEqual(self.vs.decide(results, self.spec), "supported")


class DriftDiagnosticTests(unittest.TestCase):
    def test_runs_on_synthetic_and_scaling_removes_level_bias(self):
        import exp_fixed_param_drift_v1 as dd
        h1 = synthetic_h1(years=11)
        h1["spread"], h1["tick_volume"] = 0, 100
        rv = dd.vf.daily_rv(h1)["rv"]
        gvz = pd.Series(100 * np.sqrt(252 * rv.rolling(5).mean().bfill()), index=rv.index - pd.Timedelta(days=1))
        out = dd.evaluate(h1, gvz)
        for name, by in out["calibration_concept"].items():
            for y, r in by.items():
                if isinstance(y, int):
                    self.assertLessEqual(r["L_scaled"], r["L_frozen"] + 1e-9, (name, y))  # optimal scalar never hurts
        self.assertIn("ks_vs_P0", out["target_shift"]["log_rv"]["P1_2018_2023"])
        self.assertAlmostEqual(out["rebuilt_params"]["R_h5"]["train_mean_y_over_f"], 1.0, places=3)  # QLIKE FOC
        self.assertEqual(out["data_quality"][2015]["spread_zero_share"], 1.0)


def synthetic_m1(noise_before_2018=0.0, spikes_before_2018=0, zero_spread_before_2018=False, seed=0,
                 days_per_month=3, ar_noise_before_2018=None, stale_share_before_2018=0.0, daily_drift_after_2018=0.0):
    """Server days 2015-2019 (days_per_month each), 1440 M1 bars each. Before 2018 optionally: i.i.d. quote noise,
    AR(1) noise (sd, half-life in minutes) that persists across bars, one-minute bad ticks, stale (repeated) quotes."""
    rng = np.random.default_rng(seed)
    days = [d for m in pd.date_range("2015-01-01", "2019-12-01", freq="MS") for d in pd.bdate_range(m, periods=days_per_month)]
    rows, price, u = [], 1300.0, 0.0
    for d in days:
        early = d < pd.Timestamp("2018-01-01")
        r = rng.normal(0, 0.0003, 1440)
        if not early and daily_drift_after_2018:
            r = r + rng.normal(0, daily_drift_after_2018) / 1440  # trending days: real market, no quote noise
        true = price * np.exp(np.cumsum(r))
        obs = true * np.exp(rng.normal(0, noise_before_2018, 1440)) if early and noise_before_2018 else true.copy()
        if early and ar_noise_before_2018:
            sd, half = ar_noise_before_2018
            phi = 0.5 ** (1 / half)
            noise = np.empty(1440)
            for k in range(1440):
                u = phi * u + sd * np.sqrt(1 - phi ** 2) * rng.normal()
                noise[k] = u
            obs = obs * np.exp(noise)
        if early and stale_share_before_2018:
            keep = rng.random(1440) >= stale_share_before_2018
            keep[0] = True
            obs = obs[np.maximum.accumulate(np.where(keep, np.arange(1440), 0))]
        if early:
            for k in rng.choice(np.arange(10, 1430), spikes_before_2018, replace=False):
                obs[k] *= 1.01  # one-minute bad tick that reverts next minute
        price = true[-1]
        t = d + pd.to_timedelta(np.arange(1440), unit="min")
        o = np.r_[obs[0], obs[:-1]]
        spread = np.where(early and zero_spread_before_2018, 0, 30)
        rows.append(pd.DataFrame({"TIME_DT": t, "OPEN": o, "HIGH": np.maximum(o, obs), "LOW": np.minimum(o, obs),
                                  "CLOSE": obs, "SPREAD": spread, "TV": 100}))
    return pd.concat(rows, ignore_index=True)


def aggregate(m1, freq):
    g = m1.groupby(m1["TIME_DT"].dt.floor(freq))
    return pd.DataFrame({"OPEN": g["OPEN"].first(), "HIGH": g["HIGH"].max(), "LOW": g["LOW"].min(), "CLOSE": g["CLOSE"].last(),
                         "SPREAD": g["SPREAD"].first(), "TV": g["TV"].sum()}).rename_axis("TIME_DT").reset_index()


class DataQualityTests(unittest.TestCase):
    spec = {"seed": 0, "bootstrap_draws": 200, "min_m1_bars": 600, "ohlc_defect_rate": 0.001,
            "mismatch_defect_rate": 0.01, "vr_inflation_delta": 0.10, "vr_equivalence_band": 0.15,
            "vr_week_tolerance": 0.25}

    def run_case(self, m1):
        import exp_data_quality_v1 as dq
        h1, d1 = aggregate(m1, "h"), aggregate(m1, "D")
        out = dq.evaluate(m1, h1, d1, None, None, self.spec)
        return dq.classify(out, self.spec), out

    def test_clean_history_shows_no_inflation(self):
        (verdict, findings), out = self.run_case(synthetic_m1(days_per_month=8))
        p = out["primary"]
        self.assertNotEqual(verdict, "supported", p)
        self.assertLess(p["delta_vr_day_ci95"][0], 0)
        self.assertGreater(p["delta_vr_day_ci95"][1], 0)
        self.assertFalse([f for f in findings if f[1] == "confirmed_defect"])
        self.assertEqual(out["mismatch"]["H1_vs_M1"].max(), 0.0)

    def test_persistent_noise_invisible_to_roll_is_detected(self):
        # critic B1: AR noise with a 4-hour half-life barely moves the one-lag M1 autocovariance but inflates H1 RV
        (verdict, findings), out = self.run_case(synthetic_m1(days_per_month=8, ar_noise_before_2018=(0.004, 240),
                                                              zero_spread_before_2018=True))
        p = out["primary"]
        self.assertLess(out["by_period"]["P0"]["roll_noise_share_h1"], 0.02)
        self.assertEqual(verdict, "supported", p)
        self.assertIn(("M1 spread field", "confirmed_defect"), [(a, b) for a, b, _ in findings])

    def test_post_2018_trending_days_are_not_a_pre_2018_defect(self):
        # critic B7: clean P0, intraday drift only after 2018 raises VR_day(P1), which must not count as P0 noise
        (verdict, _), out = self.run_case(synthetic_m1(days_per_month=8, daily_drift_after_2018=0.008))
        self.assertGreater(out["primary"]["vr_day_P1"], 1.1, out["primary"])
        self.assertNotEqual(verdict, "supported", out["primary"])

    def test_stale_quotes_do_not_explode_spike_count(self):
        # critic B4: > 50% zero returns used to set the spike threshold to zero
        out = self.run_case(synthetic_m1(stale_share_before_2018=0.6))[1]
        self.assertGreater(out["by_period"]["P0"]["m1_zero_return_share"], 0.5)
        self.assertLess(out["by_period"]["P0"]["spikes_per_1000_days"], 50)

    def test_not_explained_branch_wording(self):
        import exp_data_quality_v1 as dq
        out = {"primary": {"delta_vr_day": 0.01, "delta_vr_day_ci95": [-0.05, 0.07], "delta_vr_week": 0.02,
                           "vr_day_P0": 0.98, "vr_day_P1": 0.99, "vr_day_P0_ci95": [0.9, 1.06],
                           "vr_day_P1_ci95": [0.93, 1.05]},
               "integrity": {}, "mismatch": {}, "sessions": {}, "change_months": {}, "hours": {}}
        verdict, findings = dq.classify(out, self.spec)
        self.assertEqual(verdict, "not_explained_by_quote_noise")
        self.assertNotIn("market_difference", [b for _, b, _ in findings])

    def test_cross_rate_identity_residual(self):
        import exp_data_quality_v1 as dq
        rng = np.random.default_rng(5)
        t = pd.date_range("2017-09-01", periods=24 * 600, freq="h")
        lg, le = np.cumsum(rng.normal(0, 1e-3, len(t))), np.cumsum(rng.normal(0, 5e-4, len(t)))
        noise = np.where(t < "2018-01-01", rng.normal(0, 1e-3, len(t)), 0.0)  # gold quotes noisy only in 2017
        mk = lambda x: pd.DataFrame({"TIME_DT": t, "CLOSE": np.exp(x)})
        out = dq.cross_residual(mk(lg + noise), mk(le), mk(lg - le))
        self.assertGreater(out["2017_08_12"]["var_ratio"], 1.0)
        self.assertLess(out["2018"]["var_ratio"], 0.001)  # only the first 2018 return touches a noisy 2017 quote


def synthetic_spread_m1(informative=True, zero_share=0.02, seed=0, q4_shift=False):
    """2019 M1 bars, server hours 8-15: hour-dependent spreads, 3-minute widenings (+20 points) on ~1% of minutes;
    if informative, volatility doubles for the 30 minutes after a widening starts."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", "2019-12-31")
    t = (days.values[:, None] + (np.arange(8 * 60, 16 * 60) * 60_000_000_000)[None, :].astype("timedelta64[ns]")).ravel()
    n = len(t)
    hour = pd.DatetimeIndex(t).hour.to_numpy()
    spread = 20 + 3 * (hour % 4) + rng.integers(0, 3, n)
    start = np.flatnonzero(rng.random(n) < 0.01)
    wide = np.zeros(n, bool)
    hot = np.zeros(n, bool)
    for s in start:
        wide[s:s + 3] = True
        hot[s + 1:s + 31] = True
    spread = np.where(wide, spread + 20, spread)
    sigma = np.where(hot & informative, 4e-4, 2e-4)
    if q4_shift:  # period-level drift only: Q4 spreads +60% and volatility +50%, unrelated minute by minute
        q4 = pd.DatetimeIndex(t).month.to_numpy() >= 10
        spread, sigma = np.where(q4, np.rint(spread * 1.6), spread), np.where(q4, sigma * 1.5, sigma)
    close = 1300 * np.exp(np.cumsum(rng.normal(0, sigma)))
    spread = np.where(rng.random(n) < zero_share, 0, spread)
    return pd.DataFrame({"TIME_DT": pd.DatetimeIndex(t), "OPEN": close, "HIGH": close, "LOW": close, "CLOSE": close,
                         "SPREAD": spread})


SPREAD_SPEC = {"spread_start": "2018-10-01", "train": ["2019-01-01", "2019-07-01"], "test": ["2019-07-01", "2020-01-01"],
               "bootstrap_draws": 200, "min_cell_bars": 50, "min_cell_bars_adaptive": 20, "adaptive_days": 90,
               "mae_ratio_max": 0.95, "mean_bias_max": 0.10, "horizon_minutes": 30, "max_span_minutes": 40,
               "anomaly_quantile": 0.9, "vol_ratio_threshold": 1.2, "min_anomalies": 500, "min_months": 6,
               "year_share_min": 0.75, "anomaly_share_band": [0.05, 0.20],
               "min_fills": 100, "point": 0.01}


class SpreadCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.informative, cls.noise = synthetic_spread_m1(True), synthetic_spread_m1(False, seed=1)

    def evaluate(self, m1):
        import exp_spread_cost_v1 as sc
        rng = np.random.default_rng(0)
        return sc.cost_model(m1, SPREAD_SPEC, 0.95, rng), sc.anomaly(m1, SPREAD_SPEC, 0.95, rng)

    def test_informative_widening_is_detected_and_cost_model_generalizes(self):
        import exp_spread_cost_v1 as sc
        cm, an = self.evaluate(self.informative)
        self.assertGreater(an["vol_ratio"], 1.2, an)  # top-decile z also catches ordinary noise
        self.assertTrue(cm["C1_passes"], cm)  # the hour pattern beats one global median out of sample
        exe = {"fills": 200}
        q1, q2, ready = sc.decide(cm, an, {"label": "min"}, exe, SPREAD_SPEC)
        self.assertEqual((q1, q2, ready), ("static_generalizes", "vol_clause_not_refuted", True))
        d = sc.describe(self.informative, SPREAD_SPEC)
        self.assertAlmostEqual(d["by_year"][2019]["missing_share"], 0.02, delta=0.005)
        self.assertGreaterEqual(d["by_year"][2019]["median"], 20)  # zeros are missing, never zero cost

    def test_uninformative_widening_is_refuted(self):
        import exp_spread_cost_v1 as sc
        cm, an = self.evaluate(self.noise)
        self.assertLess(an["vol_ratio_ci"][1], 1.2, an)
        self.assertEqual(sc.decide(cm, an, {"label": "min"}, {"fills": 200}, SPREAD_SPEC)[1], "refuted")
        small = {**an, "anomalies_test": 10}
        self.assertEqual(sc.decide(cm, small, {"label": "min"}, {}, SPREAD_SPEC)[1:], ("inconclusive_sample", False))

    def test_norms_trail_and_period_drift_is_not_an_anomaly(self):
        import exp_spread_cost_v1 as sc
        v = sc.with_cells(self.noise)
        start = pd.Timestamp("2019-09-01")
        w = sc.trailing(v, start, SPREAD_SPEC)
        self.assertLess(w["TIME_DT"].max(), start - pd.Timedelta(minutes=30))  # no forward window reaches the month
        self.assertGreaterEqual(w["TIME_DT"].min(), start - pd.Timedelta(days=90))
        # critic_agent_6 B1: level drift in Q4 (spread and volatility together) must not pass as H-009
        cm, an = self.evaluate(synthetic_spread_m1(False, seed=2, q4_shift=True))
        self.assertNotIn(sc.decide(cm, an, {"label": "min"}, {"fills": 200}, SPREAD_SPEC)[1],
                         ("vol_clause_not_refuted", "vol_clause_not_refuted_but_redundant"), an)

    def test_bar_field_semantics_and_fill_cost_sign(self):
        import exp_spread_cost_v1 as sc
        minute = pd.Timestamp("2025-03-04 10:00")
        base = minute.value // 10**6
        ticks = pd.DataFrame({"time_msc": [base + 1000, base + 20000, base + 50000, base + 61000],
                              "bid": [100.00, 100.00, 100.00, 100.00], "ask": [100.30, 100.25, 100.40, 100.35]})
        m1 = pd.DataFrame({"TIME_DT": [minute, minute + pd.Timedelta(minutes=1)], "SPREAD": [25, 35]})
        out = sc.field_semantics(ticks, m1, 0.01)
        self.assertEqual(out["label"], "min")
        fills = pd.DataFrame({"side": ["buy", "sell"], "price": [100.32, 99.98], "bid": [100.0, 100.0],
                              "ask": [100.30, 100.30], "reason": [3, 4], "lag_ms": [120, 900]})
        exe = sc.execution(fills, 0.01)
        self.assertAlmostEqual(exe["adverse_points"]["mean"], 2.0)  # buy paid 2 above ask, sell got 2 below bid


class FakeMt5:
    def terminal_info(self):
        return type("T", (), {"path": "C:/mt5", "build": 1, "name": "MT5"})()

    def account_info(self):
        return type("A", (), {"server": "demo", "company": "broker"})()


class RunIntegrationTests(unittest.TestCase):
    def test_research_run_passes_training_run_validator(self):
        import research_run_v1 as rr
        import training_run_history as trh
        tmp = Path(tempfile.mkdtemp())
        run_dir = trh.create_run("research_fixture", Path(rr.__file__), "python -B research_run_v1.py E-0",
                                 seed_note="fixture", runs_root=tmp)
        (run_dir / "stdout.log").write_text("ok", encoding="utf-8")
        out = {"outcome": "refuted", "report": "# fixture", "metrics": {"mean_r": -0.1},
               "manifest_data": rr.mt5_manifest_data(FakeMt5(), ["GOLD#"], "2020-01-01", "2021-01-01", 10,
                                                     [["2020-01-01", "2021-01-01"]], "fixture")}
        self.assertEqual(rr.finalize(run_dir, out, "H-1", "E-0"), [])
        self.assertTrue((run_dir / "FINALIZED.json").exists())


if __name__ == "__main__":
    unittest.main()
