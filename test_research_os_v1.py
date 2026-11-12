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
                         "metrics": ["net_mean", "sharpe", "calmar", "max_dd"]}}
        events = self.add(good)
        self.assertEqual(rc.critique(events[-1], events, root=self.dir)[0], [])

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
