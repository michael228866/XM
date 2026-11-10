"""Offline synthetic tests. MT5 and XGBoost are replaced before importing gemini."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo_runtime_foundation_v1 import (
    RuntimeRun, TradingSafetyError, canonical_bytes, intrabar_evidence, sha256_file,
)
from forward_trading_ledger_v1 import build_ledger, calculate_metrics, read_records, main as ledger_main

ROOT = Path(__file__).resolve().parent


class FakeMT5(types.ModuleType):
    ACCOUNT_TRADE_MODE_DEMO = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    DEAL_ENTRY_IN = 0

    def __init__(self):
        super().__init__("MetaTrader5")
        self.account = types.SimpleNamespace(login=123, server="Research-Demo", currency="USD",
                                             trade_mode=0, balance=1000)
        self.positions = ()
        self.deals = ()
        self.sent = []
        self.rates_calls = []

    def __getattr__(self, name):
        if name.startswith("TIMEFRAME_"):
            return name
        raise AttributeError(name)

    def initialize(self, *args, **kwargs):
        raise AssertionError("Tests must never initialize MT5")

    def account_info(self):
        return self.account

    def positions_get(self, **kwargs):
        return self.positions

    def history_deals_get(self, *args):
        return self.deals

    def last_error(self):
        return (-1, "synthetic failure")

    def symbol_info_tick(self, symbol):
        return types.SimpleNamespace(ask=2000.3, bid=2000.0)

    def order_send(self, request):
        self.sent.append(request)
        return types.SimpleNamespace(retcode=10009, order=11, deal=12, request_id=13,
                                     price=request["price"], volume=request["volume"],
                                     bid=2000.0, ask=2000.3, comment="fake fill")

    def order_calc_profit(self, side, symbol, volume, entry, stop):
        return (stop - entry) * volume * 100


class Sink:
    def __init__(self):
        self.run_id = "synthetic-run"
        self.identity = {"run_id": self.run_id, "account_login": "123",
                         "account_server": "Research-Demo", "currency": "USD",
                         "model_sha256": "model-A", "configuration_sha256": "config-A"}
        self.records = []

    def append(self, stream, value):
        self.records.append((stream, {**value, **{k: v for k, v in self.identity.items()
                                                if k.startswith("account_") or k == "currency"}}))


def load_runtime(fake):
    spec = importlib.util.spec_from_file_location("gemini_offline_under_test", ROOT / "gemini.py")
    module = importlib.util.module_from_spec(spec)
    xgb = types.ModuleType("xgboost")
    def forbidden(*args, **kwargs):
        raise AssertionError("No model construction/training in foundation tests")
    xgb.XGBClassifier = forbidden
    with patch.dict(sys.modules, {"MetaTrader5": fake, "xgboost": xgb}):
        spec.loader.exec_module(module)
    return module


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"XM_EXPECTED_LOGIN": "123",
                                           "XM_EXPECTED_SERVER": "Research-Demo"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mt5 = FakeMT5()
        self.g = load_runtime(self.mt5)
        self.g.SIGNAL_LOG_FILE = str(Path(self.tmp.name) / "signals.csv")
        self.g.TRADE_HISTORY_LOG_FILE = str(Path(self.tmp.name) / "deals.csv")
        self.g.RUNTIME_RUN = Sink()
        self.position = types.SimpleNamespace(type=0, volume=.1, ticket=9, time=0,
                                              magic=self.g.MAGIC_NUMBER)

    def test_positions_failure_blocks_entry(self):
        self.mt5.positions = None
        with self.assertRaises(TradingSafetyError):
            self.g.execute_trade(0, .1, 2, 3)
        self.assertEqual(self.mt5.sent, [])

    def test_direct_order_boundary_cannot_bypass_entry_query_checks(self):
        self.mt5.positions = None
        with self.assertRaises(TradingSafetyError):
            self.g.send_demo_order({"action": self.mt5.TRADE_ACTION_DEAL})
        self.assertEqual(self.mt5.sent, [])

    def test_history_failure_blocks_entry(self):
        self.mt5.deals = None
        with self.assertRaises(TradingSafetyError):
            self.g.execute_trade(0, .1, 2, 3)
        self.assertEqual(self.mt5.sent, [])

    def test_sync_failure_does_not_use_stale_csv(self):
        self.mt5.deals = None
        Path(self.g.TRADE_HISTORY_LOG_FILE).write_text("deal_id\nold\n", encoding="utf-8")
        with self.assertRaises(TradingSafetyError):
            self.g.sync_trade_history()
        self.assertEqual(Path(self.g.TRADE_HISTORY_LOG_FILE).read_text(), "deal_id\nold\n")

    def test_empty_success_is_distinct(self):
        self.assertEqual(self.g.get_strategy_positions(), [])
        self.assertEqual(self.g.get_daily_realized_pnl(), 0)
        self.assertEqual(self.g.get_daily_entry_count(), 0)

    def test_daily_count_failure_is_not_zero(self):
        self.mt5.deals = None
        with self.assertRaises(TradingSafetyError):
            self.g.get_daily_entry_count()

    def test_dry_run_all_order_paths(self):
        self.g.DRY_RUN = True
        self.mt5.account = None
        self.assertIsNone(self.g.execute_trade(0, .1, 2, 3))
        self.assertFalse(self.g.close_position(self.position))
        self.g.manage_open_positions()
        self.assertIsNone(self.g.send_demo_order({"action": "SLTP"}))
        self.assertEqual(self.mt5.sent, [])

    def test_account_modes_and_identity(self):
        for field, invalid in (("trade_mode", 2), ("login", 999), ("server", "Other")):
            with self.subTest(field=field):
                saved = getattr(self.mt5.account, field)
                setattr(self.mt5.account, field, invalid)
                with self.assertRaises(TradingSafetyError):
                    self.g.close_position(self.position)
                setattr(self.mt5.account, field, saved)
        self.assertEqual(self.mt5.sent, [])

    def test_missing_identity_is_not_autolearned(self):
        with patch.dict(os.environ, {"XM_EXPECTED_LOGIN": "", "XM_EXPECTED_SERVER": ""}):
            with self.assertRaises(TradingSafetyError):
                self.g.execute_trade(0, .1, 2, 3)

    def test_run_account_cannot_switch_by_changing_environment(self):
        self.mt5.account.login = 999
        with patch.dict(os.environ, {"XM_EXPECTED_LOGIN": "999"}):
            with self.assertRaises(TradingSafetyError):
                self.g.execute_trade(0, .1, 2, 3)

    def test_provenance_required_before_order(self):
        self.g.RUNTIME_RUN = None
        with self.assertRaises(TradingSafetyError):
            self.g.execute_trade(0, .1, 2, 3)

    def test_successful_entry_records_request_fill_and_risk(self):
        signal = {"event_time": "2026-01-01T00:00:00+00:00"}
        self.g.execute_trade(0, .1, 2, 3, signal)
        self.assertEqual(len(self.mt5.sent), 1)
        request = self.mt5.sent[0]
        self.assertEqual(request["price"], 2000.3)
        self.assertEqual(request["sl"], 1998.3)
        result = [v for s, v in self.g.RUNTIME_RUN.records if s == "orders" and v["stage"] == "result"][0]
        self.assertAlmostEqual(result["initial_risk"], 20)
        self.assertEqual(result["model_sha256"], "model-A")
        self.assertEqual(result["order_event_id"], signal["order_event_id"])

    def test_optional_risk_failure_still_records_actual_fill(self):
        with patch.object(self.mt5, "order_calc_profit", return_value=None):
            self.g.execute_trade(0, .1, 2, 3)
        result = self.g.RUNTIME_RUN.records[-1][1]
        self.assertIsNone(result["initial_risk"])
        self.assertEqual(result["deal_id"], 12)

    def test_close_uses_opposite_side_and_position(self):
        self.assertTrue(self.g.close_position(self.position))
        self.assertEqual(self.mt5.sent[0]["type"], 1)
        self.assertEqual(self.mt5.sent[0]["price"], 2000)
        self.assertEqual(self.mt5.sent[0]["position"], 9)

    def test_csv_schema_mismatch_is_not_migrated(self):
        path = Path(self.g.SIGNAL_LOG_FILE)
        path.write_bytes(b"old,columns\r\nx,y\r\n")
        before = path.read_bytes()
        with self.assertRaises(TradingSafetyError):
            self.g.append_signal_log({"status": "fake"})
        self.assertEqual(path.read_bytes(), before)

    def test_only_one_mt5_order_send_boundary(self):
        tree = ast.parse((ROOT / "gemini.py").read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "mt5" and n.func.attr == "order_send"]
        self.assertEqual(len(calls), 1)

    def test_deal_observation_dedup_and_manual_exit_capture(self):
        entry = types.SimpleNamespace(ticket=1, order=11, position_id=9, time=1767225600,
                                      time_msc=1767225600000, symbol="GOLD#", magic=self.g.MAGIC_NUMBER,
                                      type=0, entry=0, volume=.1, price=2000, profit=0,
                                      commission=-1, swap=0, fee=0)
        exit_deal = types.SimpleNamespace(**{**vars(entry), "ticket": 2, "order": 12,
                                             "magic": 0, "entry": 1, "type": 1})
        self.mt5.deals = (entry, exit_deal)
        self.g.sync_trade_history()
        self.g.sync_trade_history()
        observations = [v for s, v in self.g.RUNTIME_RUN.records if s == "deals"]
        self.assertEqual(len(observations), 2)
        self.assertNotIn("run_id", observations[0])
        self.assertEqual(observations[1]["magic"], 0)
        entry.fee = -2
        self.g.sync_trade_history()
        observations = [v for s, v in self.g.RUNTIME_RUN.records if s == "deals"]
        self.assertEqual(len(observations), 3)
        self.assertEqual(observations[-1]["fee"], -2)

    def test_order_error_is_recorded_without_retry(self):
        with patch.object(self.mt5, "order_send", side_effect=RuntimeError("fake uncertain send")):
            with self.assertRaises(RuntimeError):
                self.g.execute_trade(0, .1, 2, 3)
        self.assertEqual(self.g.RUNTIME_RUN.records[-1][1]["stage"], "error")

    def test_runtime_journals_to_complete_versioned_ledger(self):
        self.g.execute_trade(0, .1, 2, 3, {"status": "synthetic"})
        base = int(BASE_TIME.timestamp())
        entry = types.SimpleNamespace(ticket=12, order=11, position_id=999, time=base,
                                      time_msc=base*1000, symbol="GOLD#", magic=self.g.MAGIC_NUMBER,
                                      type=0, entry=0, volume=.1, price=2000.3, profit=0,
                                      commission=-1, swap=0, fee=0)
        exit_deal = types.SimpleNamespace(**{**vars(entry), "ticket": 13, "order": 14,
                                             "time": base+60, "time_msc": (base+60)*1000,
                                             "entry": 1, "type": 1, "profit": 3})
        self.mt5.deals = (entry, exit_deal)
        self.g.sync_trade_history()
        records = self.g.RUNTIME_RUN.records
        ledger = build_ledger([r for s, r in records if s == "deals"],
                              [r for s, r in records if s == "orders"])
        trade = ledger["trades"][0]
        self.assertEqual(trade["state"], "complete")
        self.assertEqual(trade["net_pnl"], 1)
        self.assertEqual(trade["run_ids"], ["synthetic-run"])
        self.assertEqual(trade["provenance"], "known")
        self.assertAlmostEqual(trade["r_multiple"], .05)


BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def deal(ticket, entry, volume, profit=0, *, position="P1", minute=0, **extra):
    row = dict(deal_id=str(ticket), order_id="O" + str(ticket), position_id=position,
               deal_time=(BASE_TIME + timedelta(minutes=minute)).isoformat(),
               account_login="123", account_server="Research-Demo", currency="USD",
               symbol="GOLD#", magic=20260514, type=0 if entry == 0 else 1,
               entry=entry, volume=volume, price=2000, profit=profit,
               commission=0, swap=0, fee=0)
    row.update(extra)
    return row


def risk_order(row, amount=10, run="run-A"):
    return dict(stage="result", order_id=row["order_id"], account_login=row["account_login"],
                account_server=row["account_server"], initial_risk=amount,
                filled_volume=row["volume"], run_id=run,
                model_sha256="model-A", configuration_sha256="config-A")


class LedgerTests(unittest.TestCase):
    def test_full_costs_and_no_second_spread_charge(self):
        rows = [deal(1, 0, 1, commission=-2, fee=-.5),
                deal(2, 1, 1, 10, minute=2, commission=-1, swap=-.3, fee=-.2)]
        trade = build_ledger(rows, [risk_order(rows[0])])["trades"][0]
        self.assertEqual(trade["state"], "complete")
        self.assertEqual(trade["net_pnl"], 6)
        self.assertEqual(trade["explicit_cost"], 4)
        self.assertEqual(trade["r_multiple"], .6)
        self.assertFalse(trade["spread_deducted_again"])

    def test_split_entry_and_partial_exit(self):
        rows = [deal(1, 0, .4, commission=-1, order_id="O1"),
                deal(2, 0, .6, minute=1, commission=-1, order_id="O1"),
                deal(3, 1, .3, 3, minute=2), deal(4, 1, .7, -1, minute=3)]
        order = risk_order(rows[0], amount=20)
        order["filled_volume"] = 1
        trade = build_ledger(rows, [order])["trades"][0]
        self.assertEqual(trade["state"], "complete")
        self.assertEqual(trade["net_pnl"], 0)
        self.assertEqual(trade["r_multiple"], 0)

    def test_partial_exit_stays_open(self):
        ledger = build_ledger([deal(1, 0, 1), deal(2, 1, .4, 4, minute=1)])
        self.assertEqual(ledger["trades"][0]["state"], "open")
        self.assertIsNone(ledger["trades"][0]["net_pnl"])
        self.assertEqual(self.metrics(ledger)["trades"], 0)

    def test_exact_duplicate_is_not_double_counted(self):
        first = deal(1, 0, 1, commission=-2)
        repeat = {**first, "capture_run_id": "later-observer"}
        ledger = build_ledger([first, repeat, deal(2, 1, 1, 10, minute=1)])
        self.assertEqual(ledger["duplicate_deals"], 1)
        self.assertEqual(ledger["trades"][0]["net_pnl"], 8)

    def test_conflicting_duplicate_is_unknown(self):
        first = deal(1, 0, 1)
        ledger = build_ledger([first, {**first, "profit": 999}, deal(2, 1, 1, minute=1)])
        self.assertEqual(ledger["trades"][0]["state"], "Unknown")

    def test_equivalent_numeric_serializations_are_duplicates(self):
        row = deal(1, 0, 1)
        repeat = {**row, "volume": "1.0", "profit": "0.00"}
        ledger = build_ledger([row, repeat, deal(2, 1, 1, minute=1)])
        self.assertEqual(ledger["trades"][0]["state"], "complete")
        self.assertEqual(ledger["duplicate_deals"], 1)

    def test_missing_entry_and_excess_exit_are_unknown(self):
        for rows in ([deal(2, 1, 1)], [deal(1, 0, 1), deal(2, 1, 2, minute=1)]):
            self.assertEqual(build_ledger(rows)["trades"][0]["state"], "Unknown")

    def test_invalid_fields_are_never_zero_filled(self):
        for name, value in (("fee", ""), ("profit", "NaN"), ("volume", -1),
                            ("entry", 2), ("currency", ""), ("deal_time", "2026-01-01T00:00:00")):
            with self.subTest(field=name):
                first = deal(1, 0, 1)
                first[name] = value
                self.assertEqual(build_ledger([first])["trades"][0]["state"], "Unknown")

    def test_out_by_is_supported(self):
        rows = [deal(1, 0, 1), deal(2, 3, 1, minute=1)]
        self.assertEqual(build_ledger(rows)["trades"][0]["state"], "complete")

    def test_order_link_resolves_missing_position_only_when_unambiguous(self):
        first = deal(1, 0, .4, order_id="split-order")
        second = deal(2, 0, .6, order_id="split-order", position_id="", minute=1)
        ledger = build_ledger([first, second, deal(3, 1, 1, minute=2)])
        self.assertEqual(len(ledger["trades"]), 1)
        self.assertEqual(ledger["trades"][0]["state"], "complete")

    def test_old_observation_does_not_invent_origin_run(self):
        rows = [deal(1, 0, 1, capture_run_id="new-observer"), deal(2, 1, 1, minute=1)]
        trade = build_ledger(rows)["trades"][0]
        self.assertEqual(trade["provenance"], "Unknown")
        self.assertEqual(trade["run_ids"], [])
        self.assertIsNone(trade["r_multiple"])

    def test_partial_risk_capture_excluded_from_mean_r(self):
        rows = [deal(1, 0, 1), deal(2, 1, 1, 10, minute=1)]
        order = risk_order(rows[0]); order["filled_volume"] = .5
        metrics = self.metrics(build_ledger(rows, [order]))
        self.assertEqual(metrics["mean_r_samples"], 0)
        self.assertEqual(metrics["missing_risk_samples"], 1)

    def metrics(self, ledger):
        return calculate_metrics(ledger, "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")

    def test_metrics_counts_units_drawdown_and_window(self):
        rows, orders = [], []
        for i, pnl in enumerate([10, -4, -3, 2]):
            entry = deal(2*i+1, 0, 1, position=str(i+1), minute=2*i)
            rows += [entry, deal(2*i+2, 1, 1, pnl, position=str(i+1), minute=2*i+1)]
            orders.append(risk_order(entry))
        metrics = self.metrics(build_ledger(rows, orders))
        self.assertEqual(metrics["net_pnl"], 5)
        self.assertEqual(metrics["win_rate"], .5)
        self.assertAlmostEqual(metrics["profit_factor"], 12/7)
        self.assertEqual(metrics["mean_r"], .125)
        self.assertEqual(metrics["realized_max_drawdown"], -7)
        self.assertEqual(metrics["trades_per_day"], 2)
        self.assertEqual(metrics["average_win"], 6)
        self.assertEqual(metrics["average_loss"], -3.5)

    def test_no_loss_pf_is_explicit_infinity(self):
        metrics = self.metrics(build_ledger([deal(1, 0, 1), deal(2, 1, 1, 10, minute=1)]))
        self.assertIsNone(metrics["profit_factor"])
        self.assertEqual(metrics["profit_factor_status"], "infinite")

    def test_accounts_are_not_pooled(self):
        rows = [deal(1, 0, 1), deal(2, 1, 1, 10, minute=1),
                deal(1, 0, 1, account_login="999"),
                deal(2, 1, 1, minute=1, account_login="999")]
        with self.assertRaises(ValueError):
            self.metrics(build_ledger(rows))

    def test_metric_window_is_required_and_utc(self):
        for start, end in (("2026-01-01", "2026-01-02"),
                           ("2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z")):
            with self.assertRaises(ValueError):
                calculate_metrics(build_ledger([]), start, end)

    def test_partial_exit_cash_drawdown_includes_entry_costs(self):
        rows = [deal(1, 0, 1, commission=-2), deal(2, 1, .4, 10, minute=1),
                deal(3, 1, .6, -20, minute=2)]
        metrics = self.metrics(build_ledger(rows))
        self.assertEqual(metrics["net_pnl"], -12)
        self.assertEqual(metrics["realized_max_drawdown"], -20)
        self.assertEqual(metrics["observed_realized_cash_pnl"], -12)

    def test_same_timestamp_cash_events_do_not_invent_a_sequence(self):
        rows = [deal(1, 0, 1, position="P1"), deal(2, 1, 1, 10, position="P1", minute=1),
                deal(3, 0, 1, position="P2"), deal(4, 1, 1, -10, position="P2", minute=1)]
        self.assertEqual(self.metrics(build_ledger(rows))["realized_max_drawdown"], 0)

    def test_millisecond_cash_order_is_preserved_when_available(self):
        stamp = int((BASE_TIME + timedelta(minutes=1)).timestamp()*1000)
        rows = [deal(1, 0, 1, position="P1"), deal(2, 1, 1, 10, position="P1", minute=1, time_msc=stamp+100),
                deal(3, 0, 1, position="P2"), deal(4, 1, 1, -10, position="P2", minute=1, time_msc=stamp+200)]
        self.assertEqual(self.metrics(build_ledger(rows))["realized_max_drawdown"], -10)

    def test_ambiguous_same_timestamp_entry_exit_is_unknown(self):
        ledger = build_ledger([deal(1, 0, 1), deal(2, 1, 1, 10)])
        self.assertEqual(ledger["trades"][0]["state"], "Unknown")

    def test_missing_position_can_use_explicit_signal_link(self):
        first = deal(1, 0, 1, position_id="")
        signal = {"order_id": "O1", "position_ticket": "P1",
                  "account_login": "123", "account_server": "Research-Demo"}
        ledger = build_ledger([first, deal(2, 1, 1, minute=1)], signals=[signal])
        self.assertEqual(ledger["trades"][0]["state"], "complete")

    def test_missing_account_is_unknown_not_guessed(self):
        rows = [deal(1, 0, 1, account_login=""), deal(2, 1, 1, minute=1, account_login="")]
        metrics = self.metrics(build_ledger(rows))
        self.assertEqual(metrics["trades"], 0)
        self.assertEqual(metrics["unknown_samples"], 1)

    def test_cli_synthetic_export_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "deals.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in [deal(1, 0, 1), deal(2, 1, 1, 10, minute=1)]), encoding="utf-8")
            before = source.read_bytes()
            args = ["ledger", "--deals", str(source), "--start", "2026-01-01T00:00:00Z",
                    "--end", "2026-01-03T00:00:00Z", "--output", str(root / "report")]
            with patch.object(sys, "argv", args):
                ledger_main()
                saved = (root / "report" / "metrics.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    ledger_main()
            self.assertEqual((root / "report" / "metrics.json").read_bytes(), saved)
            self.assertEqual(source.read_bytes(), before)


class ProvenanceTests(unittest.TestCase):
    def test_unique_runs_hashes_and_exclusive_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.json"; model.write_text("synthetic-model", encoding="utf-8")
            account = FakeMT5().account
            config = {"CONF_THRESHOLD": .70, "hours": {1, 0}, "risk": .014}
            first = RuntimeRun(root / "runs", config, model, ROOT / "gemini.py", account)
            second = RuntimeRun(root / "runs", config, model, ROOT / "gemini.py", account)
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertEqual(first.identity["configuration_sha256"], second.identity["configuration_sha256"])
            manifest = json.loads((first.directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["CONF_THRESHOLD"], .70)
            self.assertEqual(manifest["model_sha256"], sha256_file(model))
            self.assertTrue(manifest["started_at"].endswith("+00:00"))
            self.assertEqual(manifest["account_trade_mode"], 0)
            before = (first.directory / "manifest.json").read_bytes()
            first.append("signals", {"run_id": first.run_id, "status": "fake"})
            first.append("deals", {"deal_id": "historical"})
            self.assertEqual((first.directory / "manifest.json").read_bytes(), before)
            historical = read_records(first.directory / "deals.jsonl")[0]
            self.assertNotIn("run_id", historical)
            self.assertEqual(historical["capture_run_id"], first.run_id)

    def test_config_hash_is_order_independent_but_setting_sensitive(self):
        self.assertEqual(canonical_bytes({"x": {2, 1}, "threshold": .7}),
                         canonical_bytes({"threshold": .7, "x": {1, 2}}))
        self.assertNotEqual(canonical_bytes({"threshold": .7}), canonical_bytes({"threshold": .75}))

    def test_holdout_path_is_rejected_before_read(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("must not open")):
            with self.assertRaises(PermissionError):
                read_records(ROOT / "future_holdout" / "gold_s4_v4" / "not-read.csv")


class SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.g = load_runtime(FakeMT5())

    def test_threshold_boundaries(self):
        for score, expected in ((.6499, ""), (.65, "base"), (.7499, "base"), (.75, "base")):
            self.assertEqual(self.g.get_entry_mode(True, False, score, score, None), expected)
        self.assertEqual(self.g.get_entry_mode(False, False, .99, .99, None), "")

    def test_occupancy_and_reentry_remain_barwise(self):
        self.assertEqual(self.g.get_position_gate([], .8, None), (True, "base_position"))
        self.assertFalse(self.g.get_position_gate([types.SimpleNamespace(profit=1)], .8, None)[0])
        self.assertTrue(self.g.get_position_gate([], .8, None)[0])

    def test_cooldown_and_rolling_guard_boundaries(self):
        now = BASE_TIME
        for minutes, allowed in ((14.99, False), (15, True)):
            with patch.object(self.g, "get_closed_trade_results", return_value=[
                    {"pnl": -1, "exit_time": now - timedelta(minutes=minutes)}]):
                self.assertEqual(self.g.get_runtime_risk_guard(now)["allow_entry"], allowed)
        rows = [{"pnl": -1, "exit_time": now-timedelta(days=1)}] * 18
        with patch.object(self.g, "get_closed_trade_results", return_value=rows):
            self.assertEqual(self.g.get_runtime_risk_guard(now)["risk_mult"], .5)

    def test_minimum_tp_spread_behavior_preserved(self):
        self.assertEqual(self.g.get_spread_limit_points(1.5), 45)
        self.assertEqual(self.g.get_spread_to_tp_ratio(45, 1.5), .3)

    def runtime_assignments(self, names, values):
        tree = ast.parse((ROOT / "gemini.py").read_text(encoding="utf-8-sig"))
        loop = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "live_trading_loop")
        selected = {n.targets[0].id: n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in names}
        environment = {**vars(self.g), **values}
        for name in names:
            exec(compile(ast.Module(body=[selected[name]], type_ignores=[]), "<runtime-expression>", "exec"), environment)
        return environment

    def test_session_and_rsi_actual_runtime_expressions(self):
        for hour, weekday, rsi, expected_session, expected_rsi in (
                (0, 0, 22, True, True), (5, 0, 50, False, True),
                (0, 5, 50, False, True), (0, 0, 35, True, False),
                (0, 0, 45, True, False), (0, 0, 45.01, True, True)):
            result = self.runtime_assignments(["base_session", "expanded_session", "in_session", "rsi_ok", "rsi_floor_ok"],
                                              dict(hour=hour, weekday=weekday, rsi=rsi))
            self.assertEqual(result["in_session"], expected_session)
            self.assertEqual(result["rsi_ok"], expected_rsi)
            self.assertEqual(result["rsi_floor_ok"], rsi >= 22)

    def test_actual_tp_sl_formula_and_local_daily_boundary_preserved(self):
        for atr in (.1, 2):
            result = self.runtime_assignments(["sl_distance", "tp_distance"], {"atr": atr})
            self.assertEqual(result["sl_distance"], max(atr*1.6, .6))
            self.assertEqual(result["tp_distance"], max(atr*1.3, 1.5))
        fake = self.g.mt5
        with patch.object(fake, "history_deals_get", return_value=()) as history:
            self.g.get_daily_realized_pnl()
        start, end = history.call_args.args
        self.assertIsNone(start.tzinfo)
        self.assertEqual((start.hour, start.minute, start.second), (0, 0, 0))
        self.assertEqual(start.date(), end.date())

    def test_timeout_boundary_uses_wallclock(self):
        self.g.DRY_RUN = False
        position = types.SimpleNamespace(time=1000, ticket=7)
        for held, count in ((5399.9, 0), (5400, 1)):
            with patch.object(self.g, "get_strategy_positions", return_value=[position]), \
                    patch.object(self.g.time, "time", return_value=1000+held), \
                    patch.object(self.g, "close_position", return_value=True) as close:
                self.g.manage_open_positions()
                self.assertEqual(close.call_count, count)

    def test_closed_m1_and_all_mtf_queries(self):
        frame = pd.DataFrame({"TIME_DT": pd.date_range("2026-01-01", periods=400, freq="min", tz="UTC"),
                              "OPEN": np.arange(400)+2000., "HIGH": np.arange(400)+2002.,
                              "LOW": np.arange(400)+1998., "CLOSE": np.arange(400)+2001.})
        queried = []
        def history(symbol, timeframe, count, start_pos=1):
            queried.append((timeframe, start_pos))
            return frame.copy()
        with patch.object(self.g, "get_mt5_data", side_effect=history):
            features, _, bar_time = self.g.get_current_features()
        self.assertEqual(bar_time, frame["TIME_DT"].iloc[-1])
        self.assertEqual(len(queried), 21)
        self.assertTrue(all(pos == 1 for _, pos in queried))
        self.assertEqual(list(features.columns), self.g.FEATURE_COLUMNS)

    def test_late_new_bar_is_still_scored(self):
        last = pd.Timestamp("2026-01-01 12:00", tz="UTC")
        def bar(ts):
            return np.array([(int(ts.timestamp()),)], dtype=[("time", "i8")])
        # 12:01:00 the terminal still shows 12:00 as last closed; at 12:01:01 the 12:01 bar appears.
        for rates, seen, expected in ((bar(last), last, False),
                                      (bar(last + pd.Timedelta(minutes=1)), last, True),
                                      (None, last, False), (bar(last), None, True)):
            self.g.mt5.copy_rates_from_pos = lambda *args, r=rates: r
            self.assertEqual(self.g.closed_bar_changed(seen), expected)

    def test_mtf_boundary_difference_is_isolated_not_fixed(self):
        higher = pd.DataFrame({"TIME_DT": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 10:00"]),
                               "trend": [1, -1, 1]})
        higher["trend"] = higher["trend"].shift(1)
        minute = pd.DataFrame({"TIME_DT": pd.to_datetime(["2026-01-01 09:59", "2026-01-01 10:00"])})
        research = pd.merge_asof(minute, higher, on="TIME_DT")["trend"].shift(1).iloc[-1]
        self.assertEqual(research, 1)
        self.assertNotEqual(research, -1)  # Runtime latest completed H1 trend.

    def test_same_bar_both_is_unknown_with_declared_research_convention(self):
        self.assertEqual(intrabar_evidence("long", 100, 2, 3, 103, 96),
                         {"observed_order": "Unknown", "research_convention": "stop_first"})
        self.assertEqual(intrabar_evidence("short", 100, 2, 3, 104, 97)["observed_order"], "Unknown")
        self.assertEqual(intrabar_evidence("long", 100, 2, 3, 103, 99)["observed_order"], "tp")

    def test_strategy_constants_and_core_functions_match_prechange_commit(self):
        original = subprocess.check_output(["git", "show", "003c2ff:gemini.py"], cwd=ROOT).decode("utf-8-sig")
        before, after = ast.parse(original), ast.parse((ROOT / "gemini.py").read_text(encoding="utf-8-sig"))
        def literals(tree):
            result = {}
            for n in tree.body:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    try: result[n.targets[0].id] = ast.literal_eval(n.value)
                    except (ValueError, TypeError): pass
            return result
        old, new = literals(before), literals(after)
        # Deliberate strategy changes made by the user after 003c2ff (2026-11-03/05).
        intended = {"CONF_THRESHOLD": 0.65, "RISK_PER_TRADE": 0.02}
        for name, value in old.items():
            self.assertEqual(new[name], intended.get(name, value), name)
        # normalize_lot deliberately changed: rounds up to the broker minimum within MIN_LOT_MAX_RISK_MULT.
        functions = {"get_entry_mode", "get_position_gate", "get_runtime_risk_guard", "add_indicators",
                     "get_current_features", "get_trend", "get_spread_limit_points",
                     "get_closed_trade_results"}
        for name in functions:
            a = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)

    def test_model_artifact_unchanged(self):
        self.assertEqual(sha256_file(ROOT / "gold_long_recent_candidate_xgb.json"),
                         "2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2")


if __name__ == "__main__":
    unittest.main()
