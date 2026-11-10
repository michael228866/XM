"""Synthetic accounting/report tests; no actual Forward records or MT5 are loaded."""
import argparse
import ast
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from demo_runtime_foundation_v1 import canonical_bytes
from forward_test_report_v2 import (
    LEGACY_CLOCK, build_report, digest, parse_snapshot, run_report, safe_path,
)
from forward_trading_ledger_v1 import build_ledger

UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOWS = {"UTC": (START, START+timedelta(days=2)),
           LEGACY_CLOCK: (START.replace(tzinfo=None), (START+timedelta(days=2)).replace(tzinfo=None))}


def deal(ticket, entry, pnl=0, position="P1", minute=0, naive=False, **extra):
    when = START+timedelta(minutes=minute)
    if naive:
        when = when.replace(tzinfo=None)
    row = {"deal_id": str(ticket), "order_id": "O"+str(ticket), "position_id": position,
           "entry": entry, "type": 0 if entry == 0 else 1,
           "deal_time": when.isoformat(), "volume": 1, "price": 2000,
           "profit": pnl, "commission": 0, "swap": 0, "fee": 0,
           "symbol": "GOLD#", "magic": 20260514}
    row.update(extra)
    return row


def pair(position="P1", pnl=10, first=1, minute=0, **extra):
    return [deal(first, 0, position=position, minute=minute, **extra),
            deal(first+1, 1, pnl, position=position, minute=minute+1, **extra)]


def report(rows, signals=(), orders=(), manifests=()):
    return build_report(rows, signals, orders, manifests, WINDOWS)


class ReportTests(unittest.TestCase):
    def test_unknown_identity_does_not_become_a_verified_account(self):
        result = report(pair())
        trade = result["ledger"]["trades"][0]
        self.assertEqual(trade["state"], "Unknown")
        self.assertEqual(trade["account_login"], "Unknown")
        self.assertIsNone(trade["currency"])
        self.assertEqual(trade["accounting_state"], "complete")
        self.assertEqual(result["summary"]["total_net_pnl"]["value"], 10)
        self.assertEqual(result["summary"]["total_net_pnl"]["n"], 1)

    def test_naive_time_is_not_assigned_utc(self):
        result = report(pair(naive=True))
        trade = result["ledger"]["trades"][0]
        self.assertEqual(trade["time_basis"], LEGACY_CLOCK)
        self.assertNotIn("+00:00", trade["accounting_entry_time"])
        self.assertEqual(result["sessions"][0]["group"], "Unknown_timezone")
        self.assertEqual(result["time_partitions"][0]["trades_per_day"], .5)

    def test_clocks_are_not_combined_for_drawdown_or_frequency(self):
        result = report(pair(naive=True)+pair("P2", -4, first=3, minute=5))
        self.assertEqual(len(result["time_partitions"]), 2)
        self.assertIsNone(result["summary"]["realized_max_drawdown"]["value"])
        self.assertIsNone(result["summary"]["trades_per_day"]["value"])

    def test_cost_net_win_rate_pf_payoff_and_metric_samples(self):
        rows = pair(pnl=12, commission=-1)+pair("P2", pnl=-4, first=3, minute=2, fee=-1)
        result = report(rows)
        metrics = result["summary"]
        self.assertEqual(metrics["total_net_pnl"]["value"], 4)
        self.assertEqual(metrics["win_rate"]["value"], .5)
        self.assertAlmostEqual(metrics["profit_factor"]["value"], 10/6)
        self.assertEqual(metrics["average_win"]["value"], 10)
        self.assertEqual(metrics["average_win"]["n"], 1)
        self.assertEqual(metrics["average_loss"]["value"], -6)
        self.assertAlmostEqual(metrics["payoff_ratio"]["value"], 10/6)
        self.assertEqual(metrics["explicit_cost"]["value"], 4)
        self.assertEqual(metrics["mean_r"]["n"], 0)

    def test_exact_duplicates_and_conflicting_duplicates(self):
        rows = pair()
        self.assertEqual(report(rows+[dict(rows[0])])["summary"]["total_net_pnl"]["value"], 10)
        result = report(rows+[{**rows[0], "fee": -5}])
        self.assertEqual(result["summary"]["total_net_pnl"]["n"], 0)
        self.assertEqual(result["integrity"]["conflicting_cost_duplicates"], 1)

    def test_partial_exit_open_costs_in_cash_drawdown_not_win_rate(self):
        rows = [deal(1, 0, commission=-2), deal(2, 1, 5, minute=1, volume=.4)]
        result = report(rows)
        self.assertEqual(result["summary"]["win_rate"]["n"], 0)
        self.assertEqual(result["integrity"]["recorded_accounting_states"], {"open": 1})
        self.assertEqual(result["time_partitions"][0]["realized_max_drawdown"], -2)

    def test_partial_close_complete_and_reconciliation(self):
        rows = [deal(1, 0, commission=-2), deal(2, 1, 3, minute=1, volume=.4),
                deal(3, 1, -1, minute=2, volume=.6)]
        result = report(rows)
        self.assertEqual(float(result["integrity"]["reconciliation_difference"]), 0)
        self.assertEqual(result["summary"]["total_net_pnl"]["value"], 0)
        self.assertEqual(result["summary"]["total_net_pnl"]["n"], 1)

    def test_missing_fee_excludes_sample_instead_of_assuming_zero(self):
        rows = pair(); del rows[0]["fee"]
        result = report(rows)
        self.assertEqual(result["summary"]["total_net_pnl"]["n"], 0)
        self.assertEqual(result["integrity"]["invalid_numeric_rows"], 1)

    def test_daily_zero_days_weekly_totals_and_session(self):
        result = report(pair())
        self.assertEqual(len(result["daily"]), 2)
        self.assertEqual(result["daily"][1]["total_net_pnl_n"], 0)
        self.assertEqual(result["weekly"][0]["total_net_pnl"], 10)
        self.assertEqual(result["sessions"][0]["group"], "recorded_UTC_00_08")
        self.assertEqual(result["sessions"][0]["assessment"], "insufficient_sample")

    def test_explicit_window_does_not_shrink_to_active_days(self):
        result = report(pair())
        self.assertEqual(result["time_partitions"][0]["observation_days"], 2)
        self.assertEqual(result["time_partitions"][0]["trades_per_day"], .5)

    def test_legacy_signal_score_link_is_not_a_model_version(self):
        signal = {"status": "order_opened", "deal_id": "1", "order_id": "O1",
                  "position_ticket": "P1", "buy_prob": ".72"}
        result = report(pair(), [signal])
        trade = result["ledger"]["trades"][0]
        self.assertEqual(trade["recorded_buy_score"], .72)
        self.assertEqual(trade["conf_threshold"], "Unknown")
        self.assertEqual(trade["model_sha256"], "Unknown")
        self.assertEqual(result["score_bands"][0]["group"], "0.70_to_below_0.75")

    def test_empty_threshold_band_is_explicitly_reported(self):
        bands = {r["group"]: r for r in report(pair())["score_bands"]}
        self.assertEqual(bands["0.70_to_below_0.75"]["total_net_pnl_n"], 0)
        self.assertIsNone(bands["0.70_to_below_0.75"]["win_rate"])
        self.assertEqual(bands["0.70_to_below_0.75"]["assessment"], "insufficient_sample")

    def test_conflicting_signal_ids_do_not_guess_a_score(self):
        signal = {"status": "order_opened", "deal_id": "1", "order_id": "WRONG", "buy_prob": ".72"}
        trade = report(pair(), [signal])["ledger"]["trades"][0]
        self.assertIsNone(trade["recorded_buy_score"])
        self.assertEqual(trade["signal_link_issue"], "conflicting_ids")

    def test_verified_manifest_required_for_version_attribution(self):
        identity = {"account_login": "123", "account_server": "Demo", "currency": "USD"}
        config = {"CONF_THRESHOLD": .7}
        manifest = {**identity, "run_id": "R1", "model_sha256": "a"*64,
                    "configuration": config, "configuration_sha256": digest(canonical_bytes(config)),
                    "git_commit": "historical-commit", "runtime_source_sha256": "b"*64}
        signal = {**identity, "run_id": "R1", "model_sha256": "a"*64,
                  "configuration_sha256": manifest["configuration_sha256"], "status": "order_opened",
                  "deal_id": "1", "order_id": "O1", "position_ticket": "P1", "buy_prob": .72}
        trade = report(pair(**identity), [signal], manifests=[manifest])["ledger"]["trades"][0]
        self.assertEqual(trade["conf_threshold"], .7)
        self.assertEqual(trade["git_commit"], "historical-commit")
        invalid = {**manifest, "configuration_sha256": "wrong"}
        with self.assertRaises(ValueError):
            report(pair(**identity), [signal], manifests=[invalid])

    def test_scope_mixture_is_not_added_together(self):
        rows = pair(account_login="123", account_server="Demo", currency="USD")
        rows += pair("P2", first=3, account_login="456", account_server="Demo", currency="EUR")
        with self.assertRaises(ValueError):
            report(rows)

    def test_time_mixed_inside_one_position_is_unknown(self):
        rows = pair(); rows[0]["deal_time"] = "2026-01-01T00:00:00"
        self.assertEqual(build_ledger(rows)["trades"][0]["accounting_state"], "Unknown")

    def test_losing_streak_and_cash_drawdown(self):
        rows = []
        for i, pnl in enumerate((10, -3, -4, 2)):
            rows.extend(pair(str(i+1), pnl, 2*i+1, i*2))
        result = report(rows)
        self.assertEqual(result["summary"]["max_losing_streak"]["value"], 2)
        self.assertEqual(result["summary"]["realized_max_drawdown"]["value"], -7)

    def test_csv_missing_fields_duplicate_header_and_truncated_tail(self):
        rows, rejected = parse_snapshot(b"a,b\n1,2\n3\n4,5", ".csv")
        self.assertEqual(rows, [{"a": "1", "b": "2"}])
        self.assertEqual(len(rejected), 2)
        with self.assertRaises(ValueError):
            parse_snapshot(b"a,a\n1,2\n", ".csv")

    def test_no_mt5_model_or_runtime_imports(self):
        tree = ast.parse(Path(__file__).with_name("forward_test_report_v2.py").read_text(encoding="utf-8-sig"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        self.assertFalse({"MetaTrader5", "gemini", "xgboost"} & set(imports))

    def test_holdout_rejected_before_open(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("must not open")):
            with self.assertRaises(PermissionError):
                safe_path(Path(__file__).parent / "future_holdout" / "gold_s4_v4" / "not-read.csv")

    def test_export_snapshot_integrity_reproduce_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/"deals.jsonl"
            source.write_text("".join(json.dumps(r)+"\n" for r in pair()), encoding="utf-8")
            before = source.read_bytes()
            args = argparse.Namespace(deals=[str(source)], signals=[], orders=[], manifests=[],
                                      start=START.isoformat(), end=(START+timedelta(days=2)).isoformat(),
                                      recorded_start=None, recorded_end=None, output=str(root/"report"))
            first = run_report(args)
            manifest = json.loads((root/"report"/"manifest.json").read_text(encoding="utf-8"))
            retained = root/"report"/manifest["inputs"][0]["snapshot"]
            self.assertEqual(retained.read_bytes(), before)
            self.assertEqual(digest(retained.read_bytes()), manifest["inputs"][0]["sha256"])
            with self.assertRaises(FileExistsError):
                run_report(args)
            args.deals = [str(retained)]; args.output = str(root/"second")
            second = run_report(args)
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["ledger"], second["ledger"])
            self.assertEqual(source.read_bytes(), before)
            for name in ("report.md", "report.json", "trades.csv", "summary.csv", "daily.csv", "weekly.csv"):
                self.assertTrue((root/"report"/name).exists())


if __name__ == "__main__":
    unittest.main()
