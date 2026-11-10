"""Synthetic V3 tests: no real Forward records, MT5 import, training or strategy replay."""
import argparse
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import forward_edge_diagnosis_v3 as v3
from forward_test_report_v2 import run_report
from test_forward_test_report_v2 import START, pair, report


def trade(pnl, minute=0):
    return {"accounting_net_pnl": pnl, "profit": pnl, "explicit_cost": 0,
            "accounting_exit_time": (START+timedelta(minutes=minute)).isoformat(),
            "holding_minutes": minute, "entry_volume": 1}


def args(root, **extra):
    values = dict(v2_dir=str(root/"v2"), output=str(root/"diagnosis"), costs="0,0.5,1",
                  bootstrap=100, seed=42, context_manifest=None, context_seal=None,
                  archive_deals=None, archive_seal=None, enum_source=None,
                  git_revision=[], audit_log=[], workbook=[], retained_evidence=[])
    values.update(extra)
    return argparse.Namespace(**values)


def make_v2(root):
    source = root/"deals.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in pair(pnl=12)+pair("P2", pnl=-4, first=3, minute=2)), encoding="utf-8")
    run_report(argparse.Namespace(deals=[str(source)], signals=[], orders=[], manifests=[],
               start=START.isoformat(), end=(START+timedelta(days=14)).isoformat(),
               recorded_start=None, recorded_end=None, output=str(root/"v2")))
    return source


class FinancialTests(unittest.TestCase):
    def test_exact_decimal_totals_payoff_and_breakeven(self):
        result = v3.financial([trade(12), trade(-4)])
        self.assertEqual(result["net_pnl"], 8)
        self.assertEqual(result["net_pf"], 3)
        self.assertEqual(result["payoff_ratio"], 3)
        self.assertEqual(result["win_rate"], .5)
        self.assertEqual(result["break_even_win_rate"], .25)
        self.assertEqual(result["net_expectancy_per_trade"], 4)
        self.assertEqual(v3.financial([trade(.1), trade(.2), trade(-.3)])["net_pnl"], 0)

    def test_flat_trades_use_same_win_rate_denominator(self):
        result = v3.financial([trade(12), trade(-4), trade(0)])
        self.assertAlmostEqual(result["break_even_win_rate"], 1/6)
        self.assertAlmostEqual(result["win_rate"], 1/3)
        self.assertEqual(result["n_zero"], 1)

    def test_extra_cost_is_once_per_closed_trade_without_spread(self):
        rows = [trade(12), trade(-4)]
        result = v3.financial(rows, .5)
        self.assertEqual(result["net_pnl"], 7)
        self.assertEqual(result["recorded_explicit_cost"], 0)
        self.assertEqual(rows[0]["accounting_net_pnl"], 12)

    def test_cost_can_change_winner_to_loser(self):
        self.assertEqual(v3.financial([trade(.1)], .25)["n_losses"], 1)

    def test_invalid_cost_rejected(self):
        for cost in (-1, "nan", "inf"):
            with self.assertRaises(ValueError):
                v3.financial([], cost)

    def test_no_losses_and_empty_not_finite_pf(self):
        self.assertEqual(v3.financial([trade(1)])["pf_status"], "infinite")
        self.assertIsNone(v3.financial([trade(1)])["net_pf"])
        self.assertEqual(v3.financial([])["pf_status"], "undefined")
        self.assertIsNone(v3.financial([])["win_rate"])

    def test_concentration_retains_all_losses(self):
        rows = [trade(-20), trade(-10), trade(5)]
        result = v3.concentration(rows)
        self.assertAlmostEqual(result[0]["share_of_gross_loss"], 2/3)
        self.assertEqual(result[1]["loss_amount"], 30)
        self.assertEqual(len(rows), 3)

    def test_tied_ranks_and_constant_correlation(self):
        self.assertEqual(v3.average_ranks([2, 1, 2]), [2.5, 1, 2.5])
        self.assertIsNone(v3.correlation([1]*3, [1, 2, 3]))
        self.assertIsNone(v3.correlation([1], [1]))

    def test_duration_is_association_not_risk_or_cause(self):
        result = v3.holding_relationship([trade(1, 1), trade(2, 2), trade(3, 3)])
        self.assertAlmostEqual(result["spearman_duration_pnl"], 1)
        self.assertIn("outcome", result["limitation"])
        self.assertIn("not R", result["limitation"])


class BootstrapTests(unittest.TestCase):
    window = [START.isoformat(), (START+timedelta(days=28)).isoformat()]

    def test_reproducible_empty_week_blocks_and_dependence_disclosure(self):
        rows = [trade(12), trade(-4)]
        first = v3.week_bootstrap(rows, self.window, 200, 42)
        self.assertEqual(first, v3.week_bootstrap(rows, self.window, 200, 42))
        self.assertEqual(first["active_week_blocks"], 1)
        self.assertGreater(first["calendar_week_blocks"], 1)
        self.assertGreater(first["empty_resamples"], 0)
        self.assertEqual(first["mean_pnl_95_interval"], [4, 4])
        self.assertEqual(first["label"], "Insufficient Evidence")
        self.assertTrue(any("cross-week" in s for s in first["limitations"]))

    def test_infinite_pf_is_not_reported_as_finite(self):
        result = v3.week_bootstrap([trade(1)], self.window, 100)
        self.assertEqual(result["pf_95_interval"], [None, None])
        self.assertGreater(result["pf_infinite_resamples"], 0)
        json.dumps(result, allow_nan=False)

    def test_mixed_clocks_rejected(self):
        row = trade(1)
        row["accounting_exit_time"] = START.replace(tzinfo=None).isoformat()
        with self.assertRaises(ValueError):
            v3.week_bootstrap([row], self.window, 100)

    def test_invalid_window_and_repetitions_rejected(self):
        with self.assertRaises(ValueError):
            v3.week_bootstrap([], self.window, 1)
        with self.assertRaises(ValueError):
            v3.week_bootstrap([], list(reversed(self.window)), 100)


class AttributionTests(unittest.TestCase):
    def test_broker_exit_reason_not_holding_time(self):
        rows = pair(reason=5, comment="[tp 2000]")
        row = report(rows)["ledger"]["trades"][0]
        enums = {"DEAL_REASON_TP": 5, "DEAL_REASON_SL": 4}
        self.assertEqual(v3.classify_exit(row, rows, enums)["exit_category"], "TP")
        rows[1]["reason"] = 3
        result = v3.classify_exit(row, rows, enums)
        self.assertEqual(result["exit_category"], "Unknown")
        self.assertIn("Unknown", result["timeout"])

    def test_comment_conflict_and_missing_enum_are_unknown(self):
        rows = pair(reason=5, comment="[sl 1990]")
        row = report(rows)["ledger"]["trades"][0]
        self.assertIn("Unknown", v3.classify_exit(row, rows, {"DEAL_REASON_TP": 5})["exit_category"])
        self.assertEqual(v3.classify_exit(row, rows, {})["exit_category"], "Unknown")

    def test_source_read_uses_ast_without_executing_it(self):
        source = b'CONF_THRESHOLD = 0.70\nraise Exception("must not run")\nfee = getattr(deal,"fee",0.0)\n'
        constants, defaults = v3.source_constants(source)
        self.assertEqual(constants["CONF_THRESHOLD"], .70)
        self.assertEqual(defaults, [0.0])

    def test_overlapping_gates_are_not_independent_trades(self):
        rows = [{"event_time": START.isoformat(), "status": "not_valid", "message": "session;entry_mode"}]
        utc = next(r for r in v3.signal_layers(rows) if r["time_basis"] == "UTC")
        self.assertEqual(utc["n_signal_rows"], 1)
        self.assertEqual(sum(utc["not_valid_reasons_overlapping"].values()), 2)
        self.assertEqual(utc["causal_attribution"], "Insufficient Evidence")

    def test_account_context_not_used_to_assign_unknown_trade(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); make_v2(root)
            context = root/"manifest.json"
            context.write_text(json.dumps({"data": {"mt5_fetch": {"broker_info": {"login": 123}}}}))
            seal = root/"seal.json"
            seal.write_text(json.dumps({"file_sha256": {"manifest.json": v3.digest(context.read_bytes())}}))
            result = v3.run(args(root, context_manifest=str(context), context_seal=str(seal)))
            self.assertEqual(result["coverage"]["account_verified_n"], 0)
            self.assertEqual(result["coverage"]["model_version_verified_n"], 0)
            self.assertEqual(result["evidence_findings"][0]["trade_identity_coverage_added"], 0)


class IntegrationTests(unittest.TestCase):
    def test_read_only_seals_snapshots_and_reproduction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = make_v2(root)
            before = source.read_bytes()
            enum = root/"sdk.py"
            enum.write_text("DEAL_REASON_TP = 5\nDEAL_REASON_SL = 4\nraise Exception('no import')\n")
            with patch.object(v3.subprocess, "check_output", side_effect=[b"CONF_THRESHOLD = .75\n", "abc\n", ""]):
                original = v3.run(args(root, enum_source=str(enum), git_revision=["abcdef0"]))
            manifest = json.loads((root/"diagnosis"/"manifest.json").read_text(encoding="utf-8"))
            specifications = []
            for artifact in manifest["evidence"]:
                item = {k: artifact[k] for k in ("snapshot", "role", "sha256", "original_name")}
                item["path"] = str(root/"diagnosis"/artifact["snapshot"])
                if "revision" in artifact:
                    item["revision"] = artifact["revision"]
                specifications.append(json.dumps(item))
            second = v3.run(args(root, output=str(root/"second"), retained_evidence=specifications))
            self.assertEqual(original, second)
            self.assertEqual(source.read_bytes(), before)
            for name, sha in manifest["output_sha256"].items():
                self.assertEqual(v3.digest((root/"diagnosis"/name).read_bytes()), sha)
                self.assertEqual((root/"diagnosis"/name).read_bytes(), (root/"second"/name).read_bytes())
            with self.assertRaises(FileExistsError):
                v3.run(args(root))

    def test_v2_tamper_detected_before_output_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); make_v2(root)
            (root/"v2"/"report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                v3.run(args(root))
            self.assertFalse((root/"diagnosis").exists())

    def test_retained_evidence_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root/"source.py"; data.write_text("tamper")
            specification = json.dumps({"path": str(data), "sha256": "bad", "role": "source_history_only"})
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                v3.evidence_inventory(args(root, retained_evidence=[specification]), [])

    def test_context_seal_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); context = root/"manifest.json"; context.write_text("{}")
            with self.assertRaises(ValueError):
                v3.evidence_inventory(args(root, context_manifest=str(context)), [])
            seal = root/"seal.json"; seal.write_text('{"file_sha256":{"manifest.json":"bad"}}')
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                v3.evidence_inventory(args(root, context_manifest=str(context), context_seal=str(seal)), [])

    def test_holdout_refused_without_opening(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("must not open")):
            with self.assertRaises(PermissionError):
                v3.load_v2(Path(__file__).parent/"future_holdout"/"gold_s4_v4")


if __name__ == "__main__":
    unittest.main()
