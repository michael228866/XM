"""Synthetic acceptance tests; no MT5 import, actual Forward data or strategy execution."""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import forward_data_quality_v4 as q
from demo_runtime_foundation_v1 import canonical_bytes
from test_demo_trading_foundation_v1 import BASE_TIME, deal


EXPECTED = {"account_login": "123", "account_server": "Research-Demo", "currency": "USD"}


def fixture():
    config = {"CONF_THRESHOLD": .70}
    manifest = {**EXPECTED, "schema_version": 1, "run_id": "run-A", "account_trade_mode": 0,
                "started_at": (BASE_TIME-timedelta(minutes=1)).isoformat(),
                "model_sha256": "a"*64, "configuration_sha256": q.digest(canonical_bytes(config)),
                "runtime_source_sha256": "b"*64, "foundation_source_sha256": "c"*64,
                "git_commit": "d"*40, "configuration": config}
    identity = {k: manifest[k] for k in ("run_id", "account_login", "account_server", "currency", "model_sha256", "configuration_sha256")}
    deals = [deal(1, 0, 1, commission=-1), deal(2, 1, 1, profit=12, commission=-1, fee=-.5, minute=2)]
    order = {**identity, "stage": "result", "order_id": "O1", "initial_risk": 10,
             "filled_volume": 1, "broker_price": 2000,
             "risk_source": "order_calc_profit_fill_to_requested_initial_sl",
             "request": {"type": 0, "price": 2000, "sl": 1990}}
    signal = {**identity, "status": "order_opened", "buy_prob": .72,
              "event_time": BASE_TIME.isoformat(), "order_id": "O1", "deal_id": "1", "position_ticket": "P1"}
    broker = copy.deepcopy(deals)
    broker[1].update(reason=5, comment="[tp 2012]")
    return dict(deals=deals, orders=[order], signals=[signal], manifests=[manifest], broker=broker, expected=EXPECTED)


def cli(root, **extra):
    values = dict(v2_dir=None, input_seal=None, deals=[], orders=[], signals=[], manifests=[], broker_deals=[],
                  expected_login="123", expected_server="Research-Demo", expected_currency="USD",
                  output=str(root/"quality"))
    values.update(extra)
    return argparse.Namespace(**values)


class AcceptanceTests(unittest.TestCase):
    def test_complete_receipts_and_costs_pass(self):
        result = q.inspect(**fixture())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["summary"]["reconciliation_difference"], "0.0")
        self.assertEqual(result["trades"][0]["recorded_net_pnl"], 9.5)
        self.assertEqual(result["trades"][0]["exit_category"], "TP")
        self.assertTrue(all(r["coverage"] == 1 for r in result["coverage"]))

    def test_expected_account_is_never_inferred(self):
        data = fixture(); data["expected"] = None
        result = q.inspect(**data)
        self.assertEqual(result["status"], "NOT_READY")
        self.assertIn("expected_account_missing", result["summary"]["issue_counts"])

    def test_wrong_account_or_currency_cannot_pass(self):
        for field in EXPECTED:
            data = fixture(); data["expected"] = {**EXPECTED, field: "other"}
            self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["account"])

    def test_missing_demo_evidence_and_real_mode_rejected(self):
        for value in (None, 2, False):
            data = fixture(); data["manifests"][0]["account_trade_mode"] = value
            self.assertIn("manifest_invalid", q.inspect(**data)["summary"]["issue_counts"])

    def test_bad_configuration_hash_is_reported(self):
        data = fixture(); data["manifests"][0]["configuration"]["CONF_THRESHOLD"] = .75
        self.assertEqual(q.inspect(**data)["status"], "NOT_READY")

    def test_malformed_manifest_and_nested_request_do_not_pass(self):
        data = fixture(); data["manifests"] = [[]]
        self.assertEqual(q.inspect(**data)["status"], "NOT_READY")
        data = fixture(); data["orders"][0]["request"] = "not a parsed JSON object"
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["initial_risk"])

    def test_capture_id_cannot_replace_origin_run(self):
        data = fixture()
        for row in data["orders"]+data["signals"]:
            row["capture_run_id"] = row.pop("run_id")
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["version"])

    def test_later_capture_cannot_claim_earlier_trade(self):
        data = fixture(); data["manifests"][0]["started_at"] = (BASE_TIME+timedelta(days=1)).isoformat()
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["version"])

    def test_risk_budget_or_incomplete_fill_is_not_initial_risk(self):
        for edit in ({"initial_risk": None, "risk_budget": 10}, {"filled_volume": .5}, {"risk_source": "estimate"}):
            data = fixture(); data["orders"][0].update(edit)
            self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["initial_risk"])

    def test_wrong_side_sl_and_missing_decision_quote_fail(self):
        data = fixture(); data["orders"][0]["request"]["sl"] = 2010
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["initial_risk"])
        data = fixture(); del data["orders"][0]["request"]["price"]
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["decision_and_fill"])

    def test_receipt_fill_price_must_match_actual_deals(self):
        data = fixture(); data["orders"][0]["broker_price"] = 2001
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["decision_and_fill"])

    def test_naive_or_future_signal_time_fails(self):
        for when in (BASE_TIME.replace(tzinfo=None).isoformat(), (BASE_TIME+timedelta(hours=1)).isoformat()):
            data = fixture(); data["signals"][0]["event_time"] = when
            self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["decision_and_fill"])

    def test_missing_cost_is_unknown_not_zero(self):
        data = fixture(); del data["deals"][0]["fee"]
        self.assertEqual(q.inspect(**data)["status"], "NOT_READY")
        self.assertNotIn("fee", data["deals"][0])

    def test_zero_costs_still_require_export_match(self):
        data = fixture(); data["broker"] = []
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["broker_match"])

    def test_cost_or_exit_reason_mismatch_is_not_silently_accepted(self):
        data = fixture(); data["broker"][1]["fee"] = -1
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["broker_match"])
        data = fixture(); data["deals"][1]["reason"] = 4
        self.assertFalse(q.inspect(**data)["trades"][0]["checks"]["broker_match"])

    def test_duplicate_deal_deduplicated_but_conflict_fails(self):
        data = fixture(); data["deals"].append(copy.deepcopy(data["deals"][0]))
        result = q.inspect(**data)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["summary"]["runtime_duplicates"], 1)
        data["deals"][-1]["fee"] = -5
        self.assertEqual(q.inspect(**data)["status"], "NOT_READY")

    def test_expert_exit_and_conflicting_comment_do_not_become_timeout(self):
        for edit in ({"reason": 3, "comment": "timeout?"}, {"reason": 5, "comment": "[sl 1990]"}):
            data = fixture(); data["broker"][1].update(edit)
            result = q.inspect(**data)
            self.assertFalse(result["trades"][0]["checks"]["exit_reason"])
            self.assertEqual(result["trades"][0]["exit_category"], "Unknown")

    def test_open_position_is_pending_not_a_completed_sample(self):
        data = fixture(); data["deals"] = data["deals"][:1]; data["broker"] = data["broker"][:1]
        result = q.inspect(**data)
        self.assertEqual(result["summary"]["complete"], 0)
        self.assertIsNone(result["trades"][0]["checks"]["exit_reason"])
        self.assertEqual(next(r for r in result["coverage"] if r["check"] == "exit_reason")["applicable_n"], 0)

    def test_partial_closes_and_separate_fills_keep_costs(self):
        data = fixture(); data["deals"][1]["volume"] = .4
        extra = {**data["deals"][1], "deal_id": "3", "order_id": "O3", "volume": .6, "profit": -2, "fee": -.1}
        data["deals"].append(extra)
        data["broker"] = copy.deepcopy(data["deals"])
        for row in data["broker"][1:]:
            row["reason"] = 5
        result = q.inspect(**data)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["trades"][0]["recorded_net_pnl"], 6.4)

    def test_broker_only_deal_and_empty_inputs_fail(self):
        data = fixture(); data["broker"].append({**data["broker"][0], "deal_id": "99"})
        self.assertIn("broker_only_deal", q.inspect(**data)["summary"]["issue_counts"])
        self.assertEqual(q.inspect([])["status"], "NOT_READY")

    def test_different_versions_are_not_accepted_as_one_frozen_period(self):
        data = fixture(); second = fixture()
        second["manifests"][0]["model_sha256"] = "e"*64
        second["manifests"][0]["run_id"] = "run-B"
        for row in second["orders"]+second["signals"]:
            row["model_sha256"] = "e"*64; row["run_id"] = "run-B"
            if "order_id" in row:
                row["order_id"] = "B"+row["order_id"]
            if "deal_id" in row:
                row["deal_id"] = "B"+row["deal_id"]
            if "position_ticket" in row:
                row["position_ticket"] = "P2"
        for row in second["deals"]+second["broker"]:
            row["position_id"] = "P2"; row["deal_id"] = "B"+row["deal_id"]; row["order_id"] = "B"+row["order_id"]
        for key in ("deals", "orders", "signals", "manifests", "broker"):
            data[key].extend(second[key])
        self.assertIn("mixed_strategy_versions", q.inspect(**data)["summary"]["issue_counts"])


class ExportTests(unittest.TestCase):
    def test_export_reproduce_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = fixture(); paths = {}
            for kind in ("deals", "orders", "signals", "broker"):
                path = root/(kind+".jsonl")
                path.write_text("".join(json.dumps(r)+"\n" for r in data[kind]), encoding="utf-8")
                paths["broker_deals" if kind == "broker" else kind] = [str(path)]
            manifest_file = root/"run.json"
            manifest_file.write_text(json.dumps(data["manifests"][0]), encoding="utf-8")
            paths["manifests"] = [str(manifest_file)]
            before = {p: Path(p).read_bytes() for values in paths.values() for p in values}
            result = q.run(cli(root, **paths))
            self.assertEqual(result["status"], "PASS")
            manifest = json.loads((root/"quality"/"manifest.json").read_text(encoding="utf-8"))
            command = manifest["reproduce_argv"]; command[0] = sys.executable
            subprocess.run(command, cwd=root/"quality", check=True, stdout=subprocess.DEVNULL)
            for name, sha in manifest["output_sha256"].items():
                self.assertEqual(q.digest((root/"quality"/name).read_bytes()), sha)
                self.assertEqual((root/"quality"/name).read_bytes(), (root/"quality"/"reproduced"/name).read_bytes())
            for path, original in before.items():
                self.assertEqual(Path(path).read_bytes(), original)
            with self.assertRaises(FileExistsError):
                q.run(cli(root, **paths))

    def test_holdout_and_same_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                q.run(cli(root, deals=[str(root/"d.jsonl")], broker_deals=[str(root/"d.jsonl")]))
        with patch("pathlib.Path.open", side_effect=AssertionError("must not open")):
            with self.assertRaises(PermissionError):
                q.run(cli(Path("."), deals=[str(q.ROOT/"future_holdout"/"gold_s4_v4"/"x.jsonl")]))

    def test_rejected_last_line_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root/"deals.jsonl"
            path.write_text(json.dumps(fixture()["deals"][0]), encoding="utf-8")
            result = q.run(cli(root, deals=[str(path)]))
            self.assertEqual(result["status"], "NOT_READY")
            self.assertIn("input_capture", result["summary"]["issue_counts"])

    def test_reproduction_seal_detects_changed_retained_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/"d.jsonl"
            source.write_text("".join(json.dumps(r)+"\n" for r in fixture()["deals"]), encoding="utf-8")
            q.run(cli(root, deals=[str(source)]))
            seal = root/"quality"/"manifest.json"
            retained = root/"quality"/"inputs"/"deals_000.jsonl"
            retained.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                q.run(cli(root, deals=[str(retained)], input_seal=str(seal), output=str(root/"second")))
            self.assertFalse((root/"second").exists())

    def test_sealed_v2_legacy_import_cannot_become_verified(self):
        from test_forward_edge_diagnosis_v3 import make_v2
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); make_v2(root)
            result = q.run(cli(root, v2_dir=str(root/"v2")))
            self.assertEqual(result["status"], "NOT_READY")
            self.assertEqual(next(r for r in result["coverage"] if r["check"] == "version")["passed_n"], 0)


if __name__ == "__main__":
    unittest.main()
