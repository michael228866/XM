import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from calendar_values_qualify_v1 import COLUMNS, check, main, sha256

HEADER = ",".join(COLUMNS)
NFP, UNEMP, CRUDE = 840030016, 840030015, 840200001
EXPORT = 1763000000
R1, R2 = 1700000000, 1702600000


def row(vid, event, sched, period, rev, actual, forecast, prev="1", rprev="", name="Nonfarm"):
    return f"{vid},{event},3,{sched},{period},{rev},{actual},{forecast},{prev},{rprev},1,{EXPORT},{name}"


class QualifyTests(unittest.TestCase):
    def run_check(self, lines, encoding="utf-8", raw=None, **kw):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "wb") as f:
            f.write(raw if raw is not None else "\n".join([HEADER] + lines).encode(encoding) + b"\n")
        self.addCleanup(os.remove, path)
        before = sha256(path)
        report = check(path, **kw)
        self.assertEqual(sha256(path), before)
        return report

    def test_clean_file_zero_is_valid_and_missing_forecast_is_not_zero(self):
        r = self.run_check([
            row(1, NFP, R1, R1, 0, "0.000000", "180.000000"),     # actual 0 is a real value
            row(2, NFP, R2, R2, 0, "250.000000", ""),            # missing forecast stays missing
            row(3, UNEMP, R1, R1, 0, "4.300000", "4.100000", name="Unemployment"),
            row(4, CRUDE, R1, R1, 0, "-1.200000", "0.500000", name="Crude"),
        ])
        self.assertTrue(r["format_ok"], r["errors"])
        self.assertTrue(r["data_ok"])
        self.assertEqual(r["exploratory_usable_rows"], 2)
        self.assertEqual(r["excluded"]["缺 forecast"], 1)
        self.assertEqual(r["excluded"]["不在極性表"], 1)
        self.assertEqual(r["coverage"][str(NFP)]["zero_actual"], 1)
        self.assertEqual(r["coverage"][str(NFP)]["forecast_present"], 1)
        self.assertTrue(r["point_in_time"].startswith("Unknown"))
        self.assertEqual(r["time_range"][0], "2023-11-14 22:13:20 (server)")   # epoch formatted, no tz shift

    def test_revision_zero_does_not_make_point_in_time(self):
        r = self.run_check([row(1, NFP, R1, R1, 0, "250.000000", "180.000000")])
        self.assertEqual(len(r["unconfirmable"]), 3)
        self.assertTrue(r["point_in_time"].startswith("Unknown"))

    def test_different_revisions_keep_lowest_only(self):
        r = self.run_check([
            row(1, NFP, R1, R1, 0, "250.000000", "180.000000"),
            row(2, NFP, R1, R1, 1, "240.000000", "180.000000"),
        ])
        self.assertTrue(r["format_ok"])
        self.assertEqual(r["multi_revision_periods"], 1)
        self.assertEqual(r["exploratory_usable_rows"], 1)
        self.assertEqual(r["excluded"]["同期較高 revision"], 1)

    def test_duplicates_and_conflicts_fail(self):
        r = self.run_check([
            row(1, NFP, R1, R1, 0, "250.000000", "180.000000"),
            row(1, NFP, R1, R1, 0, "250.000000", "180.000000"),   # exact duplicate (and duplicate value_id)
            row(5, NFP, R1, R1, 0, "260.000000", "180.000000"),   # same event/period/revision, other actual
        ])
        self.assertFalse(r["format_ok"])
        self.assertEqual(r["exact_duplicates"], 1)
        self.assertEqual(r["duplicate_value_ids"], 1)
        self.assertEqual(r["version_conflicts"], 1)
        self.assertEqual(r["exploratory_usable_rows"], 0)

    def test_bad_format(self):
        r = self.run_check([
            row(1, NFP, R1, R1, 0, "abc", "180.000000"),
            row(2, NFP, R2, R2, 0, "-9223372036854.775808", "180.000000"),   # LONG_MIN scaled, not blanked
            "3,840030016,3",
        ])
        self.assertFalse(r["format_ok"])
        self.assertEqual(r["bad_rows"], {"型別錯誤": 2, "欄位數錯誤": 1})
        self.assertEqual(r["parsed_rows"], 0)

    def test_wrong_header_and_encoding(self):
        r = self.run_check([], raw=b"server_time,event_id,actual\n1,2,3\n")
        self.assertFalse(r["format_ok"])
        r = self.run_check([row(1, NFP, R1, R1, 0, "1.0", "1.0", name="非農")], encoding="gbk")
        self.assertFalse(r["format_ok"])
        self.assertIn("gbk", r["errors"][0])

    def test_explicit_gbk_ok(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "wb") as f:
            f.write(("\n".join([HEADER, row(1, NFP, R1, R1, 0, "1.0", "1.0", name="非農")]) + "\n").encode("gbk"))
        self.addCleanup(os.remove, path)
        self.assertTrue(check(path, "gbk")["format_ok"])

    def test_header_only_is_well_formed_but_not_usable(self):
        r = self.run_check([])
        self.assertTrue(r["format_ok"])
        self.assertFalse(r["data_ok"])
        self.assertEqual(r["data_problems"], ["沒有資料列"])

    def test_cli_exit_codes(self):
        for lines, code in (([], 3), ([row(1, NFP, R1, R1, 0, "250.0", "180.0")], 0), (["bad"], 2)):
            fd, path = tempfile.mkstemp(suffix=".csv")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join([HEADER] + lines) + "\n")
            self.addCleanup(os.remove, path)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([path]), code)

    def test_part_file_rejected_even_if_content_is_fine(self):
        fd, path = tempfile.mkstemp(suffix=".csv.part")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join([HEADER, row(1, NFP, R1, R1, 0, "250.0", "180.0")]) + "\n")
        self.addCleanup(os.remove, path)
        r = check(path)
        self.assertFalse(r["format_ok"])
        self.assertFalse(r["data_ok"])
        self.assertIn(".part", r["errors"][0])

    def test_low_importance_excluded(self):
        line = row(1, NFP, R1, R1, 0, "250.0", "180.0").replace(f"{NFP},3,", f"{NFP},1,")
        r = self.run_check([line])
        self.assertTrue(r["format_ok"])
        self.assertEqual(r["excluded"], {"importance 不是 3": 1})
        self.assertEqual(r["exploratory_usable_rows"], 0)
        self.assertFalse(r["data_ok"])

    def test_negative_revision_is_error_and_never_chosen(self):
        r = self.run_check([
            row(1, NFP, R1, R1, 0, "250.0", "180.0"),
            row(2, NFP, R1, R1, -1, "999.0", "180.0"),
        ])
        self.assertFalse(r["format_ok"])
        self.assertIn("revision=-1", r["errors"][0])
        self.assertEqual(r["exploratory_usable_rows"], 1)
        self.assertEqual(r["coverage"][str(NFP)]["rows"], 1)

    def test_unrepresentable_time_is_readable_error(self):
        r = self.run_check([row(1, NFP, 10**15, R1, 0, "250.0", "180.0")])
        self.assertFalse(r["format_ok"])
        self.assertIn("scheduled_time", r["errors"][0])

    def test_actual_after_export_is_contradiction(self):
        r = self.run_check([row(1, NFP, EXPORT + 3600, EXPORT + 3600, 0, "250.000000", "180.000000")])
        self.assertEqual(r["excluded"]["公布時間晚於匯出卻有 actual"], 1)


if __name__ == "__main__":
    unittest.main()
