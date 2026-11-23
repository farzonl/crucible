"""Focused CTS parser, history, and Pages preservation regressions."""

import io
import json
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from cts_data import preserve_cts_data
from record_vulkan_cts import build_tree, parse_qpa, record, select_cases, summarize
import record_bench_history
import record_offload_compliance


def qpa_result(name: str, code: str) -> str:
    return (f'#beginTestCaseResult {name}\n'
            f'<TestCaseResult><Result StatusCode="{code}">details &amp; text</Result>'
            '</TestCaseResult>\n#endTestCaseResult\n')


class CTSResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qpa = self.root / "results.qpa"

    def test_status_denominator_and_directory_aggregation(self) -> None:
        codes = ["Pass", "Fail", "NotSupported", "QualityWarning",
                 "CompatibilityWarning", "InternalError"]
        names = [f"dEQP-VK.spirv_assembly.group.case{index}"
                 for index in range(len(codes) + 1)]
        self.qpa.write_text("".join(qpa_result(name, code)
                                   for name, code in zip(names, codes)))
        tests = parse_qpa(self.qpa, names)
        summary = summarize(tests)
        self.assertEqual((summary["total"], summary["passed"], summary["failed"],
                          summary["warnings"], summary["notRun"]), (7, 1, 2, 2, 1))
        self.assertEqual(summary["passRate"], 14.29)
        self.assertEqual(tests[0]["details"], "details & text")
        tree = build_tree(tests)
        self.assertEqual(tree["total"], 7)
        self.assertEqual(tree["children"][0]["children"][0]["passRate"], 14.29)
        self.assertEqual(tree["children"][0]["children"][0]["children"][0]["code"], "Pass")

    def test_interrupted_and_terminated_results(self) -> None:
        names = ["dEQP-VK.a.timeout", "dEQP-VK.a.truncated", "dEQP-VK.a.unrun"]
        self.qpa.write_text(
            f"#beginTestCaseResult {names[0]}\n"
            "#terminateTestCaseResult Watchdog timeout occurred.\n"
            f"#beginTestCaseResult {names[1]}\n<TestCaseResult>")
        self.assertEqual([test["code"] for test in parse_qpa(self.qpa, names)],
                         ["Timeout", "InternalError", "NotRun"])

    def test_bad_qpa_is_not_success(self) -> None:
        name = "dEQP-VK.a.test"
        for content in ("#beginSession\n",
                        qpa_result(name, "Unknown"),
                        qpa_result(name, "Pass") * 2,
                        qpa_result("dEQP-VK.other.test", "Pass"),
                        f"#beginTestCaseResult {name}\n<broken\n#endTestCaseResult\n"):
            with self.subTest(content=content):
                self.qpa.write_text(content)
                with self.assertRaises((ValueError, KeyError, ET.ParseError)):
                    parse_qpa(self.qpa, [name])

    def test_selection_resolves_includes_and_samples_entire_prefix(self) -> None:
        source = self.root / "mustpass.txt"
        source.write_text("cases.txt\n")
        names = [f"dEQP-VK.spirv_assembly.a.case{index:02}" for index in range(20)]
        (self.root / "cases.txt").write_text("\n".join(names + ["dEQP-VK.api.other"]))
        self.assertEqual(select_cases(source, "dEQP-VK.spirv_assembly.", 4),
                         [names[index] for index in (0, 5, 10, 15)])
        self.assertEqual(len(select_cases(source, "dEQP-VK.spirv_assembly.", 256)), 20)
        with self.assertRaises(ValueError):
            select_cases(source, "absent.", 4)
        (self.root / "cases.txt").write_text("mustpass.txt")
        with self.assertRaises(ValueError):
            select_cases(source, "dEQP-VK.", 4)

    def test_commit_rerun_merges_cache_preserves_other_data(self) -> None:
        name = "dEQP-VK.a.test"
        caselist = self.root / "cases.txt"
        caselist.write_text(name + "\n")
        self.qpa.write_text(qpa_result(name, "Fail"))
        data = self.root / "data"
        data.mkdir()
        other = data / "bench-history.json"
        other.write_text('[{"keep":true}]')
        cache = self.root / "cache.json"
        metadata = {"sha": "abc1234", "timestamp": "2026-11-22T00:00:00Z"}
        record(self.qpa, caselist, metadata, data, cache)
        cache.write_text(json.dumps([{"sha": "older", "timestamp": "2026-11-21T00:00:00Z"}]))
        self.qpa.write_text(qpa_result(name, "Pass"))
        metadata["timestamp"] = "2026-11-22T01:00:00Z"
        latest = record(self.qpa, caselist, metadata, data, cache)
        history = json.loads((data / "vulkan-cts/history.json").read_text())
        self.assertEqual([entry["sha"] for entry in history], ["older", "abc1234"])
        self.assertEqual(latest["passed"], 1)
        self.assertEqual(latest["failures"], [])
        self.assertNotIn("tree", history[-1])
        self.assertEqual(other.read_text(), '[{"keep":true}]')
        self.assertEqual(json.loads(cache.read_text()), history)

    def test_publishers_preserve_cts_and_surface_transport_errors(self) -> None:
        data = self.root / "data"
        payloads = {"latest.json": {"tree": {"total": 2}, "tests": ["keep"]},
                    "history.json": [{"sha": "old"}], "badge.json": {"message": "50%"}}

        def response(url: str, timeout: int) -> io.BytesIO:
            return io.BytesIO(json.dumps(payloads[url.rsplit("/", 1)[1]]).encode())

        with patch("cts_data.urllib.request.urlopen", side_effect=response):
            preserve_cts_data("https://example.test", data)
        for name, value in payloads.items():
            self.assertEqual(json.loads((data / "vulkan-cts" / name).read_text()), value)
        with patch("cts_data.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(urllib.error.URLError):
                preserve_cts_data("https://example.test", self.root / "another")

    def test_both_existing_pages_publishers_retain_complete_cts_dataset(self) -> None:
        site = self.root / "pages"
        site.mkdir()
        (site / "vulkan-cts.html").write_text("dashboard")
        lit = self.root / "lit.json"
        lit.write_text(json.dumps({"tests": [{"name": "suite :: test", "code": "PASS"}]}))
        bench = self.root / "bench.json"
        bench.write_text(json.dumps({"results": [{"ran": True, "medianMs": 1}]}))
        payload = {"latest.json": {"tests": [{"code": "Fail"}], "tree": {"total": 1}},
                   "history.json": [{"sha": "prior"}], "badge.json": {"message": "0%"}}

        def response(url: str, timeout: int) -> io.BytesIO:
            return io.BytesIO(json.dumps(payload[url.rsplit("/", 1)[1]]).encode())

        for module, option, source in (
                (record_offload_compliance, "--lit-json", lit),
                (record_bench_history, "--bench-json", bench)):
            with self.subTest(publisher=module.__name__):
                output = self.root / module.__name__
                argv = [module.__name__, option, str(source), "--site-src", str(site),
                        "--out-dir", str(output), "--sha", "abc",
                        "--pages-url", "https://example.test"]
                with patch("sys.argv", argv), \
                        patch("record_offload_compliance.fetch_json", return_value=None), \
                        patch("record_bench_history.fetch_json_url", return_value=None), \
                        patch("cts_data.urllib.request.urlopen", side_effect=response):
                    module.main()
                for name, value in payload.items():
                    self.assertEqual(
                        json.loads((output / "data/vulkan-cts" / name).read_text()), value)
                self.assertEqual((output / "vulkan-cts.html").read_text(), "dashboard")


if __name__ == "__main__":
    unittest.main()
