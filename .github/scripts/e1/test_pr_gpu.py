import copy
import json
import tempfile
import unittest
from pathlib import Path

from colossalai_suite import validate_report
from pr_gpu import has_colossalai_pass, has_gpu_pass, select_idle, validate_event

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
GPU_C = "GPU-00000000-0000-0000-0000-000000000003"


class PrGpuTests(unittest.TestCase):
    def test_colossalai_requires_real_success_marker(self):
        self.assertFalse(has_colossalai_pass(""))
        self.assertFalse(has_colossalai_pass(json.dumps({"result": "E1_GPU_SMOKE_PASS", "gpu_tested": True})))
        self.assertFalse(
            has_colossalai_pass(json.dumps({"result": "E1_COLOSSALAI_PASS", "gpu_tested": True, "tests_passed": 0}))
        )
        self.assertTrue(
            has_colossalai_pass(json.dumps({"result": "E1_COLOSSALAI_PASS", "gpu_tested": True, "tests_passed": 2}))
        )

    def test_junit_rejects_skips_failures_missing_and_extra_tests(self):
        good = '<testcase name="test_accelerator"/><testcase name="test_dp_plugin_dataloader"/>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            path.write_text("<testsuite>" + good + "</testsuite>")
            self.assertEqual(validate_report(path), 2)
            for bad in (
                "",
                good + '<testcase name="unexpected"/>',
                good.replace('name="test_accelerator"/>', 'name="test_accelerator"><skipped/></testcase>'),
                good.replace('name="test_accelerator"/>', 'name="test_accelerator"><failure/></testcase>'),
            ):
                path.write_text("<testsuite>" + bad + "</testsuite>")
                with self.assertRaises(RuntimeError):
                    validate_report(path)

    def test_busy_memory_is_not_idle_even_at_zero_utilization(self):
        rows = f"0,{GPU_A},26000,0\n1,{GPU_B},4,0\n2,{GPU_C},4,0"
        self.assertEqual(select_idle(rows, ""), [(1, GPU_B), (2, GPU_C)])

    def test_existing_process_disqualifies_gpu(self):
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},4,0\n1,{GPU_B},4,0", GPU_A)

    def test_unknown_occupancy_fails_closed(self):
        for rows, processes in [(f"0,{GPU_A},N/A,0", ""), (f"0,{GPU_A},4,0", "N/A")]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                select_idle(rows, processes)

    def test_duplicate_uuid_rejected(self):
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},4,0\n1,{GPU_A},4,0", "")

    def test_cpu_pass_or_empty_log_is_not_gpu_success(self):
        self.assertFalse(has_gpu_pass(""))
        self.assertFalse(has_gpu_pass(json.dumps({"result": "E1_CPU_CONTROL_PASS", "gpu_tested": False})))
        self.assertTrue(
            has_gpu_pass(
                json.dumps({"result": "E1_GPU_SMOKE_PASS", "gpu_tested": True, "backend": "nccl", "world_size": 2})
            )
        )

    def test_only_authorized_internal_pr_allowed(self):
        event = {
            "repository": {"full_name": "hpcaitech/ColossalAI"},
            "pull_request": {
                "head": {"repo": {"full_name": "hpcaitech/ColossalAI"}, "ref": "ci/e1-runner-bootstrap"},
                "base": {"ref": "main", "repo": {"full_name": "hpcaitech/ColossalAI"}},
                "user": {"login": "richardoo-707"},
            },
        }
        validate_event(event)
        for change in ("author", "branch", "base_branch"):
            other = copy.deepcopy(event)
            if change == "author":
                other["pull_request"]["user"]["login"] = "another-developer"
            elif change == "branch":
                other["pull_request"]["head"]["ref"] = "feature/another-internal-branch"
            else:
                other["pull_request"]["base"]["ref"] = "release/test"
            with self.subTest(allowed=change):
                validate_event(other)
        for change in ("fork", "repository", "base_repository"):
            other = copy.deepcopy(event)
            if change == "fork":
                other["pull_request"]["head"]["repo"]["full_name"] = "other/ColossalAI"
            elif change == "repository":
                other["repository"]["full_name"] = "other/ColossalAI"
            else:
                other["pull_request"]["base"]["repo"]["full_name"] = "other/ColossalAI"
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_event(other)


if __name__ == "__main__":
    unittest.main()
