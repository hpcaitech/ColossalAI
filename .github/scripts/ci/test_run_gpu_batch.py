import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from run_gpu_batch import GPU_COUNTS, find_idle, parse_pool, select_idle, stream_process
from run_with_timeout import run_with_timeout

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
GPU_C = "GPU-00000000-0000-0000-0000-000000000003"


class GpuSelectionTests(unittest.TestCase):
    def test_selects_requested_idle_devices_in_index_order(self):
        rows = f"2,{GPU_C},4,0\n0,{GPU_A},4,0\n1,{GPU_B},4,0"
        self.assertEqual(select_idle(rows, "", 2), [(0, GPU_A), (1, GPU_B)])

    def test_lists_all_idle_devices(self):
        rows = f"2,{GPU_C},4,0\n0,{GPU_A},4,0\n1,{GPU_B},257,0"
        self.assertEqual(find_idle(rows, ""), [(0, GPU_A), (2, GPU_C)])

    def test_respects_authorized_pool(self):
        rows = f"0,{GPU_A},4,0\n1,{GPU_B},4,0\n2,{GPU_C},4,0"
        self.assertEqual(select_idle(rows, "", 1, {2}), [(2, GPU_C)])

    def test_memory_utilization_and_processes_all_disqualify(self):
        rows = f"0,{GPU_A},257,0\n1,{GPU_B},4,1\n2,{GPU_C},4,0"
        with self.assertRaises(RuntimeError):
            select_idle(rows, GPU_C, 1)

    def test_unknown_occupancy_fails_closed(self):
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},N/A,0", "", 1)
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},4,0", "N/A", 1)

    def test_pool_validation(self):
        self.assertEqual(parse_pool("0,2"), {0, 2})
        for value in ("0,0", "0,a", "0, 1"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                parse_pool(value)

    def test_qwen2_batch_uses_four_gpus(self):
        self.assertEqual(GPU_COUNTS["9q"], 4)

    @unittest.skipUnless(os.name == "posix", "process groups require Linux")
    def test_exited_parent_does_not_wait_for_worker_stdout(self):
        command = [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('parent completed', flush=True); sys.exit(5)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "test.log"
            start = time.monotonic()
            self.assertEqual(stream_process(command, log, os.environ.copy()), 5)
            self.assertLess(time.monotonic() - start, 15)
            self.assertIn("parent completed", log.read_text())

    @unittest.skipUnless(os.name == "posix", "process groups require Linux")
    def test_timeout_reaps_worker_before_tee_eof(self):
        wrapper = Path(__file__).with_name("run_with_timeout.py")
        command = [
            "bash",
            "-o",
            "pipefail",
            "-c",
            '"$1" "$2" --timeout-seconds 0.2 -- "$1" -c '
            '"import subprocess, sys, time; '
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            'time.sleep(60)" | tee /dev/null',
            "test",
            sys.executable,
            str(wrapper),
        ]
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            self.assertEqual(stream_process(command, Path(directory) / "timeout.log", os.environ.copy()), 124)
            self.assertLess(time.monotonic() - start, 25)

    @unittest.skipUnless(os.name == "posix", "process groups require Linux")
    def test_timeout_preserves_normal_exit_status(self):
        self.assertEqual(run_with_timeout([sys.executable, "-c", "import sys; sys.exit(7)"], 5), 7)


if __name__ == "__main__":
    unittest.main()
