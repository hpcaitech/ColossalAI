import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class CpuTracebackTests(unittest.TestCase):
    def run_script(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        script = root / "worker.py"
        script.write_text(textwrap.dedent(source))
        logs = root / "tracebacks"
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parent / "cpu_tracebacks"),
            "COLOSSAL_CI_TRACEBACK_DIR": str(logs),
            "COLOSSAL_CI_TRACEBACK_SECONDS": "0.1",
            "PYTEST_CURRENT_TEST": "traceback smoke test",
        }
        result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=15)
        return result, [path.read_text() for path in logs.glob("*.log")]

    def test_spawned_worker_has_its_own_stack_log(self):
        result, logs = self.run_script(
            """
            import multiprocessing as mp
            import time

            def wait_in_worker():
                time.sleep(0.5)

            if __name__ == "__main__":
                worker = mp.get_context("spawn").Process(target=wait_in_worker)
                worker.start()
                worker.join()
                raise SystemExit(worker.exitcode)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any("wait_in_worker" in log for log in logs), logs)
        self.assertTrue(any("join" in log for log in logs), logs)
        self.assertGreaterEqual(len(logs), 2)
        self.assertTrue(all("test=traceback smoke test" in log for log in logs), logs)

    def test_failure_exit_status_is_preserved(self):
        result, logs = self.run_script("raise SystemExit(7)")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertTrue(any("EXIT" in log for log in logs), logs)


if __name__ == "__main__":
    unittest.main()
