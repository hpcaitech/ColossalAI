import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_test_ci import run_test_ci


class RunTestCiTest(unittest.TestCase):
    def test_missing_script_fails_without_starting_bash(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("run_test_ci.subprocess.run") as run:
            result = run_test_ci(Path(temp_dir) / "test_ci.sh")

        self.assertEqual(result, 1)
        run.assert_not_called()

    def test_empty_script_fails_without_starting_bash(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("run_test_ci.subprocess.run") as run:
            script = Path(temp_dir) / "test_ci.sh"
            script.touch()
            result = run_test_ci(script)

        self.assertEqual(result, 1)
        run.assert_not_called()

    def test_nonempty_script_runs_from_its_example_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("run_test_ci.subprocess.run") as run:
            script = Path(temp_dir) / "test_ci.sh"
            script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
            run.return_value = subprocess.CompletedProcess(["bash", script.name], 0)

            result = run_test_ci(script)

        self.assertEqual(result, 0)
        run.assert_called_once_with(["bash", "test_ci.sh"], cwd=script.parent, check=False)

    def test_script_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("run_test_ci.subprocess.run") as run:
            script = Path(temp_dir) / "test_ci.sh"
            script.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
            run.return_value = subprocess.CompletedProcess(["bash", script.name], 7)

            result = run_test_ci(script)

        self.assertEqual(result, 7)


if __name__ == "__main__":
    unittest.main()
