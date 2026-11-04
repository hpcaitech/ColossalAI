import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "source checkout runs in Linux Bash")
class SourceCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.origin = self.root / "origin"
        self.git("init", "--quiet", str(self.origin))
        (self.origin / "tracked.txt").write_text("pinned source\n")
        self.git("-C", str(self.origin), "add", "tracked.txt")
        self.git(
            "-C",
            str(self.origin),
            "-c",
            "user.name=CI checkout test",
            "-c",
            "user.email=ci-checkout@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "test source",
        )
        self.sha = self.git("-C", str(self.origin), "rev-parse", "HEAD")
        self.environment = self.root / "github-env"
        self.environment.touch()

    @staticmethod
    def git(*args):
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()

    def fetch(self, sha):
        workflow = Path(__file__).resolve().parents[2] / "workflows/ci_reusable_gpu_batches.yml"
        step = workflow.read_text().split("- name: Fetch the exact GitHub source commit", 1)[1]
        step = step.split("- name: Validate the batch runner", 1)[0]
        script = textwrap.dedent(step.split("run: |\n", 1)[1])
        # Exercise the production shell against a local Git server fixture,
        # without network access or credentials during the CPU gate.
        script = script.replace("${{ inputs.layer }}", "test-source")
        script = script.replace('"https://github.com/$GITHUB_REPOSITORY.git"', shlex.quote(self.origin.as_uri()))
        return subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "CI_SOURCE_SHA": sha,
                "RUNNER_TEMP": str(self.root),
                "GITHUB_ENV": str(self.environment),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_pinned_checkout_retains_git_metadata_in_private_copy(self):
        fetched = self.fetch(self.sha)
        self.assertEqual(fetched.returncode, 0, fetched.stderr)
        source = Path(self.environment.read_text().strip().split("=", 1)[1])
        private = self.root / "private-copy"
        shutil.copytree(source, private)
        self.assertTrue((private / ".git").is_dir())
        self.assertFalse((private / ".git/objects/info/alternates").exists())
        self.assertEqual(self.git("-C", str(private), "rev-parse", "HEAD"), self.sha)
        self.assertEqual(self.git("-C", str(private), "status", "--porcelain"), "")
        self.assertEqual((private / "tracked.txt").read_text(), "pinned source\n")
        self.assertIn("Archive SHA256:", fetched.stdout)

    def test_missing_commit_never_falls_back_to_current_branch(self):
        fetched = self.fetch("0" * 40)
        self.assertNotEqual(fetched.returncode, 0)
        self.assertEqual(self.environment.read_text(), "")


if __name__ == "__main__":
    unittest.main()
