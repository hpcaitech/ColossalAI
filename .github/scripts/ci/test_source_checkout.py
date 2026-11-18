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
        (self.origin / "executable.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.origin / "executable.sh").chmod(0o755)
        (self.origin / "linked.txt").symlink_to("tracked.txt")
        self.git("-C", str(self.origin), "add", "--all")
        self.git("-C", str(self.origin), "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},vendor/submodule")
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

    @staticmethod
    def step_script(name):
        workflow = Path(__file__).resolve().parents[2] / "workflows/ci_reusable_gpu_batches.yml"
        step = workflow.read_text().split(f"- name: {name}\n", 1)[1].split("\n      - name:", 1)[0]
        return textwrap.dedent(step.split("run: |\n", 1)[1])

    def run_script(self, script, **environment):
        return subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "CI_SOURCE_SHA": self.sha,
                "RUNNER_TEMP": str(self.root),
                "GITHUB_ENV": str(self.environment),
                **environment,
            },
            capture_output=True,
            text=True,
            timeout=30,
        )

    def checkout(self, sha=None, wrapper=None):
        # Exercise the production shell against a real local Git server fixture.
        script = self.step_script("Fetch and verify the exact source commit")
        script = script.replace('"https://github.com/$GITHUB_REPOSITORY.git"', shlex.quote(self.origin.as_uri()))
        environment = {}
        if wrapper is not None:
            binary = self.root / "bin"
            binary.mkdir()
            executable = binary / "git"
            executable.write_text(
                "#!/usr/bin/env python3\nimport os, subprocess, sys\n"
                + "real_git = "
                + repr(shutil.which("git"))
                + "\n"
                + textwrap.dedent(wrapper)
            )
            executable.chmod(0o755)
            environment["PATH"] = str(binary) + os.pathsep + os.environ["PATH"]
        return self.run_script(script, CI_SOURCE_SHA=sha or self.sha, **environment)

    def assert_rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.environment.read_text(), "")

    def test_pinned_checkout_retains_git_metadata_in_private_copy(self):
        checked_out = self.checkout()
        self.assertEqual(checked_out.returncode, 0, checked_out.stderr)
        source = Path(self.environment.read_text().strip().split("=", 1)[1])
        private = self.root / "private-copy"
        shutil.copytree(source, private, symlinks=True)
        self.assertTrue((private / ".git").is_dir())
        self.assertFalse((private / ".git/objects/info/alternates").exists())
        self.assertEqual(self.git("-C", str(private), "rev-parse", "HEAD"), self.sha)
        self.assertEqual(self.git("-C", str(private), "-c", "protocol.allow=never", "status", "--porcelain"), "")
        self.assertEqual((private / "tracked.txt").read_text(), "pinned source\n")
        self.assertTrue(os.access(private / "executable.sh", os.X_OK))
        self.assertTrue((private / "linked.txt").is_symlink())
        self.assertTrue(self.git("-C", str(private), "ls-files", "--stage", "vendor/submodule").startswith("160000 "))
        self.assertEqual(
            self.git("-C", str(private), "-c", "protocol.allow=never", "show", "HEAD:tracked.txt"), "pinned source"
        )
        config = self.git("-C", str(private), "config", "--local", "--list").lower()
        self.assertNotIn("extraheader", config)
        self.assertNotIn("credential", config)
        self.assertNotIn("promisor", config)
        (private / "tracked.txt").write_text("changed source\n")
        self.assertIn(
            "tracked.txt", self.git("-C", str(private), "-c", "protocol.allow=never", "status", "--porcelain")
        )
        self.assertIn("Source commit: " + self.sha, checked_out.stdout)

    def test_missing_commit_never_falls_back_to_current_branch(self):
        self.assert_rejected(self.checkout("0" * 40))

    def test_invalid_commit_is_rejected_before_fetch(self):
        self.assert_rejected(self.checkout("main"))
        self.assertEqual(list(self.root.glob("colossalai-source.*")), [])

    def test_transient_fetch_failure_retries_the_same_commit(self):
        marker = self.root / "fetch-attempts"
        result = self.checkout(
            wrapper=f"""
            from pathlib import Path
            if 'fetch' in sys.argv:
                path = Path({str(marker)!r})
                previous = path.read_text() if path.exists() else ''
                path.write_text(previous + sys.argv[-1] + '\\n')
                if not previous:
                    sys.exit(1)
            os.execv(real_git, [real_git, *sys.argv[1:]])
        """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker.read_text().splitlines(), [self.sha, self.sha])

    def test_fetched_commit_mismatch_is_rejected(self):
        result = self.checkout(
            wrapper="""
            if sys.argv[-2:] == ['rev-parse', 'FETCH_HEAD']:
                print('0' * 40)
                sys.exit(0)
            os.execv(real_git, [real_git, *sys.argv[1:]])
        """
        )
        self.assert_rejected(result)

    def test_modified_checkout_is_rejected(self):
        result = self.checkout(
            wrapper="""
            from pathlib import Path
            if 'checkout' in sys.argv:
                subprocess.run([real_git, *sys.argv[1:]], check=True)
                repo = Path(sys.argv[sys.argv.index('-C') + 1])
                (repo / 'tracked.txt').write_text('uncommitted modification')
                sys.exit(0)
            os.execv(real_git, [real_git, *sys.argv[1:]])
        """
        )
        self.assert_rejected(result)
        self.assertIn("differs from the pinned commit", result.stderr)


if __name__ == "__main__":
    unittest.main()
