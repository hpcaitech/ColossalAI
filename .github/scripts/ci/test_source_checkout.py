import io
import os
import shlex
import shutil
import subprocess
import tarfile
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
        self.git("-C", str(self.origin), "config", "uploadpack.allowFilter", "true")
        (self.origin / "tracked.txt").write_text("pinned source\n")
        self.git("-C", str(self.origin), "add", "tracked.txt")
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
        self.archive = self.root / "source.tar.gz"
        self.git(
            "-C", str(self.origin), "archive", "--format=tar.gz", "--prefix=source/", f"--output={self.archive}", "HEAD"
        )
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
        curl_start = script.index("curl --fail")
        tar_start = script.index("tar --extract", curl_start)
        script = script[:curl_start] + f'cp {shlex.quote(str(self.archive))} "$archive"\n' + script[tar_start:]
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
        self.assertNotIn("not allowed", fetched.stderr)
        self.assertNotIn("invalid object", fetched.stderr)
        source = Path(self.environment.read_text().strip().split("=", 1)[1])
        private = self.root / "private-copy"
        shutil.copytree(source, private)
        self.assertTrue((private / ".git").is_dir())
        self.assertFalse((private / ".git/objects/info/alternates").exists())
        self.assertEqual(self.git("-C", str(private), "rev-parse", "HEAD"), self.sha)
        self.assertEqual(self.git("-C", str(private), "-c", "protocol.allow=never", "status", "--porcelain"), "")
        self.assertEqual((private / "tracked.txt").read_text(), "pinned source\n")
        blob = self.git("-C", str(self.origin), "rev-parse", "HEAD:tracked.txt")
        local_blob = subprocess.run(
            [
                "git",
                "-C",
                str(private),
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.file.allow=never",
                "cat-file",
                "-e",
                blob,
            ],
            capture_output=True,
        )
        self.assertEqual(local_blob.returncode, 0, "the archive must reconstruct blob objects for offline use")
        (private / "tracked.txt").write_text("changed source\n")
        self.assertIn(
            "tracked.txt", self.git("-C", str(private), "-c", "protocol.allow=never", "status", "--porcelain")
        )
        self.assertIn("Archive SHA256:", fetched.stdout)

    def test_missing_commit_never_falls_back_to_current_branch(self):
        fetched = self.fetch("0" * 40)
        self.assertNotEqual(fetched.returncode, 0)
        self.assertEqual(self.environment.read_text(), "")

    def test_archive_must_match_the_requested_commit_tree(self):
        data = b"not the pinned source\n"
        with tarfile.open(self.archive, "w:gz") as archive:
            entry = tarfile.TarInfo("source/tracked.txt")
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
        fetched = self.fetch(self.sha)
        self.assertNotEqual(fetched.returncode, 0)
        self.assertEqual(self.environment.read_text(), "")


if __name__ == "__main__":
    unittest.main()
