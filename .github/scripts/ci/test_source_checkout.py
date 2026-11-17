import hashlib
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
        self.output = self.root / "github-output"
        self.output.touch()

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
                "GITHUB_OUTPUT": str(self.output),
                **environment,
            },
            capture_output=True,
            text=True,
            timeout=30,
        )

    def package(self, sha=None):
        # Run the actual producer shell against a local Git server fixture.
        # GitHub artifact transport is replaced only by a byte-for-byte copy.
        script = self.step_script("Package the exact GitHub source commit")
        script = script.replace('"https://github.com/$GITHUB_REPOSITORY.git"', shlex.quote(self.origin.as_uri()))
        result = self.run_script(script, CI_SOURCE_SHA=sha or self.sha)
        if result.returncode == 0:
            outputs = dict(line.split("=", 1) for line in self.output.read_text().splitlines())
            self.download = self.root / "download"
            self.download.mkdir()
            self.archive = self.download / "source.tar.gz"
            shutil.copyfile(outputs["archive"], self.archive)
            self.archive_sha256 = outputs["archive_sha256"]
        return result

    def restore(self, sha=None):
        return self.run_script(
            self.step_script("Restore and verify the exact source commit"),
            CI_SOURCE_SHA=sha or self.sha,
            CI_SOURCE_ARTIFACT_DIR=str(self.download),
            SOURCE_ARCHIVE_SHA256=self.archive_sha256,
        )

    def alter_archive(self, mutate):
        unpacked = self.root / "unpacked"
        unpacked.mkdir()
        # Use the same tar executable as production, including on runners
        # whose system Python predates tarfile extraction filters.
        subprocess.run(["tar", "-xzf", str(self.archive), "-C", str(unpacked)], check=True)
        mutate(unpacked / "repo")
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(unpacked / "repo", arcname="repo")
        # Simulate an intact transport carrying an incorrect producer payload.
        self.archive_sha256 = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def assert_rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.environment.read_text(), "")

    def test_pinned_checkout_retains_git_metadata_in_private_copy(self):
        packaged = self.package()
        self.assertEqual(packaged.returncode, 0, packaged.stderr)
        restored = self.restore()
        self.assertEqual(restored.returncode, 0, restored.stderr)
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
        self.assertIn("Archive SHA256:", restored.stdout)

    def test_missing_commit_never_falls_back_to_current_branch(self):
        self.assert_rejected(self.package("0" * 40))
        self.assertEqual(self.output.read_text(), "")

    def test_restore_rejects_a_different_commit(self):
        self.assertEqual(self.package().returncode, 0)
        self.assert_rejected(self.restore("0" * 40))

    def test_corrupted_transport_fails_digest_check_before_extraction(self):
        self.assertEqual(self.package().returncode, 0)
        with self.archive.open("ab") as archive:
            archive.write(b"corrupted transport")
        self.assert_rejected(self.restore())
        self.assertFalse((self.download / "repo").exists())

    def test_archive_must_match_the_requested_commit_tree(self):
        self.assertEqual(self.package().returncode, 0)
        self.alter_archive(lambda repo: (repo / "tracked.txt").write_text("not the pinned source\n"))
        restored = self.restore()
        self.assert_rejected(restored)
        self.assertIn("does not match commit tree", restored.stderr)

    def test_missing_git_objects_are_rejected(self):
        self.assertEqual(self.package().returncode, 0)
        self.alter_archive(lambda repo: shutil.rmtree(repo / ".git/objects"))
        self.assert_rejected(self.restore())

    def test_repositories_using_external_alternates_are_rejected(self):
        self.assertEqual(self.package().returncode, 0)
        self.alter_archive(
            lambda repo: (repo / ".git/objects/info/alternates").write_text(str(self.origin / ".git/objects"))
        )
        self.assert_rejected(self.restore())


if __name__ == "__main__":
    unittest.main()
