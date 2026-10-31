import tempfile
import textwrap
import unittest
from pathlib import Path

from check_pr_runner_policy import validate_fork_workflow, validate_private_pr_runners


class PullRequestRunnerPolicyTest(unittest.TestCase):
    def write_workflow(self, directory: Path, name: str, source: str) -> Path:
        path = directory / name
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def test_private_pull_request_runner_requires_head_repository_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflows = Path(temp_dir)
            self.write_workflow(
                workflows,
                "unsafe.yml",
                """
                on: pull_request
                jobs:
                  test:
                    runs-on: [self-hosted, linux]
                    steps: []
                """,
            )

            errors = validate_private_pr_runners(workflows)

        self.assertEqual(len(errors), 1)
        self.assertIn("unsafe.yml:test", errors[0])

    def test_guarded_private_pull_request_runner_is_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflows = Path(temp_dir)
            self.write_workflow(
                workflows,
                "guarded.yml",
                """
                on:
                  workflow_dispatch:
                  pull_request:
                jobs:
                  test:
                    if: >-
                      github.event_name != 'pull_request' ||
                      github.event.pull_request.head.repo.full_name == github.repository
                    runs-on: project-runner
                    steps: []
                """,
            )

            errors = validate_private_pr_runners(workflows)

        self.assertEqual(errors, [])

    def test_non_pull_request_private_runner_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflows = Path(temp_dir)
            self.write_workflow(
                workflows,
                "schedule.yml",
                """
                on: workflow_dispatch
                jobs:
                  test:
                    runs-on: self-hosted
                    steps: []
                """,
            )

            errors = validate_private_pr_runners(workflows)

        self.assertEqual(errors, [])

    def test_private_runner_after_merge_is_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflows = Path(temp_dir)
            self.write_workflow(
                workflows,
                "release.yml",
                """
                on:
                  pull_request:
                    types: [closed]
                jobs:
                  release:
                    if: github.event.pull_request.merged == true
                    runs-on: self-hosted
                    steps: []
                """,
            )

            errors = validate_private_pr_runners(workflows)

        self.assertEqual(errors, [])

    def test_fork_workflow_rejects_private_runner_and_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow = self.write_workflow(
                Path(temp_dir),
                "fork_pr_checks.yml",
                """
                on: pull_request
                permissions:
                  contents: write
                jobs:
                  lint:
                    runs-on: self-hosted
                    steps:
                      - uses: actions/checkout@v7
                  cpu-unit:
                    runs-on: ubuntu-24.04
                    steps:
                      - run: echo '${{ secrets.TOKEN }}'
                """,
            )

            errors = validate_fork_workflow(workflow)

        self.assertTrue(any("contents: read" in error for error in errors))
        self.assertTrue(any("must not reference secrets" in error for error in errors))
        self.assertTrue(any("GitHub-hosted" in error for error in errors))
        self.assertTrue(any("persist-credentials" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
