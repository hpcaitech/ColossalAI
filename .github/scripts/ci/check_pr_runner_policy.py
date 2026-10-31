#!/usr/bin/env python3
"""Validate the repository's fork-safe pull request runner policy."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

GITHUB_HOSTED_LABEL = re.compile(r"^(ubuntu|windows|macos)-")
HEAD_REPOSITORY = "github.event.pull_request.head.repo.full_name"
CURRENT_REPOSITORY = "github.repository"
MERGED_PULL_REQUEST = "github.event.pull_request.merged == true"


def load_workflow(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        workflow = yaml.load(file, Loader=yaml.BaseLoader)
    if not isinstance(workflow, dict):
        raise ValueError("workflow root must be a mapping")
    return workflow


def event_enabled(workflow: dict[str, Any], event: str) -> bool:
    triggers = workflow.get("on")
    if isinstance(triggers, str):
        return triggers == event
    if isinstance(triggers, list):
        return event in triggers
    if isinstance(triggers, dict):
        return event in triggers
    return False


def runner_labels(runs_on: Any) -> list[str]:
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(label) for label in runs_on]
    return []


def is_github_hosted(runs_on: Any) -> bool:
    labels = runner_labels(runs_on)
    return len(labels) == 1 and GITHUB_HOSTED_LABEL.match(labels[0]) is not None


def workflow_paths(workflows_dir: Path) -> Iterable[Path]:
    return sorted((*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")))


def validate_private_pr_runners(workflows_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in workflow_paths(workflows_dir):
        try:
            workflow = load_workflow(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{path.name}: cannot parse workflow: {error}")
            continue
        if not event_enabled(workflow, "pull_request"):
            continue
        jobs = workflow.get("jobs", {})
        if not isinstance(jobs, dict):
            errors.append(f"{path.name}: jobs must be a mapping")
            continue
        for job_id, job in jobs.items():
            if not isinstance(job, dict) or is_github_hosted(job.get("runs-on")):
                continue
            condition = str(job.get("if", ""))
            guarded_head = HEAD_REPOSITORY in condition and CURRENT_REPOSITORY in condition
            trusted_after_merge = MERGED_PULL_REQUEST in condition
            if not guarded_head and not trusted_after_merge:
                errors.append(
                    f"{path.name}:{job_id}: a private pull_request runner must be limited "
                    "to a head repository matching github.repository or an already merged PR"
                )
    return errors


def validate_fork_workflow(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        workflow = load_workflow(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [f"{path.name}: cannot parse workflow: {error}"]

    if not event_enabled(workflow, "pull_request"):
        errors.append(f"{path.name}: must use the pull_request event")
    if event_enabled(workflow, "pull_request_target"):
        errors.append(f"{path.name}: must not use pull_request_target")

    permissions = workflow.get("permissions")
    if not isinstance(permissions, dict) or permissions.get("contents") != "read":
        errors.append(f"{path.name}: top-level permissions must include contents: read")
    elif any(value not in {"read", "none"} for value in permissions.values()):
        errors.append(f"{path.name}: fork-safe permissions must be read-only")

    raw_workflow = path.read_text(encoding="utf-8")
    if re.search(r"\bsecrets\.", raw_workflow):
        errors.append(f"{path.name}: fork-safe workflow must not reference secrets")

    jobs = workflow.get("jobs", {})
    required_jobs = {"lint", "cpu-unit"}
    if not isinstance(jobs, dict):
        return [*errors, f"{path.name}: jobs must be a mapping"]
    if not required_jobs.issubset(jobs):
        errors.append(f"{path.name}: must define lint and cpu-unit jobs")
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            errors.append(f"{path.name}:{job_id}: job must be a mapping")
            continue
        if not is_github_hosted(job.get("runs-on")):
            errors.append(f"{path.name}:{job_id}: fork-safe jobs must use a GitHub-hosted runner")
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            errors.append(f"{path.name}:{job_id}: steps must be a list")
            continue
        for step in steps:
            if not isinstance(step, dict) or not str(step.get("uses", "")).startswith("actions/checkout@"):
                continue
            inputs = step.get("with", {})
            if not isinstance(inputs, dict) or inputs.get("persist-credentials") != "false":
                errors.append(f"{path.name}:{job_id}: checkout must set persist-credentials: false")
    return errors


def validate_repository(repository: Path) -> list[str]:
    workflows_dir = repository / ".github" / "workflows"
    fork_workflow = workflows_dir / "fork_pr_checks.yml"
    return [
        *validate_private_pr_runners(workflows_dir),
        *validate_fork_workflow(fork_workflow),
    ]


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    errors = validate_repository(repository)
    if errors:
        print("Pull request runner policy violations:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Pull request runner policy is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
