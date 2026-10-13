"""CPU-only regression checks for Draft admission and GPU-pool isolation."""

import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from check_pr_runner_policy import event_enabled, is_github_hosted, load_workflow, workflow_paths

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
REUSABLE = "./.github/workflows/ci_reusable_gpu_batches.yml"
UPSTREAM = "hpcaitech/ColossalAI"


def evaluate_condition(expression, context):
    """Evaluate only the typed boolean/string subset used by these job guards.

    This is a test helper, not a general GitHub expression engine. Unknown
    syntax fails the test; actual workflow syntax is also checked by actionlint.
    """
    tree = ast.parse(expression.replace("&&", " and ").replace("||", " or ").strip(), mode="eval")

    def folded(value):
        return value.casefold() if isinstance(value, str) else value

    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in {"true", "false"}:
            return node.id == "true"
        if isinstance(node, ast.Attribute):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                return context[".".join([node.id, *reversed(parts)])]
        if isinstance(node, ast.BoolOp):
            values = (visit(value) for value in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = folded(visit(node.left)), folded(visit(node.comparators[0]))
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            values = [visit(value) for value in node.args]
            if node.func.id == "fromJSON" and len(values) == 1:
                return json.loads(values[0])
            if node.func.id == "contains" and len(values) == 2:
                return folded(values[1]) in [folded(value) for value in values[0]]
        raise AssertionError(f"Unsupported workflow condition syntax: {ast.dump(node)}")

    return bool(visit(tree.body))


def github_context(draft=False, association="OWNER", head=UPSTREAM, repository=UPSTREAM, event="pull_request"):
    return {
        "github.repository": repository,
        "github.event_name": event,
        "github.event.pull_request.draft": draft,
        "github.event.pull_request.head.repo.full_name": head,
        "github.event.pull_request.author_association": association,
    }


class DraftPullRequestRoutingTests(unittest.TestCase):
    def setUp(self):
        self.layer2 = load_workflow(WORKFLOWS / "ci_layer_2_gpu_pr.yml")

    def test_trusted_draft_and_ready_prs_have_distinct_jobs(self):
        for association in ("OWNER", "MEMBER", "COLLABORATOR"):
            for draft in (True, False):
                with self.subTest(association=association, draft=draft):
                    context = github_context(draft=draft, association=association)
                    jobs = self.layer2["jobs"]
                    self.assertEqual(evaluate_condition(jobs["container-smoke"]["if"], context), draft)
                    self.assertEqual(evaluate_condition(jobs["gpu-regression"]["if"], context), not draft)

    def test_forks_untrusted_authors_and_other_repositories_never_admit_gpu_jobs(self):
        for draft in (True, False):
            for overrides in (
                {"head": "contributor/ColossalAI"},
                {"association": "CONTRIBUTOR"},
                {"association": "NONE"},
                {"repository": "contributor/ColossalAI", "head": "contributor/ColossalAI"},
                {"event": "push"},
            ):
                with self.subTest(draft=draft, overrides=overrides):
                    context = github_context(draft=draft, **overrides)
                    for job in self.layer2["jobs"].values():
                        self.assertFalse(evaluate_condition(job["if"], context))

    def test_manual_dispatch_retains_full_regression(self):
        context = github_context(event="workflow_dispatch")
        self.assertFalse(evaluate_condition(self.layer2["jobs"]["container-smoke"]["if"], context))
        self.assertTrue(evaluate_condition(self.layer2["jobs"]["gpu-regression"]["if"], context))

    def test_smoke_membership_check_name_and_fast_mode_are_explicit(self):
        smoke, core = (self.layer2["jobs"][key] for key in ("container-smoke", "gpu-regression"))
        self.assertNotEqual(smoke["name"], core["name"])
        self.assertEqual(core["name"], "Torch 2.13 GPU core (batches 2 and 5)")
        self.assertEqual(smoke["uses"], REUSABLE)
        self.assertEqual(core["uses"], REUSABLE)
        self.assertEqual(smoke["with"]["source_sha"], "${{ github.sha }}")
        self.assertEqual(smoke["with"]["batches"], "4a")
        self.assertEqual(smoke["with"]["gpu_pool_variable"], "COLOSSAL_CI_SMOKE_GPU_POOL")
        self.assertEqual(smoke["with"]["fast_mode"], "false")
        self.assertLessEqual(int(smoke["with"]["timeout_minutes"]), 15)
        self.assertEqual(core["with"]["batches"], "2 4a 5")
        self.assertEqual(core["with"]["fast_mode"], "true")
        self.assertNotIn("gpu_pool_variable", core["with"])

    def test_ready_and_converted_to_draft_events_reconcile_layer2(self):
        types = self.layer2["on"]["pull_request"]["types"]
        self.assertIn("ready_for_review", types)
        self.assertIn("converted_to_draft", types)
        self.assertEqual(self.layer2["concurrency"]["cancel-in-progress"], "true")

    def test_retired_e1_host_workflow_cannot_run_alongside_containers(self):
        self.assertFalse((WORKFLOWS / "e1-gpu-on-pr.yml").exists())
        self.assertFalse((WORKFLOWS / "e1-gpu-on-pr.yaml").exists())
        for path in workflow_paths(WORKFLOWS):
            source = path.read_text(encoding="utf-8")
            for legacy_entry in (
                "E1_COLOSSALAI_PYTHON",
                ".github/scripts/e1/pr_gpu.py",
                ".github/scripts/e1/colossalai_suite.py",
            ):
                with self.subTest(workflow=path.name, entry=legacy_entry):
                    self.assertNotIn(legacy_entry, source)

    def test_retired_e1_tests_remain_in_container_core_batches(self):
        manifest = (ROOT / ".github/scripts/ci/run_colossalai_batch.sh").read_text(encoding="utf-8")
        for case, path, test_name in (
            ("2|02", "tests/test_booster/test_accelerator.py", "test_accelerator"),
            ("5|05", "tests/test_booster/test_plugin/test_dp_plugin_base.py", "test_dp_plugin_dataloader"),
        ):
            with self.subTest(batch=case, test=test_name):
                batch = manifest.split(f"    {case})\n", 1)[1].split("        ;;", 1)[0]
                self.assertIn(path, batch)
                module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
                self.assertIn(test_name, [node.name for node in module.body if isinstance(node, ast.FunctionDef)])

    def test_legacy_gpu_entrypoints_skip_drafts_but_keep_ready_prs(self):
        for filename in ("compatibility_test_on_pr.yml", "cuda_ext_check_before_merge.yml"):
            workflow = load_workflow(WORKFLOWS / filename)
            self.assertIn("ready_for_review", workflow["on"]["pull_request"]["types"])
            for job_id, job in workflow["jobs"].items():
                with self.subTest(workflow=filename, job=job_id):
                    self.assertFalse(evaluate_condition(job["if"], github_context(draft=True)))
                    self.assertTrue(evaluate_condition(job["if"], github_context(draft=False)))
                    self.assertFalse(evaluate_condition(job["if"], github_context(head="contributor/ColossalAI")))

    def test_cuda_extension_manual_dispatch_remains_available(self):
        workflow = load_workflow(WORKFLOWS / "cuda_ext_check_before_merge.yml")
        for job in workflow["jobs"].values():
            self.assertTrue(evaluate_condition(job["if"], github_context(event="workflow_dispatch")))

    def test_all_other_private_pr_jobs_have_a_draft_or_post_merge_gate(self):
        for path in workflow_paths(WORKFLOWS):
            workflow = load_workflow(path)
            if not event_enabled(workflow, "pull_request"):
                continue
            jobs = workflow["jobs"]

            def gated(job_id):
                job = jobs[job_id]
                condition = " ".join(job.get("if", "").split())
                if any(
                    guard in condition
                    for guard in (
                        "github.event.pull_request.draft == false",
                        "github.event.pull_request.merged == true",
                    )
                ):
                    return True
                needs = job.get("needs", [])
                needs = [needs] if isinstance(needs, str) else needs
                return "always()" not in condition and any(gated(parent) for parent in needs)

            for job_id, job in jobs.items():
                if is_github_hosted(job.get("runs-on")):
                    continue
                if path.name == "ci_layer_2_gpu_pr.yml" and job_id == "container-smoke":
                    continue
                with self.subTest(workflow=path.name, job=job_id):
                    self.assertTrue(gated(job_id), "Private Draft PR jobs must be gated before runner allocation")


class LegacyPrBuildRetirementTests(unittest.TestCase):
    def test_legacy_pr_build_and_coverage_workflows_are_removed(self):
        for name in ("build_on_pr", "report_test_coverage"):
            for suffix in (".yml", ".yaml"):
                with self.subTest(workflow=name + suffix):
                    self.assertFalse((WORKFLOWS / (name + suffix)).exists())

    def test_active_workflows_do_not_reference_retired_pr_builds(self):
        for path in workflow_paths(WORKFLOWS):
            source = path.read_text(encoding="utf-8")
            for entry in ("Build on PR", "Report Test Coverage", "build_on_pr.yml", "report_test_coverage.yml"):
                with self.subTest(workflow=path.name, entry=entry):
                    self.assertNotIn(entry, source)

    def test_layer3_retains_batch_three_aot_compilation(self):
        workflow = load_workflow(WORKFLOWS / "ci_layer_3_full.yml")
        default = workflow["on"]["workflow_dispatch"]["inputs"]["batches"]["default"]
        self.assertIn("3", default.split())
        self.assertIn(f"|| '{default}'", workflow["jobs"]["torch-213"]["with"]["batches"])
        entrypoint = (ROOT / ".github/scripts/ci/run_container_batch.sh").read_text(encoding="utf-8")
        build_mode = "BUILD_EXT=0\n" + entrypoint.split("\nBUILD_EXT=0\n", 1)[1].split("\nexport BUILD_EXT\n", 1)[0]
        for batch, expected in (("3", "1"), ("2", "0"), ("4a", "0"), ("5", "0")):
            with self.subTest(batch=batch):
                result = subprocess.run(
                    ["bash", "-eu", "-c", build_mode + '\nprintf "%s" "$BUILD_EXT"'],
                    env={"PATH": os.defpath, "BATCH": batch},
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                self.assertEqual(result.stdout, expected)


class GpuPoolConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.workflow = load_workflow(WORKFLOWS / "ci_reusable_gpu_batches.yml")
        self.job = self.workflow["jobs"]["run-batches"]

    def test_pool_lookup_has_no_truthy_fallback_and_core_default_is_unchanged(self):
        inputs = self.workflow["on"]["workflow_call"]["inputs"]
        self.assertEqual(inputs["gpu_pool_variable"]["default"], "COLOSSAL_GPU_POOL")
        self.assertEqual(self.job["env"]["COLOSSAL_GPU_POOL"], "${{ vars[inputs.gpu_pool_variable] }}")
        self.assertEqual(self.job["env"]["CI_GPU_LOCK"], "${{ vars.COLOSSAL_GPU_LOCK }}")
        for filename in ("ci_layer_3_full.yml", "ci_gpu_idle_monitor.yml"):
            for job in load_workflow(WORKFLOWS / filename)["jobs"].values():
                if job.get("uses") == REUSABLE:
                    self.assertNotIn("gpu_pool_variable", job["with"])

    def validate_configuration(self, pool_variable, pool, batches="4a"):
        script = self.job["steps"][0]["run"]
        for key, value in (("inputs.layer", "draft-smoke"), ("github.run_id", "123"), ("github.run_attempt", "1")):
            script = script.replace("${{ " + key + " }}", value)
        self.assertNotIn("${{", script)
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "github-env"
            result = subprocess.run(
                ["bash", "-c", script],
                env={
                    "PATH": os.defpath,
                    "COLOSSAL_CI_IMAGE": "sha256:" + "a" * 64,
                    "CI_GPU_LOCK": str(Path(directory) / "shared.lock"),
                    "GPU_POOL_VARIABLE": pool_variable,
                    "COLOSSAL_GPU_POOL": pool,
                    "REQUESTED_BATCHES": batches,
                    "RUNNER_TEMP": directory,
                    "GITHUB_ENV": str(env_file),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(env_file.exists(), result.returncode == 0)
        return result

    def test_empty_smoke_pool_fails_before_results_or_gpu_work(self):
        result = self.validate_configuration("COLOSSAL_CI_SMOKE_GPU_POOL", "")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("COLOSSAL_CI_SMOKE_GPU_POOL", result.stderr)
        self.assertIn("no pool fallback", result.stderr)

    def test_smoke_accepts_an_explicit_pool_including_gpu_zero(self):
        for pool in ("0", "3", "3,7"):
            with self.subTest(pool=pool):
                result = self.validate_configuration("COLOSSAL_CI_SMOKE_GPU_POOL", pool)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_main_pool_still_accepts_multicard_batches(self):
        result = self.validate_configuration("COLOSSAL_GPU_POOL", "2,3", batches="2 4a 5")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_pool_or_profile_and_smoke_scope_expansion_fail(self):
        for variable, pool, batches in (
            ("OTHER_POOL", "3", "4a"),
            ("COLOSSAL_GPU_POOL", "", "2 4a 5"),
            ("COLOSSAL_CI_SMOKE_GPU_POOL", "3", "2 4a 5"),
            ("COLOSSAL_CI_SMOKE_GPU_POOL", "all", "4a"),
            ("COLOSSAL_CI_SMOKE_GPU_POOL", "3,", "4a"),
            ("COLOSSAL_CI_SMOKE_GPU_POOL", "3\n4", "4a"),
        ):
            with self.subTest(variable=variable, pool=pool, batches=batches):
                self.assertNotEqual(self.validate_configuration(variable, pool, batches).returncode, 0)


if __name__ == "__main__":
    unittest.main()
