import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import container_batch as container
import run_gpu_batch as runner

IMAGE = "sha256:" + "a" * 64
REFERENCE = "ghcr.io/example/ci@sha256:" + "b" * 64
GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
SCOPE = "gh-123-1-layer-2"


class ContainerCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        script = self.root / ".github/scripts/ci/run_container_batch.sh"
        script.parent.mkdir(parents=True)
        script.touch()
        self.output = self.root / "reports"
        self.output.mkdir()

    def command(self, **env):
        return container.create_command(
            "test-container",
            SCOPE,
            IMAGE,
            self.root,
            self.output,
            [GPU_A, GPU_B],
            "5",
            env,
        )

    def test_uuid_filter_and_logical_device_mapping(self):
        command = self.command(CUDA_VISIBLE_DEVICES="4,7")
        self.assertEqual(command[command.index("--gpus") + 1], f'"device={GPU_A},{GPU_B}"')
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", command)
        self.assertIn(f"NVIDIA_VISIBLE_DEVICES={GPU_A},{GPU_B}", command)
        self.assertEqual(command[-2:], ["5", "0,1"])

    def test_only_source_and_reports_are_mounted(self):
        command = self.command(COLOSSAL_VENV="/host/private/venv", SECRET_TOKEN="not-for-container")
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertEqual(
            mounts,
            [
                f"type=bind,src={self.root.resolve()},dst=/ci-source,readonly",
                f"type=bind,src={self.output.resolve()},dst=/ci-results",
            ],
        )
        self.assertNotIn("SECRET_TOKEN=not-for-container", command)
        self.assertNotIn("COLOSSAL_VENV=/host/private/venv", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn(f"{os.getuid()}:{os.getgid()}", command)
        self.assertIn("--pull=never", command)

    def test_resource_limits(self):
        command = self.command(COLOSSAL_CI_CPUS="4", COLOSSAL_CI_MEMORY="32g", COLOSSAL_CI_SHM="4g")
        for flag, expected in (
            ("--cpus", "4"),
            ("--memory", "32g"),
            ("--memory-swap", "32g"),
            ("--shm-size", "4g"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        for env in (
            {"COLOSSAL_CI_CPUS": "0"},
            {"COLOSSAL_CI_MEMORY": "unlimited"},
            {"COLOSSAL_CI_SHM": "0g"},
        ):
            with self.subTest(env=env), self.assertRaises(RuntimeError):
                self.command(**env)

    def test_mutable_or_empty_image_is_rejected_before_docker(self):
        with patch.object(container, "docker") as docker:
            for reference in ("", "ci:latest", "ci:torch213", "ci@sha256:invalid", "sha256:abc"):
                with self.subTest(reference=reference), self.assertRaises(RuntimeError):
                    container.resolve_image(reference)
            docker.assert_not_called()

    def test_registry_digest_resolves_to_local_content_id(self):
        with patch.object(container, "docker", return_value=IMAGE) as docker:
            self.assertEqual(container.resolve_image(REFERENCE), IMAGE)
            docker.assert_called_once_with("image", "inspect", "--format", "{{.Id}}", REFERENCE)

    def test_local_image_id_needs_no_registry(self):
        with patch.object(container, "docker", return_value=IMAGE) as docker:
            self.assertEqual(container.resolve_image(IMAGE), IMAGE)
            docker.assert_called_once_with("image", "inspect", "--format", "{{.Id}}", IMAGE)

    def test_missing_local_image_fails_without_pulling(self):
        with patch.object(container, "docker", side_effect=RuntimeError("No such image")) as docker:
            with self.assertRaisesRegex(RuntimeError, "No such image"):
                container.resolve_image(IMAGE)
            docker.assert_called_once_with("image", "inspect", "--format", "{{.Id}}", IMAGE)

    def test_wrong_local_image_id_fails_closed(self):
        with patch.object(container, "docker", return_value="sha256:" + "c" * 64):
            with self.assertRaisesRegex(RuntimeError, "different local image ID"):
                container.resolve_image(IMAGE)

    def run_batch(self, stream):
        result = {}
        code = container.run_container(
            "5",
            [(4, GPU_A), (7, GPU_B)],
            self.root,
            self.output,
            IMAGE,
            {"CI_CONTAINER_SCOPE": SCOPE},
            stream,
            result,
        )
        return code, result

    def test_success_and_test_failure_preserve_exit_and_cleanup(self):
        for exit_code in (0, 1, 5, 124):
            with (
                self.subTest(exit_code=exit_code),
                patch.object(container.subprocess, "run"),
                patch.object(
                    container,
                    "docker",
                    return_value=json.dumps(
                        {
                            "Running": False,
                            "Status": "exited",
                            "ExitCode": exit_code,
                        }
                    ),
                ),
                patch.object(container, "remove_owned_container") as cleanup,
            ):
                code, result = self.run_batch(Mock(return_value=exit_code))
                self.assertEqual(code, exit_code)
                self.assertEqual(result["container_cleanup"], "completed")
                cleanup.assert_called_once_with(result["container_name"], SCOPE)

    def test_create_failure_still_attempts_cleanup(self):
        for error in (
            subprocess.CalledProcessError(125, ["docker", "create"]),
            subprocess.TimeoutExpired("docker", 60),
        ):
            with (
                self.subTest(error=error),
                patch.object(container.subprocess, "run", side_effect=error),
                patch.object(container, "remove_owned_container") as cleanup,
            ):
                with self.assertRaises(subprocess.SubprocessError):
                    self.run_batch(Mock())
                cleanup.assert_called_once()

    def test_interruption_cleans_container_and_restores_signal_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(container.subprocess, "run"), patch.object(container, "remove_owned_container") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_batch(Mock(side_effect=RuntimeError("cancelled")))
            cleanup.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_running_or_oom_container_is_not_a_pass(self):
        for state in (
            {"Running": True},
            {"Running": False, "Status": "exited", "OOMKilled": True, "ExitCode": 0},
        ):
            with (
                self.subTest(state=state),
                patch.object(container.subprocess, "run"),
                patch.object(container, "docker", return_value=json.dumps(state)),
                patch.object(container, "remove_owned_container") as cleanup,
            ):
                with self.assertRaises(RuntimeError):
                    self.run_batch(Mock(return_value=0))
                cleanup.assert_called_once()

    def test_cleanup_failure_cannot_report_success(self):
        with (
            patch.object(container.subprocess, "run"),
            patch.object(
                container,
                "docker",
                return_value=json.dumps(
                    {
                        "Running": False,
                        "Status": "exited",
                        "ExitCode": 0,
                    }
                ),
            ),
            patch.object(
                container,
                "remove_owned_container",
                side_effect=RuntimeError("daemon unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "daemon unavailable"):
                self.run_batch(Mock(return_value=0))


class ContainerOwnershipTests(unittest.TestCase):
    def test_absent_container_needs_no_removal(self):
        with patch.object(container, "docker", return_value="") as docker:
            container.remove_owned_container("owned", SCOPE)
            self.assertEqual(docker.call_count, 1)

    def test_removal_checks_exact_name_and_both_labels(self):
        good = {
            "Name": "/owned",
            "Config": {
                "Labels": {
                    container.OWNER_LABEL: container.OWNER,
                    container.SCOPE_LABEL: SCOPE,
                }
            },
        }
        with patch.object(container, "docker", side_effect=["container-id", json.dumps([good]), ""]) as docker:
            container.remove_owned_container("owned", SCOPE)
            self.assertEqual(docker.call_args.args, ("rm", "--force", "container-id"))
        for info in (
            {"Name": "/someone-else", "Config": good["Config"]},
            {
                "Name": "/owned",
                "Config": {"Labels": {container.OWNER_LABEL: container.OWNER}},
            },
            {"Name": "/owned", "Config": {"Labels": {container.SCOPE_LABEL: SCOPE}}},
        ):
            with (
                self.subTest(info=info),
                patch.object(container, "docker", side_effect=["container-id", json.dumps([info])]) as docker,
            ):
                with self.assertRaises(RuntimeError):
                    container.remove_owned_container("owned", SCOPE)
                self.assertEqual(docker.call_count, 2)

    def test_invalid_cleanup_scope_is_rejected(self):
        with patch.object(container, "docker") as docker:
            for scope in ("", "*", "other\njob", "x" * 102):
                with self.subTest(scope=scope), self.assertRaises(RuntimeError):
                    container.cleanup_scope(scope)
            docker.assert_not_called()


class ImageRecipeTests(unittest.TestCase):
    def test_recipe_must_match_image_including_locks_and_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            source, image = Path(directory) / "source", Path(directory) / "image"
            source.mkdir()
            image.mkdir()
            for name in container.IMAGE_INPUTS:
                (source / name).write_text("same\n")
                (image / name).write_text("same\n")
            container.check_image_recipe(source, image)
            for name in container.IMAGE_INPUTS:
                (source / name).write_text("changed\n")
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "does not match"):
                    container.check_image_recipe(source, image)
                (source / name).write_text("same\n")
            (image / "requirements.lock").unlink()
            with self.assertRaises(FileNotFoundError):
                container.check_image_recipe(source, image)


@unittest.skipUnless(os.name == "posix", "flock requires POSIX")
class LockedContainerTests(unittest.TestCase):
    def test_container_lifecycle_is_inside_gpu_lock(self):
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "gpu.lock"
            command = [
                "run_gpu_batch.py",
                "--batch",
                "5",
                "--output",
                str(root / "out"),
                "--lock",
                str(lock_path),
                "--image",
                REFERENCE,
            ]

            def fake_container(*args):
                with lock_path.open("w") as second:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(args[1], [(4, GPU_A), (7, GPU_B)])
                return 0

            with (
                patch("sys.argv", command),
                patch.dict(os.environ, {"COLOSSAL_GPU_POOL": "4,7"}),
                patch.object(runner, "resolve_image", return_value=IMAGE),
                patch.object(runner, "query_gpus", return_value=(f"4,{GPU_A},4,0\n7,{GPU_B},4,0", "")),
                patch.object(runner, "run_container", side_effect=fake_container) as launch,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 0)
                launch.assert_called_once()
            with lock_path.open("w") as unlocked:
                fcntl.flock(unlocked, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_empty_explicit_image_never_falls_back_to_host(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                "run_gpu_batch.py",
                "--batch",
                "4a",
                "--output",
                str(root / "out"),
                "--lock",
                str(root / "gpu.lock"),
                "--image",
                "",
            ]
            with (
                patch("sys.argv", command),
                patch.object(runner, "query_gpus") as query,
                patch.object(runner, "stream_process") as stream,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 1)
                query.assert_not_called()
                stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
