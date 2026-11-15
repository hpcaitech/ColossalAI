"""CPU contract and integration checks for the official RoboLab bridge."""

import ast
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.worldmodel.cosmos3_harness import robolab
from examples.worldmodel.cosmos3_harness.robolab_policy import (
    DroidWamPolicy,
    build_sample,
    format_actions,
    initial_noise,
    pack_inference_inputs,
    validate_observation,
)


def _git_checkout(path, files):
    path.mkdir()
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return path


@pytest.fixture
def evaluation(tmp_path):
    rows = [
        {
            "task_name": name,
            "filename": f"benchmark/{name}.py",
            "difficulty_label": difficulty,
            "instruction_variants": {level: f"{level}: {name}" for level in robolab.INSTRUCTION_TYPES},
        }
        for name, difficulty in (("BananaTask", "simple"), ("StackTask", "complex"))
    ]
    root = _git_checkout(
        tmp_path / "RoboLab",
        {
            "robolab/tasks/_metadata/task_metadata.json": json.dumps(rows),
            **{f"robolab/tasks/{row['filename']}": "# fixture\n" for row in rows},
        },
    )
    cosmos = _git_checkout(tmp_path / "cosmos", {"sampler.py": "# fixture\n"})
    manifest = robolab.build_manifest(root, cosmos, model="droid-model", model_revision="checkpoint-sha256", num_envs=2)
    return root, cosmos, manifest


def _episode(manifest, level, task, env, success=True):
    return {
        "env_name": task,
        "task_name": task,
        "run": 0,
        "env_id": env,
        "episode": env,
        "policy": "cosmos3",
        "instruction_type": level,
        "instruction": manifest["tasks"][task]["instructions"][level],
        "success": success,
        "score": 1.0,
        "dt": 1 / 15,
    }


def _results(path, manifest, backend="npu"):
    for level in manifest["evaluation"]["instruction_types"]:
        robolab.write_json(
            {
                "manifest_fingerprint": manifest["fingerprint"],
                "backend": backend,
                "simulator_versions": {"isaacsim": "5.0.0", "isaaclab": "2.2.0"},
                "server_metadata": {
                    "manifest_fingerprint": manifest["fingerprint"],
                    "backend": backend,
                    "model_revision": manifest["model_revision"],
                    "policy": manifest["policy"],
                    "checkpoint_fingerprint": "a" * 64,
                },
            },
            path / level / "harness_run.json",
        )
        rows = [
            _episode(manifest, level, task, env, success=env == 0) for task in manifest["tasks"] for env in range(2)
        ]
        (path / level / "episode_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_manifest_portable_verified_and_pinned(evaluation, tmp_path):
    root, cosmos, manifest = evaluation
    assert manifest["benchmark"] == "RoboLab-subset"
    assert manifest["policy"]["chunk_size"] == 32
    assert manifest["policy"]["shift"] == 5
    assert str(root) not in json.dumps(manifest)
    path = robolab.write_json(manifest, tmp_path / "manifest.json")
    assert robolab.load_manifest(path) == manifest
    robolab.verify_checkout(manifest, root, "robolab")
    manifest["evaluation"]["seed"] += 1
    robolab.write_json(manifest, path)
    with pytest.raises(ValueError, match="fingerprint"):
        robolab.load_manifest(path)
    with pytest.raises(ValueError, match="immutable"):
        robolab.build_manifest(root, cosmos, model="m", model_revision="main")
    (root / "robolab/tasks/benchmark/BananaTask.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="dirty"):
        robolab.verify_checkout(manifest, root, "robolab")


def test_official_commands_explicitly_pin_instruction_and_trial_counts(evaluation):
    root, _, manifest = evaluation
    commands = robolab.evaluation_commands(
        manifest, root, python="/python with spaces", remote_uri="ws://npu-host:8000", run_name="npu-eval"
    )
    assert len(commands) == 3
    for level, command in commands:
        assert command[:2] == ["/python with spaces", str(root / "policies/cosmos3/run.py")]
        assert command[command.index("--instruction-type") + 1] == level
        assert command[command.index("--num-envs") + 1] == "2"
        assert command[command.index("--num-runs") + 1] == "1"
        assert command[command.index("--output-folder-name") + 1] == f"npu-eval/{level}"
        assert "--headless" in command
    with pytest.raises(ValueError, match="run_name"):
        robolab.evaluation_commands(manifest, root, python="python", remote_uri="ws://host", run_name="../other")


def test_score_boolean_success_and_pair_gpu_npu(evaluation, tmp_path):
    _, _, manifest = evaluation
    a = robolab.summarize_results(manifest, _results(tmp_path / "gpu", manifest, "cuda"))
    b = robolab.summarize_results(manifest, _results(tmp_path / "npu", manifest))
    assert a["complete"] and a["by_instruction"]["default"]["episodes"] == 4
    assert a["by_instruction"]["default"]["task_mean_success_rate"] == 0.5
    assert a["by_instruction"]["default"]["by_difficulty"]["complex"]["successes"] == 1
    # Every subtask score is 1, but only half the episodes succeeded.
    assert a["by_instruction"]["default"]["successes"] == 2
    comparison = robolab.compare_results(a, b)
    assert comparison["both_success"] == comparison["both_failure"] == 6
    assert comparison["by_instruction"]["default"]["delta_right_minus_left"] == 0
    b["manifest_fingerprint"] = "different"
    with pytest.raises(ValueError, match="different.*manifests"):
        robolab.compare_results(a, b)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows.append(rows[0]), "Duplicate"),
        (lambda rows: rows[0].update(success="false"), "explicit boolean"),
        (lambda rows: rows[0].update(instruction_type="specific"), "instruction type"),
        (lambda rows: rows[0].update(instruction="wrong"), "instruction text"),
        (lambda rows: rows[0].update(env_id=4), "out-of-range"),
        (lambda rows: rows[0].update(dt=0.1), "15 Hz"),
        (lambda rows: rows[0].update(dt=float("nan")), "15 Hz"),
        (lambda rows: rows[0].update(episode=False), "episode index"),
        (lambda rows: rows[0].update(task_name="UnknownTask"), "Unexpected task"),
    ],
)
def test_result_corruption_is_not_scored(evaluation, tmp_path, mutation, message):
    _, _, manifest = evaluation
    path = _results(tmp_path / "output", manifest) / "vague/episode_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match=message):
        robolab.summarize_results(manifest, path.parent.parent)


def test_partial_and_truncated_runs_never_look_complete(evaluation, tmp_path):
    _, _, manifest = evaluation
    root = _results(tmp_path / "output", manifest)
    path = root / "specific/episode_results.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="1 episodes missing"):
        robolab.summarize_results(manifest, root)
    partial = robolab.summarize_results(manifest, root, allow_partial=True)
    assert not partial["complete"] and len(partial["missing_episodes"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        robolab.compare_results(partial, partial)
    path.write_text(path.read_text() + '\n{"success":')
    with pytest.raises(json.JSONDecodeError):
        robolab.summarize_results(manifest, root, allow_partial=True)


def test_comparison_requires_all_receipts_and_matching_checkpoint_contents(evaluation, tmp_path):
    _, _, manifest = evaluation
    root = _results(tmp_path / "output", manifest)
    complete = robolab.summarize_results(manifest, root)
    receipt = root / "vague/harness_run.json"
    receipt.unlink()
    imported = robolab.summarize_results(manifest, root, allow_partial=True)
    assert imported["complete"] and not imported["provenance_verified"]
    with pytest.raises(ValueError, match="provenance"):
        robolab.compare_results(complete, imported)
    other = robolab.summarize_results(manifest, _results(tmp_path / "other", manifest))
    other["run_receipts"]["vague"]["server_metadata"]["checkpoint_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint contents"):
        robolab.compare_results(complete, other)


def test_checkpoint_digest_follows_contents_not_paths(tmp_path):
    checkpoint = tmp_path / "original"
    replica = tmp_path / "replica"
    for component in ("transformer", "vae", "text_tokenizer"):
        (checkpoint / component).mkdir(parents=True)
        (replica / component).mkdir(parents=True)
        (checkpoint / component / "config.json").write_text('{"value": 1}')
        (replica / component / "config.json").symlink_to(checkpoint / component / "config.json")
    digest = robolab.checkpoint_fingerprint(checkpoint)
    assert digest == robolab.checkpoint_fingerprint(replica)
    (replica / "vae/config.json").unlink()
    (replica / "vae/config.json").write_text('{"value": 2}')
    assert digest != robolab.checkpoint_fingerprint(replica)


def test_launcher_records_provenance_and_rejects_endpoint_mismatch(evaluation, monkeypatch):
    pytest.importorskip("msgpack")
    from examples.worldmodel.cosmos3_harness import robolab_transport

    root, _, manifest = evaluation
    metadata = {
        "manifest_fingerprint": manifest["fingerprint"],
        "backend": "npu",
        "model_revision": manifest["model_revision"],
        "policy": manifest["policy"],
        "checkpoint_fingerprint": "a" * 64,
    }
    monkeypatch.setattr(robolab_transport, "server_metadata", lambda uri: metadata)
    original_check_output = subprocess.check_output

    def check_output(command, **kwargs):
        if command[1:2] == ["-c"]:
            return json.dumps({"isaacsim": "5.0.0", "isaaclab": "2.2.0"})
        return original_check_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    original_run = subprocess.run
    launches = []

    def run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        assert kwargs["cwd"] == root and kwargs["check"]
        launches.append(command)
        level = command[command.index("--instruction-type") + 1]
        folder = root / "output/eval" / level
        rows = [_episode(manifest, level, task, env) for task in manifest["tasks"] for env in range(2)]
        (folder / "episode_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    monkeypatch.setattr(subprocess, "run", run)
    args = SimpleNamespace(
        robolab_root=root,
        python="isaac-python",
        remote_uri="ws://policy:8000",
        run_name="eval",
        video_mode="none",
        dry_run=False,
        external_server=False,
        backend="npu",
    )
    robolab.run_evaluation(args, manifest)
    summary = json.loads((root / "output/eval/summary.json").read_text())
    assert summary["complete"] and summary["provenance_verified"] and len(launches) == 3
    # Generated output does not invalidate the source pin. Matching receipts permit official resume.
    robolab.run_evaluation(args, manifest)
    assert len(launches) == 6
    metadata["backend"] = "cuda"
    with pytest.raises(ValueError, match="endpoint"):
        robolab.run_evaluation(args, manifest)
    assert len(launches) == 6


def _observation():
    return {
        "observation/image": np.zeros((540, 640, 3), dtype=np.uint8),
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.25], dtype=np.float32),
        "prompt": "Put the banana in the bowl",
        "session_id": "episode-1-env-2",
    }


def test_droid_state_convention_and_output_trim():
    config = robolab.POLICY_CONFIG
    observation = _observation()
    sample = build_sample(observation, config, lambda value, _: value)
    assert sample["video"].shape == (3, 33, 540, 640)
    assert sample["action"].shape == (33, 8)
    assert sample["action"][0, -1] == 0.75
    assert sample["domain_id"].item() == 8
    assert sample["mode"] == "wam"
    generated = torch.zeros(33, 64)
    generated[0] = 99  # This is state, not an executable action.
    generated[1:, :7] = 0.2
    generated[1:, 7] = 0.75
    response = format_actions(generated, config)
    assert response.shape == (32, 8)
    assert np.all(response[:, :7] == np.float32(0.2))
    assert np.all(response[:, -1] == 0.25)
    assert torch.all(generated[1:, 7] == 0.75)
    observation["observation/joint_position"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_observation(observation, config)


def test_state_time_zero_aligns_actions_and_video_and_keeps_fractional_positions():
    config = SimpleNamespace(
        latent_patch_size=2, action_dim=16, base_fps=24, unified_3d_mrope_temporal_modality_margin=15000
    )
    inputs = pack_inference_inputs(torch.zeros(1, 2, 9, 4, 6), torch.zeros(33, 16), torch.arange(3), config, 15)
    position = inputs["position_ids"]
    assert position.dtype == torch.float32
    vision_start = 3
    action_start = 3 + 9 * 6
    assert position[0, action_start] == position[0, vision_start]
    assert position[0, action_start + 4] == position[0, vision_start + 6]
    assert position[0, action_start + 1] != position[0, action_start + 1].long()
    assert inputs["vision_mse_loss_indexes"][0] == vision_start + 6
    assert inputs["action_mse_loss_indexes"][0] == action_start + 1


def test_denoising_preserves_conditioning_and_resets_solvers():
    config = {**robolab.POLICY_CONFIG, "chunk_size": 4}
    calls, solvers = [], []

    class Model:
        config = SimpleNamespace(
            latent_patch_size=1, action_dim=16, base_fps=24, unified_3d_mrope_temporal_modality_margin=15000
        )

        def __call__(self, **kwargs):
            calls.append(kwargs)
            # Verify normalized VAE mean and inverted state remain clean at every step/CFG branch.
            assert torch.all(kwargs["vision_tokens"][0][:, :, 0] == 2)
            assert kwargs["action_tokens"][0][0, 7] == 0.75
            assert torch.all(kwargs["action_tokens"][0][:, 8:] == 0)
            assert kwargs["vision_timesteps"].dtype == torch.float32
            return [torch.zeros_like(kwargs["vision_tokens"][0])], None, [torch.zeros_like(kwargs["action_tokens"][0])]

    class Solver:
        def __init__(self, **kwargs):
            solvers.append(self)

        def set_timesteps(self, steps, **kwargs):
            self.timesteps = torch.tensor([999, 500])

        def step(self, velocity, timestep, sample, **kwargs):
            # Deliberately perturb clean tokens: the inference loop must restore them.
            return (sample + 1,)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if messages[0]["content"] else [1]

    class Vae:
        config = SimpleNamespace(latents_mean=[1.0], latents_std=[2.0])

        def encode(self, pixels):
            assert pixels.shape[2] == 1
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.full((1, 1, 1, 2, 2), 5.0)))

    def transform(value, resolution):
        value["action"] = torch.nn.functional.pad(value["action"], (0, 8))
        return value

    policy = DroidWamPolicy(Model(), Vae(), Tokenizer(), transform, Solver, config, "cpu")
    first = policy.infer(_observation())["action"]
    second = policy.infer(_observation())["action"]
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 8)
    assert len(solvers) == 4 and len(calls) == 8
    assert calls[0]["vision_timesteps"][0].item() == 999


def test_websocket_interoperates_with_official_numpy_markers(monkeypatch):
    pytest.importorskip("websockets.sync.client")
    pytest.importorskip("msgpack")
    from websockets.sync.client import connect

    from examples.worldmodel.cosmos3_harness.robolab_transport import pack, server_metadata, unpack
    from examples.worldmodel.cosmos3_harness.serve_robolab import serve_policy

    monkeypatch.setenv("COSMOS3_API_TOKEN", "test-token")

    class Policy:
        def infer(self, observation):
            validate_observation(observation, robolab.POLICY_CONFIG)
            return {"action": np.ones((32, 8), dtype=np.float32)}

    metadata = {"manifest_fingerprint": "pinned", "backend": "npu"}
    with serve_policy(Policy(), metadata, host="127.0.0.1", port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        try:
            assert server_metadata(uri) == metadata
            with connect(uri, additional_headers={"Authorization": "Bearer test-token"}) as ws:
                assert unpack(ws.recv()) == metadata
                ws.send(pack(_observation()))
                np.testing.assert_array_equal(unpack(ws.recv())["action"], np.ones((32, 8), dtype=np.float32))
                ws.send(pack({"prompt": "bad"}))
                assert unpack(ws.recv())["type"] == "error"
            monkeypatch.setenv("COSMOS3_API_TOKEN", "wrong-token")
            with pytest.raises(Exception, match="401"):
                server_metadata(uri)
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_packed_inputs_run_through_real_diffusers_transformer():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "Cosmos3OmniTransformer"):
        pytest.skip("Installed diffusers predates Cosmos3OmniTransformer")
    model = diffusers.Cosmos3OmniTransformer(
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        latent_channel=2,
        latent_patch_size=2,
        patch_latent_dim=8,
        action_dim=16,
        action_gen=True,
        vocab_size=32,
        rope_axes_dim=[2, 1, 1],
    ).eval()
    inputs = pack_inference_inputs(torch.randn(1, 2, 3, 4, 6), torch.randn(9, 16), torch.arange(3), model.config, 15)
    inputs["vision_timesteps"] = torch.full((len(inputs["vision_mse_loss_indexes"]),), 999.0)
    inputs["action_timesteps"] = torch.full((8,), 999.0)
    with torch.inference_mode():
        vision, sound, action = model(**inputs)
    assert sound is None
    assert vision[0].shape == (1, 2, 3, 4, 6) and action[0].shape == (9, 16)
    assert torch.isfinite(vision[0]).all() and torch.isfinite(action[0]).all()
    assert torch.count_nonzero(vision[0][:, :, :1]) == torch.count_nonzero(action[0][0]) == 0
    colossal = pytest.importorskip("colossalai.shardformer.modeling.cosmos3")
    for module in model.modules():
        if module.__class__.__name__ == "Cosmos3PackedMoTAttention":
            module.set_processor(colossal.ColossalTwoWayAttnProcessor(backend="naive"))
    with torch.inference_mode():
        colossal_vision, _, colossal_action = model(**inputs)
    torch.testing.assert_close(colossal_vision[0], vision[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(colossal_action[0], action[0], rtol=1e-5, atol=1e-6)


def test_optional_official_transform_sampler_and_mrope_parity():
    """Run in CPU CI with COSMOS3_FRAMEWORK_ROOT pointing at the pinned official checkout."""
    root = os.environ.get("COSMOS3_FRAMEWORK_ROOT")
    if not root:
        pytest.skip("COSMOS3_FRAMEWORK_ROOT is needed for the upstream parity test")
    sys.path.insert(0, root)
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # Isolate this pure upstream helper from misc.py's unrelated CUDA/distributed/IO imports.
    path = Path(root) / "cosmos_framework/utils/misc.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "arch_invariant_rand"
    )
    namespace = {"torch": torch, "np": np, "List": list, "Tuple": tuple}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    arch_invariant_rand = namespace["arch_invariant_rand"]

    config = robolab.POLICY_CONFIG
    for shape in ((2, 9, 4, 6), (33, 16)):
        torch.testing.assert_close(
            initial_noise(shape, 7, "cpu"), arch_invariant_rand(shape, torch.float32, "cpu", 7), rtol=0, atol=0
        )
    sample = build_sample(
        _observation(), config, ActionTransformPipeline(max_action_dim=16, format_prompt_as_json=True)
    )
    # Use the official resolution-table choice, including its aspect-preserving resize/padding.
    assert sample["video"].shape == (3, 33, 544, 736)
    assert sample["ai_caption"]["resolution"] == {"H": 544, "W": 736}
    assert sample["ai_caption"]["cinematography"]["framing"].endswith("with the robot visible.")
    assert sample["sequence_plan"].action_start_frame_offset == 0
    model_config = SimpleNamespace(
        latent_patch_size=2, action_dim=16, base_fps=24, unified_3d_mrope_temporal_modality_margin=15000
    )
    inputs = pack_inference_inputs(torch.zeros(1, 2, 9, 4, 6), sample["action"], torch.arange(3), model_config, 15)
    official, _ = get_3d_mrope_ids_vae_tokens(
        33,
        1,
        1,
        temporal_offset=15003,
        fps=15,
        base_fps=24,
        temporal_compression_factor=1,
        base_temporal_compression_factor=4,
        start_frame_offset=0,
    )
    torch.testing.assert_close(inputs["position_ids"][:, -33:], official, rtol=0, atol=0)
    solver = FlowUniPCMultistepScheduler(shift=1.0)
    solver.set_timesteps(4, device="cpu", shift=5.0)
    assert solver.timesteps.tolist() == [999, 937, 833, 624]
    state = torch.ones(1, 8, 2, 2)
    for timestep in solver.timesteps:
        state = solver.step(torch.zeros_like(state), timestep, state, return_dict=False)[0]
    torch.testing.assert_close(state, torch.ones_like(state))
