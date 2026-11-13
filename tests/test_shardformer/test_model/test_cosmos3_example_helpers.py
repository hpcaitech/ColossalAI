"""Tests for the Cosmos 3 example helpers that run without diffusers or an accelerator."""

import copy
import json
import sys

import pytest
import torch

from examples.worldmodel.cosmos3_harness.evaluate_reasoner import build_manifest, load_responses, score_choice
from examples.worldmodel.cosmos3_harness.heldout import (
    load_split_manifest,
    split_by_group,
    split_fingerprint,
    write_split_manifest,
)
from examples.worldmodel.cosmos3_harness.loss import combine_modality_losses, normalise_packed_loss, velocity_mse
from examples.worldmodel.cosmos3_harness.train import _build_optimizer, _check_heldout_manifest


def _records():
    return [
        {"sample_id": f"{group}-{index}", "trajectory_id": f"trajectory-{group}"}
        for group in range(12)
        for index in range(3)
    ]


def test_grouped_split_is_deterministic_and_never_overlaps():
    train_ids, validation_ids, manifest = split_by_group(_records(), validation_fraction=0.25, seed=5)

    assert not set(train_ids) & set(validation_ids)
    assert set(train_ids) | set(validation_ids) == {record["sample_id"] for record in _records()}
    assert manifest["overlap"] == []

    repeated = split_by_group(_records(), validation_fraction=0.25, seed=5)
    assert repeated[0] == train_ids and repeated[1] == validation_ids

    # A different seed should not be required to move every group, but it must not produce an overlap.
    other_train, other_validation, _ = split_by_group(_records(), validation_fraction=0.25, seed=6)
    assert not set(other_train) & set(other_validation)


def test_grouped_split_rejects_duplicate_sample_ids():
    records = [
        {"sample_id": "a", "trajectory_id": "t0"},
        {"sample_id": "a", "trajectory_id": "t1"},
    ]

    with pytest.raises(ValueError, match="Duplicate sample id"):
        split_by_group(records, validation_fraction=0.5, seed=0)


def test_split_manifest_detects_an_edited_sample_list(tmp_path):
    _, _, manifest = split_by_group(_records(), validation_fraction=0.25, seed=1)
    path = write_split_manifest(manifest, tmp_path / "heldout.json")
    restored = load_split_manifest(path)

    assert restored["train_samples"] == manifest["train_samples"]
    assert restored["fingerprint"] == split_fingerprint(manifest)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["validation_samples"] = payload["validation_samples"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_split_manifest(path)


def test_token_normalised_loss_equals_a_global_token_mean():
    # Rank 0: sum 10 over 10 tokens. Rank 1: sum 30 over 30 tokens. The global mean is 1.
    rank0_sum = torch.tensor(10.0, requires_grad=True)
    rank1_sum = torch.tensor(30.0, requires_grad=True)

    local0 = normalise_packed_loss(rank0_sum, local_tokens=10, global_tokens=40, data_parallel_size=2)
    local1 = normalise_packed_loss(rank1_sum, local_tokens=30, global_tokens=40, data_parallel_size=2)
    # ZeRO averages the two local gradients, so this reproduces the global token mean instead of 0.5 + 1.5.
    zero_mean = (local0 + local1) / 2

    assert torch.isclose(zero_mean, torch.tensor(1.0))
    zero_mean.backward()
    assert torch.isclose(rank0_sum.grad, torch.tensor(0.025))
    assert torch.isclose(rank1_sum.grad, torch.tensor(0.025))


def test_velocity_mse_keeps_the_autograd_graph_and_honours_the_mask():
    prediction = torch.zeros(2, 3, requires_grad=True)
    target = torch.ones(2, 3)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    loss_sum, count = velocity_mse(prediction, target, mask)
    assert count == 3
    assert torch.isclose(loss_sum, torch.tensor(3.0))
    loss_sum.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad[0, :2], torch.full((2,), -2.0))

    combined = combine_modality_losses([{"sum": loss_sum, "count": count}])
    assert combined["count"] == 3
    assert torch.isclose(combined["sum"], loss_sum)

    with pytest.raises(ValueError, match="selected no element"):
        velocity_mse(prediction, target, torch.zeros(2, 3, dtype=torch.bool))


def test_reasoner_manifest_pins_the_shared_protocol_and_rejects_unknown_benchmarks():
    manifest = build_manifest(
        ["MMBench-Dev", "HallusionBench"],
        model="nvidia/Cosmos3-Edge",
        model_revision="abc123",
        decoder={"max_new_tokens": 4096, "temperature": 0.0, "do_sample": False},
    )

    assert manifest["model_revision"] == "abc123"
    assert set(manifest["benchmarks"]) == {"MMBench-Dev", "HallusionBench"}
    assert manifest["benchmarks"]["MMBench-Dev"]["vlmevalkit_dataset"] == "MMBench_DEV_EN"
    assert "not a reproduction of General Avg" in manifest["note"]
    assert len(manifest["fingerprint"]) == 64

    with pytest.raises(ValueError, match="Unknown benchmark"):
        build_manifest(
            ["NotABenchmark"],
            model="m",
            model_revision="r",
            decoder={"max_new_tokens": 1, "temperature": 0.0, "do_sample": False},
        )

    with pytest.raises(ValueError, match="decoder is missing"):
        build_manifest(["MMBench-Dev"], model="m", model_revision="r", decoder={})


def test_reasoner_responses_reject_empty_and_score_choices(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "benchmark": "MMBench-Dev", "response": "Answer: B"}),
                json.dumps({"sample_id": "b", "benchmark": "MMBench-Dev", "response": "C"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = load_responses(path)
    assert len(responses) == 2
    assert score_choice(responses[0]["response"], ["x", "y"], "B")
    assert score_choice(responses[1]["response"], ["x", "y"], "C")
    assert not score_choice(responses[1]["response"], ["x", "y"], "A")

    path.write_text(json.dumps({"sample_id": "c", "benchmark": "MMBench-Dev", "response": "  "}) + "\n")
    with pytest.raises(ValueError, match="empty response"):
        load_responses(path)


def test_adamw_auto_arms_take_the_same_step_from_the_same_gradients():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 3)
    plain_model = copy.deepcopy(base)
    npu_model = copy.deepcopy(base)
    plain = _build_optimizer(plain_model, 1e-3, backend="naive")
    foreach = _build_optimizer(npu_model, 1e-3, backend="naive", device_type="npu")

    assert plain.defaults["foreach"] in (None, False), "off NPU the builder must not force the foreach path"
    assert foreach.defaults["foreach"] is True

    grads = [torch.randn_like(param) for param in base.parameters()]
    for model, optimizer in ((plain_model, plain), (npu_model, foreach)):
        for param, grad in zip(model.parameters(), grads):
            param.grad = grad.clone()
        optimizer.step()

    for expected, actual in zip(plain_model.parameters(), npu_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-8)


def test_adamw_fused_mode_rejects_unsupported_devices_and_missing_torch_npu(monkeypatch):
    # An explicit optimizer choice cannot silently change the implementation.
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    # On an Ascend host ``torch_npu.optim`` may already be imported; ``from torch_npu.optim import ...`` then returns
    # the cached submodule without consulting the blocked parent, so hide the submodule as well.
    monkeypatch.setitem(sys.modules, "torch_npu.optim", None)
    with pytest.raises(ValueError, match="requires an NPU"):
        _build_optimizer(torch.nn.Linear(2, 2), 1e-5, backend="npu", mode="fused")
    with pytest.raises(ValueError, match="requires torch_npu"):
        _build_optimizer(torch.nn.Linear(2, 2), 1e-5, backend="auto", device_type="npu", mode="fused")


def test_adamw_attention_backend_does_not_select_the_optimizer_device():
    for backend in ("npu", "cuda", "naive"):
        optimizer = _build_optimizer(torch.nn.Linear(2, 2), 1e-5, backend=backend)
        assert optimizer.defaults["foreach"] in (None, False)


def test_adamw_builder_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown AdamW mode"):
        _build_optimizer(torch.nn.Linear(2, 2), 1e-5, backend="naive", mode="bogus")


def test_operator_bench_runs_on_cpu_and_records_its_conditions(tmp_path):
    from examples.worldmodel.cosmos3_harness import bench_ops

    output = tmp_path / "bench.json"
    assert (
        bench_ops.main(
            [
                "--op",
                "all",
                "--device",
                "cpu",
                "--num-tensors",
                "2",
                "--numel",
                "8",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["env"]["device_type"] == "cpu"
    assert report["conditions"]["num_tensors"] == 2
    assert set(report["results"]) == {"unscale_clip", "adamw"}
    assert set(report["results"]["unscale_clip"]) >= {"per_tensor_loop", "foreach_multi_tensor"}
    assert report["results"]["unscale_clip"]["foreach_multi_tensor"]["median_ms"] >= 0


def _packs(sample_ids):
    return [[{"sample_id": sample_id, "und_len": 4, "sequence_length": 12} for sample_id in sample_ids]]


def _split_with_manifest(tmp_path):
    records = [
        {"sample_id": f"{group}-{index}", "trajectory_id": f"trajectory-{group}"}
        for group in range(6)
        for index in range(2)
    ]
    train_ids, validation_ids, manifest = split_by_group(records, validation_fraction=0.34, seed=0)
    return train_ids, validation_ids, manifest, write_split_manifest(manifest, tmp_path / "heldout.json")


def test_heldout_manifest_check_accepts_packs_that_follow_the_split(tmp_path):
    train_ids, validation_ids, manifest, path = _split_with_manifest(tmp_path)

    summary = _check_heldout_manifest(path, _packs(train_ids), _packs(validation_ids))

    assert summary["training_samples"] == len(train_ids)
    assert summary["eval_samples"] == len(validation_ids)
    assert summary["fingerprint"] == split_fingerprint(manifest)


def test_heldout_manifest_check_rejects_training_on_a_validation_sample(tmp_path):
    train_ids, validation_ids, _, path = _split_with_manifest(tmp_path)

    with pytest.raises(ValueError, match="assigns to validation"):
        _check_heldout_manifest(path, _packs(train_ids + [validation_ids[0]]), _packs(validation_ids))


def test_heldout_manifest_check_rejects_an_eval_sample_outside_the_split(tmp_path):
    train_ids, validation_ids, _, path = _split_with_manifest(tmp_path)

    with pytest.raises(ValueError, match="outside the manifest"):
        _check_heldout_manifest(path, _packs(train_ids), _packs(validation_ids + ["someone-else-0"]))


def test_heldout_manifest_check_requires_explicit_sample_ids(tmp_path):
    train_ids, _, _, path = _split_with_manifest(tmp_path)

    with pytest.raises(ValueError, match="carries no 'sample_id'"):
        _check_heldout_manifest(path, _packs(train_ids) + [[{"und_len": 4, "sequence_length": 12}]], None)
