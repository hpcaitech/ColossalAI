"""CPU contracts for the online SFT entry, including real accumulated optimizer updates."""

from types import SimpleNamespace

import pytest
import torch

from colossalai.shardformer.layer.two_way_attn import TwoWayPackedMeta
from examples.worldmodel.cosmos3 import train_sft as entry
from examples.worldmodel.cosmos3.runtime import AccumulationWindow, resolve_attention_backend, resolve_compile_backend


def args(*extra):
    return entry.parse_args(["--data", "/data", "--checkpoint", "/checkpoint", *extra])


def test_aligned_defaults_and_legacy_aliases():
    baseline = args()
    assert baseline.vae_dtype == "bfloat16"
    assert baseline.vae_chunk_frames == 0 and baseline.vae_compile == "none"
    assert not baseline.fuse_vae_norm and not baseline.fast_vae_encoder
    assert not baseline.cast_inputs and not baseline.cast_buffers
    assert baseline.reinit_action_head and baseline.activation_checkpointing == "full"
    legacy = args("--backend", "naive", "--checkpoint-mode", "selective", "--compile-vae-aot")
    assert legacy.attn_backend == "naive" and legacy.activation_checkpointing == "selective"
    assert legacy.vae_compile == "aot" and legacy.vae_chunk_frames == 4


@pytest.mark.parametrize(
    "extra",
    [
        ["--grad-accum", "0"],
        ["--vae-chunk-frames", "-4"],
        ["--vae-encode-batch", "-1"],
        ["--compile-vae", "--compile-vae-aot"],
        ["--vae-compile", "jit", "--compile-vae-aot"],
        ["--adamw-mode", "fused"],
        ["--hsdp-shard-size", "2", "--zero-stage", "0"],
        ["--vae-gate-atol", "nan"],
        ["--latents", "/old-latents"],
    ],
)
def test_invalid_configurations_fail_before_loading_weights(extra):
    with pytest.raises(SystemExit):
        args(*extra)


def test_backend_and_compiler_resolution_are_independent():
    assert resolve_attention_backend("auto", "npu") == "npu"
    assert resolve_attention_backend("auto", "cuda") == "cuda"
    assert resolve_attention_backend("auto", "cpu") == "naive"
    assert resolve_attention_backend("naive", "cuda") == "naive"
    assert resolve_compile_backend("auto", "cuda") == "inductor"
    assert resolve_compile_backend("auto", "npu") == "npugraph_ex"
    with pytest.raises(ValueError, match="requires"):
        resolve_attention_backend("npu", "cuda")
    with pytest.raises(ValueError, match="requires"):
        resolve_compile_backend("npugraph_ex", "cuda")


def test_accumulation_counts_all_microsteps_and_deduplicates_cp_copies():
    leader, follower = AccumulationWindow(), AccumulationWindow(count_samples=False)
    for value, meta in [(1.0, TwoWayPackedMeta([2], [3])), (7.0, TwoWayPackedMeta([4, 5], [6, 7]))]:
        for window in (leader, follower):
            window.add(meta.num_samples, meta, torch.tensor(value), {"vision": torch.tensor(value)}, wait_ms=2.0)
    assert (leader.samples, leader.ar_tokens, leader.dm_tokens) == (3, 11, 16)
    assert (follower.samples, follower.ar_tokens, follower.dm_tokens) == (0, 0, 0)
    assert leader.loss_values() == follower.loss_values() == {"total": 4.0, "vision": 4.0}
    assert leader.wait_ms == 4.0


def test_the_understanding_key_norm_stays_frozen_like_the_reference():
    model = torch.nn.Module()
    model.time_embedder = torch.nn.Linear(2, 2)
    model.add_q_proj = torch.nn.Linear(2, 2)
    model.k_norm_und_for_gen = torch.nn.Linear(2, 2)
    model.frozen_tower = torch.nn.Linear(2, 2)

    trainable, frozen = entry.select_trainable_parameters(model)

    assert model.time_embedder.weight.requires_grad and model.add_q_proj.weight.requires_grad
    assert not model.k_norm_und_for_gen.weight.requires_grad
    assert not model.frozen_tower.weight.requires_grad
    assert trainable == 12 and frozen == 12


def test_build_optimizer_names_and_rates_both_recipe_groups():
    model = torch.nn.Module()
    model.time_embedder = torch.nn.Linear(2, 2)
    model.action_proj_in = torch.nn.Linear(2, 2)

    optimizer = entry.build_optimizer(model, choice="adamw", mode="plain", device_type="cpu")

    assert [(group["name"], group["lr"]) for group in optimizer.param_groups] == [
        ("base", entry.LR),
        ("action_heads", entry.LR * 5.0),
    ]


def test_the_recorded_learning_rates_come_from_each_parameter_group():
    optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 2e-4, "name": "base"},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3, "name": "action_heads"},
        ]
    )

    assert entry.optimizer_learning_rates(optimizer) == {"lr/base": 2e-4, "lr/action_heads": 1e-3}


def test_a_package_without_metadata_is_recorded_as_missing(monkeypatch):
    assert isinstance(entry.package_version("pytest"), str)

    def missing(name):
        raise entry.PackageNotFoundError(name)

    monkeypatch.setattr(entry, "version", missing)
    assert entry.package_version("diffusers") is None


class ActionHead(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.input_size = width
        self.fc = torch.nn.Embedding(3, width * width)
        self.bias = torch.nn.Embedding(3, width)


class ActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_proj_in = ActionHead()
        self.action_proj_out = ActionHead()
        self.action_modality_embed = torch.nn.Parameter(torch.ones(4))
        self.frozen_tower = torch.nn.Parameter(torch.ones(2))


def export_head(model):
    return {
        "action_modality_embed": model.action_modality_embed.detach().clone(),
        **{
            f"{reference}.{name}": param.detach().clone()
            for reference, head in (("action2llm", model.action_proj_in), ("llm2action", model.action_proj_out))
            for name, param in head.named_parameters()
        },
    }


def test_action_head_initialization_and_reference_load_are_complete_and_atomic(tmp_path):
    torch.manual_seed(9)
    model = ActionModel()
    assert set(entry.reinitialize_action_head(model)) == {"action_proj_in", "action_proj_out", "action_modality_embed"}
    assert model.action_proj_in.bias.weight.count_nonzero() == 0
    assert model.action_proj_out.fc.weight.abs().max() <= 3 / 4**0.5
    assert torch.equal(model.frozen_tower, torch.ones(2))
    state = export_head(model)
    path = tmp_path / "head.pt"
    torch.save(state, path)
    other = ActionModel()
    entry.load_action_head(other, path)
    for key, value in export_head(other).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    before = export_head(other)
    state["llm2action.bias.weight"] = torch.zeros(99)
    torch.save(state, path)
    with pytest.raises(ValueError, match="shape"):
        entry.load_action_head(other, path)
    for key, value in export_head(other).items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_domain_slice_leaves_other_action_domains_untouched(tmp_path):
    model = ActionModel()
    before = export_head(model)
    state = {
        key: torch.full_like(value if key == "action_modality_embed" else value[1], 0.25)
        for key, value in before.items()
    }
    torch.save({"format": "action-head-domain-slice", "domain_id": 1, **state}, tmp_path / "slice.pt")
    entry.load_action_head(model, tmp_path / "slice.pt")
    for key, value in export_head(model).items():
        if key != "action_modality_embed":
            torch.testing.assert_close(value[[0, 2]], before[key][[0, 2]], rtol=0, atol=0)
            assert torch.equal(value[1], state[key])


def test_sft_cycles_short_loader_and_reports_the_whole_optimizer_window(monkeypatch, tmp_path):
    config = args(
        "--out",
        str(tmp_path),
        "--samples-per-rank",
        "1",
        "--grad-accum",
        "3",
        "--max-iter",
        "2",
        "--zero-stage",
        "0",
        "--num-workers",
        "0",
        "--metrics-every",
        "1",
        "--log-every",
        "1",
    )
    monkeypatch.setattr(entry, "parse_args", lambda: config)
    monkeypatch.setattr(entry.colossalai, "launch_from_torch", lambda **kwargs: None)
    monkeypatch.setattr(entry.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(entry.dist, "get_rank", lambda group=None: 0)
    accelerator = SimpleNamespace(
        name="cpu",
        communication_backend="gloo",
        get_current_device=lambda: torch.device("cpu"),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(entry, "get_accelerator", lambda: accelerator)
    monkeypatch.setattr(
        entry, "DistCoordinator", lambda: SimpleNamespace(rank=0, world_size=1, print_on_master=lambda *a: None)
    )
    policy = SimpleNamespace(processor=SimpleNamespace(backend="naive"), packed_meta=None)
    policy.set_packed_meta = lambda meta: setattr(policy, "packed_meta", meta)
    monkeypatch.setattr(entry, "Cosmos3OmniPolicy", lambda **kwargs: policy)

    class Plugin:
        dp_group = None

        def __init__(self, **kwargs):
            assert kwargs["cast_inputs"] is False and kwargs["cast_buffers"] is False

        def prepare_dataloader(self, dataset, batch_size, distributed_sampler_cls, **kwargs):
            sampler = distributed_sampler_cls(dataset, num_replicas=1, rank=0, shuffle=False)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    monkeypatch.setattr(entry, "HybridParallelPlugin", Plugin)
    # The entry defaults to the vendored reader, so the stub stands in for both names; the default
    # path would otherwise reach for the real upstream pipeline and its dependencies.
    monkeypatch.setattr(
        entry,
        "DroidActionPolicyDataset",
        lambda *a, **kw: [{"goal": 1.0, "sample_id": "a"}, {"goal": 3.0, "sample_id": "b"}],
    )
    monkeypatch.setattr(
        entry,
        "UpstreamDroidDataset",
        lambda *a, **kw: [{"goal": 1.0, "sample_id": "a"}, {"goal": 3.0, "sample_id": "b"}],
    )

    class Model(torch.nn.Module):
        config = None

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def enable_gradient_checkpointing(self):
            pass

        def unwrap(self):
            return self

        def forward(self):
            return SimpleNamespace(sample=[self.weight.expand(1, 1, 1, 1, 1)], action=[self.weight.expand(1, 1)])

    model, reference = Model(), Model()
    monkeypatch.setattr(entry, "build_model", lambda *a: model)
    monkeypatch.setattr(entry, "build_tokenizer", lambda *a: None)
    monkeypatch.setattr(entry, "build_vae", lambda *a, **kw: torch.nn.Linear(1, 1))
    monkeypatch.setattr(entry, "build_optimizer", lambda model, *a: torch.optim.SGD(model.parameters(), lr=0.01))

    class Booster:
        def __init__(self, **kwargs):
            pass

        def boost(self, model, optimizer, dataloader):
            return model, optimizer, None, dataloader, None

        def backward(self, loss, optimizer):
            loss.backward()

    monkeypatch.setattr(entry, "Booster", Booster)

    def targets(goal):
        return {
            "vision": [torch.tensor(goal).expand(1, 1, 1, 1, 1)],
            "action": [torch.tensor(goal).expand(1, 1)],
            "vision_noisy_mask": [torch.ones(1, 1, 1, 1, 1)],
            "action_noisy_mask": [torch.ones(1, 1)],
            "raw_action_dim": 1,
        }

    def pack(batch, *a, **kw):
        policy.set_packed_meta(TwoWayPackedMeta([2], [4]))
        return {}, targets(float(batch["goal"][0]))

    monkeypatch.setattr(entry, "pack_batch", pack)
    records, summaries, manifests = [], [], []
    monkeypatch.setattr(
        entry,
        "TrainingMetrics",
        lambda **kwargs: SimpleNamespace(
            start=manifests.append, step=lambda step, **sections: records.append(sections), finish=summaries.append
        ),
    )
    monkeypatch.setattr(entry, "parallel_topology", lambda plugin: [])
    monkeypatch.setattr(entry, "device_memory_gib", lambda: {})
    entry.main()
    expected_losses = []
    reference_optim = torch.optim.SGD(reference.parameters(), lr=0.01)
    for goals in ((1.0, 3.0, 1.0), (3.0, 1.0, 3.0)):
        losses = [entry.compute_loss(reference(), targets(goal)) for goal in goals]
        loss = torch.stack(losses).mean()
        expected_losses.append(loss.item())
        loss.backward()
        reference_optim.step()
        reference_optim.zero_grad()
    torch.testing.assert_close(model.weight, reference.weight)
    assert [record["loss"]["total"] for record in records] == pytest.approx(expected_losses)
    assert [record["data"]["samples"] for record in records] == [3, 3]
    assert [record["data"]["ar_tokens"] for record in records] == [6, 6]
    assert all(record["optim"]["lr/0"] == 0.01 for record in records)
    assert manifests[0]["effective_global_batch"] == 3
    assert manifests[0]["save_kind"] == "model_export"
    assert summaries[0]["steps"] == 2 and summaries[0]["completed"]
