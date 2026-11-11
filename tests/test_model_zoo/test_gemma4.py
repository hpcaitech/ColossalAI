import copy

import pytest
import torch
import torch.distributed as dist
from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn
from examples.language.gemma4.train import freeze_non_text
from tests.kit.model_zoo import model_zoo
from tests.kit.model_zoo.transformers.gemma4 import gemma4_data, gemma4_numerics_data, tiny_gemma4
from tests.kit.numerics import assert_aligned


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_gemma4_native_alignment(variant, checkpoint):
    torch.manual_seed(42)
    reference = tiny_gemma4(variant).double().train()
    candidate = copy.deepcopy(reference)
    former = ShardFormer(ShardConfig(enable_tensor_parallelism=False))
    candidate, _ = former.optimize(candidate)
    if checkpoint:
        candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert candidate.get_input_embeddings().weight is candidate.get_output_embeddings().weight
    batch = gemma4_data()
    outputs, grads, params, losses = [], [], [], []
    for model in (reference, candidate):
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_curve = []
        for _ in range(3):
            optim.zero_grad()
            output = model(**batch)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert any("per_layer" in name for name, _ in model.named_parameters())
            if not loss_curve:
                outputs.append({"logits": output.logits.detach(), "loss": output.loss.detach()})
                grads.append({name: p.grad.clone() for name, p in model.named_parameters()})
            loss_curve.append(output.loss.detach())
            optim.step()
        params.append({name: p.detach().clone() for name, p in model.named_parameters()})
        losses.append({"curve": torch.stack(loss_curve)})
    for traces in (outputs, grads, params, losses):
        assert_aligned(traces[1], traces[0], traces[0], tol=1e-8)


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_hf_roundtrip(variant, tmp_path):
    model = tiny_gemma4(variant).eval()
    batch = gemma4_data()
    with torch.no_grad():
        expected = model(**batch).logits
    model.save_pretrained(tmp_path)
    restored = Gemma4ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch).logits, expected, rtol=0, atol=0)


def test_gemma4_registered():
    assert len(model_zoo.get_sub_registry("transformers_gemma4_")) == 2


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_numerics_spec(variant):
    entry = model_zoo[f"transformers_gemma4_{variant}_for_causal_lm"]
    model = entry[0]()
    spec = entry[4].numerics
    spec.prepare_fn(model)
    assert model.is_gradient_checkpointing
    params = dict(model.named_parameters(remove_duplicate=False))
    for names in spec.tied_params:
        assert all(params[name] is params[names[0]] for name in names)
    assert spec.unused_params == ()
    batch = gemma4_data()
    output = model(**batch)
    summed = spec.token_loss.loss(output, batch)
    count = spec.token_loss.count(batch)
    assert count.item() == 12  # seven causal targets in the first row, five in the padded row
    torch.testing.assert_close(summed / count, output.loss)
    summed.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any("per_layer" in name for name, _ in model.named_parameters())

    empty = dict(batch, labels=torch.full_like(batch["labels"], -100))
    assert spec.token_loss.count(empty).item() == 0
    assert spec.token_loss.loss(output, empty).item() == 0


def test_gemma4_numerics_batches_vary_valid_tokens():
    # Repeating a seed must reproduce the complete generated batches, while successive batches exercise padding.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        batches = [gemma4_numerics_data() for _ in range(8)]
        torch.manual_seed(17)
        repeated = [gemma4_numerics_data() for _ in range(8)]
    counts = set()
    for batch, again in zip(batches, repeated):
        for name in batch:
            torch.testing.assert_close(batch[name], again[name], rtol=0, atol=0)
        assert batch["attention_mask"][0].all()
        assert torch.equal(batch["labels"].eq(-100), batch["attention_mask"].eq(0))
        counts.add(batch["labels"][:, 1:].ne(-100).sum().item())
    assert len(counts) > 1


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_numerics_fp64_padding(variant):
    model = tiny_gemma4(variant).double()
    for length in (5, 6):
        batch = gemma4_data()
        batch["input_ids"][1, length:] = 0
        batch["attention_mask"][1, length:] = 0
        batch["labels"][1, length:] = -100
        model.zero_grad()
        output = model(**batch)
        assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
        output.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_dense_structure(variant):
    model = tiny_gemma4(variant)
    cycle = model.config.num_hidden_layers // 2
    masks = {}

    def capture(index):
        def hook(module, args, kwargs):
            masks[index] = kwargs["attention_mask"].detach().clone()

        return hook

    handles = [
        model.model.layers[index].self_attn.register_forward_pre_hook(capture(index), with_kwargs=True)
        for index in (0, cycle - 1)
    ]
    try:
        model(**gemma4_data()).loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    # The last token cannot attend to the first through the sliding window,
    # while full attention retains both the first and the preceding token.
    local, full = masks[0][0, 0], masks[cycle - 1][0, 0]
    assert local[-1, 0] != local[-1, -2]
    assert full[-1, 0] == full[-1, -2]
    for index, layer in enumerate(model.model.layers):
        shared = index >= cycle
        assert layer.self_attn.is_kv_shared_layer == shared
        assert hasattr(layer.self_attn, "k_proj") != shared
        width = model.config.intermediate_size * (2 if shared and variant == "e2b" else 1)
        assert layer.mlp.gate_proj.weight.shape[0] == width
        assert layer.self_attn.q_proj.weight.grad is not None


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_conditional_text_path(variant):
    config = Gemma4Config(text_config=tiny_gemma4(variant).config.to_dict())
    model = Gemma4ForConditionalGeneration(config)
    freeze_non_text(model)
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    model(**gemma4_data()).loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_fp32_baseline(variant):
    torch.manual_seed(17)
    truth_model = tiny_gemma4(variant).double()
    reference = copy.deepcopy(truth_model).float()
    candidate, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(copy.deepcopy(reference))
    traces = []
    for model in (truth_model, reference, candidate):
        output = model(**gemma4_data())
        output.loss.backward()
        traces.append(
            {
                "logits": output.logits.detach(),
                "loss": output.loss.detach(),
                **{f"grad/{name}": p.grad.detach() for name, p in model.named_parameters()},
            }
        )
    assert_aligned(traces[2], traces[1], traces[0])


def test_gemma4_rejects_unimplemented_fusion():
    with pytest.raises(NotImplementedError, match="native Gemma"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)).optimize(tiny_gemma4())


def _ddp_equivalence(rank, world_size, port, variant):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", seed=42)
    try:
        reference = tiny_gemma4(variant)
        truth = copy.deepcopy(reference).double()
        accumulated = copy.deepcopy(reference)
        candidate = copy.deepcopy(reference)
        candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
        booster = Booster(plugin=TorchDDPPlugin())
        candidate, optimizer, _, _, _ = booster.boost(model=candidate, optimizer=optimizer)
        batch = gemma4_data()
        # Causal losses average valid target tokens, not examples. Preserve unequal padding.
        total_tokens = batch["labels"][:, 1:].ne(-100).sum()
        reference(**batch).loss.backward()
        truth(**batch).loss.backward()
        for index in range(batch["input_ids"].shape[0]):
            micro_batch = {k: v[index : index + 1] for k, v in batch.items()}
            tokens = micro_batch["labels"][:, 1:].ne(-100).sum()
            (accumulated(**micro_batch).loss * (tokens / total_tokens)).backward()
        local_batch = {k: v[rank::world_size] for k, v in batch.items()}
        local_tokens = local_batch["labels"][:, 1:].ne(-100).sum()
        # DDP averages rank gradients; compensate using the global valid-token count.
        booster.backward(candidate(**local_batch).loss * (world_size * local_tokens / total_tokens), optimizer)
        grads = lambda m: {n: p.grad.detach() for n, p in m.named_parameters()}
        assert_aligned(grads(accumulated), grads(reference), grads(truth))
        assert_aligned(grads(candidate.unwrap()), grads(accumulated), grads(truth))
        optimizer.step()
        for model in (reference, truth, accumulated):
            torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        params = lambda m: {n: p.detach() for n, p in m.named_parameters()}
        assert_aligned(params(candidate.unwrap()), params(accumulated), params(truth))
        assert_aligned(params(accumulated), params(reference), params(truth))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
@pytest.mark.parametrize("world_size", [1, 2])
def test_gemma4_cpu_ddp_equivalence(variant, world_size):
    spawn(_ddp_equivalence, world_size, variant=variant)


@pytest.mark.parametrize("variant", ["e2b", "e4b"])
def test_gemma4_cpu_optimizer_resume(variant, tmp_path):
    from tests.kit.numerics.scenarios.resume import unsharded_optimizer_reloads

    if not unsharded_optimizer_reloads():
        pytest.skip(
            "Legacy optimizer checkpoint loading requires torch >= 2.10; covered on the maintained Torch 2.13 image"
        )
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        booster = Booster(plugin=TorchDDPPlugin())
        model = tiny_gemma4(variant)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
        batch = gemma4_data()

        def step(m, opt):
            opt.zero_grad()
            loss = m(**batch).loss
            booster.backward(loss, opt)
            opt.step()
            return loss.detach()

        step(model, optimizer)
        booster.save_model(model, str(tmp_path / "model.pt"))
        booster.save_optimizer(optimizer, str(tmp_path / "optimizer.pt"))
        expected_loss = step(model, optimizer)
        restored = tiny_gemma4(variant)
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored, restored_opt, _, _, _ = booster.boost(model=restored, optimizer=restored_opt)
        booster.load_model(restored, str(tmp_path / "model.pt"))
        booster.load_optimizer(restored_opt, str(tmp_path / "optimizer.pt"))
        torch.testing.assert_close(step(restored, restored_opt), expected_loss, rtol=0, atol=0)
        for n, p in model.unwrap().state_dict().items():
            torch.testing.assert_close(restored.unwrap().state_dict()[n], p, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
