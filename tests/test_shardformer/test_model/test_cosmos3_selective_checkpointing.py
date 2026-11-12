"""CPU check for Cosmos 3 selective activation checkpointing.

The point of the feature is where the checkpoint boundary falls, so the check is: the same statements, the same
gradients, different work on the backward pass. A two-layer stand-in model is run once with the stock forward and once
with the patched one, and the two gradient sets have to match. The MLP and attention call counters then show what
actually got recomputed: the MLP twice, the attention once.

No device and no diffusers install is needed.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from colossalai.shardformer.modeling.cosmos3 import enable_cosmos3_selective_checkpointing

HIDDEN = 8


class _PathAttention(torch.nn.Module):
    """A stand-in for the packed MoT attention: two pathways, cross-attending, no masks."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.to_q = torch.nn.Linear(hidden, hidden, bias=False)
        self.to_k = torch.nn.Linear(hidden, hidden, bias=False)
        self.to_v = torch.nn.Linear(hidden, hidden, bias=False)
        self.calls = 0

    def forward(self, und_seq, gen_seq, rotary_emb):
        self.calls += 1
        q_und, q_gen = self.to_q(und_seq), self.to_q(gen_seq)
        k_und, k_gen = self.to_k(und_seq), self.to_k(gen_seq)
        v_und, v_gen = self.to_v(und_seq), self.to_v(gen_seq)
        scale = q_und.shape[-1] ** -0.5
        und_out = F.scaled_dot_product_attention(q_und, k_gen, v_gen, scale=scale)
        gen_out = F.scaled_dot_product_attention(q_gen, k_und, v_und, scale=scale)
        return und_out, gen_out


class _CountingMLP(torch.nn.Module):
    """The 4x up-projection a block's MLP carries, with a call counter."""

    def __init__(self, hidden: int = HIDDEN, intermediate: int = 4 * HIDDEN):
        super().__init__()
        self.up = torch.nn.Linear(hidden, intermediate)
        self.down = torch.nn.Linear(intermediate, hidden)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.down(F.silu(self.up(x)))


class Cosmos3VLTextMoTDecoderLayer(torch.nn.Module):
    """The diffusers decoder layer, statement for statement, with counters on the two halves.

    The class name is what ``enable_cosmos3_selective_checkpointing`` keys on, exactly as the policy keys on the
    attention class name.
    """

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.self_attn = _PathAttention(hidden)
        self.mlp = _CountingMLP(hidden)
        self.mlp_moe_gen = _CountingMLP(hidden)
        self.input_layernorm = torch.nn.LayerNorm(hidden)
        self.input_layernorm_moe_gen = torch.nn.LayerNorm(hidden)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden)
        self.post_attention_layernorm_moe_gen = torch.nn.LayerNorm(hidden)

    def forward(self, und_seq, gen_seq, rotary_emb):
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen


class _Stack(torch.nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(Cosmos3VLTextMoTDecoderLayer() for _ in range(num_layers))

    def forward(self, und_seq, gen_seq, rotary_emb):
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        return und_seq, gen_seq


def _inputs(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    und_seq = torch.randn(5, HIDDEN, generator=generator).requires_grad_(True)
    gen_seq = torch.randn(4, HIDDEN, generator=generator).requires_grad_(True)
    rotary_emb = (None, None, None, None)  # this stand-in attention does not rotate
    return und_seq, gen_seq, rotary_emb


def _gradients(model, inputs):
    model.zero_grad(set_to_none=True)
    und_out, gen_out = model(*inputs)
    (und_out.square().sum() + gen_out.square().sum()).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def _counters(model):
    layers = [m for m in model.modules() if m.__class__.__name__ == "Cosmos3VLTextMoTDecoderLayer"]
    return {
        "attn": sum(layer.self_attn.calls for layer in layers),
        "mlp": sum(layer.mlp.calls + layer.mlp_moe_gen.calls for layer in layers),
    }


def test_selective_checkpointing_matches_the_stock_forward():
    """Same statements, same gradients; the MLP is recomputed and the attention is not."""
    torch.manual_seed(11)
    stock = _Stack()
    patched = copy.deepcopy(stock)

    reference = _gradients(stock, _inputs())
    assert _counters(stock) == {"attn": 2, "mlp": 4}

    assert enable_cosmos3_selective_checkpointing(patched) == 2
    got = _gradients(patched, _inputs())

    assert len(got) == len(reference)
    for expected, actual in zip(reference, got):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # Two layers: the attention ran once per layer in the forward and was never re-run; each of the two MLPs per
    # layer ran once in the forward and once during the recomputation.
    assert _counters(patched) == {"attn": 2, "mlp": 8}


def test_budgeted_sac_path_matches_and_recomputes_a_whole_layer():
    """Forcing the SAC path with no budget must match the stock gradients and recompute both halves."""
    torch.manual_seed(12)
    stock = _Stack()
    patched = copy.deepcopy(stock)

    generator = torch.Generator().manual_seed(4)
    und_seq = torch.randn(5, HIDDEN, generator=generator).requires_grad_(True)
    gen_seq = torch.randn(4, HIDDEN, generator=generator).requires_grad_(True)
    rotary_emb = tuple(torch.zeros(1) for _ in range(4))
    reference = _gradients(stock, (und_seq, gen_seq, rotary_emb))

    assert enable_cosmos3_selective_checkpointing(patched, budget_gib=0.0) == 2
    got = _gradients(patched, (und_seq, gen_seq, rotary_emb))

    for expected, actual in zip(reference, got):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # A zero budget saves nothing, so the whole layer is recomputed: attention and both MLPs run twice.
    assert _counters(patched) == {"attn": 4, "mlp": 8}


def test_installer_is_idempotent_and_rejects_a_foreign_model():
    model = _Stack(num_layers=3)
    assert enable_cosmos3_selective_checkpointing(model) == 3
    assert enable_cosmos3_selective_checkpointing(model) == 0

    with pytest.raises(RuntimeError, match="Cosmos3VLTextMoTDecoderLayer"):
        enable_cosmos3_selective_checkpointing(torch.nn.Linear(2, 2))
