"""Picklable tiny Ouro entries for the shared numerical-alignment tool."""

from functools import partial
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tests.kit.numerics import NumericsSpec, TokenLoss
from tests.kit.remote_models import build_remote_model, tiny_config


def reset_calls(module, inputs):
    """Count real decoder invocations in every forward, including plugin runs."""
    for layer in module.model.layers:
        layer._d2d4_calls = 0


def count_call(module, inputs, output):
    module._d2d4_calls += 1


def check_calls(module, inputs, output):
    expected = module.config.total_ut_steps
    actual = [layer._d2d4_calls for layer in module.model.layers]
    if actual != [expected] * len(actual):
        raise AssertionError(f"Ouro loop coverage: expected {expected} calls/layer, got {actual}")


def make_model(key="ouro14", loops=4, size="tiny"):
    config = tiny_config(key)
    config.total_ut_steps = loops
    if size == "small":
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_hidden_layers = 4
        config.head_dim = 32
        config.layer_types = ["full_attention"] * 4
    model = build_remote_model(key, config=config)
    if loops == 1:
        # A single-loop control has no differentiable gate contribution.
        # Freeze this intentionally unused branch on both sides; otherwise
        # Gemini rejects missing gradients before the accumulation comparison.
        model.model.early_exit_gate.requires_grad_(False)
    for layer in model.model.layers:
        layer.register_forward_hook(count_call)
    model.register_forward_pre_hook(reset_calls)
    model.register_forward_hook(check_calls)
    return model


def make_data():
    """Draw once per batch; prepare() reuses these exact tensors across runs."""
    ids = torch.randint(5, 128, (2, 16))
    return dict(input_ids=ids, labels=ids.clone(), use_cache=False, return_dict=True)


def select_logits(output):
    """The shared tool uses its fixed random-projection loss on these logits."""
    return {"logits": output.logits}


def token_loss(output, batch):
    """Actual causal language-model CE, summed before global normalization."""
    return F.cross_entropy(
        output.logits[:, :-1].float().flatten(0, 1),
        batch["labels"][:, 1:].flatten(),
        ignore_index=-100,
        reduction="sum",
    )


def token_count(batch):
    return (batch["labels"][:, 1:] != -100).sum()


def entry(key="ouro14", loops=4, size="tiny"):
    unused = ("model.early_exit_gate.weight", "model.early_exit_gate.bias") if loops == 1 else ()
    spec = NumericsSpec(unused_params=unused, token_loss=TokenLoss(token_loss, token_count))
    return (partial(make_model, key, loops, size), make_data, select_logits, None, SimpleNamespace(numerics=spec))
