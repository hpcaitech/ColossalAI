"""Pinned Ouro fixtures and an independent official-computation reference.

Importing this module never downloads code. Pinned snapshots come from the offline cache or the
hash-checked source-only fixtures. The reference only bridges HF5 construction and
builds its own triangular mask; it does not call ColossalAI's Ouro adapter.
"""

import sys
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.kit.numerics import NumericsSpec, TokenLoss
from tests.kit.remote_models import load_remote_classes, tiny_config


def reset_calls(module, inputs):
    """Reset counts for the next full model forward."""
    for layer in module.model.layers:
        layer._ouro_calls = 0


def count_call(module, inputs):
    """A pre-hook also observes non-reentrant checkpoint recomputation."""
    module._ouro_calls += 1


def check_calls(module, inputs, output):
    """Ensure that every shared decoder really executes the requested loop."""
    expected = module.config.total_ut_steps
    actual = [layer._ouro_calls for layer in module.model.layers]
    if actual != [expected] * len(actual):
        raise AssertionError(f"Ouro loop coverage: {actual}, expected {expected} per layer")


def _manual_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
    """Call the official loop using an independently constructed eager mask."""
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch, length = inputs_embeds.shape[:2]
    allowed = torch.ones(length, length, device=inputs_embeds.device, dtype=torch.bool).tril()
    allowed = allowed[None, None].expand(batch, 1, length, length)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()
    mask = torch.zeros_like(allowed, dtype=inputs_embeds.dtype).masked_fill(
        ~allowed, torch.finfo(inputs_embeds.dtype).min
    )
    return type(self).forward(self, inputs_embeds=inputs_embeds, attention_mask={"full_attention": mask}, **kwargs)


def make_ouro(key="ouro14", loops=4, checkpoint=False, reference=False, apply_policy=True):
    """Build the two-layer tiny model, retaining all official loop/gate logic."""
    _, _, original = load_remote_classes(key)
    config = tiny_config(key)
    config.total_ut_steps = loops
    if reference:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        module = sys.modules[original.__module__]
        compute = Qwen2RotaryEmbedding.compute_default_rope_parameters
        # Construction-only, test-only bridge. Restore every upstream symbol
        # before executing either the candidate or reference forward.
        with patch.object(module, "ROPE_INIT_FUNCTIONS", {**module.ROPE_INIT_FUNCTIONS, "default": compute}):
            with patch.object(
                module.OuroRotaryEmbedding, "compute_default_rope_parameters", staticmethod(compute), create=True
            ):
                model = original(config)
        model.model.forward = MethodType(_manual_forward, model.model)
    else:
        from colossalai.shardformer import ShardConfig, ShardFormer
        from colossalai.shardformer.modeling.ouro import get_ouro_model_class

        model = get_ouro_model_class(original)(config)
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if apply_policy:
            model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)

    for layer in model.model.layers:
        layer.register_forward_pre_hook(count_call)
    model.register_forward_pre_hook(reset_calls)
    model.register_forward_hook(check_calls)
    return model


def ouro_data():
    """Vary valid-token counts across micro-batches without model-side RNG."""
    ids = torch.randint(5, 128, (2, 16))
    length = int(torch.randint(8, 16, ()).item())
    mask = torch.ones_like(ids)
    mask[1, length:] = 0
    ids[1, length:] = 0
    return dict(input_ids=ids, attention_mask=mask, labels=ids.masked_fill(mask == 0, -100), use_cache=False)


def ouro_output(output):
    """Include the actual language-model loss in the numerical comparison."""
    return {"logits": output.logits, "loss": output.loss}


def ouro_loss(output):
    """The model zoo's training criterion."""
    return output["loss"]


def token_loss(output, batch):
    """Summed causal CE, normalized by the tool over the entire global step."""
    return F.cross_entropy(
        output.logits[:, :-1].float().flatten(0, 1),
        batch["labels"][:, 1:].flatten(),
        ignore_index=-100,
        reduction="sum",
    )


def token_count(batch):
    """Number of next-token targets, excluding padding."""
    return batch["labels"][:, 1:].ne(-100).sum()


def ouro_entry(key="ouro14", loops=4, checkpoint=False):
    """A picklable numerical-tool entry with a separate official reference."""
    spec = NumericsSpec(
        reference_fn=partial(make_ouro, key, loops, False, True),
        token_loss=TokenLoss(token_loss, token_count),
    )
    return (
        partial(make_ouro, key, loops, checkpoint),
        ouro_data,
        ouro_output,
        ouro_loss,
        SimpleNamespace(has_control_flow=True, has_stochastic_depth_prob=False, numerics=spec),
    )
