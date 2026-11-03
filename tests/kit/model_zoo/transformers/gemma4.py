"""Small native Gemma 4 dense models retaining PLE and shared KV layers."""

from functools import partial

import torch

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
except ImportError:
    Gemma4ForCausalLM = None


def tiny_gemma4_config(variant="e2b"):
    if variant not in ("e2b", "e4b"):
        raise ValueError(f"Unknown dense variant: {variant}")
    cycle = 5 if variant == "e2b" else 6
    return Gemma4TextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2 * cycle,
        num_attention_heads=2 if variant == "e2b" else 4,
        num_key_value_heads=1 if variant == "e2b" else 2,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=cycle,
        use_double_wide_mlp=variant == "e2b",
        layer_types=(["sliding_attention"] * (cycle - 1) + ["full_attention"]) * 2,
        sliding_window=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        final_logit_softcapping=30.0,
        use_cache=False,
        attn_implementation="eager",
        pad_token_id=0,
    )


def tiny_gemma4(variant="e2b"):
    return Gemma4ForCausalLM(tiny_gemma4_config(variant))


def gemma4_data():
    ids = torch.tensor([[2, 5, 9, 12, 21, 34, 55, 3], [2, 7, 11, 23, 31, 47, 0, 0]])
    mask = ids.ne(0).long()
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


if Gemma4ForCausalLM is not None:
    for variant in ("e2b", "e4b"):
        model_zoo.register(
            f"transformers_gemma4_{variant}_for_causal_lm",
            partial(tiny_gemma4, variant),
            gemma4_data,
            lambda output: {"logits": output.logits, "loss": output.loss},
            lambda output: output["loss"],
            ModelAttribute(has_control_flow=True),
        )
