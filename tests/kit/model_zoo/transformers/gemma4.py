"""Small native Gemma 4 dense models retaining PLE and shared KV layers."""

from functools import partial

import torch
import torch.nn.functional as F

from tests.kit.numerics import NumericsSpec, TokenLoss

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


def gemma4_numerics_data():
    # Different micro-batches must have different valid-token counts to distinguish global token normalization
    # from an average of per-micro-batch means. All randomness belongs here, before any reference or plugin runs.
    batch = gemma4_data()
    # Keep one visible key in the last sliding window, even for padded queries. HF eager attention upcasts
    # softmax to FP32: a completely masked FP64 row becomes all -inf and yields NaNs instead of a valid truth.
    length = int(torch.randint(5, 7, ()).item())
    batch["input_ids"][1, length:] = 0
    batch["attention_mask"][1, length:] = 0
    batch["labels"][1, length:] = -100
    return batch


def gemma4_numerics_prepare(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def gemma4_summed_token_loss(output, batch):
    # Native HF causal LM loss upcasts logits to float32 and predicts each token from the preceding position.
    logits = output.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")


def gemma4_valid_tokens(batch):
    return batch["labels"][:, 1:].ne(-100).sum()


def gemma4_output(output):
    return {"logits": output.logits, "loss": output.loss}


def gemma4_loss(output):
    return output["loss"]


if Gemma4ForCausalLM is not None:
    for variant in ("e2b", "e4b"):
        model_zoo.register(
            f"transformers_gemma4_{variant}_for_causal_lm",
            partial(tiny_gemma4, variant),
            gemma4_numerics_data,
            gemma4_output,
            gemma4_loss,
            ModelAttribute(
                has_control_flow=True,
                numerics=NumericsSpec(
                    prepare_fn=gemma4_numerics_prepare,
                    token_loss=TokenLoss(loss=gemma4_summed_token_loss, count=gemma4_valid_tokens),
                    tied_params=(("model.embed_tokens.weight", "lm_head.weight"),),
                    # Native FP16 backward overflows near the first RMSNorm with scale 65536. Both runs use
                    # this declared scale; skipped optimizer updates remain failures.
                    fp16_loss_scale=1024.0,
                ),
            ),
        )
