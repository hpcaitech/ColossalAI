import torch
import transformers

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import Qwen3MoeConfig

    HAS_QWEN3_MOE = True
except ImportError:
    HAS_QWEN3_MOE = False


if HAS_QWEN3_MOE:

    def data_gen():
        input_ids = torch.tensor([[1, 22, 55, 77, 53, 34, 43, 22]], dtype=torch.long)
        return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def data_gen_for_causal_lm():
        data = data_gen()
        data["labels"] = data["input_ids"].clone()
        return data

    output_transform_fn = lambda x: x
    loss_fn = lambda output: output["last_hidden_state"].mean()
    loss_fn_for_causal_lm = lambda output: output["loss"]

    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        mlp_only_layers=[1],
        max_position_embeddings=64,
        use_cache=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        experts_implementation="eager",
        attn_implementation="eager",
        output_router_logits=True,
    )
    config.pad_token_id = 0

    def is_router(name, module):
        return name.endswith("mlp.gate")

    attributes = ModelAttribute(has_control_flow=True)
    if hasattr(attributes, "numerics"):
        from tests.kit.numerics.spec import NumericsSpec, Routers

        attributes.numerics = NumericsSpec(routers=Routers(select=is_router, top_k=config.num_experts_per_tok))

    model_zoo.register(
        name="transformers_qwen3_moe",
        model_fn=lambda: transformers.Qwen3MoeModel(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn,
        model_attribute=attributes,
    )
    model_zoo.register(
        name="transformers_qwen3_moe_for_causal_lm",
        model_fn=lambda: transformers.Qwen3MoeForCausalLM(config),
        data_gen_fn=data_gen_for_causal_lm,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_causal_lm,
        model_attribute=attributes,
    )
