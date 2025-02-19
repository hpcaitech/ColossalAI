# modified from tests/kit/model_zoo/transformers/mistral.py
from types import MethodType

import torch
import transformers
from transformers import AutoConfig

# ===============================
# Register single-sentence Mixtral
# ===============================


def data_gen():
    # Generated from following code snippet
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("mixtralai/Mixtral-7B-v0.1")
    # input = 'My favourite condiment is vinegar' (last two words repeated to satisfy length requirement)
    # tokenized_input = tokenizer([input], return_tensors="pt")
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[1, 22, 55, 77, 532, 349, 43, 22]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data["labels"] = data["input_ids"].clone()
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn = lambda x: x[0].mean()
loss_fn_for_lm = lambda x: x.loss


def init_deepseek():

    config = AutoConfig.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        hidden_size=128,
        intermediate_size=320,
        kv_lora_rank=4,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_experts_per_tok=4,
        n_group=4,
        num_hidden_layers=3,
        num_key_value_heads=4,
        first_k_dense_replace=1,
        q_lora_rank=8,
        torch_dtype="bfloat16",
        n_routed_experts=16,
        topk_group=2,
        v_head_dim=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        trust_remote_code=True,
        vocab_size=2048,
    )

    if hasattr(config, "pad_token_id"):
        config.pad_token_id = config.eos_token_id
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # enable grad for moe layers
    for m in model.modules():
        if m.__class__.__name__ == "DeepseekV3MoE":
            m.moe_infer = MethodType(m.moe_infer.__wrapped__, m)
    return model
