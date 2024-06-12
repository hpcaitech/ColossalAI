import torch
from torch.nn import init
from transformers import AutoConfig, AutoModelForCausalLM

from ..registry import ModelAttribute, model_zoo

# ================================
# Register single-sentence ChatGLM
# ================================


def data_gen():
    input_ids = torch.tensor([[5941, 15, 2670, 3543, 632, 2075, 632, 2075]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_conditional_generation():
    # token classification data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    labels = data["input_ids"].clone()
    data["labels"] = labels
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_chatglm_model = lambda x: torch.nn.functional.mse_loss(
    x["last_hidden_state"], torch.ones_like(x["last_hidden_state"])
)
loss_fn = lambda x: x["loss"]


infer_config = AutoConfig.from_pretrained(
    "THUDM/chatglm2-6b",
    trust_remote_code=True,
    num_layers=2,
    padded_vocab_size=65024,
    hidden_size=128,
    num_attention_heads=8,
    multi_query_attention=True,
    multi_query_group_num=2,
    kv_channels=16,
    rmsnorm=True,
    original_rope=True,
    use_cache=True,
    torch_dtype=torch.float32,
)


def init_chatglm():
    config = AutoConfig.from_pretrained(
        "THUDM/chatglm2-6b",
        trust_remote_code=True,
        num_layers=2,
        padded_vocab_size=65024,
        hidden_size=64,
        ffn_hidden_size=214,
        num_attention_heads=8,
        kv_channels=16,
        rmsnorm=True,
        original_rope=True,
        use_cache=True,
        multi_query_attention=False,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config, empty_init=False, trust_remote_code=True)
    for m in model.modules():
        if m.__class__.__name__ == "RMSNorm":
            init.ones_(m.weight)
    return model


model_zoo.register(
    name="transformers_chatglm_for_conditional_generation",
    model_fn=init_chatglm,
    data_gen_fn=data_gen_for_conditional_generation,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
