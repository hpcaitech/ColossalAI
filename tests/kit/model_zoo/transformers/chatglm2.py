import torch

from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel
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

config = ChatGLMConfig(
    num_layers=2,
    padded_vocab_size=65024,
    hidden_size=64,
    num_attention_heads=8,
    kv_channels=16,
    rmsnorm=True,
    original_rope=True,
    use_cache=True,
    torch_dtype=torch.float32,
)

infer_config = ChatGLMConfig(
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

model_zoo.register(
    name="transformers_chatglm",
    model_fn=lambda: ChatGLMModel(config, empty_init=False),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_chatglm_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_chatglm_for_conditional_generation",
    model_fn=lambda: ChatGLMForConditionalGeneration(config, empty_init=False),
    data_gen_fn=data_gen_for_conditional_generation,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
