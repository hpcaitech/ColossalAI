# modified from tests/kit/model_zoo/transformers/mistral.py
import torch
import transformers
from transformers import MixtralConfig

from ..registry import ModelAttribute, model_zoo

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


def data_gen_for_sequence_classification():
    # sequence classification data gen
    data = data_gen()
    data["labels"] = torch.tensor([1], dtype=torch.int64)
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_mixtral_model = lambda x: x[0].mean()
loss_fn = lambda x: x.loss
loss_fn_for_seq_classification = lambda output: output.logits.mean()

config = MixtralConfig(
    hidden_size=32,
    intermediate_size=32,
    num_attention_heads=8,
    num_hidden_layers=2,
    vocab_size=1000,
    attn_implementation="flash_attention_2",
    torch_dtype="float16",
    output_router_logits=True,
)

if hasattr(config, "pad_token_id"):
    config.pad_token_id = config.eos_token_id

model_zoo.register(
    name="transformers_mixtral",
    model_fn=lambda: transformers.MixtralModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_mixtral_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
# model_zoo.register(
#     name="transformers_mixtral_for_casual_lm",
#     model_fn=lambda: transformers.MixtralForCausalLM(config),
#     data_gen_fn=data_gen_for_lm,
#     output_transform_fn=output_transform_fn,
#     loss_fn=loss_fn,
#     model_attribute=ModelAttribute(has_control_flow=True),
# )
# model_zoo.register(
#     name="transformers_mixtral_for_sequence_classification",
#     model_fn=lambda: transformers.MixtralForSequenceClassification(config),
#     data_gen_fn=data_gen_for_sequence_classification,
#     output_transform_fn=output_transform_fn,
#     loss_fn=loss_fn_for_seq_classification,
#     model_attribute=ModelAttribute(has_control_flow=True),
# )
