import torch
import transformers
from transformers import MistralConfig

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence Mistral
# ===============================


def data_gen():
    # Generated from following code snippet
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # input = 'My favourite condiment is vinegar' (last two words repeated to satisfy length requirement)
    # tokenized_input = tokenizer([input], return_tensors="pt")
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[1, 1984, 16020, 2076, 2487, 349, 21375, 4749]], dtype=torch.int64)
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
loss_fn_for_mistral_model = lambda x: torch.nn.functional.mse_loss(
    x.last_hidden_state, torch.ones_like(x.last_hidden_state)
)
loss_fn = lambda x: x.loss
loss_fn_for_seq_classification = lambda output: output.logits.mean()

config = MistralConfig(
    hidden_size=256, intermediate_size=256, num_attention_heads=64, num_hidden_layers=2, vocab_size=50258
)

if hasattr(config, "pad_token_id"):
    config.pad_token_id = config.eos_token_id

model_zoo.register(
    name="transformers_mistral",
    model_fn=lambda: transformers.MistralModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_mistral_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_mistral_for_casual_lm",
    model_fn=lambda: transformers.MistralForCausalLM(config),
    data_gen_fn=data_gen_for_lm,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_mistral_for_sequence_classification",
    model_fn=lambda: transformers.MistralForSequenceClassification(config),
    data_gen_fn=data_gen_for_sequence_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_seq_classification,
    model_attribute=ModelAttribute(has_control_flow=True),
)
