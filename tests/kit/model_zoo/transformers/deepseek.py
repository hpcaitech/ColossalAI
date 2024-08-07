# modified from tests/kit/model_zoo/transformers/mistral.py
import torch
import transformers
from transformers import AutoConfig

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


def init_deepseek():

    config = AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-moe-16b-base",
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        # vocab_size=2200,
        first_k_dense_replace=1,
        attn_implementation="flash_attention_2",
        torch_dtype="float16",
        n_routed_experts=8,
        trust_remote_code=True,
    )

    if hasattr(config, "pad_token_id"):
        config.pad_token_id = config.eos_token_id
    model = transformers.AutoModel.from_config(config, trust_remote_code=True)

    return model


model_zoo.register(
    name="transformers_deepseek",
    model_fn=init_deepseek,
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_mixtral_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
