import copy

import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence GPT
# ===============================


def data_gen():
    # Generated from following code snippet
    #
    # from transformers import AutoTokenizer
    # input = 'Hello, my dog is cute is cute' (last two words repeated to satisfy length requirement)
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data["labels"] = data["input_ids"].clone()
    return data


def data_gen_for_question_answering():
    # question answering data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    start_positions = torch.tensor([0], dtype=torch.int64)
    data["start_positions"] = start_positions
    end_positions = torch.tensor([1], dtype=torch.int64)
    data["end_positions"] = end_positions
    return data


def data_gen_for_sequence_classification():
    # sequence classification data gen
    data = data_gen()
    data["labels"] = torch.tensor([1], dtype=torch.int64)
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_gptj_model = lambda x: torch.nn.functional.mse_loss(
    x.last_hidden_state, torch.ones_like(x.last_hidden_state)
)
loss_fn = lambda x: x.loss

config = transformers.GPTJConfig(
    n_layer=2,
    n_head=4,
    vocab_size=50258,
    n_embd=256,
    hidden_size=256,
    n_positions=512,
    attn_pdrop=0,
    embd_pdrop=0,
    resid_pdrop=0,
    hidden_dropout=0,
    problem_type="single_label_classification",
    pad_token_id=50256,
)

config_for_token_classification = copy.deepcopy(config)
config_for_token_classification.num_labels = 2

# register the following models
model_zoo.register(
    name="transformers_gptj",
    model_fn=lambda: transformers.GPTJModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_gptj_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gptj_lm",
    model_fn=lambda: transformers.GPTJForCausalLM(config),
    data_gen_fn=data_gen_for_lm,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gptj_for_question_answering",
    model_fn=lambda: transformers.GPTJForQuestionAnswering(config),
    data_gen_fn=data_gen_for_question_answering,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gptj_for_sequence_classification",
    model_fn=lambda: transformers.GPTJForSequenceClassification(config_for_token_classification),
    data_gen_fn=data_gen_for_sequence_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
