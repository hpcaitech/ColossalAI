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
    # from transformers import GPT2Tokenizer
    # input = 'Hello, my dog is cute is cute' (last two words repeated to satisfy length requirement)
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    # input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
    # attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    input_ids = torch.tensor(
        [
            [15496, 11, 616, 3290, 318, 13779, 318, 13779, 15496, 11, 616, 3290, 318, 13779, 318, 13779],
            [15496, 11, 616, 3290, 318, 13779, 318, 13779, 15496, 11, 616, 3290, 318, 13779, 318, 13779],
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int64,
    )

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
    start_positions = torch.tensor([[0], [0]], dtype=torch.int64)
    data["start_positions"] = start_positions
    end_positions = torch.tensor([[1], [1]], dtype=torch.int64)
    data["end_positions"] = end_positions
    return data


def data_gen_for_token_classification():
    # token classification data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    data["labels"] = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.int64,
    )
    return data


def data_gen_for_sequence_classification():
    # sequence classification data gen
    data = data_gen()
    data["labels"] = torch.tensor([[1], [1]], dtype=torch.int64)
    return data


def date_gen_for_double_heads():
    num_choices = 2
    batch_size = 2
    input_ids = torch.tensor(
        [
            [15496, 11, 616, 3290, 318, 13779, 318, 13779, 15496, 11, 616, 3290, 318, 13779, 318, 13779],
            [15496, 11, 616, 3290, 318, 13779, 318, 13779, 15496, 11, 616, 3290, 318, 13779, 318, 13779],
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.int64,
    )

    mc_labels = torch.zeros(input_ids.shape[0], dtype=torch.int64)
    mc_token_ids = torch.arange(0, num_choices, dtype=torch.int64)
    mc_token_ids = mc_token_ids.expand((batch_size, num_choices))
    multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, num_choices, -1).contiguous()
    multiple_choice_input_mask = attention_mask.unsqueeze(1).expand(-1, num_choices, -1).contiguous()

    inputs = {
        "input_ids": multiple_choice_inputs_ids,
        "mc_token_ids": mc_token_ids,
        "attention_mask": multiple_choice_input_mask,
        "labels": multiple_choice_inputs_ids,
        "mc_labels": mc_labels,
    }
    return inputs


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_gpt2_model = lambda x: torch.nn.functional.mse_loss(
    x["last_hidden_state"], torch.ones_like(x["last_hidden_state"])
)
loss_fn = lambda x: x["loss"]

config = transformers.GPT2Config(
    n_layer=2,
    n_head=4,
    n_embd=128,
    vocab_size=50258,
    attn_pdrop=0,
    embd_pdrop=0,
    resid_pdrop=0,
    summary_first_dropout=0,
    hidden_dropout=0,
    problem_type="single_label_classification",
    pad_token_id=50256,
    tie_word_embeddings=False,
)

config_for_token_classification = copy.deepcopy(config)
config_for_token_classification.num_labels = 2

# register the following models
model_zoo.register(
    name="transformers_gpt",
    model_fn=lambda: transformers.GPT2Model(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_gpt2_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gpt_lm",
    model_fn=lambda: transformers.GPT2LMHeadModel(config),
    data_gen_fn=data_gen_for_lm,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gpt_double_heads",
    model_fn=lambda: transformers.GPT2DoubleHeadsModel(config),
    data_gen_fn=date_gen_for_double_heads,
    output_transform_fn=output_transform_fn,
    loss_fn=lambda x: x.loss + x.mc_loss,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gpt_for_question_answering",
    model_fn=lambda: transformers.GPT2ForQuestionAnswering(config),
    data_gen_fn=data_gen_for_question_answering,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gpt_for_token_classification",
    model_fn=lambda: transformers.GPT2ForTokenClassification(config_for_token_classification),
    data_gen_fn=data_gen_for_token_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_gpt_for_sequence_classification",
    model_fn=lambda: transformers.GPT2ForSequenceClassification(config_for_token_classification),
    data_gen_fn=data_gen_for_sequence_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
