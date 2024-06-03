import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence OPT
# ===============================
BATCH_SIZE = 2
SEQ_LENGTH = 16


def data_gen():
    input_ids = torch.Tensor([[1, 15043, 29892, 590, 11203, 338, 274, 1082]]).long()
    attention_mask = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1]]).long()
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_causal_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    labels = data["input_ids"].clone()
    data["labels"] = labels
    return data


def data_gen_for_sequence_classification():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data["input_ids"].clone()
    data["labels"] = torch.tensor([1])
    return data


def data_gen_for_question_answering():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data["start_positions"] = torch.tensor([0])
    data["end_positions"] = torch.tensor([1])
    return data


output_transform_fn = lambda x: x
loss_fn_for_opt_model = lambda x: torch.nn.functional.mse_loss(
    x["last_hidden_state"], torch.ones_like(x["last_hidden_state"])
)
loss_fn_for_lm = lambda x: x["loss"]
config = transformers.OPTConfig(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    dropout=0,
)

# register the following models
# transformers.OPTModel,
# transformers.OPTForCausalLM,
model_zoo.register(
    name="transformers_opt",
    model_fn=lambda: transformers.OPTModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_opt_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_opt_for_causal_lm",
    model_fn=lambda: transformers.OPTForCausalLM(config),
    data_gen_fn=data_gen_for_causal_lm,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_lm,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_opt_for_question_answering",
    model_fn=lambda: transformers.OPTForQuestionAnswering(config),
    data_gen_fn=data_gen_for_question_answering,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_lm,
    model_attribute=ModelAttribute(has_control_flow=True),
)

# TODO The loss and gradient check in the test are failing, to be fixed.
# model_zoo.register(name='transformers_opt_for_sequence_classification',
#                    model_fn=lambda: transformers.OPTForSequenceClassification(config),
#                    data_gen_fn=data_gen_for_sequence_classification,
#                    output_transform_fn=output_transform_fn,
#                    loss_fn=loss_fn_for_lm,
#                    model_attribute=ModelAttribute(has_control_flow=True))
