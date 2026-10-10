import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence ALBERT
# ===============================
BATCH_SIZE = 2
SEQ_LENGTH = 16


def data_gen_fn():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def data_gen_for_pretrain():
    inputs = data_gen_fn()
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["sentence_order_label"] = torch.zeros(BATCH_SIZE, dtype=torch.int64)
    return inputs


output_transform_fn = lambda x: x

config = transformers.AlbertConfig(
    embedding_size=128, hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256
)

model_zoo.register(
    name="transformers_albert",
    model_fn=lambda: transformers.AlbertModel(config, add_pooling_layer=False),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_albert_for_pretraining",
    model_fn=lambda: transformers.AlbertForPreTraining(config),
    data_gen_fn=data_gen_for_pretrain,
    output_transform_fn=lambda x: dict(loss=x.loss),
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_albert_for_masked_lm",
    model_fn=lambda: transformers.AlbertForMaskedLM(config),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_albert_for_sequence_classification",
    model_fn=lambda: transformers.AlbertForSequenceClassification(config),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_albert_for_token_classification",
    model_fn=lambda: transformers.AlbertForTokenClassification(config),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)

# ===============================
# Register multi-sentence ALBERT
# ===============================


def data_gen_for_qa():
    input_ids = torch.arange(SEQ_LENGTH, dtype=torch.int64).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.cat(
        [torch.zeros(1, SEQ_LENGTH // 2), torch.ones(1, SEQ_LENGTH - SEQ_LENGTH // 2)], dim=1
    ).to(torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


def data_gen_for_mcq():
    input_ids = torch.arange(2 * SEQ_LENGTH, dtype=torch.int64).reshape(1, 2, SEQ_LENGTH)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[:, :, SEQ_LENGTH // 2 :] = 1
    return dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


model_zoo.register(
    name="transformers_albert_for_question_answering",
    model_fn=lambda: transformers.AlbertForQuestionAnswering(config),
    data_gen_fn=data_gen_for_qa,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="transformers_albert_for_multiple_choice",
    model_fn=lambda: transformers.AlbertForMultipleChoice(config),
    data_gen_fn=data_gen_for_mcq,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
