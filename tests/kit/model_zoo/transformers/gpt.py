import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence GPT
# ===============================
BATCH_SIZE = 1    # it can only be 1 as GPT cannot handle batch sizes > 1 if no padding token is defined.
SEQ_LENGTH = 16


def data_gen():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def seq_classification_data_gen():
    # batch sizes should be 1 if no padding token is defined.
    input_ids = torch.zeros((1, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((1, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((1, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


output_transform_fn = lambda x: x

config = transformers.GPT2Config(n_position=64, n_layer=2, n_head=4)

# register the following models
model_zoo.register(name='transformers_gpt',
                   model_fn=lambda: transformers.GPT2Model(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_lm',
                   model_fn=lambda: transformers.GPT2LMHeadModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_double_heads',
                   model_fn=lambda: transformers.GPT2DoubleHeadsModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_for_token_classification',
                   model_fn=lambda: transformers.GPT2ForTokenClassification(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_for_sequence_classification',
                   model_fn=lambda: transformers.GPT2ForSequenceClassification(config),
                   data_gen_fn=seq_classification_data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
