import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence T5
# ===============================
BATCH_SIZE = 2
SEQ_LENGTH = 16


def data_gen():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    decoder_input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def data_gen_for_encoder_only():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids)


output_transform_fn = lambda x: x

config = transformers.T5Config(d_model=128, num_layers=2)

# register the following models
# transformers.T5Model,
# transformers.T5ForConditionalGeneration,
# transformers.T5EncoderModel,
model_zoo.register(name='transformers_t5',
                   model_fn=lambda: transformers.T5Model(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_t5_for_conditional_generation',
                   model_fn=lambda: transformers.T5ForConditionalGeneration(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_t5_encoder_model',
                   model_fn=lambda: transformers.T5EncoderModel(config),
                   data_gen_fn=data_gen_for_encoder_only,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
