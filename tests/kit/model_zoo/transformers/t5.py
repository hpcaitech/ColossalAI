import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence T5
# ===============================


# define data gen function
def data_gen_for_encoder_only():
    # Generated from following code snippet
    #
    # from transformers import T5Config, T5Tokenizer
    # config = T5Config(decoder_start_token_id=0)
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    input_ids = torch.Tensor([[13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1]]).long()
    return dict(input_ids=input_ids)


def data_gen_for_conditional_generation():
    # labels is generated with the following code
    #
    # labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
    data = data_gen_for_encoder_only()
    labels = torch.Tensor([[644, 4598, 229, 19250, 5, 1]]).long()
    data['labels'] = labels
    return data


def data_gen_for_t5_model():
    # decoder_inputs_ids is obtained with the following code
    #
    # decoder_input_ids = model._shift_right(input_ids)
    data = data_gen_for_encoder_only()
    decoder_input_ids = torch.Tensor([[0, 13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5]]).long()
    data['decoder_input_ids'] = decoder_input_ids
    return data


# output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn_for_t5_model = lambda x: x.last_hidden_state.mean()
loss_fn_for_encoder_only = lambda x: x.last_hidden_state.mean()
loss_fn_for_conditional_generation = lambda x: x.loss

# define model config
config = transformers.T5Config(d_model=128, num_layers=2, dropout_rate=0, decoder_start_token_id=0)

# register the following models
# transformers.T5Model,
# transformers.T5ForConditionalGeneration,
# transformers.T5EncoderModel,
model_zoo.register(name='transformers_t5',
                   model_fn=lambda: transformers.T5Model(config),
                   data_gen_fn=data_gen_for_t5_model,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_t5_model,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_t5_for_conditional_generation',
                   model_fn=lambda: transformers.T5ForConditionalGeneration(config),
                   data_gen_fn=data_gen_for_conditional_generation,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_conditional_generation,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_t5_encoder_model',
                   model_fn=lambda: transformers.T5EncoderModel(config),
                   data_gen_fn=data_gen_for_encoder_only,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_encoder_only,
                   model_attribute=ModelAttribute(has_control_flow=True))
