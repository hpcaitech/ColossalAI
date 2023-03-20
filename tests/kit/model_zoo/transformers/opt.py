import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence OPT
# ===============================
BATCH_SIZE = 2
SEQ_LENGTH = 16


def data_gen():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


output_transform_fn = lambda x: x

config = transformers.OPTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)

# register the following models
# transformers.OPTModel,
# transformers.OPTForCausalLM,
model_zoo.register(name='transformers_opt',
                   model_fn=lambda: transformers.OPTModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_opt_for_causal_lm',
                   model_fn=lambda: transformers.OPTForCausalLM(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
