import torch
import transformers
from chatglm2_6b.configuration_chatglm import ChatGLMConfig

from ..registry import ModelAttribute, model_zoo

# ================================
# Register single-sentence ChatGLM
# ================================

config = ChatGLMConfig(num_hidden_layers=4,
                        hidden_size=128,
                        intermediate_size=256,
                        num_attention_heads=4)

def data_gen():

    input_ids = torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)

# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_chatglm_model = lambda x: x.last_hidden_state.mean()
loss_fn = lambda x: x.loss

model_zoo.register(name='transformers_chatglm',
                   model_fn=lambda: transformers.ChatGLMModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_chatglm_model,
                   model_attribute=ModelAttribute(has_control_flow=True))

