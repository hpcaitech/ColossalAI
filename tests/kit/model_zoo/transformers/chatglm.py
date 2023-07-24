import torch
import transformers

from ..registry import ModelAttribute, model_zoo
from .chatglm2_6b.configuration_chatglm import ChatGLMConfig
from .chatglm2_6b.modeling_chatglm import ChatGLMModel

# ================================
# Register single-sentence ChatGLM
# ================================


def data_gen():
    input_ids = torch.tensor([[5941, 15, 2670, 3543, 632, 2075]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    return dict(input_ids=input_ids, attention_mask=attention_mask)


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_chatglm_model = lambda x: x.last_hidden_state.mean()
loss_fn = lambda x: x.loss
config = ChatGLMConfig(num_layers=1,
                       padded_vocab_size=65024,
                       hidden_size=64,
                       num_attention_heads=8,
                       rmsnorm=False,
                       original_rope=True,
                       use_cache=True)

model_zoo.register(name='transformers_chatglm',
                   model_fn=lambda: ChatGLMModel(config, empty_init=False),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_chatglm_model,
                   model_attribute=ModelAttribute(has_control_flow=True))
