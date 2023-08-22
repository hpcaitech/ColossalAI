import torch
import torch.distributed as dist
import transformers
from modeling.gpt2 import GPT2PipelineForwards

import colossalai
from colossalai.ppinference import InferenceConfig, PPInferEngine

colossalai.launch_from_torch(config={})


def data_gen():
    input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


inputs = data_gen()
for k, v in inputs.items():
    if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__:
        new_shape = [1] * v.dim()
        new_shape[0] = 8
        inputs[k] = v.to('cuda').repeat(*new_shape)

model = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=8))
infer_config = InferenceConfig(pp_size=4, new_length=8, micro_batch_size=2)
engine = PPInferEngine(infer_config, model, GPT2PipelineForwards.gpt2_lmhead_model_forward)

output = engine.inference([inputs])
if dist.get_rank() == 3:
    print(len(output))
