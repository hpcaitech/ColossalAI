import logging

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.nn_modules.qlinear import GeneralQuantLinear
from torch import distributed as dist
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline

from colossalai.gptq import CaiQuantLinear
from colossalai.gptq.gptq_tp import replace_autogptq_linear

logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")
dist.init_process_group(backend="nccl")
pretrained_model_dir = "/data/scratch/llama-7b-hf"
# quantized_model_dir = "llama-7b-with-act-4bit"
quantized_model_dir = "/home/lcxk/data3/test_gptq_llama/llama-7b-no-act-4bit"
rank = dist.get_rank()
world_size = dist.get_world_size()
# rank = 1
# world_size=2
torch.cuda.set_device(rank)
print("world size {0} rank {1} deivce {2}".format(world_size, rank, torch.cuda.current_device()))
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.")
]

# quantize_config = BaseQuantizeConfig(
#     bits=4,    # quantize model to 4-bit
#     group_size=128,    # it is recommended to set the value to 128
#     desc_act=False,    # set to False can significantly speed up inference but the perplexity may slightly bad
# )

# # load un-quantized model, by default, the model will always be loaded into CPU memory
# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
# model.quantize(examples)

# # save quantized model
# model.save_quantized(quantized_model_dir)

# # save quantized model using safetensors
# model.save_quantized(quantized_model_dir, use_safetensors=True)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                                           device=torch.cuda.current_device(),
                                           inject_fused_attention=False)

replace_autogptq_linear(model, tp_size=world_size, tp_rank=rank)

# if rank == 0:
#     print(model.config)
#     print(model)
# download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# inference with model.generate
print("input is:", "auto-gptq is")
print(
    tokenizer.decode(
        model.generate(**tokenizer("auto-gptq is", return_tensors="pt").to(model.device), max_new_tokens=128)[0]))
dist.barrier()
print("input is:", "today is")
print(
    tokenizer.decode(
        model.generate(**tokenizer("today is ", return_tensors="pt").to(model.device), max_new_tokens=128)[0]))
