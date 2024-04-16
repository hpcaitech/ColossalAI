import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM, BloomConfig, BloomModel, BloomForCausalLM

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.models.bloom import BloomModel, BloomForCausalLM
from colossalai.inference.modeling.policy.bloom import BloomModelInferPolicy
from colossalai.inference.modeling.policy.nopadding_llama import NoPaddingLlamaModelInferPolicy
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

from transformers import AutoTokenizer, AutoModelForCausalLM

def check_llama_model_forward():
    # model_path_or_name = "/home/lixingjian/models/bloom-560m"
    model_path_or_name = "/home/lishenggui/projects/trt/models/Llama-2-7b-hf"
    
    model = LlamaForCausalLM.from_pretrained(model_path_or_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    inference_config = InferenceConfig(
        dtype="fp16",
        max_batch_size=1,
        max_input_len=256,
        max_output_len=256,
        prefill_ratio=1.2,
        block_size=16,
    )

    # Your policy
    policy = NoPaddingLlamaModelInferPolicy()
    engine = InferenceEngine(model, tokenizer, inference_config, model_policy=policy, verbose=True)

    prompt = "Introduce some landmarks in the United Kingdom. "
    # prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions. "
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=128,
        num_beams=1,
        do_sample=False,
    )
    out = engine.generate(prompts=[prompt], generation_config=generation_config)
    print(out)


def check_bloom_model_forward():

    model_path_or_name = "/home/lixingjian/models/bloom-560m"
    
    # model = ChatGLMForConditionalGeneration.from_pretrained(model_path_or_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
    
    model = BloomForCausalLM.from_pretrained(model_path_or_name)#.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    inference_config = InferenceConfig(
        dtype="fp16",
        max_batch_size=1,
        max_input_len=256,
        max_output_len=256,
        prefill_ratio=1.2,
        block_size=16,
    )

    # Your policy
    policy = BloomModelInferPolicy()
    engine = InferenceEngine(model, tokenizer, inference_config, model_policy=policy, verbose=True)
    # engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

    # prompt = "Introduce some landmarks in the United Kingdom. "
    prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=128,
        num_beams=1,
        do_sample=False,
    )
    out = engine.generate(prompts=[prompt], generation_config=generation_config)
    print(out)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_bloom_model_forward()
    # check_llama_model_forward()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()
