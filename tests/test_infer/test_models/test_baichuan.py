import os
import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

BAICHUAN_MODEL_NAME_OR_PATH = "baichuan-inc/Baichuan2-7B-Base"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(use_engine=False, prompt_template=None):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained(BAICHUAN_MODEL_NAME_OR_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BAICHUAN_MODEL_NAME_OR_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model = model.eval()

    inputs = [
        "介绍一下今天的北京,比如故宫，天安门，长城或者其他的一些景点,",
    ]

    output_len = 38
    do_sample = False

    if use_engine:
        inference_config = InferenceConfig(
            max_output_len=output_len, prompt_template=prompt_template, dtype="fp32", use_cuda_kernel=True
        )
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        assert inference_engine.generation_config.max_new_tokens == output_len
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(do_sample=do_sample)
        outputs = inference_engine.generate(generation_config=generation_config)
    else:
        if prompt_template:
            # apply prompt template
            inputs = [_DEFAULT_PROMPT_TEMPLATES[prompt_template].format(input_text=input_text) for input_text in inputs]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.cuda()
        generation_config = GenerationConfig(
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=output_len,
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs


@parameterize("prompt_template", [None, "baichuan"])
def check_output_consistency(prompt_template):
    cai_outputs = check_inference_engine(use_engine=True, prompt_template=prompt_template)
    transformer_outputs = check_inference_engine(use_engine=False, prompt_template=prompt_template)

    for s1, s2 in zip(cai_outputs, transformer_outputs):
        assert s1 == s2, f"\nColossalAI Output: {s1}\nTransformers Output: {s2}"

    # clear singleton flash decoding tensors
    FDIntermTensors._instances = {}


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_output_consistency()


@pytest.mark.skipif(
    not os.path.exists(BAICHUAN_MODEL_NAME_OR_PATH),
    reason="There is no local model address included, please replace this address with a valid one.",
)
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()
