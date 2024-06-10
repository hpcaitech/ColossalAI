import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import Manager
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.modeling.policy import NoPaddingBaichuanModelInferPolicy
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

BAICHUAN_MODEL_NAME_OR_PATH = "baichuan-inc/Baichuan2-13B-Base"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(use_engine=False, do_sample=False, use_cuda_kernel=False, prompt_template=None, policy=None):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained(BAICHUAN_MODEL_NAME_OR_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BAICHUAN_MODEL_NAME_OR_PATH, trust_remote_code=True).half().cuda()
    model = model.eval()

    inputs = [
        "介绍一下今天的北京,比如故宫，天安门，长城或者其他的一些景点,",
    ]

    output_len = 38

    if do_sample:
        top_p = 0.5
        top_k = 50
    else:
        top_p = None
        top_k = None

    if use_engine:
        inference_config = InferenceConfig(
            max_output_len=output_len,
            prompt_template=prompt_template,
            use_cuda_kernel=use_cuda_kernel,
            tp_size=dist.get_world_size(),
        )
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True, model_policy=policy)
        assert inference_engine.generation_config.max_new_tokens == output_len
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(do_sample=do_sample, top_p=top_p, top_k=top_k, max_new_tokens=output_len)
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
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=output_len,
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs


def run_engine(world_size, **kwargs):
    manager = Manager()
    result_list = manager.list([-1] * world_size)  # Create a shared list

    spawn(run_dist, world_size, func_to_run=check_inference_engine, ret=result_list, **kwargs)
    return result_list[0]


def run_dist(rank, world_size, port, func_to_run, ret=None, **kwargs):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    if ret:
        ret[rank] = func_to_run(**kwargs)
    else:
        func_to_run(**kwargs)


# NOTE(caidi) If do_sample is set to True or use_cuda_kernel is set to False, the inference result will be different from that of the transformer.
@parameterize("prompt_template", [None, "baichuan"])
@parameterize("do_sample", [False])
@parameterize("use_cuda_kernel", [True])
def check_tp_engine(prompt_template, do_sample, use_cuda_kernel):
    kwargs1 = {
        "use_engine": True,
        "prompt_template": prompt_template,
        "do_sample": do_sample,
        "policy": NoPaddingBaichuanModelInferPolicy(),
        "use_cuda_kernel": use_cuda_kernel,
    }

    kwargs2 = {
        "use_engine": False,
        "prompt_template": prompt_template,
        "do_sample": do_sample,
        "policy": None,
        "use_cuda_kernel": use_cuda_kernel,
    }

    colossal_tp_1_output = run_engine(1, **kwargs1)
    colossal_tp_2_output = run_engine(2, **kwargs1)
    transformer_tp_1_output = run_engine(1, **kwargs2)

    for s1, s2, s3 in zip(colossal_tp_1_output, colossal_tp_2_output, transformer_tp_1_output):
        assert s1 == s3, f"\nColossalAI TP=1 Output: {s1}\nTransformers Output: {s3}"
        assert s1 == s2, f"\nColossalAI TP=1 Output: {s1}\nColossalAI TP=2 Output: {s2}"


@pytest.mark.skipif(
    not os.path.exists(BAICHUAN_MODEL_NAME_OR_PATH),
    reason="There is no local model address included, please replace this address with a valid one.",
)
@pytest.mark.largedist
@rerun_if_address_is_in_use()
def test_inference_engine():
    check_tp_engine()


if __name__ == "__main__":
    test_inference_engine()
