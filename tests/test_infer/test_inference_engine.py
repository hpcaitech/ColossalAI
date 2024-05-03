import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import Manager
from transformers import BloomForCausalLM, BloomTokenizerFast, GenerationConfig

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.modeling.policy import NoPaddingBloomModelInferPolicy
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

MODEL_PATH = "/home/lixingjian/models/bloom-560m"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(use_engine=False, prompt_template=None, do_sample=True, policy=None):
    setup_seed(20)
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_PATH)
    model = BloomForCausalLM.from_pretrained(MODEL_PATH).cuda()
    model = model.eval()

    inputs = [
        "Introduce a landmark in China",
    ]

    output_len = 38
    do_sample = do_sample
    top_p = 0.5
    top_k = 50

    if use_engine:
        inference_config = InferenceConfig(
            max_output_len=output_len,
            prompt_template=prompt_template,
            dtype="fp32",
            use_cuda_kernel=True,
            tp_size=dist.get_world_size(),
        )
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True, model_policy=policy)
        assert inference_engine.generation_config.max_new_tokens == output_len
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(do_sample=do_sample, top_p=top_p, top_k=top_k)
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
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")

    if ret:
        ret[rank] = func_to_run(**kwargs)
    else:
        func_to_run(**kwargs)


@parameterize("prompt_template", [None, "llama"])
@parameterize("do_sample", [False])
def test_tp_engine(prompt_template, do_sample):
    kwargs1 = {
        "use_engine": True,
        "prompt_template": prompt_template,
        "do_sample": do_sample,
        "policy": NoPaddingBloomModelInferPolicy(),
    }

    kwargs2 = {"use_engine": False, "prompt_template": prompt_template, "do_sample": do_sample, "policy": None}

    colossal_tp_1_output = run_engine(1, **kwargs1)
    transformer_tp_1_output = run_engine(1, **kwargs2)

    for s1, s3 in zip(colossal_tp_1_output, transformer_tp_1_output):
        assert s1 == s3, f"\nColossalAI TP=1 Output: {s1}\nTransformers Output: {s3}"


# @parameterize("num_layers", [1])
# @parameterize("max_length", [64])
# def test_spec_dec(num_layers, max_length):
#     spawn(run_dist, 1, func_to_run=check_spec_dec, num_layers=num_layers, max_length=max_length)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_inference_engine():
    test_tp_engine()
    # test_spec_dec()


if __name__ == "__main__":
    test_inference_engine()
