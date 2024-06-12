import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import Manager
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer

import colossalai
import colossalai.inference.modeling.policy as policy
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

# NOTE: To test a model with the inference engine, you need to provide the path to your
# local pretrained model weights in the MODEL_MAP dictionary
MODEL_MAP = {
    "baichuan": {
        "model": AutoModelForCausalLM,
        "tokenizer": AutoTokenizer,
        "policy": policy.NoPaddingBaichuanModelInferPolicy,
        "model_name_or_path": "baichuan-inc/Baichuan2-13B-Base",  # provide the path to local model weights
    },
    "llama": {
        "model": LlamaForCausalLM,
        "tokenizer": LlamaTokenizer,
        "policy": policy.NoPaddingLlamaModelInferPolicy,
        "model_name_or_path": "meta-llama/Llama-2-70b-hf",
    },
}

MODELS_TO_TEST = ["llama", "baichuan"]  # Specify the models to test


@parameterize("model", MODELS_TO_TEST)
@parameterize("prompt_template", [None, "model_specific"])
@parameterize("do_sample", [False])
@parameterize("use_cuda_kernel", [True])
@pytest.mark.largedist
@rerun_if_address_is_in_use()
def test_model(model, prompt_template, do_sample, use_cuda_kernel):
    model_path = MODEL_MAP[model]["model_name_or_path"]
    if not os.path.exists(model_path):
        pytest.skip(
            f"There is no local model address included for {model}, please replace this address with a valid one."
        )

    if prompt_template == "model_specific":
        prompt_template = model

    model_config = MODEL_MAP[model]

    kwargs1 = {
        "model": model,
        "use_engine": True,
        "prompt_template": prompt_template,
        "do_sample": do_sample,
        "policy": model_config["policy"](),
        "use_cuda_kernel": use_cuda_kernel,
    }

    kwargs2 = {
        "model": model,
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


def run_engine(world_size, **kwargs):
    manager = Manager()
    result_list = manager.list([-1] * world_size)  # Create a shared list
    spawn(run_dist, world_size, func_to_run=_run_engine, ret=result_list, **kwargs)
    return result_list[0]


def run_dist(rank, world_size, port, func_to_run, ret=None, **kwargs):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    if ret:
        ret[rank] = func_to_run(**kwargs)
    else:
        func_to_run(**kwargs)


def _run_engine(model, use_engine=False, do_sample=False, use_cuda_kernel=False, prompt_template=None, policy=None):
    setup_seed(20)
    model_config = MODEL_MAP[model]
    model_name_or_path = model_config["model_name_or_path"]
    tokenizer = model_config["tokenizer"].from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    model = model_config["model"].from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
    model = model.eval()

    inputs = [
        "Introduce some landmarks in Paris:",
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    test_model()
