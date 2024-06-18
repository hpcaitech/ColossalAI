import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.rpc_engine import RPCInferenceEngine
from colossalai.inference.modeling.policy import NoPaddingLlamaModelInferPolicy
from colossalai.testing import parameterize, rerun_if_address_is_in_use


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(tp_size, use_engine=False, prompt_template=None, do_sample=True, policy=None):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = "meta-llama/Llama-2-7b-hf"  # remote mode path
    inputs = [
        "介绍一下今天的北京,比如故宫，天安门，长城或者其他的一些景点,",
        "介绍一下武汉,",
    ]

    output_len = 38
    top_p = 0.5
    top_k = 50

    if use_engine:
        inference_config = InferenceConfig(
            max_output_len=output_len,
            prompt_template=prompt_template,
            dtype="fp32",
            use_cuda_kernel=True,
            tp_size=tp_size,
        )
        inference_engine = RPCInferenceEngine(model, tokenizer, inference_config, verbose=True, model_policy=policy)
        assert inference_engine.generation_config.max_new_tokens == output_len
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(
            max_new_tokens=output_len, do_sample=do_sample, dtype="fp32", top_p=top_p, top_k=top_k
        )
        outputs = inference_engine.generate(generation_config=generation_config)
    else:
        if prompt_template:
            # apply prompt template
            inputs = [_DEFAULT_PROMPT_TEMPLATES[prompt_template].format(input_text=input_text) for input_text in inputs]
        model = AutoModelForCausalLM.from_pretrained(model).cuda()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.cuda()
        generation_config = GenerationConfig(
            do_sample=do_sample,
            dtype="fp32",
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=output_len,
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs


def run_engine(tp_size, **kwargs):
    return check_inference_engine(tp_size=tp_size, **kwargs)


# TODO: fix the test
@pytest.mark.skip("model is too large")
@pytest.mark.largedist
@parameterize("prompt_template", [None, "llama"])
@parameterize("do_sample", [False])
@rerun_if_address_is_in_use()
def test_tp_engine(prompt_template, do_sample):
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method("spawn")
    kwargs1 = {
        "use_engine": True,
        "prompt_template": prompt_template,
        "do_sample": do_sample,
        "policy": NoPaddingLlamaModelInferPolicy(),
    }

    kwargs2 = {"use_engine": False, "prompt_template": prompt_template, "do_sample": do_sample, "policy": None}

    colossal_tp_1_output = run_engine(1, **kwargs1)
    colossal_tp_2_output = run_engine(2, **kwargs1)
    transformer_tp_1_output = run_engine(1, **kwargs2)

    for s1, s2, s3 in zip(colossal_tp_1_output, colossal_tp_2_output, transformer_tp_1_output):
        assert s1 == s3, f"\nColossalAI TP=1 Output: {s1}\nTransformers Output: {s3}"
        assert s1 == s2, f"\nColossalAI TP=1 Output: {s1}\nColossalAI TP=2 Output: {s2}"


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # this code will not be ok for settings to fork to subprocess
    test_tp_engine()
