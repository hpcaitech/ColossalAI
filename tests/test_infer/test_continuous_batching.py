import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_inputs(num_sequences, min_length, max_length):
    sequences = []
    for _ in range(num_sequences):
        length = torch.randint(low=min_length, high=max_length + 1, size=(1,)).item()
        # generating randomly lengthed sequences
        sequence = torch.randint(10, 30000, size=(length,))
        sequences.append(sequence)
    return sequences


@parameterize(
    "max_batch_size", 8, "max_output_len", 512, "max_input_len", 64, "do_sample", True, "top_p", 0.5, "top_k", 50
)
def check_inference_engine(use_engine=False, prompt_template=None):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").cuda().half()
    model = model.eval()

    inputs_token_ids = generate_inputs(10 * max_batch_size, min_length=10, max_length=max_input_len)

    if use_engine:
        inference_config = InferenceConfig(
            max_batch_size=max_batch_size, max_output_len=max_output_len, prompt_template=prompt_template
        )
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        assert inference_engine.generation_config.max_new_tokens == max_output_len
        inference_engine.add_request(prompts_token_ids=inputs_token_ids)
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
            max_new_tokens=max_output_len,
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    assert len(outputs) == 10 * max_batch_size


@parameterize("prompt_template", [None, "llama"])
def check_continuous_batching(prompt_template):
    check_inference_engine(use_engine=True, prompt_template=prompt_template)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_continuous_batching()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_continuous_batching():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_continuous_batching()
