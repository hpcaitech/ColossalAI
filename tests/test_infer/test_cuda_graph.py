import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import rerun_if_address_is_in_use, spawn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(use_cuda_graph=False, batch_size=32):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=50000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=16
            )
        )
        .cuda()
        .half()
    )
    model = model.eval()

    prompts_token_ids = []
    for i in range(batch_size):
        prompts_token_ids.append(
            np.random.randint(low=0, high=100, size=random.randint(1, max(1024 // batch_size, 32))).tolist()
        )

    input_len = 1024
    output_len = 128
    do_sample = False
    top_p = 0.5
    top_k = 50

    if use_cuda_graph:
        inference_config = InferenceConfig(
            max_batch_size=batch_size,
            max_input_len=input_len,
            max_output_len=output_len,
            use_cuda_kernel=False,
            use_cuda_graph=True,
            block_size=16,
        )
    else:
        inference_config = InferenceConfig(
            max_batch_size=batch_size,
            max_input_len=input_len,
            max_output_len=output_len,
            use_cuda_kernel=False,
            use_cuda_graph=False,
            block_size=16,
        )

    inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
    assert inference_engine.generation_config.max_new_tokens == output_len
    generation_config = GenerationConfig(do_sample=do_sample, top_p=top_p, top_k=top_k)
    outputs = inference_engine.generate(prompts_token_ids=prompts_token_ids, generation_config=generation_config)

    return outputs


def check_output_consistency(batch_size):
    cuda_graph_output = check_inference_engine(use_cuda_graph=True, batch_size=batch_size)
    naive_model_output = check_inference_engine(use_cuda_graph=False, batch_size=batch_size)

    for s1, s2 in zip(cuda_graph_output, naive_model_output):
        assert s1 == s2, f"\nCUDA Graph Output: {s1}\nOrigin Output: {s2}"


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_output_consistency(32)
    check_output_consistency(64)
    check_output_consistency(128)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
def test_cuda_graph_infer():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_cuda_graph_infer()
