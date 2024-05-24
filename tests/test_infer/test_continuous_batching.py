import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import InferenceConfig
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


@parameterize("n_multiple", [10])
@parameterize("max_batch_size", [8])
@parameterize("max_input_len", [128])
@parameterize("max_output_len", [128])
def check_inference_engine(n_multiple, max_batch_size, max_input_len, max_output_len):
    setup_seed(20)

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = LlamaForCausalLM(LlamaConfig(num_hidden_layers=2)).cuda()
    model = model.eval()

    inputs_token_ids = generate_inputs(
        n_multiple * max_batch_size, min_length=max_input_len // 2, max_length=max_input_len
    )
    inference_config = InferenceConfig(
        max_batch_size=max_batch_size, max_input_len=max_input_len, max_output_len=max_output_len
    )
    inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
    assert inference_engine.generation_config.max_new_tokens == max_output_len

    inference_engine.add_request(prompts_token_ids=inputs_token_ids)
    assert inference_engine.request_handler._has_waiting()

    outputs = inference_engine.generate()
    assert not inference_engine.request_handler._has_waiting()
    assert len(outputs) == n_multiple * max_batch_size


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_inference_engine()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_continuous_batching():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_continuous_batching()
