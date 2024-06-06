import random

import numpy as np
import torch
from torch.multiprocessing import Manager
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import rerun_if_address_is_in_use, spawn


def data_gen(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=torch.cuda.current_device())
    return input_ids


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_streamingllm():
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=50000,
            hidden_size=512,
            intermediate_size=1536,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=16,
        )
    ).cuda()
    model = model.eval()

    input_token_ids = data_gen(1, 4)

    output_len = 128

    inference_config = InferenceConfig(
        max_batch_size=1,
        max_output_len=output_len,
        dtype="fp32",
        use_cuda_kernel=True,
        enable_streamingllm=True,
        start_token_size=4,
        generated_token_size=32,
    )

    inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
    assert inference_engine.generation_config.max_new_tokens == output_len
    inference_engine.add_request(prompts_token_ids=input_token_ids)
    assert inference_engine.request_handler._has_waiting()

    assert inference_config.start_token_size == inference_config.block_size

    request_handler = inference_engine.request_handler
    running_bb = request_handler.running_bb

    for _ in range(12):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, -1, -1, -1]
    assert running_bb.seq_lengths[0].item() == 16

    for _ in range(16):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, 1, -1, -1]
    assert running_bb.seq_lengths[0].item() == 32

    for _ in range(16):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, 1, 2, -1]
    assert running_bb.seq_lengths[0].item() == 48

    for _ in range(16):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, 2, 3, -1]
    assert running_bb.seq_lengths[0].item() == 48

    for _ in range(1):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, 2, 3, 1]
    assert running_bb.seq_lengths[0].item() == 49

    for _ in range(15):
        inference_engine.step()

    assert running_bb.block_tables[0].tolist() == [0, 3, 1, -1]
    assert running_bb.seq_lengths[0].item() == 48


def run_dist(rank, world_size, port, func_to_run, ret=None, **kwargs):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    if ret:
        ret[rank] = func_to_run(**kwargs)
    else:
        func_to_run(**kwargs)


@rerun_if_address_is_in_use()
def test_engine():
    manager = Manager()
    result_list = manager.list([-1] * 1)  # Create a shared list

    spawn(run_dist, 1, func_to_run=check_streamingllm, ret=result_list)
    return result_list[0]


if __name__ == "__main__":
    test_engine()
