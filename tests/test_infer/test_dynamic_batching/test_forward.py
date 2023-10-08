import argparse

import pytest
import torch
from packaging import version
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.inference.dynamic_batching.io_struct import Req
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.inference.manager import start_dynamic_batching
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.test_infer.test_llama_infer import init_to_get_rotary

TP_SIZE = 2
MAX_BATCH_SIZE = 2
MAX_INPUT_LEN = 5
MAX_OUTPUT_LEN = 16
CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_total_token_num", type=int, default=42, help="max_total_token_num")
    parser.add_argument("-b", "--batch_max_tokens", type=int, default=42, help="max tokens of one batch")
    parser.add_argument("--eos_id", type=int, default=0, help="The end token of a seq")
    parser.add_argument("--disable_log_stats", type=bool, default=False)
    parser.add_argument("--log_stats_interval", type=int, default=10)
    args = parser.parse_args()
    sampling_params = SamplingParams()

    req1 = Req(0, [0, 0, 10, 6, 8], sampling_params)
    req2 = Req(1, [10, 10, 10, 10, 10], sampling_params)
    req3 = Req(2, [10, 10, 10, 10, 10], sampling_params)
    req4 = Req(3, [10, 10, 10, 9, 1], sampling_params)

    waiting_list = []
    waiting_list.append(req1)
    waiting_list.append(req2)
    waiting_list.append(req3)
    waiting_list.append(req4)

    tokenizer = LlamaTokenizer.from_pretrained("/data/scratch/llama-7b-hf")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = LlamaForCausalLM.from_pretrained("/data/scratch/llama-7b-hf", pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    init_to_get_rotary(model.model, base=10000)
    shard_config = ShardConfig(enable_tensor_parallelism=True if TP_SIZE > 1 else False, inference_only=True)

    infer_engine = TPInferEngine(model, shard_config, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    start_dynamic_batching(args=args, tp_engine=infer_engine, waiting_req_list=waiting_list)
    print("done")


def check_dynamic_forward(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_dynamic_batching():
    spawn(check_dynamic_forward, TP_SIZE)


if __name__ == "__main__":
    test_dynamic_batching()
