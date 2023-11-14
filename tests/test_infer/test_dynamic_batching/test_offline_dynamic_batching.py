from dataclasses import dataclass

import pytest
import torch
from packaging import version
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import colossalai
from colossalai.inference.dynamic_batching.io_struct import Req
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.inference.manager import start_dynamic_batching
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

TP_SIZE = 1
MAX_BATCH_SIZE = 2
MAX_INPUT_LEN = 5
MAX_OUTPUT_LEN = 16
CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


@dataclass
class args:
    max_total_token_num: int
    batch_max_tokens: int
    model: str
    eos_id: int
    disable_log_stats: bool
    log_stats_interval: int


def run():
    arg = args(
        max_total_token_num=42,
        model="llama",
        batch_max_tokens=42,
        eos_id=0,
        disable_log_stats=False,
        log_stats_interval=10,
    )
    sampling_params = SamplingParams()

    req1 = Req(0, [0, 0, 10, 6, 8], sampling_params)
    req2 = Req(1, [10, 10, 10, 10, 10], sampling_params)
    req3 = Req(2, [0, 0, 10, 10, 10], sampling_params)
    req4 = Req(3, [0, 0, 10, 10, 10], sampling_params)

    waiting_list = []
    waiting_list.append(req1)
    waiting_list.append(req2)
    waiting_list.append(req3)
    waiting_list.append(req4)

    llama_config = LlamaConfig(num_hidden_layers=2, bos_token_id=0, eos_token_id=1, vocab_size=30000, hidden_size=1024)
    model = LlamaForCausalLM(llama_config)
    model = model.half()

    shard_config = ShardConfig(
        enable_tensor_parallelism=True if TP_SIZE > 1 else False, extra_kwargs={"inference_only": True}
    )

    infer_engine = TPInferEngine(model, shard_config, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    batch_manager = start_dynamic_batching(arg, tp_engine=infer_engine, waiting_req_list=waiting_list)

    ans_gen = batch_manager.generate(request_id=5, prompts="hello", sampling_params=sampling_params)
    for result in ans_gen:
        assert result is not None


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
