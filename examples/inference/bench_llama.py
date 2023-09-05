import os
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
BATCH_SIZE = 32
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 256


def init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)
    inv_freq = 1.0 / (base**(torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) /
                             self.config.head_dim_))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return


def print_perf_stats(latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2

        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1 / avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1 / avg * num_parameters * num_bytes * BATCH_SIZE / 1e12))


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config):

    llama_model_path = "/data/scratch/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.eos_token_id)
    init_to_get_rotary(model.model, base=10000)
    model = model.half()

    model_config = model.config

    infer_engine = TPInferEngine(model.half(), BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    shard_config = ShardConfig(enable_tensor_parallelism=False, inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine.prepare_with_shard_config(shard_config)
    infer_engine.shard_model_by(shardformer)

    batch_size = 2
    max_new_tokens = 128
    input_len = 1024

    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    input_tokens = {
        "input_ids": torch.randint(1, 1000, (batch_size, input_len), device='cuda'),
        "attention_mask": torch.ones((batch_size, input_len), device='cuda')
    }

    iters = 10
    times = []

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = infer_engine.generate(input_tokens, generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print("generation time {} s".format(str(end - start)))
        times.append((end - start) / (out_len - input_len))
        infer_engine.cache_manager.free_all()

    print("outputs, ", len(outputs))
    outputs = tokenizer.batch_decode(outputs)

    print_perf_stats(times, model_config)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            torch.cuda.synchronize()
            outputs = infer_engine.generate(input_tokens, generate_kwargs)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()
