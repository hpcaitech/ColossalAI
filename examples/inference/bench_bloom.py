import os
import time

import pytest
import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
MAX_BATCH_SIZE = 32
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 128


def print_perf_stats(latency_set, config, bs, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2    # float16

        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1 / avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1 / avg * num_parameters * num_bytes * bs / 1e12))
        print("Avg Throughput: tokens/s: {}".format((1000 / (avg * 1000)) * bs))


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def bench_bloom(test_config):

    model_path = "/home/lczyh/data3/models/bloom-7b1"
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()
    # To benchmark torch original, uncommment the following line
    # model.to(torch.cuda.current_device())

    # init TPInferEngine and shard original model by shardformer
    # To benchmark torch original, comment out lines of creating, preparing, and sharding by the shardformer
    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                               inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)
    infer_engine.prepare_with_shard_config(shard_config)
    infer_engine.shard_model_by(shardformer)

    # prepare data for generation
    batch_size = MAX_BATCH_SIZE
    input_len = MAX_INPUT_LEN
    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
    input_tokens = {
        "input_ids": torch.randint(10, 1000, (batch_size, input_len)),
        "attention_mask": torch.ones((batch_size, input_len))
    }
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
            print(f" input_tokens[{t}].shape: {input_tokens[t].shape}")

    iters = 10
    times = []
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = infer_engine.generate(input_tokens, generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        infer_engine.cache_manager.free_all()
        out_len = outputs.shape[1]
        print(f" iter {i}: out len {str(out_len)}, generation time {str(end - start)} s")
        times.append((end - start) / (out_len - input_len))

    print_perf_stats(times, model.config, batch_size)


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    bench_bloom()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom():
    spawn(check_bloom, TPSIZE)


if __name__ == "__main__":
    test_bloom()
