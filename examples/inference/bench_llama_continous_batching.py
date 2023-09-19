import argparse
import os
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import LlamaForCausalLM, LlamaTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.inference.tensor_parallel.utils import init_to_get_rotary
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


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
        num_bytes = 2

        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1 / avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1 / avg * num_parameters * num_bytes * bs / 1e12))


def run_llama_test(args):
    llama_model_path = args.path
    max_batch_size = args.batch_size
    max_input_len = args.input_len
    max_output_len = args.output_len
    tokenizer = args.tokenizer
    test_mode = args.test_mode
    test_continous_batching = args.test_continous_batching

    if (tokenizer == None):
        tokenizer = llama_model_path

    tmp_tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
    tmp_tokenizer.pad_token_id = tmp_tokenizer.unk_token_id
    tmp_model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tmp_tokenizer.eos_token_id)

    model_config = tmp_model.config

    generate_kwargs = dict(max_new_tokens=max_output_len, do_sample=False)
    if test_mode == "colossalai" and not test_continous_batching:
        input_tokens = {
            "input_ids": torch.randint(1, 1000, (max_batch_size, max_input_len), device='cuda'),
            "attention_mask": torch.ones((max_batch_size, max_input_len), device='cuda')
        }
    else:
        input_tokens = [np.random.randint(1, 1000, [max_input_len]).tolist() for _ in range(max_batch_size)]

    iters = 10
    times = []

    if test_mode == "colossalai":
        if test_continous_batching:
            print("test_continous_batching: ", test_continous_batching)
            infer_engine = TPInferEngine(
                model=llama_model_path,
                max_output_len=max_output_len,
                use_continous_batching=test_continous_batching,
                tokenizer=tokenizer,
            )
        else:
            shard_config = ShardConfig(enable_tensor_parallelism=True if args.tp_size > 1 else False,
                                       inference_only=True)
            infer_engine = TPInferEngine(
                model=llama_model_path,
                shard_config=shard_config,
                max_batch_size=max_batch_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                use_continous_batching=test_continous_batching,
                tokenizer=tokenizer,
            )
            infer_engine.optimize_model()
    elif test_mode == "vllm":
        infer_engine = LLM(model=llama_model_path, tokenizer=tokenizer)

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        if test_mode == "colossalai":
            outputs = infer_engine.generate(prompt_token_ids=input_tokens, **generate_kwargs)
        elif test_mode == "vllm":
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_output_len)
            outputs = infer_engine.generate(prompt_token_ids=input_tokens, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end = time.time()
        if test_mode == "colossalai" and not test_continous_batching:
            out_len = outputs.shape[1]
        else:
            # out_len = len(outputs[0].outputs[0].token_ids)
            out_len = 1024 + 128
        print("generation time {} s".format(str(end - start)))
        times.append((end - start) / (out_len - max_input_len))

    print("outputs, ", len(outputs))
    print_perf_stats(times, model_config, max_batch_size)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         torch.cuda.synchronize()
    #         if test_mode == "colossalai":
    #             outputs = infer_engine.generate(prompt_token_ids=input_tokens, **generate_kwargs)
    #         elif test_mode == "vllm":
    #             sampling_params = SamplingParams(temperature=0.0, max_tokens=max_output_len)
    #             outputs = infer_engine.generate(prompt_token_ids=input_tokens, sampling_params=sampling_params)
    #         torch.cuda.synchronize()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def check_llama(rank, world_size, port, args):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama(args):
    spawn(check_llama, args.tp_size, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Model path', required=True)
    parser.add_argument('-t', '--tokenizer', type=str, default=None, help='Tokenizer path')
    parser.add_argument('-tp', '--tp_size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Maximum batch size')
    parser.add_argument('--input_len', type=int, default=1024, help='Maximum input length')
    parser.add_argument('--output_len', type=int, default=128, help='Maximum output length')
    parser.add_argument('--test_mode', type=str, default="colossalai", help='test colossalai or vllm')
    parser.add_argument('--test_continous_batching',
                        type=bool,
                        default=False,
                        help='whether to test continous_batching')

    args = parser.parse_args()

    test_llama(args)
