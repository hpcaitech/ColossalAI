import argparse
import os
import time

import torch
from _utils import print_perf_stats
from transformers import BloomForCausalLM, BloomTokenizerFast

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def bench_bloom(args):
    model_path = args.path
    max_batch_size = args.batch_size
    max_input_len = args.input_len
    max_output_len = args.output_len

    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    # init TPInferEngine and shard the original model
    # To benchmark torch original, comment out the line of optimizing model
    shard_config = ShardConfig(enable_tensor_parallelism=True if args.tp_size > 1 else False, inference_only=True)
    infer_engine = TPInferEngine(model, shard_config, max_batch_size, max_input_len, max_output_len)

    # prepare data for generation
    generate_kwargs = dict(max_new_tokens=max_output_len, do_sample=False)
    input_tokens = {
        "input_ids": torch.randint(10, 1000, (max_batch_size, max_input_len)),
        "attention_mask": torch.ones((max_batch_size, max_input_len)),
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
        outputs = infer_engine.generate(input_tokens, **generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print(f" iter {i}: out len {str(out_len)}, generation time {str(end - start)} s")
        times.append((end - start) / (out_len - max_input_len))

    print_perf_stats(times, model.config, max_batch_size)


def check_bloom(rank, world_size, port, args):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    bench_bloom(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom(args):
    spawn(check_bloom, args.tp_size, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Model path", required=True)
    parser.add_argument("-tp", "--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Maximum batch size")
    parser.add_argument("--input_len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=128, help="Maximum output length")

    args = parser.parse_args()

    test_bloom(args)
