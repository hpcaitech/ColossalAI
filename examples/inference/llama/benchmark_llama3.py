import argparse
import time
from contextlib import nullcontext

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.cluster import DistCoordinator
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

GIGABYTE = 1024**3
MEGABYTE = 1024**2
N_WARMUP_STEPS = 2

TORCH_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


CONFIG_MAP = {
    "toy": transformers.LlamaConfig(num_hidden_layers=4),
    "llama-7b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        max_position_embeddings=2048,
    ),
    "llama-13b": transformers.LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        max_position_embeddings=2048,
    ),
    "llama2-7b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
    ),
    "llama2-13b": transformers.LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        max_position_embeddings=4096,
    ),
    "llama3-8b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
    ),
    "llama3-70b": transformers.LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
        max_position_embeddings=8192,
    ),
}


def data_gen(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_accelerator().get_current_device())
    return input_ids.tolist()


def print_details_info(model_config, whole_end2end, total_token_num, dtype, coordinator=None):
    if coordinator is None:
        coordinator = DistCoordinator()
    msg = "-------Perf Summary-------\n"
    whole_avg_latency = whole_end2end / (total_token_num)
    num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
    num_parameters = num_layers * model_config.hidden_size * model_config.hidden_size * 12
    if dtype in ["fp16", "bf16"]:
        num_bytes = 2
    elif dtype == "fp32":
        num_bytes = 4
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    msg += f"Whole batch end2end time: {whole_end2end * 1000:.2f} ms\n"
    msg += f"Whole batch per token latency: {whole_avg_latency * 1000:.2f} ms\n"
    msg += f"Throughput: {total_token_num / whole_end2end:.2f} tokens/s\n"
    msg += f"Flops: {num_parameters * num_bytes / whole_avg_latency / 1e12:.2f} TFLOPS\n"
    if torch.cuda.is_available():
        msg += f"-------Memory Summary Device:{get_accelerator().current_device()}-------\n"
        msg += f"Max memory allocated: {get_accelerator().max_memory_allocated() / GIGABYTE:.2f} GB\n"
        msg += f"Max memory reserved: {get_accelerator().max_memory_reserved() / GIGABYTE:.2f} GB\n"

    coordinator.print_on_master(msg)


def benchmark_inference(args):
    coordinator = DistCoordinator()

    torch_dtype = TORCH_DTYPE_MAP.get(args.dtype, None)
    config = CONFIG_MAP[args.model]
    config.torch_dtype = torch_dtype
    config.pad_token_id = config.eos_token_id

    if args.model_path is not None:
        model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        # Random weights
        model = transformers.LlamaForCausalLM(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    if args.dtype == "fp16":
        model = model.half()
    elif args.dtype == "bf16":
        model = model.to(torch.bfloat16)

    inference_config = InferenceConfig(
        dtype=args.dtype,
        max_batch_size=args.batch_size,
        max_input_len=args.max_seq_len,
        max_output_len=args.max_output_len,
        prefill_ratio=1.2,
        block_size=32,
        tp_size=args.tp_size,
        use_cuda_kernel=True,
    )
    engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

    data = data_gen(args.batch_size, args.max_seq_len)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_seq_len + args.max_output_len,
        # max_new_tokens=args.max_output_len,
    )
    coordinator.print_on_master(f"Generation Config: \n{generation_config.to_dict()}")

    ctx = (
        torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=N_WARMUP_STEPS, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./tb_log_{args.batch_size}_{args.max_seq_len}_{args.max_output_len}"
            ),
        )
        if args.profile
        else nullcontext()
    )
    with ctx:
        for _ in range(N_WARMUP_STEPS):
            engine.generate(prompts_token_ids=data, generation_config=generation_config)
            if args.profile:
                ctx.step()
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStart()

        torch.cuda.synchronize()
        whole_end2end = time.perf_counter()
        output, output_tokens_list = engine.generate(
            prompts_token_ids=data, generation_config=generation_config, return_token_ids=True
        )
        torch.cuda.synchronize()
        whole_end2end = time.perf_counter() - whole_end2end

        total_token_num = sum([len(output_tokens) for output_tokens in output_tokens_list])
        coordinator.print_on_master(f"total_token_num: {total_token_num}")
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStop()
        if args.profile:
            ctx.step()

    print_details_info(model.config, whole_end2end, total_token_num, args.dtype, coordinator=coordinator)


def inference(rank, world_size, port, args):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    benchmark_inference(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def benchmark(args):
    spawn(inference, nprocs=args.tp_size, args=args)


# python benchmark_llama3.py -m llama3-8b -b 16 -s 256 -o 256
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="llama3-8b",
        help="The version of Llama model",
        choices=["toy", "llama-7b", "llama-13b", "llama2-7b", "llama2-13b", "llama3-8b", "llama3-70b"],
    )
    parser.add_argument("-p", "--model_path", type=str, default=None, help="The pretrained weights path")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-s", "--max_seq_len", type=int, default=8, help="input sequence length")
    parser.add_argument("-o", "--max_output_len", type=int, default=128, help="Output length")
    parser.add_argument("-t", "--tp_size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("-d", "--dtype", type=str, default="fp16", help="Data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--profile", default=False, action="store_true", help="enable torch profiler")
    parser.add_argument("--nsys", default=False, action="store_true", help="enable nsys profiler")

    args = parser.parse_args()

    benchmark(args)
