import argparse
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, GenerationConfig

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

GIGABYTE = 1024**3
MEGABYTE = 1024 * 1024

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
}


def data_gen(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_accelerator().get_current_device())
    return input_ids


def print_details_info(model_config, args, whole_end2end):
    msg: str = ""

    if dist.get_rank() == 0:
        msg += "-------Perf Summary-------\n"
        whole_avg_latency = whole_end2end / (args.output_len * args.batch_size)
        num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
        num_parameters = num_layers * model_config.hidden_size * model_config.hidden_size * 12 / args.pp_size
        if args.dtype in ["fp16", "bf16"]:
            num_bytes = 2
        else:
            num_bytes = 4

        msg += f"Whole batch end2end time: {whole_end2end * 1000:.2f} ms\n"
        msg += f"Whole batch per token latency: {whole_avg_latency * 1000:.2f} ms\n"
        msg += f"Throughput: {args.output_len * args.batch_size / whole_end2end:.2f} tokens/s\n"
        msg += f"Flops: {num_parameters * num_bytes / whole_avg_latency / 1e12:.2f} TFLOPS\n"

    if torch.cuda.is_available():
        msg += f"-------Memory Summary Device:{get_accelerator().current_device()}-------\n"
        msg += f"Max memory allocated: {get_accelerator().max_memory_allocated() / GIGABYTE:.2f} GB\n"
        msg += f"Max memory reserved: {get_accelerator().max_memory_reserved() / GIGABYTE:.2f} GB\n"

    print(msg)


def benchmark_inference(args):
    with torch.no_grad():
        config = CONFIG_MAP[args.model]
        config.pad_token_id = config.eos_token_id
        model = transformers.LlamaForCausalLM(config).cuda()
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

        if args.dtype == "fp16":
            model = model.half()
        elif args.dtype == "bf16":
            model = model.to(torch.bfloat16)

        if args.continous_batching:
            mbsz = args.mbsz
        else:
            mbsz = args.batch_size
        if args.mode == "caiinference":
            inference_config = InferenceConfig(
                dtype=args.dtype,
                micro_batch_size=args.mb_size,
                max_batch_size=mbsz,
                max_input_len=args.seq_len,
                max_output_len=args.output_len,
                prefill_ratio=1.2,
            )
            engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        else:
            engine = model

        data = data_gen(mbsz, args.seq_len)
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.output_len,
        )

        N_WARMUP_STEPS = 2

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
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_log_" + args.mode),
            )
            if args.profile
            else nullcontext()
        )

        with ctx:
            for _ in range(N_WARMUP_STEPS):
                if args.mode == "caiinference":
                    engine.generate(prompts_token_ids=data, generation_config=generation_config)
                else:
                    engine.generate(data, generation_config=generation_config)
                if args.profile:
                    ctx.step()

            if args.nsys:
                torch.cuda.cudart().cudaProfilerStart()

            torch.cuda.synchronize()

            whole_end2end = time.perf_counter()
            if args.mode == "caiinference":
                for _ in range(args.batch_size // mbsz):
                    engine.generate(prompts_token_ids=data, generation_config=generation_config)
            else:
                for _ in range(args.batch_size // mbsz):
                    engine.generate(data, generation_config=generation_config)
            whole_end2end = time.perf_counter() - whole_end2end
            if args.nsys:
                torch.cuda.cudart().cudaProfilerStop()
            if args.profile:
                ctx.step()

    print_details_info(model.config, args, whole_end2end)


def hybrid_inference(rank, world_size, port, args):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    benchmark_inference(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def benchmark(args):
    spawn(hybrid_inference, nprocs=args.tp_size * args.pp_size, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="toy",
        help="the size of model",
        choices=["toy", "llama-7b", "llama-13b", "llama2-7b", "llama2-13b"],
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--mbsz", type=int, default=8, help="batch size for one step")
    parser.add_argument("-s", "--seq_len", type=int, default=8, help="input sequence length")
    parser.add_argument("--mb_size", type=int, default=1, help="micro_batch_size")
    parser.add_argument("--pp_size", type=int, default=1, help="pipeline size")
    parser.add_argument("--tp_size", type=int, default=1, help="pipeline size")
    parser.add_argument("--output_len", type=int, default=128, help="Output length")
    parser.add_argument("--dtype", type=str, default="fp16", help="data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true", help="enable torch profiler")
    parser.add_argument("--nsys", default=False, action="store_true", help="enable nsys profiler")
    parser.add_argument(
        "--mode",
        default="caiinference",
        choices=["caiinference", "transformers"],
        help="decide which inference framework to run",
    )
    parser.add_argument(
        "-cb", "--continous_batching", default=False, action="store_true", help="enable continous batching"
    )
    args = parser.parse_args()
    benchmark(args)
