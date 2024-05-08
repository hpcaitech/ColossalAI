import argparse
import time

import torch
import torch.distributed as dist
import transformers

import colossalai
from colossalai.inference import PPInferEngine
from colossalai.inference.pipeline.policies import LlamaModelInferPolicy

GIGABYTE = 1024**3
MEGABYTE = 1024 * 1024

colossalai.launch_from_torch()


def data_gen(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (1, seq_len), dtype=torch.int32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.int32)
    data = dict(input_ids=input_ids, attention_mask=attention_mask)
    for k, v in data.items():
        if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
            new_shape = [1] * v.dim()
            new_shape[0] = batch_size
            data[k] = v.to("cuda").repeat(*new_shape)
    return data


def print_details_info(timestamps, model_config, args, whole_end2end):
    if dist.get_rank() == 0:
        prefill = []
        encoder = []
        end2end = []
        for timestamp in timestamps:
            prefill.append(timestamp[1] - timestamp[0])
            encoder.append(
                sum(timestamp[i + 1] - timestamp[i] for i in range(1, len(timestamp) - 1)) / (len(timestamp) - 2)
            )
            end2end.append(timestamp[-1] - timestamp[0])
        print(whole_end2end)
        with open(
            f"{args.log_path}/llama-{args.model}{args.dtype}_pp{args.pp_size}_{args.seq_len}_{args.new_length}_bsz{args.batch_size}_mbsz{args.mb_size}.log",
            "w+",
        ) as f:
            mb_avg_end2end = sum(end2end) / len(end2end)
            mb_avg_latency = mb_avg_end2end / (args.new_length * args.mb_size)
            whole_avg_latency = whole_end2end / (args.new_length * args.batch_size)
            num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
            num_parameters = num_layers * model_config.hidden_size * model_config.hidden_size * 12 / args.pp_size
            if args.dtype in ["fp16", "bf16"]:
                num_bytes = 2
            else:
                num_bytes = 4

            f.write(
                f"llama-{args.model}{args.dtype}_pp{args.pp_size}, input_len:{args.seq_len}, output_len:{args.new_length}, bsz:{args.batch_size}, mbsz:{args.mb_size}\n"
            )
            f.write("Average prefill time: {0:8.2f} ms\n".format(sum(prefill) / len(prefill) * 1000))
            f.write("Average encode time: {0:8.2f} ms\n".format(sum(encoder) / len(encoder) * 1000))
            f.write("Average micro batch end2end time: {0:8.2f} ms\n".format(mb_avg_end2end * 1000))
            f.write("Average micro batch Per Token Latency: {0:8.2f} ms\n".format(mb_avg_latency * 1000))
            f.write("Whole batch end2end time: {0:8.2f} ms\n".format(whole_end2end * 1000))
            f.write("Whole batch Per Token Latency: {0:8.2f} ms\n".format(whole_avg_latency * 1000))
            f.write("Throughput: {} tokens/s\n".format((1000 / (whole_avg_latency * 1000))))
            f.write("flops: {0:8.2f} TFlops/s\n".format(1 / whole_avg_latency * num_parameters * num_bytes / 1e12))
            f.write("----------------------------------------------------------\n")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()

        # free memory and the total available memory in bytes
        global_free_memory, total_GPU_memory_occupied = torch.cuda.mem_get_info()
        memory_allocated = torch.cuda.memory_allocated()
        max_memory_allocated = torch.cuda.max_memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        max_memory_reserved = torch.cuda.max_memory_reserved()
        with open(
            f"{args.log_path}/llama-{args.model}{args.dtype}_pp{args.pp_size}_{args.seq_len}_{args.new_length}_bsz{args.batch_size}_mbsz{args.mb_size}.log",
            "a",
        ) as f:
            f.write(
                f"\nCurrently using GPU: {current_device}\n"
                f"free memory : {global_free_memory / GIGABYTE:.4f} GB,\n"
                f"total memory: {total_GPU_memory_occupied / GIGABYTE:.4f} GB,\n"
                f"memory allocated: {memory_allocated / GIGABYTE:.4f} GB,\n"
                f"Max CUDA memory allocated: {max_memory_allocated / GIGABYTE:.4f} GB,\n"
                f"memory reserved/cached: {memory_reserved / GIGABYTE:.4f} GB,\n"
                f"Max CUDA memory reserved/cached: {max_memory_reserved / GIGABYTE:.4f} GB,\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="toy", help="the size of model")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-s", "--seq_len", type=int, default=8, help="sequence length")
    parser.add_argument("--new_length", type=int, default=4, help="new tokens length")
    parser.add_argument("--mb_size", type=int, default=1, help="micro_batch_size")
    parser.add_argument("--pp_size", type=int, default=2, help="pipeline size")
    parser.add_argument("--log_path", type=str, default="./log", help="where to store the benchmark log")
    parser.add_argument("--dtype", type=str, default="fp16", help="data type")
    args = parser.parse_args()

    if args.model == "toy":
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(num_hidden_layers=8))
    elif args.model == "7b":
        model = transformers.LlamaForCausalLM(transformers.AutoConfig.from_pretrained("decapoda-research/llama-7b-hf"))
    elif args.model == "13b":
        model = transformers.LlamaForCausalLM(transformers.AutoConfig.from_pretrained("decapoda-research/llama-13b-hf"))
    else:
        raise NotImplementedError

    engine = PPInferEngine(
        pp_size=args.pp_size,
        dtype=args.dtype,
        micro_batch_size=args.mb_size,
        new_length=args.new_length,
        model=model,
        model_policy=LlamaModelInferPolicy(),
        verbose=True,
        max_batch_size=args.mb_size,
        max_input_len=args.seq_len,
        max_output_len=args.seq_len + args.new_length + 256,
    )
    data = data_gen(args.batch_size, args.seq_len)

    torch.cuda.synchronize()
    whole_end2end = time.time()
    output, timestamps = engine.inference([data])
    torch.cuda.synchronize()
    whole_end2end = time.time() - whole_end2end

    print_details_info(timestamps, model.config, args, whole_end2end)
