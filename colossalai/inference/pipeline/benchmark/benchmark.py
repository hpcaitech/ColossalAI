import torch
import torch.distributed as dist
import transformers

import colossalai
from colossalai.inference import PPInferEngine
from colossalai.inference.pipeline.policy.llama_ppinfer import LlamaForCausalLMPipelinePolicy
import argparse
GIGABYTE = 1024 ** 3
MEGABYTE = 1024 * 1024

colossalai.launch_from_torch(config={})

def data_gen(batch_size: int=4, seq_len: int=512):
    input_ids = torch.randint(10, 30000, (1, seq_len), dtype=torch.int32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.int32)
    data = dict(input_ids=input_ids, attention_mask=attention_mask)
    for k, v in data.items():
        if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__:
            new_shape = [1] * v.dim()
            new_shape[0] = batch_size
            data[k] = v.to('cuda').repeat(*new_shape)
    return data

def print_details_info(timestamps, model_config, args):
    if dist.get_rank() == 0:
        prefill = []
        encoder = []
        end2end = []
        for timestamp in timestamps:
            prefill.append(timestamp[1] - timestamp[0])
            encoder.append(
                sum(timestamp[i + 1] - timestamp[i] for i in range(1,len(timestamp) - 1)) / (len(timestamp) - 2))
            end2end.append(timestamp[-1] - timestamp[0])
        with open(f"llama-{args.model}{'fp16' if args.fp16 is True else 'fp32'}_pp{args.pp_size}_{args.seq_len}_{args.new_length}_bsz{args.batch_size}_mbsz{args.mb_size}.log","w+") as f:
            avg_latency = sum(end2end)/(args.new_length * args.batch_size)
            num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
            num_parameters = num_layers * model_config.hidden_size * model_config.hidden_size * 12 / args.pp_size
            if args.fp16:
                num_bytes = 2
            else:
                num_bytes = 4

            f.write(f"llama-{args.model} {'fp16' if args.fp16 is True else 'fp32'} {args.pp_size}, input_len:{args.seq_len}, output_len:{args.new_length}, bsz:{args.batch_size}, mbsz:{args.mb_size}\n")
            f.write("Average prefill time: {0:8.2f} ms\n".format(sum(prefill)/len(prefill)*1000))
            f.write("Average encode time: {0:8.2f} ms\n".format(sum(encoder)/len(encoder)*1000))
            f.write("Average end2end time: {0:8.2f} ms\n".format(sum(end2end)/len(end2end)*1000))
            f.write("Average Per Token Latency: {0:8.2f} ms\n".format(avg_latency * 1000))
            f.write("Avg flops: {0:8.2f} TFlops/s\n".format(1/avg_latency * num_parameters * num_bytes / 1e12))
            f.write("Average Throughput: {} tokens/s\n".format((1000/(avg_latency * 1000))))
            f.write("----------------------------------------------------------\n")


    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()

        # free memory and the total available memory in bytes
        global_free_memory, total_GPU_memory_occupied = torch.cuda.mem_get_info()
        memory_allocated = torch.cuda.memory_allocated()
        max_memory_allocated = torch.cuda.max_memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        max_memory_reserved = torch.cuda.max_memory_reserved()
        with open(f"llama-{args.model}{'fp16' if args.fp16 is True else 'fp32'}_pp{args.pp_size}_{args.seq_len}_{args.new_length}_bsz{args.batch_size}_mbsz{args.mb_size}.log","a") as f:
            f.write(
                f"\nCurrently using GPU: {current_device}\n"
                f"free memory : {global_free_memory / GIGABYTE:.4f} GB,\n"
                f"total memory: {total_GPU_memory_occupied / GIGABYTE:.4f} GB,\n"
                f"memory allocated: {memory_allocated / GIGABYTE:.4f} GB,\n"
                f"Max CUDA memory allocated: {max_memory_allocated / GIGABYTE:.4f} GB,\n"
                f"memory reserved/cached: {memory_reserved / GIGABYTE:.4f} GB,\n"
                f"Max CUDA memory reserved/cached: {max_memory_reserved / GIGABYTE:.4f} GB,\n"
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='toy', help='the size of model')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-s', '--seq_len', type=int, default=8, help='sequence length')
    parser.add_argument('--new_length', type=int, default=4, help='new tokens length')
    parser.add_argument('--mb_size', type=int, default=1, help='micro_batch_size')
    parser.add_argument('--pp_size', type=int, default=2, help='pipeline size')
    parser.add_argument('--fp16', action="store_true", help='wheather to use fp16')
    args = parser.parse_args()

    if args.model == 'toy':
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(num_hidden_layers=8))
    elif args.model == '7b':
        model = transformers.LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
    elif args.model == '13b':
        model = transformers.LlamaForCausalLM.from_pretrained('decapoda-research/llatma-13b-hf')
    else:
        raise NotImplementedError
  
    engine = PPInferEngine(pp_size=args.pp_size, fp16=args.fp16, micro_batch_size=args.mb_size, new_length=args.new_length, model=model, model_policy=LlamaForCausalLMPipelinePolicy(),verbose=True)
    data = data_gen(args.batch_size, args.seq_len)
    output, timestamps = engine.inference([data])
    if dist.get_rank() == 0:
        print(len(output), len(output[0]))
    print_details_info(timestamps, model.config, args)
    
