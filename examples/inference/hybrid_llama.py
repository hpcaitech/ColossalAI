import argparse
import time

import torch
import torch.distributed as dist
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.inference import CaiInferEngine
from colossalai.testing import spawn


def run_inference(args):
    llama_model_path = args.path
    max_input_len = args.max_input_len
    max_output_len = args.max_output_len
    max_batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    rank = dist.get_rank()

    tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=20000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=4
        )
    )

    engine = CaiInferEngine(
        tp_size=tp_size,
        pp_size=pp_size,
        model=model,
        max_output_len=max_output_len,
        micro_batch_size=micro_batch_size,
    )

    input_tokens = {
        "input_ids": torch.randint(1, 1000, (max_batch_size, max_input_len), device="cuda"),
        "attention_mask": torch.ones((max_batch_size, max_input_len), device="cuda"),
    }

    iters = 10
    warmup = 3
    times = []

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = engine.generate(input_tokens)
        torch.cuda.synchronize()
        end = time.time()
        if rank == 0:
            out_len = len(outputs[0])
            print("generation time {} s".format(str(end - start)))
            print(out_len)
            times.append((end - start) / out_len)
    if rank == 0:
        times = times[warmup:]
        latency = sum(times) / len(times)
        print("total process latency is : " + str(latency) + " s")
        print("total throughput is : " + str(1 / latency * max_batch_size))


def run_tp_pipeline_inference(rank, world_size, port, args):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Model path", required=True)
    parser.add_argument("-tp", "--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("-pp", "--pp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Maximum batch size")
    parser.add_argument("--max_input_len", type=int, default=512, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=256, help="Maximum output length")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size")

    args = parser.parse_args()
    spawn(run_tp_pipeline_inference, nprocs=args.tp_size * args.pp_size, args=args)
