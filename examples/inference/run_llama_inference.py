import argparse

import torch
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.inference import InferenceEngine
from colossalai.testing import spawn


def run_inference(args):
    llama_model_path = args.model_path
    llama_tokenize_path = args.tokenizer_path

    max_input_len = args.max_input_len
    max_output_len = args.max_output_len
    max_batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    rank = dist.get_rank()

    tokenizer = LlamaTokenizer.from_pretrained(llama_tokenize_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.unk_token_id

    if args.quant is None:
        model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.unk_token_id)
        model = model.half()
    elif args.quant == "gptq":
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            llama_model_path, inject_fused_attention=False, device=torch.cuda.current_device()
        )
    elif args.quant == "smoothquant":
        from colossalai.inference.quant.smoothquant.models.llama import SmoothLlamaForCausalLM

        model = SmoothLlamaForCausalLM.from_quantized(llama_model_path, model_basename=args.smoothquant_base_name)
        model = model.cuda()

    engine = InferenceEngine(
        tp_size=tp_size,
        pp_size=pp_size,
        model=model,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        micro_batch_size=micro_batch_size,
        quant=args.quant,
    )

    input_tokens = {
        "input_ids": torch.randint(1, 1000, (max_batch_size, max_input_len), device="cuda"),
        "attention_mask": torch.ones((max_batch_size, max_input_len), device="cuda"),
    }

    outputs = engine.generate(input_tokens)
    if rank == 0:
        print(tokenizer.batch_decode(outputs))


def run_tp_pipeline_inference(rank, world_size, port, args):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_path", type=str, help="Model path", required=True)
    parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path", required=True)

    parser.add_argument(
        "-q",
        "--quant",
        type=str,
        choices=["gptq", "smoothquant"],
        default=None,
        help="quantization type: 'gptq' or 'smoothquant'",
    )
    parser.add_argument("--smoothquant_base_name", type=str, default=None, help="soothquant base name")
    parser.add_argument("-tp", "--tp_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("-pp", "--pp_size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Maximum batch size")
    parser.add_argument("--max_input_len", type=int, default=32, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=16, help="Maximum output length")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")

    args = parser.parse_args()
    spawn(run_tp_pipeline_inference, nprocs=args.tp_size * args.pp_size, args=args)
