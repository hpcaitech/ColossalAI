import argparse
import os

import torch
import torch.distributed as dist
from auto_gptq import AutoGPTQForCausalLM

import colossalai
from colossalai.inference import CaiInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.testing import spawn

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def run_llama_inference(args):
    quantized_model_dir = args.quantized_path
    max_batch_size = args.max_batch_size
    max_input_len = args.max_input_len
    max_output_len = args.max_output_len
    micro_batch_size = args.micro_batch_size
    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir, inject_fused_attention=False, device=torch.cuda.current_device()
    )

    engine = CaiInferEngine(
        tp_size=2,
        pp_size=2,
        model=model,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        micro_batch_size=micro_batch_size,
        quant="gptq",
    )

    def data_gen():
        input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779, 318, 13779]], dtype=torch.int64)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    inputs = data_gen()
    for k, v in inputs.items():
        if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
            new_shape = [1] * v.dim()
            new_shape[0] = 16
            inputs[k] = v.to("cuda").repeat(*new_shape)

    output = engine.generate(inputs)
    if dist.get_rank() == 0:
        assert len(output[0]) == max_output_len, f"{len(output)}, {max_output_len}"


def run_gptq_infernece(rank, world_size, port, args):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_llama_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quantized_path", type=str, help="Model path", required=True)
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--max_input_len", type=int, default=32, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=32, help="Maximum output length")
    args = parser.parse_args()

    spawn(run_gptq_infernece, args.tp_size * args.pp_size, args=args)
