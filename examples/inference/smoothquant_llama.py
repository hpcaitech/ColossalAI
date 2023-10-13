import argparse
import os

import torch
from datasets import load_dataset
from transformers import LlamaTokenizer

from colossalai.inference.quant.smoothquant.models.llama import SmoothLlamaForCausalLM


def build_model_and_tokenizer(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = SmoothLlamaForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(torch.float32)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="model name")
    parser.add_argument(
        "--output-path",
        type=str,
        help="where to save the checkpoint",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="location of the calibration dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model_path = args.model_name
    dataset_path = args.dataset_path
    output_path = args.output_path
    num_samples = 10
    seq_len = 512

    model, tokenizer = build_model_and_tokenizer(model_path)
    if not os.path.exists(dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        raise FileNotFoundError
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    model.quantized(tokenizer, dataset, num_samples=num_samples, seq_len=seq_len)
    model = model.cuda()

    model.save_quantized(output_path, model_basename="llama-7b")

    model = SmoothLlamaForCausalLM.from_quantized(output_path, model_basename="llama-7b")
    model = model.cuda()

    generate_kwargs = dict(max_new_tokens=16, do_sample=False, use_cache=True)
    input_tokens = tokenizer(["today is "], return_tensors="pt").to("cuda")
    out = model.generate(**input_tokens, **generate_kwargs)
    text = tokenizer.batch_decode(out)
    print("out is:", text)


if __name__ == "__main__":
    main()
