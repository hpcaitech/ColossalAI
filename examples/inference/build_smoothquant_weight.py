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
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        help="pretrained model directory",
        required=True,
    )
    parser.add_argument(
        "--quantized_model_dir",
        help="the path of out smoothquant model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="location of the dataset",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model_path = args.pretrained_model_dir
    dataset_path = args.dataset_path
    output_path = args.quantized_model_dir
    num_samples = args.num_samples
    seq_len = args.seq_len

    model, tokenizer = build_model_and_tokenizer(model_path)
    model.config.pad_token_id = model.config.eos_token_id
    if dataset_path is not None:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Cannot find the dataset at {args.dataset_path}")
        else:
            dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = ["Written by Adele and its producer, ", " brings you the latest celebrity & royal news from the U"]
    model.quantized(tokenizer, dataset, num_samples=num_samples, seq_len=seq_len)
    model = model.cuda()

    model.save_quantized(output_path, model_basename="llama-7b")


if __name__ == "__main__":
    main()
