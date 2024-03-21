import argparse
import time

import torch
from sentencepiece import SentencePieceProcessor
from transformers import AutoModelForCausalLM


class Bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_output(text, output):
    print(f"-----\n{Bcolors.OKBLUE}{text}{Bcolors.ENDC}{output[len(text):]}")


@torch.no_grad()
def inference(model, sp, text, max_new_tokens):
    input_ids = sp.encode(text)
    input_ids = torch.tensor([input_ids]).cuda()
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
    }
    outputs = model.generate(**inputs)
    return outputs[0].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="hpcaitech/grok-1")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--text", type=str, nargs="+", default=["Hi, what's your name?"])
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args = parser.parse_args()
    start = time.time()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    for text in args.text:
        output = inference(model, sp, text, args.max_new_tokens)
        print_output(text, sp.decode(output))
    print(f"Overall time: {time.time() - start} seconds.")
