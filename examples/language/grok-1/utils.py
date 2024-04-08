import argparse

import torch


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
def inference(model, tokenizer, text, **generate_kwargs):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **generate_kwargs,
    }
    outputs = model.generate(**inputs)
    return outputs[0].tolist()


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="hpcaitech/grok-1")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--text", type=str, nargs="+", default=["Hi, what's your name?"])
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.3, help="Set temperature value")
    parser.add_argument("--top_k", type=int, default=50, help="Set top_k value for top-k-filtering")
    parser.add_argument("--top_p", type=float, default=0.95, help="Set top_p value for generation")
    return parser
