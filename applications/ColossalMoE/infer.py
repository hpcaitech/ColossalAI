from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="base", type=str, help="model path", choices=["base", "8b", "test"])
    return parser.parse_args()


def inference(args):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    model = model.eval().bfloat16()
    print(f"param num: {sum(p.numel() for p in model.parameters())/ 1000.0 ** 3}GB")
    model = model.to(torch.cuda.current_device())

    text = "Hello my name is"
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    args = parse_args()
    inference(args)
