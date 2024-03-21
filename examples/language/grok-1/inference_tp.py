import argparse
import time

import torch
from grok1_policy import Grok1ForCausalLMPolicy
from sentencepiece import SentencePieceProcessor
from transformers import AutoModelForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.utils import get_current_device


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
def inference(model, sp, text, **generate_kwargs):
    input_ids = sp.encode(text)
    input_ids = torch.tensor([input_ids]).cuda()
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **generate_kwargs,
    }
    outputs = model.generate(**inputs)
    return outputs[0].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="hpcaitech/grok-1")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.model")
    parser.add_argument("--text", type=str, nargs="+", default=["Hi, what's your name?"])
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.3, help="Set temperature value")
    parser.add_argument("--top_k", type=int, default=50, help="Set top_k value for top-k-filtering")
    parser.add_argument("--top_p", type=float, default=0.95, help="Set top_p value for generation")
    args = parser.parse_args()
    start = time.time()
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    plugin = HybridParallelPlugin(
        tp_size=coordinator.world_size,
        pp_size=1,
        precision="bf16",
        parallel_output=False,
        custom_policy=Grok1ForCausalLMPolicy(),
    )
    booster = Booster(plugin=plugin)
    torch.set_default_dtype(torch.bfloat16)
    with LazyInitContext(default_device=get_current_device()):
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    model, *_ = booster.boost(model)
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    for text in args.text:
        output = inference(
            model.unwrap(),
            sp,
            text,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if coordinator.is_master():
            print_output(text, sp.decode(output))
    coordinator.print_on_master(f"Overall time: {time.time() - start} seconds.")
