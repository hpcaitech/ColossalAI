import time

import torch
from grok1_policy import Grok1ForCausalLMPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_default_parser, inference, print_output

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.utils import get_current_device

if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    start = time.time()
    colossalai.launch_from_torch()
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

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, trust_remote_code=True)

    with LazyInitContext(default_device=get_current_device()):
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    model, *_ = booster.boost(model)
    model.eval()
    init_time = time.time() - start

    for text in args.text:
        output = inference(
            model.unwrap(),
            tokenizer,
            text,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if coordinator.is_master():
            print_output(text, tokenizer.decode(output))

    overall_time = time.time() - start
    gen_latency = overall_time - init_time
    avg_gen_latency = gen_latency / len(args.text)
    coordinator.print_on_master(
        f"Initializing time: {init_time:.2f} seconds.\n"
        f"Overall time: {overall_time:.2f} seconds. \n"
        f"Generation latency: {gen_latency:.2f} seconds. \n"
        f"Average generation latency: {avg_gen_latency:.2f} seconds. \n"
    )
