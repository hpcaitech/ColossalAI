from argparse import ArgumentParser

import torch
from model.modeling_openmoe import OpenMoeForCausalLM, set_openmoe_args
from transformers import T5Tokenizer
from transformers.models.llama import LlamaConfig


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="base", type=str, help="model path", choices=["base", "8b", "test"])
    return parser.parse_args()


def inference(args):
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    if args.model == "test":
        config = LlamaConfig.from_pretrained("hpcai-tech/openmoe-base")
        set_openmoe_args(
            config, num_experts=config.num_experts, moe_layer_interval=config.moe_layer_interval, enable_kernel=True
        )
        model = OpenMoeForCausalLM(config)
    else:
        config = LlamaConfig.from_pretrained(f"hpcai-tech/openmoe-{args.model}")
        set_openmoe_args(
            config, num_experts=config.num_experts, moe_layer_interval=config.moe_layer_interval, enable_kernel=False
        )
        model = OpenMoeForCausalLM.from_pretrained(f"hpcai-tech/openmoe-{args.model}", config=config)
    model = model.eval().bfloat16()
    model = model.to(torch.cuda.current_device())

    input_str = """```
y = list(map(int, ['1', 'hello', '2']))
```
What error does this program produce?
ValueError: invalid literal for int() with base 10: 'hello'

```
sum = 0
for i in range(100):
        sum += i
```
What is the value of sum immediately after the 10th time line 3 is executed?"""

    # print("model config: ", model.config)
    input_ids = tokenizer("<pad>" + input_str, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.input_ids.to(torch.cuda.current_device())
    generation_output = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=64)
    out = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    print(f"output: \n{out}\n")


if __name__ == "__main__":
    args = parse_args()
    inference(args)
