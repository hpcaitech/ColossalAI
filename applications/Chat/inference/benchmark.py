# Adapted from https://github.com/tloen/alpaca-lora/blob/main/generate.py

import argparse
from time import time

import torch
from llama_gptq import load_quant
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    n_new_tokens = s.size(0) - input_ids.size(1)
    return output.split("### Response:")[1].strip(), n_new_tokens


instructions = [
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
    # ===
    "How to play support in legends of league",
    "Write a Python program that calculate Fibonacci numbers.",
]
inst = [instructions[0]] * 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pretrained',
        help='Path to pretrained model. Can be a local path or a model name from the HuggingFace model hub.')
    parser.add_argument('--quant',
                        choices=['8bit', '4bit'],
                        default=None,
                        help='Quantization mode. Default: None (no quantization, fp16).')
    parser.add_argument(
        '--gptq_checkpoint',
        default=None,
        help='Path to GPTQ checkpoint. This is only useful when quantization mode is 4bit. Default: None.')
    parser.add_argument('--gptq_group_size',
                        type=int,
                        default=128,
                        help='Group size for GPTQ. This is only useful when quantization mode is 4bit. Default: 128.')
    args = parser.parse_args()

    if args.quant == '4bit':
        assert args.gptq_checkpoint is not None, 'Please specify a GPTQ checkpoint.'

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    if args.quant == '4bit':
        model = load_quant(args.pretrained, args.gptq_checkpoint, 4, args.gptq_group_size)
        model.cuda()
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.pretrained,
            load_in_8bit=(args.quant == '8bit'),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if args.quant != '8bit':
            model.half()    # seems to fix bugs for some users.
        model.eval()

    total_tokens = 0
    start = time()
    for instruction in instructions:
        print(f"Instruction: {instruction}")
        resp, tokens = evaluate(model, tokenizer, instruction, temperature=0.2, num_beams=1)
        total_tokens += tokens
        print(f"Response: {resp}")
        print('\n----------------------------\n')
    duration = time() - start
    print(f'Total time: {duration:.3f} s, {total_tokens/duration:.3f} tokens/s')
    print(f'Peak CUDA mem: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB')
