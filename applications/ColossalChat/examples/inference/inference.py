import argparse
import json
import os
from typing import Dict

import torch
from chatio import dummy_io, rich_io, simple_io
from coati.dataset.conversation import setup_conversation_template
from coati.models import generate_streaming
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def get_gpu_memory(max_gpus=None):
    """
    Get the available memory for each GPU.

    Args:
        max_gpus (int, optional): The maximum number of GPUs to consider. Defaults to None.

    Returns:
        list: A list of available memory for each GPU.
    """
    gpu_memory = []
    num_gpus = torch.cuda.device_count() if max_gpus is None else min(max_gpus, torch.cuda.device_count())

    for gpu_id in range(num_gpus):
        # Code to get GPU memory goes here
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda", **kwargs):
    """
    Load the model and tokenizer from the specified paths and move the model to the specified device.

    Args:
        model_path (str): The path to the pre-trained model.
        tokenizer_path (str): The path to the pre-trained tokenizer.
        device (str, optional): The device to move the model to. Defaults to "cuda".
        **kwargs: Additional keyword arguments to be passed to the `AutoModelForCausalLM.from_pretrained` function.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs, trust_remote_code=True).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    return model, tokenizer


def _set_default_generate_kwargs(model: PreTrainedModel) -> Dict:
    """
    Set default keyword arguments for generation based on the given model.

    Args:
        model (PreTrainedModel): The model used for generation.

    Returns:
        Dict: A dictionary containing the default keyword arguments for generation.
    """
    unwrapped_model = model
    new_kwargs = {}
    # Use huggingface models method directly
    if hasattr(unwrapped_model, "prepare_inputs_for_generation"):
        new_kwargs["prepare_inputs_fn"] = unwrapped_model.prepare_inputs_for_generation

    if hasattr(unwrapped_model, "_update_model_kwargs_for_generation"):
        new_kwargs["update_model_kwargs_fn"] = unwrapped_model._update_model_kwargs_for_generation
    return new_kwargs


def generation_wrapper(*args, **kwargs):
    input_ids = args[1]
    tokenizer = args[2]
    for output in generate_streaming(*args, **kwargs):
        yield tokenizer.batch_decode(output[:, input_ids.size(1) :], skip_special_tokens=True)[0]


def main(args):
    conversation_template_config = json.load(open(args.conversation_template_config, "r", encoding="utf8"))

    max_new_tokens = args.max_new_tokens
    model_max_length = args.model_max_length
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path or args.model_path, local_files_only=True
    )

    assert max_new_tokens <= model_max_length
    if hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
            tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    tokenizer.padding_side = "left"

    model_kwargs = {
        "max_new_tokens": max_new_tokens,
        # 'early_stopping': True,
        # 'top_k': -1,
        # 'top_p': 1.0,
        # 'temperature': 1.0,
        # 'temperature':0.1,
    }
    round = 1

    conv = setup_conversation_template(tokenizer, conversation_template_config, args.conversation_template_config)

    while True:
        if args.io == "simple":
            chat_io = simple_io
        elif args.io == "rich":
            chat_io = rich_io
        elif args.io == "dummy":
            chat_io = dummy_io
        else:
            raise ValueError(f"Unknown io type: {args.io}")
        # raw_text = print(">>> Human:", end=" ")
        inp = chat_io.prompt_for_input("user")

        if not inp:
            print("prompt should not be empty!")
            continue

        if inp.strip() == "clear":
            conv.clear()
            os.system("clear")
            continue

        if inp.strip() == "exit":
            print("End of chat.")
            break

        query_text = inp.strip()

        conv.append_message("user", query_text)

        chat_io.prompt_for_output("assistant")

        prompt = conv.get_prompt(add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            torch.cuda.current_device()
        )
        default_generate_kwargs = _set_default_generate_kwargs(model)
        model_kwargs.update(default_generate_kwargs)
        output_stream = generation_wrapper(
            model,
            input_ids,
            tokenizer,
            max_length=model_max_length,
            temperature=0.7,
            early_stopping=True,
            stop_token_ids=conversation_template_config["stop_ids"],
            **model_kwargs,
        )

        # print(f">>> Assistant:", end=" ")
        outputs = chat_io.stream_output(output_stream)

        conv.append_message("assistant", outputs.strip())

        with open("round.txt", mode="a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 10 + "\n")
            f.write(f"round {round}:\n{conv.save_prompt()}\n\n")
            f.write("=" * 10 + "\n")

        # print(f">>> Assistant:", end=" ")

        round += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--conversation_template_config", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--io", type=str, default="rich", choices=["simple", "rich", "dummy"])
    args = parser.parse_args()
    main(args)
