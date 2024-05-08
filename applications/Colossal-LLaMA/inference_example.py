import argparse

import torch
from colossal_llama.dataset.conversation import default_conversation
from transformers import AutoModelForCausalLM, AutoTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def load_model(model_path, device="cuda", **kwargs):
    logger.info("Please check whether the tokenizer and model weights are properly stored in the same folder.")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    except OSError:
        raise ImportError("Tokenizer not found. Please check if the tokenizer exists or the model path is correct.")

    return model, tokenizer


@torch.inference_mode()
def generate(args):
    model, tokenizer = load_model(model_path=args.model_path, device=args.device)

    if args.prompt_style == "sft":
        conversation = default_conversation.copy()
        conversation.append_message("Human", args.input_txt)
        conversation.append_message("Assistant", None)
        input_txt = conversation.get_prompt()
    else:
        BASE_INFERENCE_SUFFIX = "\n\n->\n\n"
        input_txt = f"{args.input_txt}{BASE_INFERENCE_SUFFIX}"

    inputs = tokenizer(input_txt, return_tensors="pt").to(args.device)
    num_input_tokens = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=1,
    )
    response = tokenizer.decode(output.cpu()[0, num_input_tokens:], skip_special_tokens=True)
    logger.info(f"\nHuman: {args.input_txt} \n\nAssistant: \n{response}")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colossal-LLaMA-2 inference Process.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="hpcai-tech/Colossal-LLaMA-2-7b-base",
        help="HF repo name or local path of the model",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Set the device")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help=" Set maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
    )
    parser.add_argument("--do_sample", type=bool, default=True, help="Set whether or not to use sampling")
    parser.add_argument("--temperature", type=float, default=0.3, help="Set temperature value")
    parser.add_argument("--top_k", type=int, default=50, help="Set top_k value for top-k-filtering")
    parser.add_argument("--top_p", type=float, default=0.95, help="Set top_p value for generation")
    parser.add_argument("--input_txt", type=str, default="明月松间照，", help="The prompt input to the model")
    parser.add_argument("--prompt_style", choices=["sft", "pretrained"], default="sft", help="The style of the prompt")
    args = parser.parse_args()
    generate(args)
