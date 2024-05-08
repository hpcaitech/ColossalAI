import argparse

from colossal_llama.utils.stream_chat_patch import streaming_chat
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    past_key_values, history = None, []
    roles = ["", "Human", "Assistant"]

    history = []
    history.append({"role": roles[0], "message": SYSTEM})

    while True:
        input_query = input(f"\n{roles[1]}: ")
        if input_query.strip() == "exit":
            break
        if input_query.strip() == "clear":
            past_key_values, history = None, []
            continue

        print(f"\n{roles[2]}: ", end="")
        gen_len = 0
        for response, history, past_key_values in streaming_chat(
            model,
            tokenizer,
            input_query,
            history=history,
            roles=roles,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            length_penalty=args.length_penalty,
            max_new_tokens=args.max_new_tokens,
            past_key_values=past_key_values,
            return_past_key_values=True,
        ):
            output = response[gen_len:]
            print(output, end="", flush=True)
            gen_len = len(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="path to chat version model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to chat version tokenizer")
    parser.add_argument("--temperature", type=float, default=0.8, help="set temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="set top p value")
    parser.add_argument("--top_k", type=int, default=50, help="set top k value")
    parser.add_argument("--do_sample", type=bool, default=True, help="whether turn on do_sample or not")
    parser.add_argument("--length_penalty", type=float, default=1.2, help="set length penalty")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="set max new tokens")
    args = parser.parse_args()
    main(args)
