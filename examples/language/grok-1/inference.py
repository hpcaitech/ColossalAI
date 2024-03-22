import time

import torch
from sentencepiece import SentencePieceProcessor
from transformers import AutoModelForCausalLM
from utils import get_defualt_parser, inference, print_output

if __name__ == "__main__":
    parser = get_defualt_parser()
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
        output = inference(
            model,
            sp,
            text,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print_output(text, sp.decode(output))
    print(f"Overall time: {time.time() - start} seconds.")
