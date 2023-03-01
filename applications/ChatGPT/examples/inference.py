import argparse
import torch

from chatgpt.nn import BLOOMActor, GPTActor, OPTActor
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


def eval(args):
    # configure model
    if args.model == 'gpt2':
        model = GPTActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    elif args.model == 'bloom':
        model = BLOOMActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    elif args.model == 'opt':
        model = OPTActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    model.eval()
    input = args.input
    input_ids = tokenizer.encode(input, return_tensors='pt').to(torch.cuda.current_device())
    outputs = model.generate(input_ids,
                             max_length=args.max_length,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95,
                             num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--input', type=str, default='Q: How are you ? A:')
    parser.add_argument('--max_length', type=int, default=100)
    args = parser.parse_args()
    eval(args)
