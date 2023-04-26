import argparse
from random import randint

import loralib as lora
import torch
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMRM
from coati.models.deberta import DebertaRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.models.roberta import RoBERTaRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast, DebertaV2Tokenizer, LlamaTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from colossalai.nn.optimizer import HybridAdam

from tqdm import tqdm

def normalize(args):
    # configure model
    if args.model == 'bloom':
        model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    elif args.model == 'opt':
        model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    elif args.model == 'gpt2':
        model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    elif args.model == 'deberta':
        model = DebertaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    elif args.model == 'llama':
        model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    elif args.model == 'roberta':
        model = RoBERTaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    model = model.to(torch.float16)

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'deberta':
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    max_len = args.max_len

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(args.dataset, data_dir=args.subset)
    else:
        data = load_dataset(args.dataset)

    if args.test:
        train_data = data['train'].select(range(10000))
    else:
        train_data = data['train']

    if args.dataset == 'Dahoas/rm-static':
        train_dataset = RmStaticDataset(train_data, tokenizer, max_len)
    elif args.dataset == 'Anthropic/hh-rlhf':
        train_dataset = HhRlhfDataset(train_data, tokenizer, max_len)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model.eval()
    with torch.no_grad():
        bar = tqdm(train_dataloader)
        output = []
        for chosen_ids, c_mask, reject_ids, r_mask in train_dataloader:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
            chosen_reward = model(chosen_ids, attention_mask=c_mask)
            reject_reward = model(reject_ids, attention_mask=r_mask)
            output.append(chosen_reward)
            output.append(reject_reward)
            bar.update()
        bar.close()
    output = torch.cat(output)
    mean = output.mean()
    std = output.std()
    print(mean.item(), std.item())
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'deberta', 'llama', 'roberta'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset',
                        type=str,
                        choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static'],
                        default='Dahoas/rm-static')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--lora_rank', type=int, default=0)
    args = parser.parse_args()
    normalize(args)
