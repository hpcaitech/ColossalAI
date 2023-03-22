import argparse

import loralib as lora
import torch
from chatgpt.dataset import HhRlhfDataset, RmStaticDataset
from chatgpt.models import LogSigLoss, LogExpLoss
from chatgpt.models.base import RewardModel
from chatgpt.models.bloom import BLOOMRM
from chatgpt.models.gpt import GPTRM
from chatgpt.models.opt import OPTRM
from chatgpt.models.roberta import RoBERTaRM
from chatgpt.trainer import RewardModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from datasets import load_dataset
from random import randint
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam

def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'roberta':
            model = RoBERTaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')
        
        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
        
    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    max_len = args.max_len

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=1.5e-5)
    else:
        optim = Adam(model.parameters(), lr=1.5e-5)
    
    # configure loss function
    if args.loss_fn == 'log_sig':
        loss_fn = LogSigLoss()
    elif args.loss_fn == 'log_exp':
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')
    
    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(args.dataset, data_dir=args.subset)
    else:
        data = load_dataset(args.dataset)
    
    if args.test:
        train_data = data['train'].select(range(100))
        eval_data = data['test'].select(range(10)) 
    else:
        train_data = data['train']
        eval_data = data['test']
    valid_data = data['test'].select((randint(0, len(eval_data) - 1) for _ in range(len(eval_data)//10)))
    
    if args.dataset == 'Dahoas/rm-static':
        train_dataset = RmStaticDataset(train_data, tokenizer, max_len)
        valid_dataset = RmStaticDataset(valid_data, tokenizer, max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, max_len)
    elif args.dataset == 'Anthropic/hh-rlhf':
        train_dataset = HhRlhfDataset(train_data, tokenizer, max_len)
        valid_dataset = HhRlhfDataset(valid_data, tokenizer, max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, max_len)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')
    
    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 loss_fn = loss_fn,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs)

    trainer.fit()
    # save model checkpoint after fitting on only rank0
    strategy.save_model(trainer.model, args.save_path, only_rank0=True)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer, 'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()), only_rank0=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'roberta'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--dataset', type=str,
                        choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static'],
                        default='Dahoas/rm-static')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='rm_ckpt.pt')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_sig', choices=['log_sig', 'log_exp'])
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    train(args)
