import argparse

import loralib as lora
import torch
from chatgpt.dataset import RewardDataset
from chatgpt.nn import BLOOMRM
from chatgpt.trainer import RewardModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from datasets import load_dataset
from torch.optim import Adam
from transformers import BloomTokenizerFast

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
    tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
    tokenizer.pad_token = tokenizer.eos_token
    with strategy.model_init_context():
        model = BLOOMRM(pretrained=args.pretrain).cuda()
    max_len = 1024

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=5e-5)
    else:
        optim = Adam(model.parameters(), lr=5e-5)

    # prepare for data and dataset
    data = load_dataset(args.dataset)
    train_data = data["train"].select(range(100))
    eval_data = data['test'].select(range(5))
    train_dataset = RewardDataset(train_data, tokenizer, max_len)
    eval_dataset = RewardDataset(eval_data, tokenizer, max_len)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs)

    trainer.fit(use_lora=args.lora_rank)

    if args.lora_rank > 0:
        torch.save({'model_state_dict': lora.lora_state_dict(trainer.model)}, args.save_path)
    else:
        torch.save(trainer.model, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    parser.add_argument('--save_path', type=str, default='rm_ckpt.pth')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    args = parser.parse_args()
    train(args)
