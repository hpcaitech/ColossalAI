import argparse

import loralib as lora
import torch
from chatgpt.dataset import RewardDataset
from chatgpt.nn import BLOOMRM
from chatgpt.trainer import RewardModelTrainer
from datasets import load_dataset
from transformers import BloomTokenizerFast


def train(args):
    tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
    tokenizer.pad_token = tokenizer.eos_token
    model = BLOOMRM(pretrained=args.pretrain)

    model.cuda()

    max_len = 1024

    # prepare for data and dataset
    data = load_dataset(args.dataset)
    train_data = data["train"]
    eval_data = data['test']
    train_dataset = RewardDataset(train_data, tokenizer, max_len)
    eval_dataset = RewardDataset(eval_data, tokenizer, max_len)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = RewardModelTrainer(model=model,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 num_epochs=args.max_epochs)

    trainer.fit(use_lora=args.lora_rank)

    if args.lora_rank > 0:
        torch.save({'model_state_dict': lora.lora_state_dict(trainer.model)}, args.save_path)
    else:
        torch.save(trainer.model, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    parser.add_argument('--save_path', type=str, default='rm_ckpt.pth')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    args = parser.parse_args()
    train(args)
