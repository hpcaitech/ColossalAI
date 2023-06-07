import os
import random

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, GPT2LMHeadModel, get_scheduler

import colossalai
from colossalai.shardformer.shard import ShardConfig, shard_model
from colossalai.utils import get_current_device, print_rank_0

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument("--mode", type=str, default='inference')
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--model", type=str, default='bert-base-uncased')
    return parser.parse_args()


def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # tokenizer.pad_token_id = 0
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    # datasets=load_dataset("yelp_review_full")
    tokenized_datasets = datasets.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # tokenized_datasets=tokenized_datasets.rename_column("label","labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    datacollector = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=datacollector)
    eval_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=datacollector)
    return train_dataloader, eval_dataloader


def inference(model: nn.Module, args):
    print(model)
    # print(model.wte.weight.shape)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = 0
    token = "Hello, my dog is cute"
    inputs = tokenizer(token, return_tensors="pt")
    inputs.to("cuda")
    model.eval()
    model.to("cuda")
    outputs = model(**inputs)
    print(outputs[0])


def train(model: nn.Module, args, num_epoch: int = 3):
    train_dataloader, eval_dataloader = load_data(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training = num_epoch * len(train_dataloader)
    progress_bar = tqdm(range(num_training))
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training)
    best_test_loss = float("inf")
    model.to("cuda")
    model.train()
    for epoch in range(num_epoch):
        progress_bar.set_description(f"Rank {get_current_device()} epoch {epoch}")
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
        train_loss = loss

        loss = 0.0
        for batch in eval_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            # loss = outputs.loss
            assert not torch.isnan(outputs.loss), f"{batch}"
            loss += outputs.loss.item()
            # loss = criterion(outputs.logits, batch["input_ids"])
        test_loss = loss / len(eval_dataloader)
        print_rank_0(f"Train Loss: {train_loss:.4f} Test Loss:{test_loss:.4f}")
        if args.save_model and test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")


if __name__ == "__main__":
    args = get_args()
    colossalai.launch_from_torch(config=args.config)
    if args.model == 'bert-base-uncased':
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    elif args.model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        raise AttributeError("model not supported")
    shard_config = ShardConfig(
        rank=int(str(get_current_device()).split(':')[-1]),
        world_size=int(os.environ['WORLD_SIZE']),
    )
    sharded_model = shard_model(model, shard_config)

    if args.mode == "train":
        train(sharded_model, args)
    elif args.mode == "inference":
        inference(sharded_model, args)
    else:
        raise NotImplementedError
