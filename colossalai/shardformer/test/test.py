from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BertForMaskedLM
import colossalai
from colossalai.shardformer.shard.shardmodel import ShardModel
from colossalai.utils import get_current_device, print_rank_0
from colossalai.logging import get_dist_logger
from colossalai.shardformer.shard.shardconfig import ShardConfig
import inspect
import argparse
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument("--mode", type=str, default='inference')
    return parser.parse_args()

def load_data():
    datasets=load_dataset('wikitext', 'wikitext-2-raw-v1')
    # datasets=load_dataset("yelp_review_full")
    tokenized_datasets=datasets.map(lambda examples:tokenizer(examples["text"],truncation=True,padding="max_length"),batched=True)
    tokenized_datasets=tokenized_datasets.remove_columns(["text"])
    # tokenized_datasets=tokenized_datasets.rename_column("label","labels")
    tokenized_datasets.set_format("torch")
    
    train_dataset=tokenized_datasets["train"].select(range(1000))
    test_dataset=tokenized_datasets["test"].select(range(100))

    datacollector = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")
    train_dataloader=DataLoader(train_dataset,batch_size=8,shuffle=True, collate_fn=datacollector)
    eval_dataloader=DataLoader(test_dataset,batch_size=8, collate_fn=datacollector)
    return train_dataloader,eval_dataloader

def inference(model: nn.Module):
    print(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    token = "Hello, my dog is cute"
    inputs = tokenizer(token, return_tensors="pt")
    inputs.to("cuda")
    model.to("cuda")
    outputs = model(**inputs)
    print(outputs)

def train(model: nn.Module, num_epoch: int=2):
    train_dataloader, eval_dataloader=load_data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    progress_bar = tqdm(range((num_epoch)*len(train_dataloader)))
    criterion = nn.CrossEntropyLoss()
    model.to("cuda")
    model.train()
    for epoch in range(num_epoch):
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            # loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f"loss: {loss.item()}")
        print(f"Rank {os.environ['RANK']} Epoch:{epoch} Train Loss:{loss:.4f}")        
        
        for batch in eval_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            # loss = criterion(outputs.logits, batch["input_ids"])
        print(f"Rank {os.environ['RANK']} Epoch:{epoch} Test Loss:{loss:.4f}")        
        



if __name__ == "__main__":
    args = get_args()
    colossalai.launch_from_torch(config=args.config)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    shard_config = ShardConfig(
        rank = int(str(get_current_device()).split(':')[-1]),
        world_size= int(os.environ['WORLD_SIZE']),
    )
    shardmodel = ShardModel(model, shard_config)
    if args.mode == "train":
        train(shardmodel.model)
    elif args.mode == "inference":
        inference(shardmodel.model) 
