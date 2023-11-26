import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from colossalai.legacy.registry import DATASETS


@DATASETS.register_module
class WebtextDataset(Dataset):
    def __init__(self, path: Optional[str] = None, seq_len=1024) -> None:
        super().__init__()
        if path is not None:
            root = os.path.dirname(path)
            encoded_data_cache_path = os.path.join(root, f"gpt_webtext_{seq_len}.pt")
            if os.path.isfile(encoded_data_cache_path):
                seq_len_, data, attention_mask = torch.load(encoded_data_cache_path)
                if seq_len_ == seq_len:
                    self.data = data
                    self.attention_mask = attention_mask
                    return
            raw_data = []
            with open(path) as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line)["text"])
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.unk_token
            encoded_data = tokenizer(raw_data, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
            self.data = encoded_data["input_ids"]
            self.attention_mask = encoded_data["attention_mask"]
        else:
            self.data = torch.randint(0, 50257, (10240, seq_len))
            self.attention_mask = torch.ones_like(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"input_ids": self.data[index], "attention_mask": self.attention_mask[index]}, self.data[index]
