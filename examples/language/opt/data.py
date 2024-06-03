import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class NetflixDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.txt_list = netflix_descriptions = load_dataset("hugginglearners/netflix-shows", split="train")[
            "description"
        ]
        self.max_length = max([len(self.tokenizer.encode(description)) for description in netflix_descriptions])

        for txt in self.txt_list:
            encodings_dict = self.tokenizer(
                "</s>" + txt + "</s>", truncation=True, max_length=self.max_length, padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def netflix_collator(data):
    return {
        "input_ids": torch.stack([x[0] for x in data]),
        "attention_mask": torch.stack([x[1] for x in data]),
        "labels": torch.stack([x[0] for x in data]),
    }
