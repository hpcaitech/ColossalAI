import copy
import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class EasySupervisedDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        super(EasySupervisedDataset, self).__init__()
        with open(data_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
        # split to source and target ,source the characters before "回答：" including "回答：", target the characters after "回答："
        sources, targets = [], []
        for line in all_lines:
            if "回答：" in line:
                sep_index = line.index("回答：")
                sources.append(line[: sep_index + 3])
                targets.append(line[sep_index + 3 :] + tokenizer.eos_token)
            else:
                sources.append(line)
                targets.append("" + tokenizer.eos_token)
        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.data_file = data_file

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __repr__(self):
        return f"LawSupervisedDataset(data_file={self.data_file}, input_ids_len={len(self.input_ids)}, labels_len={len(self.labels)})"

    def __str__(self):
        return f"LawSupervisedDataset(data_file={self.data_file}, input_ids_len={len(self.input_ids)}, labels_len={len(self.labels)})"


class EasyPromptsDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length: int = 96) -> None:
        super(EasyPromptsDataset, self).__init__()
        with open(data_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
            all_lines = [line if "回答：" not in line else line[: line.index("回答：") + 3] for line in all_lines]
        self.prompts = [
            tokenizer(line, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)[
                "input_ids"
            ]
            .to(torch.cuda.current_device())
            .squeeze(0)
            for line in tqdm(all_lines)
        ]
        self.data_file = data_file

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __repr__(self):
        return f"LawPromptsDataset(data_file={self.data_file}, prompts_len={len(self.prompts)})"

    def __str__(self):
        return f"LawPromptsDataset(data_file={self.data_file}, prompts_len={len(self.prompts)})"


class EasyRewardDataset(Dataset):
    def __init__(self, train_file: str, tokenizer: AutoTokenizer, special_token=None, max_length=512) -> None:
        super(EasyRewardDataset, self).__init__()
        self.chosen = []
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        print(self.end_token)
        # read all lines in the train_file to a list
        with open(train_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
        for line in tqdm(all_lines):
            data = json.loads(line)
            prompt = "提问：" + data["prompt"] + " 回答："

            chosen = prompt + data["chosen"] + self.end_token
            chosen_token = tokenizer(
                chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.chosen.append(
                {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}
            )

            reject = prompt + data["rejected"] + self.end_token
            reject_token = tokenizer(
                reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.reject.append(
                {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}
            )

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return (
            self.chosen[idx]["input_ids"],
            self.chosen[idx]["attention_mask"],
            self.reject[idx]["input_ids"],
            self.reject[idx]["attention_mask"],
        )

    # python representation of the object and the string representation of the object
    def __repr__(self):
        return f"LawRewardDataset(chosen_len={len(self.chosen)}, reject_len={len(self.reject)})"

    def __str__(self):
        return f"LawRewardDataset(chosen_len={len(self.chosen)}, reject_len={len(self.reject)})"


"""
Easy SFT just accept a text file which can be read line by line. However the datasets will group texts together to max_length so LLM will learn the texts meaning better.
If individual lines are not related, just set is_group_texts to False.
"""


class EasySFTDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length=512, is_group_texts=True) -> None:
        super().__init__()
        # read the data_file line by line
        with open(data_file, "r", encoding="UTF-8") as f:
            # encode the text data line by line and put raw python list input_ids only to raw_input_ids list
            raw_input_ids = []
            for line in f:
                encoded_ids = tokenizer.encode(line)
                # if the encoded_ids is longer than max_length, then split it into several parts
                if len(encoded_ids) > max_length:
                    for i in range(0, len(encoded_ids), max_length):
                        raw_input_ids.append(encoded_ids[i : i + max_length])
                else:
                    raw_input_ids.append(encoded_ids)

        grouped_input_ids = []
        current_input_ids = []
        attention_mask = []
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if is_group_texts:
            for input_ids in raw_input_ids:
                if len(current_input_ids) + len(input_ids) > max_length:
                    # pad the current_input_ids to max_length with tokenizer.pad_token_id
                    padded_length = max_length - len(current_input_ids)
                    current_input_ids.extend([tokenizer.pad_token_id] * padded_length)
                    grouped_input_ids.append(torch.tensor(current_input_ids, dtype=torch.long))
                    attention_mask.append(
                        torch.tensor([1] * (max_length - padded_length) + [0] * padded_length, dtype=torch.long)
                    )
                    current_input_ids = []
                else:
                    current_input_ids.extend(input_ids)
            if len(current_input_ids) > 0:
                padded_length = max_length - len(current_input_ids)
                current_input_ids.extend([tokenizer.pad_token_id] * padded_length)
                grouped_input_ids.append(torch.tensor(current_input_ids, dtype=torch.long))
                attention_mask.append(
                    torch.tensor([1] * (max_length - padded_length) + [0] * padded_length, dtype=torch.long)
                )
        else:
            # just append the raw_input_ids to max_length
            for input_ids in raw_input_ids:
                padded_length = max_length - len(input_ids)
                input_ids.extend([tokenizer.pad_token_id] * padded_length)
                attention_mask.append(
                    torch.tensor([1] * (max_length - padded_length) + [0] * padded_length, dtype=torch.long)
                )
                grouped_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        self.input_ids = grouped_input_ids
        self.labels = copy.deepcopy(self.input_ids)
        self.file_name = data_file
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    # get item from dataset
    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], attention_mask=self.attention_mask[idx])

    # generate the dataset description to be printed by print in python
    def __repr__(self):
        return f"EasySFTDataset(len={len(self)},\nfile_name is {self.file_name})"

    # generate the dataset description to be printed by print in python
    def __str__(self):
        return f"EasySFTDataset(len={len(self)},\nfile_name is {self.file_name})"
