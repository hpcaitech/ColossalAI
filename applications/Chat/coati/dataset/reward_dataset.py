import json
from typing import Callable

import torch
from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.llama import LlamaActor
from coati.models.opt import OPTActor
from coati.models.utils import calc_masked_log_probs
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0


# Dahoas/rm-static
class RmStaticDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token

        chosen = [data["prompt"] + data["chosen"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [data["prompt"] + data["rejected"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}

    def __len__(self):
        length = self.chosen["input_ids"].shape[0]
        return length

    def __getitem__(self, idx):
        return (
            self.chosen["input_ids"][idx],
            self.chosen["attention_mask"][idx],
            self.reject["input_ids"][idx],
            self.reject["attention_mask"][idx],
        )


# Anthropic/hh-rlhf
class HhRlhfDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token

        chosen = [data["chosen"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [data["rejected"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}

    def __len__(self):
        length = self.chosen["input_ids"].shape[0]
        return length

    def __getitem__(self, idx):
        return (
            self.chosen["input_ids"][idx],
            self.chosen["attention_mask"][idx],
            self.reject["input_ids"][idx],
            self.reject["attention_mask"][idx],
        )


# Anthropic/hh-rlhf
class HhRlhfDatasetDPO(Dataset):
    """
    Dataset for dpo training, in addition to standard reward dataset, it caches reward scores to
    save VRAM and timing

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(
        self,
        dataset,
        model,
        pretrain,
        dataset_cache_dir,
        strategy,
        tokenizer: Callable,
        max_length: int,
        special_token=None,
    ) -> None:
        super().__init__()
        # prefix = "You are an AI assistant, your answer honestly to help users. You should avoid any inaccurate, misleading information in your answer. Please carry on the following conversation."
        prefix = ""
        dataset = [i for i in dataset if i["chosen"].count("Human:") == 1]
        print(f"Length of dataset after filtering: {len(dataset)}")

        self.end_token = tokenizer.eos_token if special_token is None else special_token

        chosen = [prefix + data["chosen"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [prefix + data["rejected"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}

        if not "chosen_reward" in dataset[0]:
            print("caching reward...")
            with strategy.model_init_context():
                # configure model
                # TODO: add support for llama
                if model == "gpt2":
                    ref_model = GPTActor(pretrained=pretrain)
                elif model == "bloom":
                    ref_model = BLOOMActor(pretrained=pretrain)
                elif model == "opt":
                    ref_model = OPTActor(pretrained=pretrain)
                elif model == "llama":
                    # Note: llama disable dropout by default
                    ref_model = LlamaActor(pretrained=pretrain)
                else:
                    raise ValueError(f'Unsupported actor model "{model}"')

                ref_model.to(torch.bfloat16).to(torch.cuda.current_device())
            ref_model = strategy.prepare(ref_model)
            ref_model.eval()
            self.chosen["reward"] = []
            self.reject["reward"] = []
            with torch.no_grad():
                for i in tqdm(range(int(len(self.chosen["input_ids"]) / 5)), disable=not is_rank_0()):
                    chosen_input_ids = torch.stack(
                        [self.chosen["input_ids"][i * 5 + j].clone().to(torch.cuda.current_device()) for j in range(5)]
                    )
                    reject_input_ids = torch.stack(
                        [self.reject["input_ids"][i * 5 + j].clone().to(torch.cuda.current_device()) for j in range(5)]
                    )
                    chosen_mask = torch.stack(
                        [
                            self.chosen["attention_mask"][i * 5 + j].clone().to(torch.cuda.current_device())
                            for j in range(5)
                        ]
                    )
                    reject_mask = torch.stack(
                        [
                            self.reject["attention_mask"][i * 5 + j].clone().to(torch.cuda.current_device())
                            for j in range(5)
                        ]
                    )
                    chosen_atten_mask = torch.stack(
                        [
                            self.chosen["attention_mask"][i * 5 + j].clone().to(torch.cuda.current_device())
                            for j in range(5)
                        ]
                    )
                    reject_atten_mask = torch.stack(
                        [
                            self.reject["attention_mask"][i * 5 + j].clone().to(torch.cuda.current_device())
                            for j in range(5)
                        ]
                    )
                    first_diff_position = torch.argmax((chosen_input_ids != reject_input_ids).float(), dim=1)
                    for j in range(chosen_input_ids.shape[0]):
                        chosen_mask[j, : first_diff_position[j]] = False
                        reject_mask[j, : first_diff_position[j]] = False
                    ref_all_logits = ref_model(
                        torch.cat([chosen_input_ids, reject_input_ids]),
                        torch.cat([chosen_atten_mask, reject_atten_mask]),
                    )["logits"].to(torch.float32)
                    ref_chosen_logits = ref_all_logits[:5]
                    ref_reject_logits = ref_all_logits[5:]
                    # print(chosen_mask)
                    logprob_ref_chosen = calc_masked_log_probs(
                        ref_chosen_logits, chosen_input_ids, chosen_mask[:, 1:]
                    ).sum(-1)
                    logprob_ref_reject = calc_masked_log_probs(
                        ref_reject_logits, reject_input_ids, reject_mask[:, 1:]
                    ).sum(-1)
                    for j in range(5):
                        self.chosen["reward"].append(logprob_ref_chosen[j : j + 1].cpu().clone())
                        self.reject["reward"].append(logprob_ref_reject[j : j + 1].cpu().clone())
            ref_model.to("cpu")
            del ref_model
            if is_rank_0() and dataset_cache_dir is not None:
                with open(dataset_cache_dir, "w", encoding="utf8") as f:
                    for idx, line in tqdm(enumerate(dataset), disable=not is_rank_0()):
                        if idx == len(self.chosen["reward"]):
                            break
                        line["chosen_reward"] = self.chosen["reward"][idx].item()
                        line["reject_reward"] = self.reject["reward"][idx].item()
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.chosen["input_ids"] = self.chosen["input_ids"][: len(self.chosen["reward"])]
            self.chosen["attention_mask"] = self.chosen["attention_mask"][: len(self.chosen["reward"])]
            self.reject["input_ids"] = self.reject["input_ids"][: len(self.reject["reward"])]
            self.reject["attention_mask"] = self.reject["attention_mask"][: len(self.reject["reward"])]
        else:
            self.chosen["reward"] = [torch.tensor([data["chosen_reward"]]) for data in dataset]
            self.reject["reward"] = [torch.tensor([data["reject_reward"]]) for data in dataset]

    def __len__(self):
        length = self.chosen["input_ids"].shape[0]
        return length

    def __getitem__(self, idx):
        return (
            self.chosen["input_ids"][idx],
            self.chosen["attention_mask"][idx],
            self.chosen["reward"][idx],
            self.reject["input_ids"][idx],
            self.reject["attention_mask"][idx],
            self.reject["reward"][idx],
        )
