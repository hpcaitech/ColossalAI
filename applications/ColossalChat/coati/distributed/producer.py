from collections import defaultdict
import os
from typing import Any, Dict, Optional

import numpy as np
import ray
import ray.util.collective as cc
import torch
from coati.dataset.loader import RawConversationDataset
import wandb
from applications.ColossalChat.coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from applications.ColossalChat.build.lib.coati.models.utils import read_jsonl_file
from applications.ColossalChat.coati.dataset.loader import AIMEDataset
from colossalai.utils import get_current_device

from .comm import ray_broadcast_tensor_dict
from .inference_backend import BACKEND_MAP
from .utils import pre_send


class BaseProducer:
    def __init__(
        self,
        producer_idx: int,
        num_producers: int,
        num_consumer_procs: int,
        num_episodes: int,
        batch_size: int,
        dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer_config: Optional[Dict[str, Any]] = None,
        microbatch_size: int = 1,
        backend: str = "transformers",
    ):
        self.producer_idx = producer_idx
        self.num_producers = num_producers
        self.num_consumer_procs = num_consumer_procs
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        assert batch_size % microbatch_size == 0
        self.num_microbatches = batch_size // microbatch_size

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.generate_config = generate_config
        self.tokenizer_config = tokenizer_config

        # init tokenizer
        if tokenizer_config is None:
            tokenizer_path = model_config["path"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer_path = tokenizer_config.pop("path")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        self.tokenizer.padding_side = "left"

        # init dataloader
        dataset_path = dataset_config.pop("path")
        self.dataset = RawConversationDataset(self.tokenizer, dataset_path, **dataset_config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=microbatch_size,
            sampler=DistributedSampler(
                self.dataset,
                num_replicas=num_producers,
                rank=producer_idx,
                shuffle=True,
                drop_last=True,
                seed=42,
            ),
            num_workers=4,
        )
        self.device = get_current_device()

        # init backend
        if backend in BACKEND_MAP:
            self.backend_cls = BACKEND_MAP[backend]
        else:
            raise ValueError(f"Unexpected backend {backend}")

    def setup(self) -> None:
        cc.init_collective_group(1 + self.num_consumer_procs, 0, group_name=f"sync_data_{self.producer_idx}")
        cc.init_collective_group(self.num_producers + 1, self.producer_idx, group_name="sync_model")

    def rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def loop(self) -> None:
        num_update_per_episode = len(self.dataloader) // self.num_microbatches
        num_valid_microbatches = num_update_per_episode * self.num_microbatches
        print(
            f"[P{self.producer_idx}] num_valid_microbatches {num_valid_microbatches}, nmb: {self.num_microbatches}, dl: {len(self.dataloader)}"
        )
        for episode in range(self.num_episodes):
            self.dataloader.sampler.set_epoch(episode)
            for i, batch in enumerate(self.dataloader):
                valid_metrics = self.validate()
                
                if i >= num_valid_microbatches:
                    break
                outputs = self.rollout(**batch)
                outputs.update(valid_metrics)
                
                print(f"[P{self.producer_idx}] Send data {[(k, v.shape) for k, v in outputs.items()]}")
                outputs["temperature"] = torch.tensor(
                    [self.model.generate_config.temperature] * outputs["input_ids"].size(0)
                ).to(outputs["input_ids"].device)
                outputs = pre_send(outputs)
                ray_broadcast_tensor_dict(
                    outputs, src=0, device=self.device, group_name=f"sync_data_{self.producer_idx}"
                )

                if (i + 1) % self.num_microbatches == 0 and (
                    episode != self.num_episodes - 1 or i != num_valid_microbatches - 1
                ):
                    # don't sync model for last iteration
                    print(
                        f"[P{self.producer_idx}] Sync model episode {episode} step {(i + 1) // self.num_microbatches - 1}"
                    )

                    state_dict = ray_broadcast_tensor_dict(
                        None, self.num_producers, device=self.device, group_name="sync_model"
                    )
                    self.load_state_dict(state_dict)
                    del state_dict
                    torch.cuda.empty_cache()
                # linear annealing for 1 episode, temperature from initial to 0.7
                if episode <= 0:
                    ratio = 1 - (len(self.dataloader) - i) / len(self.dataloader)
                    self.model.generate_config.temperature = (1 - ratio) * self.generate_config[
                        "temperature"
                    ] + ratio * 0.7


@ray.remote
class SimpleProducer(BaseProducer):
    def __init__(
        self,
        producer_idx,
        num_producers,
        num_consumer_procs,
        num_episodes,
        batch_size,
        dataset_config,
        dataloaders_config,
        model_config,
        generate_config,
        tokenizer_config=None,
        microbatch_size=1,
        backend="transformers",
        num_generations: int = 8,
    ):
        super().__init__(
            producer_idx,
            num_producers,
            num_consumer_procs,
            num_episodes,
            batch_size,
            dataset_config,
            dataloaders_config,
            model_config,
            generate_config,
            tokenizer_config,
            microbatch_size,
            backend,
        )
        self.model = self.backend_cls(model_config, generate_config, self.tokenizer, num_generations)

    @torch.no_grad()
    def rollout(self, input_ids, attention_mask, **kwargs):
        rollouts = self.model.generate(input_ids, attention_mask, **kwargs)
        if self.producer_idx == 1:
            print("Rollout example:\n", self.tokenizer.decode(rollouts["input_ids"][0][0], skip_special_tokens=True))

        return rollouts

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def validate(self):
        all_rewards = []     
        all_formats = []
        all_accs = []
        batch_reward_means = []  
    
        self.val_dataset = AIMEDataset(
            tokenizer=self.tokenizer,
            input_file="/home/yanglibing/workspace/PRIME/eval/data/AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl",
            max_length=300,
        )
        # Initialize verifiable reward.
        response_format_tags = {
            "think_start": {"text": "<think>", "num_occur": 1},
            "think_end": {"text": "</think>", "num_occur": 1},
            "answer_start": {"text": "<answer>", "num_occur": 1},
            "answer_end": {"text": "</answer>", "num_occur": 1},
        }
        self.reward_model = VerifiableReward(
            reward_fns=[math_reward_fn], tokenizer=self.tokenizer, tags=response_format_tags
        )
        
        def collate_fn(data_list: list[dict]) -> dict: 
            tensors = defaultdict(list)
            non_tensors = defaultdict(list)

            for data in data_list:
                for key, val in data.items():
                    if isinstance(val, torch.Tensor):
                        tensors[key].append(val)
                    else:
                        non_tensors[key].append(val)

            for key, val in tensors.items():
                tensors[key] = torch.stack(val, dim=0)

            for key, val in non_tensors.items():
                non_tensors[key] = np.array(val, dtype=object)

            return {**tensors, **non_tensors}
        
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)
        
        all_rewards = torch.tensor([], device=self.device)
        all_formats = torch.tensor([], device=self.device)
        all_accs = torch.tensor([], device=self.device)
    
        for test_batch in self.val_dataloader:
            test_output =  self.rollout(**test_batch)
            num_generations = test_output["response_idx"].size(1)
            print("num_generations", num_generations)
            data = {k: v.view(-1, v.size(-1)) for k, v in test_output.items()}
            reward_group = self.reward_model(
                data["input_ids"], gt_answer=data["gt_answer"], response_idx=data["response_idx"])

            rewards = torch.stack([x[0] for x in reward_group])
            format_rewards = torch.stack([x[1] for x in reward_group])
            acc_rewards = torch.stack([x[2] for x in reward_group])
            
            all_rewards = torch.cat([all_rewards, rewards])
            all_formats = torch.cat([all_formats, format_rewards])
            all_accs = torch.cat([all_accs, acc_rewards])
        
        avg_reward = torch.mean(all_rewards)
        avg_format = torch.mean(all_formats)
        avg_acc = torch.mean(all_accs)
        
        valid_metrics = {
            "avg_reward": torch.tensor(avg_reward).unsqueeze(0),
            "avg_format": torch.tensor(avg_format).unsqueeze(0),
            "avg_acc": torch.tensor(avg_acc).unsqueeze(0),
        }
        print(
            f"[P{self.producer_idx}] Validation metrics: "
            f"reward={valid_metrics['avg_reward'].item():.4f}, "
            f"format={valid_metrics['avg_format'].item():.4f}, "
            f"acc={valid_metrics['avg_acc'].item():.4f}"
        )
        return valid_metrics
                    