from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
from coati.dataset.loader import RawConversationDataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

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
                if i >= num_valid_microbatches:
                    break
                outputs = self.rollout(**batch)

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
                # linear annealing for 1 episode, temperature from initial to 0.9
                if episode <= 0:
                    ratio = 1 - (len(self.dataloader) - i) / len(self.dataloader)
                    self.model.generate_config.temperature = (1 - ratio) * self.generate_config[
                        "temperature"
                    ] + ratio * 0.9


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
