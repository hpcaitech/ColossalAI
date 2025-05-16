import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.initialize import launch
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from .comm import ray_broadcast_tensor_dict
from .utils import bind_batch, post_recv, unbind_batch


class BaseConsumer:
    def __init__(
        self,
        num_producers: int,
        num_episodes: int,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        num_update_per_episode: int,
        num_recv_per_update: int,
        batch_size: int,
        model_config: Dict[str, Any],
        plugin_config: Dict[str, Any],
        minibatch_size: int = 1,
        save_interval: int = 100,
        save_dir: str = "./model",
    ):
        self.num_producers = num_producers
        self.num_episodes = num_episodes
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.num_update_per_episode = num_update_per_episode
        self.num_recv_per_update = num_recv_per_update
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.save_interval = save_interval
        self.save_dir = save_dir
        assert batch_size % minibatch_size == 0, "batch_size should be divisible by microbatch_size"
        self.num_microbatches = batch_size // minibatch_size

        self.model_config = model_config
        self.plugin_config = plugin_config

        self.device = get_current_device()
        self.lr_scheduler = None

    def setup(self) -> None:
        launch(self.rank, self.world_size, self.master_addr, self.master_port, local_rank=0)

        plugin_config = dict(tp_size=1, pp_size=1, precision="bf16", zero_stage=2)
        if (
            self.plugin_config.get("pp_size", 1) > 1
            and "num_microbatches" not in self.plugin_config
            and "microbatch_size" not in self.plugin_config
        ):
            plugin_config["microbatch_size"] = max(1, self.minibatch_size // plugin_config.get("pp_size", 1))
        plugin_config.update(self.plugin_config)
        self.plugin = HybridParallelPlugin(**plugin_config)
        self.booster = Booster(plugin=self.plugin)
        self.dp_rank = dist.get_rank(self.plugin.dp_group)
        self.tp_rank = dist.get_rank(self.plugin.tp_group)
        self.pp_rank = dist.get_rank(self.plugin.pp_group)

        self.dp_size = dist.get_world_size(self.plugin.dp_group)
        self.tp_size = dist.get_world_size(self.plugin.tp_group)
        self.pp_size = dist.get_world_size(self.plugin.pp_group)

        # Init Hybrid ray process group
        for i in range(self.num_producers):
            cc.init_collective_group(self.world_size + 1, self.rank + 1, group_name=f"sync_data_{i}")
        if self.pp_size > 1:
            # use hybrid tp + pp
            if self.tp_rank == 0 and self.dp_rank == 0:
                cc.init_collective_group(
                    self.num_producers + 1, self.num_producers, group_name=f"sync_model_{self.pp_rank}"
                )
        else:
            if self.rank == 0:
                cc.init_collective_group(self.num_producers + 1, self.num_producers, group_name="sync_model")

        self.buffer = []
        self.recv_cnt = 0

    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def step(self, step_idx: int, **kwargs) -> Optional[float]:
        raise NotImplementedError

    def loop(self) -> None:
        print(
            f"Consumer{self.rank} num_update: {self.num_update_per_episode}, num_recv: {self.num_recv_per_update}, nmb: {self.num_microbatches}"
        )
        for episode in range(self.num_episodes):
            with tqdm(
                range(self.num_update_per_episode),
                desc=f"Episode {episode} with rollout step(s)",
                disable=self.rank != 0,
            ) as pbar:
                for step in pbar:
                    i = 0
                    allow_sync_model = True
                    for _ in range(self.num_recv_per_update):
                        # receive data from producers
                        for r in range(self.num_producers):
                            print(f"[T{dist.get_rank()}] Recv data episode {episode} step {step} from {r}")
                            raw_batch = unbind_batch(
                                ray_broadcast_tensor_dict(None, src=0, device=self.device, group_name=f"sync_data_{r}")
                            )
                            processed_batch = [
                                self.prompt_level_filtering(self.calculate_group_reward(group)) for group in raw_batch
                            ]
                            filtered_batch = [t for t in processed_batch if t is not None]
                            if self.filter_range is not None:
                                print(
                                    f"[T{dist.get_rank()}] Filter recv data: {len(processed_batch)} -> {len(filtered_batch)}"
                                )

                            self.buffer.extend(filtered_batch)
                        while len(self.buffer) >= self.dp_size * self.minibatch_size:
                            batches = self.buffer[
                                self.dp_rank * self.minibatch_size : (self.dp_rank + 1) * self.minibatch_size
                            ]
                            batch = bind_batch(batches)
                            batch = post_recv(batch)
                            loss = self.step(i, pbar, **batch)
                            self.buffer = self.buffer[self.dp_size * self.minibatch_size :]
                            if loss is not None:
                                allow_sync_model = True
                                pbar.set_postfix({"loss": loss})
                            i += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if (step + 1) % self.save_interval == 0 or (step + 1) == self.num_update_per_episode:
                        if self.rank == 0:
                            print(f"Start saving policy model at step {step + 1}.")
                        save_path = os.path.join(self.save_dir, f"modeling-episode-{episode}-step-{step + 1}")
                        self.booster.save_model(self.policy_model, save_path, shard=True)
                        if self.rank == 0:
                            print(f"Saved model checkpoint at step {step + 1} in folder {save_path}")

                    if episode != self.num_episodes - 1 or step != self.num_update_per_episode - 1:
                        if allow_sync_model:
                            if self.pp_size > 1:
                                print(
                                    f"[T{dist.get_rank()}] Sync model PP stage {self.pp_rank} episode {episode} step {step}"
                                )
                            else:
                                print(f"[T{dist.get_rank()}] Sync model episode {episode} step {step}")
                            torch.cuda.empty_cache()
                            state_dict = self.state_dict()
                            if self.pp_size > 1:
                                if self.tp_rank == 0 and self.dp_rank == 0:
                                    ray_broadcast_tensor_dict(
                                        state_dict,
                                        src=self.num_producers,
                                        device=self.device,
                                        group_name=f"sync_model_{self.pp_rank}",
                                    )
                            else:
                                if self.rank == 0:
                                    ray_broadcast_tensor_dict(
                                        state_dict, src=self.num_producers, device=self.device, group_name="sync_model"
                                    )
                            del state_dict
                            torch.cuda.empty_cache()
                            allow_sync_model = True


@ray.remote
class SimpleConsumer(BaseConsumer):
    def __init__(
        self,
        num_producers,
        num_episodes,
        rank,
        world_size,
        master_addr,
        master_port,
        num_update_per_episode,
        num_recv_per_update,
        batch_size,
        model_config,
        plugin_config,
        minibatch_size=1,
        save_interval: int = 100,
        save_dir="./model",
    ):
        super().__init__(
            num_producers,
            num_episodes,
            rank,
            world_size,
            master_addr,
            master_port,
            num_update_per_episode,
            num_recv_per_update,
            batch_size,
            model_config,
            plugin_config,
            minibatch_size,
            save_interval,
            save_dir,
        )
        path = model_config.pop("path")
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.optimizer = HybridAdam(self.model.parameters(), lr=1e-3)
        self.accum_loss = torch.zeros(1, device=self.device)

    def setup(self):
        super().setup()
        self.model, self.optimizer, *_ = self.booster.boost(self.model, self.optimizer)

    def step(self, step_idx: int, pbar: Any, **kwargs) -> Optional[float]:
        labels = kwargs["input_ids"].clone()
        labels[kwargs["attention_mask"] == 0] = -100
        kwargs["labels"] = labels
        assert kwargs.pop("action_mask").shape == kwargs.pop("action_log_probs").shape

        need_update = (step_idx + 1) % self.num_microbatches == 0

        ctx = nullcontext() if need_update else self.booster.no_sync(self.model, self.optimizer)
        with ctx:
            out = self.model(**kwargs)
            loss = out.loss / self.num_microbatches
            self.accum_loss.add_(loss.data)
            self.booster.backward(loss, self.optimizer)
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_scalar = self.accum_loss.item()
            self.accum_loss.zero_()
            return loss_scalar

    def state_dict(self):
        self.model._force_wait_all_gather()
        model = self.model.unwrap()
        state_dict = model.state_dict()
        return state_dict
