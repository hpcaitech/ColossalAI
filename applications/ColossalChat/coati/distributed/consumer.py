from contextlib import nullcontext
from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
import torch.distributed as dist
from coati.distributed.profiling_utils import CustomProfiler
from coati.utils import save_checkpoint
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
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
        enable_profiling: bool = False,
        n_behind: int = 0,
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
        self.enable_profiling = enable_profiling
        assert batch_size % minibatch_size == 0, "batch_size should be divisible by microbatch_size"
        self.num_microbatches = batch_size // minibatch_size
        self.checkpoint_path = model_config.pop("checkpoint_path", None)

        self.model_config = model_config
        self.plugin_config = plugin_config

        self.device = get_current_device()
        self.lr_scheduler = None
        self.n_behind = n_behind
        self.total_prompt_trained = 0  # for setting start index when resume training

    def setup(self) -> None:
        launch(self.rank, self.world_size, self.master_addr, self.master_port, local_rank=0)
        self.coordinator = DistCoordinator()

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
        self.profiler = CustomProfiler(f"C{self.rank}", disabled=not self.enable_profiling)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def step(self, step_idx: int, **kwargs) -> Optional[float]:
        raise NotImplementedError

    def prepare_mini_batch(self, effective_group_to_raw_group_mapping: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """
        Prepare a mini-batch from the effective group to raw group mapping.
        This method is used to create a mini-batch for training.
        """
        batches = [
            self.buffer[effective_group_to_raw_group_mapping[i]]
            for i in range(self.dp_rank * self.minibatch_size, (self.dp_rank + 1) * self.minibatch_size)
        ]
        # every dp_rank will receive a complete mini-batch, no need to sync within step() later
        # each mini-batch use the first self.dp_size * minibatch_size effective samples
        raw_mini_batches = self.buffer[
            : effective_group_to_raw_group_mapping[self.dp_size * self.minibatch_size - 1] + 1
        ]  # include the last effective sample
        raw_mini_batches_metric_dict = {
            "raw_train_mini_batch_reward": [t[1] for t in raw_mini_batches],
            "raw_train_mini_batch_format_acc": [t[2] for t in raw_mini_batches],
            "raw_train_mini_batch_ans_acc": [t[3] for t in raw_mini_batches],
            "raw_train_mini_batch_response_len": [t[4] for t in raw_mini_batches],
        }
        batch = bind_batch([t[0] for t in batches])
        batch = post_recv(batch)
        return batch, raw_mini_batches_metric_dict

    def calculate_effective_group_to_raw_group_mapping(self, step):
        effective_group_to_raw_group_mapping = {}
        for buffer_idx in range(len(self.buffer)):
            if self.buffer[buffer_idx][0] is not None:
                if self.n_behind == 0:
                    effective_group_to_raw_group_mapping[len(effective_group_to_raw_group_mapping)] = buffer_idx
                else:
                    if self.buffer[buffer_idx][-1] <= step - self.n_behind:
                        effective_group_to_raw_group_mapping[len(effective_group_to_raw_group_mapping)] = buffer_idx
        return effective_group_to_raw_group_mapping

    def loop(self) -> None:
        self.profiler.enter("sync_model")
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
        self.profiler.exit("sync_model")

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
                    torch.cuda.reset_peak_memory_stats()
                    i = 0

                    self.profiler.enter(f"rollout_episode_{episode}_step_{step}")
                    for _ in range(self.num_recv_per_update):
                        if self.n_behind > 0:
                            # after sync model, do not wait for more data to arrive as rollout takes time, use buffered data
                            effective_group_to_raw_group_mapping = self.calculate_effective_group_to_raw_group_mapping(
                                step=step
                            )
                            while len(effective_group_to_raw_group_mapping) >= self.dp_size * self.minibatch_size:
                                self.profiler.log(
                                    f"Still have {len(effective_group_to_raw_group_mapping)} effective groups, greater than {self.dp_size * self.minibatch_size}, start training"
                                )
                                batch, raw_mini_batches_metric_dict = self.prepare_mini_batch(
                                    effective_group_to_raw_group_mapping
                                )
                                self.profiler.enter("step")
                                loss = self.step(i, pbar, **batch, **raw_mini_batches_metric_dict)
                                self.profiler.exit("step")
                                self.buffer = self.buffer[
                                    effective_group_to_raw_group_mapping[self.dp_size * self.minibatch_size - 1] + 1 :
                                ]
                                # recalculate the effective group to raw group mapping
                                effective_group_to_raw_group_mapping_size_before = len(
                                    effective_group_to_raw_group_mapping
                                )
                                effective_group_to_raw_group_mapping = (
                                    self.calculate_effective_group_to_raw_group_mapping(step=step)
                                )
                                assert (
                                    len(effective_group_to_raw_group_mapping)
                                    == effective_group_to_raw_group_mapping_size_before
                                    - self.dp_size * self.minibatch_size
                                )
                                if loss is not None:
                                    pbar.set_postfix({"loss": loss})
                                i += 1

                        # receive data from producers
                        for r in range(self.num_producers):
                            print(f"[T{dist.get_rank()}] Recv data episode {episode} step {step} from {r}")
                            self.profiler.enter(f"recv_broadcast_data_P{r}")
                            raw_batch = ray_broadcast_tensor_dict(
                                None, src=0, device=self.device, group_name=f"sync_data_{r}"
                            )
                            self.profiler.exit(f"recv_broadcast_data_P{r}")
                            # calculate group reward et al. filtering. As only the filtered group will be used for training (which is incomplete),
                            # we need to calculate the metrics before filtering here for logging
                            # [batch_size, num_generations, ...] -> [batch_size * num_generations, ...]
                            raw_batch = {
                                k: v.view(-1, self.num_generations, v.size(-1)) if k != "temperature" else v
                                for k, v in raw_batch.items()
                            }
                            # [batch_size, num_generations] -> [batch_size]
                            self.total_prompt_trained += raw_batch["reward"].size(0)
                            reward = raw_batch["reward"][:, :, 0]
                            format_acc = raw_batch["format_acc"][:, :, 0]
                            ans_acc = raw_batch["ans_acc"][:, :, 0]
                            response_len = (
                                raw_batch["response_idx"][:, :, 1] - raw_batch["response_idx"][:, :, 0] + 1
                            ).type(torch.float32)
                            effective_group_mask = None
                            if self.filter_range is not None and self.grpo_config.get("dynamic_batching", True):
                                # filter the group based on the reward and accuracy
                                group_ans_acc_mean = ans_acc.mean(dim=1)
                                effective_group_mask = torch.logical_and(
                                    group_ans_acc_mean > self.filter_range[0], group_ans_acc_mean < self.filter_range[1]
                                )
                            raw_batch = unbind_batch(raw_batch)  # List[Dict[str, torch.Tensor]]
                            for group_idx, group_with_reward in enumerate(raw_batch):
                                self.buffer.append(
                                    [
                                        (
                                            group_with_reward
                                            if effective_group_mask is None or effective_group_mask[group_idx]
                                            else None
                                        ),
                                        reward[group_idx],
                                        format_acc[group_idx],
                                        ans_acc[group_idx],
                                        response_len[group_idx],
                                        step,
                                    ]
                                )
                            if effective_group_mask is not None:
                                print(
                                    f"[T{dist.get_rank()}] Filter recv data: {len(raw_batch)} -> {torch.sum(effective_group_mask).cpu().item()} effective groups"
                                )
                        # mapping the effective group to the raw group for indexing
                        effective_group_to_raw_group_mapping = self.calculate_effective_group_to_raw_group_mapping(
                            step=step
                        )
                        print(
                            f"[T{dist.get_rank()}] Collect Effective Prompt: {len(effective_group_to_raw_group_mapping)}/{self.dp_size * self.minibatch_size}"
                        )

                        if self.n_behind == 0:
                            # If n_behind is 0, we start training after receiving data from producers.
                            while len(effective_group_to_raw_group_mapping) >= self.dp_size * self.minibatch_size:
                                self.profiler.log(
                                    f"Collect {len(effective_group_to_raw_group_mapping)} effective groups, greater than {self.dp_size * self.minibatch_size}, start training"
                                )
                                batch, raw_mini_batches_metric_dict = self.prepare_mini_batch(
                                    effective_group_to_raw_group_mapping
                                )
                                self.profiler.enter("step")
                                loss = self.step(i, pbar, **batch, **raw_mini_batches_metric_dict)
                                self.profiler.exit("step")
                                self.buffer = self.buffer[
                                    effective_group_to_raw_group_mapping[self.dp_size * self.minibatch_size - 1] + 1 :
                                ]
                                # recalculate the effective group to raw group mapping
                                effective_group_to_raw_group_mapping_size_before = len(
                                    effective_group_to_raw_group_mapping
                                )
                                effective_group_to_raw_group_mapping = (
                                    self.calculate_effective_group_to_raw_group_mapping(step=step)
                                )
                                assert (
                                    len(effective_group_to_raw_group_mapping)
                                    == effective_group_to_raw_group_mapping_size_before
                                    - self.dp_size * self.minibatch_size
                                )
                                if loss is not None:
                                    pbar.set_postfix({"loss": loss})
                                i += 1

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if (step + 1) % self.save_interval == 0 or (step + 1) == self.num_update_per_episode:
                        if self.rank == 0:
                            print(f"Start saving policy model at step {step + 1}.")
                        save_checkpoint(
                            save_dir=self.save_dir,
                            booster=self.booster,
                            model=self.policy_model,
                            optimizer=self.optimizer,
                            lr_scheduler=self.lr_scheduler,
                            epoch=episode,
                            step=step,
                            batch_size=int(self.total_prompt_trained / step),
                            coordinator=self.coordinator,
                        )  # for setting start index when resuming training
                        if self.rank == 0:
                            print(f"Saved model checkpoint at step {step + 1} in folder {self.save_dir}")

                    if (episode != self.num_episodes - 1 or step != self.num_update_per_episode - 1) and (
                        episode != 0 or step >= self.n_behind
                    ):
                        if self.pp_size > 1:
                            print(
                                f"[T{dist.get_rank()}] Sync model PP stage {self.pp_rank} episode {episode} step {step}"
                            )
                        else:
                            print(f"[T{dist.get_rank()}] Sync model episode {episode} step {step}")
                        self.profiler.enter("sync_model")
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
                        self.profiler.exit("sync_model")
                    self.profiler.log(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
                    self.profiler.exit(f"rollout_episode_{episode}_step_{step}")

    def __del__(self):
        if hasattr(self, "profiler"):
            self.profiler.close()


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
        self.optimizer = HybridAdam(self.model.parameters(), lr=1e-3, weight_decay=0.01)
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
