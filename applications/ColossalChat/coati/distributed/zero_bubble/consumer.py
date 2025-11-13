import os
import threading
import time
from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
import torch.distributed as dist
from coati.distributed.comm import SharedVariableActor, ray_broadcast_tensor_dict
from coati.distributed.profiling_utils import CustomProfiler
from coati.distributed.utils import bind_batch, post_recv, unbind_batch
from tqdm import tqdm

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.initialize import launch
from colossalai.utils import get_current_device


class BaseConsumer:
    def __init__(
        self,
        shared_sync_data_actor: SharedVariableActor,
        shared_signal_actor: SharedVariableActor,
        num_producers: int,
        num_episodes: int,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        train_dataset_size: int,
        batch_size: int,
        model_config: Dict[str, Any],
        plugin_config: Dict[str, Any],
        minibatch_size: int = 1,
        save_interval: int = 100,
        save_dir: str = "./model",
        enable_profiling: bool = False,
    ):
        self.num_producers = num_producers
        self.num_episodes = num_episodes
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.train_dataset_size = train_dataset_size
        self.received_prompts = 0
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.enable_profiling = enable_profiling
        assert batch_size % minibatch_size == 0, "batch_size should be divisible by microbatch_size"
        self.num_microbatches = batch_size // minibatch_size
        self.data_uid = 0
        self.sync_model_thread_started = False

        self.model_config = model_config
        self.plugin_config = plugin_config

        self.device = get_current_device()
        self.lr_scheduler = None

        self.shared_sync_data_actor = shared_sync_data_actor
        self.shared_signal_actor = shared_signal_actor
        self.state_dict_cpu = {}

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

        self.buffer = []
        self.recv_cnt = 0
        self.profiler = CustomProfiler(f"C{self.rank}", disabled=not self.enable_profiling)

    def get_ddp_config(self) -> Dict[str, Any]:
        """
        Get the DDP configuration for the consumer.
        This method is used to get the DDP configuration for the consumer.
        """
        return {
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def init_collective_group(
        self,
        world_size: int,
        rank: int,
        backend: str = "nccl",
        group_name: str = "default",
        gloo_timeout: int = 3000000,
    ):
        cc.init_collective_group(
            world_size=world_size, rank=rank, backend=backend, group_name=group_name, gloo_timeout=gloo_timeout
        )
        print(f"[C{self.rank}] Initialized {group_name} collective group", flush=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def step(self, **kwargs) -> Optional[float]:
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

    def calculate_effective_group_to_raw_group_mapping(self):
        effective_group_to_raw_group_mapping = {}
        for buffer_idx in range(len(self.buffer)):
            if self.buffer[buffer_idx][0] is not None:
                effective_group_to_raw_group_mapping[len(effective_group_to_raw_group_mapping)] = buffer_idx
        return effective_group_to_raw_group_mapping

    def loop(self) -> None:
        print(f"Consumer{self.rank}, nmb: {self.num_microbatches}")
        for episode in range(self.num_episodes):
            with tqdm(
                range(self.train_dataset_size),
                desc=f"Episode {episode} with rollout step(s)",
                disable=self.rank != 0,
            ) as pbar:
                while self.received_prompts < self.train_dataset_size:
                    torch.cuda.reset_peak_memory_stats()
                    effective_group_to_raw_group_mapping = {}
                    self.profiler.enter(f"recv_data")
                    while len(effective_group_to_raw_group_mapping) < self.dp_size * self.minibatch_size:
                        # receive data from producers
                        raw_batch = ray.get(
                            self.shared_sync_data_actor.get_data.remote(self.data_uid)
                        )  # get the first queued data
                        self.profiler.log(f"enter sleep")
                        while raw_batch is None:
                            print(
                                f"[T{dist.get_rank()}] No data received by consumer {self.rank}, skipping. Consider increasing the data actor buffer limit"
                            )
                            time.sleep(1)
                            raw_batch = ray.get(self.shared_sync_data_actor.get_data.remote(self.data_uid))
                            continue
                        self.profiler.log(f"exit sleep")
                        self.data_uid += 1
                        raw_batch = {k: v.to(self.device) for k, v in raw_batch.items()}
                        # calculate group reward et al. filtering. As only the filtered group will be used for training (which is incomplete),
                        # we need to calculate the metrics before filtering here for logging
                        # [batch_size, num_generations] -> [batch_size]
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
                        self.received_prompts += len(raw_batch)
                        pbar.update(len(raw_batch))
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
                                ]
                            )
                        if effective_group_mask is not None:
                            print(
                                f"[T{dist.get_rank()}] Filter recv data: {len(raw_batch)} -> {torch.sum(effective_group_mask).cpu().item()} effective groups"
                            )
                        # mapping the effective group to the raw group for indexing
                        effective_group_to_raw_group_mapping = self.calculate_effective_group_to_raw_group_mapping()
                        print(
                            f"[T{dist.get_rank()}] Collect Effective Prompt: {len(effective_group_to_raw_group_mapping)}/{self.dp_size * self.minibatch_size}"
                        )
                    self.profiler.exit(f"recv_data")
                    need_sync_model = False
                    while len(effective_group_to_raw_group_mapping) >= self.dp_size * self.minibatch_size:
                        # after we have enough effective groups, we can start training
                        # on each dp_rank, we use minibatch_size effective samples to form a batch
                        batch, raw_mini_batches_metric_dict = self.prepare_mini_batch(
                            effective_group_to_raw_group_mapping
                        )
                        self.profiler.enter("step")
                        loss = self.step(pbar, **batch, **raw_mini_batches_metric_dict)
                        self.profiler.exit("step")
                        self.buffer = self.buffer[
                            effective_group_to_raw_group_mapping[self.dp_size * self.minibatch_size - 1] + 1 :
                        ]
                        # recalculate the effective group to raw group mapping
                        effective_group_to_raw_group_mapping_size_before = len(effective_group_to_raw_group_mapping)
                        effective_group_to_raw_group_mapping = self.calculate_effective_group_to_raw_group_mapping()
                        assert (
                            len(effective_group_to_raw_group_mapping)
                            == effective_group_to_raw_group_mapping_size_before - self.dp_size * self.minibatch_size
                        )
                        # cc.barrier(group_name="consumer_pg")
                        if loss is not None:
                            pbar.set_postfix({"loss": loss})
                            need_sync_model = True
                            ray.get(self.shared_signal_actor.set_signal.remote("global_step", self.global_step + 1))
                    if need_sync_model and (
                        (self.global_step + 1) % self.save_interval == 0
                        or self.received_prompts >= self.train_dataset_size
                    ):
                        if self.rank == 0:
                            print(f"Start saving policy model at step {self.global_step + 1}.")
                        save_path = os.path.join(
                            self.save_dir, f"modeling-episode-{episode}-step-{self.global_step + 1}"
                        )
                        self.booster.save_model(self.policy_model, save_path, shard=True)
                        if self.rank == 0:
                            print(f"Saved model checkpoint at step {self.global_step + 1} in folder {save_path}")

                    if need_sync_model and (
                        episode != self.num_episodes - 1 or self.received_prompts != self.train_dataset_size
                    ):

                        def sync_model_thread():
                            # sync model weights to all producers, if no model update or it is the last training step, skip syncing
                            if self.pp_size > 1:
                                print(
                                    f"[T{dist.get_rank()}] Sync model PP stage {self.pp_rank} episode {episode} step {self.global_step}"
                                )
                            else:
                                print(f"[T{dist.get_rank()}] Sync model episode {episode} step {self.global_step}")
                            torch.cuda.empty_cache()
                            if self.pp_size > 1:
                                if self.tp_rank == 0 and self.dp_rank == 0:
                                    self.profiler.enter("sync_model")
                                    ray.get(
                                        self.shared_signal_actor.set_signal.remote(
                                            f"consumer_pp_{self.pp_rank}", "ready_sync_model"
                                        )
                                    )
                                    print(
                                        f"[T{dist.get_rank()}] Sync model PP stage {self.pp_rank} episode {episode} step {self.global_step}"
                                    )
                                    ray_broadcast_tensor_dict(
                                        self.state_dict_cpu,
                                        src=0,
                                        device=torch.device("cpu"),
                                        group_name=f"sync_model_consumer_pp_{self.pp_rank}",
                                        backend="gloo",
                                    )
                                    self.profiler.exit("sync_model")
                            else:
                                if self.rank == 0:
                                    self.profiler.enter("sync_model")
                                    ray.get(self.shared_signal_actor.set_signal.remote("consumer", "ready_sync_model"))
                                    print(f"[T{dist.get_rank()}] Sync model episode {episode} step {self.global_step}")
                                    ray_broadcast_tensor_dict(
                                        self.state_dict_cpu,
                                        src=0,
                                        device=torch.device("cpu"),
                                        group_name="sync_model_consumer",
                                        backend="gloo",
                                    )
                                    self.profiler.exit("sync_model")

                        if not self.sync_model_thread_started:
                            # only sync model when the thread is not started and no other thread is broadcasting
                            self.sync_model_thread_started = True
                            state_dict_ = self.state_dict()
                            if (self.pp_size > 1 and self.tp_rank == 0 and self.dp_rank == 0) or (
                                self.pp_size == 1 and self.rank == 0
                            ):
                                if len(self.state_dict_cpu) == 0:
                                    # use pinned memory to speed up the transfer
                                    self.state_dict_cpu = {k: v.cpu().pin_memory() for k, v in state_dict_.items()}
                                    torch.cuda.synchronize()
                                for k, v in state_dict_.items():
                                    self.state_dict_cpu[k].copy_(v, non_blocking=True)
                                torch.cuda.synchronize()
                            cc.barrier(
                                group_name="consumer_pg"
                            )  # to make sure all ranks have state dict offloaded to CPU before starting the thread
                            time_before_starting_thread = time.time()
                            threading.Thread(target=sync_model_thread).start()
                            # sync_model_thread()
                            self.profiler.log(
                                f"Sync model, took {time.time() - time_before_starting_thread:.2f} seconds"
                            )
                            self.sync_model_thread_started = False
                            # ray.get(self.shared_signal_actor.release_process_lock.remote("broadcasting_lock"))
                    self.profiler.log(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
                self.received_prompts = 0
        ray.get(self.shared_signal_actor.set_signal.remote("consumer", "terminate"))

    def __del__(self):
        if hasattr(self, "profiler"):
            self.profiler.close()
