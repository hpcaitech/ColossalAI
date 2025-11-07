import time

import ray
import ray.util.collective as cc
import torch
from coati.distributed.comm import SharedVariableActor, ray_broadcast_tensor_dict
from coati.distributed.profiling_utils import CustomProfiler

from colossalai.utils import get_current_device


@ray.remote
class Distributor:
    def __init__(
        self,
        distributor_id,
        consumer_pp_size,
        num_producers,
        shared_signal_actor: SharedVariableActor,
        enable_profiling: bool = True,
    ):
        self.distributor_id = distributor_id
        self.weight_version = [0] * consumer_pp_size
        self.consumer_pp_size = consumer_pp_size
        self.state_dict_cpu = {}
        self.num_producers = num_producers
        self.shared_signal_actor = shared_signal_actor
        self.device = get_current_device()
        self.profiler = CustomProfiler(f"D{self.distributor_id}", disabled=not enable_profiling)

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
        print(f"[D] Initialized {group_name} collective group", flush=True)

    def loop(self):
        last_weight_version = self.get_weight_version()
        while True:
            time.sleep(1)
            signal = ray.get(self.shared_signal_actor.get_signal.remote())
            if self.consumer_pp_size > 1:
                if all(
                    [signal.get(f"consumer_pp_{i}", None) == "ready_sync_model" for i in range(self.consumer_pp_size)]
                ):
                    cc.barrier(group_name="distributor_pg")
                    for i in range(self.consumer_pp_size):
                        self.profiler.enter(f"sync_model_consumer_pp_{i}")
                        ray.get(self.shared_signal_actor.set_signal.remote(f"consumer_pp_{i}", "not_ready_sync_model"))
                        # Broadcast the model state dict from consumer to shared variable actor
                        self.state_dict_cpu[i] = ray_broadcast_tensor_dict(
                            None,
                            0,
                            device=torch.device("cpu"),
                            group_name=f"sync_model_consumer_pp_{i}",
                            backend="gloo",
                        )
                        self.profiler.exit(f"sync_model_consumer_pp_{i}")
                        self.weight_version[i] += 1
                if all(
                    [
                        signal.get(f"producer_{self.distributor_id}_pp_{i}", None) == "ready_sync_model"
                        for i in range(self.consumer_pp_size)
                    ]
                ):
                    for i in range(self.consumer_pp_size):
                        self.profiler.enter(f"sync_model_producer_{self.distributor_id}_pp_{i}")
                        # Broadcast the model state dict to all producers
                        ray.get(
                            self.shared_signal_actor.set_signal.remote(
                                f"producer_{self.distributor_id}_pp_{i}", "not_ready_sync_model"
                            )
                        )
                        ray_broadcast_tensor_dict(
                            self.state_dict_cpu[i],
                            1,
                            device=torch.device("cpu"),
                            group_name=f"sync_model_producer_{self.distributor_id}_pp_{i}",
                            backend="gloo",
                        )
                        self.profiler.exit(f"sync_model_producer_{self.distributor_id}_pp_{i}")
            else:
                if signal.get("consumer", None) == "ready_sync_model":
                    self.profiler.enter("sync_model_consumer")
                    cc.barrier(group_name="distributor_pg")
                    ray.get(self.shared_signal_actor.set_signal.remote("consumer", "not_ready_sync_model"))
                    # Broadcast the model state dict from consumer to shared variable actor
                    self.state_dict_cpu = ray_broadcast_tensor_dict(
                        None, 0, device=torch.device("cpu"), group_name="sync_model_consumer", backend="gloo"
                    )
                    self.profiler.exit("sync_model_consumer")
                    self.weight_version[0] += 1
                if signal.get(f"producer_{self.distributor_id}", None) == "ready_sync_model":
                    self.profiler.enter(f"sync_model_producer_{self.distributor_id}")
                    # Broadcast the model state dict to all producers
                    ray.get(
                        self.shared_signal_actor.set_signal.remote(
                            f"producer_{self.distributor_id}", "not_ready_sync_model"
                        )
                    )
                    ray_broadcast_tensor_dict(
                        self.state_dict_cpu,
                        1,
                        device=torch.device("cpu"),
                        group_name=f"sync_model_producer_{self.distributor_id}",
                        backend="gloo",
                    )
                    self.profiler.exit(f"sync_model_producer_{self.distributor_id}")
            if signal.get("consumer", None) == "terminate":
                self.profiler.log("terminate sync model worker")
                break
            if last_weight_version != self.get_weight_version():
                last_weight_version = self.get_weight_version()
                ray.get(self.shared_signal_actor.set_signal.remote("distributor_weight_version", last_weight_version))

    def get_weight_version(self):
        return self.weight_version[0]
