import copy
from typing import Any, Dict

import ray
import ray.util.collective as cc
import torch
import torch.distributed.distributed_c10d as c10d
from packaging.version import Version


def ray_broadcast_object(obj: Any, src: int = 0, device=None, group_name: str = "default") -> Any:
    rank = cc.get_rank(group_name)
    if rank == src:
        if Version(torch.__version__) >= Version("2.3.0"):
            obj_tensor, size_tensor = c10d._object_to_tensor(obj, device=device, group=None)
        elif Version(torch.__version__) >= Version("1.13.0"):
            obj_tensor, size_tensor = c10d._object_to_tensor(obj, device=device)
        else:
            obj_tensor, size_tensor = c10d._object_to_tensor(obj)
        obj_tensor = obj_tensor.to(device)
        size_tensor = size_tensor.to(device)
    else:
        size_tensor = torch.empty(1, dtype=torch.int64, device=device)
    cc.broadcast(size_tensor, src, group_name)
    if rank != src:
        obj_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
    cc.broadcast(obj_tensor, src, group_name)
    if rank != src:
        if Version(torch.__version__) >= Version("2.3.0"):
            obj = c10d._tensor_to_object(obj_tensor, size_tensor.item(), group=None)
        else:
            obj = c10d._tensor_to_object(obj, size_tensor.item())
    return obj


def ray_broadcast_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor],
    src: int = 0,
    device=None,
    group_name: str = "default",
    backend: str = "nccl",
    offload_to_cpu: bool = False,
    pin_memory: bool = False,
) -> Dict[str, torch.Tensor]:
    rank = cc.get_rank(group_name)
    if tensor_dict is None:
        tensor_dict = {}
    if rank == src:
        metadata = []
        for k, v in tensor_dict.items():
            metadata.append((k, v.shape, v.dtype))
    else:
        metadata = None
    metadata = ray_broadcast_object(metadata, src, device, group_name)
    for k, shape, dtype in metadata:
        if rank == src:
            if offload_to_cpu:
                tensor = tensor_dict[k].to(device)
            else:
                tensor = tensor_dict[k]
        else:
            tensor = tensor_dict.get(k, torch.zeros(shape, dtype=dtype, device=device, pin_memory=pin_memory))
        if backend == "gloo" and dtype == torch.bfloat16:
            # Gloo does not support bfloat16, convert to float16
            tensor = tensor.view(torch.float16)
        cc.broadcast(tensor, src, group_name)
        if backend == "gloo" and dtype == torch.bfloat16:
            # Convert back to bfloat16 if it was converted to float16
            tensor = tensor.view(torch.bfloat16)
        if rank != src:
            if offload_to_cpu:
                tensor_dict[k] = tensor.cpu()
            else:
                tensor_dict[k] = tensor
    return tensor_dict


@ray.remote
class SharedVariableActor:
    def __init__(self, number_of_readers: int = 0, buffer_size_limit: int = 1000):
        self.data_queue = []
        self.data_uid = 0
        self.number_of_readers = number_of_readers
        self.queue_size = 0
        self.signals = {}
        self.process_locks = {}
        self.signal_procs_meet_count = {}
        self.buffer_size_limit = buffer_size_limit

    def pickup_rollout_task(self, num_tasks: int):
        """
        use queue size to control whether producers should generating new rollouts or wait
        for consumer to consumer more data. if queue size is less than threshold,
        it means consumer is consuming data fast enough, so producers can generate new rollouts.
        if queue size is greater than threshold, it means consumer is consuming data slowly,
        so producers should wait for consumer to consume more data.

        Any free producer can pick up the task to generate rollout then increase the queued_data_size
        to prevent other producer to pick up the task redundantly, Note it is not the real
        queue length as data may still be generating
        """
        ret = False
        if self.queue_size < (self.buffer_size_limit / max(0.1, self.signals.get("sample_utilization", 1.0))):
            ret = True
            self.queue_size += num_tasks
        return ret

    def append_data(self, data):
        self.data_queue.append([self.data_uid, data, 0])  # [data_uid, data, access_count]
        self.data_uid += 1
        return True

    def get_data(self, data_uid: int):
        # for multi-process data reading
        if not self.data_queue:
            # no data in the queue, return None
            return None
        to_pop_index = None
        ret = None
        for i, (uid, data, access_count) in enumerate(self.data_queue):
            if uid == data_uid:
                # found the data with the given uid
                self.data_queue[i][2] += 1
                ret = copy.deepcopy(data)
                if self.data_queue[i][2] == self.number_of_readers:
                    to_pop_index = i
                break
        if to_pop_index is not None:
            # remove the data from the queue if it has been accessed by all readers
            self.data_queue.pop(to_pop_index)
            self.queue_size -= data["input_ids"].size(0)
        return ret

    def acquire_process_lock(self, key: str):
        # atomic lock for process
        if key not in self.process_locks:
            self.process_locks[key] = 1  # locked
            return 0
        if self.process_locks[key] == 0:
            self.process_locks[key] = 1  # lock the process
            return 0
        else:
            return 1

    def release_process_lock(self, key: str):
        # atomic unlock for process
        assert self.process_locks.get(key, 0) == 1, f"Releasing a process lock {key} that is not locked."
        self.process_locks[key] = 0

    def set_signal(self, key: str, signal: str):
        self.signals[key] = signal

    def get_signal(self):
        return self.signals
