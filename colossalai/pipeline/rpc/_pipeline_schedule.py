import threading
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
from torch._C._distributed_rpc import PyRRef
from torch.futures import Future

from colossalai.pipeline.pipeline_process_group import ppg
from colossalai.pipeline.rpc._pipeline_base import Phase, PipelineEngineBase, UniqueKey, WorkerBase, WorkItem

# Implementation of different Pipeline schedule
# <strategy>Worker defines the worker for each stage
# <strategy>PipelineEngine is the class for use


class FillDrainWorker(WorkerBase):

    def _get_work_item_key(self) -> UniqueKey:
        # execute backward first (if backward phase in work_list)
        num_microbatches = self.num_microbatches

        if self.forward_times < num_microbatches:
            target_phase = Phase.FORWARD
            target_microbatch_id = self.forward_times
        else:
            target_phase = Phase.BACKWARD
            target_microbatch_id = self.backward_times

        target_key = UniqueKey(target_microbatch_id, target_phase)

        return target_key


class FillDrainPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 partition_fn: Callable,
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 chunk: int = 1,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False,
                 data_process_func: Callable = None) -> None:

        if chunk > 1:
            assert num_microbatches % stage_num == 0, \
                "if you use interleaving strategy, make sure 'num_microbatches' is a multiple of stage_num!"
        use_1F1B = False

        super().__init__(FillDrainWorker, partition_fn, stage_num, num_microbatches, device, use_1F1B, chunk, criterion,
                         metric, checkpoint, data_process_func)


class OneFOneBWorker(WorkerBase):

    def _get_work_item_key(self) -> UniqueKey:
        # execute backward first (if backward phase in work_list)
        pp_rank = self.pp_rank
        actual_stage_num = self.actual_stage_num
        num_microbatches = self.num_microbatches
        is_last_stage = pp_rank == actual_stage_num - 1

        if self.outstanding <= self.outstanding_range[0]:
            target_phase = Phase.FORWARD
            target_microbatch_id = self.forward_times
        elif self.outstanding >= self.outstanding_range[1]:
            target_phase = Phase.BACKWARD
            target_microbatch_id = self.backward_times
        else:
            raise ValueError("outstanding_range[1] - outstanding_range[0] must be in [0, 1]")

        target_key = UniqueKey(target_microbatch_id, target_phase)

        # change outstanding_range at:
        # 1. forward times reach actual_stage_num, this is the end of continuous forward
        # 2. forward times reach num_microbatches, this is the end of 1F1B mode
        if not is_last_stage and \
            target_key.phase == Phase.FORWARD:
            if target_key.microbatch_id == actual_stage_num - 1 and num_microbatches > 2:
                # Why need num_microbatches > 2 ? Because there is no steady stage when num_microbatches <= 2
                outstanding_min = actual_stage_num - pp_rank - 1
                outstanding_max = actual_stage_num - pp_rank
                self.outstanding_range = (outstanding_min, outstanding_max)
            if target_key.microbatch_id == num_microbatches - 1:
                self.outstanding_range = (0, 0)

        return target_key


class OneFOneBPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 partition_fn: Callable,
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 chunk: int = 1,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False,
                 data_process_func: Callable = None) -> None:

        if chunk > 1:
            assert num_microbatches % stage_num == 0, \
                "if you use interleaving strategy, make sure 'num_microbatches' is a multiple of stage_num!"
        # assert num_microbatches > stage_num * chunk, "num_microbatches must be greater than stage_num * chunk"
        use_1F1B = True

        super().__init__(OneFOneBWorker, partition_fn, stage_num, num_microbatches, device, use_1F1B, chunk, criterion,
                         metric, checkpoint, data_process_func)


class ChimeraWorker(WorkerBase):

    def _get_producer_consumer(self) -> None:
        rank = self.pp_rank
        min_pp_rank = (rank // self.actual_stage_num) * self.actual_stage_num
        max_pp_rank = min_pp_rank + self.actual_stage_num - 1

        assert self.producer_stage_ids is None, f"all the producers of rank {rank} has been subscribed"
        assert self.consumer_stage_ids is None, f"all the consumers of rank {rank} has been subscribed"

        # should be aranged in order, the order of the input of current forward
        self.producer_stage_ids = []
        self.consumer_stage_ids = []

        # Just for demo
        prev_rank = rank - 1
        next_rank = rank + 1
        if prev_rank >= min_pp_rank:
            self.producer_stage_ids.append(prev_rank)
        if next_rank <= max_pp_rank:
            self.consumer_stage_ids.append(next_rank)

    def _get_work_item_key(self) -> UniqueKey:
        pp_rank = self.pp_rank
        stage_num = self.actual_stage_num
        real_microbatch_num = self.num_microbatches // 2

        forward_block_size = 1 if self.num_microbatches < stage_num else self.num_microbatches // stage_num
        forward_block_num = self.forward_times // forward_block_size

        if self.forward_times >= real_microbatch_num or \
            ((pp_rank + 1) % stage_num == 0 and forward_block_num > self.backward_times):
            target_phase = Phase.BACKWARD
            target_microbatch_id = self.backward_times
        else:    # others
            target_phase = Phase.FORWARD
            target_microbatch_id = self.forward_times

        # In up pipeline, microbatch_id to consume is 0, 2, 4 (2n)
        # In down pipeline, microbatch_id to consume is 1, 3, 5 (2n + 1)
        real_target_microbatch_id = target_microbatch_id * 2
        if pp_rank >= stage_num:
            real_target_microbatch_id += 1
        target_key = UniqueKey(real_target_microbatch_id, target_phase)

        with self.work_list_condition_lock:
            self.work_list_condition_lock.wait_for(lambda: target_key in self.work_list)
        return target_key

    def _initialize_partition(self):
        # In order to ensure the down pipeline share the same parameter
        # with the up pipeline, partition of down partition will be copied
        # from corresponding up stage
        pp_rank = self.pp_rank
        stage_num = self.actual_stage_num
        device = self.device
        if pp_rank < stage_num:
            super()._initialize_partition()
        else:
            # if it is down pipeline, create partition by origin method
            co_up_pp_worker_rref = self.pp_rank_to_worker_rref[pp_rank - stage_num]
            # get the coresponding model state dict and wait for its init
            state_dict = co_up_pp_worker_rref.rpc_sync().get_partition_state_dict()
            super()._initialize_partition()
            self.module_partition.load_state_dict(state_dict)

        # init group for chimera in ppg
        ppg.get_chimera_all_reduce_group(pp_rank)

        # lock for step sync
        self.step_sync_lock = threading.Lock()
        self.step_sync_lock.acquire()

        self.have_grad_lock = threading.Lock()
        self.have_grad_lock.acquire()

    def _get_lock_gradient(self):
        self.have_grad_lock.acquire()
        grads = self.get_parameter_gradients()
        self.step_sync_lock.release()
        return grads

    def is_first_stage(self):
        return (self.pp_rank % self.actual_stage_num) == 0

    def is_last_stage(self):
        return (self.pp_rank % self.actual_stage_num) == self.actual_stage_num - 1

    def _is_last_step(self, work_item: WorkItem) -> bool:
        if work_item.forward_only:
            last_phase = Phase.FORWARD
        else:
            last_phase = Phase.BACKWARD
        is_last_phase = work_item.phase == last_phase
        last_microbatch_id = self.num_microbatches - 1
        if self.pp_rank < self.actual_stage_num:
            last_microbatch_id -= 1
        is_last_microbatch = work_item.microbatch_id == last_microbatch_id
        return is_last_phase and is_last_microbatch

    def _get_step_order(self) -> List[int]:
        # TODO : If you want to extend it to multi head chimera, overwrite here
        stage_num = self.actual_stage_num
        pp_rank = self.pp_rank
        # pp_rank in the same device
        local_device_pp_ranks = [pp_rank, stage_num * 2 - pp_rank - 1]
        local_device_pp_ranks.sort(reverse=min(local_device_pp_ranks) < stage_num // 2)
        return local_device_pp_ranks

    def _hook_before_step(self):
        self.have_grad_lock.release()
        pp_rank = self.pp_rank
        stage_num = self.actual_stage_num
        co_pp_rank = (pp_rank + stage_num) % (2 * stage_num)

        # if currrent pp_rank is not the first to do step
        # wait its previous pp_rank finish step
        grads = self.get_parameter_gradients()

        # send
        co_worker = self.pp_rank_to_worker_rref[co_pp_rank]
        co_grads = co_worker.rpc_sync()._get_lock_gradient()
        # sync
        self.step_sync_lock.acquire()
        for i in range(len(grads)):
            grads[i] += co_grads[i]


class ChimeraPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 partition_fn: Callable,
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False,
                 data_process_func: Callable = None) -> None:

        assert num_microbatches % stage_num == 0, \
            "In Chimera, num_microbatches must be the multiply of stage_num!"
        use_1F1B = False
        chunk = 1

        super().__init__(ChimeraWorker, partition_fn, stage_num, num_microbatches, device, use_1F1B, chunk, criterion,
                         metric, checkpoint, data_process_func)

    def _consume_constraint(self, microbatch_id: int, forward_only: bool, input_pp_ranks: List[int],
                            output_pp_ranks: List[int], ret_future):
        pass

    def _create_pp_rank_to_rpc_worker_id(self) -> None:
        stage_num = self.stage_num
        self.pp_rank_to_rpc_worker_id = [0] * (stage_num * 2)
        for pp_rank in range(stage_num):
            self.pp_rank_to_rpc_worker_id[pp_rank] = pp_rank
            self.pp_rank_to_rpc_worker_id[pp_rank + stage_num] = stage_num - pp_rank - 1

    def _create_pp_rank_to_module_partition_id(self) -> None:
        stage_num = self.stage_num
        self.pp_rank_to_module_partition_id = [0] * (stage_num * 2)
        for pp_rank in range(stage_num):
            self.pp_rank_to_module_partition_id[pp_rank] = pp_rank
            self.pp_rank_to_module_partition_id[pp_rank + stage_num] = pp_rank

    def _create_ret_future(self, output_pp_ranks: List[int]) -> Dict[int, List[Future]]:
        num_microbatches = self.num_microbatches
        stage_num = self.stage_num
        up_ret_future = {pp_rank: [None] * num_microbatches for pp_rank in output_pp_ranks}
        down_ret_future = {pp_rank + stage_num: [None] * num_microbatches for pp_rank in output_pp_ranks}
        # merge up and down
        return {**up_ret_future, **down_ret_future}

    def _set_input(self, input_pp_ranks: List[int], microbatch_id: int, microbatch, forward_only: bool):
        # offset is 0 for all the ranks in up pipeline
        # offset is stage_num for all the ranks in down pipeline
        offset = (microbatch_id % 2) * self.stage_num
        for pp_rank in input_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank + offset]
            worker_rref.remote().set_input(microbatch_id, microbatch, forward_only)

    def _set_labels(self, output_pp_ranks: List[int], microbatch_id: int, microlabels):
        # offset is 0 for all the ranks in up pipeline
        # offset is stage_num for all the ranks in down pipeline
        offset = (microbatch_id % 2) * self.stage_num
        for pp_rank in output_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank + offset]
            worker_rref.remote().set_labels(microbatch_id, microlabels)

    def _subscribe_forward(self, microbatch_id: int, output_pp_ranks: List[int], ret_future: Dict[int, List[Future]]):
        key = UniqueKey(microbatch_id, Phase.FORWARD)
        offset = (microbatch_id % 2) * self.stage_num
        for pp_rank in output_pp_ranks:
            worker_rref = self.pp_rank_to_worker_rref[pp_rank + offset]
            ret_future[pp_rank + offset][microbatch_id] = worker_rref.rpc_async().get_output_by_key(key)

    def _ensure_backward(self, forward_only: bool, input_pp_ranks: List[int]):
        stage_num = self.stage_num
        num_microbatches = self.num_microbatches
        if not forward_only:
            for pp_rank in input_pp_ranks:
                up_last_microbatch_id = num_microbatches - 2
                down_last_microbatch_id = num_microbatches - 1

                up_worker_rref = self.pp_rank_to_worker_rref[pp_rank]
                down_worker_rref = self.pp_rank_to_worker_rref[pp_rank + stage_num]

                up_key = UniqueKey(up_last_microbatch_id, Phase.BACKWARD)
                down_key = UniqueKey(down_last_microbatch_id, Phase.BACKWARD)
                up_worker_rref.rpc_sync().get_output_by_key(up_key)
                down_worker_rref.rpc_sync().get_output_by_key(down_key)

    def _collect_forward_result(self, output_pp_ranks: List[int], ret_future: Dict[PyRRef, List[Future]]):
        """Logic of collection of forward in Chimera.
        Currently, only one input one output model is supported
        """
        stage_num = self.stage_num
        forward_result = []
        for pp_rank in output_pp_ranks:
            worker_forward_result = [None] * self.num_microbatches
            for microbatch_id in range(self.num_microbatches):
                offset = (microbatch_id % 2) * stage_num
                ret = ret_future[pp_rank + offset][microbatch_id].wait()
                ret = [ret] if isinstance(ret, torch.Tensor) else ret
                worker_forward_result[microbatch_id] = ret

            worker_forward_result = list(zip(*worker_forward_result))
            forward_result.extend(worker_forward_result)

        return forward_result
