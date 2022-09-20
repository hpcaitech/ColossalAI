from typing import List, Callable, Dict

import torch.nn as nn
from torch.futures import Future
from torch._C._distributed_rpc import PyRRef

from colossalai.pipeline.rpc._pipeline_base import PipelineEngineBase, WorkerBase, UniqueKey, Phase

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

        with self.work_list_condition_lock:
            self.work_list_condition_lock.wait_for(lambda: target_key in self.work_list)

        return target_key


class FillDrainPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 module_partitions: List[nn.Module],
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 chunk: int = 1,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False) -> None:

        if chunk > 1:
            assert num_microbatches % stage_num == 0, \
                "if you use interleaving strategy, make sure 'num_microbatches' is a multiple of stage_num!"
        use_1F1B = False

        super().__init__(FillDrainWorker, module_partitions, stage_num, num_microbatches, device, use_1F1B, chunk,
                         criterion, metric, checkpoint)


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
            if target_key.microbatch_id == actual_stage_num - 1:
                outstanding_min = actual_stage_num - pp_rank - 1
                outstanding_max = actual_stage_num - pp_rank
                self.outstanding_range = (outstanding_min, outstanding_max)
            elif target_key.microbatch_id == num_microbatches - 1:
                self.outstanding_range = (0, 0)

        with self.work_list_condition_lock:
            self.work_list_condition_lock.wait_for(lambda: target_key in self.work_list)

        return target_key


class OneFOneBPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 module_partitions: List[nn.Module],
                 stage_num: int,
                 num_microbatches: int,
                 device: str,
                 chunk: int = 1,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False) -> None:

        if chunk > 1:
            assert num_microbatches % stage_num == 0, \
                "if you use interleaving strategy, make sure 'num_microbatches' is a multiple of stage_num!"
        use_1F1B = True

        super().__init__(OneFOneBWorker, module_partitions, stage_num, num_microbatches, device, use_1F1B, chunk,
                         criterion, metric, checkpoint)


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

        if self.forward_times < real_microbatch_num:
            if (pp_rank + 1) % stage_num == 0:    # last rank
                forward_blocks = self.forward_times // (self.num_microbatches // stage_num)
                if forward_blocks > self.backward_times:
                    target_phase = Phase.BACKWARD
                    target_microbatch_id = self.backward_times
                else:
                    target_phase = Phase.FORWARD
                    target_microbatch_id = self.forward_times
            else:    # others
                target_phase = Phase.FORWARD
                target_microbatch_id = self.forward_times
        else:
            target_phase = Phase.BACKWARD
            target_microbatch_id = self.backward_times

        # In up pipeline, microbatch_id to consume is 0, 2, 4 (2n)
        # In down pipeline, microbatch_id to consume is 1, 3, 5 (2n + 1)
        real_target_microbatch_id = target_microbatch_id * 2
        if pp_rank >= stage_num:
            real_target_microbatch_id += 1
        target_key = UniqueKey(real_target_microbatch_id, target_phase)

        with self.work_list_condition_lock:
            self.work_list_condition_lock.wait_for(lambda: target_key in self.work_list)

        return target_key

    def is_first_stage(self):
        return (self.pp_rank % self.actual_stage_num) == 0

    def is_last_stage(self):
        return (self.pp_rank % self.actual_stage_num) == self.actual_stage_num - 1


class ChimeraPipelineEngine(PipelineEngineBase):

    def __init__(self,
                 module_partitions,
                 stage_num,
                 num_microbatches,
                 device: str,
                 criterion: Callable = None,
                 metric: Callable = None,
                 checkpoint: bool = False) -> None:

        assert num_microbatches % stage_num == 0, \
            "In Chimera, num_microbatches must be the multiply of stage_num!"
        use_1F1B = False
        chunk = 1
        super().__init__(ChimeraWorker, module_partitions, stage_num, num_microbatches, device, use_1F1B, chunk,
                         criterion, metric, checkpoint)

    def _consume_constraint(self, microbatch_id: int, forward_only: bool, ret_future: Dict[PyRRef, List[Future]],
                            input_worker_rrefs: List[PyRRef], output_worker_rrefs: List[PyRRef]):
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
                worker_forward_result[microbatch_id] = ret

            worker_forward_result = list(zip(*worker_forward_result))
            forward_result.extend(worker_forward_result)

        return forward_result
