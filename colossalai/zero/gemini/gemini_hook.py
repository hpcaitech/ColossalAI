from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import Dict, List

import torch
import torch.distributed as dist

from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.utils import is_ddp_ignored
from colossalai.zero.gemini import TensorState
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from colossalai.zero.gemini.memory_tracer.param_runtime_order import OrderedParamGenerator
from colossalai.logging import DistributedLogger

from .chunk import Chunk

class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


logger = DistributedLogger("gemini_hook")

class GeminiZeROHook(ColoParamOpHook):
    def __init__(
        self, gemini_manager: GeminiManager, param_order: OrderedParamGenerator, max_prefetch: int = 0
    ) -> None:
        super().__init__()
        self._gemini_manager = gemini_manager
        self._chunk_manager = gemini_manager.chunk_manager
        self._training_phase = TrainingPhase.FORWARD
        # param_visited_order might be updated somewhere else
        self._param_visited_order = param_order.param_visited_order
        self._max_prefetch = max_prefetch
        self._async_works: Dict[Chunk, dist.work] = {}
        # used by get_prefetch_chunks to track current param
        self._cur_param_idx = 0

    def get_prefetch_chunks(self, all_params: List[ColoParameter], cur_chunks: List[Chunk]) -> List[Chunk]:
        chunks_to_prefetch = set()
        if self._training_phase == TrainingPhase.FORWARD:  # forward phrase: increase
            self._cur_param_idx += len(all_params)  # need to update first
            idx = self._cur_param_idx + 1
            # still have params and prefetched chunks don't exceed the limit
            while idx < len(self._param_visited_order) and len(chunks_to_prefetch) + 1 < self._max_prefetch:
                param = self._param_visited_order[idx]
                if is_ddp_ignored(param):
                    idx += 1
                    continue
                chunk = self._chunk_manager.get_chunk(param)
                if chunk not in cur_chunks:
                    chunks_to_prefetch.add(chunk)
                idx += 1
        else:
            self._cur_param_idx -= len(all_params)
            idx = self._cur_param_idx - 1
            chunks_to_prefetch = set()
            while idx >= 0 and len(chunks_to_prefetch) + 1 < self._max_prefetch:
                param = self._param_visited_order[idx]
                if is_ddp_ignored(param):
                    idx -= 1
                    continue
                chunk = self._chunk_manager.get_chunk(self._param_visited_order[idx])
                if chunk not in cur_chunks:
                    chunks_to_prefetch.add(chunk)
                idx -= 1
        print(f"cur id {self._cur_param_idx}")
        return list(chunks_to_prefetch)

    def wait_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        non_prefetched_chunks = []
        for chunk in chunks:
            if chunk in self._async_works:
                print(f"prefetched {chunk.count_id}")
                self._async_works[chunk].wait()
                del self._async_works[chunk]
            else:
                non_prefetched_chunks.append(chunk)
        return non_prefetched_chunks

    def pre_op(self, all_params):
        # def find_idx(param):
        #     for i, p in enumerate(self._param_visited_order):
        #         if param is p:
        #             return i
        #     assert False
        
        # idxs = [find_idx(p) for p in all_params]
        # max_id = min(idxs)
        # idxs = [i - max_id for i in idxs]
        # assert list(range(len(idxs))) == sorted(idxs), f'{idxs}'

        # deal with current needed chunks
        params = [p for p in all_params if not is_ddp_ignored(p)]
        all_chunks = self._chunk_manager.get_chunks(params)
        chunks_need_to_fetch_sync = tuple(self.wait_chunks(all_chunks))
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._gemini_manager.sample_overall_data()
        self._gemini_manager.adjust_layout(chunks_need_to_fetch_sync)

        # deal with chunks that are to be async fetched
        chunks_can_be_fetch_async = self.get_prefetch_chunks(all_params=all_params, cur_chunks=chunks_need_to_fetch_sync)

        print(f"cur_chunks {' '.join([str(x.count_id) for x in chunks_need_to_fetch_sync])}, prefetch {' '.join([str(x.count_id) for x in chunks_can_be_fetch_async])}")
        # deal with chunks that are to be fetched now
        for chunk in chunks_need_to_fetch_sync:
            self._chunk_manager.access_chunk(chunk)

        # deal with chunks that are to be pre fetched TODO @botbw: the order here matters?
        for chunk in chunks_can_be_fetch_async:
            if chunk in self._async_works:
                continue
            maybe_work = self._chunk_manager.access_chunk(chunk, async_access=True)
            if maybe_work is not None:
                print(f"prefetch {chunk.count_id}")
                self._async_works[chunk] = maybe_work

        # record cuda model data of the current OP
        self._gemini_manager.record_model_data_volume()

    def post_op(self, params):
        params = [p for p in params if not is_ddp_ignored(p)]
        for p in params:
            tensor_state = (
                TensorState.HOLD
                if self._training_phase == TrainingPhase.FORWARD or not p.requires_grad
                else TensorState.HOLD_AFTER_BWD
            )
            self._chunk_manager.trans_tensor_state(p, tensor_state)

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    @contextmanager
    def switch_training_phase(self, training_phase: TrainingPhase = TrainingPhase.BACKWARD):
        if training_phase == TrainingPhase.FORWARD:
            self._cur_param_idx = 0
        else:
            self._cur_param_idx = len(self._param_visited_order) - 1

        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)
