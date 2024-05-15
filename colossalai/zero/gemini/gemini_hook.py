from chunk import Chunk
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


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


DEBUG = True  # TODO @botbw: remove


class GeminiZeROHook(ColoParamOpHook):
    def __init__(
        self, gemini_manager: GeminiManager, param_order: OrderedParamGenerator, max_prefetch: int = 0
    ) -> None:
        super().__init__()
        self._gemini_manager = gemini_manager
        self._chunk_manager = gemini_manager.chunk_manager
        self._training_phase = TrainingPhase.FORWARD
        self._cur_param = None
        # param_visited_order might be updated somewhere else
        self._param_visited_order = param_order.param_visited_order
        self._max_prefetch = max_prefetch
        self._async_works: Dict[Chunk, dist.work] = {}

        # used by get_prefetch_chunks to track current param
        self._cur_param_idx = 0

    def get_prefetch_chunks(self, all_params: List[ColoParameter]) -> List[Chunk]:
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
                chunks_to_prefetch.add(chunk)
                idx += 1
        else:
            assert self._training_phase == TrainingPhase.BACKWARD
            self._cur_param_idx -= len(all_params)
            idx = self._cur_param_idx - 1
            chunks_to_prefetch = set()
            while idx >= 0 and len(chunks_to_prefetch) + 1 < self._max_prefetch:
                param = self._param_visited_order[idx]
                if is_ddp_ignored(param):
                    idx -= 1
                    continue
                chunk = self._chunk_manager.get_chunk(self._param_visited_order[idx])
                chunks_to_prefetch.add(chunk)
                idx -= 1
        return list(chunks_to_prefetch)

    def wait_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        non_prefetched_chunks = []
        for chunk in chunks:
            if chunk in self._async_works:
                self._async_works[chunk].wait()
                del self._async_works[chunk]
            else:
                non_prefetched_chunks.append(chunk)
        return non_prefetched_chunks

    def pre_op(self, all_params):
        if DEBUG:  # TODO @botbw: remove
            idxs = list(map(lambda x: self._linked_param_order.param_visited_order.index(x), all_params))
            mx = max(idxs)
            idxs = sorted(map(lambda x: x - mx, idxs))
            assert list(range(len(idxs))) == idxs, f"{idxs=}"

        # deal with current needed chunks
        params = [p for p in all_params if not is_ddp_ignored(p)]
        all_chunks = self._chunk_manager.get_chunks(params)
        chunks_wo_work = self.wait_chunks(all_chunks)
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._gemini_manager.sample_overall_data()
        self._gemini_manager.adjust_layout(chunks_wo_work)

        # deal with chunks that are to be async fetched
        prefetch_chunks = self.get_prefetch_chunks(all_params)

        # deal with chunks that are to be fetched now
        for chunk in chunks_wo_work:
            self._chunk_manager.access_chunk(chunk)

        # deal with chunks that are to be pre fetched TODO @botbw: the order here matters?
        for chunk in prefetch_chunks:
            self._async_works[chunk] = self._chunk_manager.access_chunk(chunk, async_access=True)

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
        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)
