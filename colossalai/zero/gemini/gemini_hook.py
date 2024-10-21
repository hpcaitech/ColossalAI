from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.accelerator import get_accelerator
from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.utils import is_ddp_ignored
from colossalai.zero.gemini import TensorState
from colossalai.zero.gemini.gemini_mgr import GeminiManager


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class GeminiZeROHook(ColoParamOpHook):
    def __init__(self, gemini_manager: GeminiManager) -> None:
        super().__init__()
        self._gemini_manager = gemini_manager
        self._chunk_manager = gemini_manager.chunk_manager
        self._training_phase = TrainingPhase.FORWARD

    def pre_op(self, params):
        # map params to chunks
        params = [p for p in params if not is_ddp_ignored(p)]
        all_chunks = self._chunk_manager.get_chunks(params)

        # wait for prefetched chunks, filter those are not prefetched
        chunks_fetch_sync = self._gemini_manager.wait_chunks(all_chunks)

        # transfer state
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._gemini_manager.sample_overall_data()

        # evit chunks, aware of async fetched
        self._gemini_manager.adjust_layout(
            all_chunks, record_anyway=self._gemini_manager.placement_policy.max_prefetch > 0
        )

        # fetch the rest synchronously
        for chunk in chunks_fetch_sync:
            self._chunk_manager.access_chunk(chunk)

        # get possible chunks to prefetch
        chunks_fetch_async = self._gemini_manager.placement_policy.get_prefetch_chunks(
            is_warmup=self._gemini_manager.is_warmup(),
            compute_list=self._gemini_manager.compute_list,
            compute_idx=self._gemini_manager.compute_idx,
            async_works=self._gemini_manager.async_works,
        )

        # prefetch
        if self._gemini_manager.chunk_manager._prefetch_stream is not None:
            # This is when prefetch happens the first time and there is no dist.Work to sync,
            # there is possibility that the optimizer haven't finish computation on default stream,
            # thus we might prefetch outdated chunks there.
            #
            # Other than that, self._gemini_manager.wait_chunks will have synced with default stream
            # by calling dist.Work.wait() and this line makes no diff.
            self._gemini_manager.chunk_manager._prefetch_stream.wait_stream(get_accelerator().current_stream())

        with get_accelerator().stream(self._gemini_manager.chunk_manager._prefetch_stream):
            for chunk in chunks_fetch_async:
                maybe_work = self._chunk_manager.access_chunk(chunk, async_access=True)
                if maybe_work is not None:
                    self._gemini_manager.add_work(chunk, maybe_work)

        # record cuda model data of the current OP, including memory for prefetched chunks
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
