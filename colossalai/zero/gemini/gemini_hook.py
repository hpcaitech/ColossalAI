from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.utils import is_ddp_ignored
from colossalai.zero.gemini import TensorState
from colossalai.zero.gemini.gemini_mgr import GeminiManager

from .utils import is_lora_ignored


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
        params = [p for p in params if not is_ddp_ignored(p)]
        chunks = self._chunk_manager.get_chunks(params)
        for p in params:
            self._chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        self._gemini_manager.sample_overall_data()
        self._gemini_manager.adjust_layout(chunks)
        for chunk in chunks:
            self._chunk_manager.access_chunk(chunk)

        # record cuda model data of the current OP
        self._gemini_manager.record_model_data_volume()

    def post_op(self, params):
        params = [p for p in params if not is_ddp_ignored(p)]
        for p in params:
            if not is_lora_ignored(p):
                tensor_state = (
                    TensorState.HOLD
                    if self._training_phase == TrainingPhase.FORWARD or not p.requires_grad
                    else TensorState.HOLD_AFTER_BWD
                )
                self._chunk_manager.trans_tensor_state(p, tensor_state)
            else:
                self.lora_ignored_post_op_handler(p)

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def lora_ignored_post_op_handler(self, p):
        # This handler is desgined for model parameters not requiring gradients when lora is enabled.

        # When doing forward, just transfer state to HOLD.
        if self._training_phase == TrainingPhase.FORWARD:
            self._chunk_manager.trans_tensor_state(p, TensorState.HOLD)
            return

        chunk = self._chunk_manager.get_chunk(p)

        # Transfer state to READY_FOR_REDUCE after backward.
        chunk.tensor_trans_state(p, TensorState.HOLD_AFTER_BWD)
        chunk.tensor_trans_state(p, TensorState.READY_FOR_REDUCE)

        # Use chunk.can_reduce as the flag for scattering
        if chunk.can_reduce:
            # Tranfer chunk state to HOLD.
            for tensor in chunk.tensors_info.keys():
                chunk.tensor_trans_state(tensor, TensorState.HOLD)
            # Scatter parameter chunk.
            if chunk.keep_gathered:
                self._chunk_manager.fake_release_chunk(chunk)
            else:
                self._chunk_manager.release_chunk(chunk)
            # Move chunk to its resident device.
            self._chunk_manager.move_chunk(chunk, self.grads_device[p], force_copy=True)

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
