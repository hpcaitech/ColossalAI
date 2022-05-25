import torch
from colossalai.tensor import ParamOpHook, ChunkManager, TensorState
from enum import Enum
from typing import List
from contextlib import contextmanager
from functools import partial


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class ZeROHookV2(ParamOpHook):

    def __init__(self, chunk_manager: ChunkManager) -> None:
        super().__init__()
        self.chunk_manager = chunk_manager
        self.training_phase = TrainingPhase.FORWARD

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        for p in params:
            self.chunk_manager.trans_tensor_state(p, TensorState.COMPUTE)
        # TODO: evict chunks
        for p in params:
            self.chunk_manager.access_chunk(p)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        for p in params:
            tensor_state = TensorState.HOLD if self.training_phase == TrainingPhase.FORWARD or not p.requires_grad else TensorState.HOLD_AFTER_BWD
            self.chunk_manager.trans_tensor_state(p, tensor_state)
        for p in params:
            self.chunk_manager.release_chunk(p)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_forward()

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_forward()

    @contextmanager
    def switch_training_phase(self, training_phase: TrainingPhase = TrainingPhase.BACKWARD):
        try:
            old_training_phase = self.training_phase
            self.training_phase = training_phase
            yield
        finally:
            self.training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)
