from enum import Enum
from typing import Optional
import torch
from colossalai.tensor import ColoTensor


class TensorState(Enum):
    FREE = 0
    HOLD = 1
    HOLD_AFTER_FWD = 2
    HOLD_AFTER_BWD = 3
    COMPUTE = 4


class StatefulTensor(ColoTensor):
    """A Structure stores a Torch Tensor and labeled states. 
    Inspired from the paper:
    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management

    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, tensor: Optional[torch.Tensor], state: Optional[TensorState] = TensorState.HOLD) -> None:
        if tensor is not None:
            super().__init__(tensor.size(), dtype=tensor.dtype, requires_grad=tensor.requires_grad, \
                pin_memory=tensor.pin_memory, torch_tensor=tensor)
        else:
            super().__init__(0)

        self._state = state
        if self._state == TensorState.FREE:
            assert self.torch_tensor().numel() == 0, f"payload has to None if state is {self._state}"

    def data_ptr(self):
        if self.torch_tensor().numel() == 0:
            return None
        return self.torch_tensor().data_ptr()

    @property
    def state(self) -> TensorState:
        return self._state

    def set_null(self) -> None:
        self._state = TensorState.FREE
        self.del_torch_tensor()

    def is_null(self) -> bool:
        if self._state == TensorState.FREE:
            assert self.torch_tensor().numel() == 0
            return True
        return False

    def trans_state(self, state: TensorState) -> None:
        self._state = state
        if state == TensorState.FREE:
            self.del_torch_tensor()

    @property
    def payload(self) -> Optional[torch.Tensor]:
        return self.torch_tensor()

    def copy_payload(self, tensor) -> None:
        self.torch_tensor.view(-1).copy_(tensor.view(-1))

    def reset_payload(self, tensor) -> None:
        self._torch_tensor = tensor
        self.trans_state(TensorState.HOLD)

    @property
    def device(self) -> torch.device:
        return self.torch_tensor().device

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_tensor().dtype

    @property
    def shape(self):
        return self.torch_tensor().shape

    def to(self, device: torch.device):
        raise RuntimeError("Use colo_model_tensor_move install of call .to() on ShardedTensor")

    def to_(self, device: torch.device):
        raise RuntimeError("Use colo_model_tensor_move install of call .to_() on ShardedTensor")
