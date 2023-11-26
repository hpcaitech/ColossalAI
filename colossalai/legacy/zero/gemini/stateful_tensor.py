from enum import Enum
from typing import Optional, Union

import torch

from .gemini_context import GeminiMemoryManager


def sizeof_tensor(tensor: torch.Tensor):
    return tensor.numel() * tensor.element_size()


class TensorState(Enum):
    FREE = 0
    HOLD = 1
    HOLD_AFTER_FWD = 2
    HOLD_AFTER_BWD = 3
    COMPUTE = 4


class StatefulTensor(object):
    """A Structure stores a Torch Tensor and labeled states.
    Inspired from the paper:
    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management

    https://arxiv.org/abs/2108.05818
    """

    # Global Stateful Tensor Manager
    GST_MGR = GeminiMemoryManager(TensorState)

    def __init__(self, maybe_tensor: Optional[torch.Tensor], state: Optional[TensorState] = TensorState.HOLD) -> None:
        self._state = state
        self._payload = None
        self._payload_size = 0  # byte size of current payload

        StatefulTensor.GST_MGR.register_new_instance()

        if self._state == TensorState.FREE:
            # when the state is free, payload should be None
            assert maybe_tensor is None, f"payload has to None if state is {self._state}"
        else:
            # otherwise, payload should not be None
            assert maybe_tensor is not None, f"payload can't be None if state is {self._state}"
            self._payload = maybe_tensor
            self._payload_size = sizeof_tensor(maybe_tensor)
            self.__trans_state_update(TensorState.FREE, state)

    def data_ptr(self):
        if self._payload is None:
            return 0  # if a tensor has no storage, 0 should be returned
        return self._payload.data_ptr()

    def set_null(self) -> None:
        # notice that free stateful tensor do not need to become null again
        if self.state != TensorState.FREE:
            self.__trans_state_update(self.state, TensorState.FREE)
            self.__release()

    def is_null(self) -> bool:
        if self.state == TensorState.FREE:
            # check sanity here
            assert self.payload is None
            return True
        return False

    def trans_state(self, state: TensorState) -> None:
        if self.state == TensorState.FREE:
            # free stateful tensor can't change state
            assert state == TensorState.FREE, "Free stateful tensor can't change to other states"
            return

        self.__trans_state_update(self.state, state)

        if state == TensorState.FREE:
            self.__release()
        else:
            self._state = state

    def move_to(self, device: Union[torch.device, int]):
        assert self.state is not TensorState.FREE, "Can't move free stateful tensor"

        if not isinstance(device, torch.device):
            to_device = torch.device("cuda", device)
        else:
            to_device = device

        from_device_type = self.device.type
        if from_device_type == to_device.type:
            # from device == to device
            return

        # update manager's information
        self.__trans_device_update(from_device_type, to_device.type)
        self.payload.data = self.payload.data.to(to_device)

    def payload_copy(self, tensor) -> None:
        self._payload.view(-1).copy_(tensor.view(-1))

    def payload_reset(self, tensor) -> None:
        assert tensor is not None, "Can't reset None for stateful tensors, please use set_null() instead"

        if self.payload is not None:
            # release old payload
            self.__trans_state_update(self.state, TensorState.FREE)
        else:
            # otherwise, set the state to HOLD for new payload
            self._state = TensorState.HOLD
        del self._payload

        self._payload = tensor
        self._payload_size = sizeof_tensor(tensor)
        # record new payload
        self.__trans_state_update(TensorState.FREE, self.state)

    def payload_relay(self, rhs):
        # relay the payload of rhs to current stateful tensor
        # can't support null relay right now
        assert not rhs.is_null()

        # now this function only support stateful tensor that has zero-length payload
        # because it doesn't require memory manager updating
        # you can extend this function by yourself
        assert self.payload_size == 0

        self._payload = rhs.payload
        self._payload_size = rhs.payload_size
        self._state = TensorState.HOLD
        self.__trans_state_update(rhs.state, TensorState.HOLD)

        rhs.__release()

    @property
    def payload(self) -> Optional[torch.Tensor]:
        return self._payload

    @property
    def payload_size(self) -> int:
        return self._payload_size

    @property
    def state(self) -> TensorState:
        return self._state

    @property
    def device(self) -> torch.device:
        return self._payload.device

    @property
    def dtype(self) -> torch.dtype:
        return self._payload.dtype

    @property
    def shape(self):
        return self._payload.shape

    def to(self, device: torch.device):
        raise RuntimeError("Use move_to(...) instead of call .to() on StatefulTensor")

    def to_(self, device: torch.device):
        raise RuntimeError("Use move_to(...) instead of call .to_() on StatefulTensor")

    def __release(self):
        # release current payload
        # shouldn't be visible to users
        self._state = TensorState.FREE
        self._payload = None
        self._payload_size = 0

    def __trans_state_update(self, from_state: TensorState, to_state: TensorState):
        """Update global manager when changing the state of a tensor"""
        manager = StatefulTensor.GST_MGR
        size = self.payload_size
        device_type = self.device.type

        if from_state != TensorState.FREE:
            manager.state_mem[device_type][from_state] -= size
        else:
            # when from_state is FREE, the tensor is new to manager
            # we should add its memory
            manager.total_mem[device_type] += size

        if to_state != TensorState.FREE:
            manager.state_mem[device_type][to_state] += size
        else:
            # when to_state is FREE, the tensor will be deleted soon
            # we should sub its memory
            manager.total_mem[device_type] -= size

    def __trans_device_update(self, from_type: str, to_type: str):
        """Update global manager when changing the device of a tensor"""
        manager = StatefulTensor.GST_MGR
        size = self.payload_size
        state = self.state

        # update aggregated information
        manager.total_mem[from_type] -= size
        manager.total_mem[to_type] += size

        # update the information of each state
        manager.state_mem[from_type][state] -= size
        manager.state_mem[to_type][state] += size

    def __del__(self):
        self.set_null()
        StatefulTensor.GST_MGR.delete_instance()
        del self
