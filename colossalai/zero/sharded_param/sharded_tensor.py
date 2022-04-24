import torch
from colossalai.gemini.stateful_tensor import StatefulTensor, TensorState


class ShardedTensor(StatefulTensor):

    def __init__(self, tensor: torch.Tensor, state: TensorState = TensorState.HOLD) -> None:
        r"""
        A tensor sharded in multiple processes. Constructed from an existing torch.Tensor instance.
        """
        assert tensor.requires_grad is False
        super().__init__(tensor, state)

        # kept the shape, numel and dtype of the init tensor.
        self._origin_shape = tensor.shape
        self._origin_numel = tensor.numel()
        self._origin_dtype = tensor.dtype
        self._is_sharded = False

    @property
    def dtype(self) -> torch.dtype:
        assert self._payload.dtype == self._origin_dtype
        return self._payload.dtype

    @property
    def origin_numel(self) -> int:
        return self._origin_numel

    @property
    def origin_shape(self) -> int:
        return self._origin_shape

    @property
    def is_sharded(self):
        return self._is_sharded

    @is_sharded.setter
    def is_sharded(self, flag: bool):
        self._is_sharded = flag
