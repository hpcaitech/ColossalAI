import torch
from torch.autograd.profiler_util import _format_memory

from colossalai.elixir.cuda import gpu_device


class BufferStore(object):
    """A place to store parameters temporarily when computing.
    Parameters should be stored into the buffer before computaions.

    args:
        buffer_size: The size of the buffer.
        buffer_dtype: The dtype of the buffer.
        device_str: The device to store the buffer.
    """

    def __init__(self, buffer_size: torch.Tensor, buffer_dtype: torch.dtype, device_str: str = 'cuda') -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer_dtype = buffer_dtype
        self.buffer: torch.Tensor = torch.empty(buffer_size, dtype=buffer_dtype, device=device_str)
        self.buffer_occ = buffer_size * self.buffer.element_size()
        self.record_dict = dict()

    def zeros(self):
        torch.zero_(self.buffer)

    def insert(self, t: torch.Tensor, offset: int) -> int:
        assert t not in self.record_dict
        end = offset + t.numel()
        assert end <= self.buffer_size, f'buffer size is {self.buffer_size} but needs {end}'

        new_data = self.buffer[offset:end].view(t.shape)
        new_data.copy_(t.data)

        self.record_dict[t] = t.data
        t.data = new_data

        return end

    def erase(self, t: torch.Tensor):
        assert t in self.record_dict

        new_data = self.record_dict.pop(t)
        t.data = new_data

        return

    def empty_like(self, t: torch.Tensor):
        return self.buffer[:t.numel()].view(t.shape)

    def empty_1d(self, size: int):
        return self.buffer[:size]

    def __repr__(self) -> str:
        return f'Buffer(size={self.buffer_size}, dtype={self.buffer_dtype}, memo_occ={_format_memory(self.buffer_occ)})'
