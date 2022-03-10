import torch
from colossalai.zero.sharded_param import ShardedParamV2
from colossalai.utils import get_current_device
from typing import List


class BucketizedTensorCopy(object):

    def __init__(
        self,
        chunk_size: int,
    ):
        r""" 
        torch.nn.Parameter CPU (fp32) -> ShardedParam GPU (fp16)
        TODO(jiaruifang) The class is a little bit hardcoded
        I will make it more general later.
        """

        self.chunk_size = chunk_size
        self._offset = 0
        self._cpu_buffer = torch.empty(chunk_size, dtype=torch.float, device=torch.device("cpu:0"), pin_memory=True)
        self._cuda_buffer = torch.empty(chunk_size,
                                        dtype=torch.half,
                                        device=torch.device(f"cuda:{get_current_device()}"))

        self._buffered_param_list: List[ShardedParamV2] = []
        self._numel_list = []

    def copy(self, src_param: torch.nn.Parameter, target_param: ShardedParamV2):
        assert isinstance(target_param, ShardedParamV2)
        assert isinstance(src_param, torch.nn.Parameter)

        numel = src_param.numel()

        if self._offset + numel > self.chunk_size:
            self.flush()

        assert src_param.data.device.type == 'cpu'
        self._cpu_buffer.narrow(0, self._offset, numel).copy_(src_param.data.view(-1))

        self._buffered_param_list.append(target_param)
        self._numel_list.append(numel)

        self._offset += numel

    def flush(self):
        """
        flush to cuda memory
        """
        self._cuda_buffer.copy_(self._cpu_buffer)
        flush_offset = 0
        for sparam, numel in zip(self._buffered_param_list, self._numel_list):
            sparam.data.copy_payload(self._cpu_buffer.narrow(0, flush_offset, numel))
            flush_offset += numel

        self.reset()

    def reset(self):
        self._buffered_param_list = []
        self._numel_list = []
        self._offset = 0
