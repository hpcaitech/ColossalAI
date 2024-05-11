from typing import Dict, List, Tuple

import torch
from torch.fx import Node

from colossalai.utils.common import free_storage
from colossalai.zero.gemini.chunk.chunk import alloc_storage


class Region:
    """
    Region: A container owning a piece of contiguous nodes in the DNN computing graph.

    Args:
        r_id (int): the index of the region in the computing graph.
    """

    def __init__(self, r_id: int = 0) -> None:
        self.r_id: int = r_id
        self.fp16_params: List[torch.nn.Parameter] = []
        self.param_size: int = 0
        self.shared_rid: int = self.r_id

        self.param_num: int = 0
        self.grad_num: int = 0
        self.fp16_data = None
        self.fp32_data = None
        self.cpu_grad = None
        self.temp_fp32_data = None
        self.param_to_range: Dict[torch.nn.Parameter, Tuple[int, int]] = dict()

        self.need_offload: bool = False
        self.is_syn: bool = False
        self.nodes: List[Node] = []
        self.fwd_prefetch_region = None
        self.bwd_prefetch_region = None

        self.in_mem_pool_flag: bool = False

    @property
    def can_release(self) -> bool:
        """
        Check if the region can be released.
        """
        return self.grad_num == self.param_num

    @property
    def has_inf_or_nan(self) -> bool:
        """
        Check if the grad of the region has inf or nan values on CUDA.
        """
        return torch.isinf(self.fp16_data).any() | torch.isnan(self.fp16_data).any()

    def init_param_data(self, pre_alloc_tensor: torch.Tensor = None):
        """
        Map the parameters in the region to a contiguous memory space.
        """

        self.fp16_data = torch.zeros(self.param_num, dtype=torch.half, device="cuda")
        offset = 0
        for param in self.fp16_params:
            param.data = param.data.cuda()
            p_num = param.data.numel()
            self.fp16_data[offset : offset + p_num].copy_(param.data.flatten())
            param.data = self.fp16_data[offset : offset + p_num].view(param.data.shape)
            self.param_to_range[param] = (offset, offset + p_num)
            offset += p_num

        self.fp32_data = self.fp16_data.float().cpu().pin_memory()
        free_storage(self.fp16_data)
        if self.in_mem_pool_flag and pre_alloc_tensor is not None:
            self.fp16_data = pre_alloc_tensor

    def move_param_to_cuda(self):
        """
        Move parameters from CPU to GPU.
        It first moves float32 parameters to GPU and
        then transforms float32 parameters to half-precision on the GPU.
        The reason is that the performance of precision conversion on the CPU
        is much slower than the data transfer overhead.
        """

        self.temp_fp32_data.copy_(self.fp32_data, non_blocking=True)
        self.temp_fp32_data.record_stream(torch.cuda.current_stream())
        if not self.in_mem_pool_flag:
            alloc_storage(self.fp16_data)
        self.fp16_data[: self.param_num].copy_(self.temp_fp32_data)
        self.fp16_data.record_stream(torch.cuda.current_stream())

        self.__update_params_ptr()

    def move_grad_to_cpu(self):
        """
        Move gradients from GPU to CPU.
        """

        self.cpu_grad = torch.empty(self.param_num, dtype=torch.half, pin_memory=True)
        self.cpu_grad.copy_(self.fp16_data[: self.param_num], non_blocking=True)
        self.fp16_data.record_stream(torch.cuda.current_stream())
        if not self.in_mem_pool_flag:
            self.free_cuda_data()

        self.grad_num = 0

    def free_cuda_data(self):
        free_storage(self.fp16_data)

        # torch.cuda.empty_cache()

    def copy_grad_to_region_slice(self, param: torch.nn.Parameter, data_slice: torch.Tensor) -> None:
        """
        Copy data slice to the memory space indexed by the input tensor in the region.

        Args:
            param (torch.nn.Parameter): the param used to retrieve meta information
            data_slice (torch.Tensor): the tensor to be copied to the region
        """

        begin, end = self.param_to_range[param]
        self.fp16_data[begin:end].copy_(data_slice.data.flatten())
        param.data = self.fp16_data[begin:end].view(param.data.shape)

        self.grad_num += data_slice.numel()

    def split(self, cut_node_idx: int, cut_param_idx: int):
        """
        Split the region into two and return the latter.
        """
        new_reg = Region(r_id=self.r_id + 1)
        new_reg.nodes = self.nodes[cut_node_idx:]
        new_reg.fp16_params = self.fp16_params[cut_param_idx:]
        for p in new_reg.fp16_params:
            new_reg.param_size += p.data.numel() * p.data.element_size()
            new_reg.param_num += p.data.numel()

        self.nodes = self.nodes[:cut_node_idx]
        self.fp16_params = self.fp16_params[:cut_param_idx]
        self.param_size -= new_reg.param_size
        self.param_num -= new_reg.param_num

        return new_reg

    def __update_params_ptr(self) -> None:
        for param in self.fp16_params:
            begin, end = self.param_to_range[param]
            param.data = self.fp16_data[begin:end].view(param.data.shape)
