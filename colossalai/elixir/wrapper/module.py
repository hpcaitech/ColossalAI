from collections import defaultdict
from copy import copy
from functools import partial
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.utils._pytree import tree_map

from colossalai.elixir.chunk import Chunk, ChunkFetcher, ChunkGroup, MemoryPool, TensorState
from colossalai.elixir.chunk.scheduler import FIFOScheduler, PrefetchScheduler
from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.hook import BufferStore, HookParam
from colossalai.elixir.search import SearchResult
from colossalai.elixir.tensor import OutplaceTensor
from colossalai.utils.model.experimental import LazyTensor


def get_param_optim_data(param_data: torch.Tensor, param_dtype: torch.dtype):
    param_data = param_data.to(gpu_device())
    optim_data = param_data.clone() if param_data.dtype == torch.float else param_data.float()
    param_data = param_data.to(param_dtype)
    return param_data, optim_data


class ElixirModule(nn.Module):
    """Use this class to wrap your model when using Elixir. Don't know what should be written here.
    But some docstring is needed here.

    args:
        module: training module
        search_result: a SearchResult generated from a search algorithm in `elixir.search`
        process_group: the communication group, ussually dp parallel group
        prefetch: whether to use prefetch overlaping communication with computation
        dtype: the dtype used in training
    """

    def __init__(self,
                 module: nn.Module,
                 search_result: SearchResult,
                 process_group: ProcessGroup,
                 prefetch: bool = False,
                 dtype: torch.dtype = torch.float,
                 reduce_always_fp32: bool = False,
                 output_fp32: bool = False,
                 use_fused_kernels: bool = False) -> None:
        super().__init__()

        assert dtype in {torch.float, torch.float16}

        self._set_module_outplace(module)
        self.module = module
        self.dtype = dtype
        self.use_amp = (dtype == torch.float16)
        self.process_group = process_group
        self.prefetch_flag = prefetch
        self.reduce_always_fp32 = reduce_always_fp32
        self.output_fp32 = output_fp32
        self.use_fused_kernels = use_fused_kernels

        self.no_grad_state_dict = dict()
        self.grad_state_dict = dict()
        self.__init_chunk_group(search_result)
        self.__init_chunk_fetcher(search_result, prefetch)
        self.__init_buffer_storage()

        for name, param in module.named_parameters():
            if not param.requires_grad:
                assert name in self.no_grad_state_dict
                continue
            assert name in self.grad_state_dict
            param.register_hook(partial(self._gradient_handler, param=param))
            param.__class__ = HookParam

    def __init_chunk_group(self, sr: SearchResult):
        torch.cuda.empty_cache()
        state_dict = self.module.state_dict(keep_vars=True)
        for name, tensor in state_dict.items():
            if isinstance(tensor, nn.Parameter):
                assert tensor.is_floating_point(), 'the dtypes of parameters should be float dtypes'
                # deal with parameters
                if tensor.requires_grad:
                    self.grad_state_dict[name] = tensor
                else:
                    self.no_grad_state_dict[name] = tensor
                    # polish no-grad parameters
                    tensor.data = tensor.data.to(dtype=self.dtype, device=gpu_device())
            else:
                # deal with buffers
                self._lazy_init_check(tensor)
                to_dtype = self.dtype if tensor.is_floating_point() else tensor.dtype
                tensor.data = tensor.data.to(dtype=to_dtype, device=gpu_device())

        empty_mp = MemoryPool('cuda')
        empty_mp.allocate()

        self.param_chunk_group = sr.chunk_group
        self.optim_chunk_group = ChunkGroup(empty_mp)
        self.param_to_optim = dict()
        vis_set = set()

        for plan in sr.param_chunk_plans:
            assert plan.chunk_dtype == self.dtype
            # optimizer chunks should not be gathered
            optim_kwargs = copy(plan.kwargs)
            if 'rcache_fused' in optim_kwargs:
                optim_kwargs['rcache_fused'] = False

            p_chunk = self.param_chunk_group.open_chunk(chunk_size=plan.chunk_size,
                                                        chunk_dtype=plan.chunk_dtype,
                                                        process_group=self.process_group,
                                                        chunk_config=plan.kwargs)
            o_chunk = self.optim_chunk_group.open_chunk(chunk_size=plan.chunk_size,
                                                        chunk_dtype=torch.float,
                                                        process_group=self.process_group,
                                                        chunk_config=optim_kwargs)

            for name in plan.name_list:
                param = self.grad_state_dict[name]
                self._lazy_init_check(param)
                param_data, optim_data = get_param_optim_data(param.data, self.dtype)
                param.data = param_data
                p_chunk.append_tensor(param)
                o_chunk.append_tensor(optim_data)
                self.param_to_optim[param] = optim_data

                vis_set.add(param)

            self.param_chunk_group.close_chunk(p_chunk)
            self.optim_chunk_group.close_chunk(o_chunk)
            p_chunk.init_pair(o_chunk)

        # sanity check: every parameter needed gradient has been initialized
        for param in self.module.parameters():
            if param.requires_grad:
                assert param in vis_set

    def __init_chunk_fetcher(self, sr: SearchResult, prefetch: bool):
        scheduler = None
        if prefetch:
            assert sr.param_called_per_step is not None

            chunk_called_per_step = list()
            for step in sr.param_called_per_step:
                step_set = set()
                for name in step:
                    param = self.grad_state_dict[name]
                    chunk = self.param_chunk_group.ten_to_chunk[param]
                    step_set.add(chunk)
                chunk_called_per_step.append(step_set)

            scheduler = PrefetchScheduler(chunk_called_per_step=chunk_called_per_step)
        else:
            scheduler = FIFOScheduler()

        self.fetcher = ChunkFetcher(scheduler,
                                    self.param_chunk_group,
                                    overlap=prefetch,
                                    reduce_always_fp32=self.reduce_always_fp32)
        self.fetcher.reset()

    def __init_buffer_storage(self):
        buffer_size = 0
        for submodule in self.modules():
            sum_param_size = 0
            for param in submodule.parameters(recurse=False):
                if not param.requires_grad or self.fetcher.is_in_fused(param):
                    continue
                assert param.dtype == self.dtype
                sum_param_size += param.numel()
            buffer_size = max(buffer_size, sum_param_size)
        self.buffer = BufferStore(buffer_size, self.dtype)
        print('module buffer', self.buffer)

    def _gradient_handler(self, grad: torch.Tensor, param: nn.Parameter):
        # create an empty tensor
        fake_grad = self.buffer.empty_like(grad)

        with torch._C.DisableTorchFunction():
            chunk = self.fetcher.get_one_chunk(param)
            assert self.fetcher.group.is_accessed(chunk)
            if chunk.tensors_info[param].state != TensorState.HOLD_AFTER_BWD:
                raise RuntimeError()
            self.fetcher.group.tensor_trans_state(param, TensorState.READY_FOR_REDUCE)
            chunk.copy_tensor_to_chunk_slice(param, grad)
            self.fetcher.reduce_chunk(chunk)

        return fake_grad

    def _lazy_init_check(self, tensor: torch.Tensor) -> None:
        if isinstance(tensor, LazyTensor):
            tensor.materialize()

    def _set_module_outplace(self, m: nn.Module):
        # set inplace to False for all modules
        for module in m.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

    def _deattach_fetcher(self):
        self.fetcher.clear()
        HookParam.release_fetcher()
        HookParam.disable_fused_kernel()

    def _release_for_inference(self):
        torch.cuda.synchronize()

        scheduler = self.fetcher.scheduler
        param_group = self.param_chunk_group
        while True:
            maybe_chunk = scheduler.top()
            if maybe_chunk is None:
                break
            scheduler.remove(maybe_chunk)
            param_group.release_chunk(maybe_chunk)
        self._deattach_fetcher()

    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled():
            inference_mode = False
        else:
            inference_mode = True

        # reset the fetcher in this step
        self.fetcher.reset()
        HookParam.attach_fetcher(self.fetcher, self.buffer)
        if self.use_fused_kernels:
            HookParam.enable_fused_kernel()

        def to_outplace_tensor(t):
            if isinstance(t, torch.Tensor):
                if t.is_floating_point():
                    t = t.to(self.dtype)
                t = OutplaceTensor(t)
            return t

        args = tree_map(to_outplace_tensor, args)
        kwargs = tree_map(to_outplace_tensor, kwargs)

        outputs = self.module(*args, **kwargs)
        if self.output_fp32:
            outputs = outputs.float()

        if inference_mode:
            self._release_for_inference()

        return outputs

    def backward(self, loss: torch.Tensor):
        loss.backward()
        # reset the fetcher for the next step
        self._deattach_fetcher()
        # reset all attributes
        self.module.zero_grad(set_to_none=True)

    def state_dict(self,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   only_rank_0: bool = False,
                   from_param: bool = False):
        assert keep_vars is False, 'state_dict can not keep variables in ElixirModule'
        # make sure that the variables are kept, we shall detach them later
        module_state_dict = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=True)

        tensor_to_names = defaultdict(list)
        for name, tensor in module_state_dict.items():
            if isinstance(tensor, nn.Parameter) and tensor.requires_grad:
                used_tensor = self.grad_state_dict[name]
                if not from_param:
                    used_tensor = self.param_to_optim.get(used_tensor)
                tensor_to_names[used_tensor].append(name)
            else:
                module_state_dict[name] = tensor.detach()

        def update_state_dict(chunks: Iterable[Chunk]):
            for c in chunks:
                for op, cp in zip(c.get_tensors(), c.get_cpu_copy(only_rank_0)):
                    for name in tensor_to_names.get(op):
                        module_state_dict[name] = cp

        if from_param:
            used_group = self.param_chunk_group
        else:
            used_group = self.optim_chunk_group

        update_state_dict(used_group.fused_chunks)
        update_state_dict(used_group.float_chunks)

        return module_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], only_rank_0: bool = False):
        load_flag = not only_rank_0 or dist.get_rank() == 0
        if not load_flag:
            # only rank 0 loads the state dict
            assert state_dict is None

        if only_rank_0:
            # broadcast the length of the state dict
            state_length = len(state_dict) if load_flag else None
            comm_list = [state_length]
            dist.broadcast_object_list(comm_list)
            state_length = comm_list[0]
            # broadcast the keys of the state dict
            state_keys = state_dict.keys() if load_flag else [None] * state_length
            dist.broadcast_object_list(state_keys)
            # update the state dict
            if not load_flag:
                state_dict = {k: None for k in state_keys}

        # init the mapping from optim tensor to load tensor
        optim_to_load = dict()
        for name, maybe_tensor in state_dict.items():
            if name in self.no_grad_state_dict:
                no_grad_param = self.no_grad_state_dict.get(name)
                if load_flag:
                    no_grad_param.copy_(maybe_tensor)
                if only_rank_0:
                    dist.broadcast(no_grad_param, src=0)
            elif name in self.grad_state_dict:
                grad_param = self.grad_state_dict.get(name)
                optim_tensor = self.param_to_optim.get(grad_param)
                optim_to_load[optim_tensor] = maybe_tensor

        def use_state_dict(chunks: Iterable[Chunk]):
            for c in chunks:
                load_tensor_list = list()
                has_load = False
                for chunk_tensor in c.tensors_info.keys():
                    if chunk_tensor in optim_to_load:
                        has_load = True
                        load_tensor_list.append(optim_to_load[chunk_tensor])
                    else:
                        load_tensor_list.append(None)
                if has_load:
                    c.load_tensors(load_tensor_list, only_rank_0)
                    c.paired_chunk.optim_update()

        use_state_dict(self.optim_chunk_group.fused_chunks)
        use_state_dict(self.optim_chunk_group.float_chunks)

        return
