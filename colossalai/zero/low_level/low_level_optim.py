# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from colossalai.amp.naive_amp.mixed_precision_mixin import (
    BF16MixedPrecisionMixin,
    FP16MixedPrecisionMixin,
    MixedPrecisionMixin,
)
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.tensor import ColoParameter, ProcessGroup
from colossalai.utils.cuda import get_current_device

from ._utils import (
    calculate_global_norm_from_list,
    compute_norm,
    flatten,
    has_inf_or_nan,
    reduce_tensor_dp_group,
    release_param_grad,
    split_by_dtype,
    sync_param,
)
from .bookkeeping import BucketStore, GradientStore, ParameterStore, TensorBucket


class LowLevelZeroFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):

    def __init__(self,
                 num_working_param_groups: int,
                 grad_store: GradientStore,
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32) -> None:
        super().__init__(initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis,
                         max_scale)
        self.num_working_param_groups = num_working_param_groups
        self.grad_store = grad_store

    def check_local_overflow(self) -> bool:
        for group_id in range(self.num_working_param_groups):
            for avg_grad in self.grad_store.get_averaged_gradients_by_group(group_id):
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    return True
        return False


class LowLevelZeroOptimizer(ColossalaiOptimizer):
    """Optimizer used for ZeRO-1 and ZeRO-2.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            initial_scale: int = 2**16,    # grad scaler config
            min_scale: int = 1,
            growth_factor: float = 2.,
            backoff_factor: float = .5,
            growth_interval: int = 2000,
            hysteresis: int = 2,
            max_scale: int = 2**24,
            clip_grad_norm: float = 0.0,    # grad clipping
            verbose: bool = False,
            reduce_bucket_size: int = 1024 * 1024,    # communication
            communication_dtype: Optional[torch.dtype] = None,
            overlap_communication: bool = False,
            partition_grad: bool = False,    # stage 2 flag
            cpu_offload: bool = False,    # cpu offload
            forced_dtype: Optional[torch.dtype] = None):

        # TODO: add support for
        # 1. optimize the sharding
        # 2. support layer drop
        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)
        self._dtype = self.optim.param_groups[0]['params'][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        colo_pg = self._search_colo_process_group()
        if isinstance(colo_pg, ProcessGroup):
            self._local_rank = colo_pg.dp_local_rank()
            self._world_size = colo_pg.dp_world_size()
            self._dp_global_ranks = colo_pg.get_ranks_in_dp()
            self._dp_torch_group = colo_pg.dp_process_group()
            self._mp_torch_group = None
            if colo_pg.tp_world_size() > 1:
                self._mp_torch_group = colo_pg.tp_process_group()
        elif colo_pg is None:
            dp_parallel_mode = ParallelMode.DATA
            mp_parallel_mode = ParallelMode.MODEL

            self._dp_parallel_mode = dp_parallel_mode
            self._mp_parallel_mode = mp_parallel_mode
            self._local_rank = gpc.get_local_rank(dp_parallel_mode)
            self._world_size = gpc.get_world_size(dp_parallel_mode)
            self._dp_global_ranks = gpc.get_ranks_in_group(dp_parallel_mode)
            self._dp_torch_group = gpc.get_group(dp_parallel_mode)
            self._mp_torch_group = None
            if gpc.is_initialized(mp_parallel_mode) and gpc.get_world_size(mp_parallel_mode) > 1:
                self._mp_torch_group = gpc.get_group(mp_parallel_mode)
        else:
            raise NotImplementedError

        # working and master params for mixed precision training
        self._working_param_groups = dict()
        self._master_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group['params']
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self._dp_torch_group)
        self._grad_store = GradientStore(self._dp_torch_group)
        self._bucket_store = BucketStore(self._dp_torch_group)

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = list()
            for param in param_group['params']:
                if param.requires_grad:
                    group_params.append(param)

            # add the working params to working_param_groups for bookkeeping
            self._working_param_groups[group_id] = group_params

            master_param_current_rank = self._create_master_param_current_rank(group_params)

            self._master_param_groups_of_current_rank[group_id] = master_param_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group['params'] = master_param_current_rank

        # intialize communication stream for
        # communication-compuation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

        # initialize mixed precision mixin
        self.mixed_precision_mixin: Optional[MixedPrecisionMixin] = None
        if self._dtype is torch.float16:
            self.mixed_precision_mixin = LowLevelZeroFP16MixedPrecisionMixin(self.num_param_groups,
                                                                             self._grad_store,
                                                                             initial_scale=initial_scale,
                                                                             min_scale=min_scale,
                                                                             growth_factor=growth_factor,
                                                                             backoff_factor=backoff_factor,
                                                                             growth_interval=growth_interval,
                                                                             hysteresis=hysteresis,
                                                                             max_scale=max_scale)
        elif self._dtype is torch.bfloat16:
            self.mixed_precision_mixin = BF16MixedPrecisionMixin()

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_param_groups(self):
        return len(self._working_param_groups)

    def _sanity_checks(self):
        assert torch.cuda.is_available(), 'CUDA is required'
        for param_group in self.optim.param_groups:
            group_params = param_group['params']
            for param in group_params:
                assert param.dtype == self._dtype, \
                    f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    def _search_colo_process_group(self):
        colo_flag = False
        colo_pg = None
        for param_group in self.optim.param_groups:
            group_params = param_group['params']
            for param in group_params:
                if isinstance(param, ColoParameter):
                    colo_flag = True
                    if colo_pg is None:
                        colo_pg = param.get_process_group()
                    else:
                        assert colo_pg == param.get_process_group(), "All parameters should be in a same process group"
                elif colo_flag:
                    raise RuntimeError("All parameters should be ColoParameter if you use ColoParameter.")
        return colo_pg

    def _create_master_param_current_rank(self, param_list):
        # split each param evenly
        params_current_rank = []
        device = 'cpu' if self._cpu_offload else get_current_device()

        offset = 0
        for param in reversed(param_list):
            padding_size = (self._world_size - param.numel() % self._world_size) % self._world_size
            self._param_store.record_offset_in_flatten(param, offset)
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.view(-1), [0, padding_size])
                else:
                    padding_param = param.view(-1)
                splited_params = padding_param.split(param.numel() // self._world_size)
                offset += splited_params[0].numel()

                splited_param_current_rank = splited_params[self._local_rank].detach().float().to(device)
                params_current_rank.append(splited_param_current_rank)
                self._param_store.link_master_and_working_param(splited_param_current_rank, param)

        return params_current_rank

    ###########################
    # Backward Reduction Hook #
    ###########################

    def _grad_handler(self, param, group_id, grad):
        self._add_to_bucket(param, group_id, grad)
        return grad

    def _attach_reduction_hook(self):
        # we iterate over the working params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    param.register_hook(partial(self._grad_handler, param, group_id))

    #######################
    # Reduction Functions #
    #######################

    def _run_reduction(self):
        if self._bucket_store.num_elements_in_bucket() > 0:
            # self._bucket_store.flatten_grad()
            grads_in_bucket = self._bucket_store.get_grad()
            if self._overlap_communication:
                stream = self._comm_stream
            else:
                stream = torch.cuda.current_stream()

            with torch.cuda.stream(stream):
                # TODO: both zero 1 and 2 do need flatten when comm
                group_id = self._bucket_store.current_group_id
                if not self._partition_grads:
                    for rank, grad_list in grads_in_bucket.items():
                        for grad in grad_list:
                            dist.all_reduce(grad, group=self._dp_torch_group)

                            self._grad_store.append_average_gradient_by_group(group_id, rank, grad)

                else:
                    for i in range(len(grads_in_bucket[0])):
                        comm_grad_list = []
                        for rank, grad_list in grads_in_bucket.items():
                            comm_grad_list.append(grad_list[i])
                        grad = torch.zeros_like(comm_grad_list[0])
                        dist.reduce_scatter(grad, comm_grad_list, group=self._dp_torch_group)
                        self._grad_store.append_average_gradient_by_group(group_id, self._local_rank, grad)

                self._bucket_store.reset()

    def _add_to_bucket(self, param, group_id, grad):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if self._bucket_store.num_elements_in_bucket(
        ) + param_size > self._reduce_bucket_size or group_id != self._bucket_store.current_group_id:

            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store.add_param_grad(group_id, grad, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False, sync_grad=True):
        if self.mixed_precision_mixin is not None:
            loss = self.mixed_precision_mixin.pre_backward(loss)
        loss.backward(retain_graph=retain_graph)

        # finish gradient reduction
        if not self._partition_grads:
            self._reduce_grad_stage1()
        else:
            # TODO: support async comm in reduce
            self._reduce_scatter_grad_stage2()

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()

        self.zero_grad()
        # gradient synchronization
        # if sync_grad:
        # self._sync_grad()

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        if self.mixed_precision_mixin is not None:
            self.mixed_precision_mixin.pre_zero_grad()
        for _, param_group in self._working_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure=None):
        assert closure is None, 'closure is not supported by step()'

        if self.mixed_precision_mixin is not None and self.mixed_precision_mixin.should_skip_step():
            self._grad_store.reset_all_average_gradients()
            if self._verbose:
                self._logger.info(f'Found overflow. Skip step')
            self.zero_grad()
            return

        # copy the grad of working param to master param
        grad_partition_groups = []
        norm_groups = []

        for group_id in range(self.num_param_groups):
            # compute norm
            working_avg_grads = self._grad_store.get_averaged_gradients_by_group(group_id, self._local_rank)
            master_params = self._master_param_groups_of_current_rank[group_id]

            norm_group = compute_norm(gradients=working_avg_grads,
                                      params=master_params,
                                      dp_group=self._dp_torch_group,
                                      mp_group=self._mp_torch_group,
                                      master_working_map=self._param_store.master_to_working_param)
            norm_groups.append(norm_group)

            # create flat gradient for the flat fp32 master params
            for param, grad in zip(master_params, working_avg_grads):
                assert param.shape == grad.shape, \
                    f'fp32 param and grad have different shape {param.shape} vs {grad.shape}'

                dtype = param.dtype
                master_grad = grad.to(dtype)
                param.grad = master_grad.to(param.device)
                grad_partition_groups.append(grad)

            self._grad_store.reset_average_gradients_by_group(group_id)

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()
        # release the master grad

        # update working partition updated by the current rank
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])
            # for param in self._working_param_groups[group_id]:
            master_params = self._master_param_groups_of_current_rank[group_id]
            for partition_param in master_params:
                device = partition_param.device
                # print(device, self._local_rank)
                # if device == "cpu":
                #     partition_param = partition_param.to(device='cuda')
                # print(partition_param.device, self._local_rank)
                full_master_param = [torch.zeros_like(partition_param).cuda() for _ in range(self._world_size)]
                dist.all_gather(full_master_param, partition_param.cuda(), group=self._dp_torch_group)

                working_param = self._param_store.master_to_working_param[id(partition_param)]

                full_master_param = flatten(full_master_param)[:working_param.numel()].reshape_as(working_param)

                working_param.data.copy_(full_master_param)

    #############################
    # Mixed Precision Utilities #
    #############################

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        div_scale = 1.0
        if self.mixed_precision_mixin is not None:
            div_scale = self.mixed_precision_mixin.get_grad_div_scale()

        if self._clip_grad_norm > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / div_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1:
                div_scale = clip * div_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1. / div_scale)

    ############################
    # Gradient Synchronization #
    ############################

    def _sync_grad(self):
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        for group_id in range(self.num_param_groups):
            param_group = self._param_store.get_params_by_rank_group(self._local_rank, group_id)

            avg_gradients_group = self._grad_store.get_averaged_gradients_by_group(group_id)

            param_idx = 0
            for param in param_group:
                if param.grad is not None:
                    if len(avg_gradients_group) == param_idx:
                        self._grad_store.append_average_gradient_by_group(group_id, param.grad)
                    else:
                        self._grad_store.add_average_gradient_by_group(group_id, param_idx, param.grad)
                    param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    def _reduce_grad_stage1(self):
        # if not overlapping communication (no reduction hook is attached)
        # we need to manually reduce these gradients
        if not self._overlap_communication:
            for group_id in range(len(self._working_param_groups)):
                param_group = self._working_param_groups[group_id]
                for param in param_group:
                    if param.grad is not None:
                        self._add_to_reduction_bucket(param)

        # we need to reduce the gradients
        # left in the communication bucket
        self._run_reduction()

    def _reduce_scatter_grad_stage2(self):
        # when partition_grads is True, reduction hooks
        # are attached in the __init__ function, so we
        # only need to reduce the gradients
        # left in the communication bucket

        # TODO:use reduce-scatter
        self._run_reduction()
        # for reduce_rank in range(self._world_size):
        #     self._run_reduction(reduce_rank)
