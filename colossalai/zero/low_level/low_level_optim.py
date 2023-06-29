# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from contextlib import contextmanager
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
from colossalai.utils import conditional_context
from colossalai.utils.cuda import get_current_device

from ._utils import (
    calculate_global_norm_from_list,
    compute_norm,
    flatten,
    has_inf_or_nan,
    release_param_grad,
    sync_tensor,
)
from .bookkeeping import BucketStore, GradientStore, ParameterStore


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
            for avg_grad in self.grad_store.get_working_grads_by_group_id(group_id):
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
            grad_accumulate_interval: int = 1,
            forced_dtype: Optional[torch.dtype] = None):

        assert not (partition_grad and grad_accumulate_interval > 1), \
                    "gradient accumulation is not compatible with ZeRO-2"
        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)
        self._dtype = self.optim.param_groups[0]['params'][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        # grad accumulation
        self.require_grad_sync = True
        self._accumulate_intervel = grad_accumulate_interval
        self._accumulate_step = 0

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
        self._grad_store = GradientStore(self._dp_torch_group, partition_grad=partition_grad)
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
        # split each param evenly by world size
        params_current_rank = []
        device = 'cpu' if self._cpu_offload else get_current_device()

        for param in reversed(param_list):
            padding_size = (self._world_size - param.numel() % self._world_size) % self._world_size
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                else:
                    padding_param = param.data.view(-1)
                splited_params = padding_param.split(padding_param.numel() // self._world_size)

                splited_param_current_rank = splited_params[self._local_rank].detach().float().to(device)
                params_current_rank.append(splited_param_current_rank)
                self._param_store.link_master_and_working_param(splited_param_current_rank, param)

        return params_current_rank

    ###########################
    # Backward Reduction Hook #
    ###########################

    def _grad_handler(self, param, group_id, grad):
        # if run with no_sync context, would not sync grad when backward
        if self.require_grad_sync:
            self._add_to_bucket(param, group_id)
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
            self._bucket_store.build_grad_in_bucket()
            flat_grads = self._bucket_store.get_flatten_grad()
            flat_grads /= self._world_size
            if self._overlap_communication:
                stream = self._comm_stream
            else:
                stream = torch.cuda.current_stream()

            with torch.cuda.stream(stream):
                group_id = self._bucket_store.current_group_id

                grad_dtype = flat_grads.dtype
                if self._communication_dtype is not None:
                    flat_grads = flat_grads.to(self._communication_dtype)

                if not self._partition_grads:
                    dist.all_reduce(flat_grads, group=self._dp_torch_group)
                    if flat_grads.dtype != grad_dtype:
                        flat_grads = flat_grads.to(grad_dtype)

                    flat_grads_per_rank = flat_grads.split(flat_grads.numel() // self._world_size)
                    grad_in_bucket = self._bucket_store.get_grad()

                    for rank, grad_list in grad_in_bucket.items():
                        sync_tensor(flat_grads_per_rank[rank], grad_list)
                        for grad in grad_list:
                            param_id = self._bucket_store.get_param_id_of_grad(grad)
                            self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)

                else:
                    flat_grads_list = list(flat_grads.split(len(flat_grads) // self._world_size))
                    recieved_grad = torch.zeros_like(flat_grads_list[0])
                    dist.reduce_scatter(recieved_grad, flat_grads_list, group=self._dp_torch_group)

                    if recieved_grad.dtype != grad_dtype:
                        recieved_grad = recieved_grad.to(grad_dtype)

                    grad_in_bucket_current_rank = self._bucket_store.get_grad()[self._local_rank]
                    sync_tensor(recieved_grad, grad_in_bucket_current_rank)
                    for grad in grad_in_bucket_current_rank:
                        param_id = self._bucket_store.get_param_id_of_grad(grad)
                        self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)

                self._bucket_store.reset()

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if self._bucket_store.num_elements_in_bucket() + param_size > self._reduce_bucket_size or \
            group_id != self._bucket_store.current_group_id:
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store.add_param_grad(group_id, param, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False):
        if self.mixed_precision_mixin is not None:
            loss = self.mixed_precision_mixin.pre_backward(loss)

        self._accumulate_step += 1
        no_sync = self._accumulate_step < self._accumulate_intervel
        with conditional_context(self.no_sync(), enable=no_sync):
            loss.backward(retain_graph=retain_graph)

        if no_sync:
            return

        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()

        self.zero_grad()

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

        if not self._accumulate_step == self._accumulate_intervel:
            return

        if self.mixed_precision_mixin is not None and self.mixed_precision_mixin.should_skip_step():
            self._grad_store.reset_all_gradients()
            if self._verbose:
                self._logger.info(f'Found overflow. Skip step')
            self.zero_grad()
            self._accumulate_step -= 1
            return

        # record all grads for unscale and clip
        grad_partition_groups = []
        norm_groups = []

        # sometimes not all params are 'really' working
        # for instance, when layer drop, the dropped layer has no grad
        # and should not be updated
        real_working_params = dict()
        real_master_params = dict()

        grad_index = 0 if self._partition_grads else self._local_rank

        for group_id in range(self.num_param_groups):
            master_params = self._master_param_groups_of_current_rank[group_id]
            real_working_params[group_id] = []
            real_master_params[group_id] = []
            for splited_param in master_params:
                working_param = self._param_store.master_to_working_param[id(splited_param)]
                # if a working param requires grad and has no grad
                # it is not 'really' working, e.g. the droped layer
                # else the splited grad should be attached to the splited param
                grads = self._grad_store.get_partitioned_gradients_by_param_id(group_id, id(working_param))
                if len(grads) > 0:
                    real_working_params[group_id].append(working_param)
                    grad = grads[grad_index].to(splited_param.dtype).to(splited_param.device)
                    splited_param.grad = grad
                    grad_partition_groups.append(grad)
                    real_master_params[group_id].append(splited_param)

            # compute norm
            working_grads = self._grad_store.get_working_grads_by_group_id(group_id)
            norm_group = compute_norm(gradients=working_grads,
                                      params=real_working_params[group_id],
                                      dp_group=self._dp_torch_group,
                                      mp_group=self._mp_torch_group)
            norm_groups.append(norm_group)

            self._grad_store.reset_grads_by_group_id(group_id)

            # update the params in the optimizer
            self.optim.param_groups[group_id]['params'] = real_master_params[group_id]

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()

        # release the grad
        grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])

        # update working partition updated by the current rank
        for group_id in range(self.num_param_groups):
            master_working_param = self.optim.param_groups[group_id]['params']

            for idx, splited_param in enumerate(master_working_param):
                full_master_param = [torch.zeros_like(splited_param).cuda() for _ in range(self._world_size)]
                dist.all_gather(full_master_param, splited_param.cuda(), group=self._dp_torch_group)
                working_param = real_working_params[group_id][idx]
                full_master_param = flatten(full_master_param)[:working_param.numel()].reshape_as(working_param)
                working_param.data.copy_(full_master_param)

            self.optim.param_groups[group_id]['params'] = self._master_param_groups_of_current_rank[group_id]

        # reset accumulate step
        self._accumulate_step = 0

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

    def _reduce_grad(self, partition_grad):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not partition_grad and not self._overlap_communication:
            for group_id in range(len(self._working_param_groups)):
                param_group = self._working_param_groups[group_id]
                for param in param_group:
                    if param.grad is not None:
                        self._add_to_bucket(param, group_id)

        # run reduction
        self._run_reduction()

    # this context comes from pytorch DDP
    @contextmanager
    def no_sync(self):
        old_require_grad_sync = self.require_grad_sync
        self.require_grad_sync = False
        try:
            yield
        finally:
            self.require_grad_sync = old_require_grad_sync
