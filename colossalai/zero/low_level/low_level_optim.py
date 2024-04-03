# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor, inf
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_mixin import (
    BF16MixedPrecisionMixin,
    FP16MixedPrecisionMixin,
    MixedPrecisionMixin,
)
from colossalai.interface import OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.tensor.moe_tensor.api import is_moe_tensor

from ._utils import calculate_global_norm_from_list, flatten, has_inf_or_nan, release_param_grad, sync_tensor
from .bookkeeping import BucketStore, GradientStore, ParameterStore


class LowLevelZeroFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):
    def __init__(
        self,
        num_working_param_groups: int,
        grad_store: GradientStore,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__(
            initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis, max_scale
        )
        self.num_working_param_groups = num_working_param_groups
        self.grad_store = grad_store

    def check_local_overflow(self) -> bool:
        for group_id in range(self.num_working_param_groups):
            for avg_grad in self.grad_store.get_working_grads_by_group_id(group_id):
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    return True
        return False


class LowLevelZeroOptimizer(OptimizerWrapper):
    """Optimizer used for ZeRO-1 and ZeRO-2."""

    def __init__(
        self,
        optimizer: Optimizer,
        initial_scale: int = 2**16,  # grad scaler config
        min_scale: int = 1,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        max_scale: int = 2**24,
        clip_grad_norm: float = 0.0,  # grad clipping
        verbose: bool = False,
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        dp_process_group: Optional[ProcessGroup] = None,  # the dp pg for comm
        forced_dtype: Optional[torch.dtype] = None,
        moe_extra_dp_process_group: Optional[ProcessGroup] = None,
        master_weights: bool = True,  # master weights
    ):
        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        # grad accumulation
        self.require_grad_sync = True

        # if process_group is none, will use the default one
        self.dp_pg = dp_process_group
        self._local_rank = dist.get_rank(group=self.dp_pg)
        self._world_size = dist.get_world_size(group=self.dp_pg)

        # extra dp
        # This group is used to sync moe param, dp_world_size = moe_duplicates * extra_dp_size.
        # Non moe param will be sync by global dp pg, moe param will be sync by extra dp pg.
        # Moe param grad is be split as non moe param by global dp pg, and grad will be merged in step.
        # And moe working and master param are split by extra dp pg.
        self.moe_extra_dp_pg = moe_extra_dp_process_group
        if self.moe_extra_dp_pg is not None:
            self.moe_extra_dp_pg_size = dist.get_world_size(group=self.moe_extra_dp_pg)
            self.moe_extra_dp_pg_rank = dist.get_rank(group=self.moe_extra_dp_pg)

        # working and master params for mixed precision training
        self._working_param_groups = dict()
        self._master_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        # master weights copy
        self._master_weights = master_weights

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self.dp_pg)
        self._grad_store = GradientStore(self.dp_pg, partition_grad=partition_grad)
        self._bucket_store = BucketStore(self.dp_pg)

        # moe param should not be stored in working_groups
        # because they have different parallel strategy
        # so we need to store them separately in param_groups
        # instead of working_groups
        self.working_moe_params = list()

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = list()
            for param in param_group["params"]:
                if param.requires_grad:
                    if self.moe_extra_dp_pg is None:
                        # skip moe param
                        if is_moe_tensor(param):
                            self.working_moe_params.append(param)
                            continue
                    group_params.append(param)

            # add the working params to working_param_groups for bookkeeping
            self._working_param_groups[group_id] = group_params

            master_param_current_rank = self._create_master_param_current_rank(group_params)
            self._master_param_groups_of_current_rank[group_id] = master_param_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group["params"] = master_param_current_rank

        # if there are moe params, store in addtional group in optim
        if len(self.working_moe_params) > 0:
            self._sync_master_param = False
            param_group = dict()
            # create fp32 master param
            for key, value in self.optim.param_groups[0].items():
                if key != "params":
                    param_group[key] = value
            self.master_moe_params = []
            for param in self.working_moe_params:
                self.master_moe_params.append(param.clone().to(torch.float32).detach())
            # create mapping from master to working for optimizer io
            self.moe_master_to_working_map = {}
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                self.moe_master_to_working_map[id(master_moe_param)] = working_moe_param
            # add to optim
            param_group["params"] = self.master_moe_params
            self.optim.param_groups.append(param_group)

        # initialize communication stream for
        # communication-computation overlapping
        if self._overlap_communication:
            self._comm_stream = get_accelerator().Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

        # initialize mixed precision mixin
        self.mixed_precision_mixin: Optional[MixedPrecisionMixin] = None
        if self._dtype is torch.float16:
            self.mixed_precision_mixin = LowLevelZeroFP16MixedPrecisionMixin(
                self.num_param_groups,
                self._grad_store,
                initial_scale=initial_scale,
                min_scale=min_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                hysteresis=hysteresis,
                max_scale=max_scale,
            )
        elif self._dtype is torch.bfloat16:
            self.mixed_precision_mixin = BF16MixedPrecisionMixin()

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_param_groups(self):
        return len(self._working_param_groups)

    def _sanity_checks(self):
        assert get_accelerator().name in ["cuda", "npu"], "device is required"
        for param_group in self.optim.param_groups:
            group_params = param_group["params"]
            for param in group_params:
                assert (
                    param.dtype == self._dtype
                ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    def _create_master_param_current_rank(self, param_list):
        # split each param evenly by world size
        params_current_rank = []
        device = "cpu" if self._cpu_offload else get_accelerator().get_current_device()

        for param in param_list:
            padding_size = (self._world_size - param.numel() % self._world_size) % self._world_size
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                    # reset working params' ptr when no master weights
                    if self._master_weights == False:
                        param.data = padding_param[: param.numel()].view(param.shape)
                else:
                    padding_param = param.data.view(-1)

                if self.moe_extra_dp_pg is not None and is_moe_tensor(param):
                    splited_params = padding_param.split(padding_param.numel() // self.moe_extra_dp_pg_size)
                    splited_params = splited_params[self.moe_extra_dp_pg_rank]
                else:
                    splited_params = padding_param.split(padding_param.numel() // self._world_size)
                    splited_params = splited_params[self._local_rank]

                # use fp32 when master_weights is True
                if self._master_weights is True:
                    splited_param_current_rank = splited_params.detach().float().to(device)
                else:
                    splited_param_current_rank = splited_params

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

            if self.moe_extra_dp_pg is None:
                flat_grads = self._bucket_store.get_flatten_grad()
                flat_grads /= self._world_size
            else:
                # record moe and non moe param
                moe_list = []
                for param in self._bucket_store._param_list:
                    moe_list.append(is_moe_tensor(param))

                # divide them into different groups
                moe_grad_list = []
                non_moe_grad_list = []
                for grad_list in self._bucket_store._grad_in_bucket.values():
                    non_moe_cur_grad = []
                    moe_cur_grad = []
                    for i in range(len(grad_list)):
                        if moe_list[i] == True:
                            moe_cur_grad.append(grad_list[i])
                        else:
                            non_moe_cur_grad.append(grad_list[i])
                    if len(moe_cur_grad) > 0:
                        moe_grad_list.append(moe_cur_grad)
                    if len(non_moe_cur_grad) > 0:
                        non_moe_grad_list.append(non_moe_cur_grad)

                if len(non_moe_grad_list) > 0:
                    non_moe_flat_grads = []
                    for grad_list in non_moe_grad_list:
                        non_moe_flat_grads.append(_flatten_dense_tensors(grad_list))
                    non_moe_flat_grads = _flatten_dense_tensors(non_moe_flat_grads)
                    non_moe_flat_grads /= self._world_size

                if len(moe_grad_list) > 0:
                    moe_flat_grads = []
                    for grad_list in moe_grad_list:
                        moe_flat_grads.append(_flatten_dense_tensors(grad_list))
                    moe_flat_grads = _flatten_dense_tensors(moe_flat_grads)

            # ready to add other tensors to bucket
            self._bucket_store.reset_num_elements_in_bucket()

            if self._overlap_communication:
                stream = self._comm_stream
                # in case of the memory being reused in the default stream
                if self.moe_extra_dp_pg is None:
                    flat_grads.record_stream(stream)
                else:
                    if len(non_moe_grad_list) > 0:
                        non_moe_flat_grads.record_stream(stream)
                    if len(moe_grad_list) > 0:
                        moe_flat_grads.record_stream(stream)
                # waiting for ops in the default stream finishing
                stream.wait_stream(get_accelerator().current_stream())
            else:
                stream = get_accelerator().current_stream()

            with get_accelerator().stream(stream):
                group_id = self._bucket_store.current_group_id

                if self.moe_extra_dp_pg is None:
                    grad_dtype = flat_grads.dtype
                    if self._communication_dtype is not None:
                        flat_grads = flat_grads.to(self._communication_dtype)

                if not self._partition_grads:
                    if self.moe_extra_dp_pg is None:
                        dist.all_reduce(flat_grads, group=self.dp_pg)
                        if flat_grads.dtype != grad_dtype:
                            flat_grads = flat_grads.to(grad_dtype)

                        flat_grads_per_rank = flat_grads.split(flat_grads.numel() // self._world_size)
                        grad_in_bucket = self._bucket_store.get_grad()
                        self._update_unpartitoned_grad(grad_in_bucket.values(), flat_grads_per_rank, group_id)

                    # sync extra zero group
                    else:
                        # sync non moe param in global dp group
                        if len(non_moe_grad_list) > 0:
                            dist.all_reduce(non_moe_flat_grads, group=self.dp_pg)
                            flat_grads_per_rank = non_moe_flat_grads.split(
                                non_moe_flat_grads.numel() // self._world_size
                            )
                            self._update_unpartitoned_grad(non_moe_grad_list, flat_grads_per_rank, group_id)

                        # sync moe param only in zero group
                        if len(moe_grad_list) > 0:
                            dist.all_reduce(moe_flat_grads, group=self.moe_extra_dp_pg)
                            flat_grads_per_rank = moe_flat_grads.split(moe_flat_grads.numel() // self._world_size)
                            self._update_unpartitoned_grad(moe_grad_list, flat_grads_per_rank, group_id)

                else:
                    if self.moe_extra_dp_pg is None:
                        flat_grads_list = list(flat_grads.split(len(flat_grads) // self._world_size))
                        recieved_grad = torch.zeros_like(flat_grads_list[0])
                        dist.reduce_scatter(recieved_grad, flat_grads_list, group=self.dp_pg)

                        if recieved_grad.dtype != grad_dtype:
                            recieved_grad = recieved_grad.to(grad_dtype)

                        grad_in_bucket_current_rank = self._bucket_store.get_grad()[self._local_rank]
                        self._update_partitoned_grad(grad_in_bucket_current_rank, recieved_grad, group_id, 1)
                    else:
                        # categorize moe and non moe param
                        grad_in_bucket_current_rank = self._bucket_store.get_grad()[self._local_rank]
                        moe_grad_in_bucket_current_rank = []
                        non_moe_grad_in_bucket_current_rank = []
                        for idx, grad in enumerate(grad_in_bucket_current_rank):
                            if moe_list[idx] == True:
                                moe_grad_in_bucket_current_rank.append(grad)
                            else:
                                non_moe_grad_in_bucket_current_rank.append(grad)

                        if len(non_moe_grad_list) > 0:
                            flat_grads_list = list(
                                non_moe_flat_grads.split(len(non_moe_flat_grads) // self._world_size)
                            )
                            recieved_grad = torch.zeros_like(flat_grads_list[0])
                            dist.reduce_scatter(recieved_grad, flat_grads_list, group=self.dp_pg)
                            self._update_partitoned_grad(
                                non_moe_grad_in_bucket_current_rank, recieved_grad, group_id, 1
                            )

                        if len(moe_grad_list) > 0:
                            flat_grads_list = list(
                                moe_flat_grads.split(len(moe_flat_grads) // self.moe_extra_dp_pg_size)
                            )
                            recieved_grad = torch.zeros_like(flat_grads_list[0])
                            dist.reduce_scatter(recieved_grad, flat_grads_list, group=self.moe_extra_dp_pg)
                            param_slice = self._world_size // self.moe_extra_dp_pg_size
                            recieved_grad = list(recieved_grad.split(len(recieved_grad) // param_slice))
                            for split_recieved_grad in recieved_grad:
                                split_recieved_grad = _unflatten_dense_tensors(
                                    split_recieved_grad, moe_grad_in_bucket_current_rank
                                )
                                for real_grad, grad in zip(split_recieved_grad, moe_grad_in_bucket_current_rank):
                                    param_id = self._bucket_store.get_param_id_of_grad(grad)
                                    self._add_grad(real_grad, param_slice, group_id, param_id)

                self._bucket_store.reset()

    def _update_unpartitoned_grad(self, origin_grad_list: List, flat_grad_list: List, group_id: int) -> None:
        for rank, grad_list in enumerate(origin_grad_list):
            sync_tensor(flat_grad_list[rank], grad_list)
            for grad in grad_list:
                param_id = self._bucket_store.get_param_id_of_grad(grad)
                self._add_grad(grad, self._world_size, group_id, param_id, rank)

    def _update_partitoned_grad(
        self, origin_grad_list: List, flat_grad: torch.Tensor, group_id: int, partition_num: int
    ) -> None:
        sync_tensor(flat_grad, origin_grad_list)
        for grad in origin_grad_list:
            param_id = self._bucket_store.get_param_id_of_grad(grad)
            self._add_grad(grad, partition_num, group_id, param_id)

    def _add_grad(self, grad: torch.Tensor, partition_num: int, group_id: int, param_id: int, rank: int = 0) -> None:
        if len(self._grad_store.get_partitioned_gradients_by_param_id(group_id, param_id)) < partition_num:
            self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)
        else:
            self._grad_store.add_gradients_by_param_id(grad, rank, group_id, param_id)

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self._bucket_store.num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or group_id != self._bucket_store.current_group_id
        ):
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store.add_param_grad(group_id, param, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and no_sync are not compatible"

        if self.mixed_precision_mixin is not None:
            loss = self.mixed_precision_mixin.pre_backward(loss)

        loss.backward(retain_graph=retain_graph)

        if not self.require_grad_sync:
            return

        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()
        self.zero_grad()

    def backward_by_grad(self, tensor, grad):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and gradient accumulation(no_sync) are not compatible"

        if self.mixed_precision_mixin is not None:
            grad = self.mixed_precision_mixin.pre_backward_by_grad(tensor, grad)
        torch.autograd.backward(tensor, grad)

        if not self.require_grad_sync:
            return
        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()

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
        assert closure is None, "closure is not supported by step()"
        if not self.require_grad_sync:
            return

        if self.mixed_precision_mixin is not None and self.mixed_precision_mixin.should_skip_step():
            self._grad_store.reset_all_gradients()
            if self._verbose:
                self._logger.info(f"Found overflow. Skip step")
            self.zero_grad()
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
                    # moe hybrid zero
                    if self.moe_extra_dp_pg is not None and is_moe_tensor(working_param):
                        real_working_params[group_id].append(working_param)
                        if self._partition_grads:
                            grad = grads
                        else:
                            param_slice = self._world_size // self.moe_extra_dp_pg_size
                            grad = grads[
                                self.moe_extra_dp_pg_rank * param_slice : (self.moe_extra_dp_pg_rank + 1) * param_slice
                            ]
                        grad = flatten(grad)
                    else:
                        real_working_params[group_id].append(working_param)
                        grad = grads[grad_index]
                    # no need to copy fp32 grad if master_weights is False
                    if self._master_weights:
                        grad = grad.to(splited_param.dtype).to(splited_param.device)
                    splited_param.grad = grad
                    grad_partition_groups.append(grad)
                    real_master_params[group_id].append(splited_param)

            # compute norm
            working_grads = self._grad_store.get_working_grads_by_group_id(group_id)
            norm_group = self._compute_grad_norm(gradients=working_grads)
            norm_groups.append(norm_group)

            self._grad_store.reset_grads_by_group_id(group_id)

            # update the params in the optimizer
            self.optim.param_groups[group_id]["params"] = real_master_params[group_id]

        # update param for moe ep
        # move grad to master param and compute norm
        if len(self.working_moe_params) > 0:
            moe_grads = []
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                if master_moe_param.grad is not None:
                    raise RuntimeError("Moe param should not have grad here")
                grad = working_moe_param.grad
                # no need to copy fp32 grad if master_weights is False
                if self._master_weights:
                    grad = grad.to(master_moe_param.dtype).to(master_moe_param.device)
                master_moe_param.grad = grad
                working_moe_param.grad = None
                moe_grads.append(grad)
                grad_partition_groups.append(grad)
            norm_group = self._compute_grad_norm(gradients=moe_grads)
            norm_groups.append(norm_group)
            self.optim.param_groups[-1]["params"] = self.master_moe_params
            del moe_grads

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()

        # release moe grad
        if len(self.working_moe_params) > 0:
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                master_moe_param.grad = None
                working_moe_param.data = (
                    master_moe_param.data.to(working_moe_param.device).to(working_moe_param.dtype).detach()
                )

        # release the grad
        grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])

        # update working partition updated by the current rank
        device = get_accelerator().get_current_device()
        for group_id in range(self.num_param_groups):
            master_working_param = self.optim.param_groups[group_id]["params"]
            for idx, splited_param in enumerate(master_working_param):
                working_param = real_working_params[group_id][idx]
                if self.moe_extra_dp_pg is not None and is_moe_tensor(working_param):
                    all_splited_param = [
                        torch.zeros(splited_param.shape, device=device, dtype=self._dtype)
                        for _ in range(self.moe_extra_dp_pg_size)
                    ]
                    dist.all_gather(
                        all_splited_param, splited_param.to(device).to(self._dtype), group=self.moe_extra_dp_pg
                    )
                else:
                    all_splited_param = [
                        torch.zeros(splited_param.shape, device=device, dtype=self._dtype)
                        for _ in range(self._world_size)
                    ]
                    dist.all_gather(all_splited_param, splited_param.to(device).to(self._dtype), group=self.dp_pg)
                working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].reshape_as(working_param))
            self.optim.param_groups[group_id]["params"] = self._master_param_groups_of_current_rank[group_id]

    def _compute_grad_norm(self, gradients: List[Tensor], norm_type: int = 2) -> float:
        r"""
        Compute and return the gradient norm for gradient clipping.

        Args:
            gradients (List[Tensor]): The gradients to compute norm
            norm_type (int, optional): type of the used p-norm, Can be ``'inf'`` for infinity norm. Defaults to 2.

        Returns:
            float: The total norm of given gradients
        """

        if len(gradients) == 0:
            return 0.0

        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(grad.data.abs().max() for grad in gradients)
            total_norm_cuda = torch.tensor(
                [float(total_norm)], device=get_accelerator().get_current_device(), dtype=torch.float
            )
            dist.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=self.dp_pg)
            total_norm = total_norm_cuda.item()

        else:
            total_norm_exponentiated = 0.0
            for grad in gradients:
                grad_norm_exponentiated = grad.data.double().norm(norm_type) ** norm_type
                total_norm_exponentiated += grad_norm_exponentiated

            # Sum across all model parallel GPUs.
            total_norm_exponentiated_cuda = torch.tensor(
                [float(total_norm_exponentiated)], device=get_accelerator().get_current_device(), dtype=torch.float
            )
            torch.distributed.all_reduce(
                total_norm_exponentiated_cuda, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg
            )
            total_norm = total_norm_exponentiated_cuda.item() ** (1.0 / norm_type)

        return total_norm

    #############################
    # Mixed Precision Utilities #
    #############################

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        div_scale = 1.0
        if self.mixed_precision_mixin is not None:
            div_scale = self.mixed_precision_mixin.get_grad_div_scale()

        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm / div_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1:
                div_scale = clip * div_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / div_scale)

    ############################
    # Gradient Synchronization #
    ############################

    # this method is used to sync gradient manually
    def _sync_grad(self):
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                if param.requires_grad and param.grad is not None:
                    self._add_to_bucket(param, group_id)

        self._run_reduction()

    def _reduce_grad(self, partition_grad):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not partition_grad and not self._overlap_communication:
            self._sync_grad()
        else:
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

    ##############
    # State Dict #
    ##############

    def _pack_state(self, state: Dict) -> Dict:
        # comes from pytorch optimizer.state_dict()
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.optim.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in state.items()}

        return {"state": packed_state, "param_groups": param_groups}

    def state_dict(self) -> Dict:
        """Return a state_dict same with DDP

        Returns:
            Dict: the pytorch form state_dict
        """
        zero_state = dict()
        device = get_accelerator().get_current_device()
        for param, state in self.optim.state.items():
            zero_state[param] = copy.deepcopy(state)
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    working_param = self._param_store.master_to_working_param[id(param)]
                    if self.moe_extra_dp_pg is not None and is_moe_tensor(v):
                        gather_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self.moe_extra_dp_pg_size)
                        ]
                        dist.all_gather(gather_tensor, v.to(device), group=self.moe_extra_dp_pg)
                    else:
                        gather_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self._world_size)
                        ]
                        dist.all_gather(gather_tensor, v.to(device), group=self.dp_pg)
                    param_state = (
                        torch.stack(gather_tensor).view(-1)[: working_param.numel()].reshape_as(working_param).cpu()
                    )
                    zero_state[param][k] = param_state

        states_dict = self._pack_state(zero_state)

        return states_dict

    def load_state_dict(self, state_dict: Dict):
        """Load state dict, requires the state_dict be the pytorch form

        Args:
            state_dict (dict): A pytorch form state_dict
        """
        zero_state_dict = copy.deepcopy(state_dict)
        for param_idx, state in zero_state_dict["state"].items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    padding_size = (self._world_size - v.numel() % self._world_size) % self._world_size
                    with torch.no_grad():
                        v = v.flatten()
                        if padding_size > 0:
                            v = torch.nn.functional.pad(v, [0, padding_size])
                        if self.moe_extra_dp_pg is not None and is_moe_tensor(v):
                            v_list = v.split(v.numel() // self.moe_extra_dp_pg_size)
                            zero_state_dict["state"][param_idx][k] = v_list[self.moe_extra_dp_pg_rank].detach().clone()
                        else:
                            v_list = v.split(v.numel() // self._world_size)
                            zero_state_dict["state"][param_idx][k] = v_list[self._local_rank].detach().clone()

        self.optim.load_state_dict(zero_state_dict)

    def state_dict_shard(self, max_shard_size: int = 1024) -> Iterator[Tuple[Dict, int]]:
        """Returns dictionaries containing a whole state of the module one by one. The max size of dictionary shard is specified by ``max_shard_size``.
           Only include the 'state' in state_dict.

        Args:
            max_shard_size (int, optional): max size of state shard (in MB). Defaults to 1024.

        Yields:
            Iterator[OrderedDict]: A generator of state dict shard
        """
        ret_block = dict()
        ret_block_size = 0

        device = get_accelerator().get_current_device()
        local_states = self.optim.state_dict()["state"]
        for param_idx, states in local_states.items():
            current_block_size = 0
            current_block = copy.deepcopy(states)

            # find the working param of current param_id
            for group_id, pg in self._master_param_groups_of_current_rank.items():
                if (group_id + 1) * len(pg) < param_idx:
                    continue
                master_param = pg[param_idx - (group_id) * len(pg)]
                working_param = self._param_store.master_to_working_param[id(master_param)]

            for k, v in states.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    if self.moe_extra_dp_pg is not None and is_moe_tensor(v):
                        state_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self.moe_extra_dp_pg_size)
                        ]
                        dist.all_gather(state_tensor, v.to(device), group=self.moe_extra_dp_pg)
                    else:
                        state_tensor = [
                            torch.zeros(v.shape, device=device, dtype=v.dtype) for _ in range(self._world_size)
                        ]
                        dist.all_gather(state_tensor, v.to(device), group=self.dp_pg)
                    state_tensor = (
                        torch.stack(state_tensor).view(-1)[: working_param.numel()].reshape_as(working_param).cpu()
                    )
                    current_block_size += state_tensor.numel()
                    current_block[k] = state_tensor

            if ret_block_size + current_block_size > max_shard_size and len(ret_block) > 0:
                yield ret_block, ret_block_size
                ret_block = dict()
                ret_block_size = 0

            ret_block[param_idx] = current_block
            ret_block_size += current_block_size

        yield ret_block, ret_block_size

    def update_master_params(self, model: nn.Module) -> None:
        """Update master params from working params

        Args:
            model (nn.Module): The model to update master params
        """
        for p in model.parameters():
            p_id = id(p)
            if p_id in self._param_store.working_to_master_param:
                master_param = self._param_store.working_to_master_param[p_id]
                padding_size = self._param_store.get_param_padding_size(p)
                working_param = p.data.view(-1)
                if padding_size > 0:
                    working_param = torch.nn.functional.pad(working_param, [0, padding_size])
                if self.moe_extra_dp_pg is not None and is_moe_tensor(p):
                    master_param.copy_(working_param.chunk(self.extra_dp_pg_size)[self.extra_dp_pg_rank])
                else:
                    master_param.copy_(working_param.chunk(self._world_size)[self._local_rank])
        if hasattr(self, "master_moe_params"):
            for master_moe_param, working_moe_param in zip(self.master_moe_params, self.working_moe_params):
                master_moe_param.copy_(working_moe_param)

    def get_working_to_master_map(self) -> Dict[int, torch.Tensor]:
        return self._param_store.working_to_master_param

    def get_master_to_working_map(self) -> Dict[int, torch.Tensor]:
        if hasattr(self, "moe_master_to_working_map"):
            return {**self._param_store.master_to_working_param, **self.moe_master_to_working_map}
        return self._param_store.master_to_working_param
