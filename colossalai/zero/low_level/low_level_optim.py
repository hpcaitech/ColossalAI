# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple, Union
from weakref import proxy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor, inf
from torch.distributed import ProcessGroup
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_mixin import (
    BF16MixedPrecisionMixin,
    FP16MixedPrecisionMixin,
    MixedPrecisionMixin,
)
from colossalai.checkpoint_io.utils import calculate_tensor_size
from colossalai.interface import OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.quantization.fp8 import all_gather_fp8, all_reduce_fp8, reduce_scatter_fp8
from colossalai.tensor.moe_tensor.api import is_moe_tensor

from ._utils import (
    all_gather_into_flat_tensor_nd,
    calculate_global_norm_from_list,
    get_nd_rank,
    get_nd_world_size,
    has_inf_or_nan,
    release_param_grad,
    sync_tensor,
)
from .bookkeeping import BucketStore, GradientStore, TensorBucket
from .zero_hook import set_all_gather_handle, wait_all_gather_handle


class LowLevelZeroFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):
    def __init__(
        self,
        num_working_param_groups: int,
        pg_to_grad_store: Dict[ProcessGroup, GradientStore],
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__(
            initial_scale,
            min_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            hysteresis,
            max_scale,
        )
        self.num_working_param_groups = num_working_param_groups
        self.pg_to_grad_store = pg_to_grad_store

    def check_local_overflow(self) -> bool:
        for store in self.pg_to_grad_store.values():
            for group_id in range(self.num_working_param_groups):
                for avg_grad in store.get_working_grads_by_group_id(group_id):
                    if avg_grad is not None and has_inf_or_nan(avg_grad):
                        return True
        return False


class LowLevelZeroOptimizer(OptimizerWrapper):
    """Optimizer used for ZeRO-1 and ZeRO-2."""

    def __init__(
        self,
        optimizer: Optimizer,
        pg_to_param_list: Optional[Dict[Union[ProcessGroup, Tuple[ProcessGroup, ...]], List[nn.Parameter]]] = None,
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
        dp_process_group: Optional[ProcessGroup] = None,
        extra_dp_group: Optional[ProcessGroup] = None,
        forced_dtype: Optional[torch.dtype] = None,
        master_weights: bool = True,  # master weights
        overlap_allgather: bool = False,
        fp8_communication: bool = False,
        backward_context=None,
    ):
        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        if (dp_process_group is not None) and (pg_to_param_list is not None):
            raise ValueError("dp_process_group and pg_to_param_list should not be provided at the same time.")
        if pg_to_param_list is None and extra_dp_group is not None and dp_process_group is None:
            raise ValueError("dp_process_group should be provided when extra_dp_group is provided.")
        if pg_to_param_list is None and extra_dp_group is not None and fp8_communication:
            raise ValueError(
                "fp8_communication is not supported when pg_to_param_list is None and extra_dp_group is provided."
            )

        if pg_to_param_list is None:
            unique_dp_group = dist.group.WORLD if dp_process_group is None else dp_process_group
            if extra_dp_group is not None:
                unique_dp_group = (extra_dp_group, unique_dp_group)
            pg_to_param_list = {unique_dp_group: []}
            for group in self.optim.param_groups:
                pg_to_param_list[unique_dp_group].extend(group["params"])

        self.pg_to_param_list = pg_to_param_list
        param_to_pg = {}
        for grp, param_list in pg_to_param_list.items():
            for p in param_list:
                assert isinstance(p, nn.Parameter), f"got {type(p)}"
                param_to_pg[p] = grp
        self.param_to_pg = param_to_pg

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        # grad accumulation
        self.require_grad_sync = True

        # working and master params for mixed precision training
        self._working_param_groups = dict()
        self._master_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._overlap_allgather = overlap_allgather
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype
        self._fp8_communication = fp8_communication
        self._backward_context = backward_context

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

        # record the padding size of each param
        self._padding_map = dict()
        # padded working param is all-gather buffer and it shares the same memory with working param
        self._working_param_to_padded_working_param = dict()

        # mapping working param and master param
        self.master_to_working_param = dict()
        self.working_to_master_param = dict()

        # NOTE need to gurantee the order of process group is the same accross all ranks
        # process_group <---> xxx_store
        # process_group <---> [param1 param2 ...]
        # each process group have its own stores
        # param belonging to one process_group will use corresponding store
        self.pg_to_grad_store = {
            pg: GradientStore(pg, partition_grad=self._partition_grads) for pg in self.pg_to_param_list
        }
        # param id to grad store, have to use id(param) as key since it is used in stores
        self.pid_to_grad_store = {id(param): self.pg_to_grad_store[param_to_pg[param]] for param in param_to_pg}
        self.pg_to_bucket_store = {pg: BucketStore(pg, reduce_bucket_size) for pg in self.pg_to_param_list}
        # param id to bucket store, have to use id(param) as key since it is used in stores
        self.pid_to_bucket_store = {id(param): self.pg_to_bucket_store[param_to_pg[param]] for param in param_to_pg}

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = list()
            for param in param_group["params"]:
                if param.requires_grad:
                    group_params.append(param)

            # add the working params to working_param_groups for bookkeeping
            self._working_param_groups[group_id] = group_params

            master_param_current_rank = self._create_master_param_current_rank(group_params)
            self._master_param_groups_of_current_rank[group_id] = master_param_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group["params"] = master_param_current_rank

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        self.grad_handles = []
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

        # initialize mixed precision mixin
        self.mixed_precision_mixin: Optional[MixedPrecisionMixin] = None
        if self._dtype is torch.float16:
            self.mixed_precision_mixin = LowLevelZeroFP16MixedPrecisionMixin(
                self.num_param_groups,
                self.pg_to_grad_store,
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
        self._current_grad_norm: Optional[float] = None

    def __del__(self):
        for hook in self.grad_handles:
            hook.remove()

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
                if not hasattr(param, "skip_zero_check") or param.skip_zero_check is False:
                    assert (
                        param.dtype == self._dtype
                    ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    def _create_master_param_current_rank(self, param_list):
        # split each param evenly by world size
        params_current_rank = []
        device = "cpu" if self._cpu_offload else get_accelerator().get_current_device()

        for param in param_list:
            padding_size = (
                self.pid_to_bucket_store[id(param)].world_size
                - param.numel() % self.pid_to_bucket_store[id(param)].world_size
            ) % self.pid_to_bucket_store[id(param)].world_size
            self.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                    # # reset working params' ptr when no master weights
                    # if self._master_weights == False:
                    param.data = padding_param[: param.numel()].view(param.shape)
                else:
                    padding_param = param.data.view(-1)
                self._working_param_to_padded_working_param[param] = padding_param

                splited_params = padding_param.split(
                    padding_param.numel() // self.pid_to_bucket_store[id(param)].world_size
                )
                splited_params = splited_params[self.pid_to_bucket_store[id(param)].local_rank]

                # use fp32 when master_weights is True
                if self._master_weights is True:
                    splited_param_current_rank = splited_params.detach().clone().float().to(device)
                else:
                    splited_param_current_rank = splited_params

                params_current_rank.append(splited_param_current_rank)
                self.link_master_and_working_param(splited_param_current_rank, param)

        return params_current_rank

    ###########################
    # Backward Reduction Hook #
    ###########################

    def _attach_reduction_hook(self):
        # we iterate over the working params
        # on each param, we register a hook to its AccumulateGrad object
        self_weakref = proxy(self)

        def _grad_handler(param, group_id):
            # if run with no_sync context, would not sync grad when backward
            if self_weakref.require_grad_sync:
                self_weakref._add_to_bucket(param, group_id)

        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    self.grad_handles.append(
                        param.register_post_accumulate_grad_hook(partial(_grad_handler, group_id=group_id))
                    )

    #######################
    # Reduction Functions #
    #######################

    def _run_reduction(self):
        for bucket_store in self.pg_to_bucket_store.values():
            if bucket_store.num_elements_in_bucket() <= 0:
                continue

            bucket_store.build_grad_in_bucket()

            flat_grads = bucket_store.get_flatten_grad()
            flat_grads /= bucket_store.world_size

            # ready to add other tensors to bucket
            bucket_store.reset_num_elements_in_bucket()

            if self._overlap_communication:
                stream = bucket_store.comm_stream
                # in case of the memory being reused in the default stream
                flat_grads.record_stream(stream)
                # waiting for ops in the default stream finishing
                stream.wait_stream(get_accelerator().current_stream())
            else:
                stream = get_accelerator().current_stream()

            with get_accelerator().stream(stream):
                group_id = bucket_store.current_group_id

                grad_dtype = flat_grads.dtype
                if self._communication_dtype is not None:
                    flat_grads = flat_grads.to(self._communication_dtype)

                if not self._partition_grads:
                    for i, sz in enumerate(bucket_store.sizes):
                        grp = bucket_store.torch_pg if len(bucket_store.sizes) == 1 else bucket_store.torch_pg[i]
                        if self._fp8_communication:
                            all_reduce_fp8(flat_grads, group=grp)
                        else:
                            dist.all_reduce(flat_grads, group=grp)
                    if flat_grads.dtype != grad_dtype:
                        flat_grads = flat_grads.to(grad_dtype)

                    flat_grads_per_rank = flat_grads.split(flat_grads.numel() // bucket_store.world_size)
                    grad_in_bucket = bucket_store.get_grad()
                    self._update_unpartitoned_grad(bucket_store, grad_in_bucket.values(), flat_grads_per_rank, group_id)
                else:
                    cur_flat_grads = flat_grads
                    for i, sz in enumerate(bucket_store.sizes):
                        grp = bucket_store.torch_pg if len(bucket_store.sizes) == 1 else bucket_store.torch_pg[i]
                        flat_grads_list = list(cur_flat_grads.split(len(cur_flat_grads) // sz))
                        received_grad = torch.empty_like(flat_grads_list[0])
                        if self._fp8_communication:
                            reduce_scatter_fp8(
                                received_grad,
                                flat_grads_list,
                                group=grp,
                            )
                        else:
                            dist.reduce_scatter_tensor(received_grad, cur_flat_grads, group=grp)
                        cur_flat_grads = received_grad

                    if received_grad.dtype != grad_dtype:
                        received_grad = received_grad.to(grad_dtype)

                    grad_in_bucket_current_rank = bucket_store.get_grad()[bucket_store.local_rank]
                    self._update_partitoned_grad(bucket_store, grad_in_bucket_current_rank, received_grad, group_id, 1)

                bucket_store.reset()

    def _update_unpartitoned_grad(
        self, bucket_store: BucketStore, origin_grad_list: List, flat_grad_list: List, group_id: int
    ) -> None:
        for rank, grad_list in enumerate(origin_grad_list):
            sync_tensor(flat_grad_list[rank], grad_list)
            for grad in grad_list:
                param_id = bucket_store.get_param_id_of_grad(grad)
                self._add_grad(grad, bucket_store.world_size, group_id, param_id, rank)

    def _update_partitoned_grad(
        self,
        bucket_store: BucketStore,
        origin_grad_list: List,
        flat_grad: torch.Tensor,
        group_id: int,
        partition_num: int,
    ) -> None:
        sync_tensor(flat_grad, origin_grad_list)
        for grad in origin_grad_list:
            param_id = bucket_store.get_param_id_of_grad(grad)
            self._add_grad(grad, partition_num, group_id, param_id)

    def _add_grad(
        self,
        grad: torch.Tensor,
        partition_num: int,
        group_id: int,
        param_id: int,
        rank: int = 0,
    ) -> None:
        if (
            len(self.pid_to_grad_store[param_id].get_partitioned_gradients_by_param_id(group_id, param_id))
            < partition_num
        ):
            self.pid_to_grad_store[param_id].append_gradients_by_param_id(grad, group_id, param_id)
        else:
            self.pid_to_grad_store[param_id].add_gradients_by_param_id(grad, rank, group_id, param_id)

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self.pid_to_bucket_store[id(param)].num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or group_id != self.pid_to_bucket_store[id(param)].current_group_id
        ):
            self._run_reduction()

        padding_size = self.get_param_padding_size(param)
        self.pid_to_bucket_store[id(param)].add_param_grad(group_id, param, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, inputs=None, retain_graph=False):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and no_sync are not compatible"

        if self.mixed_precision_mixin is not None:
            loss = self.mixed_precision_mixin.pre_backward(loss)

        ctx = nullcontext() if self._backward_context is None else self._backward_context()
        with ctx:
            loss.backward(inputs=inputs, retain_graph=retain_graph)

        if not self.require_grad_sync:
            return

        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()

    def backward_by_grad(self, tensor, grad, inputs: Tensor = None, retain_graph: bool = False):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and gradient accumulation(no_sync) are not compatible"

        if self.mixed_precision_mixin is not None:
            grad = self.mixed_precision_mixin.pre_backward_by_grad(tensor, grad)
        torch.autograd.backward(
            tensor,
            grad,
            inputs=inputs,
            retain_graph=retain_graph,
        )

        if not self.require_grad_sync:
            return
        self._reduce_grad(self._partition_grads)

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()

    def zero_bucket_stores(self):
        for bucket_store in self.pg_to_bucket_store.values():
            bucket_store.reset_all()

    def zero_grad_stores(self):
        for grad_store in self.pg_to_grad_store.values():
            grad_store.reset_all_gradients()

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
        self.zero_grad_stores()
        self.zero_bucket_stores()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure=None):
        assert closure is None, "closure is not supported by step()"
        if not self.require_grad_sync:
            return

        if self.mixed_precision_mixin is not None and self.mixed_precision_mixin.should_skip_step():
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

        for group_id in range(self.num_param_groups):
            master_params = self._master_param_groups_of_current_rank[group_id]
            working_params = self._working_param_groups[group_id]
            real_working_params[group_id] = []
            real_master_params[group_id] = []
            working_grads = []
            for working_param, master_param in zip(working_params, master_params):
                # if a working param requires grad and has no grad
                # it is not 'really' working, e.g. the droped layer
                # else the splited grad should be attached to the splited param
                grad_store = self.pid_to_grad_store[id(working_param)]
                grads = grad_store.get_partitioned_gradients_by_param_id(group_id, id(working_param))
                grad_index = 0 if self._partition_grads else grad_store.local_rank
                if len(grads) > 0:
                    real_working_params[group_id].append(working_param)
                    grad = grads[grad_index]
                    # no need to copy fp32 grad if master_weights is False
                    if self._master_weights:
                        grad = grad.to(master_param.dtype).to(master_param.device)
                    master_param.grad = grad
                    grad_partition_groups.append(grad)
                    real_master_params[group_id].append(master_param)

            # compute norm
            norm_group = 0
            for grad_store in self.pg_to_grad_store.values():
                working_grads = grad_store.get_working_grads_by_group_id(group_id)
                norm_group += self._compute_grad_norm(dp_pg=grad_store.torch_pg, gradients=working_grads)

            norm_groups.append(norm_group)

            # update the params in the optimizer
            self.optim.param_groups[group_id]["params"] = real_master_params[group_id]

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._current_grad_norm = global_norm
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()

        # release the grad
        grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])

        self.pg_to_tensor_bucket = {
            pg: TensorBucket(self.pg_to_bucket_store[pg].reduce_bucket_size) for pg in self.pg_to_param_list
        }

        # update working partition updated by the current rank
        device = get_accelerator().get_current_device()
        for group_id in range(self.num_param_groups):
            master_working_param = self.optim.param_groups[group_id]["params"]
            for idx, master_param in enumerate(master_working_param):
                working_param = real_working_params[group_id][idx]
                param_to_gather = master_param.to(device).to(self._dtype)
                pg = self.param_to_pg[working_param]
                padded_working_param = self._working_param_to_padded_working_param[working_param]
                if self._overlap_allgather:
                    # handle = dist.all_gather_into_tensor(padded_working_param, param_to_gather, pg, async_op=True)
                    handle = all_gather_into_flat_tensor_nd(padded_working_param, param_to_gather, pg, async_op=True)
                    set_all_gather_handle(working_param, handle)
                else:
                    if param_to_gather.numel() > self.pg_to_tensor_bucket[pg].max_size:
                        if self._fp8_communication:
                            # TODO: fit fp8 communication
                            all_gather_fp8(
                                list(padded_working_param.chunk(dist.get_world_size(pg))),
                                param_to_gather,
                                pg,
                                fp8_format="e4m3",
                            )
                        else:
                            # dist.all_gather_into_tensor(padded_working_param, param_to_gather, pg)
                            all_gather_into_flat_tensor_nd(padded_working_param, param_to_gather, pg)
                        continue
                    try:
                        self.pg_to_tensor_bucket[pg].add_to_bucket(param_to_gather, write_back_tensor=working_param)
                    except RuntimeError:
                        self.pg_to_tensor_bucket[pg].all_gather(pg, fp8_communication=self._fp8_communication)
                        self.pg_to_tensor_bucket[pg].add_to_bucket(param_to_gather, write_back_tensor=working_param)
            self.optim.param_groups[group_id]["params"] = self._master_param_groups_of_current_rank[group_id]
        if not self._overlap_allgather:
            for pg, tensor_bucket in self.pg_to_tensor_bucket.items():
                if not tensor_bucket.is_empty():
                    tensor_bucket.all_gather(pg, fp8_communication=self._fp8_communication)

    def _compute_grad_norm(
        self, dp_pg: Union[ProcessGroup, Tuple[ProcessGroup, ...]], gradients: List[Tensor], norm_type: int = 2
    ) -> float:
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
                [float(total_norm)],
                device=get_accelerator().get_current_device(),
                dtype=torch.float,
            )
            if isinstance(dp_pg, tuple):
                for grp in dp_pg:
                    dist.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grp)
            else:
                dist.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_pg)
            total_norm = total_norm_cuda.item()

        else:
            total_norm_exponentiated = 0.0
            for grad in gradients:
                grad_norm_exponentiated = grad.data.double().norm(norm_type) ** norm_type
                total_norm_exponentiated += grad_norm_exponentiated

            # Sum across all model parallel GPUs.
            total_norm_exponentiated_cuda = torch.tensor(
                [float(total_norm_exponentiated)],
                device=get_accelerator().get_current_device(),
                dtype=torch.float,
            )
            if isinstance(dp_pg, tuple):
                for grp in dp_pg:
                    dist.all_reduce(
                        total_norm_exponentiated_cuda,
                        op=torch.distributed.ReduceOp.SUM,
                        group=grp,
                    )
            else:
                torch.distributed.all_reduce(
                    total_norm_exponentiated_cuda,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dp_pg,
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
                if is_moe_tensor(param) and param.requires_grad and param.grad is None:
                    # TODO better of of doing this
                    # assign zero grad to unrouted expert to avoid hang during grad reduction
                    param.grad = torch.zeros_like(param)

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

    def state_dict(
        self, pinned_state_dicts: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, only_on_master: bool = False
    ) -> Dict:
        """Return a state_dict same with DDP

        Returns:
            Dict: the pytorch form state_dict
        """
        zero_state = dict()
        device = get_accelerator().get_current_device()
        for param_group in self.optim.param_groups:
            for param in param_group["params"]:
                if param not in self.optim.state:
                    continue
                state = self.optim.state[param]
                working_param = self.master_to_working_param[id(param)]
                pg = self.param_to_pg[working_param]
                if not only_on_master or get_nd_rank(pg) == 0:
                    zero_state[param] = copy.deepcopy(state)
                else:
                    zero_state[param] = {}

                if pinned_state_dicts is not None and param not in pinned_state_dicts:
                    pinned_state_dicts[param] = {}

                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and k != "step":
                        gathered_tensor = torch.empty(v.numel() * get_nd_world_size(pg), device=device, dtype=v.dtype)
                        all_gather_into_flat_tensor_nd(gathered_tensor, v.to(device).flatten(), pg)
                        param_state = gathered_tensor[: working_param.numel()].reshape_as(working_param)
                        if not only_on_master or get_nd_rank(pg) == 0:
                            if pinned_state_dicts is not None and k not in pinned_state_dicts[param]:
                                pinned_state_dicts[param][k] = torch.empty_like(
                                    param_state, pin_memory=True, device="cpu"
                                )
                            if pinned_state_dicts is not None:
                                pinned_state_dicts[param][k].copy_(param_state)
                                zero_state[param][k] = pinned_state_dicts[param][k]
                            else:
                                zero_state[param][k] = param_state.cpu()

        states_dict = self._pack_state(zero_state)

        return states_dict

    def load_state_dict(self, state_dict: Dict):
        """Load state dict, requires the state_dict be the pytorch form

        Args:
            state_dict (dict): A pytorch form state_dict
        """
        zero_state_dict = copy.deepcopy(state_dict)
        idx2master = {}
        cnt = 0
        for param_group in self.optim.param_groups:
            for param in param_group["params"]:
                idx2master[cnt] = param
                cnt += 1
        for param_idx, state in zero_state_dict["state"].items():
            pg = self.param_to_pg[self.master_to_working_param[id(idx2master[param_idx])]]
            world_size = get_nd_world_size(pg)
            rank = get_nd_rank(pg)
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    padding_size = (world_size - v.numel() % world_size) % world_size
                    with torch.no_grad():
                        v = v.flatten()
                        if padding_size > 0:
                            v = torch.nn.functional.pad(v, [0, padding_size])
                        v_list = v.split(v.numel() // world_size)
                        zero_state_dict["state"][param_idx][k] = v_list[rank].detach().clone()

        self.optim.load_state_dict(zero_state_dict)

    def state_dict_shard(
        self,
        max_shard_size: int = 1024,
        pinned_state_dicts: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        only_on_master: bool = False,
    ) -> Iterator[Tuple[Dict, int]]:
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

        master2idx = {}
        cnt = 0
        for param_group in self.optim.param_groups:
            for param in param_group["params"]:
                master2idx[param] = cnt
                cnt += 1

        for param_group in self.optim.param_groups:
            for master_param in param_group["params"]:
                param_idx = master2idx[master_param]
                states = local_states[param_idx]

                current_block_size = 0
                if pinned_state_dicts is not None and param_idx not in pinned_state_dicts:
                    pinned_state_dicts[param_idx] = {}
                working_param = self.master_to_working_param[id(master_param)]
                pg = self.param_to_pg[working_param]
                if not only_on_master or get_nd_rank(pg) == 0:
                    current_block = copy.deepcopy(states)
                else:
                    current_block = {}

                for k, v in states.items():
                    if isinstance(v, torch.Tensor) and k != "step":
                        state_tensor = torch.empty(v.numel() * get_nd_world_size(pg), device=device, dtype=v.dtype)
                        all_gather_into_flat_tensor_nd(state_tensor, v.to(device).flatten(), pg)
                        state_tensor = state_tensor[: working_param.numel()].reshape_as(working_param)
                        if not only_on_master or get_nd_rank(pg) == 0:
                            if pinned_state_dicts is not None and k not in pinned_state_dicts[param_idx]:
                                pinned_state_dicts[param_idx][k] = torch.empty_like(
                                    state_tensor, pin_memory=True, device="cpu"
                                )
                            if pinned_state_dicts is not None:
                                pinned_state_dicts[param_idx][k].copy_(state_tensor)
                                current_block[k] = pinned_state_dicts[param_idx][k]
                            else:
                                current_block[k] = state_tensor.cpu()
                        current_block_size += calculate_tensor_size(state_tensor)

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
            if p_id in self.working_to_master_param:
                pg = self.param_to_pg[p]
                world_size = get_nd_world_size(pg)
                rank = get_nd_rank(pg)
                master_param = self.working_to_master_param[p_id]
                padding_size = self.get_param_padding_size(p)
                working_param = p.data.view(-1)
                if padding_size > 0:
                    working_param = torch.nn.functional.pad(working_param, [0, padding_size])
                master_param.copy_(working_param.chunk(world_size)[rank])

    def get_working_to_master_map(self) -> Dict[int, torch.Tensor]:
        return self.working_to_master_param

    def get_master_to_working_map(self) -> Dict[int, torch.Tensor]:
        return self.master_to_working_param

    def get_param_padding_map(self) -> Dict[int, torch.Tensor]:
        return self._padding_map

    def record_param_padding_size(self, param: Tensor, padding_size: int):
        """Record the padding size of a param

        Args:
            param (Tensor): The parameter
            padding_size (int): The padding size of the parameter
        """

        self._padding_map[id(param)] = padding_size

    def get_param_padding_size(self, param: Tensor) -> int:
        """Return the padding size of the parameter

        Args:
            param (Tensor): The parameter

        Returns:
            int: the padding size of the parameter
        """

        return self._padding_map[id(param)]

    def link_master_and_working_param(self, master_param: Tensor, working_param: Tensor):
        """Mapping master parameter and working parameter

        Args:
            master_param (Tensor): The parameter copy in optimizer
            working_param (Tensor): The parameter of the model
        """

        self.master_to_working_param[id(master_param)] = working_param
        self.working_to_master_param[id(working_param)] = master_param

    def get_padding_map(self) -> Dict[int, Tensor]:
        """Return the padding map

        Returns:
            Dict[int, Tensor]: The padding map
        """

        return self._padding_map

    def get_param_grad(self, working_param: nn.Parameter) -> Tensor:
        grad_store = self.pid_to_grad_store[id(working_param)]
        grad = grad_store.get_working_grad_by_param_id(id(working_param))
        if grad is None:
            return None
        grad_flat = grad.flatten()
        output_grad = torch.empty(
            grad_flat.numel() * grad_store.world_size, device=grad_flat.device, dtype=grad_flat.dtype
        )
        all_gather_into_flat_tensor_nd(output_grad, grad_flat, grad_store.torch_pg)
        return output_grad.view(-1)[: working_param.numel()].view_as(working_param)

    def get_working_grads_by_group_id(self, group_id: int) -> List[Tensor]:
        working_grads = []
        for grad_store in self.pg_to_grad_store.values():
            working_grads.extend(grad_store.get_working_grads_by_group_id(group_id))
        return working_grads

    def get_param_id_for_grad(self, grad: Tensor) -> int:
        param_id = None
        for grad_store in self.pg_to_grad_store.values():
            id_maybe_none = grad_store.get_param_id_for_grad(grad)
            if id_maybe_none is not None:
                if param_id is not None:
                    raise ValueError("The grad mapping is not unique")
                param_id = id_maybe_none
        return param_id

    def get_working_grad_by_param_id(self, param_id: int) -> Tensor:
        grad_store = self.pid_to_grad_store[param_id]
        return grad_store.get_working_grad_by_param_id(param_id)

    def get_partitioned_gradients_by_param_id(self, group_id: int, param_id: int) -> List:
        grad_store = self.pid_to_grad_store[param_id]
        return grad_store.get_partitioned_gradients_by_param_id(group_id, param_id)

    def _force_wait_all_gather(self):
        for param in self._working_param_to_padded_working_param.keys():
            wait_all_gather_handle(param)

    def get_grad_norm(self, norm_type=2, **kwargs):
        return self._current_grad_norm
