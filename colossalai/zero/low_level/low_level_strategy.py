# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.accelerator import get_accelerator
from colossalai.tensor.moe_tensor.api import is_moe_tensor

from ._utils import flatten, release_param_grad, sync_tensor
from .bookkeeping import BucketStore, GradientStore, ParameterStore


class LowLevelOptStrategyBase(ABC):
    """
    Base class for low-level optimization strategies, this is to reduce the
    coupling between different param group and corresponding process group

    This class contains necessary stores/data for optimizer:
        1. params bucket
        2. grads bucket
        3. reduce buckets
    and necessary methods to do communication
    """

    # the store before refactoring supports multiple param groups
    # but currently only one is used
    DEFAULT_STORE_GROUP_ID = 0

    def __init__(
        self,
        param_group,
        process_group,
        master_weights,
        partition_grad,
        cpu_offload,
        overlap_communication,
        reduce_bucket_size,
        communication_dtype,
    ):
        # param_group that current strategy is working on
        self.param_group = param_group
        self._dtype = self.param_group["params"][0].dtype

        if process_group is None:  # if process_group is none, convert to default explicitly
            process_group = dist.group.WORLD

        self.process_group = process_group

        # if process_group is none, will use the default one
        self._local_rank = dist.get_rank(group=self.process_group)
        self._world_size = dist.get_world_size(group=self.process_group)

        # master weights copy
        self._master_weights = master_weights

        self._cpu_offload = cpu_offload

        # stage 2
        self._partition_grad = partition_grad

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(process_group)
        self._grad_store = GradientStore(process_group, partition_grad=partition_grad)
        self._bucket_store = BucketStore(process_group)

        # working and master params for mixed precision training
        group_params = []
        for param in param_group["params"]:
            if param.requires_grad:
                group_params.append(param)
        master_param_current_rank = self._create_master_param_current_rank(group_params)
        param_group["params"] = master_param_current_rank
        self.working_param_group: List[torch.Tensor] = group_params
        self.master_param_group: List[torch.Tensor] = master_param_current_rank

        # by default this shouldn't be manipulate
        self.require_grad_sync = True

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # initialize communication stream for
        # communication-computation overlapping
        if self._overlap_communication:
            self._comm_stream = get_accelerator().Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grad:
            # we iterate over the working params
            # on each param, we register a hook to its AccumulateGrad object
            param_group = self.working_param_group
            for param in param_group:
                if param.requires_grad:

                    def _grad_handler(grad, param):
                        # if run with no_sync context, would not sync grad when backward
                        if self.require_grad_sync:
                            self._add_to_bucket(param)
                        return grad

                    param.register_hook(partial(_grad_handler, param=param))

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

    def _add_to_bucket(self, param):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self._bucket_store.num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID != self._bucket_store.current_group_id
        ):
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store.add_param_grad(LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID, param, padding_size)

    def _reduce_grad(self):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not self._partition_grad and not self._overlap_communication:
            self._sync_grad()
        else:
            self._run_reduction()

    def _sync_grad(self):
        param_group = self.working_param_group
        for param in param_group:
            if param.requires_grad and param.grad is not None:
                self._add_to_bucket(param)

        self._run_reduction()

    def _run_reduction(self):
        if self._bucket_store.num_elements_in_bucket() <= 0:
            return

        self._bucket_store.build_grad_in_bucket()

        flat_grads = self._bucket_store.get_flatten_grad()
        flat_grads /= self._world_size

        # ready to add other tensors to bucket
        self._bucket_store.reset_num_elements_in_bucket()

        if self._overlap_communication:
            stream = self._comm_stream
            # in case of the memory being reused in the default stream
            flat_grads.record_stream(stream)
            # waiting for ops in the default stream finishing
            stream.wait_stream(get_accelerator().current_stream())
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            group_id = self._bucket_store.current_group_id
            assert group_id == LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID, "after refactoring, group_id should be 0"

            grad_dtype = flat_grads.dtype
            if self._communication_dtype is not None:
                flat_grads = flat_grads.to(self._communication_dtype)

            if not self._partition_grad:
                dist.all_reduce(flat_grads, group=self.process_group)
                if flat_grads.dtype != grad_dtype:
                    flat_grads = flat_grads.to(grad_dtype)

                flat_grads_per_rank = flat_grads.split(flat_grads.numel() // self._world_size)
                grad_in_bucket = self._bucket_store.get_grad()
                self._update_unpartitoned_grad(grad_in_bucket.values(), flat_grads_per_rank, group_id)
            else:
                flat_grads_list = list(flat_grads.split(len(flat_grads) // self._world_size))
                recieved_grad = torch.zeros_like(flat_grads_list[0])
                dist.reduce_scatter(recieved_grad, flat_grads_list, group=self.process_group)

                if recieved_grad.dtype != grad_dtype:
                    recieved_grad = recieved_grad.to(grad_dtype)

                grad_in_bucket_current_rank = self._bucket_store.get_grad()[self._local_rank]
                self._update_partitoned_grad(grad_in_bucket_current_rank, recieved_grad, group_id, 1)

        self._bucket_store.reset()

    ######################################################################
    # interfaces for child classes to manipulate the params, grads and buckets (and their stores)
    @property
    def master_params(self):
        return self.master_param_group

    @property
    def working_params(self):
        return self.working_param_group

    @property
    def working_grads(self):
        return self._grad_store.get_working_grads_by_group_id(LowLevelOptStrategyBase.DEFAULT_STORE_GROUP_ID)

    def get_param_padding_size(self, param):
        return self._param_store.get_param_padding_size(param)

    def get_working_param_grads(self, working_param):
        return self._grad_store.get_partitioned_gradients_by_param_id(
            LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID, id(working_param)
        )

    def update_master_params(self, working_param):
        for working_param, master_param in zip(self.working_params, self.master_params):
            padding_size = self.get_param_padding_size(working_param)
            if padding_size > 0:
                working_param = torch.nn.functional.pad(working_param, [0, padding_size])
            master_param.copy_(working_param.chunk(self._world_size)[self._local_rank])

    def get_grad_norm(self, norm_type: int = 2) -> float:
        r"""
        Compute and return the gradient norm for gradient clipping.

        Args:
            gradients (List[Tensor]): The gradients to compute norm
            norm_type (int, optional): type of the used p-norm, Can be ``'inf'`` for infinity norm. Defaults to 2.

        Returns:
            float: The total norm of given gradients
        """
        gradients = self.working_grads

        norm_type = float(norm_type)
        if norm_type == torch.inf:
            total_norm = max(grad.data.abs().max() for grad in gradients)
            total_norm_cuda = torch.tensor(
                [float(total_norm)], device=get_accelerator().get_current_device(), dtype=torch.float
            )
            dist.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
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
                total_norm_exponentiated_cuda, op=torch.distributed.ReduceOp.SUM, group=self.process_group
            )
            total_norm = total_norm_exponentiated_cuda.item() ** (1.0 / norm_type)

        return total_norm

    def zero_grad(self, set_to_none=True):
        param_group = self.working_param_group
        for param in param_group:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    param.grad.detach()
                    param.grad.zero_()

    def zero_working_grad(self):
        self._grad_store.reset_grads_by_group_id(LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID)

    def allgather_optim_state(self, master_param, master_state) -> torch.Tensor:
        device = get_accelerator().get_current_device()
        working_param = self._param_store.master_to_working_param[id(master_param)]
        gather_tensor = [
            torch.zeros(master_state.shape, device=device, dtype=master_state.dtype) for _ in range(self._world_size)
        ]
        dist.all_gather(gather_tensor, master_state, group=self.process_group)
        param_state = torch.stack(gather_tensor).view(-1)[: working_param.numel()].reshape_as(working_param).cpu()
        return param_state

    def scatter_optim_state(self, optim_state):
        with torch.no_grad():
            param_group = self.param_group
            for param in param_group["params"]:
                state = optim_state
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and k != "step":
                        padding_size = (self._world_size - v.numel() % self._world_size) % self._world_size
                        v = v.flatten()
                        if padding_size > 0:
                            v = torch.nn.functional.pad(v, [0, padding_size])
                        v_list = v.split(v.numel() // self._world_size)
                        state[k] = v_list[self._local_rank].detach().clone()

    def get_param_grad(self, param):
        grad_maybe_partial = self.get_working_param_grads(param)
        if len(grad_maybe_partial) == 0:
            return None
        if self._partition_grad:
            tensor_list = [torch.empty_like(grad_maybe_partial[0]) for _ in range(self._world_size)]
            dist.all_gather(tensor_list, grad_maybe_partial[0], group=self.process_group)
            grad_flat = torch.cat(tensor_list, dim=0)
        else:
            grad_flat = torch.cat(grad_maybe_partial, dim=0)
        return grad_flat[: param.numel()].reshape_as(param)

    ######################################################################
    # interfaces for child classes to implement, which will be called at
    # corresponding stage in LowLevelOptimizer

    @abstractmethod
    def pre_backward(self, loss, retain_graph=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def post_backward(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def pre_backward_by_grad(self, tensor, grad) -> None:
        raise NotImplementedError

    @abstractmethod
    def post_backward_by_grad(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def pre_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def post_step(self) -> None:
        raise NotImplementedError


class LowLevelOptStrategy(LowLevelOptStrategyBase):
    def __init__(
        self,
        param_group: Dict[str, Any],  # from optimizer.param_groups
        process_group: Optional[ProcessGroup] = None,  # the dp pg for comm
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        master_weights: bool = True,  # master weights
    ):
        super().__init__(
            param_group=param_group,
            process_group=process_group,
            cpu_offload=cpu_offload,
            partition_grad=partition_grad,
            master_weights=master_weights,
            reduce_bucket_size=reduce_bucket_size,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
        )

        # temporary variables
        self.__saved_master_params = None
        self.__saved_working_params = None

    ######################################################################
    # pre-backward: sanity check
    # post-backward: deal with grads

    def pre_backward(self, loss, retain_graph=False):
        assert not (
            self._partition_grad and not self.require_grad_sync
        ), "ZeRO2(partition_grad) and no_sync are not compatible"

    def post_backward(self):
        if not self.require_grad_sync:
            return

        self._reduce_grad()

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()

        for param in self.working_param_group:
            assert param.grad is None, "unreduced grad are not removed"

    def pre_backward_by_grad(self, tensor, grad):
        assert not (
            self._partition_grad and not self.require_grad_sync
        ), "ZeRO2(partition_grad) and no_sync are not compatible"

    def post_backward_by_grad(self):
        self.post_backward()

    def pre_step(self) -> None:
        # sometimes not all params are 'really' working
        # for instance, when layer drop, the dropped layer has no grad
        # and should not be updated
        grad_index = 0 if self._partition_grad else self._local_rank
        real_master_params, real_working_params = [], []
        for working_param, master_param in zip(self.working_param_group, self.master_param_group):
            # if a working param requires grad and has no grad
            # it is not 'really' working, e.g. the droped layer
            # else the splited grad should be attached to the splited param
            grads = self.get_working_param_grads(working_param)
            if len(grads) > 0:
                real_master_params.append(master_param)
                real_working_params.append(working_param)
                grad = grads[grad_index]
                # no need to copy fp32 grad if master_weights is False
                if self._master_weights:
                    grad = grad.to(master_param.dtype).to(master_param.device)
                # TODO @botbw: in original code, grad_partition_groups is used
                # however it seems it's the same as working_grads as long as
                # we update the grads in store correctly
                grads[grad_index] = master_param.grad = grad

        # update the params in the optimizer and the working partition
        # @botbw: to me, it seems like the original author only wants to keep the "real_xxx_params" when do the optimizer
        # computation, and add "non real_xxx_params" back after since we might still need them for checkpoint
        # not sure if it's necessary since None grads don't really bring lots of overhead
        self.__saved_working_params = self.working_param_group
        self.__saved_master_params = self.master_param_group
        self.working_param_group = real_working_params
        self.master_param_group = self.param_group["params"] = real_master_params

    def post_step(self):
        release_param_grad(self.master_param_group)

        # update working partition updated by the current rank
        device = get_accelerator().get_current_device()
        for working_param, master_param in zip(self.working_param_group, self.master_param_group):
            all_splited_param = [
                torch.zeros(master_param.shape, device=device, dtype=self._dtype) for _ in range(self._world_size)
            ]
            dist.all_gather(all_splited_param, master_param.to(device).to(self._dtype), group=self.process_group)
            working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].reshape_as(working_param))

        # restore saved values
        self.working_param_group = self.__saved_working_params
        self.master_param_group = self.__saved_master_params
        self.__saved_master_params = self.__saved_working_params = None
        self.param_group["params"] = self.master_param_group


class MoeZeroStrategy(LowLevelOptStrategy):
    def __init__(
        self,
        param_group: Dict[str, Any],  # from optimizer.param_groups
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        process_group: Optional[ProcessGroup] = None,  # the dp pg for comm
        master_weights: bool = True,  # master weights
    ):
        for param in param_group["params"]:
            if not is_moe_tensor(param):
                raise ValueError(f"Mixture-of-Experts parameters are required for MoeZeroStrategy {type(param)}")

        super().__init__(
            param_group=param_group,
            process_group=process_group,
            cpu_offload=cpu_offload,
            partition_grad=partition_grad,
            master_weights=master_weights,
            reduce_bucket_size=reduce_bucket_size,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
        )

    # def get_param_grad(self, param):  # TODO @botbw: discuss whether it's intuitive to return grad of divided of full moe tensor
    #     moe_partial_grad = super().get_param_grad(param)
    #     moe_grad_list = [torch.empty_like(moe_partial_grad) for _ in range(self._world_size)]
    #     dist.all_gather(moe_grad_list, moe_partial_grad, group=self.process_group)
    #     moe_grad = torch.cat(moe_grad_list, dim=0).reshape(param.shape[0] * self._world_size, *param.shape[1:])
    #     return moe_grad
