from tokenize import group
from colossalai.utils.cuda import get_current_device
import torch
from colossalai.logging import get_dist_logger
from torch.optim import Optimizer
from .bookkeeping import ParameterStore
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.amp.naive_amp._fp16_optimizer import DynamicGradScaler
from colossalai.nn.optimizer import ColossalaiOptimizer
from ._utils import count_numel, calculate_padding, to_cpu, shuffle_by_round_robin, flatten_dense_tensors_with_padding, is_nccl_aligned

# optimization given by DeepSpeed
# fp16 is 2 bytes
# 4-byte NCCL alignment / 2 = 2
# thus, we put 2 fp16 floating point into one unit
NCCL_COMMUNICATION_UNIT_SIZE = 2


class PartitionedOptimizer(ColossalaiOptimizer):

    def __init__(
        self,
        optimizer: Optimizer,
        postscale_gradients=True,
        gradient_predivide_factor=1.0,
        ignore_unused_parameters=True,
        initial_scale=2**32,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
        max_scale: int = 2**32,
        verbose=False,
        allgather_bucket_size=5000000000,
        reduce_bucket_size=500000000,
        round_robin=True,
    ):
        # TODO: add support for
        # 1. fp16 master weights
        # 2. contiguous gradients
        # 3. cpu offload
        # 4. add support for communication overlap

        self.optimizer = optimizer
        self._logger = get_dist_logger()
        self._partition_id = gpc.get_local_rank(ParallelMode.DATA)
        self._partition_count = gpc.get_world_size(ParallelMode.DATA)

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore()
        self._fp16_param_groups = []
        self._fp32_param_partition_updated_by_current_rank = []

        # communication params
        self._reduce_bucket_size = reduce_bucket_size
        self._all_gather_bucket_size = allgather_bucket_size

        # partitioning-related
        self._round_robin = round_robin

        # fp16 optimizer

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale,
                                             verbose=verbose)

        self._sanity_checks()

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optimizer.param_groups):
            params = param_group['params']

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups.append(params)

            # calculate the number of padding needed to make it align
            # with the NCCL communication alignment
            if gpc.is_last_rank(ParallelMode.DATA):
                total_numel = count_numel(params)
                padding = calculate_padding(
                    total_numel, unit_size=NCCL_COMMUNICATION_UNIT_SIZE)
            else:
                padding = 0
            self._param_store.add_group_padding(group_id, padding)

            # move all the parameters to cpu to free up GPU space for creating flat buffer
            to_cpu(params)

            # reorder the tensors in the param group
            if self._round_robin:
                tensor_list, tensor_indices = shuffle_by_round_robin(
                    params, self._partition_count)
            else:
                tensor_list = params
                tensor_indices = {i: i for i in range(len(tensor_list))}

            self._param_store.add_reordered_fp16_params(tensor_list,
                                                        tensor_indices)

            # flatten the reordered tensors
            parallel_nccl_unit_size = NCCL_COMMUNICATION_UNIT_SIZE * self._partition_count
            flat_tensor = flatten_dense_tensors_with_padding(
                tensor_list, parallel_nccl_unit_size)
            flat_tensor = flat_tensor.cuda()
            self._param_store.add_flat_fp16_param(flat_tensor)

            # sync the weights after reordering and flattening
            # the resulting fp16 weight will be slices of the flat buffer with the new tensor order
            self._param_store.sync_fp16_param(
                group_id=group_id,
                fp16_params=self._fp16_param_groups[group_id])

            # partition the flat tensors into N partitions where N is the data parallel world size
            # flat_fp16_dp_partitions is a list of tensor
            # each element in the list is a slice of the whole fp16 flat param tensor
            flat_fp16_dp_partitions = self._param_store.partition_flat_fp16_param(
            )
            for tensor in flat_fp16_dp_partitions:
                assert is_nccl_aligned(
                    tensor
                ), 'Flat FP16 partition tensor is not in line with NCCL alighment'

            # add to parameter store for bookkeeping
            for partiton_id, tensor in enumerate(flat_fp16_dp_partitions):
                self._param_store.add_data_parallel_partition(
                    group_id=group_id, partition_id=partiton_id, tensor=tensor)

            # add tensor partition updated by the optimizer on the current rank
            fp32_partition_tensor = flat_fp16_dp_partitions[
                self._partition_id].clone().float().detach()
            fp32_partition_tensor = fp32_partition_tensor.to(
                get_current_device())
            fp32_partition_tensor.requires_grad = True
            self._fp32_param_partition_updated_by_current_rank.append(
                fp32_partition_tensor)

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group['params'] = [fp32_partition_tensor]

            self._param_store.update_partition_size()
            self._param_store.update_partition_ownership()

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    def initialize_optimizer_states(self):
        for group_id, param_group in enumerate(self._fp16_param_groups):
            fp32_partition_param = self._fp32_param_partition_updated_by_current_rank[
                group_id]
            partition_size = self._param_store.get_partition_size(group_id)
            device = get_current_device()

            fp32_partition_grad = torch.zeros(partition_size,
                                              dtype=fp32_partition_param.dtype,
                                              device=device)
            fp32_partition_param.grad = fp32_partition_grad

        self.optimizer.step()

        for partition_tensor in self._fp32_param_partition_updated_by_current_rank:
            partition_tensor.grad = None

        return

    def _sanity_checks(self):
        assert self._all_gather_bucket_size % NCCL_COMMUNICATION_UNIT_SIZE == 0, f"argument 'all_gather_bucket_size' ({self._all_gather_bucket_size}) must be a multiple of NCCL_COMMUNICATION_UNIT_SIZE({NCCL_COMMUNICATION_UNIT_SIZE})"
        assert torch.cuda.is_available(), 'CUDA is required'

    def backward(self, loss):
        loss = self.grad_scaler.scale_loss(loss)
        loss.backward(loss, retain_graph=True)

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for param_group in self._fp16_param_groups:
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()

    def step(self, closure=None):
        pass

    def sync_grad_across_dp(self):
        # add an if-else statement here when stage 2 is ready
        self._sync_grad_across_dp_stage1()

    ########################################
    # Gradient Synchronization for Stage 1 #
    ########################################

    def _sync_grad_across_dp_stage1(self):
        self._reduce_partition_grads()

        for i, group in enumerate(self._fp16_param_groups):
            for param in group:
                if param.grad is not None:
                    self.reduce_ready_partitions_and_remove_grads(param, i)

        for group_id in range(self._param_store.num_param_groups):
            if group_id not in self.averaged_gradients or \
                self.averaged_gradients[group_id] is None:
                self.averaged_gradients[group_id] = self.get_flat_partition(
                    self.params_in_partition[group_id],
                    self.first_offset[group_id],
                    self.partition_size[group_id],
                    dtype=self.dtype,
                    device=torch.cuda.current_device(),
                    return_tensor_list=True)
            else:
                avg_new = self.get_flat_partition(
                    self.params_in_partition[group_id],
                    self.first_offset[group_id],
                    self.partition_size[group_id],
                    dtype=self.dtype,
                    device=torch.cuda.current_device(),
                    return_tensor_list=True)

                for accumulated_grad, new_avg_grad in zip(
                        self.averaged_gradients[i], avg_new):
                    accumulated_grad.add_(new_avg_grad)

        # clear the gradients
        self.zero_grad()

    def _sync_grad_across_dp_stage2(self):
        pass

    def _reduce_partition_grads(self):
        pass

    def _copy_model_grads_to_main_grads(self):
        for group_id in range(self._param_store.num_param_groups):
            for partition_id in range(self._partition_count):
                fp16_param = self._param_store.get_data_parallel_partition(
                    group_id, partition_id)
                fp32_param = self._fp32_param_partition_updated_by_current_rank[
                    group_id]
                fp32_param.grad = fp16_param.grad
