from math import sqrt

import torch
import torch.distributed as dist
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import is_model_parallel_parameter, report_memory_usage
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch._six import inf
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.distributed_c10d import _get_global_rank

from .loss_scaler import DynamicLossScaler, LossScaler

# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False


def split_half_float_double(tensors):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor"
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def get_alignment_padding(tensor_list, alignment):
    num_elements = sum([tensor.numel() for tensor in tensor_list])
    remainder = num_elements % alignment
    return (alignment - remainder) if remainder else remainder


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor


def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    return sqrt(total_norm)


def is_moe_parallel_parameter(param):
    return False


class ZeroRedundancyOptimizer_Level_1(object):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.
    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054
    For usage examples, refer to TODO: DeepSpeed Tutorial
    """

    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 cpu_offload=False,
                 mpu=None,
                 clip_grad=0.0,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):
        self.logger = get_dist_logger()
        self.verbose = verbose
        self.log(f"Reduce bucket size {reduce_bucket_size}")
        self.log(f"Allgather bucket size {allgather_bucket_size}")
        self.log(f"CPU Offload: {cpu_offload}")
        self.log(f'Round robin gradient partitioning: {round_robin_gradients}')
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        self.elastic_checkpoint = elastic_checkpoint
        if dp_process_group is None:
            dp_process_group = gpc.get_group(ParallelMode.DATA)
        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        # Load pre-built or JIT compile (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        # ZeRO stage 1 (False) or 2 (True)
        self.partition_gradients = partition_grads

        self.reduce_scatter = reduce_scatter

        self.overlap_comm = overlap_comm

        self.cpu_offload = cpu_offload

        self.deepspeed_adam_offload = cpu_offload

        self.device = torch.cuda.current_device() if not self.cpu_offload else 'cpu'

        self.dp_process_group = dp_process_group

        # expert parallel group
        self.ep_process_group = expert_parallel_group

        # data parallel group for experts
        self.expert_dp_process_group = expert_data_parallel_group

        # data parallel size for non-experts
        dp_size = dist.get_world_size(group=self.dp_process_group)

        # For MoE models this maybe different for different param group
        # It will be modified during MoE setup later in the init
        self.real_dp_process_group = [
            dp_process_group for i in range(len(self.optimizer.param_groups))
        ]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        self.contiguous_gradients = contiguous_gradients or cpu_offload

        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            self._configure_moe_settings()
        self._global_grad_norm = 0.

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = gpc.get_group(ParallelMode.MODEL)
            self.model_parallel_rank = gpc.get_local_rank(ParallelMode.MODEL)

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients

        self.extra_large_param_to_reduce = None
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients

        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [
                DeepSpeedCPUAdam], f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32. Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}. Either disable fp16_master_weights_and_gradients or enable ZeRO-2 Offload with DeepSpeedCPUAdam"

        if self.reduce_scatter:
            assert self.communication_data_type in (
                torch.float16, torch.bfloat16), f"ZeRO-2 supports only float16 or bfloat16 communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-2 with reduce scatter enabled"

        # param flattened by groups
        self.bit16_groups = []
        self.bit16_groups_flat = []

        # param partitioned by data parallel degree
        # this will contain a list of equal sized tensors
        # each of which will be updated by a different process
        self.parallel_partitioned_bit16_groups = []

        # a single 32-bit partition of the parallel partitioned parameters
        # that this process will update
        self.single_partition_of_fp32_groups = []

        # param partition info

        # These are the parameters in each group that will not be updated by this process directly
        self.params_not_in_partition = []

        # These are the parameters that will be updated by this process directly
        self.params_in_partition = []

        # Offset from the first parameter in the the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        self.first_offset = []

        # number of elements per partition in each group
        self.partition_size = []

        # align nccl all-gather send buffers to 4-bye boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        assert (allgather_bucket_size % self.nccl_start_alignment_factor ==
                0), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype

        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        self.groups_padding = []
        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            self.bit16_groups.append(param_group['params'])

            # Record padding required to align group to world size
            if partition_id == dist.get_world_size(
                    group=self.real_dp_process_group[i]) - 1:
                padding = get_alignment_padding(self.bit16_groups[i],
                                                self.partition_count[i])
            else:
                padding = 0
            self.groups_padding.append(padding)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            self.report_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer
            move_to_cpu(self.bit16_groups[i])
            self.report_memory_usage(f"After moving param group {i} to CPU")

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i],
                    dist.get_world_size(group=self.real_dp_process_group[i])
                )
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # create flat buffer in CPU and move to GPU
            self.bit16_groups_flat.append(
                self.flatten_dense_tensors_aligned(
                    self.round_robin_bit16_groups[i],
                    self.nccl_start_alignment_factor *
                    dist.get_world_size(group=self.real_dp_process_group[i])).cuda(
                        torch.cuda.current_device()))
            self.report_memory_usage(f"After flattening and moving param group {i} to GPU")

            self.report_memory_usage(
                f"After Flattening and after emptying param group {i} cache")

            # set model bit16 weight to slices of flattened buffer
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            data_parallel_partitions = self.get_data_parallel_partitions(
                self.bit16_groups_flat[i],
                i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() %
                        (2 * self.nccl_start_alignment_factor) == 0)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() %
                        (2 * self.nccl_start_alignment_factor) == 0)

            # a partition of the fp32 master weights that will be updated by this process
            if not fp16_master_weights_and_gradients:
                self.single_partition_of_fp32_groups.append(
                    self.parallel_partitioned_bit16_groups[i][partition_id].to(
                        self.device).clone().float().detach())
            else:
                self.single_partition_of_fp32_groups.append(
                    self.parallel_partitioned_bit16_groups[i][partition_id].to(
                        self.device).clone().half().detach())

            # modify optimizer of have flat master weight
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(
                group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i],
                partition_size,
                partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        # for rank in range(dist.get_world_size()):
        #     if dist.get_rank() == rank:
        #         print(
        #             f"Rank: {rank} partition count {self.partition_count} and sizes{[(p.numel(), self.is_moe_param_group[i] if hasattr(self, 'is_moe_param_group') else False) for i,p in enumerate(self.single_partition_of_fp32_groups)]} "
        #         )
        #         dist.barrier()
        # exit(0)
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.reduction_stream = torch.cuda.Stream()
        self.cpu_computation_stream = torch.cuda.Stream()
        self.copy_grad_stream = torch.cuda.Stream()
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False

        # simplified param id
        self.param_id = {}

        # interesting code: unique ids being assigned to individual parameters
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = torch.zeros(
                largest_param_numel,
                device=self.device,
                dtype=self.dtype).pin_memory()
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(
                largest_param_numel,
                device=torch.cuda.current_device(),
                dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i,
                                       self.params_in_partition[i],
                                       self.first_offset[i],
                                       self.partition_size[i])

        # mapping from parameter to partition that it belongs to
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        self.total_grads_in_partition = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # stores the offset at which a parameter gradient needs to be inserted in a partition
        self.grad_partition_insertion_offset = {}

        # the offset in the gradient at which it must be inserted at the beginning of the partition
        self.grad_start_offset = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}

        # store index of first parameter in each partition
        self.first_param_index_in_partition = {}

        # initializes all data structures for implementing gradient partitioning
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        self.reset_partition_gradient_structures()

        # creates backward hooks for gradient partitioning
        if self.partition_gradients or self.overlap_comm:
            self.create_reduce_and_remove_grad_hooks()

        # we may have a way of fusing dynamic scale. Do not support for now
        if self.dtype == torch.float or self.dtype == torch.bfloat16 or not dynamic_loss_scale:
            loss_scale_value = 1.0 if (
                (self.dtype == torch.float) or
                (self.dtype == torch.bfloat16)) else static_loss_scale

            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=loss_scale_value)
            cur_iter = 0
        else:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        self.report_memory_usage("Before initializing optimizer states")
        self.initialize_optimizer_states()
        self.report_memory_usage("After initializing optimizer states")

        self.log(f"optimizer state initialized")

        self.report_memory_usage(f"After initializing ZeRO optimizer")

    def is_moe_group(self, group):
        return 'moe' in group and group['moe']

    def _configure_moe_settings(self):
        assert self.contiguous_gradients, "Contiguous Gradients in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"
        assert self.reduce_scatter, "Reduce Scatter in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"

        assert any([self.is_moe_group(group) for group in self.optimizer.param_groups]
                   ), "The model has moe layers, but None of the param groups are marked as MoE. Create a param group with 'moe' key set to True before creating optimizer"
        self.is_moe_param_group = []
        for i, group in enumerate(self.optimizer.param_groups):
            if self.is_moe_group(group):
                assert all([is_moe_parallel_parameter(param) for param in group['params']]
                           ), "All params in MoE group must be MoE params"
                self.real_dp_process_group[i] = self.expert_dp_process_group[
                    group['name']]
                self.partition_count[i] = dist.get_world_size(
                    group=self.expert_dp_process_group[group['name']])
                self.is_moe_param_group.append(True)
            else:
                self.is_moe_param_group.append(False)

        assert self.expert_dp_process_group is not None, "Expert data parallel group should be configured with MoE"
        assert self.ep_process_group is not None, "Expert parallel group should be configured with MoE"

    def _update_model_bit16_weights(self, group_index):
        updated_params = self.unflatten(self.bit16_groups_flat[group_index],
                                        self.round_robin_bit16_groups[group_index])
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data

        # set model fp16 weight to slices of reordered flattened buffer
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    def _round_robin_reorder(self, tensor_list, num_partitions):

        # disable round robin if need to debug something
        # return tensor_list, list(range(len(tensor_list)))

        partition_tensors = {}

        for i, tensor in enumerate(tensor_list):
            j = i % num_partitions
            if not j in partition_tensors:
                partition_tensors[j] = []
            partition_tensors[j].append((i, tensor))

        reordered_tensors = []
        reordered_indices = {}

        for partition_index in partition_tensors.keys():
            for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                reordered_indices[original_index] = len(reordered_tensors)
                reordered_tensors.append(tensor)

        return reordered_tensors, reordered_indices

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

    def initialize_optimizer_states(self):

        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(
                int(self.partition_size[i]),
                dtype=self.single_partition_of_fp32_groups[i].dtype,
                device=self.device)
            self.single_partition_of_fp32_groups[
                i].grad = single_grad_partition.pin_memory(
            ) if self.cpu_offload else single_grad_partition

        self.optimizer.step()

        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  # class init

        return

    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################
    def allreduce_gradients(self, pipeline_parallel=False):
        if not self.partition_gradients:
            world_size = dist.get_world_size(self.dp_process_group)
            my_rank = dist.get_rank(self.dp_process_group)

            # with PP we must create ipg buffer, since backward is handled outside zero
            if pipeline_parallel and self.contiguous_gradients:
                self.ipg_buffer = []
                buf_0 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_0)
                self.ipg_index = 0

            if not self.overlap_comm:
                for i, group in enumerate(self.bit16_groups):
                    for param in group:
                        if param.grad is not None:
                            self.reduce_ready_partitions_and_remove_grads(param, i)
        # reduce any pending grads in either hook/non-hook case
        self.overlapping_partition_gradients_reduce_epilogue()

    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):

        for i, param_group in enumerate(self.round_robin_bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.total_grads_in_partition[i][partition_id] = 0
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][
                    partition_id] = self.get_first_param_index(
                        i,
                        param_group,
                        partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        #self.log("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        if self.overlap_comm:
            torch.cuda.synchronize()
            # It is safe to clear previously reduced grads of other partitions
            self._clear_previous_reduced_grads()

        if self.cpu_offload is False:
            for i, _ in enumerate(self.bit16_groups):

                if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
                    self.averaged_gradients[i] = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.dtype,
                        device=torch.cuda.current_device(),
                        return_tensor_list=True)
                else:
                    avg_new = self.get_flat_partition(self.params_in_partition[i],
                                                      self.first_offset[i],
                                                      self.partition_size[i],
                                                      dtype=self.dtype,
                                                      device=torch.cuda.current_device(),
                                                      return_tensor_list=True)

                    for accumulated_grad, new_avg_grad in zip(self.averaged_gradients[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)

        self._release_ipg_buffers()

        # No need to keep the gradients anymore.
        # All gradients required by the step
        # are in self.averaged_gradients
        self.zero_grad()
        self.report_memory_usage(f"End ipg_epilogue")

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def reset_partition_gradient_structures(self):
        for i, _ in enumerate(self.bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][
                    partition_id] = self.total_grads_in_partition[i][partition_id]

                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):
        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1

        partition_size = self.partition_size[i]

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for param in param_group:

            param_size = param.numel()
            param_id = self.get_param_id(param)

            if (current_index >= start_index and current_index < end_index):
                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][
                    param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif start_index > current_index and start_index < (current_index +
                                                                param_size):
                assert (first_offset == 0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def create_reduce_and_remove_grad_hooks(self):
        self.grad_accs = []
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)

                    wrapper(param, i)

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        self.report_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}"
        )

    # create a flat tensor aligned at the alignment boundary
    def flatten_dense_tensors_aligned(self, tensor_list, alignment):
        num_elements = sum(t.numel() for t in tensor_list)
        remaining = num_elements % alignment

        if remaining:
            elements_to_add = alignment - remaining
            pad_tensor = torch.zeros(elements_to_add,
                                     device=tensor_list[0].device,
                                     dtype=tensor_list[0].dtype)
            padded_tensor_list = tensor_list + [pad_tensor]
        else:
            padded_tensor_list = tensor_list

        return self.flatten(padded_tensor_list)

    ############### Independent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads",
                                         param.numel())
            self.reduce_ipg_grads()
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads",
                                         param.numel())

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        if param.numel() > self.reduce_bucket_size:
            self.extra_large_param_to_reduce = param

        elif self.contiguous_gradients:
            # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
            new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(
                0,
                self.elements_in_ipg_bucket,
                param.numel())
            new_grad_tensor.copy_(param.grad.view(-1))
            param.grad.data = new_grad_tensor.data.view_as(param.grad)

        self.elements_in_ipg_bucket += param.numel()

        assert param.grad is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        self.grads_in_ipg_bucket.append(param.grad)
        self.params_in_ipg_bucket.append((i, param, param_id))

        # make sure the average tensor function knows how to average the gradients
        if is_moe_parallel_parameter(param):
            self.ipg_bucket_has_moe_params = True

        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def gradient_reduction_w_predivide(self, tensor):

        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        tensor_to_allreduce = tensor

        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            if self.gradient_predivide_factor != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor / dp_world_size)
        else:
            tensor_to_allreduce.div_(dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def average_tensor(self, tensor):
        if self.overlap_comm:
            torch.cuda.synchronize()
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id = -1

            process_group = self.dp_process_group
            # count = 0
            for i, param, param_id in self.params_in_ipg_bucket:

                process_group = self.dp_process_group
                # Averages gradients at parameter level if ipg has a moe param
                # Otherwise averaging is done at the entire buffer level at the end of the loop
                # MoE param have different groups
                if self.ipg_bucket_has_moe_params:
                    process_group = self.expert_dp_process_group[
                        param.group_name] if is_moe_parallel_parameter(
                            param) else self.dp_process_group
                    param.grad.data.div_(dist.get_world_size(group=process_group))

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids]
                           ), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]
                # Get all partition ids + their offsets
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # if dist.get_rank() == 0 and count < 100:
                    #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                    # count += 1

                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    if partition_id == prev_id:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id = partition_id

            if not self.ipg_bucket_has_moe_params:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))

            async_handles = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                # if dist.get_rank() == 0:
                #     print(f"Rank {dist.get_rank()} rank offset id {i} real dp size {dist.get_world_size(group=real_dp_process_group[i])} and dst: {dst}")
                # dist.barrier()
                # dist.barrier()
                dst_rank = _get_global_rank(real_dp_process_group[i], dst)
                async_handle = dist.reduce(grad_slice,
                                           dst=dst_rank,
                                           group=real_dp_process_group[i],
                                           async_op=True)
                async_handles.append(async_handle)

            for handle in async_handles:
                handle.wait()

    ##############################################################################
    ############################# CPU Offload Methods#############################
    ##############################################################################
    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        current_offset = 0

        for i, tensor in enumerate(tensor_list):
            param_id = self.get_param_id(tensor)
            param_start_offset = 0

            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                param_start_offset = first_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_offset):
                num_elements = partition_size - current_offset

            self.grad_position[param_id] = [
                int(group_id),
                int(param_start_offset),
                int(current_offset),
                int(num_elements)
            ]
            current_offset += num_elements

    def update_overflow_tracker_for_param_grad(self, param):
        if param.grad is not None and self._has_inf_or_nan(param.grad.data):
            self.local_overflow = True

    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # copy to a preexisiting buffer to avoid memory allocation penalty
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(
            0,
            0,
            param.numel())

        # buffer for storing gradients for this parameter in CPU
        def buffer_to_accumulate_to_in_cpu():
            if not self.fp16_master_weights_and_gradients:
                return torch.zeros(param.numel(),
                                   dtype=param.dtype,
                                   device=self.device).pin_memory()
            else:
                return self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(
                    0,
                    dest_offset,
                    num_elements)

        # accumulate gradients into param.grad or parts of it that belongs to this partition
        def accumulate_gradients():
            if not self.fp16_master_weights_and_gradients:
                dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1),
                                  non_blocking=True)
                param.grad.data.view(-1).add_(dest_buffer)
            else:
                dest_buffer.narrow(0,
                                   source_offset,
                                   num_elements).copy_(
                                       self.accumulated_grads_in_cpu[param_id].view(-1),
                                       non_blocking=True)
                param.grad.data.view(-1).narrow(
                    0,
                    source_offset,
                    num_elements).add_(dest_buffer.narrow(0,
                                                          source_offset,
                                                          num_elements))

        # move accumulated gradients back to CPU
        def copy_gradients_to_cpu():
            if not self.fp16_master_weights_and_gradients:
                self.accumulated_grads_in_cpu[param_id].data.copy_(
                    param.grad.data.view(-1),
                    non_blocking=True)
            else:
                self.accumulated_grads_in_cpu[param_id].data.copy_(
                    param.grad.data.view(-1).narrow(0,
                                                    source_offset,
                                                    num_elements),
                    non_blocking=True)

        if param_id not in self.accumulated_grads_in_cpu:
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()

        if self.micro_step_id > 0:
            accumulate_gradients()

        # at the boundary we will send 32bit directly
        if not self.is_gradient_accumulation_boundary:
            copy_gradients_to_cpu()

    def set_norm_for_param_grad(self, param):
        param_id = self.get_param_id(param)
        accumulated_grad = self.accumulated_grads_in_cpu[
            param_id] if self.gradient_accumulation_steps > 1 else param.grad

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        accumulated_grad = param.grad

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(
            0,
            dest_offset,
            num_elements)

        src_tensor = param.grad.view(-1).narrow(0, source_offset, num_elements)
        if not self.fp16_master_weights_and_gradients:
            src_tensor = src_tensor.float()

        dest_tensor.copy_(src_tensor, non_blocking=True)
        param.grad = None  # offload only

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
                # so they have no norm_for_param_grads
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item()**2
                else:
                    # As unused parameters in modules may not be expected sometimes,
                    # add an explicit error msg when it occurred and an option to
                    # avoid the error
                    assert self.ignore_unused_parameters, """
                        This assert indicates that your module has parameters that
                        were not used in producing loss.
                        You can avoid this assert by
                        (1) enable ignore_unused_parameters option in zero_optimization config;
                        (2) making sure all trainable parameters and `forward` function
                            outputs participate in calculating loss.
                    """

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                        op=torch.distributed.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    ############################################################################################
    def copy_grads_in_partition(self, param):
        if self.cpu_offload:

            if self.gradient_accumulation_steps > 1:
                self.async_accumulate_grad_in_cpu_via_gpu(param)

            if self.is_gradient_accumulation_boundary:
                self.set_norm_for_param_grad_in_gpu(param)

                self.update_overflow_tracker_for_param_grad(param)

                self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)

            return
        #print(f"ID {self.get_param_id(param)} grad norm {param.grad.norm()}")
        if self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()

            self.report_memory_usage(f"before copying {total_size} gradients into partition")
            self.grads_in_partition = torch.empty(int(total_size),
                                                  dtype=self.dtype,
                                                  device=torch.cuda.current_device())
            self.report_memory_usage(f"after copying {total_size} gradients into partition")

        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(
            0,
            self.grads_in_partition_offset,
            param.numel())
        new_grad_tensor.copy_(param.grad.view(-1))
        param.grad.data = new_grad_tensor.data.view_as(param.grad)
        #print(f"Grad norm after copy to contiguous_buffer {param.grad.data.norm()}")
        self.grads_in_partition_offset += param.numel()

    def reduce_ipg_grads(self):
        if self.contiguous_gradients:
            if self.extra_large_param_to_reduce is not None:
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                _, _, param_id = self.params_in_ipg_bucket[0]
                assert self.get_param_id(
                    self.extra_large_param_to_reduce) == param_id, "param in ipg bucket does not match extra-large param"
                self.average_tensor(self.extra_large_param_to_reduce.grad.view(-1))
                self.extra_large_param_to_reduce = None
            else:
                self.average_tensor(self.ipg_buffer[self.ipg_index])
        else:
            self.buffered_reduce_fallback(
                None,
                self.grads_in_ipg_bucket,
                elements_per_buffer=self.elements_in_ipg_bucket)

        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            torch.cuda.synchronize()
            #            stream = self.copy_grad_stream
            stream = torch.cuda.current_stream()
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:

                assert self.params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"

                self.params_already_reduced[param_id] = True

                if self.partition_gradients:
                    if not self.is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.previous_reduced_grads is None:
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            param.grad = None  # only if self.partition_gradients
                    elif self.contiguous_gradients:
                        self.copy_grads_in_partition(param)
                else:  # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients:
                        self.copy_grads_in_partition(param)

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0
        #####################################################################

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):
        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None  # dead code

    def get_grads_to_reduce(self, i, partition_id):
        def get_reducible_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(
                total_elements - start,
                self.partition_size[i] -
                self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0,
                                                             int(start),
                                                             int(num_elements))
            else:
                if num_elements == total_elements:
                    return grad.clone()
                else:
                    return grad.clone().contiguous().view(-1).narrow(
                        0,
                        int(start),
                        int(num_elements))

        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducible_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zero_like(param)

    ######################Reduction Related Methods##############################
    def allreduce_bucket(self,
                         bucket,
                         communication_data_type=torch.float16,
                         rank=None,
                         log=None):
        rank = None
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if pg_correctness_test:
            communication_data_type = torch.float32

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = _get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    def _clear_previous_reduced_grads(self):
        if self.previous_reduced_grads is not None:
            for param in self.previous_reduced_grads:
                param.grad = None  # overlap enabled
            self.previous_reduced_grads = None

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        if self.overlap_comm:
            torch.cuda.synchronize()
            # It is safe to clear the previously reduced grads of other partitions
            self._clear_previous_reduced_grads()
            stream = self.reduction_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self,
                            bucket,
                            numel_per_bucket=500000000,
                            rank=None,
                            log=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []

        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    # allows using reduction of gradients instead of using all_reduce

    def buffered_reduce_fallback(self,
                                 rank,
                                 grads,
                                 elements_per_buffer=500000000,
                                 log=None):
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket,
                                     numel_per_bucket=elements_per_buffer,
                                     rank=rank,
                                     log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor, group_id):
        partitions = []

        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if (current_index >= start_index and current_index < end_index):
                params_in_partition.append(tensor)

            elif start_index > current_index and start_index < (current_index +
                                                                tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.bit16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None  # epilogue and in step
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None:
            pass
        else:
            torch.distributed.all_reduce(tensor=tensor,
                                         op=op,
                                         group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.
        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            #self.log(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2
            # Sum across all model parallel GPUs.
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self,
                           tensor_list,
                           first_offset,
                           partition_size,
                           dtype,
                           device,
                           return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)

            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                    0,
                    int(tensor_offset),
                    int(num_elements)))
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(
                torch.zeros(int(partition_size - current_size),
                            dtype=dtype,
                            device=device))

        if return_tensor_list:
            return flat_tensor_list

        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None  # in step

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = -1

        self.report_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:

            self.log(
                "[deepscale] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, "
                "reducing to {}".format(dist.get_rank(),
                                        prev_scale,
                                        self.loss_scale))

            self.report_memory_usage('After overflow before clearing gradients')
            self.zero_grad()
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            self.report_memory_usage('After overflow after clearing gradients')

            return

        norm_groups = []
        single_partition_grad_groups = []
        skip = False
        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                norm_groups.append(
                    self.complete_grad_norm_calculation_for_cpu_offload(
                        self.params_in_partition[i]))
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            else:
                norm_groups.append(
                    self.get_grad_norm_direct(self.averaged_gradients[i],
                                              self.params_in_partition[i]))

                # free gradients for all the parameters that are not updated by this process
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(
                        group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(
                            self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

            single_partition_grad_groups.append(single_grad_partition)

        if self.has_moe_layers:
            self._average_expert_grad_norms(norm_groups)

        self._global_grad_norm = get_global_norm(norm_groups)
        self.unscale_and_clip_grads(single_partition_grad_groups, self._global_grad_norm)

        if self.deepspeed_adam_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
                bit16_param_groups = [
                    bit16_partitions[partition_id]
                    for bit16_partitions in self.parallel_partitioned_bit16_groups
                ]
                self.optimizer.step(fp16_param_groups=bit16_param_groups)
            else:
                self.optimizer.step()
                for bit16_partitions, fp32_partition in zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups):
                    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
        else:
            self.optimizer.step()

            # get rid of the fp32 gradients. Not needed anymore
            if not self.cpu_offload:
                for group in self.single_partition_of_fp32_groups:
                    group.grad = None  # in step

            for bit16_partitions, fp32_partition in zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups):
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

        if self.cpu_offload:
            self.reset_cpu_buffers()

        # gather the updated weights from everyone
        for group_id, partitioned_params in enumerate(self.parallel_partitioned_bit16_groups):

            # Sequential AllGather Best of both worlds
            dp_world_size = dist.get_world_size(
                group=self.real_dp_process_group[group_id])
            num_shards = max(
                1,
                partitioned_params[partition_id].numel() * dp_world_size //
                self.allgather_bucket_size)

            shard_size = partitioned_params[partition_id].numel() // num_shards

            # Enforce nccl/rccl alignment of start location of each shard
            shard_size = shard_size - (shard_size % self.nccl_start_alignment_factor)

            num_elements = shard_size

            assert shard_size * num_shards <= partitioned_params[partition_id].numel()

            for shard_id in range(num_shards):

                if shard_id == (num_shards - 1):
                    num_elements = partitioned_params[partition_id].numel(
                    ) - shard_id * shard_size

                shard_list = []
                for dp_id in range(dp_world_size):
                    curr_shard = partitioned_params[dp_id].narrow(
                        0,
                        shard_id * shard_size,
                        num_elements).detach()
                    shard_list.append(curr_shard)
                dist.all_gather(shard_list,
                                shard_list[partition_id],
                                group=self.real_dp_process_group[group_id])

        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            self._update_model_bit16_weights(i)

        self.report_memory_usage('After zero_optimizer step')

        return

    def _average_expert_grad_norms(self, norm_groups):
        for i, norm in enumerate(norm_groups):
            if self.is_moe_param_group[i]:
                scaled_norm = norm * 1.0 / float(
                    dist.get_world_size(group=self.real_dp_process_group[i]))
                scaled_norm_tensor = torch.tensor(scaled_norm,
                                                  device='cuda',
                                                  dtype=torch.float)
                dist.all_reduce(scaled_norm_tensor, group=self.real_dp_process_group[i])
                norm_groups[i] = scaled_norm_tensor.item()

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial(
            )
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            '''This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process'''
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

        else:
            params = []
            for group in self.bit16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.cuda.ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu,
                                        op=torch.distributed.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:
        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    # Return group tensor after removing paddings that are added for alignment to DP world size.
    # This method works on the assumption that each group contains a single flattened tensor.
    def _get_groups_without_padding(self, groups_with_padding):
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            lean_length = group.numel() - self.groups_padding[i]
            groups_without_padding.append(group[:lean_length])

        return groups_without_padding

    # Return optimizer state after removing paddings that are added for alignment.
    def _get_state_without_padding(self, state_with_padding, padding):
        lean_state = {}
        for key, value in state_with_padding.items():
            if torch.is_tensor(value):
                lean_length = value.numel() - padding
                lean_state[key] = value[:lean_length]
            else:
                lean_state[key] = value

        return lean_state

    # Return base optimizer states.
    # This method assumes that each param group contains a single flattened tensor.
    def _get_base_optimizer_state(self):
        optimizer_groups_state = []
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            lean_optimizer_state = self._get_state_without_padding(
                self.optimizer.state[p],
                self.groups_padding[i])
            optimizer_groups_state.append(lean_optimizer_state)

        return optimizer_groups_state

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow

        if self.elastic_checkpoint:
            state_dict['base_optimizer_state'] = self._get_base_optimizer_state()
        else:
            state_dict['base_optimizer_state'] = self.optimizer.state_dict()

        # Remove paddings for DP alignment to enable loading for other alignment values
        fp32_groups_without_padding = self._get_groups_without_padding(
            self.single_partition_of_fp32_groups)
        state_dict['single_partition_of_fp32_groups'] = fp32_groups_without_padding

        state_dict['zero_stage'] = 1
        state_dict['partition_count'] = self.partition_count

        return state_dict

    # Restore base optimizer fp32 weights from elastic checkpoint by:
    # 1) Merging fp32 weights from checkpoints of all partitions
    # 2) Extracting fp32 weights for current partition from merged weights
    # 3) Using extracted weights to update base optimizer weights directly.
    def _restore_from_elastic_fp32_weights(self, all_state_dict):
        merged_single_partition_of_fp32_groups = []

        for i in range(len(self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            merged_partitions = [
                sd['single_partition_of_fp32_groups'][i] for sd in all_state_dict
            ]
            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(
                    group_name=self.optimizer.param_groups[i]['name'])
                merged_partitions = [merged_partitions[i] for i in ranks]
            flat_merged_partitions = self.flatten_dense_tensors_aligned(
                merged_partitions,
                self.nccl_start_alignment_factor *
                dist.get_world_size(group=self.real_dp_process_group[i]))
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, i)
            merged_single_partition_of_fp32_groups.append(dp_partitions[partition_id])

        for current, saved in zip(self.single_partition_of_fp32_groups, merged_single_partition_of_fp32_groups):
            current.data.copy_(saved.data)

    # Restore base optimizer fp32 weights from ZeRO fp16 or bfloat16 weights
    def _restore_from_bit16_weights(self):
        for group_id, (bit16_partitions, fp32_partition) in enumerate(zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
            fp32_partition.data.copy_(bit16_partitions[partition_id].data)

    # Refresh the fp32 master params from the fp16 or bfloat16 copies.
    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    # Extract optimizer state for current partition from merged states of all partitions
    def _partition_base_optimizer_state(self, state_key, all_partition_states, group_id):
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
        alignment = dist.get_world_size(group=self.real_dp_process_group[group_id])
        if torch.is_tensor(all_partition_states[0]):
            flat_merged_partitions = self.flatten_dense_tensors_aligned(
                all_partition_states,
                alignment)
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions,
                                                              group_id)
            return dp_partitions[partition_id]
        else:
            # Assume non-tensor states are not partitioned and equal across ranks, so return first one
            return all_partition_states[0]

    def _restore_base_optimizer_state(self, base_optimizer_group_states):
        if type(base_optimizer_group_states) == dict:
            base_optimizer_group_states = base_optimizer_group_states['state']
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            for key, saved in base_optimizer_group_states[i].items():
                if torch.is_tensor(self.optimizer.state[p][key]):
                    dst_tensor = self.optimizer.state[p][key]
                    src_tensor = _get_padded_tensor(saved, dst_tensor.numel())
                    self.optimizer.state[p][key].data.copy_(src_tensor.data)
                else:
                    self.optimizer.state[p][key] = saved

    def get_ep_ranks(self, rank=0, group_name=None):
        from deepspeed.utils import groups
        expert_parallel_size_ = groups.get_expert_parallel_world_size(group_name)
        world_size = groups.get_data_parallel_world_size()
        rank = groups.get_expert_parallel_rank(group_name)
        ranks = range(rank, world_size, expert_parallel_size_)
        return list(ranks)

    # Restore base optimizer state from elastic checkpoint by
    # 1) Merging optimizer state from checkpoints of all partitions
    # 2) Extracting optimizer state for current partition from the merged state
    # 3) Using the extracted value to directly update the base optimizer.
    def _restore_elastic_base_optimizer_state(self, all_state_dict):
        base_optimizer_group_states = []
        for i in range(len(self.optimizer.param_groups)):
            partition_states = {}
            all_partition_group_states = [
                sd['base_optimizer_state'][i] for sd in all_state_dict
            ]

            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(
                    group_name=self.optimizer.param_groups[i]['name'])
                all_partition_group_states = [
                    all_partition_group_states[i] for i in ranks
                ]

            for key in all_partition_group_states[0].keys():
                all_partition_states = [
                    all_states[key] for all_states in all_partition_group_states
                ]
                partition_states[key] = self._partition_base_optimizer_state(
                    key,
                    all_partition_states,
                    i)
            base_optimizer_group_states.append(partition_states)

        self._restore_base_optimizer_state(base_optimizer_group_states)

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False):
        r"""Loading ZeRO checkpoint
        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
        """
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """

        # I think it should actually be ok to reload the optimizer before the model.
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        self.loss_scaler = current_rank_sd['loss_scaler']
        self.dynamic_loss_scale = current_rank_sd['dynamic_loss_scale']
        self.overflow = current_rank_sd['overflow']

        ckpt_is_rigid = isinstance(current_rank_sd['base_optimizer_state'], dict)

        if load_optimizer_states:
            if ckpt_is_rigid:
                # loading rigid ckpt into either rigid or elastic exec
                self.optimizer.load_state_dict(current_rank_sd['base_optimizer_state'])
            else:
                if self.elastic_checkpoint:
                    # loading elastic into elastic exec
                    self._restore_elastic_base_optimizer_state(state_dict_list)
                else:
                    # loading an elastic checkpoint into rigid exec
                    self._restore_base_optimizer_state(
                        current_rank_sd['base_optimizer_state'])

        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 1 if changing DP degree and option 2 otherwise.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.

        if load_from_fp32_weights:
            # option 2 from above
            if self.elastic_checkpoint and not ckpt_is_rigid:
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                # For non-elastic checkpoint, simply copying from saved weights of current rank is sufficient.
                for current, saved in zip(self.single_partition_of_fp32_groups, current_rank_sd['single_partition_of_fp32_groups']):
                    src_tensor = _get_padded_tensor(saved, current.numel())
                    current.data.copy_(src_tensor.data)
        else:
            # option 1 from above
            self._restore_from_bit16_weights()

    def log(self, message):
        if self.verbose:
            self.logger.info(message, ranks=[0])

    def report_memory_usage(self, message):
        if self.verbose:
            report_memory_usage(message)

    def clip_grad_norm(self, *args, **kwargs):
        # dummy function to retain the same function interface
        # as ColossalaiOptimizer for compatibility
        pass
