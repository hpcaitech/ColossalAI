#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.optim import Optimizer

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import OPTIMIZER_WRAPPERS
from colossalai.utils import get_current_device, print_rank_0


def get_alignment_padding(flattened_lean_size, sub_partition_id, sub_partition_size):
    sub_partition_high_limit = (sub_partition_id + 1) * sub_partition_size
    if sub_partition_high_limit <= flattened_lean_size:
        return 0
    else:
        return min(sub_partition_size, sub_partition_high_limit - flattened_lean_size)


def get_group_alignment_padding(tensor_list, sub_partition_size, sub_partition_count):
    group_paddings = []
    flattened_size = sum([tensor.numel() for tensor in tensor_list])
    for i in range(sub_partition_count):
        padding = get_alignment_padding(flattened_size, i, sub_partition_size)
        group_paddings.append(padding)

    return group_paddings


def _single_range_check(current_index, start_index, end_index, tensor_size):
    offset = 0
    if (current_index >= start_index) and (current_index < end_index):
        # Fully inside bounds
        return True, offset
    elif (start_index > current_index) and (start_index < (current_index + tensor_size)):
        # Partially contained, compute offset
        offset = start_index - current_index
        return True, offset
    else:
        return False, offset


def _range_check(current_index, element_intervals, tensor_size):
    results = []
    for comm_idx, interval in enumerate(element_intervals):
        start_index, end_index = interval
        contained, offset = _single_range_check(
            current_index, start_index, end_index, tensor_size)
        if contained:
            results.append((contained, offset, comm_idx))
    if len(results) == 0:
        return [(False, 0, -1)]
    return results


@OPTIMIZER_WRAPPERS.register_module
class ZeroRedundancyOptimizer_Level_1(Optimizer):
    """
    ZeroRedundancyOptimizer_Level_1 designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    This version aligns with stage-1 in the paper above.
    """

    def __init__(self,
                 init_optimizer: Optimizer,
                 dp_parallel_mode: ParallelMode = ParallelMode.DATA,
                 max_elements_per_comm=5e8,
                 verbose=False
                 ):
        # TODO: this class does not work with fp16 AMP_TYPE.PARALLEL, fix it
        assert get_current_device() != 'cpu', 'ZeRO optimizer cannot be used on CPU only'

        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors
        self.optimizer = init_optimizer
        self.dp_parallel_mode = dp_parallel_mode
        self.verbose = verbose

        # for compatibility with pytorch optim
        self.defaults = init_optimizer.defaults

        # param flattened by groups
        self._param_groups = []
        self._param_groups_flat = []

        # parallel_sub_partitioned_fp16_groups[group-idx] -> [comm-ids] -> [rank-ids]
        self.parallel_sub_partitioned_groups = []
        # same underlying data as above but viewed as: [groups] -> [rank-ids] -> [comm-ids]
        self.parallel_comm_sub_partitioned_groups = []

        # param partition info
        # parameters in each group that will not be updated by this process directly
        self.params_not_local = []

        # parameters that will be updated by this process directly
        self.params_in_rank_sub_partitions = []

        # parameter offsets for parameters in sub-partitions. Parameter
        # boundaries may not align with sub-partition boundaries
        # so we need to keep track of the offsets
        self.params_in_rank_sub_partitions_offsets = []

        # number of elements per sub-partition in each group
        self.sub_partition_sizes = []

        # number of communication intervals for each group
        self.num_comm_intervals_per_group = []

        self.local_rank = gpc.get_local_rank(self.dp_parallel_mode)
        self.partition_count = self.world_size = gpc.get_world_size(
            self.dp_parallel_mode)

        self.group_paddings = []
        self.default_device = self.optimizer.param_groups[0]['params'][0].device

        # max elems per param group
        self.max_elems_per_comm = []

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self._param_groups.append(param_group['params'])

            # calculate best max elements per comm based to minimize padding
            self.max_elems_per_comm.append(
                self.best_max_elems_per_comm(
                    num_elements=sum(t.numel() for t in self._param_groups[i]),
                    max_elements_per_comm=max_elements_per_comm
                )
            )

            # flattens all tensors into single 1d tensor aligned with sub-partition size for later dividing
            # RS: create aligned sub-partitions
            flat_aligned_params = self.flatten_dense_tensors_sub_partition_aligned(
                tensor_list=self._param_groups[i],
                max_elements_per_comm=self.max_elems_per_comm[i],
            )
            self._param_groups_flat.append(flat_aligned_params)

            updated_params = self.unflatten(self._param_groups_flat[i],
                                            self._param_groups[i])
            for p, q in zip(self._param_groups[i], updated_params):
                p.data = q.data

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            # RS: split into two layer list -> [comm-id] -> [sub-partitions per rank]
            comm_partitions, dp_sub_partitions, element_intervals, sub_partition_size, num_comm_intervals = \
                self.get_data_parallel_sub_partitions(
                    tensor=self._param_groups_flat[i],
                    max_elements_per_comm=self.max_elems_per_comm[i],
                )
            self.parallel_comm_sub_partitioned_groups.append(
                comm_partitions)  # comm -> rank
            self.parallel_sub_partitioned_groups.append(
                dp_sub_partitions)  # rank -> comm
            self.sub_partition_sizes.append(sub_partition_size)
            self.num_comm_intervals_per_group.append(num_comm_intervals)

            # Compute sub_partition paddings
            sub_partition_paddings = get_group_alignment_padding(
                tensor_list=self._param_groups[i],
                sub_partition_size=sub_partition_size,
                sub_partition_count=num_comm_intervals * self.partition_count)
            self.group_paddings.append(sub_partition_paddings)

            # modify optimizer of have flat master weight
            param_group['params'] = self.parallel_sub_partitioned_groups[i][self.local_rank]

            # RS: divide up the sub-partitions and keep track of offsets for each param
            # partition_size = len(self.fp16_groups_flat[i]) / dist.get_world_size(group=self.dp_process_group)
            params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local = self.get_all_sub_partition_info(
                tensor_list=self._param_groups[i],
                all_element_intervals=element_intervals,
            )

            self.params_in_rank_sub_partitions.append(
                params_in_rank_sub_partition)
            self.params_not_local.append(params_not_local)
            self.params_in_rank_sub_partitions_offsets.append(
                params_in_rank_sub_partitions_offsets)

        self.local_sub_partitions_of_groups = [
            group[self.local_rank] for group in self.parallel_sub_partitioned_groups]
        self._initialize_optimizer_states()

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, value):
        self.optimizer.state = value

    @property
    def param_groups(self):
        # LSG: return the full param groups instead of local partitions
        # of the param groups for compatibility with torch.cuda.amp
        param_groups = []

        for group_id, group in enumerate(self.optimizer.param_groups):
            group_containing_all_param = {
                'params': self._param_groups[group_id],
                **{k: v for k, v in group.items() if k != 'params'}
            }
            # LSG: for compatibility with unknown bug with lr scheduler
            # TODO: fix this
            group_containing_all_param.setdefault('initial_lr', group['lr'])
            param_groups.append(group_containing_all_param)
        return param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def _initialize_optimizer_states(self):
        for group_idx, group in enumerate(self.local_sub_partitions_of_groups):
            for idx, sub_partition_param in enumerate(group):
                sub_partition_grad = torch.zeros(int(
                    self.sub_partition_sizes[group_idx]),
                    dtype=sub_partition_param.dtype).cuda()
                sub_partition_param.grad = sub_partition_grad

        self.optimizer.step()

        # LSG: comment out for compatibility with torch.cuda.amp
        # for group in self.local_sub_partitions_of_groups:
        #     for idx, sub_partition_param in enumerate(group):
        #         sub_partition_param.grad = None

    def best_max_elems_per_comm(self, num_elements, max_elements_per_comm):
        # if we use max-elems-per-comm as is, how many comm intervals will there be
        max_comm_intervals = math.ceil(num_elements / max_elements_per_comm)
        padding_for_max_comm = (max_elements_per_comm *
                                max_comm_intervals) - num_elements

        # if we use 1 less comm interval how much extra comm padding would be required
        min_comm_intervals = num_elements // max_elements_per_comm
        if min_comm_intervals == 0:
            if self.verbose:
                print_rank_0(
                    f'Using default max_elements_per_comm {max_elements_per_comm}')
            return max_elements_per_comm

        padding_for_min_comm = math.ceil(
            num_elements / (self.world_size * min_comm_intervals))

        # choose padding that uses least amount of overhead
        if padding_for_max_comm > padding_for_min_comm:
            new_max_elements_per_comm = padding_for_min_comm + max_elements_per_comm
            if self.verbose:
                print_rank_0(
                    f'Updating max_elements_per_comm from {max_elements_per_comm} -> {new_max_elements_per_comm}')
            return new_max_elements_per_comm
        else:
            if self.verbose:
                print_rank_0(
                    f'Using default max_elements_per_comm {max_elements_per_comm}')
            return max_elements_per_comm

    def get_data_parallel_sub_partitions(self,
                                         tensor,
                                         max_elements_per_comm,
                                         ):
        total_num_elements = tensor.numel()

        # if total elements is less than our max, revert to splitting into dp partitions
        max_elements_per_comm = min(total_num_elements, max_elements_per_comm)
        sub_partition_size = int(max_elements_per_comm // self.world_size)

        # Ensure partition alignment was done correctly
        num_sub_partitions = int(total_num_elements // sub_partition_size)
        assert total_num_elements % sub_partition_size == 0, "{} % {} != 0".format(total_num_elements,
                                                                                   sub_partition_size)

        # Ensure comm interval alignment was done correctly.
        num_comm_intervals = int(num_sub_partitions // self.world_size)
        assert num_sub_partitions % self.world_size == 0, "{} % {} != 0".format(
            num_sub_partitions, self.world_size)

        if self.verbose:
            print_rank_0("**** partition info:")
            print_rank_0(f"\t total_num_elements={total_num_elements}")
            print_rank_0(f"\t world_size={self.world_size}")
            print_rank_0(f"\t max_elements_per_comm={max_elements_per_comm}")
            print_rank_0(f"\t sub_partition_size={sub_partition_size}")
            print_rank_0(f"\t num_sub_partitions={num_sub_partitions}")
            print_rank_0(f"\t num_comm_intervals={num_comm_intervals}")
            print_rank_0("****")

        # [comm_id] -> [rank]
        comm_partitions = []
        for _ in range(num_comm_intervals):
            comm_partitions.append([])

        start = 0
        comm_id = 0
        element_intervals = defaultdict(
            list)  # [rank] -> [(start,end), (start,end), ...]
        for idx in range(num_sub_partitions):
            rank_id = idx % self.world_size
            sub_partition = tensor.narrow(
                0, start, sub_partition_size).detach()
            element_intervals[rank_id].append(
                (start, start + sub_partition_size))
            comm_partitions[comm_id].append(sub_partition)
            start = start + sub_partition_size
            if rank_id == (self.world_size - 1):
                comm_id += 1

        # [rank] -> [comm_id]
        sub_partitions = []
        for _ in range(self.world_size):
            sub_partitions.append([])
        for comm_id, partitions in enumerate(comm_partitions):
            for rank_id, partition in enumerate(partitions):
                sub_partitions[rank_id].append(partition)

        return comm_partitions, sub_partitions, element_intervals, sub_partition_size, num_comm_intervals

    def get_all_sub_partition_info(self,
                                   tensor_list,
                                   all_element_intervals,
                                   ):
        params_not_local = []

        # [rank] -> [comm-id] -> [param/offset]
        params_in_rank_sub_partition = []
        params_in_rank_sub_partitions_offsets = []

        for rank in range(self.world_size):
            params_in_local_sub_partition = []
            local_sub_partition_offsets = []
            comm_tensor_list = []
            comm_offset_list = []
            current_index = 0
            prev_comm_idx = 0
            for iii, tensor in enumerate(tensor_list):
                tensor_size = tensor.numel()
                results_list = _range_check(current_index,
                                            all_element_intervals[rank],
                                            tensor_size)
                for contained, offset, comm_idx in results_list:
                    if contained:
                        if prev_comm_idx != comm_idx:
                            params_in_local_sub_partition.append(
                                comm_tensor_list)
                            comm_tensor_list = []
                            local_sub_partition_offsets.append(
                                comm_offset_list)
                            comm_offset_list = []
                        comm_tensor_list.append(tensor)
                        comm_offset_list.append(offset)
                        prev_comm_idx = comm_idx
                    elif rank == self.local_rank:
                        params_not_local.append(tensor)

                current_index = current_index + tensor_size

            # assert len(comm_tensor_list) > 0
            # assert len(comm_offset_list) > 0
            params_in_local_sub_partition.append(comm_tensor_list)
            local_sub_partition_offsets.append(comm_offset_list)

            params_in_rank_sub_partition.append(params_in_local_sub_partition)
            params_in_rank_sub_partitions_offsets.append(
                local_sub_partition_offsets)

        return params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local

    def get_flat_sub_partitions(self,
                                comm_tensor_list,
                                comm_param_offsets,
                                sub_partition_size,
                                dtype,
                                default_device,
                                num_comm_intervals=None,
                                return_partition_params=False):
        partition_params = []
        final_param_offsets = []
        flat_sub_partitions = []
        for tensor_list, param_offsets in zip(comm_tensor_list, comm_param_offsets):
            flat_tensor_list = []
            current_size = 0
            my_offsets = []
            my_params = []

            for i, tensor in enumerate(tensor_list):
                if tensor.grad is None:
                    tensor.grad = torch.zeros(tensor.size(),
                                              dtype=tensor.dtype,
                                              device=tensor.device)
                param = tensor
                tensor = tensor.grad
                num_elements = tensor.numel()
                tensor_offset = 0

                # we need to offset to get to the right element
                if i == 0 and param_offsets[i] > 0:
                    tensor_offset = param_offsets[i]
                    num_elements = num_elements - tensor_offset

                # We don't need all elements of the tensor if this tensor is
                # larger than we have space for in our curr sub-partition
                if num_elements > (sub_partition_size - current_size):
                    num_elements = sub_partition_size - current_size

                # we need a narrow view of the tensor based on the tensor offset and number of elements that
                # we need from this tensor
                if tensor_offset > 0 or num_elements < tensor.numel():
                    flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                        0,
                        int(tensor_offset),
                        int(num_elements)).to(dtype))
                else:
                    flat_tensor_list.append(tensor.to(dtype))
                my_params.append(param)

                # remember offset into partition and #elems for this tensor
                my_offsets.append((current_size, num_elements))

                current_size = current_size + num_elements

            # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
            if current_size < sub_partition_size:
                my_offsets.append((None, None))
                my_params.append(None)
                if len(tensor_list) == 0:
                    assert default_device != None
                    flat_tensor_list.append(
                        torch.zeros(int(sub_partition_size - current_size),
                                    dtype=dtype,
                                    device=default_device))
                else:
                    flat_tensor_list.append(
                        torch.zeros(int(sub_partition_size - current_size),
                                    dtype=dtype,
                                    device=tensor_list[0].device))
            partition_params.append(my_params)  # flat_tensor_list)
            final_param_offsets.append(my_offsets)
            assert len(flat_tensor_list) == len(my_offsets), "{} {}".format(
                len(flat_tensor_list), len(my_offsets))
            flat_sub_partitions.append(self.flatten(flat_tensor_list))
        if num_comm_intervals is not None and len(
                flat_sub_partitions) < num_comm_intervals:
            # print("padding w. sub partitions to ensure uniform communication")
            device = flat_sub_partitions[0].device
            for _ in range(num_comm_intervals - len(flat_sub_partitions)):
                flat_sub_partitions.append(
                    torch.zeros(int(sub_partition_size),
                                dtype=dtype,
                                device=device))
                partition_params.append([None])
                final_param_offsets.append([(None, None)])

        if return_partition_params:
            assert len(flat_sub_partitions) == len(partition_params)
            assert len(partition_params) == len(final_param_offsets), "{} {}".format(len(partition_params),
                                                                                     len(final_param_offsets))
            return flat_sub_partitions, partition_params, final_param_offsets
        return flat_sub_partitions

    def zero_grad(self, set_grads_to_None=False):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self._param_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            if isinstance(p, list):
                for _p in p:
                    _p.grad = None
            else:
                p.grad = None

    def flatten_dense_tensors_sub_partition_aligned(self,
                                                    tensor_list,
                                                    max_elements_per_comm
                                                    ):
        assert max_elements_per_comm >= self.world_size, f"max_elements_per_comm {max_elements_per_comm} < dp {self.world_size}"

        num_elements = sum(t.numel() for t in tensor_list)

        # Compute aligned partition size based on parameter count
        aligned_param_partition_size = math.ceil(
            num_elements / self.world_size)

        # Compute aligned partition size based on communication size
        aligned_comm_partition_size = int(
            max_elements_per_comm // self.world_size)

        if aligned_param_partition_size <= aligned_comm_partition_size:
            sub_partition_count = 1
            sub_partition_size = aligned_param_partition_size
        else:
            sub_partition_count = math.ceil(aligned_param_partition_size /
                                            aligned_comm_partition_size)
            sub_partition_size = aligned_comm_partition_size

        # Compute required padding  for alignment to dp and max_elements_per_comm
        padding = (sub_partition_count * sub_partition_size *
                   self.world_size) - num_elements

        if self.verbose:
            print_rank_0(
                f"sub_partition_count: {sub_partition_count}, sub_partition_size: {sub_partition_size}, padding: {padding}")
            print_rank_0(
                f"number of elements with padding: {num_elements} + {padding} = {num_elements + padding}")

        if padding == 0:
            aligned_tensor_list = tensor_list
        else:
            pad_tensor = torch.zeros(padding,
                                     device=tensor_list[0].device,
                                     dtype=tensor_list[0].dtype)
            aligned_tensor_list = tensor_list + [pad_tensor]

        flat_tensors = self.flatten(aligned_tensor_list)
        return flat_tensors

    # def reduce_gradients(self):
    #     # LSG: this reduce gradients method no longer works
    #     # after code change, please use DataParallelGradientHandler instead
    #
    #     world_size = gpc.get_world_size(self.parallel_mode)
    #     local_rank = gpc.get_local_rank(self.parallel_mode)
    #
    #     for i, group in enumerate(self._param_groups):
    #         num_comm_intervals = self.num_comm_intervals_per_group[i]
    #         all_sub_partitions = []
    #         for rank in range(world_size):
    #             # gsp is list of partitions indexed by comm_idx
    #             grad_sub_partitions = self.get_flat_sub_partitions(
    #                 comm_tensor_list=self.params_in_rank_sub_partitions[i][rank],
    #                 comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i][rank],
    #                 dtype=self.local_sub_partitions_of_groups[i][0].dtype,
    #                 default_device=self.default_device,
    #                 sub_partition_size=self.sub_partition_sizes[i],
    #                 num_comm_intervals=self.num_comm_intervals_per_group[i])
    #             all_sub_partitions.append(grad_sub_partitions)
    #
    #             assert len(grad_sub_partitions) == num_comm_intervals
    #
    #         local_comm_partitions = []
    #         for comm_idx in range(num_comm_intervals):
    #             single_comm_all_partitions = []
    #             for rank in range(world_size):
    #                 single_comm_all_partitions.append(all_sub_partitions[rank][comm_idx])
    #
    #             for partition in single_comm_all_partitions:
    #                 partition.div_(world_size)
    #
    #             dist.reduce_scatter(output=single_comm_all_partitions[local_rank],
    #                                 input_list=single_comm_all_partitions,
    #                                 group=gpc.get_group(self.parallel_mode))

    def step(self, closure=None):
        local_sub_partitions_grad_groups = []

        for i, group in enumerate(self._param_groups):
            # RS: update free grads w.r.t. sub partitions
            # free gradients for all the parameters that are not updated by this process
            self.free_grad_in_param_list(self.params_not_local[i])

            # create flat gradient partitions for parameters updated by this process
            local_grad_sub_partitions = self.get_flat_sub_partitions(
                comm_tensor_list=self.params_in_rank_sub_partitions[i][self.local_rank],
                comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i][self.local_rank],
                sub_partition_size=self.sub_partition_sizes[i],
                dtype=self.local_sub_partitions_of_groups[i][0].dtype,
                num_comm_intervals=self.num_comm_intervals_per_group[i],
                default_device=self.default_device)

            # RS: update all our local params with sub-partition grads
            for idx, sub_partition_param in enumerate(self.local_sub_partitions_of_groups[i]):
                sub_partition_param.grad = local_grad_sub_partitions[idx]

            # RS: update free grads for sub-partitions
            # release all the gradient since we have already created a necessary copy in dp_grad_partition
            self.free_grad_in_param_list(
                self.params_in_rank_sub_partitions[i][self.local_rank])

            local_sub_partitions_grad_groups.append(local_grad_sub_partitions)

        if closure is None:
            loss = self.optimizer.step()
        else:
            loss = self.optimizer.step(closure=closure)

        # RS: clear our sub partition grads
        # LSG: not needed as amp is used instead
        # get rid of the fp32 gradients. Not needed anymore
        # for group in self.local_sub_partitions_of_groups:
        #     for idx, sub_partition_param in enumerate(group):
        #         sub_partition_param.grad = None

        # RS: all_gather/broadcast sub-partitions in separate comm calls
        # gather the updated weights from everyone
        for all_sub_partitions in self.parallel_comm_sub_partitioned_groups:
            for comm_id, sub_partitions in enumerate(all_sub_partitions):
                dist.all_gather(sub_partitions,
                                sub_partitions[self.local_rank],
                                group=gpc.get_group(self.dp_parallel_mode))

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self._param_groups)):
            updated_params = self.unflatten(self._param_groups_flat[i],
                                            self._param_groups[i])
            for p, q in zip(self._param_groups[i], updated_params):
                p.data = q.data

        return loss

    def _rigid_state_dict(self):
        """Returns a dict that can be loaded for continued training with same DP degree

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
        for k, v in self.optimizer.state_dict().items():
            state_dict[k] = v
        state_dict[
            'local_sub_partitions_of_groups'] = self.local_sub_partitions_of_groups
        return state_dict

    def state_dict(self):
        """
        Returns a dict containing the current state of this Optimizer instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.

        Example::

            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        return self._rigid_state_dict()

    def load_state_dict(self,
                        state_dict,
                        load_optimizer_states=True,
                        ):
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
        self._rigid_load_state_dict(
            state_dict,
            load_optimizer_states)

    def _rigid_load_state_dict(self, state_dict, load_optimizer_states=True):
        # I think it should actually be ok to reload the optimizer before the model.
        state_dict_ = state_dict.copy()
        local_sub_partitions_of_groups = state_dict_.pop(
            'local_sub_partitions_of_groups')

        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict_)

        for curr_group, saved_group in zip(self.local_sub_partitions_of_groups,
                                           local_sub_partitions_of_groups):
            for curr_param, saved_param in zip(curr_group, saved_group):
                curr_param.data.copy_(saved_param.data)
