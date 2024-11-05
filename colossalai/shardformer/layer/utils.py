from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup, get_world_size

from colossalai.accelerator import get_accelerator


class SeqParallelUtils:
    @staticmethod
    def marked_as_sp_partial_derived_param(param):
        """
        Mark a parameter as partially derived in sequence parallelism.

        Args:
            param: The parameter to mark as partially derived.
        """
        setattr(param, "partial_derived", True)

    @staticmethod
    def is_sp_partial_derived_param(param):
        """
        Check if a parameter is marked as partially derived in sequence parallelism.

        Args:
            param: The parameter to check.

        Returns:
            bool: True if the parameter is marked as partially derived, False otherwise.
        """
        return getattr(param, "partial_derived", False)

    @staticmethod
    def allreduce_partial_data_grad(
        process_group: ProcessGroup,
        model: nn.Module = None,
        grads: List[torch.Tensor] = None,
    ):
        """
        Allreduce partial derived gradients across the specified process group.

        This function performs gradient synchronization for parameters that are marked as partially derived in sequence parallelism.

        Args:
            process_group (ProcessGroup): The process group for gradient synchronization.
            model (nn.Module): The model from which gradients will be synchronized.
            grads (List[torch.Tensor]): The list of gradients to be synchronized.
            only_sp_partial (bool): Whether handle all the parameters or only parameters marked as partial derived.
        Raises:
            AssertionError: If both `model` and `grads` are provided or neither is provided.
        """
        # Ensure that exactly one of `model` and `grads` is provided for gradient synchronization.
        assert (model is not None) ^ (grads is not None), "Exactly one of model and grads must be not None."

        # Get the size of the process group, which determines whether synchronization is needed.
        group_size = get_world_size(process_group) if process_group is not None else 1

        if group_size == 1:
            # If the process group size is 1, no synchronization is required.
            return

        if model is not None:
            # If `model` is provided, extract partial derived gradients from the model's parameters.
            grads = []

            for p in model.parameters():
                if p.grad is not None:
                    if SeqParallelUtils.is_sp_partial_derived_param(p):
                        grads.append(p.grad.data)

            # Flatten and reduce the gradients using the specified process group.
            if len(grads) == 0:
                return
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=process_group)

            # Unflatten the synchronized gradients and update the model's gradients.
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
        else:
            # If `grads` are provided explicitly, synchronize those gradients directly.
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=process_group)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


class Randomizer:
    """
    Randomizer enables the program to be executed under a different seed within the context.

    Example:

    ```python
    randomizer = Randomizer(seed=1024)

    with randomizer.fork():
        # do something here with seed 1024
        do_something()
    ```

    Args:
        seed (int): The random seed to set.
        enable_cpu (bool): fork the CPU RNG state as well.
        with_index (bool): whether to use the index of the randomizer.
    """

    _INDEX = 0

    def __init__(self, seed: int):
        self.seed = seed

        # Handle device rng state
        # 1. get the current rng state
        # 2. set the seed and store the rng state
        # 3. recover the original rng state
        device_original_rng_state = get_accelerator().get_rng_state()
        get_accelerator().manual_seed(seed)
        self.device_rng_state = get_accelerator().get_rng_state()
        get_accelerator().set_rng_state(device_original_rng_state)

        # to the same for cpu rng state
        cpu_original_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.cpu_rng_state = torch.get_rng_state()
        torch.set_rng_state(cpu_original_rng_state)

    def _set_device_rng_state(self, rng_state):
        get_accelerator().set_rng_state(rng_state)

    def _get_device_rng_state(self):
        current_state = get_accelerator().get_rng_state()
        return current_state

    def _set_cpu_rng_state(self, rng_state):
        torch.set_rng_state(rng_state)

    def _get_cpu_rng_state(self):
        current_state = torch.get_rng_state()
        return current_state

    @contextmanager
    def fork_rng(self, enable_cpu: bool = False):
        """
        This is a context manager to change the dropout state and recover the original state.

        Usage:
        ::
            >>> with _seed_manager.dropout_mode():
            >>>     input = super().forward(input)
        """
        try:
            current_device_rng_state = self._get_device_rng_state()
            self._set_device_rng_state(self.device_rng_state)

            if enable_cpu:
                current_cpu_rng_state = self._get_cpu_rng_state()
                self._set_cpu_rng_state(self.cpu_rng_state)
            yield
        finally:
            self.device_rng_state = self._get_device_rng_state()
            self._set_device_rng_state(current_device_rng_state)

            if enable_cpu:
                self.cpu_rng_state = self._get_cpu_rng_state()
                self._set_cpu_rng_state(current_cpu_rng_state)

    @staticmethod
    def index():
        """
        Return the index of the randomizer. The index is useful when the user wants
        to introduce some randomness in the program.

        Note:
        The index will increment by one each time this method is called.

        Example:

        ```python
        # assume we need a randomizer to init the weight of different layers
        # we can use the index of the randomizer to do so that
        # each layer has its own randomizer with a different seed
        base_seed = torch.random.initial_seed()
        seed = base_seed + Randomizer.index()
        randomizer = Randomizer(seed)

        with randomizer.fork():
            init_weights()
        ```

        """
        idx = Randomizer._INDEX
        return idx

    @staticmethod
    def increment_index():
        """
        Increment the index of the randomizer by one.
        """
        Randomizer._INDEX += 1

    @staticmethod
    def reset_index():
        """
        Reset the index to zero.
        """
        Randomizer._INDEX = 0

    @staticmethod
    def is_randomizer_index_synchronized(process_group: ProcessGroup = None):
        """
        Return whether the randomizer index is synchronized across processes.
        """
        index = Randomizer.index()
        if dist.is_initialized():
            # convert the index to tensor
            index_tensor = torch.tensor(index, dtype=torch.int32, device=get_accelerator().get_current_device())

            # all gather the index
            gathered_index = [torch.zeros_like(index_tensor) for _ in range(dist.get_world_size(process_group))]
            dist.all_gather(gathered_index, index_tensor, process_group)

            # make sure all the gathered index are the same
            for i in range(1, dist.get_world_size(process_group)):
                if gathered_index[i] != gathered_index[0]:
                    return False

        return True

    @staticmethod
    def synchronize_index(process_group: ProcessGroup = None):
        """
        All gather the index and pick the largest value.
        """
        index = Randomizer.index()

        if dist.is_initialized():
            # convert the index to tensor
            index_tensor = torch.tensor(index, dtype=torch.int32, device=get_accelerator().get_current_device())

            # all gather the index
            gathered_index = [torch.zeros_like(index_tensor) for _ in range(dist.get_world_size(process_group))]
            dist.all_gather(gathered_index, index_tensor, process_group)

            # pick the largest index
            for i in range(1, dist.get_world_size(process_group)):
                if gathered_index[i] > index_tensor:
                    index_tensor = gathered_index[i]

            # set the index
            Randomizer._INDEX = index_tensor.item()


def create_randomizer_with_offset(
    seed: int, process_group: ProcessGroup = None, offset_by_rank: bool = True, offset_by_index: bool = True
):
    """
    Create a randomizer with an offset. The offset is equal to the rank of the process and the index of the randomizer.

    Args:
        seed (int): The base random seed to set.
        process_group (ProcessGroup): the process group to get the rank from.
        offset_by_rank (bool): whether to offset by the rank of the process, i.e., the rank of the process will be added to the seed. Default: True.
        offset_by_index (bool): whether to offset by the index of the randomizer, i.e., the index of the randomizer will be added to the seed. Default: True.

    Returns:
        Randomizer: the randomizer with offset.
    """
    base_seed = seed

    if offset_by_rank and dist.is_initialized():
        rank = dist.get_rank(process_group)
        base_seed += rank

    if offset_by_index:
        # check if the randomizer index is synchronized
        is_synchronized = Randomizer.is_randomizer_index_synchronized(process_group)
        assert is_synchronized, (
            "We detect that the randomizer index is not synchronized across processes."
            "This is not allowed when we want to create a randomizer with offset by index."
            "Please call Randomizer.synchronize_index() first."
        )

        base_seed += Randomizer.index()
        Randomizer.increment_index()

    return Randomizer(seed=base_seed)


def split_batch_zigzag(
    batch: Union[torch.Tensor, List[torch.Tensor]], sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Split the input sequence batch . Naively spliting the attention mask in the causal setting
    will result in the preceding ranks having much less workload.
    We split after "folding" the 2D attention mask in half (https://github.com/zhuzilin/ring-flash-attention/issues/2).
    For example, for sp_size = 4 and seq_len = 8, we get | s0, s7 | s1, s6 | s2, s5 | s3, s4 |.

    Args:
        batch (List[torch.Tensor] or Tensor): The input tensor(s) to split.
        sp_group (ProcessGroup): The process group for sequence parallelism.
        seq_dim (int): The sequence dimension to split.
        is_label (bool): If True, mask and shift the tensor for next token prediction.

    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size == 1:
        return batch

    if isinstance(batch, torch.Tensor):
        batch = [batch]
    seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1

    if sp_size > 1:
        for idx, tensor in enumerate(batch):
            assert (
                tensor.shape[seq_dim] // (sp_size * 2) > 1 and tensor.shape[seq_dim] % (sp_size * 2) == 0
            ), f"Bro, the seq length {tensor.shape[seq_dim]} for tensor {idx} can't be split by {sp_size * 2}!"
            if is_label:
                assert tensor.dim() == 2, "Label shape should be (B, Seqlen)"
                tensor = torch.cat([tensor[:, 1:], torch.full_like(tensor[:, :1], -100)], dim=1)

            tensor = tensor.view(
                *tensor.shape[:seq_dim],
                2 * sp_size,
                tensor.shape[seq_dim] // (2 * sp_size),
                *tensor.shape[seq_dim + 1 :],
            )
            indices = torch.tensor([sp_rank, 2 * sp_size - 1 - sp_rank], device=tensor.device)
            tensor = tensor.index_select(seq_dim, indices).contiguous()
            # (B, 2, Sq // (2 * sp_size), ...) -> (B, Sq // sp_size, ...)
            batch[idx] = tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])

    if len(batch) == 1:
        return batch[0]
    return batch


def split_varlen_zigzag(
    batch: Union[List[torch.Tensor], torch.Tensor],
    cu_seqlens: torch.Tensor,
    sp_group: ProcessGroup,
    max_seqlen: int = 0,
    is_batched_seq: bool = False,
    is_label: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Split a packed seq/batch of padded sequences in a Zigzag fashion.
        Different from split_batch_zigzag, inputs here have variable sequence lengths.
    Args:
        batch (List[torch.Tensor]): Packed sequences of shape (T, ...), or (B, Sq, ...) if is_batched_seq,
            where T is the total number of tokens.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths of shape (B + 1) before splitting.
        sp_group (ProcessGroup): The process group for sequence parallelism.
        max_seqlen (int): The maximum sequence length in the batch before splitting.
        is_batched_seq (bool): If True, then the input is a batch of sequences padded to the same len.
        is_label (bool): If True, mask out the first token in each sequence (<Start of Sentence>).

    Returns:
        batch (List[torch.Tensor]): Packed sequences of shape (T, ..)
            or (B, max_seqlen // sp_size, ...) if is_batched_seq
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size == 1:
        return batch

    if is_batched_seq:
        assert max_seqlen > 0, "max_seqlen must be provided for 2D input"

    if isinstance(batch, torch.Tensor):
        batch = [batch]
    # seq: (B, Sq, h, n)
    # seq = seq[:, :rank * (seqlen // sp_size), ...]

    for i, packed_seq in enumerate(batch):
        device = packed_seq.device
        dtype = packed_seq.dtype

        if is_batched_seq:
            assert max_seqlen % (sp_size * 2) == 0
            # Recreate a padded tensor with the new max seqlen
            shape = (packed_seq.shape[0], max_seqlen // sp_size, *packed_seq.shape[2:])
            local_seq = torch.zeros(shape, dtype=dtype, device=device)
        else:
            total_seqlen = cu_seqlens[-1]
            assert (
                total_seqlen % (2 * sp_size) == 0
            ), f"total_seqlen {total_seqlen} must be divisible by 2 * sp_size = {2 * sp_size}"
            local_seq = []

        for j in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[j], cu_seqlens[j + 1]
            seqlen = end - start
            assert (
                seqlen % (2 * sp_size) == 0
            ), f"batch {i} seq {j}'s length ({seqlen}) must be divisible by 2 * sp_size = {2 * sp_size} for splitting"

            if is_batched_seq:
                seq = packed_seq[j][:seqlen]
                if is_label:
                    # Shift one position to the right for next token prediction
                    seq = torch.cat([seq[1:], torch.tensor([-100], dtype=dtype, device=device)])

                seq = seq.chunk(2 * sp_size, dim=0)
                half = seqlen // sp_size // 2
                local_seq[j][:half] = seq[sp_rank]
                local_seq[j][half : seqlen // sp_size] = seq[2 * sp_size - 1 - sp_rank]
            else:
                seq = packed_seq[start:end]
                if is_label:
                    seq = torch.cat(seq[1:], torch.tensor([-100], dtype=dtype, device=device))
                seq = seq.chunk(sp_size * 2)
                local_seq.extend([seq[sp_rank], seq[2 * sp_size - 1 - sp_rank]])

        if is_batched_seq:
            batch[i] = local_seq.contiguous()
        else:
            batch[i] = torch.cat(local_seq, dim=0)

    if len(batch) == 1:
        batch = batch[0]
    return batch


def is_share_sp_tp(sp_mode: str):
    """sp_mode "ring" and "split_gather" use the TP group as SP group
    to split both the vocab and sequence, so we must gather the sequence
    to correctly get logits at each positions.
    """
    return sp_mode in ["ring", "split_gather"]


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = []

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
        self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: Optional[torch.Tensor] = None,
        commit: bool = True,
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(send_tensor)
        else:
            res = recv_tensor

        # looks like batch_isend_irecv doesn't deadlock even
        # when we don't swap send recv ops based on rank
        send_op = dist.P2POp(dist.isend, send_tensor, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.extend([send_op, recv_op])

        if commit:
            self._reqs = dist.batch_isend_irecv(self._ops)
        return res

    def commit(self):
        assert len(self._ops) > 0, "No ops to commit"
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        assert len(self._reqs) > 0, "No requests to wait for"
        for req in self._reqs:
            req.wait()
        self._reqs = []
        self._ops = []


@torch.jit.script
def get_half_index(cu_seqlens, *, front: bool):
    index = torch.zeros(cu_seqlens[-1], dtype=torch.bool, device=cu_seqlens.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index
