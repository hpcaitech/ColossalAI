from contextlib import contextmanager
from typing import List

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


def zigzag_split_batch(batch: List[torch.Tensor], sp_group: ProcessGroup, varlen: bool = False):
    """
    Split the input along the sequence dimension for Ring Attention. As naively spliting sequence
    in the causual setting will result in the first ranks having much less workload than the last ranks,
    we split after "folding" the 2D attention mask in half (https://github.com/zhuzilin/ring-flash-attention/issues/2).
    For example, for sp_size = 4 and seq_len = 8, we get | s0, s7 | s1, s6 | s2, s5 | s3, s4 |.

    Args:
        batch (List[torch.Tensor]): The input tensors to split.
        sp_group (ProcessGroup): The process group for sequence parallelism.
        varlen (bool): If the input is padded (aka "packing" mode), such that
            sequences in a batch have different lengths, and we need to unpad and
            split each sequence evenly by sp_size.
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    seq_dim = 1
    if sp_size > 1:
        for idx, tensor in enumerate(batch):
            assert (
                tensor.numel() // (sp_size * 2) > 1
            ), f"Bro, the seq length for tensor {idx} in batch is too short to split!"
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

    return batch


def is_share_sp_tp(sp_mode: str):
    """sp_mode "ring" and "split_gather" use the TP group as SP group
    to split both the vocab and sequence, so we must gather the sequence
    to correctly get logits at each positions.
    """
    return sp_mode in ["ring", "split_gather"]


# Copied from https://github.com/zhuzilin/ring-flash-attention/tree/main/ring_flash_attn
# Use Triton kernel if installed else use torch
try:
    import triton
    import triton.language as tl

    @triton.jit
    def flatten_kernel(
        # pointers to matrices
        OUT,
        LSE,
        CU_SEQLENS,
        # strides
        stride_out_nheads,
        stride_out_seqlen,
        stride_lse_batch,
        stride_lse_nheads,
        stride_lse_seqlen,
        # meta-parameters
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_batch = tl.program_id(axis=1)
        pid_head = tl.program_id(axis=2)

        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads
        OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        LSE = LSE + rm[:, None] * stride_lse_seqlen
        x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

        OUT = OUT + rm[:, None] * stride_out_seqlen
        tl.store(OUT, x, mask=rm[:, None] < seqlen)

    def flatten_varlen_lse(lse, cu_seqlens):
        """
        Arguments:
            lse: (batch_size, nheads, max_seqlen)
            cu_seqlens: (batch_size + 1,)
        Return:
            flatten_lse: (nheads, total_seqlen)
        """
        total_seqlen = cu_seqlens[-1]
        batch_size, nheads, max_seqlen = lse.shape
        output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

        grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
        BLOCK_M = 4

        with torch.cuda.device(lse.device.index):
            flatten_kernel[grid](
                output,
                lse,
                cu_seqlens,
                # strides
                output.stride(0),
                output.stride(1),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                BLOCK_M,
            )
        return output

    @triton.jit
    def unflatten_kernel(
        # pointers to matrices
        OUT,
        LSE,
        CU_SEQLENS,
        # strides
        stride_out_batch,
        stride_out_nheads,
        stride_out_seqlen,
        stride_lse_seqlen,
        stride_lse_nheads,
        # meta-parameters
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_batch = tl.program_id(axis=1)
        pid_head = tl.program_id(axis=2)

        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        LSE = LSE + rm[:, None] * stride_lse_seqlen
        x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

        OUT = OUT + rm[:, None] * stride_out_seqlen
        tl.store(OUT, x, mask=rm[:, None] < seqlen)

    def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
        """
        Arguments:
            lse: (total_seqlen, nheads, 1)
            cu_seqlens: (batch_size + 1,)
            max_seqlen: int
        Return:
            unflatten_lse: (batch_size, nheads, max_seqlen)
        """
        lse = lse.unsqueeze(dim=-1)
        batch_size = len(cu_seqlens) - 1
        nheads = lse.shape[1]
        output = torch.empty(
            (batch_size, nheads, max_seqlen),
            dtype=lse.dtype,
            device=lse.device,
        )

        grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
        BLOCK_M = 4

        with torch.cuda.device(lse.device.index):
            unflatten_kernel[grid](
                output,
                lse,
                cu_seqlens,
                # strides
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                BLOCK_M,
            )
        return output

except:
    # Triton not installed, use torch instead
    @torch.jit.script
    def flatten_varlen_lse(lse, cu_seqlens):
        new_lse = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            new_lse.append(lse[i, :, : end - start])
        return torch.cat(new_lse, dim=1)

    @torch.jit.script
    def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
        num_seq = len(cu_seqlens) - 1
        num_head = lse.shape[-2]
        new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
        for i in range(num_seq):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            new_lse[i, : end - start] = lse[start:end]
        return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()
