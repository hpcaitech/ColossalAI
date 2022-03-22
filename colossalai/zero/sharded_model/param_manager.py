import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from ._zero3_utils import alloc_storage, free_storage, get_shard

# TODO: Remove the toggle-enable_nccl_base_collectives in the future
if os.getenv("ENABLE_NCCL_BASE_COLLECTIVES", "1") == "0":
    enable_nccl_base_collectives = False
else:
    enable_nccl_base_collectives = True

# TODO: add flatten params


class Zero3ParameterManager:
    def __init__(self,
                 module: nn.Module,
                 process_group: Optional[ProcessGroup],
                 mixed_precision: bool = False,
                 flatten_parameters: bool = True,
                 compute_dtype: Optional[torch.dtype] = None,
                 compute_device: Optional[torch.device] = None,
                 offload_config: Optional[dict] = None
                 ) -> None:
        """Manage parameter shards. We manage several attributes on each Parameter instance:
            ``zero_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param).
            ``zero_orig_size``: the size of the original Parameter (before sharding)
            ``zero_shard_padding``: the padding size. All paddings are right padding.
            ``zero_fp32_shard``: a single shard of the parameters in full precision
                (typically FP32, but this is dependent on the dtype of the model
                as it's passed in by the user). This can be on CPU or GPU
                depending on the value of *``offload_config``*.
            ``zero_fp16_shard``: This will be a single shard of the parameters in FP16, used for all-gather.
                This can be in FP16 or FP32 depending on the value of *``compute_dtype``* and
                if params are offloaded to CPU.
            ``zero_full_param_padded``: the full weight (padded to be evenly
                divisible by ``world_size``), used for computation in the
                forward and backward pass. This will be resized in place and
                only materialized (via all-gather) as needed.
            ``zero_cpu_grad``: the gradient saved on CPU. It's set only when using CPU offload.

        :param module: original module
        :type module: nn.Module
        :param process_group: typically data parallel process group, defaults to None
        :type process_group: Optional[ProcessGroup], optional
        :param mixed_precision: whether to use mixed precision mode, defaults to False
        :type mixed_precision: bool, optional
        :param flatten_parameters: whether to flatten parameters, useless now, defaults to True
        :type flatten_parameters: bool, optional
        :param compute_dtype: the dtype of parameters when computing, defaults to None
        :type compute_dtype: Optional[torch.dtype], optional
        :param compute_device: the device of parameters when computing, defaults to None
        :type compute_device: Optional[torch.device], optional
        :param offload_config: offload config, defaults to None
        :type offload_config: Optional[dict], optional
        """
        self.process_group = process_group
        self.shard_idx = process_group.rank()
        self.num_shards = process_group.size()
        self.mixed_precision = mixed_precision
        self.compute_dtype = compute_dtype
        self.compute_device = compute_device
        self.offload_config = offload_config

        self._cpu_offload = offload_config.get('device', None) == 'cpu' if offload_config else False

        self.params: List[Parameter] = []
        for param in module.parameters():
            if not hasattr(param, 'zero_is_sharded'):
                self.params.append(param)

        self._has_params = len(self.params) > 0
        self._has_sharded_params = False
        # Flag to indicate if the full params are gathered.
        self.has_full_params: bool = False

        self._shard_params()
        # Maybe no need, reserve to prevent bugs
        # self.delete_fp32_shards()

        self._streams: Dict[str, torch.cuda.Stream] = {}

    def _shard_params(self) -> None:
        for p in self.params:
            assert not hasattr(p, "zero_is_sharded")
            assert p.is_floating_point()
            if self.mixed_precision:
                assert p.dtype == torch.float32

            # If world_size is 1, then we all-reduce grads instead of sharding.
            p.zero_is_sharded = self.num_shards > 1
            p.zero_orig_size = p.data.size()

            if not p.zero_is_sharded:
                p.zero_shard_padding = 0
                continue

            # Replace p.data with the relevant shard.
            orig_data = p.data
            p.data, p.zero_shard_padding = get_shard(p.data, self.shard_idx, self.num_shards)
            free_storage(orig_data)

    @torch.no_grad()
    def reset_param_attr(self, p: Parameter, training: bool) -> None:
        """This should be called by ``ZeroRedundancyLevel3Model._lazy_init()``
        """
        assert hasattr(p, 'zero_is_sharded') and hasattr(p, 'zero_orig_size')
        if hasattr(p, 'zero_fp32_shard'):
            return

        # A single shard of the parameters in full precision.
        p.zero_fp32_shard = p.data

        if self.mixed_precision:
            assert p.zero_fp32_shard.dtype == torch.float32

        if self._cpu_offload:
            assert p.zero_fp32_shard.device == torch.device('cpu')
            # If we plan to keep the FP32 parameters on CPU, then pinning
            # memory allows us to later use non-blocking transfers when moving
            # the FP32 param shard to compute_device.
            p.zero_fp32_shard = p.zero_fp32_shard.pin_memory()
            p.data = p.zero_fp32_shard

        if self.mixed_precision or self._cpu_offload:

            # In mixed precision mode, we maintain a reduced precision
            # (typically FP16) parameter shard on compute_device for performing
            # the computation in the forward/backward pass. We resize the
            # storage to size 0 at init (here) and re-materialize (by copying
            # from _fp32_shard) as needed. If offloading params to CPU, the
            # dtype of the fp16 shard will depend on the *`compute_dtype`*.
            p.zero_fp16_shard = torch.zeros_like(
                p.zero_fp32_shard, device=self.compute_device, dtype=self.compute_dtype)
            free_storage(p.zero_fp16_shard)

        if self.mixed_precision:
            assert p.zero_fp32_shard.dtype == torch.float32

        if not self.mixed_precision and not self._cpu_offload:
            # use _fp32_shard if you are not in using mixed precision or
            # offloading params and grads to CPU.
            p.zero_fp16_shard = None

        # We also maintain a full-sized parameter of type self.compute_dtype
        # (FP16 for mixed_precision or FP32 otherwise). We resize the
        # storage to size 0 at init (here) and only materialize as needed. The
        # storage may contain padding elements so that it is evenly divisible by
        # world_size, although these padding elements will be removed before the
        # relevant computation.
        if p.zero_is_sharded:
            p.zero_full_param_padded = torch.zeros(
                p.data.numel() * self.num_shards, device=self.compute_device, dtype=self.compute_dtype
            )
            free_storage(p.zero_full_param_padded)

        if self._cpu_offload and training:
            p.zero_cpu_grad = torch.zeros_like(p.data, device='cpu').pin_memory()

    def setup_streams(self, streams):
        self._streams = streams

    @torch.no_grad()
    def rebuild_full_params(self, force_full_precision: bool = False) -> Optional[List[Tuple[torch.Tensor, bool]]]:
        """
        Gather all shards of params.

        Note, this is idempotent if full params are already gathered. Callers
        assume the idempotency. So please keep it that way.

        Args:
            force_full_precision (bool, Optional): by default params will be gathered
                in ``compute_dtype`` (e.g., FP16), unless *force_full_precision* is
                ``True``, in which case they will be gathered in full precision
                (e.g., FP32), possibly in fresh storage. The parameter that's being
                rebuilt will end up in full precision as well.

        Returns:
            A list of tuples, where the first element is the full-sized param
            and the second element is a bool indicating if it's safe for the
            caller to free the full-sized param. This will be ``None`` if
            ``force_full_precision=False`` and the full params are already gathered.
        """
        # Store tensor and free flag
        output_tensors: List[Tuple[torch.Tensor, bool]] = []

        def update_p_data(custom_output_tensor: Optional[torch.Tensor] = None) -> None:
            """
            Helper function to update p.data pointer.

            Args:
                custom_output_tensor (torch.Tensor, Optional): if not None, this
                tensor contains the data we just gathered.
            """
            if custom_output_tensor is not None:
                assert p.zero_is_sharded
                p.data = custom_output_tensor
                output_tensors.append((p.data, True))
            elif not p.zero_is_sharded:
                if (self.mixed_precision or self._cpu_offload) and not force_full_precision:
                    assert p.zero_fp16_shard is not None
                    p.data = p.zero_fp16_shard
                    output_tensors.append((p.data, True))
                else:
                    # Here p.data == p._fp32_shard, so it's not safe to free.
                    output_tensors.append((p.data, False))
            else:
                p.data = p.zero_full_param_padded
                output_tensors.append((p.data, True))
            # Trim any padding and reshape to match original size.
            p.data = p.data[: p.zero_orig_size.numel()].view(p.zero_orig_size)

        if self._has_sharded_params:
            # self.has_full_params flag can be out of sync if a shared param is
            # sharded by another ZeroRedundancyLevel3Model instance. An example is that in eval case
            # with reshard_after_forward=False but the sharing instance has
            # reshard_after_forward=True. Then, on the second forward, the
            # other instance can shard the shared param and but this instance
            # can mistakenly think the full param is already gathered from the
            # has_full_params flag.
            #
            # Therefore, we update the flag accordingly here.
            self.has_full_params = not any(p.zero_full_param_padded.storage().size() == 0 for p in self.params)

        # Early exit if we already have full params and don't need full precision.
        if self.has_full_params and not force_full_precision:
            for p in self.params:
                update_p_data()
            return output_tensors

        self.has_full_params = True

        with torch.cuda.stream(self._streams["all_gather"]):
            if (self.mixed_precision or self._cpu_offload) and not force_full_precision:
                self.use_fp16_shards()

            if self._cpu_offload and force_full_precision:
                # If the compute_dtype and storage dtype are the same,
                # use pinned memory. Otherwise move p.data to the compute
                # device.
                if self.params[0].dtype == self.compute_dtype:
                    self.use_fp16_shards()
                else:
                    for p in self.params:
                        p.data = p.data.to(self.compute_device)

            for p in self.params:
                if not p.zero_is_sharded:  # e.g., when world_size == 1
                    update_p_data()
                else:
                    # Skip if already built. Only shared param can be rebuilt multiple times.
                    # A corner case is p.zero_orig_size = (1,), which means the shape equality is
                    # not a perfect check. But we assume we don't share a param with shape (1,).
                    # if p.data.shape == p.zero_orig_size and hasattr(p, "zero_is_shared") and p.zero_is_shared:
                    #     continue
                    # If self._cpu_offload and force_full_precision, we need to cast
                    # the FP32 CPU param to CUDA for the all-gather.
                    p_data = p.data.to(p.zero_full_param_padded.device, non_blocking=True)

                    p_size = p.zero_full_param_padded.size()
                    assert p_size.numel() % self.num_shards == 0
                    if self.mixed_precision and force_full_precision:
                        # Allocate fresh tensor in full precision since we are in
                        # mixed precision and full precision rebuild is asked.
                        output_tensor = p_data.new_zeros(p_size)
                    else:
                        if p.zero_full_param_padded.storage().size() != p_size.numel():
                            # Allocate based on full size from all shards.
                            alloc_storage(p.zero_full_param_padded, size=p_size)
                        output_tensor = p.zero_full_param_padded

                    # Fill output_tensor with (p.data for each shard in self.world_size)
                    if hasattr(dist, "_all_gather_base") and enable_nccl_base_collectives:
                        # New version of PyTorch has all_gather_base, which is faster than chunk and then all_gather.
                        dist._all_gather_base(output_tensor, p_data, group=self.process_group)
                    else:
                        chunks = list(output_tensor.chunk(self.num_shards))
                        dist.all_gather(chunks, p_data, group=self.process_group)

                    # Set p.data = output_tensor (with padding trimmed)
                    update_p_data(output_tensor)

                    if (self.mixed_precision or self._cpu_offload) and not force_full_precision:
                        self.free_fp16_shards([p])

                    if self._cpu_offload and (self.params[0].dtype == self.compute_dtype):
                        self.free_fp16_shards([p])

        torch.cuda.current_stream().wait_stream(self._streams["all_gather"])
        return output_tensors

    @torch.no_grad()
    def use_full_params(self) -> None:
        """
        Switch p.data pointers to use the full params.

        Note: this assumes full params are already gathered.

        Note: this might be called after full_params is already in used. So please
              make sure it is idempotent in that case.
        """
        assert self.has_full_params
        for p in self.params:
            if not p.zero_is_sharded:
                if self.mixed_precision or self._cpu_offload:
                    assert p.zero_fp16_shard is not None
                    assert p.zero_fp16_shard.storage().size() != 0
                    p.data = p.zero_fp16_shard
            else:
                assert p.zero_full_param_padded.storage().size() != 0, f"{p.zero_orig_size} {id(self)}"
                p.data = p.zero_full_param_padded[: p.zero_orig_size.numel()].view(p.zero_orig_size)

    @torch.no_grad()
    def use_fp16_shards(self, params: Optional[List[Parameter]] = None) -> None:
        """Cast FP32 param shard to FP16 for a list of params."""
        if params is None:
            params = self.params
        with torch.cuda.stream(self._streams["fp32_to_fp16"]):
            for p in params:
                assert p.zero_fp16_shard is not None
                alloc_storage(p.zero_fp16_shard, size=p.zero_fp32_shard.size())
                p.zero_fp16_shard.copy_(
                    # If _cpu_offload is True, this will be non-blocking
                    # because _fp32_shard is pinned, otherwise it's a no-op.
                    p.zero_fp32_shard.to(p.zero_fp16_shard.device, non_blocking=True)
                )
                p.data = p.zero_fp16_shard
        torch.cuda.current_stream().wait_stream(self._streams["fp32_to_fp16"])

    @torch.no_grad()
    def use_fp32_shards(self, params: Optional[List[Parameter]] = None) -> None:
        """Use FP32 shard for a list of params."""
        if params is None:
            params = self.params
        for p in params:
            p.data = p.zero_fp32_shard

    @torch.no_grad()
    def free_full_params(self, params: Optional[List[Parameter]] = None) -> None:
        """Free up storage for full parameters."""
        if params is None:
            params = self.params
        self.has_full_params = False
        current_stream = torch.cuda.current_stream()
        for p in params:
            if not p.zero_is_sharded:  # e.g., world_size == 1
                if self.mixed_precision or self._cpu_offload:
                    self.free_fp16_shards([p])
                continue
            # Don't let PyTorch reuse this memory until all work in the current
            # stream is complete.
            p.zero_full_param_padded.record_stream(current_stream)
            # There may be external references to the Tensor Storage that we
            # can't modify, such as references that are created by
            # ctx.save_for_backward in the forward pass. Thus when we
            # unshard parameters, we should reuse the original Tensor
            # Storage object and unshard it in-place. For now, just resize
            # the Storage to 0 to save memory.
            free_storage(p.zero_full_param_padded)

    @torch.no_grad()
    def free_fp16_shards(self, params: Optional[List[Parameter]] = None) -> None:
        """Free storage for FP16 shards for a list of params."""
        if params is None:
            params = self.params
        current_stream = torch.cuda.current_stream()
        for p in params:
            if p.zero_fp16_shard is not None:
                # zero_fp16_shard is allocated in "fp32_to_fp16" stream, so we can't
                # free it until the work in the current stream completes.
                p.zero_fp16_shard.record_stream(current_stream)
                free_storage(p.zero_fp16_shard)

    def delete_fp32_shards(self) -> None:
        for p in self.params:
            if hasattr(p, 'zero_fp32_shard'):
                del p.zero_fp32_shard  # reset _init_param_attr
