import contextlib
import copy
import functools
import os
import traceback
from collections import OrderedDict
from enum import Enum, auto
from typing import (Any, Callable, Dict, Generator, List, NamedTuple, Optional,
                    Set, Union)

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device
from .param_manager import Zero3ParameterManager
from torch.autograd import Variable
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from ._zero3_utils import (apply_to_tensors, assert_in_engine,
                           cast_float_arguments, cast_trensor_to_fp16,
                           cast_trensor_to_fp32, chunk_and_pad, free_storage,
                           get_gradient_predivide_factor, get_shard,
                           replace_state_dict_prefix)
from .reduce_scatter import ReduceScatterBucketer

# TODO: Remove the toggle-enable_nccl_base_collectives in the future
if os.getenv("ENABLE_NCCL_BASE_COLLECTIVES", "1") == "0":
    enable_nccl_base_collectives = False
else:
    enable_nccl_base_collectives = True


class TrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()
    GATHER_FULL_PARAMS = auto()

# TODO: Add clip_grad_norm_
# TODO: Add gather_full_optim_state_dict and get_shard_from_optim_state_dict


class ShardedModel(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_process_group: Optional[ProcessGroup] = None,
                 reshard_after_forward: bool = True,
                 disable_reshard_on_root: bool = True,
                 mixed_precision: bool = False,
                 fp32_reduce_scatter: bool = False,
                 flatten_parameters: bool = True,
                 compute_dtype: Optional[torch.dtype] = None,
                 buffer_dtype: Optional[torch.dtype] = None,
                 reduce_scatter_bucket_size_mb: int = 25,
                 compute_device: Optional[torch.device] = None,
                 no_broadcast_optim_state: Optional[bool] = False,
                 state_dict_device: Optional[torch.device] = None,
                 clear_autocast_cache: bool = False,
                 force_input_to_fp32: bool = False,
                 verbose: bool = False,
                 offload_config: Optional[dict] = None,
                 state_dict_on_rank_0_only: bool = False,
                 gradient_predivide_factor: Optional[float] = 1.0) -> None:
        super().__init__()
        self.logger = get_dist_logger()

        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.reduce_scatter_process_group = reduce_scatter_process_group or self.process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        self.reshard_after_forward = self._orig_reshard_after_forward = reshard_after_forward
        self.disable_reshard_on_root = disable_reshard_on_root
        self.mixed_precision = mixed_precision
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.offload_config = offload_config
        self.compute_dtype = compute_dtype or (torch.float16 if mixed_precision else torch.float32)
        self.buffer_dtype = buffer_dtype or self.compute_dtype
        self.reduce_scatter_bucket_size_mb = reduce_scatter_bucket_size_mb
        self.compute_device = compute_device or torch.device(f'cuda:{get_current_device()}')
        self.uncollected_opt_state: Dict[int, Dict] = {}
        self.no_broadcast_optim_state = no_broadcast_optim_state
        self.state_dict_device = state_dict_device or self.compute_device
        self.clear_autocast_cache = clear_autocast_cache
        self.force_input_to_fp32 = force_input_to_fp32
        self.verbose = verbose
        self.state_dict_on_rank_0_only = state_dict_on_rank_0_only

        self._cpu_offload = offload_config.get('device', None) == 'cpu' if offload_config else False

        # We find if gradient_predivide_factor != 1.0, there may be wrong precision problem
        # So we use 1.0 as the default gradient_predivide_factor
        # However, if you set gradient_predivide_factor to None, we will set gradient_predivide_factor to a value >= 1.0 automatically
        self.gradient_predivide_factor: float = gradient_predivide_factor if gradient_predivide_factor is not None else \
            get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

        self._check_sanity()

        self.params: List[Parameter] = []

        for name, param in module.named_parameters():
            if not hasattr(param, 'zero_is_sharded'):
                self.params.append(param)

        self.module = module

        self.param_manager = Zero3ParameterManager(module, process_group=self.process_group, mixed_precision=self.mixed_precision,
                                                   flatten_parameters=flatten_parameters, compute_dtype=self.compute_dtype, compute_device=self.compute_device,
                                                   offload_config=offload_config)

        self._reset_lazy_init_info()

        # Flag to indicate if we require gradient reduction in the backward
        # pass. This will be False when inside the no_sync context manager.
        self._require_backward_grad_sync: bool = True

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState.IDLE

        # Register hook after state_dict() to remove the "_zero3_module."
        # prefix and before load_state_dict() to add it back.
        self._register_state_dict_hook(functools.partial(_post_state_dict_hook, self.state_dict_on_rank_0_only))
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)

        # Flag to indicate whether state_dict() should automatically gather the full params.
        self._return_full_state_dict = True

        # Flag to guard against preparing gradients multiple times per iteration.
        # This is reset at the end of the backward pass.
        self._pre_backward_hook_has_run = False

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._lazy_init()

        # Start of a forward pass.
        self.training_state = TrainingState.FORWARD

        # For root and mixed precision, we convert the input to FP16 (no_grad is needed for
        # the conversion).
        if self._is_root and self.mixed_precision:
            args, kwargs = cast_float_arguments(cast_trensor_to_fp16, *args, **kwargs)

        # If enabled, convert the input to FP32 if we are in full precision.
        # no_grad is not used because the input might be for a non-root instance,
        # which mean autograd needs to go through the conversion.
        if self.force_input_to_fp32 and not self.mixed_precision:
            args, kwargs = cast_float_arguments(cast_trensor_to_fp32, *args, **kwargs)

        # All-gather full parameters. This will also transfer FP32 parameters to
        # ``self.compute_dtype`` (e.g., FP16 if *mixed_precision* is ``True``).
        self.param_manager.rebuild_full_params()

        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)

        if self.reshard_after_forward:
            self.param_manager.free_full_params()
            if self.mixed_precision or self._cpu_offload:
                self.param_manager.free_fp16_shards()

        # Switch to main FP32 param shard. We maintain this invariant throughout
        # the code, i.e., ``p.data == p.zero_fp32_shard`` after each function. This
        # also ensures that after the first forward, the optimizer state will be
        # initialized with the correct dtype and (sharded) size, since optimizer
        # state is typically initialized lazily in ``optim.step()``.
        self.param_manager.use_fp32_shards()

        # Register pre-backward hooks to all-gather the params for the backward
        # pass (if output's grad was needed). This won't register anything if
        # we are in eval mode.
        #
        # Some model does forward pass multiple times, we need to register the
        # pre-backward hook on every output since the last output's hook has to
        # fire first to setup for backward. However, we use ``self._pre_backward_hook_has_run``
        # to prevent repeated overhead from multiple hook callbacks.
        outputs = self._register_pre_backward_hooks(outputs)

        # Done with a forward pass.
        self.training_state = TrainingState.IDLE

        # Only need to clear cache during forward. During backward, the cache is not used.
        if self.clear_autocast_cache:
            torch.clear_autocast_cache()

        return outputs

    def _check_sanity(self) -> None:
        if self.fp32_reduce_scatter and not self.mixed_precision:
            raise ValueError("fp32_reduce_scatter requires mixed_precision=True")
        if self.compute_device.type == 'cuda':
            input_tensor = torch.ones(1).to(self.compute_device)
            output = list(torch.zeros(self.world_size).to(self.compute_device).chunk(self.world_size))
            dist.all_gather(output, input_tensor, group=self.process_group)
            assert torch.cat(output).sum() == float(self.world_size), (
                f"found {torch.cat(output).sum()} devices in process group but "
                f"world_size={self.world_size}. Check torch.cuda.set_device is called properly"
            )

    def _reset_lazy_init_info(self) -> None:
        self._is_root: Optional[bool] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._reducer: Optional[ReduceScatterBucketer] = None
        self.param_manager.delete_fp32_shards()
        self._output_pre_backward_hook_registered: Optional[List] = None
        self.reshard_after_forward = self._orig_reshard_after_forward

    def _lazy_init(self):
        # Initialize param attributes lazily, in case the param's dtype or
        # device changes after __init__.
        for p in self.params:
            self.param_manager.reset_param_attr(p, self.training)

        # Initialize _is_root and setup streams. These steps would ideally
        # happen in __init__, but _is_root can only be determined after the
        # entire model hierarchy is setup, thus we run it lazily.
        if self._is_root is None:
            self._set_is_root()
            self._setup_streams()
            self._setup_output_hook_list()

        if self._is_root:
            # Buffers stay on GPU, and don't get sharded. Since _cast_buffers
            # applies recursively, we only call this from the root instance.
            self._cast_buffers()

            if self.disable_reshard_on_root:
                # Don't free the full params for the outer-most (root) instance,
                # since those params will be needed immediately after for the
                # backward pass.
                self.reshard_after_forward = False

            # Due to the use of streams, we need to make sure the previous
            # ``optim.step()`` is done before we all-gather parameters.
            self._wait_for_previous_optim_step()

    def _set_is_root(self) -> None:
        """If ``True``, implies that no other :class:`ShardedModel`
        instance wraps this one. Called once by :func:`_lazy_init`.
        Also sets self.children_share_process_group = True if all child
        instances share the same process group. If some child instances use a
        different process group, self.clip_grad_norm_ will raise an error.
        """
        if self._is_root is not None:
            return
        # No Zero3Model instance wraps this, else _is_root would be set to False.
        self._is_root = True
        # If final backward callback is never been queued, state should be IDLE.
        # If final backward callback is queued, the callback should be finished
        # and the state was reset to be IDLE.
        # This should be asserted at the beginning of forward pass in the root instance only.
        # For children instances, if they are checkpointed, state will not be reset to
        # IDLE after each inner forward/backward.
        self._assert_state(TrainingState.IDLE)
        # As the root, we now set all children instances to False and
        # give them a closure to try to queue a wait_for_post_backward.
        self.children_share_process_group = True
        for n, m in self.named_modules():
            # `n != ""` excludes self.
            if n != '' and isinstance(m, ShardedModel):
                # We relax the assert for non-root instance, when the nested inialized module is wrapped
                # again in ShardedModel later, for example after training to run inference.
                assert m._is_root is None or not m._is_root
                if m._is_root is None:
                    m._is_root = False
                if m.process_group != self.process_group:
                    self.children_share_process_group = False

                # if child instance in its own (smaller) world, that was probably an attempt to avoid OOM.
                # Therefore gathering this child's optim state will probably cause OOM, so we won't do it.
                m.no_broadcast_optim_state = m.no_broadcast_optim_state or (
                    (m.world_size == 1) and (m.world_size < self.world_size) and (m.process_group != self.process_group)
                )

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return

        if torch.cuda.is_available():
            # Stream to move main FP32 params (may be on CPU) to FP16 for forward.
            self._streams['fp32_to_fp16'] = torch.cuda.Stream()
            # Stream for all-gathering parameters.
            self._streams['all_gather'] = torch.cuda.Stream()
            # Stream for overlapping grad reduction with the backward pass.
            self._streams['post_backward'] = torch.cuda.Stream()

        self.param_manager.setup_streams(self._streams)
        # Helper for bucketing reduce-scatter ops. This is also shared with
        # children instances to improve bucket utilization.
        self._reducer = ReduceScatterBucketer(self.reduce_scatter_bucket_size_mb)
        # We share streams with all children instances, which allows them to
        # overlap transfers across the forward pass without synchronizing with
        # the default stream.
        for n, m in self.named_modules():
            if n != "" and isinstance(m, ShardedModel):
                m._streams = self._streams
                m._reducer = self._reducer
                m.param_manager.setup_streams(self._streams)

    def _setup_output_hook_list(self) -> None:
        """set up a list to avoid registering pre-backward hooks
        incorrectly.
        """
        assert self._is_root, "This should only be called on the root"
        self._output_pre_backward_hook_registered = []
        for n, m in self.named_modules():
            if n != "" and isinstance(m, ShardedModel):
                m._output_pre_backward_hook_registered = self._output_pre_backward_hook_registered

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`ShardedModel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return
        if self.mixed_precision or self._cpu_offload:
            self._streams["fp32_to_fp16"].wait_stream(torch.cuda.current_stream())
        else:
            self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def _cast_buffers(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, memo: Optional[Set] = None
    ) -> None:
        """Move all buffers to the given *device* and *dtype*.

        If *device* or *dtype* are not given, then they will default to
        ``self.compute_device`` and ``self.buffer_dtype``, respectively. In the
        case of nested ShardedModel instances, we will respect the child instance's
        ``compute_device`` and ``buffer_dtype`` configuration.

        Args:
            device (torch.device, Optional):
                device to cast buffers to (defaults to compute_device)
            dtype (torch.dtype, Optional):
                dtype to cast buffers to (defaults to buffer_dtype)
            memo (Set, Optional):
                set of modules that have already been processed
        """
        if memo is None:
            memo = set()
        for module in self.modules():
            if module is not self and isinstance(module, ShardedModel):
                # Allow any child Zero3Model instances to handle their own buffers.
                module._cast_buffers(device=device, dtype=dtype, memo=memo)
            elif module not in memo:
                memo.add(module)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None:
                        continue
                    buf = buf.to(device=device or self.compute_device)
                    if torch.is_floating_point(buf):
                        buf = buf.to(dtype=dtype or self.buffer_dtype)
                    setattr(module, name, buf)

    @torch.no_grad()
    def _prep_grads_for_backward(self) -> None:
        """Make sure p.grad is correctly prepared for the backward with
        right shape, device, accumulated values, etc.
        """
        for p in self.params:
            if p.grad is not None:
                if p.grad.device != p.data.device:
                    p.grad = None
                elif p.grad.size() == p.zero_orig_size:
                    if not p.zero_is_sharded:
                        p.zero_saved_grad = p.grad.data
                        p.grad = None
                    else:
                        # This is gradient accumulation with no_sync context.
                        pass
                elif p.grad.size() == p.zero_fp32_shard.shape:
                    # This is gradient accumulation without no_sync context.
                    # We save the grad shard and set p.grad to None for this backward pass.
                    # We will accumulate after this pass's grad is generated and reduced and
                    # sharded.
                    p.zero_saved_grad_shard = p.grad.data
                    p.grad = None
                else:
                    raise AssertionError(f"unexpected grad shape: {p.grad.size()}")

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward.

        Returns:
            outputs: new outputs with hooks registered if they requires gradient.
        """
        if not torch.is_grad_enabled():
            return outputs  # don't register hooks if grad isn't enabled

        if self._is_root:
            # This actually means that only root instance has
            # _post_backward_callback_queued defined. Accidentally accessing this field
            # will assert on all other instances, giving us a nice bug checker.
            self._post_backward_callback_queued = False

        def _pre_backward_hook(*unused: Any) -> None:
            # try to queue final backward callback only once for root, so
            # that final backward callback is attached to the outer most
            # backward graph task and called after all the backward
            # calls are completed.
            if self._is_root:
                self._register_final_backward_hook()

            # All-gather full parameters or switching to the full params.
            #
            # This needs to be done on every pre_backward hook, even within the same
            # iteration (i.e. for checkpointed, multiple forward pass modules). This is
            # because after the forward pass (i.e. in checkpoint inner graph), we always
            # switch to fp32_shard in the ``forward`` function.
            #
            # We used to do this only after the ``self._pre_backward_hook_has_run``
            # boolean guard below, which is incorrect. It worked in pytorch < 1.9 for
            # some unknown reason, but pytorch 1.10 nightly exposed this bug.
            #
            # Note, both ``self.param_manager.rebuild_full_params`` and ``self.param_manager.use_full_params`` are
            # idempotent.  So in case they are called unnecessarily, they don't incur much
            # overhead.
            if self.reshard_after_forward:
                self.param_manager.rebuild_full_params()
            else:
                self.param_manager.use_full_params()

            # Only run the ``self._prep_grads_for_backward`` once per iteration (i.e. in case
            # it is multiple outputs or multiple forward passes).
            if not self._pre_backward_hook_has_run:
                self._pre_backward_hook_has_run = True
                # Start of a backward pass for the first time in an iteration.
                self._assert_state([TrainingState.IDLE, TrainingState.PRE_BACKWARD])
                # Prepare p.grad so that it is in the right shape, device, accumulated values, etc.
                self._prep_grads_for_backward()

            # Transition to PRE_BACKWARD state if currently IDLE. We can transition from POST_BACKWARD
            # to IDLE when ShardedModel is within activation checkpointing and called multiple times, due to the
            # extra forward pass for re-computation.
            if self.training_state == TrainingState.IDLE:
                self.training_state = TrainingState.PRE_BACKWARD
            self._assert_state([TrainingState.PRE_BACKWARD, TrainingState.POST_BACKWARD])

        _registered = 0

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            # We don't register the pre_backward hook on the same tensor that has been
            # returned from an inner ShardedModel, unless it is the first one. This does
            # not cover all problematic cases though. A tensor not from an inner
            # ShardedModel can cause problems too:
            # ```
            #   x = layer1(input)
            #   state = [x]  # better change to x.detach(), not fixed by the following if-condition
            #   x = inner_zero3_module_layer2(x)
            #   state.append(x)  # better change to x.detach(), but fixed by the following if-condition
            #   x = layer3(x)
            #   return x, state
            # ```
            # The tensors in `state`, if not detached, can be registered with
            # backward hooks (in addition to the `x` on the last line). In that case,
            # pre-backward hook can fire multiple times in the order that causes
            # the outer ShardedModel to crash.
            #
            # The best practice is for modules to be wrapped by ShardedModel to return 1 and only
            # 1 tensor to be used for backward. All other tensors returned should be
            # detached.
            nonlocal _registered
            assert self._output_pre_backward_hook_registered is not None
            if t.requires_grad and (_registered == 0 or id(t) not in self._output_pre_backward_hook_registered):
                t.register_hook(_pre_backward_hook)
                self._output_pre_backward_hook_registered.append(id(t))
                _registered += 1
            return t

        # Attach hooks to Tensor outputs.
        outputs = apply_to_tensors(outputs, _register_hook)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        """
        Register backward hooks to reshard params and reduce-scatter grads.

        This is called during forward pass. The goal is to attach a hook
        on each of the parameter's gradient generating function (``grad_acc``
        below) so that the hook is called *after* all gradients for that
        param are computed.

        Goals:

        1. We want the hook to fire once and only once *after* all gradients
        are accumulated for a param.
        2. If it fires more than once, we end up incorrectly shard the grad
        multiple times. (could lead to dimension too small)
        3. If it fires once but too early or doesn't fire, we leave gradients
        unsharded. (could lead to dimension too large)

        Due to multiple-pass forward, this function can be called on
        the same parameter multiple times in a single forward pass. If we register
        the hook multiple time, we end up getting called multiple times. We
        could try to get a new hook every time and delete the previous one
        registered. However, due to *unknown reason* (I have debugged it for
        a long time!), in mixed precision mode, we get two different ``grad_acc``
        objects below during different calls of this function (in the same
        forward pass). If we keep the last one, the hook end up firing too
        early. In full precision mode, we luckily get the *same* ``grad_acc``
        object, so deleting and re-registering still ensured the hook fire
        once after all gradients are generated. However, we find if we use activation
        checkpoint in mixed precision mode, hook on ``grad_acc`` object won't be
        fire for *unknown reason*. So we finally register hook on parameter directly.

        Empirically, keep the first hook register per forward pass seems to
        work the best. We do need to remove the hook at the end of the
        backward pass. Otherwise, the next forward pass will not register
        a new hook, which is needed for a new forward pass.
        """
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for p in self.params:
            if p.requires_grad:
                if hasattr(p, "zero_shard_bwd_hook"):
                    continue
                # For mixed precision with activation checkpoint, hooks on GradAccumulation won't be fired normally
                # Instead we register hook on parameter
                # In this way, we can't modify param.grad and param.data directly, which leads to more memory usage
                # Register a hook on the first call, empirically, autograd
                # fires it at the end for this param, which makes sense.
                # p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                # assert p_tmp.grad_fn is not None
                # grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                # handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
                # p.zero_shard_bwd_hook = (grad_acc, handle)
                handle = p.register_hook(functools.partial(self._post_backward_hook, p))
                p.zero_shard_bwd_hook = handle

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will replace
        ``param.grad`` with a single shard of the summed gradient across all
        GPUs. This shard will align with the current GPU rank. For example::

            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]

            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8

        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by `param_manager`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        # First hook callback will see PRE state. If we have multiple params,
        # then subsequent hook callbacks will see POST state.
        self._assert_state([TrainingState.PRE_BACKWARD, TrainingState.POST_BACKWARD])
        self.training_state = TrainingState.POST_BACKWARD
        if grad is None:
            return

        assert grad is not None, param.shape
        if grad.requires_grad:
            raise RuntimeError("ShardedModel only works with gradients that don't require gradients")

        if self._require_backward_grad_sync or self.reshard_after_forward:
            # Free full params. As a special case, we don't free the full params
            # when in a ``no_sync`` context (as inversely indicated by
            # ``self._require_backward_grad_sync``), since the params will not
            # get updated before the next forward. This saves networking
            # bandwidth but uses more GPU memory.
            self.param_manager.free_full_params([param])

        if self.mixed_precision:
            # This is a no-op if reshard_after_forward is True, since we already
            # free the param shard when rebuilding the full params in the
            # pre_backward_hook.
            self.param_manager.free_fp16_shards([param])

        # Switch to FP32 shard after backward.
        # Cannot modify param.data, so we switch to FP32 in final backward hook
        # self.param_manager.use_fp32_shards([param])

        if not self._require_backward_grad_sync:
            return

        # Wait for all work in the current stream to finish, then start the
        # reductions in post_backward stream.
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._streams["post_backward"]):
            new_grad = grad.clone()

            if self.mixed_precision and self.fp32_reduce_scatter:
                # Cast grad to FP32.
                new_grad.data = new_grad.data.to(param.dtype)

            if self.gradient_predivide_factor > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                new_grad.data.div_(self.gradient_predivide_factor)

            orig_grad_data = new_grad.data
            if param.zero_is_sharded:
                assert self._reducer is not None
                # Save the unsharded grad for reduction. We will asynchronously accumulate the reduced gradient into
                # param.zero_saved_grad_shard. If this ShardedModel module was called multiple times it's possible that multiple
                # gradient reductions will happen in an undefined order. But addition commutes, so this order doesn't
                # matter, neglecting rounding.
                # Clear grad on the tensor, so any repeated gradient computations do not interfere with this reduction.
                #
                # The effect on memory consumption is not usually significant. No extra memory is allocated if this
                # module is called only once, reduction happens quickly, or the tensor is bucketed. If the module is
                # called multiple times, and the backwards pass runs far enough ahead of the `post_backward` stream,
                # then we can end up with multiple unsharded gradients allocated and queued for reduction.
                #
                # We could guard against this by using CUDA events (see record_event, wait_event in torch.cuda.Stream).
                # This ensures the `default` stream will wait for the `post_backward` stream to complete the last
                # reduction for this module, before scheduling additional reduction work. Then at most there are two
                # unsharded gradients allocated; one for a pending reduction, and one for gradient computation.
                callback_fn = functools.partial(self._reduce_scatter_callback, param)
                grad_chunks = chunk_and_pad(orig_grad_data, self.reduce_scatter_process_group.size())
                self._reducer.reduce_scatter_async(
                    grad_chunks, group=self.reduce_scatter_process_group, callback_fn=callback_fn
                )
            else:
                # Currently the only way for _is_sharded to be False is if
                # world_size == 1. This could be relaxed in the future, in which
                # case grads should be all-reduced here.
                assert self.world_size == 1
                self._reduce_scatter_callback(param, new_grad)

            # After _post_backward_hook returns, orig_grad_data will eventually
            # go out of scope, at which point it could otherwise be freed for
            # further reuse by the main stream while the div/reduce_scatter/copy
            # are underway in the post_backward stream. See:
            # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
            orig_grad_data.record_stream(self._streams["post_backward"])

    def _reduce_scatter_callback(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        """Hook to call on each param after the reduce-scatter."""
        assert torch.cuda.current_stream() == self._streams["post_backward"]
        self._assert_state(TrainingState.POST_BACKWARD)
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            reduced_grad.data.div_(self.gradient_postdivide_factor)
        # Cast grad to param's dtype (typically FP32). Note: we do this
        # before the cpu offload step so that this entire hook remains
        # non-blocking. The downside is a bit more D2H transfer in that case.
        if self.mixed_precision:
            orig_param_grad_data = reduced_grad.data
            reduced_grad.data = reduced_grad.data.to(dtype=param.zero_fp32_shard.dtype)
            # Don't let this memory get reused until after the transfer.
            orig_param_grad_data.record_stream(torch.cuda.current_stream())

        if param.zero_is_sharded:
            # Accumulate into the gradient shard.
            if getattr(param, "zero_saved_grad_shard", None) is None:
                param.zero_saved_grad_shard = reduced_grad.data
            else:
                assert (
                    param.zero_saved_grad_shard.shape == reduced_grad.shape
                ), f"{param.zero_saved_grad_shard.shape} vs {reduced_grad.shape}"
                param.zero_saved_grad_shard.data += reduced_grad.data
            reduced_grad = param.zero_saved_grad_shard.data
        else:
            # We can't modify the dtype of grad in this function
            # So we use `param.zero_saved_grad` to store gradient
            # This is useful when using mixed precision mode on single node
            if getattr(param, 'zero_saved_grad', None) is None:
                param.zero_saved_grad = reduced_grad.data
            else:
                param.zero_saved_grad.data += reduced_grad.data

        # Optionally move gradients to CPU, typically used if one is running the optimizer on the CPU. Once the full
        # backwards pass completes, we will set `.grad` to the CPU copy.
        if self._cpu_offload:
            param.zero_cpu_grad.copy_(reduced_grad.data, non_blocking=True)
            # Don't let this memory get reused until after the transfer.
            reduced_grad.data.record_stream(torch.cuda.current_stream())

    def _register_final_backward_hook(self) -> None:
        """Try to queue a `_final_backward_hook` callback.

        Only called on root and only queue one callback at the beginning of
        outer most backward.
        """
        assert self._is_root
        if not self._post_backward_callback_queued:
            self._assert_state([TrainingState.IDLE])
            self._post_backward_callback_queued = True
            Variable._execution_engine.queue_callback(self._final_backward_hook)

    @torch.no_grad()
    def _final_backward_hook(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        # None, backward runtime swallow the assert error, so we use assert_in_engine() here.
        assert_in_engine(self._is_root, "FinalBackwardHook not called on root")
        # Check if the root module has params and if any of them has
        # the `requires_grad` field set. If `requires_grad=False` for
        # all the params, the post_backward hook will not fire and the
        # state will remain in `TrainingState.PRE_BACKWARD`.
        if any([p.requires_grad for p in self.params]):
            self._assert_state(TrainingState.POST_BACKWARD)
        else:
            self._assert_state(TrainingState.PRE_BACKWARD)
        self.param_manager.use_fp32_shards()
        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self._streams["post_backward"]):
                assert_in_engine(self._reducer is not None, "FinalBackwardHook: reducer is None")
                assert self._reducer is not None  # make mypy happy
                self._reducer.flush()
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self._cpu_offload:
                # Wait for the non-blocking GPU -> CPU grad transfers to finish.
                torch.cuda.current_stream().synchronize()

        # A backward pass is done, clean up below.
        # Free reducer buffers.
        if self._reducer is not None:
            self._reducer.free()

        def _finalize_parameters(zero_module: ShardedModel) -> None:
            """Helper used below on all zero3 modules."""
            for p in zero_module.params:
                if not p.requires_grad:
                    continue
                if hasattr(p, "zero_shard_bwd_hook"):
                    p.zero_shard_bwd_hook.remove()
                    delattr(p, "zero_shard_bwd_hook")

                # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
                # remains the unsharded gradient accumulated from prior no-sync passes, and p.zero_saved_grad_shard
                # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
                # sync passes, if desired.
                if not self._require_backward_grad_sync:
                    continue

                # Parameter and gradient devices must match.
                if hasattr(p, "zero_cpu_grad"):
                    assert_in_engine(p.device == torch.device("cpu"),
                                     f"FinalBackwardHook: incorrect cpu_grad device {p.device}")
                    p.grad = p.zero_cpu_grad
                elif hasattr(p, "zero_saved_grad_shard"):
                    assert_in_engine(
                        p.device == p.zero_saved_grad_shard.device,
                        f"FinalBackwardHook: incorrect saved_grad_shard device {p.device} vs {p.zero_saved_grad_shard.device}",
                    )
                    p.grad = p.zero_saved_grad_shard
                elif hasattr(p, 'zero_saved_grad'):
                    p.grad = p.zero_saved_grad

                if hasattr(p, "zero_saved_grad_shard"):
                    delattr(p, "zero_saved_grad_shard")
                if hasattr(p, 'zero_saved_grad'):
                    delattr(p, "zero_saved_grad")

        # Update root and nested ShardedModel's hooks and flags.
        for m in self.modules():  # includes self
            if isinstance(m, ShardedModel):
                _finalize_parameters(m)
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    # Check if the module has params and if any of them has
                    # the `requires_grad` field set. If `requires_grad=False` for
                    # all the params, the post_backward hook will not fire and the
                    # state will remain in `TrainingState.PRE_BACKWARD`.
                    if any([p.requires_grad for p in m.params]):
                        m._assert_state(TrainingState.POST_BACKWARD)
                    else:
                        m._assert_state(TrainingState.PRE_BACKWARD)
                else:
                    # When `m` and its children has no params or has params but
                    # none with `requires_grad==True`, there are two cases:
                    # 1. output tensors are `requires_grad==True`. In this case,
                    # pre-backward hook is still registered, so it is in PRE_BACKWARD state.
                    # 2. output tensors are `requires_grad==False`. In this case,
                    # pre-backward hook is not registered, so it is in IDLE state.
                    m._assert_state([TrainingState.PRE_BACKWARD, TrainingState.IDLE])
                m.training_state = TrainingState.IDLE

                if m._is_root:
                    # reset this flag for cases like "one forward pass + multiple backward passes"
                    self._post_backward_callback_queued = False
                    # clear this list for next iteration
                    assert_in_engine(
                        self._output_pre_backward_hook_registered is not None,
                        "FinalBackwardHook: self._output_pre_backward_hook_registered should not be None",
                    )
                    assert self._output_pre_backward_hook_registered is not None  # make mypy happy
                    self._output_pre_backward_hook_registered.clear()

    @contextlib.contextmanager
    def gather_full_params(self, recurse: bool = True, volatile: bool = False) -> Generator:
        """
        A context manager to expose full params for the current ShardedModel instance.
        Can be useful *after* forward/backward for a model to get the params for
        additional processing or checking. Parameters will be gathered in full
        precision (e.g., FP32).

        .. note:: This can be used on inner ShardedModels.

        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.

        .. note:: The full parameters will be freed after the context manager
            exits; it is up to the caller to clone them if needed.

        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``volatile=True``, in which case there
            are no guarantees about persistence).

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                ShardedModel instances (default: True)
            volatile (bool, Optional): if ``True``, modifications to params are
                not guaranteed to persist after the context manager exists;
                enabling this can be slightly more efficient (default: False)
        """
        if recurse:
            with contextlib.ExitStack() as stack:
                # Summon all params for any nested Zero3Model instances.
                for module in self.modules():
                    if isinstance(module, ShardedModel):
                        stack.enter_context(module.gather_full_params(recurse=False, volatile=volatile))
                # Yield to the caller, with full params in all nested instances.
                yield
            # Exiting from the ExitStack will re-shard params.
            return
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            self._assert_state(TrainingState.IDLE)
            # Set the state so that we assert when trying to go into
            # forward/backward.
            self.training_state = TrainingState.GATHER_FULL_PARAMS
            full_tensors = self.param_manager.rebuild_full_params(force_full_precision=True)
            assert full_tensors is not None
            with contextlib.ExitStack() as stack:
                try:
                    yield
                finally:
                    stack.close()
                    for p, (full_tensor, safe_to_free) in zip(self.params, full_tensors):
                        if not volatile:
                            # Copy any changes made to the full params back into
                            # the corresponding local shards.
                            local_shard, _ = get_shard(full_tensor)
                            p.zero_fp32_shard.copy_(local_shard.view_as(p.zero_fp32_shard))
                        if safe_to_free:
                            free_storage(full_tensor)
                    self.has_full_params = False
                    self.param_manager.use_fp32_shards()
                    self.training_state = TrainingState.IDLE

    def apply(self, fn: Callable[[nn.Module], None]) -> "ShardedModel":
        """
        Applies ``fn`` recursively to every submodule (as returned by
        ``.children()``) as well as self. Typical use includes initializing the
        parameters of a model.

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (nn.Module): function to be applied to each submodule

        Returns:
            Module: self
        """
        is_uninitialized = self._is_root is None
        self._assert_state(TrainingState.IDLE)
        with self.gather_full_params(recurse=False):
            return_value = super().apply(fn)
        # summon_full_params will call _lazy_init, which sets _is_root. However,
        # apply() may be called directly on children instances to do weight
        # init, so we should reset the _is_root flag in this case.
        if is_uninitialized and self._is_root:
            for module in self.modules():
                if isinstance(module, ShardedModel):
                    module._reset_lazy_init_info()
        return return_value

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __getstate__(self) -> Dict[str, str]:
        """Serialize the state.

        Some properties are not serializable (e.g., process groups, streams), so
        we remove them and try to reconstruct them in :func:`__setstate__`.
        """
        state = copy.copy(self.__dict__)
        state["is_sharded"] = [p.zero_is_sharded for p in self.params]
        state["orig_sizes"] = [p.zero_orig_size for p in self.params]
        if state["process_group"] is not None:
            state["process_group"] = "MISSING"  # process_group isn't pickleable
        if state["process_group_reduce_scatter"] is not None:
            state["process_group_reduce_scatter"] = "MISSING"  # process_group_reduce_scatter isn't pickleable
        self._reset_lazy_init_info()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Intercept state setting and perform needed changes on params."""
        super().__setstate__(state)

        def fixup(p: Parameter, is_sharded: bool, size: torch.Size) -> Parameter:
            assert isinstance(p, Parameter)
            p.data = p.data.clone()  # move tensors out of shared memory
            p.zero_is_sharded = is_sharded
            p.zero_orig_size = size
            return p

        self.params = [
            fixup(p, is_sharded, size) for p, is_sharded, size in zip(self.params, self.is_sharded, self.orig_sizes)
        ]
        del self.is_sharded
        del self.orig_sizes
        self._reset_lazy_init_info()

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    @contextlib.contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across ShardedModel
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context.

        .. note:: This likely results in higher memory usage because ShardedModel will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: Gradient accumulation can be done without this context,
            avoiding the extra GPU memory overhead, but with the extra
            networking overhead.
        """
        self._lazy_init()
        assert self._is_root, "no_sync on inner ShardedModel is not supported"
        self._assert_state(TrainingState.IDLE)
        # This instance may wrap other ShardedModel instances and we
        # need to set all of them to accumulate gradients.
        old_flags = []
        for m in self.modules():  # includes self
            if isinstance(m, ShardedModel):
                old_flags.append((m, m._require_backward_grad_sync))
                m._require_backward_grad_sync = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                assert m._require_backward_grad_sync is False
                m._require_backward_grad_sync = old_flag

    def _assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = f"expected to be in states {state} but current state " f"is {self.training_state}"
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            self.logger.error(f'Zero3 instance {self} got error: {msg}', ranks=[0])
            if self.rank == 0:
                traceback.print_stack()
            raise ValueError(msg)

    def extra_repr(self) -> str:
        repr = (
            f"world_size={self.world_size}, "
            f"mixed_precision={self.mixed_precision}, "
        )
        if self.verbose:
            repr = (
                f"rank={self.rank}, " + repr + f"reshard_after_forward={self.reshard_after_forward}, "
                f"compute_dtype={self.compute_dtype}, "
                f"buffer_dtype={self.buffer_dtype}, "
                f"fp32_reduce_scatter={self.fp32_reduce_scatter}, "
                f"compute_device={self.compute_device}"
                f"reduce_scatter_bucket_size_mb={self.reduce_scatter_bucket_size_mb}, "
                f"clear_autocast_cache={self.clear_autocast_cache}"
                f"force_input_to_fp32={self.force_input_to_fp32}"
                f"offload_config={self.offload_config}"
            )
        return repr

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns the whole (unsharded) state of the module. Parameters are not
        sharded, so the resulting state_dict can be loaded directly by the
        wrapped Module without any sharding-specific logic. Returned tensors
        will be full precision (e.g., FP32).

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._lazy_init()

        def maybe_cast_buffers(dtype: Optional[torch.dtype] = None) -> None:
            if self.mixed_precision:
                self._cast_buffers(dtype=dtype)

        assert self._return_full_state_dict is True, 'Only support return full state dict now'
        if self.training_state != TrainingState.GATHER_FULL_PARAMS:
            with self.gather_full_params(recurse=False, volatile=True):
                maybe_cast_buffers(torch.float32)
                state_dict = super().state_dict()
        else:
            maybe_cast_buffers(torch.float32)
            state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        if self._cpu_offload:
            for k, tensor in state_dict.items():
                state_dict[k] = tensor.cpu()

        # In case we are in mixed precision, restore buffers back to buffer_dtype.
        maybe_cast_buffers()
        return state_dict

    def load_state_dict(
        self, state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """
        Load a whole (unsharded) state_dict.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        if self._return_full_state_dict:
            with self.gather_full_params():
                return self.module.load_state_dict(state_dict, strict)
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            return self.module.load_state_dict(state_dict, strict)


def _post_state_dict_hook(
    state_dict_on_rank_0_only: bool,
    module: Zero3ParameterManager,
    state_dict: "OrderedDict[str, torch.Tensor]",
    prefix: str,
    *args: Any,
) -> "OrderedDict[str, torch.Tensor]":
    # When state_dict_on_rank_0_only is ``True``, ``model.state_dict()`` will only
    # returns full state dict on rank 0 and return empty dict non-rank 0,
    # which allow ShardedModel to skip the GPU -> CPU copy on
    # non-rank 0 altogether and prevent OOM.
    if state_dict_on_rank_0_only and dist.get_rank() != 0:
        state_dict.clear()
        return state_dict
    # Assuming we are in a ``gather_full_params()`` context, we need to clone
    # each tensor so that it does not get freed (in-place) when the context
    # exits. At the same time, this hook can be called multiple times
    # recursively, so we need to make sure that we only clone each tensor at
    # most once. Thus we add an attribute on the tensor called "_has_been_cloned"
    # which keeps track of tensors that are no longer at risk of being freed.
    for key in state_dict.keys():
        if not key.startswith(prefix) or getattr(state_dict[key], "_has_been_cloned", False):
            continue
        if state_dict[key].device.type != module.state_dict_device.type:
            state_dict[key] = state_dict[key].to(device=module.state_dict_device)
            state_dict[key]._has_been_cloned = True
        elif module.training_state == TrainingState.GATHER_FULL_PARAMS:
            # We copy the state_dict since full param will be freed after we
            # exit the ``summon_full_params()`` context.
            state_dict[key] = state_dict[key].clone()
            state_dict[key]._has_been_cloned = True

    # Remove "_zero3_module." prefix
    replace_state_dict_prefix(state_dict, prefix + "_zero3_module.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"], prefix: str, *args: Any
) -> None:
    replace_state_dict_prefix(state_dict, prefix, prefix + "_zero3_module.")
