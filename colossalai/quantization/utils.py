import torch
import torch.distributed as dist
from packaging import version
from torch import Tensor
from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream
from torch.distributed.utils import _p_assert


def _all_gather_flat_param(
    self,
    padded_unsharded_flat_param: Tensor,
) -> Tensor:
    """
    All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

    Then switch to use the all-gathered tensor.
    """
    _p_assert(
        hasattr(self, "process_group") and hasattr(self, "world_size"),
        "Expects a process group and world size to have been set via `shard()`",
    )
    sharded_flat_param = self.flat_param.data
    expected_numel = sharded_flat_param.numel() * self.world_size
    _p_assert(
        padded_unsharded_flat_param.numel() == expected_numel,
        f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
    )

    pg = self._fake_process_group if self._use_fake_all_gather else self.process_group

    # HACK this should be handled by C10D
    if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
        tensor_list = list(torch.chunk(padded_unsharded_flat_param, dist.get_world_size(pg)))
        work = dist.all_gather(tensor_list, sharded_flat_param, group=pg)
    else:
        if self._comm_hook is None:
            dist.all_gather_into_tensor(
                padded_unsharded_flat_param,
                sharded_flat_param,
                pg,
            )
        else:
            self._comm_hook(None, padded_unsharded_flat_param, sharded_flat_param, pg)

    if self._offload_params:
        # In case of offloading, `flat_param.data` (i.e. sharded param) is
        # created on the pre-unshard stream. We need to hand it over to the
        # unshard stream for all-gather
        _no_dispatch_record_stream(
            sharded_flat_param,
            self._device_handle.current_stream(),  # unshard_stream
        )
    return padded_unsharded_flat_param


def register_params_comm_hook(self, state: object, hook: callable):
    """Register a communication hook for FlatParamHandle.

    This is an enhancement that provides a flexible hook to users where they can specify how FSDP unshards
    parameters across multiple workers.

    .. warning ::
        FSDP communication hook should be registered before running an initial forward pass
        and only once.

    Args:
        state (object): Passed to the hook to maintain any state information during the training process.
        hook (Callable): Callable, which has one of the following signatures:
                        1) ``hook: Callable[torch.Tensor] -> None``:
                        This function takes in a Python tensor, which represents
                        the full, flattened, unsharded gradient with respect to all variables
                        corresponding to the model this FSDP unit is wrapping
                        (that are not wrapped by other FSDP sub-units).
                        It then performs all necessary processing and returns ``None``;
                        2) ``hook: Callable[torch.Tensor, torch.Tensor] -> None``:
                        This function takes in two Python tensors, the first one represents
                        the full, flattened, unsharded gradient with respect to all variables
                        corresponding to the model this FSDP unit is wrapping
                        (that are not wrapped by other FSDP sub-units). The latter
                        represents a pre-sized tensor to store a chunk of a sharded gradient after
                        reduction.
                        In both cases, callable performs all necessary processing and returns ``None``.
                        Callables with signature 1 are expected to handle gradient communication for a `NO_SHARD` case.
                        Callables with signature 2 are expected to handle gradient communication for sharded cases.

    """
    if not self.check_is_root():
        raise AssertionError("register_comm_hook can only be called on a root instance.")

    # if fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
    #     raise AssertionError(
    #         f"Communication hook is not supported for hybrid strategies: {fsdp_state.sharding_strategy}"
    #     )
    if self._handle._comm_hook is not None:
        raise AssertionError("A communication hook is already registered")
    if not callable(hook):
        raise ValueError(f"The communication hook must be callable but got {hook}")
    self._handle._comm_hook = hook
    self._handle._comm_hook_state = state


def patch_fsdp_params_comm_hook():
    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._flat_param import FlatParamHandle

        FlatParamHandle._comm_hook = None
        FlatParamHandle._comm_hook_state = None
        FlatParamHandle._all_gather_flat_param = _all_gather_flat_param
        FSDP.register_params_comm_hook = register_params_comm_hook
    else:
        raise RuntimeError("This fsdp_params_comm_hook patch is not supported while torch version under 2.2.0.")
