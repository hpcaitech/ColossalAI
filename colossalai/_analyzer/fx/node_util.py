from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.autograd.profiler_util import _format_memory
from torch.fx import Node

from colossalai._analyzer.envs import MeshConfig


def intersect(a, b):
    return {k: a[k] for k in a if k in b}


def subtract(a, b):
    return {k: a[k] for k in a if k not in b}


def union(a, b):
    return {**a, **b}


def compute_size_in_bytes(elem: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Compute the size of a tensor or a collection of tensors in bytes.

    Args:
        elem (torch.Tensor | Dict | List | Tuple | int): Arbitrary nested ``torch.Tensor`` data structure.

    Returns:
        int: The size of the tensor or the collection of tensors in bytes.
    """
    nbytes = 0
    if isinstance(elem, torch.Tensor):
        if elem.is_quantized:
            nbytes += elem.numel() * torch._empty_affine_quantized([], dtype=elem.dtype).element_size()
        else:
            nbytes += elem.numel() * torch.tensor([], dtype=elem.dtype).element_size()
    elif isinstance(elem, dict):
        value_list = [v for _, v in elem.items()]
        nbytes += compute_size_in_bytes(value_list)
    elif isinstance(elem, tuple) or isinstance(elem, list) or isinstance(elem, set):
        for e in elem:
            nbytes += compute_size_in_bytes(e)
    return nbytes


@dataclass
class MetaInfo:
    r"""
    The base class to store all profiling and static graph analysis information
    needed for auto-parallel system in Colossal-AI.
    ============================================================================
                            -------------------------------
                            |          FX.Node            |    <-----
    [input/param] are  ---> |[input/param]      [grad_inp]|    [grad_inp] contributes to the
    placeholders (might be  |     | \__________     |     |    profiled peak memory in backward
    saved for backward.     |     |            \    |     |    pass. [grad_param] is calculated
                            |     |             \   |     |    separately.
                            | [interm] -------> [grad_int]|    <-----
                            |     |  \_________     |     |    [grad_interm] marks the peak
                            |    / \           \    |     |    memory in backward pass.
    [x] is not counted ---> | [x]  [interm] --> [grad_int]|    <-----
    in [interm] because     |          |  \_____    |     |
    it is not saved for     |          |        \   |     |
    backward.               |      [output]      \  |     |    <----- [output] is potentially
                            -------------------------------    [input] for the next node.
    ============================================================================

    Accumulate Size = ALL_PREVIOUS_CTX U {Interm Size + Output Size}
    Output Size = ([output] in global_ctx and not is_alias)
    Temp Size = ([output] not in global_ctx and not is_alias)
    Backward Size = ([grad_inp])

    Usage:
        >>> for node in graph.nodes:
        >>>     n_info = MetaInfo(node)     # will create a new MetaInfo instance and store in node.meta['info']
        >>>                                 # if not exist, otherwise return the existing one
        >>>     n_info.to_recompute = ...   # set the to_recompute attribute

    Remarks:
        This feature is experimental and all the entries are subject to change.
    """

    # reference
    node: Node

    # directory
    mod_dir: str = ""

    # ctx[data_ptr] = Tensor
    # mark the storage for ctx.save_for_backward
    global_ctx: Dict[str, torch.Tensor] = field(default_factory=lambda: {})  # globally shared
    curr_ctx: Dict[str, torch.Tensor] = field(default_factory=lambda: {})  # global_ctx till this node

    # should be updated after each graph manipulation
    # ============================== Update ====================================
    # parameter and buffer within ``Node``
    parameters: Dict[str, torch.nn.Parameter] = field(default_factory=lambda: {})
    buffers: Dict[str, torch.Tensor] = field(default_factory=lambda: {})

    inputs: Tuple[torch.Tensor] = ()
    outputs: Tuple[torch.Tensor] = ()
    is_alias: Tuple[bool] = ()  # whether the output is an alias of input

    # compute cost
    fwd_flop: Optional[int] = 0
    bwd_flop: Optional[int] = 0

    # communication cost (should be the size in bytes of communication)
    fwd_comm: Optional[int] = 0
    bwd_comm: Optional[int] = 0

    # should keep the same whenever manipulated
    # ============================= Invariant ==================================
    activation_checkpoint: Tuple[torch.Tensor] = ()  # (region_0, region_1, ...) support nested codegen
    to_offload: Optional[bool] = False
    sharding_spec: str = "RR"

    def __new__(cls, node: Node, **kwargs):
        orig_init = cls.__init__

        # if initialized, return the existing one
        # should disable the __init__ function
        if node.meta.get("info", None) is not None:

            def _dummy(self, *args, **kwargs):
                if getattr(self, "_is_init", False):
                    self._is_init = True
                    orig_init(self, *args, **kwargs)
                cls.__init__ = orig_init

            cls.__init__ = _dummy
            return node.meta["info"]
        return super().__new__(cls)

    def __post_init__(self):
        self.node.meta["info"] = self

    @property
    def fwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return self.fwd_flop / tflops + self.fwd_comm / bandwidth

    @property
    def bwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return self.bwd_flop / tflops + self.bwd_comm / bandwidth

    @property
    def param_size(self):
        return compute_size_in_bytes(self.parameters)

    @property
    def buffer_size(self):
        return compute_size_in_bytes(self.buffers)

    @property
    def output_size(self):
        """Used in CheckpointSolver"""
        output_ctx = {
            o.data_ptr(): o
            for o, is_alias in zip(self.outputs, self.is_alias)
            if not is_alias and isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter)
        }
        return compute_size_in_bytes(intersect(self.global_ctx, output_ctx))

    @property
    def accumulate_size(self):
        """Used in CheckpointSolver"""
        output_ctx = {
            o.data_ptr(): o
            for o, is_alias in zip(self.outputs, self.is_alias)
            if not is_alias and isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter)
        }
        return compute_size_in_bytes(union(self.curr_ctx, intersect(self.global_ctx, output_ctx)))

    @property
    def temp_size(self):
        """Used in CheckpointSolver"""
        output_ctx = {
            o.data_ptr(): o
            for o, is_alias in zip(self.outputs, self.is_alias)
            if not is_alias and isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter)
        }
        return compute_size_in_bytes(subtract(output_ctx, self.global_ctx))

    @property
    def backward_size(self):
        """Used in CheckpointSolver"""
        return compute_size_in_bytes(self.inputs)

    def __repr__(self):
        s = f"Node {self.node.name}"
        if self.parameters:
            s += f"\n\thas parameter of size {_format_memory(self.param_size)}"
        if self.buffers:
            s += f"\n\thas buffer of size {_format_memory(self.buffer_size)}"
        if self.output_size:
            s += f"\n\thas output activation of size {_format_memory(self.output_size)}"
        # if self.total_size:
        #     s += f'\n\thas total activation of size {_format_memory(self.total_size)}'
        if self.temp_size:
            s += f"\n\thas temp activation of size {_format_memory(self.temp_size)}"
        if self.backward_size:
            s += f"\n\thas backward activation of size {_format_memory(self.backward_size)}"
        s += (
            f"\n\tfwd_flop = {self.fwd_flop}"
            f"\n\tbwd_flop = {self.bwd_flop}"
            f"\n\tfwd_comm = {self.fwd_comm}"
            f"\n\tbwd_comm = {self.bwd_comm}"
            f"\n\tto_recompute = {self.to_recompute}"
            f"\n\tto_offload = {self.to_offload}"
            f"\n\tsharding_spec = {self.sharding_spec}"
        )
        return s
