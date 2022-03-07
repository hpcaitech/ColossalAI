import torch
from colossalai.registry import OPHOOKS
from colossalai.zero.shard_utils import TensorShardStrategy

from ._base_ophook import BaseOpHook


@OPHOOKS.register_module
class ZeroHook(BaseOpHook):
    """
    A hook to process sharded param for ZeRO method.
    """

    def __init__(self, process_group):
        super().__init__()
        self.shard_strategy = TensorShardStrategy(process_group=process_group)

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.gather([param.col_attr.data])
            param.data = param.col_attr.data.payload

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.shard([param.col_attr.data])
            param.data = torch.empty([], dtype=param.col_attr.data.dtype, device=param.col_attr.data.payload.device)

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.gather([param.col_attr.data])
            param.data = param.col_attr.data.payload

            # Store local accumulated grad shard
            if param.grad is not None:
                param.col_attr.grad = param.grad.data
                param.grad = None

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.shard([param.col_attr.data])
            param.data = torch.empty([], dtype=param.col_attr.data.dtype, device=param.col_attr.data.payload.device)

    def pre_iter(self):
        pass

    def post_iter(self):
        pass
