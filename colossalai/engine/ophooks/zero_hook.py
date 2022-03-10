import torch
from colossalai.registry import OPHOOKS
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.utils import get_current_device
from ._base_ophook import BaseOpHook


@OPHOOKS.register_module
class ZeroHook(BaseOpHook):
    """
    A hook to process sharded param for ZeRO method.
    """

    def __init__(self, shard_strategy: BaseShardStrategy):
        super().__init__()
        self.shard_strategy = shard_strategy
        # NOTE(jiaruifang) Now the computing device of FWD and BWD is always on GPU
        self.computing_device = torch.device(f'cuda:{get_current_device()}')

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.gather([param.col_attr.data])
            if param.col_attr.data.device != self.computing_device:
                param.col_attr.data.to(self.computing_device)
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
            if param.col_attr.data.device != self.computing_device:
                param.col_attr.data.to(self.computing_device)
            param.data = param.col_attr.data.payload
            # Store local accumulated grad shard
            if param.grad is not None:
                if param.col_attr.bwd_count == 0:
                    # We haven't stored local accumulated grad yet
                    assert param.col_attr.grad is None
                    param.col_attr.grad = param.grad.data
                    param.grad = None
                else:
                    # We have stored local accumulated grad
                    # The grad here must be locally computed full grad in this backward pass
                    assert param.grad.shape == param.col_attr.data.origin_shape
            param.col_attr.bwd_count += 1

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters():
            assert hasattr(param, 'col_attr')
            self.shard_strategy.shard([param.col_attr.data])
            param.data = torch.empty([], dtype=param.col_attr.data.dtype, device=param.col_attr.data.payload.device)

    def pre_iter(self):
        pass

    def post_iter(self):
        pass
