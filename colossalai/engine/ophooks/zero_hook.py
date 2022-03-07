import torch
from . import BaseOpHook
from colossalai.registry import OPHOOKS
from colossalai.zero.shard_utils import TensorShardStrategy


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
            assert hasattr(param, 'ca_attr')
            self.shard_strategy.gather([param.ca_attr.data])
            param.data = param.ca_attr.data

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            self.shard_strategy.shard([param.ca_attr.data])
            param.data = None

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            self.shard_strategy.gather([param.ca_attr.data])
            param.data = param.ca_attr.data

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            self.shard_strategy.shard([param.ca_attr.data])
            param.data = None

            # save param.grad to some place  in case gradient accumulation.
            # at the moment, the ca_attr.grad payload is useless.
            # gather the memory first
            self.shard_strategy.gather([param.ca_attr.grad])
            param.grad = None

    def pre_iter(self):
        pass

    def post_iter(self):
        pass
