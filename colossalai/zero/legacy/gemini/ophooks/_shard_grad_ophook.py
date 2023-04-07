import torch

from colossalai.registry import OPHOOKS

from . import BaseOpHook


@OPHOOKS.register_module
class ShardGradMemTracerHook(BaseOpHook):
    """
    A hook to process sharded param before and afther FWD and BWD operator executing.
    """

    def __init__(self):
        super().__init__()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        for param in module.parameters():
            assert hasattr(param, '_sharded_grad')
            param._sharded_grad.setup()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        pass

    def post_iter(self):
        pass
