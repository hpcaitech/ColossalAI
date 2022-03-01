import torch
from . import BaseOpHook
from colossalai.registry import OPHOOKS

@OPHOOKS.register_module
class ShardParamHook(BaseOpHook):
    """
    A hook to process sharded param before and afther FWD and BWD operator executing.
    """
    def __init__(self):
        super().__init__()

    def niter(self):
        return self._niter

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.gather()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.shard()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.gather()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.shard()

    def pre_iter(self):
        pass

    def post_iter(self):
        pass

