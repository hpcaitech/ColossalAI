import torch
import torch.distributed as dist
from colossalai.registry import OPHOOKS

from . import BaseOpHook


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
            if dist.get_rank() == 0:
                print(f'{param._name} pre fwd shape {param.ca_attr.payload("cpu").shape}')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.shard()
            if dist.get_rank() == 0:
                print(f'{param._name} post fwd shape {param.ca_attr.payload("cpu").shape}')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.gather()
            if dist.get_rank() == 0:
                print(f'{param._name} pre bwd shape {param.ca_attr.payload("cpu").shape}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters():
            assert hasattr(param, 'ca_attr')
            param.ca_attr.shard()
            if dist.get_rank() == 0:
                print(f'{param._name} post bwd shape {param.ca_attr.payload("cpu").shape}')

    def pre_iter(self):
        pass

    def post_iter(self):
        pass
