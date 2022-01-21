import torch
from colossalai.registry import OPHOOKS
from colossalai.trainer.ophooks import BaseOpHook


@OPHOOKS.register_module
class TestOpHook(BaseOpHook):
    r"""
    A simple OpHook. Print the module name before its execution.
    """
    def __init__(self):
        super().__init__()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            print(f"FWD pre {module.__class__.__name__}")

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            print(f"FWD post {module.__class__.__name__}")

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            print(f"BWD pre {module.__class__.__name__}")

    def post_bwd_exec(self, module: torch.nn.Module, input):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            print(f"BWD post {module.__class__.__name__}")
