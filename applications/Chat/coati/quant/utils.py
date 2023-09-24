from contextlib import contextmanager

import torch


def _noop(*args, **kwargs):
    pass


@contextmanager
def low_resource_init():
    """This context manager disables weight initialization and sets the default float dtype to half."""
    old_kaiming_uniform_ = torch.nn.init.kaiming_uniform_
    old_uniform_ = torch.nn.init.uniform_
    old_normal_ = torch.nn.init.normal_
    dtype = torch.get_default_dtype()
    try:
        torch.nn.init.kaiming_uniform_ = _noop
        torch.nn.init.uniform_ = _noop
        torch.nn.init.normal_ = _noop
        torch.set_default_dtype(torch.half)
        yield
    finally:
        torch.nn.init.kaiming_uniform_ = old_kaiming_uniform_
        torch.nn.init.uniform_ = old_uniform_
        torch.nn.init.normal_ = old_normal_
        torch.set_default_dtype(dtype)
