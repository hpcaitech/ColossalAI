import torch

from colossalai.elixir.ctx import tensor_creation_methods


class MetaContext(object):
    """A context manager that wraps all tensor creation methods in torch.
    By default, all tensors will be created in meta.

    args:
        device_type: The device type of the tensors to be created.
    """

    def __init__(self, device_type: str = 'meta') -> None:
        super().__init__()
        self.device_type = device_type
        return None

    def __enter__(self):

        def meta_wrap(func):

            def wrapped_func(*args, **kwargs):
                kwargs['device'] = self.device_type
                return func(*args, **kwargs)

            return wrapped_func

        for name, method in tensor_creation_methods.items():
            setattr(torch, name, meta_wrap(method))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, method in tensor_creation_methods.items():
            setattr(torch, name, method)
