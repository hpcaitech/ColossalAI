import torch

TESNOR_CREATION_METHODS = dict(tensor=torch.tensor,
                               sparse_coo_tensor=torch.sparse_coo_tensor,
                               asarray=torch.asarray,
                               as_tensor=torch.as_tensor,
                               as_strided=torch.as_strided,
                               from_numpy=torch.from_numpy,
                               from_dlpack=torch.from_dlpack,
                               frombuffer=torch.frombuffer,
                               zeros=torch.zeros,
                               zeros_like=torch.zeros_like,
                               ones=torch.ones,
                               ones_like=torch.ones_like,
                               arange=torch.arange,
                               range=torch.range,
                               linspace=torch.linspace,
                               logspace=torch.logspace,
                               eye=torch.eye,
                               empty=torch.empty,
                               empty_like=torch.empty_like,
                               empty_strided=torch.empty_strided,
                               full=torch.full,
                               full_like=torch.full_like,
                               quantize_per_tensor=torch.quantize_per_tensor,
                               quantize_per_channel=torch.quantize_per_channel,
                               dequantize=torch.dequantize,
                               complex=torch.complex,
                               polar=torch.polar,
                               heaviside=torch.heaviside)


# TODO: unify this with lazy init context
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

        for name, method in TESNOR_CREATION_METHODS.items():
            setattr(torch, name, meta_wrap(method))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, method in TESNOR_CREATION_METHODS.items():
            setattr(torch, name, method)
