import torch


def _hijack_detach_and_clone(ptensor: torch.Tensor) -> torch.Tensor:
    """
    Hijack the detach and clone methods of the tensor to make sure the dist_layout is copied.

    Args:
        tensor (torch.Tensor): The tensor to be hijacked.

    Returns:
        torch.Tensor: The hijacked tensor.
    """
    ptensor._unpad_detach = ptensor.detach
    ptensor._unpad_clone = ptensor.clone

    def new_detach(self):
        t_ = self._unpad_detach()
        t_._padding_dim = self._padding_dim
        t_._origin_length = self._origin_length
        t_._current_length = self._current_length
        return t_

    def new_clone(self, *args, **kwargs):
        t_ = self._unpad_clone(*args, **kwargs)
        t_._padding_dim = self._padding_dim
        t_._origin_length = self._origin_length
        t_._current_length = self._current_length
        return t_

    # bind the new methods to the tensor
    ptensor.detach = new_detach.__get__(ptensor)
    ptensor.clone = new_clone.__get__(ptensor)
    return ptensor


def _hijack_back_detach_and_clone(ptensor: torch.Tensor) -> torch.Tensor:
    """
    Hijack the detach and clone methods of the tensor to make sure the dist_layout is copied.

    Args:
        tensor (torch.Tensor): The tensor to be hijacked.

    Returns:
        torch.Tensor: The hijacked tensor.
    """
    ptensor.detach = ptensor._unpad_detach
    ptensor.clone = ptensor._unpad_clone

    delattr(ptensor, "_unpad_detach")
    delattr(ptensor, "_unpad_clone")

    return ptensor


def is_padded_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a padding tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a padding tensor.
    """
    return hasattr(tensor, "_padding_dim")


def to_padded_tensor(
    tensor: torch.Tensor,
    current_length: int,
    padding_dim: int,
) -> torch.Tensor:
    assert (
        padding_dim < tensor.dim()
    ), f"Please passing a valid padding_dim. the dimension of the tensor is {tensor.dim()}"

    if is_padded_tensor(tensor):
        return tensor

    origin_length = tensor.shape[padding_dim]
    padding_num = current_length - origin_length
    padding_data = torch.zeros(
        *tensor.shape[:padding_dim],
        padding_num,
        *tensor.shape[padding_dim + 1 :],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    tensor.data = torch.cat((tensor.data, padding_data), dim=padding_dim).contiguous()

    tensor._padding_dim = padding_dim
    tensor._origin_length = origin_length
    tensor._current_length = current_length

    _hijack_detach_and_clone(tensor)

    return tensor


def to_unpadded_tensor(ptensor: torch.Tensor):
    if not is_padded_tensor(ptensor):
        return ptensor

    unpad_slices = [slice(None)] * ptensor.dim()
    unpad_slices[ptensor._padding_dim] = slice(None, ptensor._origin_length)
    ptensor.data = ptensor.data[tuple(unpad_slices)]

    delattr(ptensor, "_padding_dim")
    delattr(ptensor, "_origin_length")
    delattr(ptensor, "_current_length")

    _hijack_back_detach_and_clone(ptensor)

    return ptensor


def init_as_padded_tensor(tensor: torch.Tensor, current_length: int, origin_length: int, padding_dim: int):
    if is_padded_tensor(tensor):
        return tensor

    tensor._padding_dim = padding_dim
    tensor._origin_length = origin_length
    tensor._current_length = current_length

    _hijack_detach_and_clone(tensor)

    return tensor
