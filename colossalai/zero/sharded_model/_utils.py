from typing import Any, Callable, List, Tuple

import torch
import torch.nn.functional as F
from typing import Union
from colossalai.gemini.stateful_tensor import StatefulTensor


def get_gradient_predivide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


@torch.no_grad()
def alloc_storage(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():    # no need to reallocate
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())


def cast_tensor_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, StatefulTensor):
        tensor = tensor.payload
    if torch.is_floating_point(tensor) and tensor.dtype is torch.float32:
        return tensor.half()
    return tensor


def cast_tensor_to_fp32(tensor: Union[torch.Tensor, StatefulTensor]) -> torch.Tensor:
    if isinstance(tensor, StatefulTensor):
        tensor = tensor.payload

    if torch.is_floating_point(tensor) and tensor.dtype is torch.float16:
        return tensor.float()
    return tensor


def apply_to_tensors(x: Any, fn: Callable):
    if torch.is_tensor(x):
        return fn(x)
    elif isinstance(x, list):
        return [apply_to_tensors(t, fn) for t in x]
    elif isinstance(x, tuple):
        return tuple(apply_to_tensors(t, fn) for t in x)
    elif isinstance(x, dict):
        return {key: apply_to_tensors(val, fn) for key, val in x.items()}
    else:
        return x


def cast_float_arguments(fn: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    return apply_to_tensors(args, fn), apply_to_tensors(kwargs, fn)


def chunk_and_pad(tensor: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    # torch.chunk may return fewer than num_chunks chunks, pad accordingly.
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks
