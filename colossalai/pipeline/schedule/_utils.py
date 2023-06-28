from typing import Any, List, Optional

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def to_device(x: Any, device: Optional[torch.device] = None) -> Any:
    """Move object to device if it is a tensor.

    Args:
        x (Any): Object to be moved.
        device (Optional[torch.device], optional): Target device. Defaults to None.

    Returns:
        Any: Moved object.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def get_batch_size(batch: Any) -> int:
    """Get the batch size (size of dimension-0) of the first tensor in the batch.

    Args:
        batch (Any): Batch to be inspected.

    Raises:
        RuntimeError: If no tensor is found in the batch.

    Returns:
        int: Batch size.
    """
    data_list, _ = tree_flatten(batch)
    for data in data_list:
        if isinstance(data, torch.Tensor):
            return data.size(0)
    raise RuntimeError('No tensor found in the batch')


def get_micro_batch(batch: Any, start: int, micro_batch_size: int) -> Any:
    """Get a micro batch of the original batch.

    Args:
        batch (Any): Batch to be sliced.
        start (int): Start index of the micro batch.
        micro_batch_size (int): Size of the micro batch.

    Returns:
        Any: Target micro batch.
    """

    def _get_tensor_slice(x: Any):
        if isinstance(x, torch.Tensor):
            return x[start:start + micro_batch_size]
        return x

    return tree_map(_get_tensor_slice, batch)


def model_forward(model: Module, data: Any, internal_inputs: Optional[dict]) -> Any:
    """Call model forward function with data and internal inputs.

    Args:
        model (Module): Model to be called.
        data (Any): Data loaded from data iterator.
        internal_inputs (Optional[dict]): Data from previous stage. It must be a dict or None if it's the first stage.

    Returns:
        Any: Outputs of the model.
    """
    if internal_inputs is None:
        internal_inputs = {}
    if isinstance(data, (list, tuple)):
        return model(*data, **internal_inputs)
    elif isinstance(data, dict):
        return model(**data, **internal_inputs)
    return model(data, **internal_inputs)


def retain_grad(x: Any) -> None:
    """Call retain_grad() on a tensor.

    Args:
        x (Any): Object to be called.
    """
    if isinstance(x, torch.Tensor):
        x.retain_grad()


def detach(x: Any) -> Any:
    """Call detach() on a tensor.

    Args:
        x (Any): Object to be called.

    Returns:
        Any: The detached object.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def merge_batch(data: List[Any]) -> Any:
    """Merge micro batches into a batch.

    Args:
        data (List[Any]): A list of micro batches.

    Returns:
        Any: Merge batch.
    """
    if len(data) == 0:
        return
    flattened_data = []
    tree_spec = None
    for d in data:
        elems, tree_spec = tree_flatten(d)
        flattened_data.append(elems)
    merged_data = []
    for elem_batch in zip(*flattened_data):
        if isinstance(elem_batch[0], torch.Tensor):
            merged_data.append(torch.cat(elem_batch, dim=0))
        else:
            merged_data.append(list(elem_batch))
    return tree_unflatten(merged_data, tree_spec)
