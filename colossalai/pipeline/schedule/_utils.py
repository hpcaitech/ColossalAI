from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import SUPPORTED_NODES, TreeSpec, _register_pytree_node, tree_flatten, tree_map, tree_unflatten


# this register are for torch under version 1.13.1, maybe removed in the future
def _odict_flatten(d: "OrderedDict[Any, Any]") -> Tuple[List[Any], Any]:
    return list(d.values()), list(d.keys())


def _odict_unflatten(values: List[Any], context: Any) -> "OrderedDict[Any, Any]":
    return OrderedDict((key, value) for key, value in zip(context, values))


_register_pytree_node(OrderedDict, _odict_flatten, _odict_unflatten)


def tree_map_hf(fn: Any, pytree: Any):
    flat_args, spec = tree_flatten_hf(pytree)
    return tree_unflatten([fn(i) for i in flat_args], spec)


# use this flatten function to handle the ModelingOutput Class instance.
def tree_flatten_hf(pytree: Any) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values an a TreeSpec that can be used
    to reconstruct the pytree.
    """
    if isinstance(pytree, OrderedDict):
        node_type = OrderedDict
        flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
        child_pytrees, context = flatten_fn(pytree)

        # Recursively flatten the children
        result: List[Any] = []
        children_specs: List["TreeSpec"] = []
        for child in child_pytrees:
            flat, child_spec = tree_flatten_hf(child)
            result += flat
            children_specs.append(child_spec)
        return result, TreeSpec(node_type, context, children_specs)
    else:
        result, tree_spec = tree_flatten(pytree)
        return result, tree_spec


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
    raise RuntimeError("No tensor found in the batch")


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
            return x[start : start + micro_batch_size]
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
    if isinstance(x, torch.Tensor) and x.requires_grad:
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


def merge_batch(data: List[Any], batch_size_dim=0) -> Any:
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
        # elems should be an instance of OrderedDict
        elems, tree_spec = tree_flatten_hf(d)
        flattened_data.append(elems)
    merged_data = []

    for elem_batch in zip(*flattened_data):
        if isinstance(elem_batch[0], torch.Tensor):
            if len(elem_batch[0].shape) == 0:  # set loss to None in pipeline outputs
                merged_data.append(None)
            else:
                merged_data.append(torch.cat(elem_batch, dim=batch_size_dim))
        else:
            merged_data.append(list(elem_batch))
    return tree_unflatten(merged_data, tree_spec)
