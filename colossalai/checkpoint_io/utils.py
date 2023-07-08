# coding=utf-8
import re
from collections import abc as container_abcs
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.tensor.d_tensor import is_distributed_tensor

SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_NAME = "pytorch_model.bin"
STATES_NAME = "pytorch_optim.bin"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
STATES_INDEX_NAME = "pytorch_optim.bin.index.json"
GROUP_FILE_NAME = "pytorch_optim_group.bin"

# ======================================
# General helper functions
# ======================================


def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a parameter in MB. Used to compute whether a group of params exceed the shard size.
    If so, a new shard should be created.

    Args:
        tensor (torch.Tensor): the tensor to calculate size for.

    Returns:
        float: size of the tensor in MB.
    """
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def is_safetensors_available() -> bool:
    """
    Check whether safetensors is available.

    Returns:
        bool: whether safetensors is available.
    """
    try:
        import safetensors
        return True
    except ImportError:
        return False


def is_dtensor_checkpoint(checkpoint_file_path: str) -> bool:
    """
    Check whether the checkpoint file is a dtensor checkpoint.

    Args:
        checkpoint_file_path (str): path to the checkpoint file.

    Returns:
        bool: whether the checkpoint file is a dtensor checkpoint.
    """
    if checkpoint_file_path.endswith('.*.safetensors') or checkpoint_file_path.endswith('.*.bin'):
        return True
    else:
        return False


def is_safetensor_checkpoint(checkpoint_file_path: str) -> bool:
    """
    Check whether the checkpoint file is a safetensor checkpoint.

    Args:
        checkpoint_file_path (str): path to the checkpoint file.

    Returns:
        bool: whether the checkpoint file is a safetensor checkpoint.
    """
    if checkpoint_file_path.endswith('.safetensors'):
        return True
    else:
        return False


# ======================================
# Helper functions for saving shard file
# ======================================
def unwrap_optimizer(optimizer: OptimizerWrapper):
    '''
    Unwrap a wrapped optimizer.
    This method should be used before saving/loading it to/from sharded checkpoints.
    '''

    # TODO(Baizhou): ColossalaiOptimizer will be replaced with OptimizerWrapper in the future
    unwrapped_optim = optimizer.optim
    if isinstance(unwrapped_optim, ColossalaiOptimizer):
        unwrapped_optim = unwrapped_optim.optim
    return unwrapped_optim


def shard_model_checkpoint(state_dict: torch.Tensor, max_shard_size: int = 1024) -> Iterator[Tuple[OrderedDict, int]]:
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.
    """
    current_block = {}
    current_block_size = 0

    for key, weight in state_dict.items():
        ret_block = None
        ret_block_size = 0
        if not is_distributed_tensor(weight):
            weight_size = calculate_tensor_size(weight)

            # If this weight is going to tip up over the maximal size, we split.
            if current_block_size + weight_size > max_shard_size and current_block_size > 0:
                ret_block = current_block
                ret_block_size = current_block_size
                current_block = {}
                current_block_size = 0
            current_block[key] = weight
            current_block_size += weight_size

        if ret_block != None:
            yield ret_block, ret_block_size

    yield current_block, current_block_size


def shard_optimizer_checkpoint(state_dict: dict, max_shard_size: int = 1024) -> Iterator[Tuple[OrderedDict, int]]:
    """
    Splits an optimizer state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.
    """

    # Only split state_dict['state']; state_dict['param_group'] is not considered in this function.
    states = state_dict['state']

    current_block = {}
    current_block_size = 0

    for param_id, state in states.items():

        ret_block = None
        ret_block_size = 0

        # A state might contain more than one tensors.
        # e.g. each Adam state includes: 'step', 'exp_avg', 'exp_avg_sq'
        state_size = 0
        isDTensor = False
        for state_tensor in state.values():

            # When state_tensor is not of Tensor class,
            # e.g., a SGD optimizer with momentum set to 0 can have None as state
            # The calculation of tensor size should be skipped to avoid error.
            if not isinstance(state_tensor, torch.Tensor):
                continue

            # If the states are stored as DTensors, mark isDTensor as true.
            if is_distributed_tensor(state_tensor):
                isDTensor = True
            state_size += calculate_tensor_size(state_tensor)

        if not isDTensor:

            if current_block_size + state_size > max_shard_size and current_block_size > 0:
                ret_block = current_block
                ret_block_size = current_block_size
                current_block = {}
                current_block_size = 0

            current_block[param_id] = state
            current_block_size += state_size

        if ret_block != None:
            yield ret_block, ret_block_size

    yield current_block, current_block_size


def load_shard_state_dict(checkpoint_file: Path, use_safetensors: bool = False):
    """
    load shard state dict into model
    """
    if use_safetensors and not checkpoint_file.suffix == ".safetensors":
        raise Exception("load the model using `safetensors`, but no file endwith .safetensors")
    if use_safetensors:
        from safetensors.torch import load_file as safe_load_file
        from safetensors.torch import safe_open
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata["format"] != "pt":
            raise NotImplementedError(
                f"Conversion from a {metadata['format']} safetensors archive to PyTorch is not implemented yet.")
        return safe_load_file(checkpoint_file)
    else:
        return torch.load(checkpoint_file)


def load_state_dict_into_model(model: nn.Module,
                               state_dict: torch.Tensor,
                               missing_keys: List,
                               strict: bool = False,
                               load_sub_module: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

    unexpected_keys: List[str] = []
    sub_missing_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module: nn.Module, state_dict, prefix="", load_sub_module: bool = True):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, sub_missing_keys, [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)
        if load_sub_module:
            for name, child in module._modules.items():
                if child is not None:
                    load(child, state_dict, prefix + name + ".")

    load(model, state_dict, "", load_sub_module)
    del load

    missing_keys = missing_keys.append(sub_missing_keys)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs = 'Unexpected key(s) in state_dict: {}. '.format(', '.join(
                '"{}"'.format(k) for k in unexpected_keys))
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))


def load_param_groups_into_optimizer(optimizer: Optimizer, param_group_path: str) -> dict:
    """
    Load information of param_groups into an initialized optimizer.
    """

    # Load list of param_groups from given file path.
    # The params in saved_groups are in the form of integer indices.
    saved_groups = torch.load(param_group_path)
    if not isinstance(saved_groups, List):
        raise ValueError(f'The param_groups saved at {param_group_path} is not of List type')

    # The params in param_groups are in the form of pytorch tensors.
    # For more details, please view source code of Optimizer class in pytorch.
    param_groups = optimizer.param_groups

    # Check the compatibility of saved_groups and param_groups.
    if len(param_groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of original parameter groups")
    param_lens = (len(g['params']) for g in param_groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Creating mapping from id to parameters.
    id_map = {
        old_id: p for old_id, p in zip(chain.from_iterable((g['params'] for g in saved_groups
                                                           )), chain.from_iterable((g['params'] for g in param_groups)))
    }

    # Update parameter groups, setting their 'params' value.
    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group

    updated_groups = [update_group(g, ng) for g, ng in zip(param_groups, saved_groups)]

    optimizer.__dict__.update({'param_groups': updated_groups})
    return id_map


def load_states_into_optimizer(optimizer: Optimizer, state_dict: dict, id_map: dict):
    r"""Copies states from `state_dict` into an Optimizer object.

    Args:
        optimizer(Optimizer): An initialized Optimizer object to be loaded
        state_dict(dict): a mapping from tensor index (an integer)
            to its states to be loaded (a mapping from state name to a tensor).
        id_map(dict): a mapping from tensor index (an integer)
            to its corresponding parameter (a tensor) whose states will be updated.
    """

    def cast(param, value, key=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
            if (key != "step"):
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v, key=k) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    new_states = defaultdict(dict)
    for k, v in state_dict.items():
        if k in id_map:
            param = id_map[k]
            new_states[param] = cast(param, v)
        else:
            new_states[k] = v

    optimizer.state.update(new_states)


def sharded_optimizer_loading_epilogue(optimizer: Optimizer):
    r"""Do the cleaning up work after state_dict has been loaded into optimizer

    Args:
        optimizer(Optimizer): An optimizer object whose state has just been loaded.
    """

    # Do the cleaning up as in src code of Pytorch.
    optimizer._hook_for_profile()    # To support multiprocessing pickle/unpickle.
    optimizer.defaults.setdefault('differentiable', False)


# ======================================
# Helper functions for saving state dict
# ======================================


def save_state_dict(state_dict: dict, checkpoint_file_path: str, use_safetensors: bool) -> None:
    """
    Save state dict to checkpoint.

    Args:
        state_dict (dict): state dict.
        checkpoint_file_path (str): path to the checkpoint file.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.
    """
    if use_safetensors:
        assert is_safetensors_available(), "safetensors is not available."
        assert checkpoint_file_path.endswith('.safetensors'), \
            "safetensors only supports .safetensors suffix for checkpoint file."
        from safetensors.torch import save_file as safe_save_file
        safe_save_file(state_dict, checkpoint_file_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict, checkpoint_file_path)


def save_param_groups(state_dict: dict, group_file_path: str) -> None:
    """
    Save information of param_groups to given file path.

    Args:
        state_dict (dict): state dict.
        group_file_path (str): path to the group file.
    """
    param_groups = state_dict["param_groups"]
    torch.save(param_groups, group_file_path)


def save_dtensor(name: str, tensor: torch.Tensor, index_file: "CheckpointIndexFile", use_safetensors: bool) -> None:
    """
    Save distributed tensor to checkpoint. This checkpoint will be a dictionary which contains
    only one tensor.

    Args:
        tensor (Tensor): tensor to be saved.
        index_file (CheckpointIndexFile): path to the checkpoint file.
        size_per_shard (int): size per shard in MB.
    """
    root_path = index_file.root_path
    output_root_path = root_path.joinpath('dtensor')

    # create directory
    output_root_path.mkdir(exist_ok=True)

    # save tensor to this directory
    # TODO(YuliangLiu): get index of the tensor shard
    # e.g. index =
    index = 0

    # save tensor to file
    ckpt_file_name = generate_dtensor_file_name(name, index, use_safetensors)
    ckpt_file_path = output_root_path.joinpath(ckpt_file_name)

    # dtensor ckpt file always contains only one tensor
    state_dict = {name: tensor}
    save_state_dict(state_dict, str(ckpt_file_path), use_safetensors)

    # update the weight map
    # * means all shards
    ckpt_file_name_in_weight_map = 'dtensor/' + generate_dtensor_file_name(name, '*', use_safetensors)
    index_file.append_weight_map(name, ckpt_file_name_in_weight_map)


def get_checkpoint_file_suffix(use_safetensors: bool) -> str:
    """
    Get checkpoint file suffix.

    Args:
        use_safetensors (bool): whether to use safetensors to save the checkpoint.

    Returns:
        str: checkpoint file suffix.
    """
    if use_safetensors:
        return '.safetensors'
    else:
        return '.bin'


def generate_checkpoint_shard_file_name(index: int,
                                        total_number: int,
                                        use_safetensors: bool,
                                        prefix: str = None) -> str:
    """
    Generate checkpoint shard file name.

    Args:
        index (int): index of the shard.
        total_number (int): total number of shards.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.
        prefix (str): prefix of the shard file name. Default: None.

    Returns:
        str: checkpoint shard file name.
    """
    suffix = get_checkpoint_file_suffix(use_safetensors)

    if prefix is None:
        return f"{index:05d}-of-{total_number:05d}.{suffix}"
    else:
        return f"{prefix}-{index:05d}-of-{total_number:05d}.{suffix}"


def generate_dtensor_file_name(param_name: str, index: int, use_safetensors: bool) -> str:
    """
    Generate dtensor file name.

    Args:
        param_name (str): name of the distributed parameter.
        index (int): index of the shard.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.

    Returns:
        str: dtensor file name.
    """
    suffix = get_checkpoint_file_suffix(use_safetensors)
    return f'{param_name}.{index}.{suffix}'


def save_state_dict_as_shard(
    state_dict: dict,
    checkpoint_path: str,
    index: int,
    total_number: int,
    use_safetensors: bool,
    prefix: str = None,
) -> None:
    """
    Save state dict as shard.

    Args:
        state_dict (dict): state dict.
        checkpoint_path (str): path to the checkpoint file.
        index (int): index of the shard.
        total_number (int): total number of shards.
        prefix (str): prefix of the shard file name.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.
    """
    # generate the shard name
    shard_file_name = generate_checkpoint_shard_file_name(index, total_number, use_safetensors, prefix)
    shard_file_path = Path(checkpoint_path).joinpath(shard_file_name).absolute()

    # save the shard
    save_state_dict(state_dict, str(shard_file_path), use_safetensors)


# ========================================
# Helper functions for loading state dict
# ========================================


def has_index_file(checkpoint_path: str) -> Tuple[bool, Optional[Path]]:
    """
    Check whether the checkpoint has an index file.

    Args:
        checkpoint_path (str): path to the checkpoint.

    Returns:
        Tuple[bool, Optional[Path]]: a tuple of (has_index_file, index_file_path)
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        # check if it is .index.json
        reg = re.compile("(.*?).index((\..*)?).json")
        if reg.fullmatch(checkpoint_path.name) is not None:
            return True, checkpoint_path
        else:
            return False, None
    elif checkpoint_path.is_dir():
        # check if there is only one a file ending with .index.json in this directory
        index_files = list(checkpoint_path.glob('*.index.*json'))

        # if we found a .index.json file, make sure there is only one
        if len(index_files) > 0:
            assert len(
                index_files
            ) == 1, f'Expected to find one .index.json file in {checkpoint_path}, but found {len(index_files)}'

        if len(index_files) == 1:
            return True, index_files[0]
        else:
            return False, None
    else:
        raise RuntimeError(f'Invalid checkpoint path {checkpoint_path}. Expected a file or a directory.')


def load_state_dict(checkpoint_file_path: Path):
    """
    Load state dict from checkpoint.

    Args:
        checkpoint_file_path (Path): path to the checkpoint file.

    Returns:
        dict: state dict.
    """

    assert not is_dtensor_checkpoint(checkpoint_file_path), \
        f'Cannot load state dict from dtensor checkpoint {checkpoint_file_path}, you should convert the distributed tensors to gathered tensors with our CLI offline.'

    if is_safetensor_checkpoint(checkpoint_file_path):
        assert is_safetensors_available(), \
            f'Cannot load state dict from safetensor checkpoint {checkpoint_file_path}, because safetensors is not available. Please install safetensors first with pip install safetensors.'
        # load with safetensors
        from safetensors import safe_open
        state_dict = {}
        with safe_open(checkpoint_file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        return state_dict

    else:
        # load with torch
        return torch.load(checkpoint_file_path)


def add_prefix(weights_name: str, prefix: Optional[str] = None) -> str:
    if prefix is not None and len(prefix) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [prefix] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


def get_model_base_filenames(prefix: str = None, use_safetensors: bool = False):
    """
    generate base model weight filenames
    """
    weights_name = SAFE_WEIGHTS_NAME if use_safetensors else WEIGHTS_NAME
    weights_name = add_prefix(weights_name, prefix)

    save_index_file = SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME
    save_index_file = add_prefix(save_index_file, prefix)

    return weights_name, save_index_file


def get_optimizer_base_filenames(prefix: str = None):
    """
    generate base optimizer state filenames
    """
    states_name = STATES_NAME
    states_name = add_prefix(states_name, prefix)

    save_index_file = STATES_INDEX_NAME
    save_index_file = add_prefix(save_index_file, prefix)

    param_group_file = GROUP_FILE_NAME
    param_group_file = add_prefix(param_group_file, prefix)

    return states_name, save_index_file, param_group_file


def get_shard_filename(weights_name: str, idx: int):
    """
    get shard file name
    """
    shard_file = weights_name.replace(".bin", f"-{idx+1:05d}.bin")
    shard_file = shard_file.replace(".safetensors", f"-{idx + 1:05d}.safetensors")
    return shard_file
