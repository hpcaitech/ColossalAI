# coding=utf-8
from pathlib import Path
import torch
import torch.nn as nn
from typing import List, Dict, Mapping, OrderedDict, Optional, Tuple
from colossalai.tensor.d_tensor.d_tensor import DTensor
import re

SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

# ======================================
# General helper functions
# ======================================

def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a parameter in MB. Used to compute whether a group of params exceed the shard size.
    If so, a new shard should be created.

    Args:
        tenosr (torch.Tensor): the tensor to calculate size for.

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
def shard_checkpoint(state_dict: torch.Tensor, max_shard_size: int = 1024, weights_name: str = WEIGHTS_NAME):
 
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.
    """
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    for key, weight in state_dict.items():
        if type(weight) != DTensor:
            weight_size = calculate_tensor_size(weight)

            # If this weight is going to tip up over the maximal size, we split.
            if current_block_size + weight_size > max_shard_size:
                sharded_state_dicts.append(current_block)
                current_block = {}
                current_block_size = 0

            current_block[key] = weight
            current_block_size += weight_size
            total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None
    
    # Otherwise, let's build the index
    weight_map = {}
    shards = {}

    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index

def load_shard_state_dict(checkpoint_file: Path, use_safetensors: bool =False):
    """
    load shard state dict into model
    """
    if use_safetensors and not checkpoint_file.suffix == ".safetensors":
        raise Exception("load the model using `safetensors`, but no file endwith .safetensors")
    if use_safetensors:
        from safetensors.torch import safe_open
        from safetensors.torch import load_file as safe_load_file
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata["format"] != "pt":
            raise NotImplementedError(
                f"Conversion from a {metadata['format']} safetensors archive to PyTorch is not implemented yet."
            )
        return safe_load_file(checkpoint_file)
    else:
        return torch.load(checkpoint_file)
    
def load_state_dict_into_model(model: nn.Module, state_dict: torch.Tensor, missing_keys: List, strict: bool = False):
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

    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model, state_dict, "")
    del load

    # deal with missing key
    if len(missing_keys) > 0:
        deleted_keys = []
        for key in missing_keys:
            if key not in sub_missing_keys:
                deleted_keys.append(key)
        for key in deleted_keys:
            missing_keys.remove(key)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs = 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys))
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        
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
    


def add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None and len(variant) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name
