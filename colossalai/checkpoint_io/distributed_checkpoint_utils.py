import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, OrderedDict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import _get_default_group

from colossalai.interface import ModelWrapper
from colossalai.utils import get_non_persistent_buffers_set

from .index_file import CheckpointIndexFile
from .utils import (
    StateDictSharder,
    async_save_state_dict_shards,
    create_pinned_state_dict,
    get_model_base_filenames,
    load_state_dict,
    save_state_dict,
    save_state_dict_shards,
    search_tp_partition_dim,
)

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"

MODEL_META_PREFIX = "pytorch_model-meta-dist-"
MODEL_WEIGHT_PREFIX = "pytorch_model-dist-"
SHARD_META_SUFFIX = ".index.json"


def dist_model_state_dict(model: nn.Module, prefix: str = "", keep_vars: bool = False):
    destination = dict()
    # Save parameters.
    for name, param in model.named_parameters():
        if param is None:
            continue
        destination[prefix + name] = param
    # Save buffers.
    non_persist_buffers_set = get_non_persistent_buffers_set(model)
    for name, buf in model.named_buffers():
        if buf is not None and name not in non_persist_buffers_set:
            buffer = buf if keep_vars else buf.detach()
            destination[prefix + name] = buffer

    # Save extra states.
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if (
        getattr(model.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
        is not torch.nn.Module.get_extra_state
    ):
        extra_state = model.get_extra_state()
        destination[extra_state_key] = extra_state
    return destination


def load_state_dict_into_dist_model(
    model: nn.Module, state_dict: Dict, prefix: str = "", keep_vars: bool = False, strict: bool = False
):
    destination = dict()
    # Save parameters.
    for name, param in model.named_parameters():
        if param is None:
            continue
        with torch.no_grad():
            param.copy_(state_dict[prefix + name])
    # Save buffers.
    non_persist_buffers_set = get_non_persistent_buffers_set(model)
    for name, buf in model.named_buffers():
        if buf is not None and name not in non_persist_buffers_set:
            with torch.no_grad():
                buf.copy_(state_dict[prefix + name])

    # Save extra states.
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if (
        getattr(model.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
        is not torch.nn.Module.get_extra_state
    ):
        extra_state = model.get_extra_state()
        with torch.no_grad():
            extra_state.copy_(state_dict[extra_state_key])
    return destination


def create_model_metadata(
    model: nn.Module,
    prefix: str = "",
    tp_size=None,
    tp_rank=None,
):
    param_origin_shape = model.param_origin_shape
    model = model.unwrap()
    model_metadata = {}
    for name, param in model.named_parameters():
        if param is None:
            continue
        model_metadata[prefix + name] = {}
        original_shape = param_origin_shape[name]
        tp_partition_dim = search_tp_partition_dim(
            current_shape=param.shape, original_shape=original_shape, tp_size=tp_size
        )
        model_metadata[prefix + name]["offsets"] = torch.zeros(len(original_shape), dtype=torch.int)
        model_metadata[prefix + name]["lengths"] = list(param.shape)
        model_metadata[prefix + name]["global_shape"] = list(original_shape)
        if tp_partition_dim is not None:
            partition_size = param.shape[tp_partition_dim]
            model_metadata[prefix + name]["offsets"][tp_partition_dim] = partition_size * tp_rank
            if tp_rank == tp_size - 1:
                model_metadata[prefix + name]["lengths"][tp_partition_dim] = original_shape[tp_partition_dim] - (
                    partition_size * (tp_size - 1)
                )
    return model_metadata


def save_metadata(model_metadata, metadata_file, checkpoint_file=None, total_size=None):
    metadata_dicts = {
        "checkpoint_version": "1.0",
        "total_size": total_size,
        "metadata": {},
    }
    for name, data in model_metadata.items():
        metadata_dicts["metadata"][name] = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v = v.tolist()
            metadata_dicts["metadata"][name][k] = v
        if checkpoint_file is not None:
            metadata_dicts["metadata"][name]["file"] = checkpoint_file
        metadata_dicts["metadata"][name]["rank"] = dist.get_rank(_get_default_group())
    with open(metadata_file, "w") as json_file:
        json.dump(metadata_dicts, json_file, indent=4)


def load_metadata(checkpoint: str):
    metadata_dict = {}
    for filename in os.listdir(checkpoint):
        if filename.startswith(MODEL_META_PREFIX) and filename.endswith(".json"):
            file_path = os.path.join(checkpoint, filename)
            try:
                with open(file_path, "r") as f:
                    metadata_json = json.load(f)
                    for name, item in metadata_json["metadata"].items():
                        if name not in metadata_dict:
                            metadata_dict[name] = {}
                            metadata_dict[name]["global_shape"] = item["global_shape"]
                            metadata_dict[name]["shards"] = {}
                        else:
                            assert metadata_dict[name]["global_shape"] == item["global_shape"]
                        shard = {item["rank"]: {}}
                        for k, v in item.items():
                            if k == "rank":
                                continue
                            shard[item["rank"]][k] = v
                        metadata_dict[name]["shards"].update(shard)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Unable to load file {file_path}: {e}")
    return metadata_dict


def find_covering_shards(shards, target_offsets, target_lengths):
    """
    Parameters:

    shards: A list containing information about all shards.
    target_offsets: A one-dimensional array representing the starting position of the target tensor in each dimension.
    target_lengths: A one-dimensional array representing the lengths of the target tensor in each dimension.
    Returns:

    A list of all shards that cover the target range.
    """
    target_start = target_offsets
    target_end = [start + length for start, length in zip(target_offsets, target_lengths)]

    covering_shards = {}

    global_shape = None
    total_lengths = None
    for rank, shard in shards.items():
        shard_start = shard["offsets"]
        shard_lengths = shard["lengths"]
        if global_shape == None:
            global_shape = shard["global_shape"]
            total_lengths = [0] * len(global_shape)
        shard_end = [start + length for start, length in zip(shard_start, shard_lengths)]

        overlap = any(
            not (target_end[dim] <= shard_start[dim] or target_start[dim] >= shard_end[dim])
            for dim in range(len(target_start))
        )
        if overlap:
            covering_shards.update({rank: shard})
        for dim in range(len(shard_start)):
            total_lengths[dim] = max(total_lengths[dim], shard_start[dim] + shard_lengths[dim])

    assert total_lengths == global_shape
    return covering_shards


def extract_weight_from_shard_partial(shard, target_offsets, target_lengths):
    """
    Extract the target range of weights from shard data, supporting partial overlap.

    param shard: A dictionary containing shard data, including 'offsets', 'lengths', and 'weight'.
    param target_offsets: A 1D array indicating the starting position of the target tensor in each dimension.
    param target_lengths: A 1D array indicating the length of the target tensor in each dimension.
    return: The extracted sub-tensor of the target weights and its position within the target range.
    """
    shard_offsets = shard["offsets"]
    shard_lengths = shard["lengths"]
    weight = shard["weight"]

    slices = []
    target_slices = []

    for dim, (t_offset, t_length, s_offset, s_length) in enumerate(
        zip(target_offsets, target_lengths, shard_offsets, shard_lengths)
    ):
        intersection_start = max(t_offset, s_offset)
        intersection_end = min(t_offset + t_length, s_offset + s_length)

        if intersection_start >= intersection_end:
            return None, None

        shard_slice_start = intersection_start - s_offset
        shard_slice_end = intersection_end - s_offset
        slices.append(slice(shard_slice_start, shard_slice_end))

        target_slice_start = intersection_start - t_offset
        target_slice_end = intersection_end - t_offset
        target_slices.append(slice(target_slice_start, target_slice_end))

    target_weight = weight[tuple(slices)]
    return target_weight, target_slices


def assemble_tensor_from_shards_partial(shards, target_offsets, target_lengths, dtype):
    target_tensor = torch.zeros(target_lengths, dtype=dtype)

    for rank, shard in shards.items():
        target_weight, target_slices = extract_weight_from_shard_partial(shard, target_offsets, target_lengths)

        if target_weight is not None and target_slices is not None:
            target_tensor[tuple(target_slices)] = target_weight

    return target_tensor


def is_pytorch_model_meta_dist_file(checkpoint_index_file):
    if MODEL_META_PREFIX in str(checkpoint_index_file):
        return True
    return False


def dist_model_sharder(
    model: nn.Module,
    prefix: str = "",
    keep_vars: bool = False,
    size_per_shard: int = 1024,
    pinned_state_dicts: Optional[Dict[str, torch.Tensor]] = None,
) -> Iterator[Tuple[OrderedDict, int]]:
    # An internel method that breaks state_dict of model into shards within limited size.

    state_dict_sharder = StateDictSharder(size_per_shard)

    # Save parameters.
    for name, param in model.named_parameters():
        if param is None:
            continue
        if pinned_state_dicts is not None:
            if (prefix + name) not in pinned_state_dicts:
                pinned_state_dicts[prefix + name] = torch.empty_like(param, pin_memory=True, device="cpu")
            pinned_state_dicts[prefix + name].copy_(param)
            param = pinned_state_dicts[prefix + name]
        block, block_size = state_dict_sharder.append_param(prefix + name, param)
        if block is not None:
            yield block, block_size

    # Save buffers.
    non_persist_buffers_set = get_non_persistent_buffers_set(model)
    for name, buf in model.named_buffers():
        if buf is not None and name not in non_persist_buffers_set:
            buffer = buf if keep_vars else buf.detach()
            if pinned_state_dicts is not None:
                if (prefix + name) not in pinned_state_dicts:
                    pinned_state_dicts[prefix + name] = torch.empty_like(buffer, pin_memory=True, device="cpu")
                pinned_state_dicts[prefix + name].copy_(buffer)
                buffer = pinned_state_dicts[prefix + name]
            block, block_size = state_dict_sharder.append_param(prefix + name, buffer)
            if block is not None:
                yield block, block_size

    # Save extra states.
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if (
        getattr(model.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
        is not torch.nn.Module.get_extra_state
    ):
        extra_state = model.get_extra_state()
        if pinned_state_dicts is not None:
            if extra_state_key not in pinned_state_dicts:
                pinned_state_dicts[extra_state_key] = torch.empty_like(extra_state, pin_memory=True, device="cpu")
            pinned_state_dicts[extra_state_key].copy_(extra_state)
            extra_state = pinned_state_dicts[extra_state_key]
        block, block_size = state_dict_sharder.append_param(extra_state_key, extra_state)
        if block is not None:
            yield block, block_size

    # Return the last block in sharder.
    yield state_dict_sharder.current_block, state_dict_sharder.current_block_size


def save_dist_unshard_model(
    model: ModelWrapper,
    model_metadata: Dict,
    checkpoint: str,
    use_safetensors: bool,
    use_async: bool = False,
    dist_id=0,
    pinned_state_dicts=None,
):
    """
    Save model state dict to a single file with given checkpointing path.

    Args:
        model (nn.Module): Model on local device to be saved.
        checkpoint (str): Checkpointing path which should be a file path. Can be absolute or relative path.
        gather_dtensor (bool, optional): Whether to gather dtensor, currently not used. Defaults to True.
        use_safetensors (bool, optional): Whether to use safe tensors. Defaults to False.
        use_async (bool, optional): Whether to save the state_dicts of model asynchronously. Defaults to False.
    """

    model = model.unwrap()

    # The logic of collecting parameter shards along tp degree
    # has been implemented by _save_to_state_dict method of ParallelModule in Shardformer.
    state_dict = dist_model_state_dict(model)

    Path(checkpoint).mkdir(parents=True, exist_ok=True)
    file_name = f"{MODEL_WEIGHT_PREFIX}{dist_id:05d}.bin"
    if use_async:
        file_name = file_name.replace(".bin", ".safetensors")
    checkpoint_file = os.path.join(checkpoint, file_name)
    metadata_file = os.path.join(checkpoint, f"{MODEL_META_PREFIX}{dist_id:05d}.json")
    save_metadata(model_metadata, metadata_file, file_name)

    if use_async:
        from colossalai.utils.safetensors import save

        if id(model) not in pinned_state_dicts:
            pinned_state_dicts[id(model)] = create_pinned_state_dict(state_dict)
        for name, param in state_dict.items():
            pinned_state_dicts[id(model)][name].copy_(param)
            state_dict[name] = pinned_state_dicts[id(model)][name]
        writer = save(path=checkpoint_file, state_dict=state_dict)
        return writer
    else:
        save_state_dict(state_dict, checkpoint_file, use_safetensors)
        return None


def load_dist_model(
    model: ModelWrapper,
    model_metadata: Dict,
    checkpoint: str,
    low_cpu_mem_mode: bool = True,
    num_threads: int = 1,
):
    """
    Load model from a single file with the given path of checkpoint.

    Args:
        model (nn.Module): The model to be loaded.
        checkpoint_index_file (str): Path to the checkpoint file.
        strict (bool, optional): For name matching during loading state_dict. Defaults to False.
                                    This argument should be manually set to False since not all params in checkpoint are needed for each device when pipeline is enabled.
    """

    model_before_wrapping = model
    model = model.unwrap()

    metadata_loaded = load_metadata(checkpoint)

    load_files = {}
    covered_shards = {}
    for key, item in model_metadata.items():
        offsets = item["offsets"]
        lengths = item["lengths"]
        assert (
            item["global_shape"] == metadata_loaded[key]["global_shape"]
        ), f"{item['global_shape']}, {metadata_loaded[key]['global_shape']}"
        shards = metadata_loaded[key]["shards"]
        covering_shards = find_covering_shards(shards=shards, target_offsets=offsets, target_lengths=lengths)
        covered_shards[key] = covering_shards
        for rank, shard in covering_shards.items():
            if rank not in load_files:
                load_files[rank] = set()
            load_files[rank].add(shard["file"])

    dtype = None
    for rank, files in load_files.items():
        for file in files:
            file_path = os.path.join(checkpoint, file)
            state_dict_shard = load_state_dict(file_path)
            for key, weight in state_dict_shard.items():
                if key not in covered_shards:
                    continue
                if dtype == None:
                    dtype = weight.dtype
                covered_shards[key][rank]["weight"] = weight
    state_dict = {}
    for key, shards in covered_shards.items():
        state = assemble_tensor_from_shards_partial(
            shards, model_metadata[key]["offsets"], model_metadata[key]["lengths"], dtype=dtype
        )
        state_dict[key] = state

    if not low_cpu_mem_mode:
        state_dict = create_pinned_state_dict(state_dict, empty=False, num_threads=num_threads)

    load_state_dict_into_dist_model(model=model, state_dict=state_dict)

    # Update master params if mixed-precision training is enabled.
    model_before_wrapping.update_master_params()


def save_dist_sharded_model(
    model: ModelWrapper,
    model_metadata: Dict,
    checkpoint: str,
    prefix: Optional[str] = None,
    size_per_shard: int = 1024,
    use_safetensors: bool = False,
    use_async: bool = False,
    dist_id: int = 0,
    pinned_state_dicts=None,
) -> None:
    """
    Save sharded model checkpoint under the given checkpointing path.
    The following files will be created under the path:
    - An index file (pytorch_model.bin.index.json) containing a map between model params/buffers and file names.
    - Multiple files that store state tensors of models.
        If pipeline parallelism is used, the filenames are in the form of "pytorch_model.<prefix>-stage-000XX-shard-000XX.bin".
        If pipeline parallelism is not used, "pytorch_model.<prefix>-000XX.bin"


    Args:
        model (nn.Module): Model on local device to be saved.
        checkpoint (str): Checkpointing path which should be a directory path.
        gather_dtensor (bool, optional): Whether to gather_dtensor, currently not used. Defaults to True.
        prefix (str, optional): Perfix of file to save. Defaults to None.
        size_per_shard (int, optional): Size per shard in MB. Defaults to 1024.
        use_safetensors (bool, optional): Whether to use safe tensors. Defaults to False.
        use_async (bool, optional): Whether to save the state_dicts of model asynchronously. Defaults to False.
    """

    model = model.unwrap()

    if os.path.isfile(checkpoint):
        logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
        return

    Path(checkpoint).mkdir(parents=True, exist_ok=True)
    # Devices along the same dp_group share the same copies of model.
    # So only let the device with dp_rank == 0 and sp_rank == 0 save the model.

    if use_async:
        if id(model) not in pinned_state_dicts:
            pinned_state_dicts[id(model)] = {}
        pinned_state_dicts = pinned_state_dicts[id(model)]
    else:
        pinned_state_dicts = None
    state_dict_shard = dist_model_sharder(model, size_per_shard=size_per_shard, pinned_state_dicts=pinned_state_dicts)
    weights_name, _ = get_model_base_filenames(prefix, use_safetensors)
    index_file = CheckpointIndexFile(checkpoint)

    # Manage filenames of sharded weights and index file for each pipeline stage.
    weights_name = weights_name.replace(".bin", f"-dist-{dist_id:05d}-shard.bin")
    weights_name = weights_name.replace(".safetensors", f"-dist-{dist_id:05d}-shard.safetensors")
    metadata_file = os.path.join(checkpoint, f"{MODEL_META_PREFIX}{dist_id:05d}{SHARD_META_SUFFIX}")
    async_writers = []
    if use_async:
        total_size, writers = async_save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint,
            index_file=index_file,
            base_filename=weights_name,
            is_master=True,
            state_preprocess=False,
        )
        async_writers.extend(writers)
    else:
        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint,
            index_file=index_file,
            base_filename=weights_name,
            is_master=True,
            use_safetensors=use_safetensors,
            use_pp_format=True,
        )
    for k, _ in model_metadata.items():
        model_metadata[k]["file"] = index_file.get_checkpoint_file(k)

    save_metadata(model_metadata, metadata_file, total_size=total_size)
    return async_writers
