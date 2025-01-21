import json
import os
from contextlib import contextmanager
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import _get_default_group

from colossalai.interface import ModelWrapper
from colossalai.shardformer.layer.parallel_module import ParallelModule

from .utils import load_state_dict, search_tp_partition_dim

MODEL_META_PREFIX = "pytorch_model-meta-dist-"
MODEL_WEIGHT_PREFIX = "pytorch_model-dist-"
SHARD_META_SUFFIX = ".index.json"
UNSHARD_META_SUFFIX = ".json"


@contextmanager
def RestoreDefaultStateDictBehavior(model):
    original_methods = {}
    for name, module in model.named_modules():
        if isinstance(module, ParallelModule):
            original_methods[module] = (module._save_to_state_dict, module._load_from_state_dict)
            module._save_to_state_dict = nn.Module._save_to_state_dict.__get__(module, nn.Module)
            module._load_from_state_dict = nn.Module._load_from_state_dict.__get__(module, nn.Module)
    try:
        yield model
    finally:
        for module, original_method in original_methods.items():
            module._save_to_state_dict, module._load_from_state_dict = original_method


def create_model_metadata(
    model: ModelWrapper,
    prefix: str = "",
    tp_size: int = None,
    tp_rank: int = None,
    zero_size: int = None,
    zero_rank: int = None,
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
        model_metadata[prefix + name]["offsets"] = [0] * len(original_shape)
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


def load_dist_model(
    model_metadata: Dict,
    checkpoint: str,
):
    """
    Load model from a single file with the given path of checkpoint.

    Args:
        model (nn.Module): The model to be loaded.
        checkpoint_index_file (str): Path to the checkpoint file.
        strict (bool, optional): For name matching during loading state_dict. Defaults to False.
                                    This argument should be manually set to False since not all params in checkpoint are needed for each device when pipeline is enabled.
    """
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
                if key not in covered_shards or rank not in covered_shards[key]:
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

    return state_dict


def get_dist_files_name(weights_name, dist_id):
    weights_name = weights_name.replace(".bin", f"-dist-{dist_id:05d}-shard.bin")
    weights_name = weights_name.replace(".safetensors", f"-dist-{dist_id:05d}-shard.safetensors")
    return weights_name


def get_dist_meta_file_name(checkpoint, dist_id, use_safetensors):
    if use_safetensors:
        return os.path.join(checkpoint, f"{MODEL_META_PREFIX}{dist_id:05d}{SHARD_META_SUFFIX}")
    return os.path.join(checkpoint, f"{MODEL_META_PREFIX}{dist_id:05d}{UNSHARD_META_SUFFIX}")
