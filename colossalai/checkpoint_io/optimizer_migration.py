"""Explicit offline migration of trusted legacy optimizer checkpoints.

This module is not used by training loaders. Legacy decoding can execute code.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from colossalai.utils.safetensors import load_flat, save_optimizer_file

from .index_file import CheckpointIndexFile


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trusted_legacy(path):
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=False)
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if "colossalai_optimizer_format" in metadata:
            return load_flat(path)
        objects = json.loads(metadata.get("non_tensor_keys", "[]"))
        if type(objects) is not list or any(type(key) is not str for key in objects):
            raise ValueError("Invalid legacy optimizer object list")
        if len(set(objects)) != len(objects) or not set(objects).issubset(reader.keys()):
            raise ValueError("Invalid legacy optimizer object keys")
        flat = {}
        for key in reader.keys():
            value = reader.get_tensor(key)
            if key in objects:
                import pickle

                if value.dtype != torch.uint8 or value.dim() != 1:
                    raise ValueError("Legacy optimizer object must be a byte tensor")
                # Deliberately isolated behind the caller's explicit trust opt-in.
                value = pickle.loads(value.numpy().tobytes())
            flat[key] = value
    states = {}
    groups = flat.pop("param_groups", None)
    for key, value in flat.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "state" or not parts[1].isdigit():
            raise ValueError("Invalid legacy optimizer state key")
        states.setdefault(int(parts[1]), {})[parts[2]] = value
    return states if groups is None else {"state": states, "param_groups": groups}


def _write_exclusive(path, state):
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".optimizer-migration-") as staging:
        temporary = Path(staging) / "optimizer.safetensors"
        save_optimizer_file(temporary, state)
        # Unlike replace(), link() cannot overwrite a concurrently created file.
        os.link(temporary, path)


def migrate_optimizer_checkpoint(source, destination, *, trust_legacy_pickle=False):
    """Convert a trusted file or optimizer index/directory without overwriting it.

    ``trust_legacy_pickle`` permits arbitrary code execution while reading the
    legacy input. Use an isolated environment without network or credentials.
    """
    if trust_legacy_pickle is not True:
        raise ValueError("Explicit trust_legacy_pickle=True is required; legacy inputs can execute code")
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source.is_dir():
        indices = list(source.glob("*optim*.index.json"))
        if len(indices) != 1:
            raise ValueError("Specify an optimizer index file; exactly one optimizer index is required")
        source = indices[0]
    if not source.name.endswith(".index.json") and destination.suffix != ".safetensors":
        raise ValueError("A single-file migration destination must end with .safetensors")
    inputs = {source.name: _sha256(source)}

    def read_input(path):
        inputs[str(path.relative_to(source.parent))] = _sha256(path)
        return _load_trusted_legacy(path)

    if not source.name.endswith(".index.json"):
        state = _load_trusted_legacy(source)
        if type(state) is list:
            state = {"param_groups": state}
        _write_exclusive(destination, state)
        return {"format_version": 2, "inputs": inputs, "outputs": {destination.name: _sha256(destination)}}
    index = CheckpointIndexFile.from_file(source, is_optimizer=True)
    groups_path = index.get_param_group_filename()
    if groups_path is None:
        raise ValueError("Optimizer index has no parameter group file")
    groups = read_input(Path(groups_path))
    if type(groups) is dict and "param_groups" in groups:
        groups = groups["param_groups"]
    # Reserve a new output directory. The index is published only after every
    # file has been converted; failed conversion leaves no usable index.
    destination.mkdir()
    save_optimizer_file(destination / "optimizer_group.safetensors", {"param_groups": groups})
    output = CheckpointIndexFile(destination)
    output.append_meta_data("param_groups", "optimizer_group.safetensors")
    output.append_meta_data("param_groups_format", 2)
    output.append_meta_data("optimizer_format", 2)
    converted = {}
    total_size = 0
    for filename in dict.fromkeys(index.weight_map.values()):
        state = read_input(Path(index._resolve_file(filename)))
        new_name = f"optimizer-{len(converted) + 1:05d}.safetensors"
        save_optimizer_file(destination / new_name, state)
        converted[filename] = new_name
        total_size += (destination / new_name).stat().st_size
    for param_id, filename in index.weight_map.items():
        output.append_weight_map(param_id, converted[filename])
    output.append_meta_data("total_size", total_size)
    output.write_index_file("optimizer.safetensors.index.json")
    return {
        "format_version": 2,
        "inputs": inputs,
        "outputs": {path.name: _sha256(path) for path in destination.iterdir()},
    }
