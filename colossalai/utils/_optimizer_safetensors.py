"""Data-only optimizer checkpoint schema. No Python object reconstruction."""

import json
import math
from collections import OrderedDict

import torch

FORMAT_KEY = "colossalai_optimizer_format"
FORMAT_VERSION = "2"
SCHEMA_KEY = "optimizer_schema"
KINDS = ("full", "state_shard", "param_groups")
MAX_SCHEMA_BYTES = 16 * 1024 * 1024
MAX_NODES = 1_000_000
MAX_DEPTH = 64


class LegacyOptimizerCheckpointError(ValueError):
    """An optimizer checkpoint requires legacy pickle object decoding."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate optimizer schema key: {key}")
        result[key] = value
    return result


def _integer(value):
    if len(value) > 4096:
        raise ValueError("Optimizer schema integer is too large")
    return int(value)


def _constant(value):
    raise ValueError(f"Non-standard JSON constant in optimizer schema: {value}")


def _budget(depth, count):
    count[0] += 1
    if depth > MAX_DEPTH or count[0] > MAX_NODES:
        raise ValueError("Optimizer schema exceeds the depth or node limit")


def _fields(node, expected):
    if type(node) is not dict or set(node) != set(expected):
        raise ValueError("Invalid optimizer schema fields")


def _validate_payload(payload, kind):
    if type(payload) is not dict:
        raise ValueError("Optimizer payload must be a dictionary")
    expected = {"state", "param_groups"} if kind == "full" else {"state" if kind == "state_shard" else "param_groups"}
    if set(payload) != expected:
        raise ValueError(f"Invalid fields for optimizer checkpoint kind {kind}")
    if "state" in payload:
        states = payload["state"]
        if type(states) not in (dict, OrderedDict):
            raise ValueError("Optimizer state must be a dictionary")
        for param_id, state in states.items():
            if type(param_id) is not int or param_id < 0 or type(state) not in (dict, OrderedDict):
                raise ValueError("Optimizer state requires non-negative integer IDs and dictionaries")
            if any(type(key) is not str for key in state):
                raise ValueError("Optimizer state field names must be strings")
    if "param_groups" in payload:
        groups = payload["param_groups"]
        if type(groups) is not list:
            raise ValueError("Optimizer param_groups must be a list")
        seen = set()
        for group in groups:
            if type(group) not in (dict, OrderedDict) or "params" not in group:
                raise ValueError("Each optimizer parameter group requires params")
            if any(type(key) is not str for key in group):
                raise ValueError("Optimizer parameter group keys must be strings")
            if type(group["params"]) is not list:
                raise ValueError("Optimizer params must be a list of integer IDs")
            for param_id in group["params"]:
                if type(param_id) is not int or param_id < 0 or param_id in seen:
                    raise ValueError("Optimizer parameter IDs must be non-negative and unique")
                seen.add(param_id)
        if kind == "full" and not set(payload["state"]).issubset(seen):
            raise ValueError("Optimizer state references an ID absent from param_groups")


def flatten_optimizer(state_dict, kind=None):
    if type(state_dict) not in (dict, OrderedDict):
        raise TypeError("Optimizer state_dict must be a dictionary")
    if kind is None:
        if "param_groups" in state_dict:
            kind = "full" if "state" in state_dict else "param_groups"
        else:
            kind = "state_shard"
    if kind not in KINDS:
        raise ValueError(f"Unsupported optimizer checkpoint kind: {kind}")
    payload = dict(state_dict) if kind != "state_shard" else {"state": state_dict}
    _validate_payload(payload, kind)
    tensors = {}
    active = set()
    count = [0]

    def encode(value, path, depth=0):
        _budget(depth, count)
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized:
                raise TypeError(f"Only dense strided tensors are supported at {path}")
            # Paths, rather than per-shard counters, keep pinned-cache names stable.
            key = "optimizer.tensor." + json.dumps(path, ensure_ascii=True, separators=(",", ":"))
            tensor = value.detach().contiguous()
            tensors[key] = tensor
            return {"type": "tensor_ref", "key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        value_type = type(value)
        if value is None:
            return {"type": "null"}
        if value_type in (bool, int, float, str):
            if value_type is int and value.bit_length() > 4096:
                raise ValueError(f"Optimizer integer is too large at {path}")
            if value_type is float and not math.isfinite(value):
                value = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
            return {"type": value_type.__name__, "value": value}
        if value_type not in (list, tuple, dict, OrderedDict):
            raise TypeError(f"Unsupported optimizer value {value_type.__name__} at {path}")
        if id(value) in active:
            raise ValueError(f"Cyclic optimizer data at {path}")
        active.add(id(value))
        try:
            if value_type in (list, tuple):
                items = [encode(item, path + [("item", i)], depth + 1) for i, item in enumerate(value)]
                return {"type": value_type.__name__, "items": items}
            items = []
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise TypeError(f"Unsupported optimizer dictionary key at {path}")
                items.append(
                    [encode(key, path + [("key", key)], depth + 1), encode(item, path + [("value", key)], depth + 1)]
                )
            return {"type": "dict", "items": items}
        finally:
            active.remove(id(value))

    schema = json.dumps(encode(payload, []), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(schema.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ValueError("Optimizer schema exceeds the byte limit")
    return tensors, {FORMAT_KEY: FORMAT_VERSION, "kind": kind, SCHEMA_KEY: schema}


def parse_optimizer_metadata(metadata, expected_kind=None):
    metadata = metadata or {}
    if "non_tensor_keys" in metadata:
        raise LegacyOptimizerCheckpointError(
            "Legacy pickle optimizer checkpoints are not loaded. Migrate a trusted checkpoint with "
            "colossalai checkpoint migrate-optimizer --trust-legacy-pickle."
        )
    version = metadata.get(FORMAT_KEY)
    if version is None:
        if SCHEMA_KEY in metadata or "kind" in metadata:
            raise ValueError("Optimizer format version is missing")
        if expected_kind in ("full", "param_groups"):
            raise LegacyOptimizerCheckpointError("Legacy optimizer metadata requires explicit migration")
        return None, None
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported optimizer checkpoint version: {version}")
    kind = metadata.get("kind")
    if kind not in KINDS or (expected_kind is not None and kind != expected_kind):
        raise ValueError(f"Unexpected optimizer checkpoint kind: {kind}")
    raw_schema = metadata.get(SCHEMA_KEY)
    if type(raw_schema) is not str or len(raw_schema.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ValueError("Missing or oversized optimizer schema")
    try:
        schema = json.loads(raw_schema, object_pairs_hook=_pairs, parse_int=_integer, parse_constant=_constant)
    except (RecursionError, OverflowError) as exc:
        raise ValueError("Optimizer schema exceeds JSON nesting or numeric limits") from exc
    return kind, schema


def unflatten_optimizer(tensors, metadata, expected_kind=None, separator="."):
    kind, schema = parse_optimizer_metadata(metadata, expected_kind)
    if kind is None:
        # Legacy tensor-only shards contain no serialized Python objects.
        states = {}
        for key in tensors:
            parts = key.split(separator)
            if len(parts) != 3 or parts[0] != "state" or len(parts[1]) > 4096 or not parts[1].isdigit() or not parts[2]:
                raise LegacyOptimizerCheckpointError("Legacy optimizer layout requires explicit migration")
            param_id = int(parts[1])
            if parts[2] in states.setdefault(param_id, {}):
                raise ValueError("Duplicate legacy optimizer state field")
            states[param_id][parts[2]] = tensors[key]
        return states
    count = [0]
    referenced = set()
    cache = {}

    def decode(node, depth=0):
        _budget(depth, count)
        if type(node) is not dict or type(node.get("type")) is not str:
            raise ValueError("Optimizer schema node requires a type")
        tag = node["type"]
        if tag == "null":
            _fields(node, ("type",))
            return None
        if tag in ("bool", "int", "float", "str"):
            _fields(node, ("type", "value"))
            value = node["value"]
            if tag == "float" and type(value) is str and value in ("nan", "+inf", "-inf"):
                return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[value]
            expected = {"bool": bool, "int": int, "float": float, "str": str}[tag]
            if type(value) is not expected or (tag == "float" and not math.isfinite(value)):
                raise ValueError(f"Invalid optimizer {tag} value")
            if tag == "int" and value.bit_length() > 4096:
                raise ValueError("Optimizer schema integer is too large")
            return value
        if tag == "tensor_ref":
            _fields(node, ("type", "key", "dtype", "shape"))
            key = node["key"]
            if type(key) is not str or not key.startswith("optimizer.tensor.") or key not in tensors:
                raise ValueError("Invalid optimizer tensor reference")
            shape = node["shape"]
            if type(shape) is not list or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("Invalid optimizer tensor shape")
            if key not in cache:
                cache[key] = tensors[key]
            tensor = cache[key]
            if list(tensor.shape) != shape or str(tensor.dtype) != node["dtype"]:
                raise ValueError("Optimizer tensor reference has the wrong shape or dtype")
            referenced.add(key)
            return tensor
        if tag in ("list", "tuple", "dict"):
            _fields(node, ("type", "items"))
            if type(node["items"]) is not list:
                raise ValueError("Optimizer container requires a list of items")
            if tag != "dict":
                items = [decode(item, depth + 1) for item in node["items"]]
                return items if tag == "list" else tuple(items)
            result = {}
            for pair in node["items"]:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("Invalid optimizer dictionary entry")
                key = decode(pair[0], depth + 1)
                if type(key) not in (str, int) or key in result:
                    raise ValueError("Invalid or duplicate optimizer dictionary key")
                result[key] = decode(pair[1], depth + 1)
            return result
        raise ValueError(f"Unsupported optimizer schema type: {tag}")

    payload = decode(schema)
    if referenced != set(tensors):
        raise ValueError("Optimizer checkpoint contains unreferenced tensors")
    _validate_payload(payload, kind)
    return payload["state"] if kind == "state_shard" else payload
