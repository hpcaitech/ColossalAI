"""OpenPI MsgPack/NumPy wire protocol shared with RoboLab's Cosmos3 client.

No Isaac, openpi or accelerator dependencies are needed for a handshake. Tokens are read
from COSMOS3_API_TOKEN, so they never enter manifests, subprocess arguments or receipts.
"""

from __future__ import annotations

import os
from typing import Any

import msgpack
import numpy as np

MAX_MESSAGE_SIZE = 16 * 1024 * 1024


def _dtype(value) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise ValueError(f"Unsupported wire dtype: {dtype}")
    return dtype


def _encode(value: Any) -> dict:
    if isinstance(value, np.ndarray):
        _dtype(value.dtype)
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        _dtype(value.dtype)
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value: dict) -> Any:
    if b"__ndarray__" in value:
        dtype = _dtype(value[b"dtype"])
        shape = value[b"shape"]
        if not isinstance(shape, (list, tuple)) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("Invalid ndarray shape")
        # frombuffer + reshape validates the exact byte count without allocating from untrusted dimensions.
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(tuple(shape)).copy()
    if b"__npgeneric__" in value:
        return _dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode)


def unpack(value: bytes) -> Any:
    if not isinstance(value, bytes) or len(value) > MAX_MESSAGE_SIZE:
        raise ValueError("Expected a binary MsgPack message of at most 16 MiB")
    return msgpack.unpackb(value, object_hook=_decode, strict_map_key=False)


def server_metadata(uri: str) -> dict:
    from websockets.sync.client import connect

    token = os.environ.get("COSMOS3_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with connect(uri, additional_headers=headers, compression=None, max_size=MAX_MESSAGE_SIZE, open_timeout=30) as ws:
        result = unpack(ws.recv(timeout=30))
    if not isinstance(result, dict) or result.get("type") == "error":
        raise ValueError("Policy endpoint did not return a valid metadata handshake")
    return result
