"""The action-cache converter must be bit-exact and keep the parquet row order.

The cache is read by ``DroidActionPolicyDataset._read_actions`` as ``[row_from + start]`` slices, so the only
contract that matters is: the same rows, the same values, in the same order. The full-dataset check was done on
274 real windows (max diff 0.0); this test pins the converter itself with a synthetic parquet so a schema change
fails in CI rather than hours into a run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_action_cache.py"
SPEC = importlib.util.spec_from_file_location("build_action_cache", SCRIPT)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


def test_convert_roundtrip(tmp_path):
    rows = 4
    table = pa.table(
        {
            CACHE.JOINT_ACTION: [[float(i * 7 + j) for j in range(7)] for i in range(rows)],
            CACHE.GRIPPER_ACTION: [[float(i)] for i in range(rows)],
            CACHE.STATE_JOINT: [[float(100 + i * 7 + j) for j in range(7)] for i in range(rows)],
            CACHE.STATE_GRIPPER: [[float(200 + i)] for i in range(rows)],
            "episode_index": list(range(rows)),
        }
    )
    source = tmp_path / "chunk-000" / "file-000.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(table, source)
    destination = tmp_path / "cache" / "chunk-000" / "file-000.npy"

    count, seconds = CACHE.convert(source, destination)

    assert count == rows
    assert seconds >= 0.0
    array = np.load(destination)
    assert array.shape == (rows, 17)
    assert array.dtype == np.float32
    expected_joint = np.array([[float(i * 7 + j) for j in range(7)] for i in range(rows)], dtype=np.float32)
    expected_state = np.array([[float(100 + i * 7 + j) for j in range(7)] for i in range(rows)], dtype=np.float32)
    np.testing.assert_array_equal(array[:, 0:7], expected_joint)
    np.testing.assert_array_equal(array[:, 7], np.arange(rows, dtype=np.float32))
    np.testing.assert_array_equal(array[:, 8:15], expected_state)
    np.testing.assert_array_equal(array[:, 15], np.array([200 + i for i in range(rows)], dtype=np.float32))
    np.testing.assert_array_equal(array[:, 16], np.arange(rows, dtype=np.float32))
