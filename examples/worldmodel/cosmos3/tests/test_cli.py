"""Preserve the upstream parser regression: importing the entry does not build its parser.

A malformed, empty ``add_argument()`` once survived imports and packing tests but made
all training commands fail. Keep the minimal command, recipe and required-data checks.
The backend default is now ``auto`` so the same command supports both NPU and CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from examples.worldmodel.cosmos3 import train_sft
from examples.worldmodel.cosmos3.runtime import resolve_attention_backend


def test_a_minimal_command_line_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--data", "/data", "--checkpoint", "/ckpt"])
    args = train_sft.parse_args()
    assert args.data == Path("/data")
    assert args.checkpoint == Path("/ckpt")


def test_the_recipe_defaults_are_the_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--data", "/data", "--checkpoint", "/ckpt"])
    args = train_sft.parse_args()
    assert (args.samples_per_rank, args.grad_accum, args.max_iter) == (32, 32, 10000)
    assert (args.chunk_length, args.fps, args.resolution) == (32, 15.0, "480")
    assert args.activation_checkpointing == "full"
    assert args.attn_backend == "auto"
    assert resolve_attention_backend(args.attn_backend, "npu") == "npu"
    assert resolve_attention_backend(args.attn_backend, "cuda") == "cuda"
    assert args.vae_dtype == "bfloat16"


def test_data_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--checkpoint", "/ckpt"])
    with pytest.raises(SystemExit):
        train_sft.parse_args()
