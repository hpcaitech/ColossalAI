"""The command line parses.

This exists because it did not. Dropping the precomputed-latent path removed the two options inside one
``parser.add_argument(`` call but left the call itself, as ``parser.add_argument()`` with no arguments. Python
imports that file happily; argparse raises only when the parser is built, so every test passed, the smoke
script passed -- neither builds this parser -- and the training script died on its first line of real work
with ``_get_positional_kwargs() missing 1 required positional argument: 'dest'``.

So the assertion worth making is the cheap one: build the parser and parse a line. Anything that leaves a
malformed ``add_argument`` behind fails here, whatever the option was called.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

train_sft = pytest.importorskip("train_sft", reason="needs torch, colossalai and the example's own modules")


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
    assert args.backend == "npu"
    assert args.vae_dtype == "bfloat16"


def test_data_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--checkpoint", "/ckpt"])
    with pytest.raises(SystemExit):
        train_sft.parse_args()
