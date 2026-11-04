"""Checks on the JSON prompt and the classifier-free-guidance dropout.

The prompt structure is checked without a tokenizer; the chat template and the dropout need the checkpoint's
tokenizer, so those skip unless ``COSMOS3_CHECKPOINT`` points at it.

Runs on CPU.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packing import CFG_DROPOUT_RATE, MAX_TEXT_TOKENS, build_action_json_prompt, tokenize_prompt

TASK = "Use the yellow towel to wipe the tabletop"


@pytest.fixture(scope="module")
def prompt():
    # 33 frames at 15 fps is 2.2 s, which is where the truncate/round difference shows up.
    return build_action_json_prompt(TASK, num_frames=33, fps=15.0, canvas_hw=(544, 736), aspect="4,3")


@pytest.fixture(scope="module")
def tokenizer():
    checkpoint = os.environ.get("COSMOS3_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set COSMOS3_CHECKPOINT to the Cosmos3-Edge-Policy-DROID directory")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(checkpoint) / "text_tokenizer")


def test_field_order_matches_the_formatter(prompt):
    """json.dumps preserves insertion order, so the order reaches the tokenizer and has to match upstream."""
    assert list(prompt.keys()) == [
        "cinematography",
        "actions",
        "duration",
        "fps",
        "resolution",
        "aspect_ratio",
    ]


def test_duration_truncates_while_the_time_range_rounds(prompt):
    """Two different roundings on the same 2.2 s, both taken from ActionPromptJsonFormatter."""
    assert prompt["duration"] == "2s"
    assert prompt["actions"][0]["time"] == "0:00-0:02"
    assert isinstance(prompt["fps"], float)


def test_resolution_is_the_padded_canvas(prompt):
    assert prompt["resolution"] == {"H": 544, "W": 736}
    assert prompt["aspect_ratio"] == "4,3"


def test_description_and_framing(prompt):
    assert prompt["actions"][0]["description"] == TASK + "."
    assert prompt["cinematography"]["framing"].startswith("This video contains concatenated views")


def test_idle_frame_is_absent_without_the_filter_table(prompt):
    """The recipe leaves use_filter_dict off, so there are no idle-frame counts and upstream drops the field."""
    assert "idle_frame" not in prompt["actions"][0]


def test_the_chat_template_is_applied(prompt, tokenizer):
    """A plain tokenizer(task) would skip the role markers and the generation prompt entirely."""
    ids = tokenize_prompt(prompt, tokenizer, cfg_dropout_rate=0.0)
    text = tokenizer.decode(ids)
    assert ids.numel() > 0
    assert "user" in text and "assistant" in text
    assert TASK in text

    bare = tokenizer(TASK, return_tensors="pt")["input_ids"][0]
    assert ids.numel() != bare.numel()


def test_dropping_the_condition_empties_the_content_but_keeps_the_markers(prompt, tokenizer):
    """Upstream replaces the whole string with "" and still runs the template."""
    kept = tokenize_prompt(prompt, tokenizer, cfg_dropout_rate=0.0)
    dropped = tokenize_prompt(prompt, tokenizer, cfg_dropout_rate=1.0)
    text = tokenizer.decode(dropped)
    assert 0 < dropped.numel() < kept.numel()
    assert TASK not in text
    assert "cinematography" not in text
    assert "user" in text


def test_dropout_fires_at_the_configured_rate(prompt, tokenizer):
    dropped_len = tokenize_prompt(prompt, tokenizer, cfg_dropout_rate=1.0).numel()
    generator = torch.Generator().manual_seed(0)
    trials = 2000
    hits = sum(
        1 for _ in range(trials) if tokenize_prompt(prompt, tokenizer, generator=generator).numel() == dropped_len
    )
    assert abs(hits / trials - CFG_DROPOUT_RATE) < 0.02


def test_truncation_limit():
    assert MAX_TEXT_TOKENS == 4096
