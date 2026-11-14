"""The device uint8 path must reproduce the host pad/normalise path it replaces."""

import torch

from examples.worldmodel.cosmos3.droid_dataset import resize_and_reflect_pad
from examples.worldmodel.cosmos3.video_uint8 import pad_normalize_torch


def host_reference(video, target_h, target_w, pad_columns=None):
    padded, content_hw = resize_and_reflect_pad(video.float(), target_h, target_w, pad_columns=pad_columns)
    return padded / 127.5 - 1.0, content_hw


def test_the_device_path_matches_the_host_path():
    video = torch.randint(0, 256, (3, 2, 5, 7), dtype=torch.uint8)
    pad_h, pad_w = 3, 4

    expected, _ = host_reference(video, 5 + pad_h, 7 + pad_w)
    actual = pad_normalize_torch(video, pad_h, pad_w, torch.float32)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_the_bf16_cast_before_padding_is_exact():
    video = torch.randint(0, 256, (3, 2, 5, 7), dtype=torch.uint8)
    pad_h, pad_w = 3, 4

    expected, _ = host_reference(video, 5 + pad_h, 7 + pad_w)
    actual = pad_normalize_torch(video, pad_h, pad_w, torch.bfloat16)

    # Reflection copies values, so casting before or after the pad is the same operation.
    torch.testing.assert_close(actual.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_a_content_pad_keeps_the_full_canvas_geometry():
    video = torch.randint(0, 256, (3, 2, 5, 7), dtype=torch.uint8)

    full, content_hw = host_reference(video, 8, 12)
    narrow, narrow_content_hw = host_reference(video, 8, 12, pad_columns=2)

    assert narrow_content_hw == content_hw
    assert narrow.shape[-1] < full.shape[-1]
    torch.testing.assert_close(narrow, full[..., : narrow.shape[-1]], rtol=0, atol=0)
