"""The image augmentation must be upstream's pipeline, not an equivalent-looking one.

``_augment`` is random, so it cannot be pinned by comparing one draw against a recorded number. It is pinned
the way a random transform can be: the reference below is a transcription of
``droid_lerobot_dataset.py::_compose_multi_view``'s augmentation block, both sides are driven from the same
seed, and the outputs have to agree element for element.

The reference is deliberately written out rather than imported from the module under test. A test whose
expectation comes from the implementation measures self-consistency, not correctness; the hand-rolled version
this replaces passed every check this example had while differing from upstream in four ways at once -- a
random crop ratio instead of a fixed 0.95, a per-view draw instead of one shared across the three views,
bilinear instead of antialiased resize, and no ColorJitter at all.

Runs on CPU in under a second; no checkpoint, no dataset, no device.
"""

import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_dataset import CAMS, DroidActionPolicyDataset

T_FRAMES, C, H, W = 4, 3, 180, 320
SEED = 1234


def upstream_augment(views):
    """Transcribed from droid_lerobot_dataset.py::_compose_multi_view, augmentation block only."""
    wrist, left, right = (views[c] for c in CAMS)
    _, _, h, w = wrist.shape
    augmentor = T.Compose(
        [
            T.RandomCrop((int(h * 0.95), int(w * 0.95))),
            T.Resize((h, w), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
    )
    n, m = wrist.shape[0], wrist.shape[0] + left.shape[0]
    combined = augmentor(torch.cat([wrist, left, right], dim=0))
    return {"wrist": combined[:n], "left": combined[n:m], "right": combined[m:]}


@pytest.fixture
def views():
    torch.manual_seed(0)
    return {c: torch.rand(T_FRAMES, C, H, W) for c in CAMS}


@pytest.fixture
def dataset():
    """A dataset shell: ``_augment`` reads only its own lazy transform slot."""
    ds = DroidActionPolicyDataset.__new__(DroidActionPolicyDataset)
    ds._augmentor = None
    return ds


def test_matches_upstreams_pipeline_element_for_element(dataset, views):
    """The whole point. Building a Compose does not consume the RNG, applying it does, so both sides are
    seeded immediately before the call that draws."""
    torch.manual_seed(SEED)
    expected = upstream_augment({k: v.clone() for k, v in views.items()})
    torch.manual_seed(SEED)
    got = dataset._augment({k: v.clone() for k, v in views.items()})
    for cam in CAMS:
        assert torch.equal(got[cam], expected[cam]), f"{cam} differs from upstreams pipeline"


def test_the_three_views_share_one_draw(dataset):
    """The regression guard against the per-view loop this replaces.

    One transform call over the concatenated views samples its parameters once, so three identical inputs must
    come out identical. A loop that draws a crop offset per view breaks that, and nothing else in this example
    would have noticed.
    """
    const = {c: torch.full((T_FRAMES, C, H, W), 0.5) for c in CAMS}
    torch.manual_seed(SEED)
    out = dataset._augment(const)
    assert torch.equal(out["wrist"], out["left"])
    assert torch.equal(out["left"], out["right"])


def test_output_keeps_the_input_shape(dataset, views):
    """Crop then resize back: the composition stage downstream assumes the original frame size."""
    torch.manual_seed(SEED)
    out = dataset._augment(views)
    for cam in CAMS:
        assert out[cam].shape == (T_FRAMES, C, H, W)
