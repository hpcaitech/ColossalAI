# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Viewpoint type definitions and caption augmentor for Action datasets.

Provides a ``Viewpoint`` type alias for camera perspective labels and a
``ViewpointTextInfo`` augmentor that appends a human-readable viewpoint
description to the caption string.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.action.action_caption_attribute_adapter import ViewCompositionMappings

Viewpoint = Literal[
    "ego_view",
    "third_person_view",
    "wrist_view",
    "concat_view",
    "separate_camera_views",
    "top_down_2d_view",
    "video_game_view",
]

DEFAULT_VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    "separate_camera_views": ViewCompositionMappings["separate_camera_views"],
    "top_down_2d_view": "This video is captured from a top-down view of a flat 2D scene.",
    "video_game_view": "This video is captured from an in-game camera perspective in a virtual environment.",
}


def _resize_to_width(view: torch.Tensor, target_w: int) -> torch.Tensor:  # [T,C,H,W]
    """Aspect-preserving resize to ``target_w``; height follows from the view's own aspect ratio."""
    h, w = view.shape[-2:]
    target_h = round(h * target_w / w)
    if target_w != w or target_h != h:
        view = F.interpolate(view, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return view


def compose_multiview(
    primary: torch.Tensor,  # [T,C,H,W]
    left: torch.Tensor,  # [T,C,H_l,W_l]
    right: torch.Tensor,  # [T,C,H_r,W_r]
) -> torch.Tensor:  # [T,C,H+H_bottom,W]
    """Place the primary view above two side views resized to match its width.

    ``left``/``right`` are resized aspect-preserving so their width matches
    half the primary view's width; the resulting height follows from each
    view's own native aspect ratio rather than being forced to match the
    primary's, so a view whose native aspect ratio differs from the
    primary's is not distorted.
    """
    _, width = primary.shape[-2:]
    half_width = width // 2
    left = _resize_to_width(left, half_width)  # [T,C,H_l',W/2]
    right = _resize_to_width(right, half_width)  # [T,C,H_r',W/2]
    bottom = torch.cat([left, right], dim=-1)  # [T,C,H_bottom,W]
    return torch.cat([primary, bottom], dim=-2)  # [T,C,H+H_bottom,W]


class ViewpointTextInfo(Augmentor):
    """Augmentor that appends viewpoint type description to captions.

    Reads a viewpoint label from ``data_dict[viewpoint_key]`` and appends
    the corresponding template sentence to the caption.  Designed to run
    after the raw ``ai_caption`` is set but before duration/FPS metadata
    is appended.

    Args:
        input_keys: Input keys (kept for API compatibility).
        output_keys: Output keys (kept for API compatibility).
        args: Configuration arguments:
            - caption_key (str): Key for caption in data_dict. Default: ``"ai_caption"``
            - viewpoint_key (str): Key for viewpoint label. Default: ``"viewpoint"``
            - templates (dict): Override mapping from viewpoint to sentence.
              Default: :data:`DEFAULT_VIEWPOINT_TEMPLATES`
            - separator (str): Separator between caption and metadata. Default: ``". "``
            - enabled (bool): Whether augmentation is enabled. Default: ``True``
    """

    def __init__(
        self,
        input_keys: list | None = None,
        output_keys: list | None = None,
        args: dict | None = None,
    ) -> None:
        super().__init__(input_keys or [], output_keys or [], args)

        self.caption_key: str = args.get("caption_key", "ai_caption") if args else "ai_caption"
        self.viewpoint_key: str = args.get("viewpoint_key", "viewpoint") if args else "viewpoint"
        self.templates: dict[str, str] = (
            args.get("templates", DEFAULT_VIEWPOINT_TEMPLATES) if args else DEFAULT_VIEWPOINT_TEMPLATES
        )
        self.default_separator: str = args.get("separator", ". ") if args else ". "
        self.enabled: bool = args.get("enabled", True) if args else True

    def __call__(self, data_dict: dict) -> dict | None:
        """Append viewpoint description to the caption.

        If the sample provides an ``"additional_view_description"`` key (a
        free-form string describing the concatenated camera layout), it is
        appended after the generic ``concat_view`` template. This allows each
        dataset to supply its own description of which cameras are tiled and
        how.

        Args:
            data_dict: Sample dictionary containing caption and viewpoint.

        Returns:
            The mutated *data_dict*, or the original unchanged if the
            viewpoint key is missing or unrecognized.
        """
        if not self.enabled:
            return data_dict

        viewpoint = data_dict.get(self.viewpoint_key)
        if viewpoint is None:
            raise ValueError(
                f"ViewpointTextInfo: missing key {self.viewpoint_key!r} in data_dict. "
                f"All action datasets must provide a viewpoint label."
            )

        # Append dataset-specific concat_view details after the base template.
        additional_view_description = data_dict.pop("additional_view_description", None)
        template = self.templates.get(viewpoint)

        if template is None:
            log.warning(
                f"ViewpointTextInfo: unrecognized viewpoint {viewpoint!r}. "
                f"Known viewpoints: {sorted(self.templates.keys())}. Skipping.",
                rank0_only=False,
            )
            return data_dict

        if additional_view_description:
            separator = " " if template.endswith(".") else self.default_separator
            template = template + separator + additional_view_description.rstrip()

        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict

        caption = caption.rstrip()
        separator = " " if caption.endswith(".") else self.default_separator
        data_dict[self.caption_key] = caption + separator + template

        return data_dict
