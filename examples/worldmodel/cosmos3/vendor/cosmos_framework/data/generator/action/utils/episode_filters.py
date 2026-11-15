# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Storage-independent episode keys for optional dataset filter metadata."""

from typing import Any


def normalize_episode_filter_keys(filters: dict[str, Any], *, source_prefix: str = "") -> dict[str, Any]:
    """Preserve complete relative episode IDs; URI keys require an explicit dataset prefix."""
    normalized: dict[str, Any] = {}
    for key, ranges in filters.items():
        episode = key
        if "://" in key:
            # Only composite URI keys carry the recording/trajectory delimiter.
            # Relative episode IDs are opaque, including any '--' in their names.
            recording = key.split("/recordings/MP4--", 1)[0].rstrip("/")
            if not source_prefix:
                raise ValueError(
                    "filter_dict_source_prefix is required for URI-based episode filter keys. "
                    "Set DROID_FILTER_SOURCE_PREFIX when using a DROID launcher, or pass "
                    "filter_dict_source_prefix to the dataset. Use the URI prefix before the complete episode ID."
                )
            prefix = source_prefix.rstrip("/") + "/"
            if not recording.startswith(prefix):
                raise ValueError(
                    "Episode filter URI does not match filter_dict_source_prefix. "
                    "Check DROID_FILTER_SOURCE_PREFIX against the keys in your filter JSON."
                )
            episode = recording.removeprefix(prefix)
        if not episode:
            raise ValueError("Filter entries must identify an episode")
        if episode in normalized and normalized[episode] != ranges:
            raise ValueError(f"Conflicting filter ranges for episode {episode!r}")
        normalized[episode] = ranges
    return normalized
