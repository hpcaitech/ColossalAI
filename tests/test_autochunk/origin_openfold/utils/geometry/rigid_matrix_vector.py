# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rigid3Array Transformations represented by a Matrix and a Vector."""

from __future__ import annotations

import dataclasses
from typing import List, Union

import torch
from fastfold.utils.geometry import rotation_matrix, vector

Float = Union[float, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Rigid3Array:
    """Rigid Transformation, i.e. element of special euclidean group."""

    rotation: rotation_matrix.Rot3Array
    translation: vector.Vec3Array

    def __matmul__(self, other: Rigid3Array) -> Rigid3Array:
        new_rotation = self.rotation @ other.rotation    # __matmul__
        new_translation = self.apply_to_point(other.translation)
        return Rigid3Array(new_rotation, new_translation)

    def __getitem__(self, index) -> Rigid3Array:
        return Rigid3Array(
            self.rotation[index],
            self.translation[index],
        )

    def __mul__(self, other: torch.Tensor) -> Rigid3Array:
        return Rigid3Array(
            self.rotation * other,
            self.translation * other,
        )

    def map_tensor_fn(self, fn) -> Rigid3Array:
        return Rigid3Array(
            self.rotation.map_tensor_fn(fn),
            self.translation.map_tensor_fn(fn),
        )

    def inverse(self) -> Rigid3Array:
        """Return Rigid3Array corresponding to inverse transform."""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply_to_point(-self.translation)
        return Rigid3Array(inv_rotation, inv_translation)

    def apply_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Apply Rigid3Array transform to point."""
        return self.rotation.apply_to_point(point) + self.translation

    def apply(self, point: torch.Tensor) -> vector.Vec3Array:
        return self.apply_to_point(vector.Vec3Array.from_array(point))

    def apply_inverse_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Apply inverse Rigid3Array transform to point."""
        new_point = point - self.translation
        return self.rotation.apply_inverse_to_point(new_point)

    def compose_rotation(self, other_rotation):
        rot = self.rotation @ other_rotation
        return Rigid3Array(rot, self.translation.clone())

    def compose(self, other_rigid):
        return self @ other_rigid

    def unsqueeze(self, dim: int):
        return Rigid3Array(
            self.rotation.unsqueeze(dim),
            self.translation.unsqueeze(dim),
        )

    @property
    def shape(self) -> torch.Size:
        return self.rotation.xx.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.rotation.xx.dtype

    @property
    def device(self) -> torch.device:
        return self.rotation.xx.device

    @classmethod
    def identity(cls, shape, device) -> Rigid3Array:
        """Return identity Rigid3Array of given shape."""
        return cls(rotation_matrix.Rot3Array.identity(shape, device), vector.Vec3Array.zeros(shape, device))

    @classmethod
    def cat(cls, rigids: List[Rigid3Array], dim: int) -> Rigid3Array:
        return cls(
            rotation_matrix.Rot3Array.cat([r.rotation for r in rigids], dim=dim),
            vector.Vec3Array.cat([r.translation for r in rigids], dim=dim),
        )

    def scale_translation(self, factor: Float) -> Rigid3Array:
        """Scale translation in Rigid3Array by 'factor'."""
        return Rigid3Array(self.rotation, self.translation * factor)

    def to_tensor(self) -> torch.Tensor:
        rot_array = self.rotation.to_tensor()
        vec_array = self.translation.to_tensor()
        array = torch.zeros(rot_array.shape[:-2] + (4, 4), device=rot_array.device, dtype=rot_array.dtype)
        array[..., :3, :3] = rot_array
        array[..., :3, 3] = vec_array
        array[..., 3, 3] = 1.
        return array

    def to_tensor_4x4(self) -> torch.Tensor:
        return self.to_tensor()

    def reshape(self, new_shape) -> Rigid3Array:
        rots = self.rotation.reshape(new_shape)
        trans = self.translation.reshape(new_shape)
        return Rigid3Aray(rots, trans)

    def stop_rot_gradient(self) -> Rigid3Array:
        return Rigid3Array(
            self.rotation.stop_gradient(),
            self.translation,
        )

    @classmethod
    def from_array(cls, array):
        rot = rotation_matrix.Rot3Array.from_array(array[..., :3, :3],)
        vec = vector.Vec3Array.from_array(array[..., :3, 3])
        return cls(rot, vec)

    @classmethod
    def from_tensor_4x4(cls, array):
        return cls.from_array(array)

    @classmethod
    def from_array4x4(cls, array: torch.tensor) -> Rigid3Array:
        """Construct Rigid3Array from homogeneous 4x4 array."""
        rotation = rotation_matrix.Rot3Array(array[..., 0, 0], array[..., 0, 1], array[..., 0, 2], array[..., 1, 0],
                                             array[..., 1, 1], array[..., 1, 2], array[..., 2, 0], array[..., 2, 1],
                                             array[..., 2, 2])
        translation = vector.Vec3Array(array[..., 0, 3], array[..., 1, 3], array[..., 2, 3])
        return cls(rotation, translation)
