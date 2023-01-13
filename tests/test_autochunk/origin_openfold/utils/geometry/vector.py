# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vec3Array Class."""

from __future__ import annotations

import dataclasses
from typing import List, Union

import torch
from fastfold.utils.geometry import utils

Float = Union[float, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Vec3Array:
    x: torch.Tensor = dataclasses.field(metadata={'dtype': torch.float32})
    y: torch.Tensor
    z: torch.Tensor

    def __post_init__(self):
        if hasattr(self.x, 'dtype'):
            assert self.x.dtype == self.y.dtype
            assert self.x.dtype == self.z.dtype
            assert all([x == y for x, y in zip(self.x.shape, self.y.shape)])
            assert all([x == z for x, z in zip(self.x.shape, self.z.shape)])

    def __add__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x * other,
            self.y * other,
            self.z * other,
        )

    def __rmul__(self, other: Float) -> Vec3Array:
        return self * other

    def __truediv__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x / other,
            self.y / other,
            self.z / other,
        )

    def __neg__(self) -> Vec3Array:
        return self * -1

    def __pos__(self) -> Vec3Array:
        return self * 1

    def __getitem__(self, index) -> Vec3Array:
        return Vec3Array(
            self.x[index],
            self.y[index],
            self.z[index],
        )

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    @property
    def shape(self):
        return self.x.shape

    def map_tensor_fn(self, fn) -> Vec3Array:
        return Vec3Array(
            fn(self.x),
            fn(self.y),
            fn(self.z),
        )

    def cross(self, other: Vec3Array) -> Vec3Array:
        """Compute cross product between 'self' and 'other'."""
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return Vec3Array(new_x, new_y, new_z)

    def dot(self, other: Vec3Array) -> Float:
        """Compute dot product between 'self' and 'other'."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self, epsilon: float = 1e-6) -> Float:
        """Compute Norm of Vec3Array, clipped to epsilon."""
        # To avoid NaN on the backward pass, we must use maximum before the sqrt
        norm2 = self.dot(self)
        if epsilon:
            norm2 = torch.clamp(norm2, min=epsilon**2)
        return torch.sqrt(norm2)

    def norm2(self):
        return self.dot(self)

    def normalized(self, epsilon: float = 1e-6) -> Vec3Array:
        """Return unit vector with optional clipping."""
        return self / self.norm(epsilon)

    def clone(self) -> Vec3Array:
        return Vec3Array(
            self.x.clone(),
            self.y.clone(),
            self.z.clone(),
        )

    def reshape(self, new_shape) -> Vec3Array:
        x = self.x.reshape(new_shape)
        y = self.y.reshape(new_shape)
        z = self.z.reshape(new_shape)

        return Vec3Array(x, y, z)

    def sum(self, dim: int) -> Vec3Array:
        return Vec3Array(
            torch.sum(self.x, dim=dim),
            torch.sum(self.y, dim=dim),
            torch.sum(self.z, dim=dim),
        )

    def unsqueeze(self, dim: int):
        return Vec3Array(
            self.x.unsqueeze(dim),
            self.y.unsqueeze(dim),
            self.z.unsqueeze(dim),
        )

    @classmethod
    def zeros(cls, shape, device="cpu"):
        """Return Vec3Array corresponding to zeros of given shape."""
        return cls(torch.zeros(shape, dtype=torch.float32, device=device),
                   torch.zeros(shape, dtype=torch.float32, device=device),
                   torch.zeros(shape, dtype=torch.float32, device=device))

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @classmethod
    def from_array(cls, tensor):
        return cls(*torch.unbind(tensor, dim=-1))

    @classmethod
    def cat(cls, vecs: List[Vec3Array], dim: int) -> Vec3Array:
        return cls(
            torch.cat([v.x for v in vecs], dim=dim),
            torch.cat([v.y for v in vecs], dim=dim),
            torch.cat([v.z for v in vecs], dim=dim),
        )


def square_euclidean_distance(vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6) -> Float:
    """Computes square of euclidean distance between 'vec1' and 'vec2'.

    Args:
        vec1: Vec3Array to compute    distance to
        vec2: Vec3Array to compute    distance from, should be
                    broadcast compatible with 'vec1'
        epsilon: distance is clipped from below to be at least epsilon

    Returns:
        Array of square euclidean distances;
        shape will be result of broadcasting 'vec1' and 'vec2'
    """
    difference = vec1 - vec2
    distance = difference.dot(difference)
    if epsilon:
        distance = torch.maximum(distance, epsilon)
    return distance


def dot(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.dot(vector2)


def cross(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.cross(vector2)


def norm(vector: Vec3Array, epsilon: float = 1e-6) -> Float:
    return vector.norm(epsilon)


def normalized(vector: Vec3Array, epsilon: float = 1e-6) -> Vec3Array:
    return vector.normalized(epsilon)


def euclidean_distance(vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6) -> Float:
    """Computes euclidean distance between 'vec1' and 'vec2'.

    Args:
        vec1: Vec3Array to compute euclidean distance to
        vec2: Vec3Array to compute euclidean distance from, should be
                    broadcast compatible with 'vec1'
        epsilon: distance is clipped from below to be at least epsilon

    Returns:
        Array of euclidean distances;
        shape will be result of broadcasting 'vec1' and 'vec2'
    """
    distance_sq = square_euclidean_distance(vec1, vec2, epsilon**2)
    distance = torch.sqrt(distance_sq)
    return distance


def dihedral_angle(a: Vec3Array, b: Vec3Array, c: Vec3Array, d: Vec3Array) -> Float:
    """Computes torsion angle for a quadruple of points.

    For points (a, b, c, d), this is the angle between the planes defined by
    points (a, b, c) and (b, c, d). It is also known as the dihedral angle.

    Arguments:
        a: A Vec3Array of coordinates.
        b: A Vec3Array of coordinates.
        c: A Vec3Array of coordinates.
        d: A Vec3Array of coordinates.

    Returns:
        A tensor of angles in radians: [-pi, pi].
    """
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = v1.cross(v2)
    c2 = v3.cross(v2)
    c3 = c2.cross(c1)

    v2_mag = v2.norm()
    return torch.atan2(c3.dot(v2), v2_mag * c1.dot(c2))
