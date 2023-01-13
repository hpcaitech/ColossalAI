# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rot3Array Matrix Class."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from fastfold.utils.geometry import utils, vector
from fastfold.utils.tensor_utils import tensor_tree_map

COMPONENTS = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']


@dataclasses.dataclass(frozen=True)
class Rot3Array:
    """Rot3Array Matrix in 3 dimensional Space implemented as struct of arrays."""
    xx: torch.Tensor = dataclasses.field(metadata={'dtype': torch.float32})
    xy: torch.Tensor
    xz: torch.Tensor
    yx: torch.Tensor
    yy: torch.Tensor
    yz: torch.Tensor
    zx: torch.Tensor
    zy: torch.Tensor
    zz: torch.Tensor

    __array_ufunc__ = None

    def __getitem__(self, index):
        field_names = utils.get_field_names(Rot3Array)
        return Rot3Array(**{name: getattr(self, name)[index] for name in field_names})

    def __mul__(self, other: torch.Tensor):
        field_names = utils.get_field_names(Rot3Array)
        return Rot3Array(**{name: getattr(self, name) * other for name in field_names})

    def __matmul__(self, other: Rot3Array) -> Rot3Array:
        """Composes two Rot3Arrays."""
        c0 = self.apply_to_point(vector.Vec3Array(other.xx, other.yx, other.zx))
        c1 = self.apply_to_point(vector.Vec3Array(other.xy, other.yy, other.zy))
        c2 = self.apply_to_point(vector.Vec3Array(other.xz, other.yz, other.zz))
        return Rot3Array(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

    def map_tensor_fn(self, fn) -> Rot3Array:
        field_names = utils.get_field_names(Rot3Array)
        return Rot3Array(**{name: fn(getattr(self, name)) for name in field_names})

    def inverse(self) -> Rot3Array:
        """Returns inverse of Rot3Array."""
        return Rot3Array(self.xx, self.yx, self.zx, self.xy, self.yy, self.zy, self.xz, self.yz, self.zz)

    def apply_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Applies Rot3Array to point."""
        return vector.Vec3Array(self.xx * point.x + self.xy * point.y + self.xz * point.z,
                                self.yx * point.x + self.yy * point.y + self.yz * point.z,
                                self.zx * point.x + self.zy * point.y + self.zz * point.z)

    def apply_inverse_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Applies inverse Rot3Array to point."""
        return self.inverse().apply_to_point(point)

    def unsqueeze(self, dim: int):
        return Rot3Array(*tensor_tree_map(lambda t: t.unsqueeze(dim), [getattr(self, c) for c in COMPONENTS]))

    def stop_gradient(self) -> Rot3Array:
        return Rot3Array(*[getattr(self, c).detach() for c in COMPONENTS])

    @classmethod
    def identity(cls, shape, device) -> Rot3Array:
        """Returns identity of given shape."""
        ones = torch.ones(shape, dtype=torch.float32, device=device)
        zeros = torch.zeros(shape, dtype=torch.float32, device=device)
        return cls(ones, zeros, zeros, zeros, ones, zeros, zeros, zeros, ones)

    @classmethod
    def from_two_vectors(cls, e0: vector.Vec3Array, e1: vector.Vec3Array) -> Rot3Array:
        """Construct Rot3Array from two Vectors.

        Rot3Array is constructed such that in the corresponding frame 'e0' lies on
        the positive x-Axis and 'e1' lies in the xy plane with positive sign of y.

        Args:
            e0: Vector
            e1: Vector
        Returns:
            Rot3Array
        """
        # Normalize the unit vector for the x-axis, e0.
        e0 = e0.normalized()
        # make e1 perpendicular to e0.
        c = e1.dot(e0)
        e1 = (e1 - c * e0).normalized()
        # Compute e2 as cross product of e0 and e1.
        e2 = e0.cross(e1)
        return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)

    @classmethod
    def from_array(cls, array: torch.Tensor) -> Rot3Array:
        """Construct Rot3Array Matrix from array of shape. [..., 3, 3]."""
        rows = torch.unbind(array, dim=-2)
        rc = [torch.unbind(e, dim=-1) for e in rows]
        return cls(*[e for row in rc for e in row])

    def to_tensor(self) -> torch.Tensor:
        """Convert Rot3Array to array of shape [..., 3, 3]."""
        return torch.stack([
            torch.stack([self.xx, self.xy, self.xz], dim=-1),
            torch.stack([self.yx, self.yy, self.yz], dim=-1),
            torch.stack([self.zx, self.zy, self.zz], dim=-1)
        ],
                           dim=-2)

    @classmethod
    def from_quaternion(cls,
                        w: torch.Tensor,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        z: torch.Tensor,
                        normalize: bool = True,
                        eps: float = 1e-6) -> Rot3Array:
        """Construct Rot3Array from components of quaternion."""
        if normalize:
            inv_norm = torch.rsqrt(eps + w**2 + x**2 + y**2 + z**2)
            w *= inv_norm
            x *= inv_norm
            y *= inv_norm
            z *= inv_norm
        xx = 1 - 2 * (y**2 + z**2)
        xy = 2 * (x * y - w * z)
        xz = 2 * (x * z + w * y)
        yx = 2 * (x * y + w * z)
        yy = 1 - 2 * (x**2 + z**2)
        yz = 2 * (y * z - w * x)
        zx = 2 * (x * z - w * y)
        zy = 2 * (y * z + w * x)
        zz = 1 - 2 * (x**2 + y**2)
        return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)

    def reshape(self, new_shape):
        field_names = utils.get_field_names(Rot3Array)
        reshape_fn = lambda t: t.reshape(new_shape)
        return Rot3Array(**{name: reshape_fn(getattr(self, name)) for name in field_names})

    @classmethod
    def cat(cls, rots: List[Rot3Array], dim: int) -> Rot3Array:
        field_names = utils.get_field_names(Rot3Array)
        cat_fn = lambda l: torch.cat(l, dim=dim)
        return cls(**{name: cat_fn([getattr(r, name) for r in rots]) for name in field_names})
