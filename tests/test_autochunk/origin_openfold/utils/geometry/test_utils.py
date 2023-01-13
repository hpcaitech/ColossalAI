# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared utils for tests."""

import dataclasses

import numpy as np
from fastfold.utils.geometry import rigid_matrix_vector, rotation_matrix, vector


def assert_rotation_matrix_equal(matrix1: rotation_matrix.Rot3Array, matrix2: rotation_matrix.Rot3Array):
    for field in dataclasses.fields(rotation_matrix.Rot3Array):
        field = field.name
        np.testing.assert_array_equal(getattr(matrix1, field), getattr(matrix2, field))


def assert_rotation_matrix_close(mat1: rotation_matrix.Rot3Array, mat2: rotation_matrix.Rot3Array):
    np.testing.assert_array_almost_equal(mat1.to_array(), mat2.to_array(), 6)


def assert_array_equal_to_rotation_matrix(array: np.ndarray, matrix: rotation_matrix.Rot3Array):
    """Check that array and Matrix match."""
    np.testing.assert_array_equal(matrix.xx, array[..., 0, 0])
    np.testing.assert_array_equal(matrix.xy, array[..., 0, 1])
    np.testing.assert_array_equal(matrix.xz, array[..., 0, 2])
    np.testing.assert_array_equal(matrix.yx, array[..., 1, 0])
    np.testing.assert_array_equal(matrix.yy, array[..., 1, 1])
    np.testing.assert_array_equal(matrix.yz, array[..., 1, 2])
    np.testing.assert_array_equal(matrix.zx, array[..., 2, 0])
    np.testing.assert_array_equal(matrix.zy, array[..., 2, 1])
    np.testing.assert_array_equal(matrix.zz, array[..., 2, 2])


def assert_array_close_to_rotation_matrix(array: np.ndarray, matrix: rotation_matrix.Rot3Array):
    np.testing.assert_array_almost_equal(matrix.to_array(), array, 6)


def assert_vectors_equal(vec1: vector.Vec3Array, vec2: vector.Vec3Array):
    np.testing.assert_array_equal(vec1.x, vec2.x)
    np.testing.assert_array_equal(vec1.y, vec2.y)
    np.testing.assert_array_equal(vec1.z, vec2.z)


def assert_vectors_close(vec1: vector.Vec3Array, vec2: vector.Vec3Array):
    np.testing.assert_allclose(vec1.x, vec2.x, atol=1e-6, rtol=0.)
    np.testing.assert_allclose(vec1.y, vec2.y, atol=1e-6, rtol=0.)
    np.testing.assert_allclose(vec1.z, vec2.z, atol=1e-6, rtol=0.)


def assert_array_close_to_vector(array: np.ndarray, vec: vector.Vec3Array):
    np.testing.assert_allclose(vec.to_array(), array, atol=1e-6, rtol=0.)


def assert_array_equal_to_vector(array: np.ndarray, vec: vector.Vec3Array):
    np.testing.assert_array_equal(vec.to_array(), array)


def assert_rigid_equal_to_rigid(rigid1: rigid_matrix_vector.Rigid3Array, rigid2: rigid_matrix_vector.Rigid3Array):
    assert_rot_trans_equal_to_rigid(rigid1.rotation, rigid1.translation, rigid2)


def assert_rigid_close_to_rigid(rigid1: rigid_matrix_vector.Rigid3Array, rigid2: rigid_matrix_vector.Rigid3Array):
    assert_rot_trans_close_to_rigid(rigid1.rotation, rigid1.translation, rigid2)


def assert_rot_trans_equal_to_rigid(rot: rotation_matrix.Rot3Array, trans: vector.Vec3Array,
                                    rigid: rigid_matrix_vector.Rigid3Array):
    assert_rotation_matrix_equal(rot, rigid.rotation)
    assert_vectors_equal(trans, rigid.translation)


def assert_rot_trans_close_to_rigid(rot: rotation_matrix.Rot3Array, trans: vector.Vec3Array,
                                    rigid: rigid_matrix_vector.Rigid3Array):
    assert_rotation_matrix_close(rot, rigid.rotation)
    assert_vectors_close(trans, rigid.translation)
