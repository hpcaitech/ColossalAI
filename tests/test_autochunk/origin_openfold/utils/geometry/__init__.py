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
"""Geometry Module."""

from fastfold.utils.geometry import rigid_matrix_vector, rotation_matrix, vector

Rot3Array = rotation_matrix.Rot3Array
Rigid3Array = rigid_matrix_vector.Rigid3Array

Vec3Array = vector.Vec3Array
square_euclidean_distance = vector.square_euclidean_distance
euclidean_distance = vector.euclidean_distance
dihedral_angle = vector.dihedral_angle
dot = vector.dot
cross = vector.cross
