#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import numpy as np
from pyquaternion import Quaternion

"""
In general, 4 different coordinate systems are used with 3 of them are described
in https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf
 1. The vehicle coordinate system V according to ISO 8855 with the origin
    on the ground below of the rear axis center, x pointing in driving direction,
    y pointing left, and z pointing up.
 2. The camera coordinate system C with the origin in the camera’s optical
    center and same orientation as V.
 3. The image coordinate system I with the origin in the top-left image pixel,
    u pointing right, and v pointing down.
 4. In addition, we also add the coordinate system S with the same origin as C,
    but the orientation of I, ie. x pointing right, y down, and z in the
    driving direction.

All GT annotations are given in the ISO coordinate system V and hence, the
evaluation requires the data to be available in this coordinate system.

For V and C it is:                   For S and I it is:

                    ^                         ^
                  z |    ^                   / z/d
                    |   / x                 /
                    |  /                   /
                    | /                   +------------>
                    |/                    |         x/u
       <------------+                     |
         y                                |
                                          | y/v
                                          V
"""


# Define different coordinate systems
CRS_V = 0
CRS_C = 1
CRS_S = 2


def get_K_multiplier():
    K_multiplier = np.zeros((3, 3))
    K_multiplier[0][1] = K_multiplier[1][2] = -1
    K_multiplier[2][0] = 1
    return K_multiplier


def get_projection_matrix(camera):
    K_matrix = np.zeros((3, 3))
    K_matrix[0][0] = camera.fx
    K_matrix[0][2] = camera.u0
    K_matrix[1][1] = camera.fy
    K_matrix[1][2] = camera.v0
    K_matrix[2][2] = 1
    return K_matrix


def apply_transformation_points(points, transformation_matrix):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.matmul(transformation_matrix, points.T).T
    return points


class Camera(object):
    def __init__(
        self,
        fx,
        fy,
        u0,
        v0,
        sensor_T_ISO_8855,
        imgWidth=2048,
        imgHeight=1024):
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.sensor_T_ISO_8855 = sensor_T_ISO_8855
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight


class Box3dImageTransform(object):
    def __init__(self, camera):
        self._camera = camera
        self._rotation_matrix = np.zeros((3, 3))
        self._size = np.zeros((3,))
        self._center = np.zeros((3,))

        self.loc = ["BLB", "BRB", "FRB", "FLB", "BLT", "BRT", "FRT", "FLT"]

        self._box_points_2d = np.zeros((8, 2))
        self._box_points_3d_vehicle = np.zeros((8, 3))
        self._box_points_3d_cam = np.zeros((8, 3))

        self.bottom_arrow_2d = np.zeros((2, 2))
        self._bottom_arrow_3d_vehicle = np.zeros((2, 3))
        self._bottom_arrow_3d_cam = np.zeros((2, 3))

        self._box_left_side_cropped_2d = []
        self._box_right_side_cropped_2d = []
        self._box_front_side_cropped_2d = []
        self._box_back_side_cropped_2d = []
        self._box_top_side_cropped_2d = []
        self._box_bottom_side_cropped_2d = []

    def initialize_box_from_annotation(self, csBbox3dAnnotation, coordinate_system=CRS_V):
        # Unpack annotation and call initialize_box() method
        self.initialize_box(
            csBbox3dAnnotation.dims,
            csBbox3dAnnotation.rotation,
            csBbox3dAnnotation.center,
            coordinate_system=coordinate_system
        )

    def initialize_box(self, size, quaternion, center, coordinate_system=CRS_V):
        # Internally, the box is always stored in the ISO 8855 coordinate system V
        # If the box is passed with another coordinate system, we transform it to V first.
        # "size" is always given in LxWxH
        K_multiplier = get_K_multiplier()
        quaternion_rot = Quaternion(quaternion)
        center = np.array(center)

        if coordinate_system == CRS_S:  # convert it to CRS_C first
            center = np.matmul(K_multiplier.T, center.T).T
            image_T_sensor_quaternion = Quaternion(matrix=K_multiplier)
            quaternion_rot = (
                image_T_sensor_quaternion.inverse *
                quaternion_rot *
                image_T_sensor_quaternion
            )

        # center and quaternion must be corrected
        if coordinate_system == CRS_C or coordinate_system == CRS_S:
            sensor_T_ISO_8855_4x4 = np.eye(4)
            sensor_T_ISO_8855_4x4[:3, :] = np.array(self._camera.sensor_T_ISO_8855)
            sensor_T_ISO_8855_4x4_inv = np.linalg.inv(sensor_T_ISO_8855_4x4)
            center_T = np.ones((4, 1))
            center_T[:3, 0] = center.T
            center = np.matmul(sensor_T_ISO_8855_4x4_inv, center_T)
            center = (center.T)[0, :3]

            sensor_T_ISO_8855_quaternion = Quaternion(
                matrix=np.array(self._camera.sensor_T_ISO_8855)[:3, :3])
            quaternion_rot = sensor_T_ISO_8855_quaternion.inverse * quaternion_rot

        self._size = np.array(size)
        self._rotation_matrix = np.array(quaternion_rot.rotation_matrix)
        self._center = center

        self.update()

    def get_vertices(self, coordinate_system=CRS_V):
        if coordinate_system == CRS_V:
            box_points_3d = self._box_points_3d_vehicle

        if coordinate_system == CRS_C or coordinate_system == CRS_S:
            box_points_3d = apply_transformation_points(
                self._box_points_3d_vehicle, self._camera.sensor_T_ISO_8855
            )

        if coordinate_system == CRS_S:
            K_multiplier = get_K_multiplier()
            box_points_3d = np.matmul(K_multiplier, box_points_3d.T).T

        return {l: p for (l, p) in zip(self.loc, box_points_3d)}

    def get_vertices_2d(self):
        return {l: p for (l, p) in zip(self.loc, self._box_points_2d)}

    def get_parameters(self, coordinate_system=CRS_V):
        K_multiplier = get_K_multiplier()
        quaternion_rot = Quaternion(matrix=self._rotation_matrix)
        center = self._center

        # center and quaternion must be corrected
        if coordinate_system == CRS_C or coordinate_system == CRS_S:
            sensor_T_ISO_8855_4x4 = np.eye(4)
            sensor_T_ISO_8855_4x4[:3, :] = np.array(self._camera.sensor_T_ISO_8855)
            center_T = np.ones((4, 1))
            center_T[:3, 0] = center.T
            center = np.matmul(sensor_T_ISO_8855_4x4, center_T)
            center = (center.T)[0, :3]
            sensor_T_ISO_8855_quaternion = Quaternion(
                matrix=np.array(self._camera.sensor_T_ISO_8855)[:3, :3]
            )
            quaternion_rot = sensor_T_ISO_8855_quaternion * quaternion_rot

        # change axis
        if coordinate_system == CRS_S:
            center = np.matmul(K_multiplier, center.T).T
            image_T_sensor_quaternion = Quaternion(matrix=K_multiplier)
            quaternion_rot = (
                image_T_sensor_quaternion *
                quaternion_rot *
                image_T_sensor_quaternion.inverse
            )

        return (self._size, center, quaternion_rot)

    def _get_side_visibility(self, face_center, face_normal):
        return np.dot(face_normal, face_center) < 0

    def get_all_side_visibilities(self):
        K_multiplier = get_K_multiplier()
        rotation_matrix_cam = np.matmul(
            np.matmul(K_multiplier, self._rotation_matrix), K_multiplier.T
        )

        box_vector_x = rotation_matrix_cam[:, 0]
        box_vector_y = rotation_matrix_cam[:, 1]
        box_vector_z = rotation_matrix_cam[:, 2]

        front_visible = self._get_side_visibility(
            (self._box_points_3d_cam[3] + self._box_points_3d_cam[6]) / 2, box_vector_z
        )
        back_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[5]) / 2, -box_vector_z
        )
        top_visible = self._get_side_visibility(
            (self._box_points_3d_cam[7] + self._box_points_3d_cam[5]) / 2, -box_vector_y
        )
        bottom_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[2]) / 2, box_vector_y
        )
        left_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[7]) / 2, -box_vector_x
        )
        right_visible = self._get_side_visibility(
            (self._box_points_3d_cam[1] + self._box_points_3d_cam[6]) / 2, box_vector_x
        )

        return [
            front_visible,
            back_visible,
            top_visible,
            bottom_visible,
            left_visible,
            right_visible,
        ]

    def get_all_side_polygons_2d(self):
        front_side = self._box_front_side_cropped_2d
        back_side = self._box_back_side_cropped_2d
        top_side = self._box_top_side_cropped_2d
        bottom_side = self._box_bottom_side_cropped_2d
        left_side = self._box_left_side_cropped_2d
        right_side = self._box_right_side_cropped_2d

        return [front_side, back_side, top_side, bottom_side, left_side, right_side]

    def get_amodal_box_2d(self):
        xs = []
        ys = []

        for side_polygon in self.get_all_side_polygons_2d():
            for [x, y] in side_polygon:
                xs.append(x)
                ys.append(y)

        # if the whole box is behind the camera, return [0., 0., 0., 0.]
        if len(xs) == 0:
            return [0., 0., 0., 0.]

        return [
            min(self._camera.imgWidth - 1, max(0, min(xs))),
            min(self._camera.imgHeight - 1, max(0, min(ys))),
            min(self._camera.imgWidth - 1, max(0, max(xs))),
            min(self._camera.imgHeight - 1, max(0, max(ys)))
        ]

    def _crop_side_polygon_and_project(self, side_point_indices=[], side_points=[]):
        K_matrix = get_projection_matrix(self._camera)
        camera_plane_z = 0.01

        side_points_3d_cam = [self._box_points_3d_cam[i] for i in side_point_indices]
        side_points_3d_cam += side_points

        cropped_polygon_3d = []
        for i, point in enumerate(side_points_3d_cam):
            if point[2] > camera_plane_z:  # 1 cm
                cropped_polygon_3d.append(point)
            else:
                next_index = (i + 1) % len(side_points_3d_cam)
                prev_index = i - 1

                if side_points_3d_cam[prev_index][2] > camera_plane_z:
                    delta_0 = point - side_points_3d_cam[prev_index]
                    k_0 = (camera_plane_z - point[2]) / delta_0[2]
                    point_0 = point + k_0 * delta_0
                    cropped_polygon_3d.append(point_0)

                if side_points_3d_cam[next_index][2] > camera_plane_z:
                    delta_1 = point - side_points_3d_cam[next_index]
                    k_1 = (camera_plane_z - point[2]) / delta_1[2]
                    point_1 = point + k_1 * delta_1
                    cropped_polygon_3d.append(point_1)

        if len(cropped_polygon_3d) == 0:
            cropped_polygon_2d = []
        else:
            cropped_polygon_2d = np.matmul(K_matrix, np.array(cropped_polygon_3d).T)
            cropped_polygon_2d = cropped_polygon_2d[:2, :] / cropped_polygon_2d[-1, :]
            cropped_polygon_2d = cropped_polygon_2d.T.tolist()
            cropped_polygon_2d.append(cropped_polygon_2d[0])

        return cropped_polygon_2d

    def update(self):
        self._update_box_points_3d()
        self._update_box_sides_cropped()
        self._update_box_points_2d()

    def _update_box_sides_cropped(self):
        self._box_left_side_cropped_2d = self._crop_side_polygon_and_project(
            [3, 0, 4, 7]
        )
        self._box_right_side_cropped_2d = self._crop_side_polygon_and_project(
            [1, 5, 6, 2]
        )
        self._box_front_side_cropped_2d = self._crop_side_polygon_and_project(
            [3, 2, 6, 7]
        )
        self._box_back_side_cropped_2d = self._crop_side_polygon_and_project(
            [0, 1, 5, 4]
        )
        self._box_top_side_cropped_2d = self._crop_side_polygon_and_project(
            [4, 5, 6, 7]
        )
        self._box_bottom_side_cropped_2d = self._crop_side_polygon_and_project(
            [0, 1, 2, 3]
        )
        self.bottom_arrow_2d = self._crop_side_polygon_and_project(
            side_points=[self._bottom_arrow_3d_cam[x] for x in range(2)]
        )

    def _update_box_points_3d(self):
        center_vectors = np.zeros((8, 3))
        # Bottom Face
        center_vectors[0] = np.array(
            [-self._size[0] / 2, self._size[1] / 2, -self._size[2] / 2]
            # Back Left Bottom
        )
        center_vectors[1] = np.array(
            [-self._size[0] / 2, -self._size[1] / 2, -self._size[2] / 2]
            # Back Right Bottom
        )
        center_vectors[2] = np.array(
            [self._size[0] / 2, -self._size[1] / 2, -self._size[2] / 2]
            # Front Right Bottom
        )
        center_vectors[3] = np.array(
            [self._size[0] / 2, self._size[1] / 2, -self._size[2] / 2]
            # Front Left Bottom
        )

        # Top Face
        center_vectors[4] = np.array(
            [-self._size[0] / 2, self._size[1] / 2, self._size[2] / 2]
            # Back Left Top
        )
        center_vectors[5] = np.array(
            [-self._size[0] / 2, -self._size[1] / 2, self._size[2] / 2]
            # Back Right Top
        )
        center_vectors[6] = np.array(
            [self._size[0] / 2, -self._size[1] / 2, self._size[2] / 2]
            # Front Right Top
        )
        center_vectors[7] = np.array(
            [self._size[0] / 2, self._size[1] / 2, self._size[2] / 2]
            # Front Left Top
        )

        # Rotate the vectors
        box_points_3d = np.matmul(self._rotation_matrix, center_vectors.T).T
        # Translate to box position in 3d space
        box_points_3d += self._center

        self._box_points_3d_vehicle = box_points_3d

        self._bottom_arrow_3d_vehicle = np.array(
            [
                (0.5 * (self._box_points_3d_vehicle[3] + self._box_points_3d_vehicle[2])),
                (0.5 * (self._box_points_3d_vehicle[3] + self._box_points_3d_vehicle[1])),
            ]
        )
        bottom_arrow_3d_cam = apply_transformation_points(
            self._bottom_arrow_3d_vehicle, self._camera.sensor_T_ISO_8855
        )

        # Points in ISO8855 system with origin at the sensor
        box_points_3d_cam = apply_transformation_points(
            self._box_points_3d_vehicle, self._camera.sensor_T_ISO_8855
        )
        K_multiplier = get_K_multiplier()
        self._box_points_3d_cam = np.matmul(K_multiplier, box_points_3d_cam.T).T
        self._bottom_arrow_3d_cam = np.matmul(K_multiplier, bottom_arrow_3d_cam.T).T

    def _update_box_points_2d(self):
        K_matrix = get_projection_matrix(self._camera)
        box_points_2d = np.matmul(K_matrix, self._box_points_3d_cam.T)
        box_points_2d = box_points_2d[:2, :] / box_points_2d[-1, :]
        self._box_points_2d = box_points_2d.T
