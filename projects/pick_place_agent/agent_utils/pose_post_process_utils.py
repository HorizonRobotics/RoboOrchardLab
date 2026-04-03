# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import copy

import numpy as np


class PosePostProcess:
    def __init__(self, camera_info: dict):
        self.camera_extrinsic = camera_info["extrinsic"]
        camera_instrinsics = camera_info["intrinsic"]
        self.fx, self.fy = camera_instrinsics["fx"], camera_instrinsics["fy"]
        self.cx, self.cy = camera_instrinsics["cx"], camera_instrinsics["cy"]
        self.scale = camera_instrinsics["scale"]
        self.workspace_limits = camera_info["workspace_limits"]

    def get_grasp_poses_img(self, grasp_poses_from_model):
        """Get grasp poses in the image coordinate system.

        Converts grasp poses from the model infer result
        to 4x4 homogeneous matrices in the image coordinate system.

        Args:
            grasp_poses_from_model (list of GraspGroup):
                List of grasp poses from the model output.

        Returns:
            list: List of 4x4 homogeneous matrices representing
            the grasp poses in the image coordinate system.
        """
        grasp_poses_img_homo = []
        for grasp_pose_from_model in grasp_poses_from_model:
            gg_pick_homo = np.eye(4)
            gg_pick_homo[:3, :3] = grasp_pose_from_model.rotation_matrix
            gg_pick_homo[:3, 3] = grasp_pose_from_model.translation
            grasp_poses_img_homo.append(gg_pick_homo)
        return grasp_poses_img_homo

    def grasp_poses_to_world(self, grasp_poses_img):
        """Get grasp poses in the world coordinate system.

        Converts grasp poses from image coordinate to world coordinate.

        Args:
            grasp_poses_img: list of grasp poses in the image coordinate

        Returns:
            list: List of 4x4 homogeneous transformation matrices representing
            the grasp poses in the world coordinate system.
        """
        grasp_poses_world_homo = []
        for grasp_pose_img in grasp_poses_img:
            grasp_pose_world_homo = (
                np.linalg.inv(self.camera_extrinsic) @ grasp_pose_img
            )  # get from robotwin, which is world2cam
            grasp_poses_world_homo.append(grasp_pose_world_homo)
        return grasp_poses_world_homo

    def workspace_filter(self, poses, gg_picks):
        """Filter grasp poses by workspace limits.

        Filters a list of pose objects based on whether their positions
        fall within the workspace, limits defined in the configuration.

        Args:
            poses (list): List of 4x4 homogeneous matrices representing poses
            gg_picks(GraspGroup): List of model infer result.

        Returns:
            tuple:
                filtered_poses (list):
                    List of pose matrices within the workspace limits.
                filtered_gg_picks(GraspGroup):
                    List of gg_picks after removed the poses which
                    do not meet the filtering criteria.
        """
        filtered_poses = []
        remove_indices = []
        for idx, pose in enumerate(poses):
            x, y, z = pose[:3, 3]
            if (
                self.workspace_limits["x_min"]
                <= x
                <= self.workspace_limits["x_max"]
                and self.workspace_limits["y_min"]
                <= y
                <= self.workspace_limits["y_max"]
                and self.workspace_limits["z_min"]
                <= z
                <= self.workspace_limits["z_max"]
            ):
                filtered_poses.append(pose)
            else:
                remove_indices.append(idx)
        filtered_gg_picks = gg_picks.remove(remove_indices)
        return filtered_poses, filtered_gg_picks

    def get_grounding_point(self, grounding_depth):
        xmap, ymap = (
            np.arange(grounding_depth.shape[1]),
            np.arange(grounding_depth.shape[0]),
        )
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = grounding_depth / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        mask = (points_z > 0) & (points_z < 1)
        grounding_points = np.stack([points_x, points_y, points_z], axis=-1)
        grounding_points_img = grounding_points[mask].astype(np.float32)

        cam2world = np.linalg.inv(
            self.camera_extrinsic
        )  # get from robotwin, which is world2cam
        grounding_points_world = (
            grounding_points_img @ cam2world[:3, :3].T + cam2world[:3, 3]
        )

        return grounding_points_world

    def grounding_filter(self, poses, gg_picks, grounding_depth):
        """Filter grasp poses by grounding depth.

        Filters a list of pose objects based on
        whether their positions fall within the 3D bounds
        defined by the provided grounding depth image.

        The function computes the 3D point cloud from
        the depth image using camera intrinsics and extrinsics,
        determines the minimum and maximum bounds of the point cloud,
        and removes any poses whose positions are outside these bounds.

        Args:
            poses (list): List of 4x4 homogeneous matrices representing poses.
            gg_picks (GraspGroup): List of model infer result.
            grounding_depth (np.ndarray):
                Depth image used to compute the 3D point cloud for filtering.

        Returns:
            tuple:
                filtered_poses (list):
                    List of pose matrices within the bounds
                    defined by the grounding depth.
                filtered_gg_picks:
                    List of gg_picks after removed the poses which
                    do not meet the filtering criteria.
        """
        grounding_points_world = self.get_grounding_point(grounding_depth)
        min_bound = grounding_points_world.min(axis=0)
        max_bound = grounding_points_world.max(axis=0)
        filtered_poses = []
        removed_indices = []

        for idx, pose in enumerate(poses):
            if np.all(pose[:3, 3] >= min_bound) and np.all(
                pose[:3, 3] <= max_bound
            ):
                filtered_poses.append(pose)
            else:
                removed_indices.append(idx)
        filtered_gg_picks = copy.deepcopy(gg_picks).remove(removed_indices)
        return filtered_poses, filtered_gg_picks

    def pose_reserve(self, poses, max_angle_deg=10):
        """Reserves the orientation of poses.

        Adjusts the orientation of poses to avoid excessive
        deviation between the gripper's initial pose and the world frame.
        The gripper coordinate system is defined as:
            - x-axis: grasp direction (forward)
            - y-axis: left
            - z-axis: up
        The world coordinate system is defined as:
            - x-axis: right
            - y-axis: forward
            - z-axis: up
        For each pose in the input list, if the angle
        between the pose's local y-axis and the world x-axis
        exceeds max_angle_deg, the pose's x-axis (grasp direction)
        is rotated 180 degrees counterclockwise.

        Args:
            poses (list): List of 4x4 homogeneous matrices representing poses.
            max_angle_deg (float, optional):
                The maximum allowed angle (in degrees) between
                the pose's y-axis and the world x-axis. Default is 10.

        Returns:
            list: List of pose matrices after adjustment.
        """
        for pose in poses:
            pose_y = pose[:3, :3][:, 1]
            world_x = np.array([1, 0, 0])  # world x-axis

            if np.dot(pose_y, world_x) > np.cos(np.deg2rad(max_angle_deg)):
                r_x_180 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
                new_rotation_matrix = pose[:3, :3] @ r_x_180
                new_homo = np.eye(4)
                new_homo[:3, :3] = new_rotation_matrix
                new_homo[:3, 3] = pose[:3, 3]
                pose = new_homo
        return poses

    def top_down_filter(self, poses, gg_picks, max_angle_deg=30):
        """Filter grasp poses by top-down orientation.

        Filters grasp poses to retain only those that are
         "top-down" according to the gripper and world coordinate systems.
        The gripper coordinate system is defined as:
            - x-axis: grasp direction (forward)
            - y-axis: left
            - z-axis: up
        The world coordinate system is defined as:
            - x-axis: right
            - y-axis: forward
            - z-axis: up

        Args:
            poses(list): List of 4x4 homogeneous matrices representing poses.
            gg_picks(GraspGroup):
                list of grasp poses from model output, which is GraspGroup.
            max_angle_deg(float, optional)
                The maximum allowed angle (in degrees)
                between the pose's y-axis and the world x-axis. Default is 30.

        Returns:
            tuple:
                filtered_poses (list):
                    List of pose matrices that are top-down.
                filtered_gg_picks(GraspGroup):
                    List of gg_picks after removed the poses which
                    do not meet the filtering criteria.
        """
        filtered_poses = []
        remove_indices = []
        for idx, pose in enumerate(poses):
            pose_x = pose[:3, :3][
                :, 0
            ]  # x-axis of the pose, which is the grasp direction
            pose_y = pose[:3, :3][:, 1]
            world_x = np.array([1, 0, 0])  # world x-axis
            world_z = np.array([0, 0, 1])  # world z-axis

            if np.dot(pose_y, world_x) > np.cos(np.deg2rad(30)):
                r_x_180 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
                new_rotation_matrix = pose[:3, :3] @ r_x_180
                new_homo = np.eye(4)
                new_homo[:3, :3] = new_rotation_matrix
                new_homo[:3, 3] = pose[:3, 3]
                pose = new_homo

            if np.dot(pose_x, -world_z) > np.cos(np.deg2rad(max_angle_deg)):
                filtered_poses.append(pose)
            else:
                remove_indices.append(idx)

        filtered_gg_picks = copy.deepcopy(gg_picks).remove(remove_indices)
        return filtered_poses, filtered_gg_picks

    def horizontal_filter(self, poses, gg_picks, max_angle_deg):
        """Filter grasp poses by horizontal orientation.

        Filters grasp poses to retain only those that are
         "horizontal" according to the gripper and world coordinate systems.
        The gripper coordinate system is defined as:
            - x-axis: grasp direction (forward)
            - y-axis: left
            - z-axis: up
        The world coordinate system is defined as:
            - x-axis: right
            - y-axis: forward
            - z-axis: up

        Args:
            poses(list): List of 4x4 homogeneous matrices representing poses.
            gg_picks(GraspGroup):
                list of grasp poses from model output, which is GraspGroup.
            max_angle_deg(float, optional)
                The maximum allowed angle (in degrees)
                between the pose's x-axis and the world z-axis. Default is 30.

        Returns:
            tuple:
                filtered_poses (list):
                    List of pose matrices that are top-down.
                filtered_gg_picks(GraspGroup):
                    List of gg_picks after removed the poses which
                    do not meet the filtering criteria.
        """
        filtered_poses = []
        remove_indices = []
        for idx, pose in enumerate(poses):
            pose_x = pose[:3, :3][
                :, 0
            ]  # x-axis of the pose, which is the grasp direction
            world_z = np.array([0, 0, 1])
            if np.dot(pose_x, world_z) <= np.sin(np.deg2rad(max_angle_deg)):
                filtered_poses.append(pose)
            else:
                remove_indices.append(idx)
        filtered_gg_picks = gg_picks.remove(remove_indices)
        return filtered_poses, filtered_gg_picks

    def pose_retract(self, poses, retract_length):
        """Retract grasp poses.

        Retracts the translation of each pose by a specified length
        along the pose's local x-axis(grasp direction).

        Args:
            poses(list): List of 4x4 homogeneous matrices representing poses.
            retract_length (float):
                The length by which to retract each pose's translation.

        Returns:
            list: List of retracted pose matrices.
        """  # noqa: E501
        for pose in poses:
            rotation_matrix = pose[:3, :3]
            local_direction = np.array([1, 0, 0])
            global_direction = rotation_matrix @ local_direction
            retract_vector = -retract_length * global_direction
            retracted_translation = pose[:3, 3] + retract_vector
            pose[:3, 3] = retracted_translation

        return poses
