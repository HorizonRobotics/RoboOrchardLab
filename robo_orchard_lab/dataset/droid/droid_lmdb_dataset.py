# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

import logging
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__file__)


def xyzrpy_to_matrix(input):
    output = np.eye(4)
    output[:3, :3] = Rotation.from_euler(
        "xyz", np.array(input[3:])
    ).as_matrix()
    output[:3, 3] = input[:3]
    return output


class DroidActionLmdbDataset(BaseLmdbManipulationDataset):
    """Droid LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├──uuid: str
            ├──num_steps: int
            └──success: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └──{cam_name}: [x, y, z, r, p, y]
        {uuid}/intrinsic
            ├──{cam_name}:
            └──fx, fy, cx, cy, disto, d_fov, h_fov, v_fov
        {uuid}/instructions: list(str)
        {uuid}/action/cartesian_position: (T, 6)
        {uuid}/action/cartesian_velocity: (T, 6)
        {uuid}/action/gripper_position: (T,)
        {uuid}/action/gripper_velocity: (T,)
        {uuid}/action/joint_position: (T, 7)
        {uuid}/action/joint_velocity: (T, 7)
        {uuid}/action/robot_state/cartesian_position
        {uuid}/action/robot_state/gripper_position
        {uuid}/action/robot_state/joint_positions
        {uuid}/action/robot_state/joint_torques_computed
        {uuid}/action/robot_state/joint_velocities
        {uuid}/action/robot_state/motor_torques_measured
        {uuid}/action/robot_state/prev_command_successful
        {uuid}/action/robot_state/prev_controller_latency_ms
        {uuid}/action/robot_state/prev_joint_torques_computed
        {uuid}/action/robot_state/prev_joint_torques_computed_safened
        {uuid}/action/target_cartesian_position
        {uuid}/action/target_gripper_position
        {uuid}/observation/camera_type/{cam_name}
        {uuid}/observation/controller_info/controller_on
        {uuid}/observation/controller_info/failure
        {uuid}/observation/controller_info/movement_enabled
        {uuid}/observation/controller_info/success
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/gripper_position
        {uuid}/observation/robot_state/joint_positions
        {uuid}/observation/robot_state/joint_torques_computed
        {uuid}/observation/robot_state/joint_velocities
        {uuid}/observation/robot_state/motor_torques_measured
        {uuid}/observation/robot_state/prev_command_successful
        {uuid}/observation/robot_state/prev_controller_latency_ms
        {uuid}/observation/robot_state/prev_joint_torques_computed
        {uuid}/observation/robot_state/prev_joint_torques_computed_safened
        {uuid}/observation/timestamp/cameras/{cam_name}_estimated_capture
        {uuid}/observation/timestamp/cameras/{cam_name}_frame_received
        {uuid}/observation/timestamp/cameras/{cam_name}_read_end
        {uuid}/observation/timestamp/cameras/{cam_name}_read_start
        {uuid}/observation/timestamp/control/control_start
        {uuid}/observation/timestamp/control/policy_start
        {uuid}/observation/timestamp/control/sleep_start
        {uuid}/observation/timestamp/control/step_end
        {uuid}/observation/timestamp/control/step_start
        {uuid}/observation/timestamp/robot_state/read_end
        {uuid}/observation/timestamp/robot_state/read_start
        {uuid}/observation/timestamp/robot_state/robot_timestamp_nanos
        {uuid}/observation/timestamp/robot_state/robot_timestamp_seconds
        {uuid}/observation/timestamp/skip_action

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        lazy_init=False,
        cam_names=None,
        load_ee_state=False,
        min_num_step=0,
        max_num_step=1e5,
        **kwargs,
    ):
        self.min_num_step = min_num_step
        self.max_num_step = max_num_step
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=False,
            lazy_init=lazy_init,
            **kwargs,
        )
        self.cam_names = cam_names
        self.load_ee_state = load_ee_state

    def _check_valid(self, index_data):
        if (
            index_data.num_steps > self.max_num_step
            or index_data.num_steps < self.min_num_step
        ):
            return False
        return super()._check_valid(index_data)

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = self.idx_lmdbs[lmdb_index][episode_index]
        uuid = idx_data["uuid"]
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        extrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
        intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        images = []
        extrinsic_mat = []
        intrinsic_mat = []
        for cam_name in cam_names:
            image_buffer = self.img_lmdbs[lmdb_index][
                f"{uuid}/{cam_name}/{step_index}"
            ]
            if image_buffer is None:
                logger.info(f"invalid data at index {index}")
                index = np.random.randint(self.__len__())
                return self[index]
            image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
            images.append(image)

            extrinsic_mat.append(
                xyzrpy_to_matrix(extrinsic[cam_name][step_index])
            )
            intrinsic_mat.append(
                np.array(
                    [
                        [
                            intrinsic[cam_name]["fx"],
                            0,
                            intrinsic[cam_name]["cx"],
                            0,
                        ],
                        [
                            0,
                            intrinsic[cam_name]["fy"],
                            intrinsic[cam_name]["cy"],
                            0,
                        ],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            )
        t_world2cam = np.linalg.inv(np.stack(extrinsic_mat))
        intrinsic = np.stack(intrinsic_mat)

        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        gripper_position = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/gripper_position"
        ]
        joint_state = np.concatenate(
            [joint_state, gripper_position[:, None]], axis=1
        )
        skip_action = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/timestamp/skip_action"
        ]
        skip_action[: step_index + 1] = False
        keep_action = np.logical_not(skip_action)
        joint_state = joint_state[keep_action]

        data = dict(
            uuid=uuid,
            cam_names=cam_names,
            step_index=step_index,
            imgs=images,
            depths=[x[..., 0] * 0 for x in images],  # fake depth
            intrinsic=intrinsic,
            T_world2cam=t_world2cam,
            joint_state=joint_state,
        )
        if self.load_ee_state:
            ee_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/cartesian_position"
            ]
            ee_state = ee_state[keep_action]
            data["ee_state"] = ee_state

        self.get_instruction(lmdb_index, data)
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def get_instruction(self, lmdb_index, data):
        instructions = self.meta_lmdbs[lmdb_index][
            f"{data['uuid']}/instructions"
        ]
        if len(instructions) == 0:
            text = ""
        else:
            idx = np.random.randint(len(instructions))
            text = instructions[idx]
        data["text"] = text

    def visualize(
        self,
        episode_index,
        output_path="./vis_data",
        fps=25,
        interval=1,
        **kwargs,
    ):
        from tqdm import tqdm

        end_idx = self.cumsum_steps[episode_index]
        if episode_index != 0:
            start_idx = self.cumsum_steps[episode_index - 1]
        else:
            start_idx = 0
        videoWriter = None  # noqa: N806
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        uuid = self.__getitem__(start_idx)["uuid"]
        file = os.path.join(output_path, f"{uuid.replace('/', '-')}.mp4")

        logger.info(f"episode start_idx: {start_idx}, end_idx: {end_idx}")
        logger.info(f"video save path: {file}")

        for i in tqdm(list(range(start_idx, end_idx, interval))):
            data = self.__getitem__(i)
            if i == start_idx:
                logger.info(
                    f"instruction: {data.get('text')}, "
                    f"subtask: {data.get('subtask')}"
                )

            vis_imgs = self.get_vis_imgs(
                data["imgs"],
                data.get("projection_mat"),
                data.get("hist_robot_state", [None])[-1],
                **kwargs,
            )

            if videoWriter is None:
                videoWriter = cv2.VideoWriter(  # noqa: N806
                    file,
                    fourcc,
                    fps,
                    vis_imgs.shape[:2][::-1],
                )
            videoWriter.write(vis_imgs)
        videoWriter.release()

    @staticmethod
    def get_vis_imgs(
        imgs,
        projection_mat=None,
        robot_state=None,
        ee_indices=(7,),
        channel_conversion=False,
    ):
        from scipy.spatial.transform import Rotation

        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        if isinstance(projection_mat, torch.Tensor):
            projection_mat = projection_mat.cpu().numpy()
        if isinstance(robot_state, torch.Tensor):
            robot_state = robot_state.cpu().numpy()

        vis_imgs = []
        for img_index in range(imgs.shape[0]):
            img = imgs[img_index]
            if robot_state is None or projection_mat is None:
                vis_imgs.append(img)
                continue
            for joint_index in range(robot_state.shape[0]):
                rot = Rotation.from_quat(
                    robot_state[joint_index, 4:], scalar_first=True
                ).as_matrix()
                trans = robot_state[joint_index, 1:4]

                if joint_index in ee_indices:
                    axis_length = 0.1
                else:
                    axis_length = 0.03
                points = np.float32(
                    [
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length],
                        [0, 0, 0],
                    ]
                )
                points = points @ rot.T + trans

                pts_2d = points @ projection_mat[img_index, :3, :3].T
                pts_2d = pts_2d + projection_mat[img_index, :3, 3]
                depth = pts_2d[:, 2]
                pts_2d = pts_2d[:, :2] / depth[:, None]

                if depth[3] < 0.02:
                    continue

                pts_2d = pts_2d.astype(np.int32)
                for i in range(3):
                    if depth[i] < 0.02:
                        continue
                    cv2.circle(
                        img, (pts_2d[i, 0], pts_2d[i, 1]), 6, (0, 0, 255), -1
                    )
                    if i == 3:
                        continue
                    color = [0, 0, 0]
                    color[i] = 255
                    cv2.line(
                        img,
                        (pts_2d[i, 0], pts_2d[i, 1]),
                        (pts_2d[3, 0], pts_2d[3, 1]),
                        tuple(color),
                        3,
                    )
            vis_imgs.append(img)

        if len(vis_imgs) % 2 == 0:
            num_imgs = len(vis_imgs)
            vis_imgs = np.concatenate(
                [
                    np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                    np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
                ],
                axis=0,
            )
        else:
            vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs
