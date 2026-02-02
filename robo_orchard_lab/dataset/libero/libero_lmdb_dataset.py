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

import logging
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from robo_orchard_lab.dataset.libero.utils import (
    pose_inv,
    transform_ee_rotations,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class LiberoLmdbDataset(BaseLmdbManipulationDataset):
    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        num_episode=None,
        cam_names=None,
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            num_episode=num_episode,
            **kwargs,
        )
        self.cam_names = cam_names

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)
        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name

        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        camera_extrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
        camera_intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        T_world2cam = []  # noqa: N806
        intrinsic = []

        for cam_name in cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                image = image[::-1]
                images.append(image)

            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(depth, bytes):
                    depth = cv2.imdecode(
                        np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED
                    )
                depth = depth[::-1]
                depths.append(depth.squeeze(-1))

            if camera_intrinsic[cam_name][step_index].shape == (3, 3):
                intrinsic_ = np.eye(4)
                intrinsic_[:3, :3] = camera_intrinsic[cam_name][step_index]

            T_world2cam.append(
                pose_inv(camera_extrinsic[cam_name][step_index])
            )
            intrinsic.append(intrinsic_)

        if self.load_image:
            images = np.stack(images)
        if self.load_depth:
            depths = np.stack(depths)

        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        intrinsic = np.stack(intrinsic)

        ee_states = self.meta_lmdbs[lmdb_index][f"{uuid}/observation/ee_state"]

        r_diff = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        ee_quats = transform_ee_rotations(ee_states, r_diff)
        ee_states = np.concatenate([ee_states[:, :3], ee_quats], axis=1)
        gripper_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/gripper_state"
        ]
        current_width = gripper_state[:, 0] - gripper_state[:, 1]
        MAX_WIDTH = 0.08  # noqa: N806
        gripper_state = current_width / MAX_WIDTH
        robot_states = np.concatenate(
            [gripper_state[:, None], ee_states], axis=1
        )
        robot_states = np.expand_dims(robot_states, axis=1).astype(np.float32)

        actions = self.meta_lmdbs[lmdb_index][f"{uuid}/action"]
        arm_actions = actions[:, :6]
        rot_vecs = arm_actions[:, 3:6]
        rotations = R.from_rotvec(rot_vecs)
        arm_actions_quat = rotations.as_quat()[:, [3, 0, 1, 2]]
        actions = np.concatenate(
            [actions[:, 6:], arm_actions[:, :3], arm_actions_quat], axis=1
        )
        actions = np.expand_dims(actions, axis=1).astype(np.float32)
        actions[..., :1] = (1 - actions[..., :1]) / 2
        data = dict(
            uuid=uuid,
            step_index=step_index,
            joint_state=robot_states,
            action=actions,
            task_name=task_name,
            T_world2cam=T_world2cam,
            intrinsic=intrinsic,
            cam_names=cam_names,
        )
        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instructions = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if isinstance(instructions, str):
            text = instructions
        elif len(instructions) == 0:
            text = ""

        data["text"] = text
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def visualize(
        self,
        episode_index,
        output_path="./vis_data",
        fps=30,
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
                    fps // interval,
                    vis_imgs.shape[:2][::-1],
                )
            videoWriter.write(vis_imgs)
        videoWriter.release()

    @staticmethod
    def get_vis_imgs(
        imgs,
        projection_mat=None,
        robot_state=None,
        channel_conversion=False,
    ):
        import torch
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
                        img, (pts_2d[i, 0], pts_2d[i, 1]), 2, (0, 0, 255), -1
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
                        1,
                    )
            vis_imgs.append(img)

        vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs
