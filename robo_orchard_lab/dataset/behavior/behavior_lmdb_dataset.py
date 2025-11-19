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
import logging
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class BehaviorLmdbDataset(BaseLmdbManipulationDataset):
    """Behavior LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}

    Depth storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

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
        T_base2world=None,  # noqa: N803
        T_base2ego=None,  # noqa: N803
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
        )
        if cam_names is None:
            self.cam_names = utils.ROBOT_CAMERA_NAMES["R1Pro"]

        self.T_base2world = T_base2world
        self.T_base2ego = T_base2ego

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        # task_name = idx_data.task_name

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        _T_cam2base = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]  # noqa: N806
        _intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        T_cam2base = []  # noqa: N806
        intrinsic = []
        for cam_name in self.cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/rgb_{cam_name}/{step_index}"
                ]

                if isinstance(image, bytes):
                    image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                image = image.astype(np.float32) / 255.0
                images.append(image)

            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/depth_{cam_name}/{step_index}"
                ]

                if isinstance(depth, bytes):
                    depth = np.frombuffer(depth, np.uint16)
                depth = cv2.imdecode(
                    depth, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
                )
                depth = depth / 1000.0
                depths.append(depth)

            _tmp = np.eye(4)
            _tmp[:3, :3] = _T_cam2base[cam_name][step_index][:3, :3]
            _tmp[:3, 3] = _T_cam2base[cam_name][step_index][:3, 3]
            T_cam2base.append(_tmp)

            _tmp = np.eye(4)
            _tmp[:3, :3] = _intrinsic[cam_name][:3, :3]
            intrinsic.append(_tmp)

        T_cam2base = np.stack(T_cam2base)  # noqa: N806
        intrinsic = np.stack(intrinsic)

        action = self.meta_lmdbs[lmdb_index][f"{uuid}/action"]
        mobile_traj = action[:, :3]
        action = action[:, 3:]

        # base_qvel = self.meta_lmdbs[lmdb_index][
        #    f"{uuid}/observation/robot_state/base_qvel"
        # ]

        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        joint_state = [
            # base_qvel,
            joint_state[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["torso"]],
            joint_state[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_arm"]],
            joint_state[
                :, utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_gripper"]
            ].sum(axis=-1, keepdims=True),
            joint_state[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_arm"]],
            joint_state[
                :, utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_gripper"]
            ].sum(axis=-1, keepdims=True),
        ]
        joint_state = np.concatenate(joint_state, axis=-1)

        ee_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/cartesian_position"
        ]
        if ee_state.ndim == 3:
            ee_state = ee_state.reshape(ee_state.shape[0], -1)

        data = dict(
            uuid=uuid,
            step_index=step_index,
            intrinsic=intrinsic,
            T_cam2base=T_cam2base,
            T_base2world=copy.deepcopy(self.T_base2world),
            T_base2ego=copy.deepcopy(self.T_base2ego),
            joint_state=joint_state,
            ee_state=ee_state,
            mobile_traj=mobile_traj,
            action=action,
        )
        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        subtask = self.meta_lmdbs[lmdb_index][f"{uuid}/subtask"]
        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instruction"]
        data["text"] = instruction
        data["subtask"] = subtask

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
            vis_imgs = self.get_vis_imgs(
                data["imgs"],
                data.get("projection_mat"),
                data.get("hist_robot_state", [None])[-1],
                # data.get("pred_robot_state", [None])[0],
                data.get("hist_ee_state", [None])[-1],
                data["T_cam2base"],
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

    def draw_joint(
        self, img_index, img, joint_pose, projection_mat, ee_indices=None
    ):
        if ee_indices is None:
            ee_indices = [9, 10]

        for joint_index in range(joint_pose.shape[0]):
            rot = Rotation.from_quat(
                joint_pose[joint_index, 4:], scalar_first=True
            ).as_matrix()
            trans = joint_pose[joint_index, 1:4]

            if joint_index in ee_indices:
                axis_length = 0.05
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

                color = [0, 0, 0]
                color[i] = 255
                cv2.line(
                    img,
                    (pts_2d[i, 0], pts_2d[i, 1]),
                    (pts_2d[3, 0], pts_2d[3, 1]),
                    tuple(color),
                    3,
                )
        return img

    def draw_camera(self, img_index, img, camera_pose, projection_mat):
        for j in range(camera_pose.shape[0]):
            trans = camera_pose[j, :3, 3].numpy()
            rot = camera_pose[j, :3, :3].numpy()

            points = np.float32(
                [
                    [0.05, 0, 0],
                    [0, 0.05, 0],
                    [0, 0, 0.05],
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

                color = [0, 0, 0]
                color[i] = 255
                cv2.line(
                    img,
                    (pts_2d[i, 0], pts_2d[i, 1]),
                    (pts_2d[3, 0], pts_2d[3, 1]),
                    tuple(color),
                    3,
                )

        return img

    def get_vis_imgs(
        self,
        imgs,
        projection_mat=None,
        joint_pose=None,
        eef_pose=None,
        cam2base=None,
        ee_indices=(9, 10),
        channel_conversion=False,
    ):
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()

        if isinstance(projection_mat, torch.Tensor):
            projection_mat = projection_mat.cpu().numpy()

        if isinstance(joint_pose, torch.Tensor):
            joint_pose = joint_pose.cpu().numpy()

        if isinstance(eef_pose, torch.Tensor):
            eef_pose = eef_pose.cpu().numpy()

        vis_imgs = []
        imgs = (imgs * 255).astype(np.uint8)

        # left view: only left arm
        img = self.draw_joint(0, imgs[0], joint_pose[4:12, :], projection_mat)
        vis_imgs.append(img)

        # right view: only right arm
        img = self.draw_joint(1, imgs[1], joint_pose[12:20, :], projection_mat)
        vis_imgs.append(img)

        # head view: left arm + right arm
        img = self.draw_joint(2, imgs[2], joint_pose[4:12, :], projection_mat)
        img = self.draw_joint(2, img, joint_pose[12:20, :], projection_mat)
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


if __name__ == "__main__":
    pass
