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
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class InternA1LmdbDataset(BaseLmdbManipulationDataset):
    """RoboTwin LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── user: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4] or np.ndarray[4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions
        {uuid}/observation/robot_state/master_joint_positions

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
        cam_names=None,
        load_extrinsic=True,
        load_calibration=True,
        load_ee_state=False,
        bgr2rgb=False,
        depth_scale=1000,
        hist_steps=None,
        pred_steps=None,
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
            hist_steps=hist_steps,
            pred_steps=pred_steps,
            **kwargs,
        )
        self.cam_names = cam_names
        self.load_extrinsic = load_extrinsic
        self.load_ee_state = load_ee_state
        self.load_calibration = load_calibration
        self.bgr2rgb = bgr2rgb
        self.depth_scale = depth_scale

    def get_instruction(self, lmdb_index, data, by_reader=True):
        meta = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/meta_data"]
        text = meta["task_name"].replace("_", " ") + ". "
        return {"text": text}

    def get_depths(self, lmdb_index, data):
        depths = []
        img_shape = data["imgs"].shape[1:3]
        for _ in data["cam_names"]:
            dummy_depth = np.zeros(img_shape, dtype=np.float32)
            depths.append(dummy_depth)
        depths = np.stack(depths)
        return {"depths": depths}

    def get_images(self, lmdb_index, data):
        images = []
        for cam_name in data["cam_names"]:
            img_buffer = self.img_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            img_buffer = np.ndarray(
                shape=(1, len(img_buffer)), dtype=np.uint8, buffer=img_buffer
            )
            img = cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR)
            if self.bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        images = np.stack(images)
        return {"imgs": images}

    def get_intrinsic(self, lmdb_index, data):
        if "intrinsic" in data:
            intrinsics = data["intrinsic"]
        else:
            intrinsics = self.meta_lmdbs[lmdb_index][
                f"{data['uuid']}/intrinsic"
            ]
        if intrinsics is None:
            camera_info = self.meta_lmdbs[lmdb_index][
                f"{data['uuid']}/camera_info"
            ]
            intrinsics = {}
            for cam_name in camera_info.keys():
                intrinsics[cam_name] = np.array(
                    camera_info[cam_name]["image"]["P"]
                ).reshape(3, 4)
        intrinsic = []
        for cam_name in data["cam_names"]:
            tmp = np.eye(4)
            tmp[:3, :3] = intrinsics[cam_name][:3, :3]
            intrinsic.append(tmp)
        intrinsic = np.stack(intrinsic)
        return {"intrinsic": intrinsic}

    def get_extrinsic(self, lmdb_index, data):
        extrinsics = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/extrinsic"]
        t_world2cam = []  # noqa: N806
        for cam_name in data["cam_names"]:
            _ext = extrinsics[cam_name]
            if _ext.ndim == 3:
                _ext = _ext[data["step_index"]]
            t_world2cam.append(_ext)
        t_world2cam = np.stack(t_world2cam)  # noqa: N806
        t_world2cam = np.linalg.inv(t_world2cam)
        return {"T_world2cam": t_world2cam}

    def get_calibration(self, lmdb_index, data):
        # TODO, only for agilex collected data currently
        calibration = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/extrinsics"]
        calibration = dict(
            left=calibration["piper_left_end_pose_to_camera_left_link"],
            mid=calibration["piper_left_base_link_to_camera_mid_link"],
            right=calibration["piper_right_end_pose_to_camera_right_link"],
        )
        return {"calibration": calibration}

    def get_joint_state(self, lmdb_index, data):
        step_index = data["step_index"]
        num_steps_per_shard = data["num_steps_per_shard"]
        uuid = data["uuid"]
        joint_state = self._get_meta(
            lmdb_index,
            uuid,
            "observation/robot_state/joint_positions",
            step_index,
            num_steps_per_shard,
        )
        master_joint_state = self._get_meta(
            lmdb_index,
            uuid,
            "observation/robot_state/master_joint_positions",
            step_index,
            num_steps_per_shard,
        )
        step_index_in_shard = self._get_step_index_in_shard(
            step_index,
            num_steps_per_shard,
        )
        results = {
            "joint_state": np.array(joint_state),
            "step_index_in_shard": step_index_in_shard,
        }
        if master_joint_state is not None:
            results["master_joint_state"] = np.array(master_joint_state)
        return results

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index].get(episode_index)
        )
        uuid = idx_data.uuid
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]
        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        robot_type = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"][
            "robot_type"
        ]
        data = dict(
            uuid=uuid,
            step_index=step_index,
            task_name=idx_data.task_name,
            cam_names=cam_names,
            num_steps_per_shard=num_steps_per_shard,
            robot_type=robot_type,
        )

        data.update(self.get_joint_state(lmdb_index, data))
        if self.load_ee_state:
            ee_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/cartesian_position"
            ]
            data["left_ee"] = np.array(ee_state)[lmdb_index][0]
            data["right_ee"] = np.array(ee_state)[lmdb_index][1]

        data.update(self.get_intrinsic(lmdb_index, data))
        if self.load_image:
            data.update(self.get_images(lmdb_index, data))
        if self.load_depth:
            data.update(self.get_depths(lmdb_index, data))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(lmdb_index, data))
        if self.load_calibration:
            data.update(self.get_calibration(lmdb_index, data))

        data.update(self.get_instruction(lmdb_index, data))

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

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

            if "depths" in data.keys():
                vis_depths = self.depth_visualize(data["depths"])
                if len(vis_depths) % 2 == 0:
                    num_imgs = len(vis_depths)
                    vis_depths = np.concatenate(
                        [
                            np.concatenate(
                                vis_depths[: num_imgs // 2], axis=1
                            ),
                            np.concatenate(
                                vis_depths[num_imgs // 2 :], axis=1
                            ),
                        ],
                        axis=0,
                    )
                else:
                    vis_depths = np.concatenate(vis_depths, axis=1)
                vis_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)

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
        ee_indices=(6, 13),
        channel_conversion=False,
    ):
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

    @staticmethod
    def depth_visualize(depth, min_depth=0.01, max_depth=1.2, mode="bwr"):
        import matplotlib.pyplot as plt

        mask = depth > 0
        cmap = plt.cm.get_cmap(mode, 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        cmap = cmap[::-1]

        depth_shape = depth.shape
        if max_depth is None:
            max_depth = depth.max()
        if min_depth is None:
            min_depth = depth.min()

        depth = (depth - min_depth) / (max_depth - min_depth)
        index = np.int32(depth * 255)
        index = np.clip(index, a_min=0, a_max=255)
        depth_color = cmap[index].reshape(*depth_shape, 3)
        depth_color = np.where(mask[..., None], depth_color, 0)
        depth_color = np.uint8(depth_color)

        return depth_color
