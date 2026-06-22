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

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class EgoDexLmdbDataset(BaseLmdbManipulationDataset):
    """EgoDex LMDB Dataset."""

    def __init__(self, load_reference_img=False, **kwargs):
        super().__init__(**kwargs)
        self.load_reference_img = load_reference_img

    def _get_reference_img(self, lmdb_index, episode_index, uuid):
        last_step_index = (
            self.idx_lmdbs[lmdb_index][episode_index]["num_steps"] - 1
        )
        image = self.img_lmdbs[lmdb_index][f"{uuid}/camera/{last_step_index}"]
        if image is not None:
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return {"reference_imgs": [image]}

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name

        extrinsic = np.linalg.inv(
            self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"][step_index]
        )
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        joint_transforms = self.meta_lmdbs[lmdb_index][
            f"{uuid}/joint_state/transforms"
        ]
        joint_confidence = self.meta_lmdbs[lmdb_index][
            f"{uuid}/joint_state/confidences"
        ]

        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]

        joint_names = self.meta_lmdbs[lmdb_index]["transform_names"]

        data = dict(
            uuid=uuid,
            step_index=step_index,
            intrinsic=intrinsic[None],
            T_world2cam=np.eye(4)[None],
            joint_transforms=extrinsic @ joint_transforms,
            joint_confidence=joint_confidence,
            task_name=task_name,
            text=instruction,
            joint_names=joint_names,
        )

        if self.load_image:
            image = self.img_lmdbs[lmdb_index][f"{uuid}/camera/{step_index}"]
            if isinstance(image, bytes):
                image = np.ndarray(
                    shape=(1, len(image)), dtype=np.uint8, buffer=image
                )
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            data["imgs"] = np.stack([image])
            data["depths"] = np.zeros_like(data["imgs"][..., 0])

        if self.load_reference_img:
            data.update(
                self._get_reference_img(lmdb_index, episode_index, uuid)
            )

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def visualize(
        self,
        episode_index,
        output_path="./vis_data",
        fps=10,
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
