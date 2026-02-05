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
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from robo_orchard_lab.dataset.robot.dataset import (
    ConcatRODataset,
    RODataset,
    ROMultiRowDataset,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode

logger = logging.getLogger(__file__)


class RODatasetVisualizer:
    """Visualizer for RODataset (ONLY transformed data).

    Supported:
        Episode → mp4
    """

    def __init__(
        self,
        dataset: RODataset | ROMultiRowDataset | ConcatRODataset,
        ee_indices=(7,),
    ):
        """Initialize the visualizer with a dataset and end-effector indices.

        Args:
            dataset: RODataset / ROMultiRowDataset with transforms applied.
            ee_indices: Indices of the joints to be treated as end-effectors.
        """
        self.dataset = dataset
        self.ee_indices = ee_indices

    # =========================================================================
    # Public API
    # =========================================================================

    def visualize_episode(
        self,
        episode_index,
        output_dir,
        fps=10,
        interval=1,
        with_episode_id=True,
        episode_range=None,
        with_frame_idx=False,
        with_valid_mask=False,
    ):
        """Visualize one full episode and save it as an mp4 file."""

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        start, end = self._episode_range(episode_index)

        frames = []
        for idx in tqdm(range(start, end, interval)):
            if episode_range is not None:
                frame_idx = idx - start
                if (
                    frame_idx < episode_range[0]
                    or frame_idx >= episode_range[1]
                ):
                    continue
            data = self.dataset[idx]
            idx = data["step_index"]
            frame = self._render_frame(data)

            if with_frame_idx:
                cv2.putText(
                    frame,
                    f"Frame: {idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                )

            if with_valid_mask:
                cv2.putText(
                    frame,
                    f"Valid: {data['pred_mask'][0]}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                )

            frames.append(frame)

            if idx == start:
                logger.info(f"episode: {data['uuid']}")

        uuid = data["uuid"]
        out_name = uuid.replace("/", "_")
        if with_episode_id:
            out_name = f"ep{episode_index:04d}_" + out_name
        if episode_range is not None:
            out_name = (
                f"{out_name}_frames{episode_range[0]}-{episode_range[1]}"
            )

        save_path = output_dir / f"{out_name}.mp4"

        logger.info(f"Saving episode {episode_index} → {output_dir}")
        imageio.mimwrite(str(save_path), frames, fps=fps)
        logger.info(f"Save video to {save_path.absolute()}")

    def visualize_frame(self, index):
        """Return a single visualization frame as uint8 numpy image."""
        data = self.dataset[index]
        return self._render_frame(data)

    # =========================================================================
    # Internal rendering logic
    # =========================================================================

    def _render_frame(self, data):
        """Render frame images and projected robot state.

        Render:
            Multiple camera images
            Robot joint coordinate axes (projected with projection_mat)
        """

        imgs = data["imgs"]
        depths = data["depths"]
        proj = data.get("projection_mat")
        robot_state = data.get("hist_robot_state", [None])[-1]

        # convert to numpy
        if hasattr(imgs, "cpu"):
            imgs = imgs.cpu().numpy()
        if hasattr(proj, "cpu"):
            proj = proj.cpu().numpy()
        if hasattr(robot_state, "cpu"):
            robot_state = robot_state.cpu().numpy()

        vis_imgs = self.get_vis_imgs(
            imgs, proj, robot_state, ee_indices=self.ee_indices
        )
        vis_depths = self.depth_visualize(depths)
        vis_depths = np.reshape(
            vis_depths.transpose(1, 0, 2, 3), vis_imgs.shape
        )

        final_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)

        return final_imgs

    # =========================================================================

    @staticmethod
    def get_vis_imgs(imgs, projection_mat, robot_state, ee_indices=(7,)):
        """Combine all camera views and draw coordinate frames.

        Args:
            imgs: (N, H, W, 3) input images.
            projection_mat: Camera projection matrices.
            robot_state: Joint positions and orientations.
            ee_indices: Indices of end-effector joints.
        """
        if imgs.ndim == 3:
            imgs = imgs[None]

        vis_list = []

        for cam_idx in range(imgs.shape[0]):
            img = imgs[cam_idx].copy()

            if projection_mat is None or robot_state is None:
                vis_list.append(img)
                continue

            proj_matrix = projection_mat[cam_idx]

            # draw all joints
            for j in range(robot_state.shape[0]):
                rot = Rotation.from_quat(
                    robot_state[j, 4:], scalar_first=True
                ).as_matrix()
                trans = robot_state[j, 1:4]

                axis_len = 0.1 if j in ee_indices else 0.03
                points = np.float32(
                    [
                        [axis_len, 0, 0],
                        [0, axis_len, 0],
                        [0, 0, axis_len],
                        [0, 0, 0],
                    ]  # type: ignore
                )
                points = points @ rot.T + trans

                pts3 = points @ proj_matrix[:3, :3].T + proj_matrix[:3, 3]
                depth = pts3[:, 2]
                pts2 = pts3[:, :2] / depth[:, None]

                if depth[3] < 0.02:
                    continue

                pts2 = pts2.astype(np.int32)

                # draw axis lines
                for ax in range(3):
                    color = [0, 0, 0]
                    color[ax] = 255
                    cv2.line(
                        img, tuple(pts2[3]), tuple(pts2[ax]), tuple(color), 3
                    )

                # draw axis tips
                for ax in range(3):
                    cv2.circle(img, tuple(pts2[ax]), 5, (0, 0, 255), -1)

                # draw gripper value
                if j in ee_indices:
                    gripper_value = robot_state[j, 0]
                    x, y = int(pts2[3][0]) + 5, int(pts2[3][1]) - 5
                    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                        continue
                    cv2.putText(
                        img,
                        f"G<{j}>: {gripper_value:.2f}",
                        (int(pts2[3][0]) + 5, int(pts2[3][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            vis_list.append(img[:, :, ::-1])  # BGR to RGB

        return np.uint8(np.concatenate(vis_list, axis=1))

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

    # =========================================================================

    def _episode_range(self, ep_idx):
        """Return (start_idx, end_idx) for a specific episode."""
        for ep in self.dataset.iterate_meta(Episode):
            if ep_idx == ep.index:
                start = ep.dataset_begin_index
                end = start + ep.frame_num
                return start, end

        raise KeyError(f"Episode index {ep_idx} not found in dataset")
