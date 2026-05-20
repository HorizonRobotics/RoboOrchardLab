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


import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from robo_orchard_lab.dataset.behavior.utils import ROBOT_CAMERA_NAMES
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)


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
        cam_names=None,
        hist_steps=None,
        pred_steps=None,
        reset_step=1000,
        dataset_name="behavior",
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
            reset_step=reset_step,
            dataset_name=dataset_name,
            **kwargs,
        )

        if cam_names is not None:
            self.cam_names = cam_names
        else:
            self.cam_names = ROBOT_CAMERA_NAMES["R1Pro"]

        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.dataset_name = dataset_name

    def _concat_shards(self, *shards):
        shards = [x for x in shards if x is not None]
        if len(shards) == 0:
            return None
        elif isinstance(shards[0], np.ndarray):
            return np.concatenate(shards, axis=0)
        elif isinstance(shards[0], list):
            results = []
            for x in shards:
                results.extend(x)
            return results

    def _get_meta_with_shard(
        self,
        lmdb_index,
        uuid,
        key,
        step_index,
        num_steps_per_shard,
    ):
        shard_index = step_index // num_steps_per_shard
        current_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/{shard_index}/{key}"
        ]
        step_index_in_shard = step_index % num_steps_per_shard
        if (
            self.hist_steps is not None
            and step_index_in_shard < self.hist_steps - 1
            and shard_index != 0
        ):
            pre_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index - 1}/{key}"
            ]
        else:
            pre_shard = None

        if (
            self.pred_steps is not None
            and num_steps_per_shard - step_index_in_shard < self.pred_steps
        ):
            # maybe out of bound, return None
            next_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index + 1}/{key}"
            ]
        else:
            next_shard = None

        if pre_shard is not None:
            step_index_in_shard += len(pre_shard)

        data = self._concat_shards(pre_shard, current_shard, next_shard)
        return data, step_index_in_shard

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        if num_steps_per_shard is None:
            mobile_traj = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/mobile_traj"
            ]
            joint_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/joint_position"
            ]
            action = self.meta_lmdbs[lmdb_index][
                f"{uuid}/robot_action/joint_position"
            ]

            extrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
            intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        else:
            mobile_traj, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/mobile_traj",
                step_index,
                num_steps_per_shard,
            )
            joint_state, _ = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/joint_position",
                step_index,
                num_steps_per_shard,
            )

            action, _ = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "robot_action/joint_position",
                step_index,
                num_steps_per_shard,
            )

            extrinsic, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "extrinsic",
                step_index,
                num_steps_per_shard,
            )
            extrinsic = extrinsic[step_index_in_shard]

            intrinsic, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "intrinsic",
                step_index,
                num_steps_per_shard,
            )
            intrinsic = intrinsic[step_index_in_shard]

        data = dict(
            uuid=uuid,
            step_index=(
                step_index
                if num_steps_per_shard is None
                else step_index_in_shard
            ),
            mobile_traj=mobile_traj,
            joint_state=joint_state,
            action=action,
            intrinsic=intrinsic,
            T_world2cam=extrinsic,
        )

        if num_steps_per_shard is not None:
            data["step_index_in_shard"] = step_index_in_shard

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        for cam_name in self.cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/rgb_{cam_name}/{step_index}"
                ]

                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                images.append(image)

            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/depth_{cam_name}/{step_index}"
                ]

                depth = cv2.imdecode(
                    depth, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
                )
                depth = depth / 1000.0
                depths.append(depth)

        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instruction"]
        data["text"] = instruction

        subtask_text = self.meta_lmdbs[lmdb_index][f"{uuid}/subtask_text"]
        data["subtask_text"] = subtask_text

        skill_text = self.meta_lmdbs[lmdb_index][f"{uuid}/skill_text"]
        data["skill_text"] = skill_text

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)

        return data

    def draw_depth_heatmap(self, depths):
        heatmaps = []
        for depth in depths:
            if torch.is_tensor(depth):
                depth = depth.detach().cpu().numpy()

            depth = np.nan_to_num(depth)
            dmin = np.percentile(depth, 2)
            dmax = np.percentile(depth, 98)
            if dmax - dmin < 1e-6:
                dmax = dmin + 1e-6

            depth = np.clip(depth, dmin, dmax)
            depth_norm = (depth - dmin) / (dmax - dmin)
            depth_norm = (depth_norm * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            heatmaps.append(heatmap)

        return np.concatenate(heatmaps, axis=1)

    def draw_traj_board(
        self,
        local_traj,
        *,
        board_h,
        board_w,
        scale=50,  # pixels per meter (physical)
        viz_scale=1.0,  # visualization-only scale (>=1.0 makes traj longer)
        grid_meter=1.0,  # grid spacing in meters
        traj_color=(0, 0, 255),
        robot_color=(0, 0, 0),
        axis_color=(0, 0, 0),
        grid_color=(220, 220, 220),
    ):
        """Draw trajectory on a white board.

        Args:
            local_traj: (N, 3) array, [x, y, yaw]
            board_h / board_w: board size in pixels
            scale: physical scale, pixels per meter
            viz_scale: visualization-only scale multiplier
                       - 1.0 : physical scale
                       - 1.5~2.0 : recommended for debugging
            grid_meter: grid spacing in meters
        """
        if torch.is_tensor(local_traj):
            local_traj = local_traj.detach().cpu().numpy()

        # local_traj = np.asarray(local_traj, dtype=np.float32)

        board = np.ones((board_h, board_w, 3), dtype=np.uint8) * 255
        cx, cy = board_w // 2, board_h // 2

        scale_viz = float(scale) * float(viz_scale)
        grid_px = int(grid_meter * scale_viz)
        grid_px = max(grid_px, 8)

        # vertical grid lines
        x = cx
        while x < board_w:
            cv2.line(board, (x, 0), (x, board_h), grid_color, 1)
            x += grid_px
        x = cx
        while x >= 0:
            cv2.line(board, (x, 0), (x, board_h), grid_color, 1)
            x -= grid_px

        # horizontal grid lines
        y = cy
        while y < board_h:
            cv2.line(board, (0, y), (board_w, y), grid_color, 1)
            y += grid_px
        y = cy
        while y >= 0:
            cv2.line(board, (0, y), (board_w, y), grid_color, 1)
            y -= grid_px

        axis_len = int(1.0 * scale_viz)  # 1 meter
        # x forward (up)
        cv2.arrowedLine(
            board,
            (cx, cy),
            (cx, cy - axis_len),
            axis_color,
            2,
            tipLength=0.15,
        )

        # y left
        cv2.arrowedLine(
            board,
            (cx, cy),
            (cx - axis_len, cy),
            axis_color,
            2,
            tipLength=0.15,
        )

        # robot
        cv2.circle(board, (cx, cy), 6, robot_color, -1)

        pts = []
        for x, y in local_traj[:, :2]:
            px = int(cx - y * scale_viz)  # y left
            py = int(cy - x * scale_viz)  # x forward
            pts.append((px, py))

        if len(pts) >= 2:
            cv2.polylines(
                board,
                [np.array(pts, dtype=np.int32)],
                isClosed=False,
                color=traj_color,
                thickness=3,
            )

        return board

    def draw_joint(
        self,
        imgs,
        joint_pose,
        projection_mat,
        *,
        cam_draw_plan=None,
        ee_indices=(11, 19),
        channel_conversion=False,
    ):
        if torch.is_tensor(imgs):
            imgs = imgs.detach().cpu().numpy()

        if torch.is_tensor(joint_pose):
            joint_pose = joint_pose.detach().cpu().numpy()

        if torch.is_tensor(projection_mat):
            projection_mat = projection_mat.detach().cpu().numpy()

        if cam_draw_plan is None:
            cam_draw_plan = {
                0: [slice(4, 12)],
                1: [slice(12, 20)],
                2: [slice(4, 12), slice(12, 20)],
            }

        vis_imgs = []
        for cam_id, joint_slices in cam_draw_plan.items():
            img = imgs[cam_id].copy()
            proj_r = projection_mat[cam_id, :3, :3]
            proj_t = projection_mat[cam_id, :3, 3]

            for js in joint_slices:
                joints = joint_pose[js]

                for j in range(joints.shape[0]):
                    trans = joints[j, 1:4]

                    quat = joints[j, 4:]
                    rot = Rotation.from_quat(
                        quat, scalar_first=True
                    ).as_matrix()

                    global_joint_idx = js.start + j
                    axis_len = 0.06 if global_joint_idx in ee_indices else 0.04

                    axes = np.array(
                        [
                            [axis_len, 0, 0],
                            [0, axis_len, 0],
                            [0, 0, axis_len],
                            [0, 0, 0],
                        ],
                        dtype=np.float32,
                    )

                    pts_3d = axes @ rot.T + trans
                    pts_cam = pts_3d @ proj_r.T + proj_t

                    depth = pts_cam[:, 2]
                    if depth[3] < 0.02:
                        continue

                    pts_2d = (pts_cam[:, :2] / depth[:, None]).astype(np.int32)
                    origin = tuple(pts_2d[3])

                    for i in range(3):
                        if depth[i] < 0.02:
                            continue
                        cv2.circle(img, tuple(pts_2d[i]), 3, (0, 0, 255), -1)
                        color = [0, 0, 0]
                        color[i] = 255
                        cv2.line(
                            img,
                            origin,
                            tuple(pts_2d[i]),
                            tuple(color),
                            3,
                        )

            vis_imgs.append(img)

        vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = vis_imgs.astype(np.uint8)

        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]

        return vis_imgs

    def visualize(
        self,
        episode_index,
        output_path,
        fps=30,
        traj_viz_range=5.0,  # 5 meter
        traj_viz_scale=4.0,
    ):
        end_idx = self.cumsum_steps[episode_index]
        if episode_index != 0:
            start_idx = self.cumsum_steps[episode_index - 1]
        else:
            start_idx = 0

        video_writer = None  # noqa: N806
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        uuid = self.__getitem__(start_idx)["uuid"]
        save_file = os.path.join(output_path, f"{uuid.replace('/', '-')}.mp4")

        for i in tqdm(
            range(start_idx, end_idx),
            desc=f"{uuid}",
        ):
            data = self.__getitem__(i)

            # joint
            joint_img = self.draw_joint(
                imgs=data["imgs"],
                joint_pose=data["hist_robot_state"][-1],
                projection_mat=data["projection_mat"],
            )

            board_h = joint_img.shape[0]
            board_w = joint_img.shape[0]

            # traj
            real_range_m = 2.0 * traj_viz_range
            scale = board_w / real_range_m
            traj_board = self.draw_traj_board(
                local_traj=data["mobile_traj"],
                board_h=board_h,
                board_w=board_w,
                scale=scale,
                viz_scale=traj_viz_scale,
            )

            # depth
            depth_img = self.draw_depth_heatmap(data["depths"])
            blank = np.ones_like(traj_board) * 255
            depth_img = np.concatenate([depth_img, blank], axis=1)

            # concat
            top_row = np.concatenate([joint_img, traj_board], axis=1)
            vis_img = np.concatenate([top_row, depth_img], axis=0)

            # write text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            line_height = 22
            text_color = (0, 0, 255)

            texts = [
                data["text"],
                data["subtask_text"],
                data["skill_text"],
            ]

            x0 = 10
            y0 = 30
            for i, txt in enumerate(texts):
                y = y0 + i * line_height
                cv2.putText(
                    vis_img,
                    txt,
                    (x0, y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

            if video_writer is None:
                h, w = vis_img.shape[:2]
                video_writer = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

            video_writer.write(vis_img)

        video_writer.release()
