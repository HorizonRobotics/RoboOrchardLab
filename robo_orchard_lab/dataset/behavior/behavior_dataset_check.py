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
from utils import load_config


def draw_traj_board(
    local_traj,
    *,
    board_h,
    board_w,
    scale=50,              # pixels per meter (physical)
    viz_scale=1.0,    # visualization-only scale (>=1.0 makes traj longer)
    grid_meter=1.0,        # grid spacing in meters
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

    #local_traj = np.asarray(local_traj, dtype=np.float32)

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
        px = int(cx - y * scale_viz)   # y left
        py = int(cy - x * scale_viz)   # x forward
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

def visualize_episode_joints(
    dataset,
    episode_index,
    output_path,
    fps=30,
    traj_viz_range=5.0, # 5 meter
    traj_viz_scale=4.0,
):
    os.makedirs(output_path, exist_ok=True)

    end_idx = dataset.cumsum_steps[episode_index]
    start_idx = (
        dataset.cumsum_steps[episode_index - 1] if episode_index > 0 else 0
    )

    first_data = dataset[start_idx]
    uuid = first_data["uuid"]
    save_file = os.path.join(
        output_path, f"{uuid.replace('/', '-')}_joint_traj.mp4"
    )

    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    for idx in tqdm(
        range(start_idx, end_idx),
        desc=f"{uuid}",
    ):
        data = dataset[idx]

        joint_img = draw_joint(
            imgs=data["imgs"],
            joint_pose=data["hist_robot_state"][-1],
            projection_mat=data["projection_mat"],
        )

        board_h = joint_img.shape[0]
        board_w = joint_img.shape[0]

        real_range_m = 2.0 * traj_viz_range
        scale = board_w / real_range_m
        traj_board = draw_traj_board(
            local_traj=data["mobile_traj"],
            board_h=board_h,
            board_w=board_w,
            scale=scale,
            viz_scale=traj_viz_scale,
        )

        vis_img = np.concatenate([joint_img, traj_board], axis=1)

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
            video_writer = cv2.VideoWriter(
                save_file, fourcc, fps, (w, h)
            )

        video_writer.write(vis_img)

    video_writer.release()
    return save_file

"""
copy behavior_dataset_check.py project/sem/common
python3 behavior_dataset_check.py
"""

if __name__ == "__main__":
    config_path = "configs/config_b1k_common.py"
    config = load_config(config_path)

    concat_dataset = config.build_training_dataset(config.config)
    cur_dataset = concat_dataset.datasets[0]
    print(cur_dataset.cumsum_steps)

    ep_num = len(cur_dataset.cumsum_steps)
    for i in range(ep_num):
        visualize_episode_joints(
            cur_dataset,
            episode_index=i,
            output_path="./data_check_res"
        )

    # visualize_episode_joints(
    #     cur_dataset,
    #     episode_index=8,
    #     output_path="./data_check_res"
    # )

