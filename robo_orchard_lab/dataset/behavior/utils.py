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


from collections import OrderedDict

import ffmpeg
import numpy as np

ROBOT_CAMERA_NAMES = {
    "A1": {
        "external",
        "wrist",
    },
    "R1Pro": [
        "left_wrist",
        "right_wrist",
        "head",
    ],
}


# Camera resolutions and corresponding intrinstics
HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)


# TODO: Fix A1
CAMERA_INTRINSICS = {
    "A1": {
        "external": np.array(
            [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),  # 240x240
        "wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),  # 240x240
    },
    "R1Pro": {
        "head": np.array(
            [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),  # 480x480
    },
}


# Action indices
ACTION_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:7],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    ),
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "A1": OrderedDict(
        {
            "joint_qpos": np.s_[0:8],
            "joint_qpos_sin": np.s_[8:16],
            "joint_qpos_cos": np.s_[16:24],
            "joint_qvel": np.s_[24:32],
            "joint_qeffort": np.s_[32:40],
            "eef_0_pos": np.s_[40:43],
            "eef_0_quat": np.s_[43:47],
            "grasp_0": np.s_[47:48],
            "gripper_0_qpos": np.s_[48:50],
            "gripper_0_qvel": np.s_[50:52],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],
            "base_qpos_sin": np.s_[247:250],
            "base_qpos_cos": np.s_[250:253],
            "base_qvel": np.s_[253:256],
        }
    ),
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:8],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:6],
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    ),
}

TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_INDICES_TO_NAMES = {v: k for k, v in TASK_NAMES_TO_INDICES.items()}


def decode_video_to_frames_ffmpeg(
    video_path: str,
) -> tuple[list[bytes], int, int]:
    """Robust decoder that handles AV1 / H.264 / HEVC."""
    probe = ffmpeg.probe(video_path)
    video_info = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])

    process = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="error")
        .run_async(pipe_stdout=True)
    )

    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame
    process.wait()


def decode_depth_to_frames_ffmpeg(
    video_path: str,
) -> tuple[list[bytes], int, int]:
    """Robust decoder that handles AV1 / H.264 / HEVC."""
    probe = ffmpeg.probe(video_path)
    video_info = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])

    process = (
        ffmpeg.input(video_path)
        .output(
            "pipe:", format="rawvideo", pix_fmt="gray16le", loglevel="error"
        )
        .run_async(pipe_stdout=True)
    )

    frame_size = width * height * 2
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint16).reshape([height, width])
        frame = dequantize_depth(frame)
        yield frame
    process.wait()


def dequantize_depth(
    quantized_depth: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    shift: float = 3.5,
) -> np.ndarray:
    """Dequantizes a 14-bit depth tensor back to the original depth values.

    Args:
        quantized_depth (np.ndarray): Quantized depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).

    Returns:
        np.ndarray: Dequantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = np.log(min_depth + shift)
    log_max = np.log(max_depth + shift)

    log_norm = quantized_depth / qmax
    log_depth = log_norm * (log_max - log_min) + log_min
    depth = np.clip(np.exp(log_depth) - shift, min_depth, max_depth)

    return depth


def compute_episode_keep_indices(
    mobile_traj: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    # static_threshold: float = 1e-3,
    static_threshold: float = 1e-3,
    base_time: np.ndarray | None = None,
    head_time_to_filter: float | None = None,
    tail_time_to_filter: float | None = None,
):
    num_steps = state.shape[0]
    static_mask = np.ones(num_steps, dtype=bool)

    if static_threshold > 0:

        def diff_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[:, None]
            return np.any(
                np.abs(np.diff(x, axis=0)) > static_threshold, axis=1
            )

        diff_mobile_traj = diff_func(mobile_traj)
        diff_state = diff_func(state)

        dynamic = diff_mobile_traj | diff_state

        static_mask[1:] = dynamic

    time_mask = np.ones(num_steps, dtype=bool)

    if head_time_to_filter is not None:
        time_from_start = (base_time - base_time[0]) / 1e9
        time_mask[time_from_start < head_time_to_filter] = False

    if tail_time_to_filter is not None:
        time_to_end = (base_time[-1] - base_time) / 1e9
        time_mask[time_to_end < tail_time_to_filter] = False

    keep_indices = static_mask & time_mask

    mobile_traj_filt = mobile_traj[keep_indices]
    state_filt = state[keep_indices]
    action_filt = action[keep_indices]

    extrinsic_filt = extrinsic[keep_indices]
    intrinsic_filt = intrinsic[keep_indices]

    return (
        keep_indices,
        mobile_traj_filt,
        state_filt,
        action_filt,
        extrinsic_filt,
        intrinsic_filt,
    )


def traj_world_to_local(xy_yaw_world: np.ndarray):
    xy_world = xy_yaw_world[:, :2]
    yaw = xy_yaw_world[:, 2]

    xy0 = xy_world[0]
    yaw0 = yaw[0]

    # rotation: world -> local
    c, s = np.cos(yaw0), np.sin(yaw0)
    world_to_local = np.array(
        [[c, s], [-s, c]],
        dtype=xy_world.dtype,
    )

    # transform position
    xy_local = (world_to_local @ (xy_world - xy0).T).T

    # transform yaw
    yaw_local = yaw - yaw0
    yaw_local = (yaw_local + np.pi) % (2 * np.pi) - np.pi

    traj_local = np.column_stack([xy_local, yaw_local])
    return traj_local
