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


import multiprocessing as mp
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import av
import numpy as np
from scipy.spatial.transform import Rotation

ROBOT_CAMERA_NAMES = {
    "A1": {
        # "external": "external::external_camera",
        # "wrist": "external::wrist_camera",
        "external",
        "wrist",
    },
    "R1Pro": [
        # "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        # "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        # "head": "robot_r1::robot_r1:zed_link:Camera:0",
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
            "joint_qpos": np.s_[
                0:28
            ],
            "joint_qpos_sin": np.s_[
                28:56
            ],
            "joint_qpos_cos": np.s_[
                56:84
            ],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[
                140:143
            ],
            "robot_ori_cos": np.s_[
                143:146
            ],
            "robot_ori_sin": np.s_[
                146:149
            ],
            "robot_2d_ori": np.s_[
                149:150
            ],
            "robot_2d_ori_cos": np.s_[
                150:151
            ],
            "robot_2d_ori_sin": np.s_[
                151:152
            ],
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
            "base_qpos": np.s_[
                244:247
            ],
            "base_qpos_sin": np.s_[
                247:250
            ],
            "base_qpos_cos": np.s_[
                250:253
            ],
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


@dataclass
class SkillAnno:
    skill_idx: int
    skill_id: List[int]
    skill_description: List[str]
    object_id: List[List[str]]
    manipulating_object_id: List[str] | List[Any]  # 有时为空数组
    frame_duration: Tuple[int, int]
    skill_type: List[str] | List[Any]
    memory_prefix: Optional[List[str]] = None
    spatial_prefix: Optional[List[str]] = None


@dataclass
class PrimitiveAnno:
    primitive_idx: int
    primitive_id: List[int]
    primitive_description: List[str]
    object_id: List[List[str]]
    manipulating_object_id: List[str]
    frame_duration: Tuple[int, int]
    skill_idxes: List[int]
    memory_prefix: Optional[List[str]] = None
    spatial_prefix: Optional[List[str]] = None


@dataclass
class Episode:
    task_name: str
    task_id: str
    episode_id: str
    skill_annotation: List[SkillAnno]
    primitive_annotation: List[PrimitiveAnno]


def to_skill(d):
    return SkillAnno(
        skill_idx=d["skill_idx"],
        skill_id=d.get("skill_id", []),
        skill_description=d.get("skill_description", []),
        object_id=d.get("object_id", []),
        manipulating_object_id=d.get("manipulating_object_id", []),
        frame_duration=tuple(d.get("frame_duration", (0, 0))),
        skill_type=d.get("skill_type", []),
        memory_prefix=d.get("memory_prefix"),
        spatial_prefix=d.get("spatial_prefix"),
    )


def to_primitive(d):
    return PrimitiveAnno(
        primitive_idx=d["primitive_idx"],
        primitive_id=d.get("primitive_id", []),
        primitive_description=d.get("primitive_description", []),
        object_id=d.get("object_id", []),
        manipulating_object_id=d.get("manipulating_object_id", []),
        frame_duration=tuple(d.get("frame_duration", (0, 0))),
        skill_idxes=d.get("skill_idxes", []),
        memory_prefix=d.get("memory_prefix"),
        spatial_prefix=d.get("spatial_prefix"),
    )


def decode_video(video_path, sample_rate=3):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.thread_count = mp.cpu_count()
    stream.codec_context.thread_type = "FRAME"

    is_depth = True if "depth" in video_path else False
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % sample_rate != 0:
            continue

        if is_depth:
            frame = frame.reformat(format="gray16le")
            img = frame.to_ndarray()  # uint16 single-channel
            img = dequantize_depth(img)
        else:
            img = frame.to_ndarray(format="bgr24")

        frames.append(img)

    container.close()

    frames = np.stack(frames, axis=0)
    return frames


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


def quat2mat(quaternion):
    """Convert quaternions into rotation matrices.

    Args:
        quaternion (np.ndarray): (x, y, z, w).

    Returns:
        np.ndarray: (..., 3, 3) rotation matrices.
    """
    quaternion = quaternion / np.linalg.norm(
        quaternion, axis=-1, keepdims=True
    )

    outer = np.expand_dims(quaternion, -1) * np.expand_dims(quaternion, -2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = np.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat


def euler2mat(euler):
    """Converts extrinsic euler angles into rotation matrix form.

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    return Rotation.from_euler("xyz", euler).as_matrix()
