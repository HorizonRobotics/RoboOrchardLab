# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def sample(tgt_time, src_time, src_data=None, prefix=""):
    time_diff = np.abs(tgt_time[:, None] - src_time)
    logger.info(
        f"{prefix:<50} - "
        + f"max time diff: {time_diff.min(axis=-1).max():.4f}, "
        + f"mean time diff: {time_diff.min(axis=-1).mean():.4f}"
    )
    index = np.argmin(time_diff, axis=1)
    output_time = src_time[index]
    if src_data is not None:
        output = []
        for src in src_data:
            _output = []
            for i in index:
                _output.append(src[i])
            output.append(_output)
        return output_time, output
    return output_time


def format_time(timestamp):
    timestamp = np.array(timestamp, dtype="float64")
    timestamp = timestamp[:, 0] + timestamp[:, 1] / 1e9
    return timestamp


def pose_to_mat(pose):
    if isinstance(pose, dict):
        x, y, z = pose["position"]
        qx, qy, qz, w = pose["orientation"]
    elif hasattr(pose, "position"):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, w = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    else:
        x, y, z = pose.translation.x, pose.translation.y, pose.translation.z
        qx, qy, qz, w = (
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
        )

    trans = np.array([x, y, z])
    rot = Rotation.from_quat([qx, qy, qz, w], scalar_first=False).as_matrix()
    ret = np.eye(4)
    ret[:3, 3] = trans
    ret[:3, :3] = rot
    return ret


def get_frequency(timestamp, prefix="", window_size=3):
    if not isinstance(timestamp, np.ndarray):
        timestamp = np.array(timestamp)
    time_diff = np.diff(timestamp)
    time_diff = torch.from_numpy(time_diff)[None, None]
    time_diff = torch.nn.functional.avg_pool1d(
        time_diff, window_size, 1
    ).numpy()[0, 0]
    freq = 1 / time_diff
    logger.info(
        f"{prefix:<50} - "
        f"duration: {timestamp[-1] - timestamp[0]:.2f}s, "
        + f"min frequency: {freq.min():.1f}Hz, "
        + f"mean frequency: {freq.mean():.1f}Hz"
    )
    return freq
