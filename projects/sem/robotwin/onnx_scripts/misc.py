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

import numpy as np
import torch

ENCODER_INPUT_NAMES = [
    "imgs",
    "depths",
    "pixels",
    "projection_mat_inv",
    "hist_robot_state",
    "joint_scale_shift",
    "joint_relative_pos",
]

ENCODER_OUTPUT_NAMES = [
    "image_feature",
    "robot_feature",
]

DECODER_INPUT_NAMES = [
    "noisy_action",
    "image_feature",
    "robot_feature",
    "timestep",
    "joint_relative_pos",
]

DECODER_OUTPUT_NAMES = ["pred_action"]


def unsqueeze_batch(data):
    for k, v in data.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            data[k] = v[None]
        else:
            data[k] = [v]
    return data


def get_pixels(data, strides):
    image_wh = data["image_wh"][0].to(torch.int32).tolist()
    pixels = []
    for stride in strides:
        feature_wh = [image_wh[0] // stride, image_wh[1] // stride]
        u = torch.linspace(0, image_wh[0] - stride, feature_wh[0])
        v = torch.linspace(0, image_wh[1] - stride, feature_wh[1])
        u = u[None].tile(feature_wh[1], 1)
        v = v[:, None].tile(1, feature_wh[0])
        uv = torch.stack([u, v], dim=-1).flatten(0, 1)
        pixels.append(uv)
    pixels = torch.cat(pixels, dim=0)[:, None].to(data["imgs"])
    return pixels
