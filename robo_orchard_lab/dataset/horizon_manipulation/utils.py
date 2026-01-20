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

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def decode_img(img_buffer):
    img_buffer = np.ndarray(
        shape=(1, len(img_buffer)), dtype=np.uint8, buffer=img_buffer
    )
    img = cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR)
    return img


def decode_depth(depth_buffer, depth_scale=1000):
    return (
        cv2.imdecode(
            np.frombuffer(depth_buffer, np.uint8), cv2.IMREAD_UNCHANGED
        )
        / depth_scale
    )


def depth_visualize(depth, min_depth=0.01, max_depth=1.2, mode="bwr"):
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
