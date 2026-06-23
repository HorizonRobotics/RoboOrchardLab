# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import struct

import cv2
import numpy as np

from robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_packer import (
    PiperMcapPacker,
)


def _encode_depth(depth: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", depth)
    assert success
    return buffer.tobytes()


def _corrupt_first_idat(encoded: bytes) -> bytes:
    corrupted = bytearray(encoded)
    pos = 8
    while pos + 12 <= len(corrupted):
        length = struct.unpack(">I", corrupted[pos : pos + 4])[0]
        chunk_type = bytes(corrupted[pos + 4 : pos + 8])
        data_start = pos + 8
        data_end = data_start + length
        if chunk_type == b"IDAT":
            corrupted[data_start] ^= 0xFF
            return bytes(corrupted)
        pos = data_end + 4
    raise AssertionError("encoded PNG has no IDAT chunk")


def test_replace_invalid_depths_uses_zero_png_for_bad_frames():
    valid_depth = np.arange(16, dtype=np.uint16).reshape(4, 4)
    valid_encoded = _encode_depth(valid_depth)
    invalid_encoded = _corrupt_first_idat(valid_encoded)

    sanitized = PiperMcapPacker._replace_invalid_depths(
        [valid_encoded, invalid_encoded],
        uuid="task/user/episode",
        cam="middle",
    )

    assert sanitized[0] == valid_encoded
    assert sanitized[1] != invalid_encoded
    decoded = PiperMcapPacker._decode_depth(sanitized[1])
    assert decoded is not None
    assert decoded.dtype == valid_depth.dtype
    assert decoded.shape == valid_depth.shape
    assert np.count_nonzero(decoded) == 0


def test_replace_invalid_depths_uses_expected_shape_when_all_frames_bad():
    valid_depth = np.arange(16, dtype=np.uint16).reshape(4, 4)
    invalid_encoded = _corrupt_first_idat(_encode_depth(valid_depth))

    sanitized = PiperMcapPacker._replace_invalid_depths(
        [invalid_encoded],
        uuid="task/user/episode",
        cam="middle",
        expected_shape=(3, 5),
    )

    decoded = PiperMcapPacker._decode_depth(sanitized[0])
    assert decoded is not None
    assert decoded.dtype == np.uint16
    assert decoded.shape == (3, 5)
    assert np.count_nonzero(decoded) == 0
