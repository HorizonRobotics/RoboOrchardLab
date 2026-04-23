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

import numpy as np
import pytest

from projects.holobrain.policy.robotwin_policy import (
    HoloBrainRoboTwinPolicy,
    HoloBrainRoboTwinPolicyCfg,
)
from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationOutput,
)


class _StubAction:
    def __init__(self, array: np.ndarray):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _StubPipeline:
    def __init__(self):
        self.calls = 0
        self.reset_calls = 0
        self.cfg = HoloBrainInferencePipelineCfg()
        self.model = type("Model", (), {"eval": lambda self: None})()

    def __call__(self, data):
        self.calls += 1
        return MultiArmManipulationOutput(
            action=_StubAction(
                np.array(
                    [
                        np.arange(14, dtype=np.float32),
                        np.arange(14, dtype=np.float32) + 1,
                        np.arange(14, dtype=np.float32) + 2,
                    ]
                )
            )
        )

    def reset(self, **kwargs):
        del kwargs
        self.reset_calls += 1


class _FakePoseMatrix:
    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def get_matrix(self):
        return self._matrix[None, ...]


class _FakePose:
    def __init__(self, cam2world: np.ndarray):
        self._cam2world = cam2world

    def inverse(self):
        return _FakePose(np.linalg.inv(self._cam2world))

    def as_Transform3D_M(self):  # noqa: N802
        return _FakePoseMatrix(self._cam2world)


class _FakeCameraData:
    def __init__(
        self,
        sensor_data: np.ndarray,
        intrinsic_matrices: np.ndarray,
        pose: _FakePose,
    ):
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


def _make_obs(extrinsic_shape: tuple[int, int]) -> dict:
    if extrinsic_shape == (3, 4):
        extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
    elif extrinsic_shape == (4, 4):
        extrinsic = np.eye(4, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported extrinsic shape {extrinsic_shape}")

    return {
        "observation": {
            "front_camera": {
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth": np.full((4, 4), 1000, dtype=np.float32),
                "extrinsic_cv": extrinsic,
                "intrinsic_cv": np.eye(3, dtype=np.float32),
            }
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        "instructions": "test instruction",
    }


@pytest.mark.parametrize("extrinsic_shape", [(3, 4), (4, 4)])
def test_robotwin_policy_supports_robotwin_camera_matrices(
    extrinsic_shape: tuple[int, int],
):
    policy = HoloBrainRoboTwinPolicy(
        pipeline=_StubPipeline(),
        cfg=HoloBrainRoboTwinPolicyCfg(use_action_chunk_size=2),
    )

    act1 = policy.act(_make_obs(extrinsic_shape))
    act2 = policy.act(_make_obs(extrinsic_shape))
    act3 = policy.act(_make_obs(extrinsic_shape))

    assert act1.shape == (14,)
    assert act2.shape == (14,)
    assert act3.shape == (14,)
    assert policy.pipeline.calls == 2


def _make_formatted_obs() -> dict:
    pose = _FakePose(np.eye(4, dtype=np.float64))
    return {
        "cameras": {
            "front_camera": {
                "rgb": _FakeCameraData(
                    sensor_data=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                    intrinsic_matrices=np.eye(3, dtype=np.float32)[None, ...],
                    pose=pose,
                ),
                "depth": _FakeCameraData(
                    sensor_data=np.full((1, 4, 4), 1000.0, dtype=np.float32),
                    intrinsic_matrices=np.eye(3, dtype=np.float32)[None, ...],
                    pose=pose,
                ),
            }
        },
        "joints": type(
            "Joints",
            (),
            {
                "position": np.arange(14, dtype=np.float32)[None, ...],
            },
        )(),
        "instructions": "formatted instruction",
        "tf": {},
    }


def test_robotwin_policy_supports_formatted_robotwin_observation():
    policy = HoloBrainRoboTwinPolicy(
        pipeline=_StubPipeline(),
        cfg=HoloBrainRoboTwinPolicyCfg(use_action_chunk_size=2),
    )

    act1 = policy.act(_make_formatted_obs())
    act2 = policy.act(_make_formatted_obs())
    act3 = policy.act(_make_formatted_obs())

    assert act1.shape == (14,)
    assert act2.shape == (14,)
    assert act3.shape == (14,)
    assert policy.pipeline.calls == 2
