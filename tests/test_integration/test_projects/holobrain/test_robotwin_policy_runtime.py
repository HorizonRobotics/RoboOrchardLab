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

import unittest
from unittest.mock import patch

import numpy as np

from projects.holobrain.policy import (
    HoloBrainRoboTwinPolicy,
    HoloBrainRoboTwinPolicyCfg,
)
from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipelineCfg,
)


class _StubPipeline:
    def __init__(self):
        self.cfg = HoloBrainInferencePipelineCfg()
        self.model = type("Model", (), {"eval": lambda self: None})()

    def reset(self, **kwargs):
        del kwargs


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
    def __init__(self, sensor_data: np.ndarray, intrinsic: np.ndarray, pose):
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic
        self.pose = pose


class RobotwinPolicyRuntimeTest(unittest.TestCase):
    def test_process_raw_obs_uses_float64_camera_matrices(self):
        policy = HoloBrainRoboTwinPolicy(
            cfg=HoloBrainRoboTwinPolicyCfg(),
            pipeline=_StubPipeline(),
        )
        obs = {
            "observation": {
                "front_camera": {
                    "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "depth": np.full((4, 4), 1000.0, dtype=np.float32),
                    "extrinsic_cv": np.eye(4, dtype=np.float32),
                    "intrinsic_cv": np.eye(3, dtype=np.float32),
                }
            },
            "joint_action": {"vector": np.arange(14, dtype=np.float32)},
            "instructions": "test instruction",
        }

        data = policy._process_raw_obs(obs)

        self.assertEqual(data.t_world2cam["front_camera"].dtype, np.float64)
        self.assertEqual(data.intrinsic["front_camera"].dtype, np.float64)
        self.assertAlmostEqual(
            float(data.depth["front_camera"][0][0, 0]),
            1.0,
        )

    def test_process_formatted_obs_converts_datatypes(self):
        policy = HoloBrainRoboTwinPolicy(
            cfg=HoloBrainRoboTwinPolicyCfg(),
            pipeline=_StubPipeline(),
        )
        pose = _FakePose(np.eye(4, dtype=np.float64))
        obs = {
            "cameras": {
                "front_camera": {
                    "rgb": _FakeCameraData(
                        sensor_data=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                        intrinsic=np.eye(3, dtype=np.float32)[None, ...],
                        pose=pose,
                    ),
                    "depth": _FakeCameraData(
                        sensor_data=np.full(
                            (1, 4, 4), 1000.0, dtype=np.float32
                        ),
                        intrinsic=np.eye(3, dtype=np.float32)[None, ...],
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

        data = policy._process_formatted_obs(obs)

        self.assertEqual(data.t_world2cam["front_camera"].dtype, np.float64)
        self.assertEqual(data.intrinsic["front_camera"].dtype, np.float64)
        self.assertEqual(data.image["front_camera"][0].shape, (4, 4, 3))
        self.assertEqual(data.history_joint_state[0].shape, (14,))
        self.assertAlmostEqual(
            float(data.depth["front_camera"][0][0, 0]),
            1.0,
        )

    def test_policy_loads_pipeline_from_model_dir(self):
        loaded_pipeline = _StubPipeline()
        eval_calls = {"count": 0}

        class _Model:
            def eval(self):
                eval_calls["count"] += 1

        loaded_pipeline.model = _Model()

        with patch(
            "projects.holobrain.policy.robotwin_policy."
            "HoloBrainInferencePipeline.load_pipeline",
            return_value=loaded_pipeline,
        ) as mock_load_pipeline:
            policy = HoloBrainRoboTwinPolicy(
                cfg=HoloBrainRoboTwinPolicyCfg(
                    model_dir="/tmp/model",
                    inference_prefix="robotwin2_0",
                ),
            )

        self.assertIs(policy.pipeline, loaded_pipeline)
        self.assertEqual(eval_calls["count"], 1)
        mock_load_pipeline.assert_called_once_with(
            directory="/tmp/model",
            inference_prefix="robotwin2_0",
            device="cpu",
            load_weights=True,
            load_impl="native",
            model_prefix="model",
        )


if __name__ == "__main__":
    unittest.main()
