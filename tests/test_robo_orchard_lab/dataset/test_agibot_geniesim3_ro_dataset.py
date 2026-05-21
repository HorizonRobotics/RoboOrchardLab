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

from types import SimpleNamespace

import numpy as np

from robo_orchard_lab.dataset.agibot_geniesim import (
    agibot_geniesim3_ro_dataset,
)
from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    SimpleStateSampling,
)


def _joint_item(values: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(position=np.asarray(values, dtype=np.float64)[None])


def test_geniesim3_gripper_units_stay_normalized_after_master_sampling():
    parser = agibot_geniesim3_ro_dataset.ArrowDataParse(
        cam_names=[],
        load_image=False,
        load_depth=False,
        load_extrinsic=False,
        gripper_indices=[7, 15],
        gripper_divisor=120.0,
    )
    joint_state = np.zeros((2, 16), dtype=np.float64)
    joint_state[:, 7] = [120.0, 30.0]
    joint_state[:, 15] = [60.0, 90.0]
    master_joint_state = np.zeros((2, 16), dtype=np.float64)
    master_joint_state[:, 7] = [0.25, 0.50]
    master_joint_state[:, 15] = [0.75, 0.00]

    data = {
        "joints": [_joint_item(row) for row in joint_state],
        "actions": [_joint_item(row) for row in master_joint_state],
    }
    parsed = {}
    parsed.update(parser.get_joints(data))
    parsed.update(parser.get_master_joints(data))
    parsed["step_index"] = 0

    np.testing.assert_allclose(
        parsed["joint_state"][:, [7, 15]],
        [[1.0, 0.5], [0.25, 0.75]],
    )
    np.testing.assert_allclose(
        parsed["master_joint_state"][:, [7, 15]],
        [[0.25, 0.75], [0.50, 0.00]],
    )

    sampled = SimpleStateSampling(
        hist_steps=1,
        pred_steps=1,
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=[7, 15],
        limitation=1000,
        static_threshold=0,
    )(parsed)

    np.testing.assert_allclose(
        sampled["hist_joint_state"][0, [7, 15]],
        [1.0, 0.5],
    )
    np.testing.assert_allclose(
        sampled["pred_joint_state"][0, [7, 15]],
        [0.50, 0.00],
    )
