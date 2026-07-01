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

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)
from robo_orchard_lab.utils.build import build


def test_robocasa_holobrain_config_supports_validation_and_deploy_modes():
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "projects/holobrain_internal/common/configs"
    sys.path.insert(0, str(config_dir))
    try:
        from data_configs.config_robocasa_dataset import (
            build_processors,
            build_transforms,
        )
    finally:
        sys.path.remove(str(config_dir))

    config = {
        "hist_steps": 2,
        "pred_steps": 4,
        "with_depth": False,
        "dst_wh": (16, 12),
    }

    training_transforms = build_transforms(config, "training")
    validation_transforms = build_transforms(config, "validation")
    deploy_transforms = build_transforms(config, "deploy")

    assert "state_loss_weights" in training_transforms[-1]["keys"]
    assert "fk_loss_weight" in training_transforms[-1]["keys"]
    assert "uuid" in validation_transforms[-1]["keys"]
    assert "pred_mask" in validation_transforms[-1]["keys"]
    assert "uuid" not in deploy_transforms[-2]["keys"]
    assert "pred_mask" not in deploy_transforms[-2]["keys"]
    assert deploy_transforms[-1]["type"].__name__ == "UnsqueezeBatch"

    for transforms in (
        training_transforms,
        validation_transforms,
        deploy_transforms,
    ):
        assert all(callable(build(transform)) for transform in transforms)

    processor = build_processors(config, dataset_name="robocasa")
    expected_transform_names = [
        type(build(transform)).__name__ for transform in deploy_transforms
    ]
    assert [
        type(transform).__name__ for transform in processor.transforms
    ] == expected_transform_names

    cam_names = processor.cfg.cam_names
    assert cam_names is not None
    sample = MultiArmManipulationInput(
        intrinsic={cam_name: np.eye(4) for cam_name in cam_names},
        t_base2cam={cam_name: np.eye(4) for cam_name in cam_names},
        history_joint_state=[
            np.zeros((1, 8), dtype=np.float32),
            np.zeros((1, 8), dtype=np.float32),
        ],
        image={
            cam_name: [np.zeros((8, 10, 3), dtype=np.uint8)]
            for cam_name in cam_names
        },
        instruction="Open the oven door.",
    )

    input_data = processor.struction_to_dict(sample)
    assert "T_base2cam" in input_data
    assert "T_world2cam" not in input_data

    batch = processor.pre_process(sample)

    assert batch["imgs"].shape == (1, len(cam_names), 12, 16, 3)
    assert batch["hist_robot_state"].shape == (1, 2, 1, 8)
    assert batch["pred_robot_state"].shape == (1, 4, 1, 8)
    assert batch["projection_mat"].shape == (1, len(cam_names), 4, 4)
    assert batch["embodiedment_mat"].shape == (1, 4, 4)
    assert batch["text"] == ["Open the oven door."]


def test_robocasa_policy_converts_model_pose_to_absolute_env_action():
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    policy_dir = repo_root / (
        "projects/holobrain_internal/common/holobrain_robocasa_policy"
    )
    sys.path.insert(0, str(common_dir))
    sys.path.insert(0, str(policy_dir))
    try:
        from deploy_policy import (
            convert_ee_poses_to_robocasa_actions,
            extract_eef_body_to_site_rot,
        )
    finally:
        sys.path.remove(str(policy_dir))
        sys.path.remove(str(common_dir))

    current_robot_state = np.array(
        [[0.5, 9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target_quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    target_robot_state = np.array(
        [
            [0.25, 1.25, -2.5, 0.75, *target_quat_wxyz],
            [1.25, -1.5, 2.0, 3.0, *target_quat_wxyz],
        ],
        dtype=np.float32,
    )

    actions = convert_ee_poses_to_robocasa_actions(
        current_robot_state=current_robot_state,
        target_robot_state=target_robot_state,
        valid_action_step=1,
        control_mode=1.0,
    )

    expected_rotvec = Rotation.from_quat(
        target_quat_wxyz[[1, 2, 3, 0]]
    ).as_rotvec()
    assert actions.shape == (1, 12)
    np.testing.assert_allclose(actions[0, :3], [1.25, -2.5, 0.75])
    np.testing.assert_allclose(actions[0, 3:6], expected_rotvec)
    np.testing.assert_allclose(actions[0, 6], 0.25)
    np.testing.assert_allclose(actions[0, 7:11], 0.0)
    np.testing.assert_allclose(actions[0, 11], 1.0)

    body_to_site_rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    actions = convert_ee_poses_to_robocasa_actions(
        current_robot_state=current_robot_state,
        target_robot_state=target_robot_state,
        valid_action_step=1,
        control_mode=1.0,
        eef_body_to_site_rot=body_to_site_rot,
    )
    expected_rot = (
        Rotation.from_quat(target_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        @ body_to_site_rot
    )
    np.testing.assert_allclose(
        actions[0, 3:6],
        Rotation.from_matrix(expected_rot).as_rotvec(),
    )

    class DummySimData:
        def get_body_xmat(self, body_name):
            assert body_name == "robot0_right_hand"
            return np.eye(3)

        def get_site_xmat(self, site_name):
            assert site_name == "gripper0_right_grip_site"
            return body_to_site_rot

    class DummySim:
        data = DummySimData()

    class DummyRobotModel:
        eef_name = {"right": "robot0_right_hand"}

    class DummyGripper:
        important_sites = {"grip_site": "gripper0_right_grip_site"}

    class DummyRobot:
        robot_model = DummyRobotModel()
        gripper = {"right": DummyGripper()}

    class DummyInnerEnv:
        sim = DummySim()
        robots = [DummyRobot()]

    class DummyUnwrappedEnv:
        env = DummyInnerEnv()

    class DummyEnv:
        unwrapped = DummyUnwrappedEnv()

    np.testing.assert_allclose(
        extract_eef_body_to_site_rot(DummyEnv()),
        body_to_site_rot,
    )


def test_robocasa_eval_configures_env_for_absolute_base_osc():
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    sys.path.insert(0, str(common_dir))
    try:
        from robocasa_eval import configure_robocasa_env_absolute_action
    finally:
        sys.path.remove(str(common_dir))

    class DummyController:
        name_suffix = "POSE"
        input_type = "delta"
        input_ref_frame = "base"
        control_dim = 6
        input_min = np.full(6, -1.0)
        input_max = np.full(6, 1.0)

    controller = DummyController()

    class DummyCompositeController:
        def get_controller(self, part_name):
            assert part_name == "right"
            return controller

    class DummyRobot:
        composite_controller = DummyCompositeController()

    class DummyInnerEnv:
        robots = [DummyRobot()]

    class DummyUnwrappedEnv:
        env = DummyInnerEnv()

    class DummyEnv:
        unwrapped = DummyUnwrappedEnv()

    configure_robocasa_env_absolute_action(DummyEnv())

    assert controller.input_type == "absolute"
    assert controller.input_ref_frame == "base"
    np.testing.assert_allclose(controller.input_min, -np.inf)
    np.testing.assert_allclose(controller.input_max, np.inf)
