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

import os

import cv2
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.dataset.robocasa.robocasa_lmdb_dataset import (
    RoboCasaLmdbDataset,
)
from robo_orchard_lab.dataset.robocasa.robocasa_lmdb_packer import (
    DEFAULT_CAMERA_CONFIGS,
    DEFAULT_EEF_TO_HAND,
    STATE_SLICES,
    camera_axis_correction,
    camera_calibration_from_config,
    make_pose,
    pose_inv,
    quat_wxyz_to_mat,
    state_base_pose,
    state_camera_world_pose,
    t_base2cam_from_world_pose,
)
from robo_orchard_lab.dataset.robocasa.transforms import (
    SimpleStateSampling,
    TransformRobotState,
)


class RoboCasaTestPacker(BaseLmdbManipulationDataPacker):
    def _pack(self):
        uuid = "robocasa_test_episode"
        num_steps = 4
        cam_names = ["robot0_agentview_left", "robot0_eye_in_hand"]
        self.write_index(
            0,
            {
                "uuid": uuid,
                "num_steps": num_steps,
                "task_name": "OpenOven",
                "simulation": True,
            },
        )
        self.index_pack_file.write("__len__", 1)

        self.meta_pack_file.write(f"{uuid}/camera_names", cam_names)
        self.meta_pack_file.write(f"{uuid}/instruction", "Open the oven door.")
        self.meta_pack_file.write(
            f"{uuid}/intrinsic",
            {cam: np.eye(3, dtype=np.float64) for cam in cam_names},
        )
        self.meta_pack_file.write(
            f"{uuid}/base2cam",
            {
                cam: np.tile(np.eye(4, dtype=np.float64), (num_steps, 1, 1))
                for cam in cam_names
            },
        )
        self.meta_pack_file.write(
            f"{uuid}/calibration",
            {cam: np.eye(4, dtype=np.float64) for cam in cam_names},
        )
        ee_state = np.array(
            [
                [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.meta_pack_file.write(f"{uuid}/observation/ee_state", ee_state)
        self.meta_pack_file.write(
            f"{uuid}/observation/gripper_state",
            np.array(
                [
                    [0.04, -0.04],
                    [0.03, -0.03],
                    [0.02, -0.02],
                    [0.01, -0.01],
                ],
                dtype=np.float64,
            ),
        )
        self.meta_pack_file.write(
            f"{uuid}/action/gripper",
            np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float64),
        )
        self.meta_pack_file.write(
            f"{uuid}/action/osc_action",
            np.zeros((num_steps, 6), dtype=np.float64),
        )

        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        encoded = cv2.imencode(".png", image)[1]
        for cam in cam_names:
            for step_idx in range(num_steps):
                self.image_pack_file.write(f"{uuid}/{cam}/{step_idx}", encoded)


def test_robocasa_lmdb_dataset_reads_episode_and_applies_sampling(tmp_path):
    lmdb_path = os.path.join(tmp_path, "robocasa_lmdb")
    packer = RoboCasaTestPacker(input_path="", output_path=lmdb_path)
    try:
        packer()
    finally:
        packer.close()

    meta_lmdb = Lmdb(
        uri=os.path.join(lmdb_path, "meta"),
        writable=False,
        encoding_mode="utf-8",
    )
    try:
        assert meta_lmdb["robocasa_test_episode/base2cam"] is not None
        assert meta_lmdb["robocasa_test_episode/calibration"] is not None
        assert meta_lmdb["robocasa_test_episode/extrinsic"] is None
    finally:
        meta_lmdb.close()

    dataset = RoboCasaLmdbDataset(
        paths=lmdb_path,
        transforms=[SimpleStateSampling(hist_steps=2, pred_steps=3)],
        load_depth=False,
    )

    assert len(dataset) == 4
    data = dataset[2]

    assert data["uuid"] == "robocasa_test_episode"
    assert data["text"] == "Open the oven door."
    assert len(data["imgs"]) == 2
    assert data["imgs"][0].shape == (8, 10, 3)
    assert data["intrinsic"].shape == (2, 4, 4)
    assert data["T_base2cam"].shape == (2, 4, 4)
    assert "T_world2cam" not in data
    assert data["hist_robot_state"].shape == (2, 1, 8)
    assert data["pred_robot_state"].shape == (3, 1, 8)
    assert data["pred_mask"].tolist() == [True, True, False]

    np.testing.assert_allclose(
        data["hist_robot_state"][:, 0, 0],
        np.array([-0.75, -0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        data["pred_robot_state"][:, 0, 0],
        np.array([1.0, -1.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        data["pred_robot_state"][0, 0, 1:4],
        np.array([0.2, 0.0, 0.2], dtype=np.float32),
    )


def test_transform_robot_state_applies_embodiment_matrix():
    robot_state = torch.tensor(
        [
            [[0.25, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            [[-0.5, 0.0, 1.0, 0.0, 0.70710678, 0.0, 0.0, 0.70710678]],
        ],
        dtype=torch.float64,
    )
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    transform[:3, 3] = torch.tensor([0.5, -0.25, 2.0], dtype=torch.float32)

    output = TransformRobotState()._apply_transform(robot_state, transform)
    transform = transform.to(robot_state)
    expected_pos = (
        transform[:3, :3] @ robot_state[..., 1:4].reshape(-1, 3).T
    ).T + transform[:3, 3]
    expected_rot = transform[:3, :3] @ quaternion_to_matrix(
        robot_state[..., 4:].reshape(-1, 4)
    )

    assert output.shape == robot_state.shape
    assert output.dtype == robot_state.dtype
    torch.testing.assert_close(output[..., :1], robot_state[..., :1])
    torch.testing.assert_close(output[..., 1:4].reshape(-1, 3), expected_pos)
    torch.testing.assert_close(
        quaternion_to_matrix(output[..., 4:].reshape(-1, 4)),
        expected_rot,
    )


def test_robocasa_wrist_camera_uses_fixed_hand_to_eef_offset():
    state = np.zeros(16, dtype=np.float64)
    state[STATE_SLICES["base_position"]] = [0.2, -0.1, 0.0]
    state[STATE_SLICES["base_rotation_xyzw"]] = [0.0, 0.0, 0.0, 1.0]
    state[STATE_SLICES["eef_position_relative"]] = [0.4, 0.1, 0.5]
    state[STATE_SLICES["eef_rotation_relative_xyzw"]] = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    camera = "robot0_eye_in_hand"
    cam_cfg = DEFAULT_CAMERA_CONFIGS[camera]

    base_pos, base_rot = state_base_pose(state)
    eef_pos = base_pos + state[STATE_SLICES["eef_position_relative"]]
    t_hand_to_cam = make_pose(
        np.asarray(cam_cfg["pos"], dtype=np.float64),
        quat_wxyz_to_mat(np.asarray(cam_cfg["quat"], dtype=np.float64)),
    )
    expected_cam_to_world = (
        make_pose(eef_pos, np.eye(3))
        @ DEFAULT_EEF_TO_HAND
        @ t_hand_to_cam
        @ camera_axis_correction()
    )

    cam_pos, cam_rot = state_camera_world_pose(state, camera, cam_cfg)
    actual_base2cam = t_base2cam_from_world_pose(
        cam_pos,
        cam_rot,
        base_pos,
        base_rot,
    )
    expected_base2cam = pose_inv(expected_cam_to_world) @ make_pose(
        base_pos,
        base_rot,
    )
    previous_base2cam_without_offset = pose_inv(
        make_pose(eef_pos, np.eye(3))
        @ make_pose(np.asarray(cam_cfg["pos"], dtype=np.float64), np.eye(3))
        @ camera_axis_correction()
    ) @ make_pose(base_pos, base_rot)

    np.testing.assert_allclose(actual_base2cam, expected_base2cam, atol=1e-6)
    assert not np.allclose(actual_base2cam, previous_base2cam_without_offset)


def test_robocasa_camera_calibration_semantics():
    wrist_calibration = camera_calibration_from_config(
        DEFAULT_CAMERA_CONFIGS["robot0_eye_in_hand"]
    )
    expected_wrist_calibration = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.097],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(wrist_calibration, expected_wrist_calibration)

    camera = "robot0_agentview_left"
    cam_cfg = DEFAULT_CAMERA_CONFIGS[camera]
    state = np.zeros(16, dtype=np.float64)
    state[STATE_SLICES["base_rotation_xyzw"]] = [0.0, 0.0, 0.0, 1.0]
    state[STATE_SLICES["eef_rotation_relative_xyzw"]] = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    base_pos, base_rot = state_base_pose(state)
    cam_pos, cam_rot = state_camera_world_pose(state, camera, cam_cfg)
    expected_static_calibration = t_base2cam_from_world_pose(
        cam_pos,
        cam_rot,
        base_pos,
        base_rot,
    )
    np.testing.assert_allclose(
        camera_calibration_from_config(cam_cfg),
        expected_static_calibration,
    )
