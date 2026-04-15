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

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb  # noqa: E402
from robo_orchard_lab.dataset.robo_challenge_v2 import (  # noqa: E402
    RCV2LmdbPacker,
    rc_v2_lmdb_packer as packer_mod,
)


def _pose(
    xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> list[float]:
    return [*xyz, *quat_xyzw]


def _transform(translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _camera_matrix(
    fx: float = 500.0,
    fy: float = 510.0,
    cx: float = 320.0,
    cy: float = 240.0,
) -> list[list[float]]:
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _write_robot_bundle(
    root: Path,
    bundle_dir: str,
    robot_type: str,
    cameras: dict[str, dict],
) -> None:
    bundle_path = root / bundle_dir / "final_calibrations.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "robotType": robot_type,
                "calibrations": cameras,
            }
        )
    )


def _install_fake_video_decoder(
    monkeypatch: pytest.MonkeyPatch,
    frame_count_by_path: dict[str, int],
) -> None:
    def _fake_encode(
        video_path: str,
        jpeg_quality: int,
        max_frames: int,
        keep_mask=None,
    ):
        frame_count = frame_count_by_path[video_path]
        decoded_frame_count = min(frame_count, max_frames)
        if keep_mask is None:
            kept_indices = range(decoded_frame_count)
        else:
            kept_indices = [
                i for i in range(decoded_frame_count) if bool(keep_mask[i])
            ]
        jpeg_frames = [
            packer_mod._encode_rgb_frame_to_jpeg(
                np.full((8, 8, 3), fill_value=i, dtype=np.uint8),
                jpeg_quality,
            )
            for i in kept_indices
        ]
        return jpeg_frames, decoded_frame_count

    monkeypatch.setattr(
        packer_mod,
        "_encode_selected_video_frames",
        _fake_encode,
    )


def _build_single_arm_episode(
    root: Path,
    task_name: str,
    robot_id: str,
    features: dict,
    state_rows: list[dict],
    video_camera_names: list[str],
    task_tags: list[str] | None = None,
) -> Path:
    task_dir = root / task_name
    episode_dir = task_dir / "data" / "episode_000000"
    (episode_dir / "meta").mkdir(parents=True)
    (episode_dir / "states").mkdir(parents=True)
    (episode_dir / "videos").mkdir(parents=True)
    (task_dir / "task_desc.json").write_text(
        json.dumps(
            {
                "prompt": f"{task_name} prompt",
                "description": f"{task_name} description",
                "task_tag": task_tags or ["single-arm"],
            }
        )
    )
    (episode_dir / "meta" / "episode_meta.json").write_text(
        json.dumps(
            {
                "robot_id": robot_id,
                "frames": len(state_rows),
                "features": features,
            }
        )
    )
    _write_jsonl(episode_dir / "states" / "states.jsonl", state_rows)
    for camera_name in video_camera_names:
        (episode_dir / "videos" / f"{camera_name}_rgb.mp4").write_bytes(b"")
    return episode_dir


def _build_dual_arm_episode(
    root: Path,
    task_name: str,
    robot_id: str,
    features: dict,
    left_rows: list[dict],
    right_rows: list[dict],
    video_camera_names: list[str],
    task_tags: list[str] | None = None,
) -> Path:
    task_dir = root / task_name
    episode_dir = task_dir / "data" / "episode_000000"
    (episode_dir / "meta").mkdir(parents=True)
    (episode_dir / "states").mkdir(parents=True)
    (episode_dir / "videos").mkdir(parents=True)
    (task_dir / "task_desc.json").write_text(
        json.dumps(
            {
                "prompt": f"{task_name} prompt",
                "description": f"{task_name} description",
                "task_tag": task_tags or ["dual-arm"],
            }
        )
    )
    (episode_dir / "meta" / "episode_meta.json").write_text(
        json.dumps(
            {
                "robot_id": robot_id,
                "frames": len(left_rows),
                "features": features,
            }
        )
    )
    _write_jsonl(episode_dir / "states" / "left_states.jsonl", left_rows)
    _write_jsonl(episode_dir / "states" / "right_states.jsonl", right_rows)
    for camera_name in video_camera_names:
        (episode_dir / "videos" / f"{camera_name}_rgb.mp4").write_bytes(b"")
    return episode_dir


def _read_lmdb(output_root: Path, name: str) -> Lmdb:
    return Lmdb(uri=os.path.join(output_root, name), writable=False)


class TestRCV2LmdbPacker:
    def test_rigid_inverse_helpers_match_numpy_inverse(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = packer_mod.Rotation.from_euler(
            "xyz",
            [0.2, -0.1, 0.3],
        ).as_matrix()
        transform[:3, 3] = np.array([0.4, -0.2, 0.7], dtype=np.float64)
        transforms = np.stack([transform, transform.copy()], axis=0)
        transforms[1, :3, 3] = np.array([0.1, 0.2, -0.3], dtype=np.float64)

        assert np.allclose(
            packer_mod._invert_rigid_transform(transform),
            np.linalg.inv(transform),
        )
        assert np.allclose(
            packer_mod._invert_rigid_transform_sequence(transforms),
            np.linalg.inv(transforms),
        )

    def test_load_jsonl_states_rejects_partial_field_rows(
        self,
        tmp_path: Path,
    ) -> None:
        states_path = tmp_path / "states.jsonl"
        _write_jsonl(
            states_path,
            [
                {
                    "joint_positions": [0.0] * 6,
                    "gripper_width": 0.08,
                    "timestamp": 0.0,
                },
                {
                    "gripper_width": 0.08,
                    "timestamp": 1.0,
                },
            ],
        )

        with pytest.raises(ValueError, match="joint_positions"):
            packer_mod._load_jsonl_states(
                str(states_path),
                packer_mod.SINGLE_ARM_STATE_FIELDS,
            )

    def test_dual_arm_layout_is_driven_by_task_tags(self) -> None:
        assert packer_mod._is_dual_arm_task({"task_tag": ["dual-arm"]})
        assert not packer_mod._is_dual_arm_task(
            {"task_tag": ["single-arm", "dos-w1"]}
        )

        with pytest.raises(ValueError, match="exactly one"):
            packer_mod._is_dual_arm_task({"task_tag": ["w1"]})

    def test_single_arm_ur5_alias_and_instruction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        calibration_root = tmp_path / "calibration"

        cam_in_base = _transform((1.0, 0.0, 0.0))
        cam_in_ee = _transform((0.1, 0.0, 0.0))
        states = [
            {
                "joint_positions": [0.0] * 6,
                "ee_positions": _pose((0.0, 0.0, 0.5)),
                "gripper_width": 0.08,
                "timestamp": float(i),
            }
            for i in range(3)
        ]
        states[1]["joint_positions"][0] = 0.1
        states[2]["joint_positions"][0] = 0.2
        features = {
            "cam_global": {
                "intrinsics": _camera_matrix(),
                "extrinsics": {"arms": {"arm": cam_in_base.tolist()}},
            },
            "cam_side": {
                "intrinsics": [],
                "extrinsics": {"arms": {}},
            },
        }
        episode_dir = _build_single_arm_episode(
            input_root,
            task_name="task_single",
            robot_id="rc_ur5_1",
            features=features,
            state_rows=states,
            video_camera_names=["cam_global", "cam_side"],
        )
        _write_robot_bundle(
            calibration_root,
            bundle_dir="ur5",
            robot_type="ur5",
            cameras={
                "cam_arm": {
                    "cameraId": "cam_arm",
                    "referenceFrame": "ee",
                    "intrinsics": {
                        "fx": 610.0,
                        "fy": 620.0,
                        "cx": 300.0,
                        "cy": 200.0,
                    },
                    "pose": {"matrix4": cam_in_ee.tolist()},
                }
            },
        )
        _install_fake_video_decoder(
            monkeypatch,
            {
                str(episode_dir / "videos" / "cam_global_rgb.mp4"): 3,
                str(episode_dir / "videos" / "cam_side_rgb.mp4"): 3,
            },
        )

        packer = RCV2LmdbPacker(
            input_path=str(input_root),
            output_path=str(output_root),
            static_threshold=None,
            robot_calibration_root=str(calibration_root),
        )
        packer()

        meta_lmdb = _read_lmdb(output_root, "meta")
        index_lmdb = _read_lmdb(output_root, "index")
        try:
            uuid = "task_single_episode000000"
            camera_names = meta_lmdb[f"{uuid}/camera_names"]
            intrinsic = meta_lmdb[f"{uuid}/intrinsic"]
            extrinsic = meta_lmdb[f"{uuid}/extrinsic"]
            calibration = meta_lmdb[f"{uuid}/calibration"]
            meta_data = meta_lmdb[f"{uuid}/meta_data"]
            index_data = index_lmdb[0]

            assert camera_names == ["cam_global", "cam_arm"]
            assert index_data["embodiment"] == "ur5"
            assert meta_data["embodiment"] == "ur5"
            assert meta_data["instruction"] == "task_single prompt"
            assert meta_data["description"] == "task_single description"
            assert meta_lmdb[f"{uuid}/instructions"] is None
            assert np.allclose(calibration["cam_global"], cam_in_base)
            assert np.allclose(
                extrinsic["cam_global"],
                np.linalg.inv(cam_in_base),
            )
            assert np.allclose(calibration["cam_arm"], cam_in_ee)
            assert np.allclose(
                intrinsic["cam_arm"],
                np.array(
                    [
                        [610.0, 0.0, 300.0],
                        [0.0, 620.0, 200.0],
                        [0.0, 0.0, 1.0],
                    ],
                ),
            )
            assert extrinsic["cam_arm"].shape == (3, 4, 4)
        finally:
            meta_lmdb.close()
            index_lmdb.close()

    def test_missing_wrist_calibration_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        calibration_root = tmp_path / "empty_calibration"

        states = [
            {
                "joint_positions": [0.0] * 6,
                "ee_positions": _pose((0.0, 0.0, 0.5)),
                "gripper_width": 0.08,
                "timestamp": 0.0,
            },
            {
                "joint_positions": [0.1] * 6,
                "ee_positions": _pose((0.1, 0.0, 0.5)),
                "gripper_width": 0.08,
                "timestamp": 1.0,
            },
        ]
        features = {
            "cam_arm": {
                "intrinsics": [],
                "extrinsics": {"arms": {}},
            }
        }
        episode_dir = _build_single_arm_episode(
            input_root,
            task_name="task_missing_calibration",
            robot_id="rc_arx5_1",
            features=features,
            state_rows=states,
            video_camera_names=["cam_arm"],
        )
        _install_fake_video_decoder(
            monkeypatch,
            {str(episode_dir / "videos" / "cam_arm_rgb.mp4"): 2},
        )

        packer = RCV2LmdbPacker(
            input_path=str(input_root),
            output_path=str(output_root),
            static_threshold=None,
            robot_calibration_root=str(calibration_root),
        )
        packer._init_lmdbs()
        try:
            with pytest.raises(ValueError, match="missing calibration"):
                packer._pack_one_episode(
                    ep_idx=0,
                    uuid="task_missing_calibration_episode000000",
                    task_name="task_missing_calibration",
                    ep_id="000000",
                    ep_dir=str(episode_dir),
                )
        finally:
            packer.close()

    def test_dual_arm_right_wrist_extrinsic_and_embodiment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        calibration_root = tmp_path / "calibration"

        cam_in_left = _transform((1.0, 0.0, 0.0))
        cam_in_right = _transform((0.5, 0.0, 0.0))
        cam_in_ee = _transform((0.1, 0.0, 0.0))
        left_rows = [
            {
                "joint_positions": [0.0] * 6,
                "gripper_width": 0.0,
                "ee_positions": _pose((0.0, 0.0, 0.3)),
                "timestamp": 0.0,
            },
            {
                "joint_positions": [0.1] * 6,
                "gripper_width": 0.0,
                "ee_positions": _pose((0.0, 0.0, 0.3)),
                "timestamp": 1.0,
            },
        ]
        right_rows = [
            {
                "joint_positions": [0.0] * 6,
                "gripper_width": 0.0,
                "ee_positions": _pose((0.2, 0.0, 0.3)),
                "timestamp": 0.0,
            },
            {
                "joint_positions": [0.1] * 6,
                "gripper_width": 0.0,
                "ee_positions": _pose((0.25, 0.0, 0.3)),
                "timestamp": 1.0,
            },
        ]
        features = {
            "cam_high": {
                "intrinsics": _camera_matrix(),
                "extrinsics": {
                    "arms": {
                        "left": cam_in_left.tolist(),
                        "right": cam_in_right.tolist(),
                    }
                },
            },
            "cam_right_wrist": {
                "intrinsics": [],
                "extrinsics": {"arms": {}},
            },
        }
        episode_dir = _build_dual_arm_episode(
            input_root,
            task_name="task_dual",
            robot_id="rc_aloha_1",
            features=features,
            left_rows=left_rows,
            right_rows=right_rows,
            video_camera_names=["cam_high", "cam_right_wrist"],
        )
        _write_robot_bundle(
            calibration_root,
            bundle_dir="aloha",
            robot_type="aloha",
            cameras={
                "cam_right_wrist": {
                    "cameraId": "cam_right_wrist",
                    "referenceFrame": "ee",
                    "intrinsics": {
                        "fx": 430.0,
                        "fy": 435.0,
                        "cx": 215.0,
                        "cy": 120.0,
                    },
                    "pose": {"matrix4": cam_in_ee.tolist()},
                }
            },
        )
        _install_fake_video_decoder(
            monkeypatch,
            {
                str(episode_dir / "videos" / "cam_high_rgb.mp4"): 2,
                str(episode_dir / "videos" / "cam_right_wrist_rgb.mp4"): 2,
            },
        )

        packer = RCV2LmdbPacker(
            input_path=str(input_root),
            output_path=str(output_root),
            static_threshold=None,
            robot_calibration_root=str(calibration_root),
        )
        packer()

        meta_lmdb = _read_lmdb(output_root, "meta")
        index_lmdb = _read_lmdb(output_root, "index")
        try:
            uuid = "task_dual_episode000000"
            extrinsic = meta_lmdb[f"{uuid}/extrinsic"]
            calibration = meta_lmdb[f"{uuid}/calibration"]

            right_base_in_left = cam_in_left @ np.linalg.inv(cam_in_right)
            right_ee_0 = packer_mod._quat_pose_to_matrix(
                np.asarray(right_rows[0]["ee_positions"], dtype=np.float64)
            )
            expected = np.linalg.inv(
                right_base_in_left @ right_ee_0 @ cam_in_ee
            )

            assert index_lmdb[0]["embodiment"] == "aloha"
            assert meta_lmdb[f"{uuid}/meta_data"]["embodiment"] == "aloha"
            assert np.allclose(calibration["cam_high"], cam_in_left)
            assert np.allclose(
                extrinsic["cam_high"],
                np.linalg.inv(cam_in_left),
            )
            assert extrinsic["cam_right_wrist"].shape == (2, 4, 4)
            assert np.allclose(extrinsic["cam_right_wrist"][0], expected)
        finally:
            meta_lmdb.close()
            index_lmdb.close()

    def test_w1_normalizes_to_dos_w1_and_uses_dual_arm_layout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        calibration_root = tmp_path / "calibration"

        cam_in_left = _transform((0.8, 0.0, 0.0))
        cam_in_right = _transform((0.3, 0.0, 0.0))
        cam_in_ee = _transform((0.05, 0.0, 0.0))
        left_rows = [
            {
                "joint_positions": [0.0] * 6,
                "ee_positions": _pose((0.0, 0.0, 0.4)),
                "gripper_width": 0.04,
                "timestamp": 0.0,
            },
            {
                "joint_positions": [0.0] * 6,
                "ee_positions": _pose((0.1, 0.0, 0.4)),
                "gripper_width": 0.04,
                "timestamp": 1.0,
            },
        ]
        right_rows = [
            {
                "joint_positions": [0.0] * 6,
                "ee_positions": _pose((0.2, 0.0, 0.4)),
                "gripper_width": 0.05,
                "timestamp": 0.0,
            },
            {
                "joint_positions": [0.1] * 6,
                "ee_positions": _pose((0.25, 0.0, 0.4)),
                "gripper_width": 0.05,
                "timestamp": 1.0,
            },
        ]
        features = {
            "cam_high": {
                "intrinsics": _camera_matrix(),
                "extrinsics": {
                    "arms": {
                        "left": cam_in_left.tolist(),
                        "right": cam_in_right.tolist(),
                    }
                },
            },
            "cam_right_wrist": {
                "intrinsics": [],
                "extrinsics": {"arms": {}},
            },
        }
        episode_dir = _build_dual_arm_episode(
            input_root,
            task_name="task_w1",
            robot_id="rc_w1_1",
            features=features,
            left_rows=left_rows,
            right_rows=right_rows,
            video_camera_names=["cam_high", "cam_right_wrist"],
        )
        _write_robot_bundle(
            calibration_root,
            bundle_dir="dos-w1",
            robot_type="w1",
            cameras={
                "cam_right_wrist": {
                    "cameraId": "cam_right_wrist",
                    "referenceFrame": "ee",
                    "intrinsics": {
                        "fx": 520.0,
                        "fy": 525.0,
                        "cx": 250.0,
                        "cy": 180.0,
                    },
                    "pose": {"matrix4": cam_in_ee.tolist()},
                }
            },
        )
        _install_fake_video_decoder(
            monkeypatch,
            {
                str(episode_dir / "videos" / "cam_high_rgb.mp4"): 2,
                str(episode_dir / "videos" / "cam_right_wrist_rgb.mp4"): 2,
            },
        )

        packer = RCV2LmdbPacker(
            input_path=str(input_root),
            output_path=str(output_root),
            robot_calibration_root=str(calibration_root),
        )
        packer()

        meta_lmdb = _read_lmdb(output_root, "meta")
        index_lmdb = _read_lmdb(output_root, "index")
        try:
            uuid = "task_w1_episode000000"
            camera_names = meta_lmdb[f"{uuid}/camera_names"]
            calibration = meta_lmdb[f"{uuid}/calibration"]
            extrinsic = meta_lmdb[f"{uuid}/extrinsic"]
            right_ee_positions = meta_lmdb[
                f"{uuid}/observation/robot_state/right/ee_positions"
            ]
            joint_positions = meta_lmdb[
                f"{uuid}/observation/robot_state/joint_positions"
            ]
            cartesian_position = meta_lmdb[
                f"{uuid}/observation/robot_state/cartesian_position"
            ]

            right_base_in_left = cam_in_left @ np.linalg.inv(cam_in_right)
            right_ee_0 = packer_mod._quat_pose_to_matrix(
                np.asarray(right_rows[0]["ee_positions"], dtype=np.float64)
            )
            expected_right_wrist_extrinsic = np.linalg.inv(
                right_base_in_left @ right_ee_0 @ cam_in_ee
            )

            assert camera_names == ["cam_high", "cam_right_wrist"]
            assert index_lmdb[0]["embodiment"] == "dos-w1"
            assert meta_lmdb[f"{uuid}/meta_data"]["embodiment"] == "dos-w1"
            assert np.allclose(calibration["cam_high"], cam_in_left)
            assert np.allclose(calibration["cam_right_wrist"], cam_in_ee)
            assert right_ee_positions.shape == (2, 7)
            assert joint_positions.shape == (2, 14)
            assert cartesian_position.shape == (2, 14)
            assert np.allclose(
                joint_positions[0],
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.04,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.05,
                    ],
                    dtype=np.float64,
                ),
            )
            assert np.allclose(
                extrinsic["cam_right_wrist"][0],
                expected_right_wrist_extrinsic,
            )
        finally:
            meta_lmdb.close()
            index_lmdb.close()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
