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

from __future__ import annotations
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import robo_orchard_lab.envs.robotwin.curobo_base_patch as curobo_base_patch
from robo_orchard_lab.envs.robotwin.curobo_base_patch import (
    RoboTwinCuroboPatchUnsupportedError,
    _install_robotwin_curobo_base_transform_patch,
    _parse_entity_to_base_from_curobo_yml,
    prepare_robotwin_runtime_for_cfg,
    setup_robotwin_demo_with_runtime_guards,
)

pytestmark = pytest.mark.sim_env


def _write_curobo_assets(
    tmp_path: Path,
    *,
    base_link: str = "fl_base_link",
    xyz: tuple[float, float, float] = (0.2305, 0.297, 0.782),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.02),
    frame_bias: tuple[float, float, float] = (-0.2305, -0.297, -0.782),
) -> Path:
    urdf_path = tmp_path / "robot.urdf"
    yml_path = tmp_path / "curobo_left.yml"
    urdf_path.write_text(
        f"""
<robot name="fake">
  <link name="footprint"/>
  <link name="{base_link}"/>
  <joint name="{base_link}_joint" type="fixed">
    <parent link="footprint"/>
    <child link="{base_link}"/>
    <origin xyz="{" ".join(str(v) for v in xyz)}"
            rpy="{" ".join(str(v) for v in rpy)}"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    yml_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "urdf_path": str(urdf_path),
                        "base_link": base_link,
                    }
                },
                "planner": {"frame_bias": list(frame_bias)},
            }
        ),
        encoding="utf-8",
    )
    return yml_path


def _assert_quat_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-6,
) -> None:
    alignment = abs(float(np.dot(actual, expected)))
    assert np.isclose(alignment, 1.0, atol=atol)


def _urdf_fixed_axis_rpy_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    roll_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    pitch_mat = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    yaw_mat = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return yaw_mat @ pitch_mat @ roll_mat


class _FakeNativeAlohaCuroboPlanner:
    def plan_path(self) -> None:
        if "aloha-agilex" in self.yml_path:
            rot_marker = "axangle2mat"
            transform_marker = "T_rot @ T_bias @ T_target"
            assert rot_marker and transform_marker

    def plan_batch(self) -> None:
        if "aloha-agilex" in self.yml_path:
            rot_marker = "axangle2mat"
            transform_marker = "T_rot @ T_bias @ T_target"
            assert rot_marker and transform_marker


class _FakeLegacyCuroboPlanner:
    def plan_path(self) -> None:
        target_pose_p = [0.0, 0.0, 0.0]
        target_pose_p[0] += self.frame_bias[0]

    def plan_batch(self) -> None:
        base_target_pose_p = [0.0, 0.0, 0.0]
        base_target_pose_p[0] += self.frame_bias[0]


def _install_fake_robotwin_planner_module(
    monkeypatch: pytest.MonkeyPatch,
    planner_cls: type,
) -> None:
    envs_module = ModuleType("envs")
    robot_module = ModuleType("envs.robot")
    planner_module = ModuleType("envs.robot.planner")
    planner_module.__dict__["CuroboPlanner"] = planner_cls
    robot_module.__dict__["planner"] = planner_module
    monkeypatch.setitem(sys.modules, "envs", envs_module)
    monkeypatch.setitem(sys.modules, "envs.robot", robot_module)
    monkeypatch.setitem(sys.modules, "envs.robot.planner", planner_module)


def _native_aloha_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        action_type="ee",
        patch_curobo_base_transform=True,
        get_task_config=lambda: {"embodiment_name": "aloha-agilex"},
    )


def test_parse_entity_to_base_from_curobo_yml_reads_urdf_fixed_joint(
    tmp_path: Path,
) -> None:
    yml_path = _write_curobo_assets(tmp_path)

    entity_to_base = _parse_entity_to_base_from_curobo_yml(str(yml_path))

    np.testing.assert_allclose(
        entity_to_base.entity_to_base_xyz,
        np.array([0.2305, 0.297, 0.782]),
    )
    expected_yaw = 0.02
    expected_rotation = np.array(
        [
            [np.cos(expected_yaw), -np.sin(expected_yaw), 0.0],
            [np.sin(expected_yaw), np.cos(expected_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        entity_to_base.entity_to_base_rotation_mat,
        expected_rotation,
    )


def test_parse_entity_to_base_uses_urdf_fixed_axis_rpy(
    tmp_path: Path,
) -> None:
    rpy = (0.2, 0.3, 0.4)
    yml_path = _write_curobo_assets(tmp_path, rpy=rpy)

    entity_to_base = _parse_entity_to_base_from_curobo_yml(str(yml_path))

    np.testing.assert_allclose(
        entity_to_base.entity_to_base_rotation_mat,
        _urdf_fixed_axis_rpy_matrix(*rpy),
    )


def test_parse_entity_to_base_rejects_frame_bias_mismatch(
    tmp_path: Path,
) -> None:
    yml_path = _write_curobo_assets(
        tmp_path,
        frame_bias=(-0.1, -0.2, -0.3),
    )

    with pytest.raises(ValueError, match="frame_bias"):
        _parse_entity_to_base_from_curobo_yml(str(yml_path))


def test_transform_entity_target_rejects_tiny_quaternion(
    tmp_path: Path,
) -> None:
    yml_path = _write_curobo_assets(tmp_path)
    entity_to_base = _parse_entity_to_base_from_curobo_yml(str(yml_path))

    with pytest.raises(ValueError, match="Quaternion"):
        entity_to_base.transform_entity_target_to_base(
            [0.0, 0.0, 0.0],
            [1e-12, 0.0, 0.0, 0.0],
        )


def test_prepare_runtime_skips_non_ee_action_even_when_patch_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_calls: list[bool] = []
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "_install_robotwin_curobo_base_transform_patch",
        lambda: install_calls.append(True),
    )
    cfg = SimpleNamespace(
        action_type="qpos",
        patch_curobo_base_transform=True,
    )

    prepare_robotwin_runtime_for_cfg(cfg)

    assert install_calls == []


def test_prepare_runtime_installs_patch_when_native_aloha_transform_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_robotwin_planner_module(
        monkeypatch,
        _FakeNativeAlohaCuroboPlanner,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "in_robotwin_workspace",
        nullcontext,
    )
    install_calls: list[bool] = []
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "_install_robotwin_curobo_base_transform_patch",
        lambda: install_calls.append(True),
    )

    prepare_robotwin_runtime_for_cfg(_native_aloha_cfg())

    assert install_calls == [True]


def test_prepare_runtime_installs_patch_for_legacy_aloha_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_robotwin_planner_module(
        monkeypatch,
        _FakeLegacyCuroboPlanner,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "in_robotwin_workspace",
        nullcontext,
    )
    install_calls: list[bool] = []
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "_install_robotwin_curobo_base_transform_patch",
        lambda: install_calls.append(True),
    )

    prepare_robotwin_runtime_for_cfg(_native_aloha_cfg())

    assert install_calls == [True]


def test_prepare_runtime_warns_and_falls_back_when_patch_install_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_messages: list[str] = []

    def _raise_unsupported_patch() -> None:
        raise RuntimeError("unsupported RoboTwin planner")

    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "in_robotwin_workspace",
        nullcontext,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.curobo_base_patch."
        "_install_robotwin_curobo_base_transform_patch",
        _raise_unsupported_patch,
    )
    monkeypatch.setattr(
        curobo_base_patch.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(message),
    )
    monkeypatch.setattr(
        curobo_base_patch,
        "_PATCH_INSTALLED_IN_PROCESS",
        False,
    )

    prepare_robotwin_runtime_for_cfg(_native_aloha_cfg())

    assert curobo_base_patch._PATCH_INSTALLED_IN_PROCESS is False
    assert any("Falling back" in message for message in warning_messages)


def test_setup_demo_guard_allows_worker_mode_after_patch_install_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_events: list[str] = []
    monkeypatch.setattr(
        curobo_base_patch,
        "_PATCH_INSTALLED_IN_PROCESS",
        False,
    )
    task = SimpleNamespace(
        setup_demo=lambda seed: setup_events.append(f"setup:{seed}"),
        robot=SimpleNamespace(communication_flag=True),
    )

    setup_robotwin_demo_with_runtime_guards(
        _native_aloha_cfg(),
        task,
        {"seed": 3},
    )

    assert setup_events == ["setup:3"]


def test_setup_demo_with_runtime_guard_rejects_worker_mode_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
):
    cleanup_events: list[str] = []
    monkeypatch.setattr(
        curobo_base_patch,
        "_PATCH_INSTALLED_IN_PROCESS",
        True,
    )

    class _FakeConn:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            cleanup_events.append(f"close:{self.name}")

    class _FakeProc:
        def __init__(self, name: str, *, alive: bool) -> None:
            self.name = name
            self.alive = alive

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            cleanup_events.append(f"terminate:{self.name}")
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            cleanup_events.append(f"join:{self.name}:{timeout}")

    task = SimpleNamespace(
        setup_demo=lambda seed: cleanup_events.append(f"setup:{seed}"),
        close_env=lambda clear_cache: cleanup_events.append(
            f"close_env:{clear_cache}"
        ),
        robot=SimpleNamespace(
            communication_flag=True,
            left_conn=_FakeConn("left"),
            right_conn=_FakeConn("right"),
            left_proc=_FakeProc("left", alive=True),
            right_proc=_FakeProc("right", alive=False),
        ),
    )

    with pytest.raises(
        RoboTwinCuroboPatchUnsupportedError,
        match="communication_flag",
    ):
        setup_robotwin_demo_with_runtime_guards(
            SimpleNamespace(
                action_type="ee",
                patch_curobo_base_transform=True,
            ),
            task,
            {"seed": 3},
        )

    assert cleanup_events == [
        "setup:3",
        "close_env:True",
        "close:left",
        "close:right",
        "terminate:left",
        "join:left:2.0",
        "join:right:2.0",
    ]


def test_install_patch_updates_new_planners_and_keeps_old_instance_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aloha_tmp_path = tmp_path / "aloha-agilex"
    aloha_tmp_path.mkdir()
    yml_path = _write_curobo_assets(
        aloha_tmp_path,
        xyz=(0.5, 0.0, 0.0),
        rpy=(0.0, 0.0, np.pi / 2),
        frame_bias=(-0.5, 0.0, 0.0),
    )

    class _FakeCuroboPlanner:
        _robo_orchard_original_frame_bias: list[float]

        def __init__(self, yml_path: str) -> None:
            self.yml_path = yml_path
            self.frame_bias = [-0.5, 0.0, 0.0]

        def _trans_from_world_to_base(
            self,
            base_pose: np.ndarray,
            target_pose: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            del base_pose
            return target_pose[:3].copy(), target_pose[3:].copy()

        def plan_path(self, target_pose: np.ndarray) -> np.ndarray:
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                target_pose,
            )
            target_pose_p = target_pose_p.copy()
            if "aloha-agilex" not in self.yml_path:
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else:
                target_pose_p += np.array([10.0, 0.0, 0.0])
            return np.concatenate([target_pose_p, target_pose_q])

        def plan_batch(
            self,
            target_poses: list[np.ndarray],
        ) -> list[np.ndarray]:
            return [
                self.plan_path(target_pose) for target_pose in target_poses
            ]

    envs_module = ModuleType("envs")
    robot_module = ModuleType("envs.robot")
    planner_module = ModuleType("envs.robot.planner")
    planner_module.__dict__["CuroboPlanner"] = _FakeCuroboPlanner
    robot_module.__dict__["planner"] = planner_module
    monkeypatch.setitem(sys.modules, "envs", envs_module)
    monkeypatch.setitem(sys.modules, "envs.robot", robot_module)
    monkeypatch.setitem(sys.modules, "envs.robot.planner", planner_module)

    old_planner = _FakeCuroboPlanner(str(yml_path))

    _install_robotwin_curobo_base_transform_patch()
    patched_init = _FakeCuroboPlanner.__init__
    patched_transform = _FakeCuroboPlanner._trans_from_world_to_base
    patched_plan_path = _FakeCuroboPlanner.plan_path
    patched_plan_batch = _FakeCuroboPlanner.plan_batch
    _install_robotwin_curobo_base_transform_patch()

    assert _FakeCuroboPlanner.__init__ is patched_init
    assert _FakeCuroboPlanner._trans_from_world_to_base is patched_transform
    assert _FakeCuroboPlanner.plan_path is patched_plan_path
    assert _FakeCuroboPlanner.plan_batch is patched_plan_batch

    target_pose = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    old_result = old_planner.plan_path(target_pose)
    np.testing.assert_allclose(old_result[:3], np.array([11.0, 0.0, 0.0]))

    new_planner = _FakeCuroboPlanner(str(yml_path))
    assert new_planner.frame_bias == [0.0, 0.0, 0.0]
    assert new_planner._robo_orchard_original_frame_bias == [
        -0.5,
        0.0,
        0.0,
    ]

    result = new_planner.plan_path(target_pose)
    assert new_planner.yml_path == str(yml_path)
    np.testing.assert_allclose(
        result[:3],
        np.array([0.0, -0.5, 0.0]),
        atol=1e-12,
    )
    _assert_quat_close(
        result[3:],
        np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]),
    )
    batch_result = new_planner.plan_batch([target_pose])
    np.testing.assert_allclose(batch_result[0], result)

    with pytest.raises(RuntimeError, match="fresh Python process"):
        prepare_robotwin_runtime_for_cfg(
            SimpleNamespace(
                action_type="ee",
                patch_curobo_base_transform=False,
            )
        )


def test_patched_planner_warns_and_uses_native_when_transform_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aloha_tmp_path = tmp_path / "aloha-agilex"
    aloha_tmp_path.mkdir()
    yml_path = _write_curobo_assets(
        aloha_tmp_path,
        xyz=(0.5, 0.0, 0.0),
        rpy=(0.0, 0.0, np.pi / 2),
        frame_bias=(-0.4, 0.0, 0.0),
    )
    warning_messages: list[str] = []

    class _FakeCuroboPlanner:
        def __init__(self, yml_path: str) -> None:
            self.yml_path = yml_path
            self.frame_bias = [-0.4, 0.0, 0.0]

        def _trans_from_world_to_base(
            self,
            base_pose: np.ndarray,
            target_pose: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            del base_pose
            return target_pose[:3].copy(), target_pose[3:].copy()

        def plan_path(self, target_pose: np.ndarray) -> np.ndarray:
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                target_pose,
            )
            target_pose_p = target_pose_p.copy()
            if "aloha-agilex" not in self.yml_path:
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else:
                target_pose_p += np.array([10.0, 0.0, 0.0])
            return np.concatenate([target_pose_p, target_pose_q])

        def plan_batch(
            self,
            target_poses: list[np.ndarray],
        ) -> list[np.ndarray]:
            return [
                self.plan_path(target_pose) for target_pose in target_poses
            ]

    _install_fake_robotwin_planner_module(monkeypatch, _FakeCuroboPlanner)
    monkeypatch.setattr(
        curobo_base_patch.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(message),
    )

    _install_robotwin_curobo_base_transform_patch()

    planner = _FakeCuroboPlanner(str(yml_path))
    target_pose = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = planner.plan_path(target_pose)

    assert planner.frame_bias == [-0.4, 0.0, 0.0]
    np.testing.assert_allclose(result[:3], np.array([11.0, 0.0, 0.0]))
    assert any("Falling back" in message for message in warning_messages)


def test_install_patch_handles_planner_without_native_aloha_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aloha_tmp_path = tmp_path / "aloha-agilex"
    aloha_tmp_path.mkdir()
    yml_path = _write_curobo_assets(
        aloha_tmp_path,
        xyz=(0.5, 0.0, 0.0),
        rpy=(0.0, 0.0, np.pi / 2),
        frame_bias=(-0.5, 0.0, 0.0),
    )

    class _FakeCuroboPlannerWithoutNativeAlohaBranch:
        def __init__(self, yml_path: str) -> None:
            self.yml_path = yml_path
            self.frame_bias = [-0.5, 0.0, 0.0]

        def _trans_from_world_to_base(
            self,
            base_pose: np.ndarray,
            target_pose: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            del base_pose
            return target_pose[:3].copy(), target_pose[3:].copy()

        def plan_path(self, target_pose: np.ndarray) -> np.ndarray:
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                target_pose,
            )
            target_pose_p = target_pose_p.copy()
            target_pose_p[0] += self.frame_bias[0]
            target_pose_p[1] += self.frame_bias[1]
            target_pose_p[2] += self.frame_bias[2]
            return np.concatenate([target_pose_p, target_pose_q])

        def plan_batch(
            self,
            target_poses: list[np.ndarray],
        ) -> list[np.ndarray]:
            return [
                self.plan_path(target_pose) for target_pose in target_poses
            ]

    _install_fake_robotwin_planner_module(
        monkeypatch,
        _FakeCuroboPlannerWithoutNativeAlohaBranch,
    )

    _install_robotwin_curobo_base_transform_patch()

    planner = _FakeCuroboPlannerWithoutNativeAlohaBranch(str(yml_path))
    target_pose = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = planner.plan_path(target_pose)

    assert planner.yml_path == str(yml_path)
    np.testing.assert_allclose(
        result[:3],
        np.array([0.0, -0.5, 0.0]),
        atol=1e-12,
    )
    _assert_quat_close(
        result[3:],
        np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]),
    )
