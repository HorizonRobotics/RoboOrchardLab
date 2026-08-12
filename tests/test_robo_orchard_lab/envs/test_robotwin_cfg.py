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

import copy
import os
from pathlib import Path

import cloudpickle
import pytest
import yaml

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnv, RoboTwinEnvCfg


@pytest.fixture()
def robotwin_task_config_assets(tmp_path: Path, monkeypatch):
    robotwin_root = tmp_path / "robotwin"
    task_config_dir = robotwin_root / "task_config"
    task_config_dir.mkdir(parents=True)

    def _write_robot_asset(
        name: str,
        *,
        dual_arm: bool,
        joint_count: int,
    ) -> Path:
        robot_dir = robotwin_root / "robots" / name
        robot_dir.mkdir(parents=True)
        arm_joints = [f"joint_{idx}" for idx in range(joint_count)]
        robot_config = {
            "urdf_path": "robot.urdf",
            "srdf_path": "robot.srdf",
            "move_group": ["left_eef", "right_eef"],
            "ee_joints": ["left_eef_joint", "right_eef_joint"],
            "arm_joints_name": [arm_joints, arm_joints],
            "gripper_name": [
                {"base": "gripper_joint", "mimic": []},
                {"base": "gripper_joint", "mimic": []},
            ],
            "gripper_bias": 0.1,
            "gripper_scale": [-0.01, 0.04],
            "dual_arm": dual_arm,
        }
        (robot_dir / "config.yml").write_text(
            yaml.safe_dump(robot_config),
            encoding="utf-8",
        )
        (robot_dir / "robot.urdf").write_text(
            "<robot name='fixture'/>", encoding="utf-8"
        )
        (robot_dir / "robot.srdf").write_text(
            "<robot name='fixture'/>", encoding="utf-8"
        )
        (robot_dir / "collision.yml").write_text("{}\n", encoding="utf-8")
        curobo_config = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": str(robot_dir / "robot.urdf"),
                    "collision_spheres": str(robot_dir / "collision.yml"),
                }
            }
        }
        curobo_names = (
            ("curobo_left.yml", "curobo_right.yml")
            if dual_arm
            else ("curobo.yml",)
        )
        for curobo_name in curobo_names:
            (robot_dir / curobo_name).write_text(
                yaml.safe_dump(curobo_config),
                encoding="utf-8",
            )
        return robot_dir

    _write_robot_asset("combined", dual_arm=True, joint_count=6)
    _write_robot_asset("left_arm", dual_arm=False, joint_count=6)
    _write_robot_asset("right_arm", dual_arm=False, joint_count=7)

    task_config = {
        "data_type": {
            "rgb": False,
            "depth": True,
            "endpose": False,
        },
        "camera": {"head_camera_type": "default_head"},
        "embodiment": ["combined"],
    }
    task_config_path = task_config_dir / "task.yml"
    task_config_path.write_text(
        yaml.safe_dump(task_config),
        encoding="utf-8",
    )
    for preset_name in ("demo_clean", "demo_randomized"):
        (task_config_dir / f"{preset_name}.yml").write_text(
            yaml.safe_dump(task_config),
            encoding="utf-8",
        )

    (task_config_dir / "_embodiment_config.yml").write_text(
        yaml.safe_dump(
            {
                "combined": {"file_path": "./robots/combined"},
                "left_arm": {"file_path": "./robots/left_arm"},
                "right_arm": {"file_path": "./robots/right_arm"},
            }
        ),
        encoding="utf-8",
    )

    (task_config_dir / "_camera_config.yml").write_text(
        yaml.safe_dump(
            {
                "default_head": {"h": 480, "w": 640},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.env.config_robotwin_path",
        lambda: str(robotwin_root),
    )
    return task_config_path


class TestRoboTwinEnvCfg:
    @pytest.mark.parametrize(
        ("task_config_path", "expected_name"),
        [
            (None, "demo_clean.yml"),
            ("demo_clean", "demo_clean.yml"),
            ("demo_randomized", "demo_randomized.yml"),
        ],
    )
    def test_resolves_task_config_presets(
        self,
        robotwin_task_config_assets: Path,
        task_config_path: str | None,
        expected_name: str,
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=task_config_path,
        )

        expected_path = robotwin_task_config_assets.parent / expected_name
        assert cfg.task_config_path == str(expected_path)

    def test_rejects_unknown_task_config_preset_name(
        self, robotwin_task_config_assets: Path
    ):
        with pytest.raises(FileNotFoundError, match="demo_randomize"):
            RoboTwinEnvCfg(
                task_name="place_object_basket",
                check_expert=False,
                check_task_init=False,
                task_config_path="demo_randomize",
            )

    def test_model_validate_resolves_task_config_preset(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg.model_validate(
            {
                "task_name": "place_object_basket",
                "check_expert": False,
                "check_task_init": False,
                "task_config_path": "demo_randomized",
            }
        )

        expected_path = (
            robotwin_task_config_assets.parent / "demo_randomized.yml"
        )
        assert cfg.task_config_path == str(expected_path)

    def test_json_round_trip_serializes_class_type(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        json_str = cfg.to_str(format="json")
        loaded_cfg = RoboTwinEnvCfg.from_str(json_str, format="json")

        assert loaded_cfg.class_type is RoboTwinEnv
        assert loaded_cfg == cfg
        assert loaded_cfg.patch_curobo_base_transform is False

    def test_patch_curobo_base_transform_defaults_to_disabled(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        assert cfg.patch_curobo_base_transform is False

    def test_eval_mode_keeps_cfg_seed_as_start_seed(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            eval_mode=True,
            seed=3,
        )

        task_config = cfg.get_task_config()

        assert cfg.seed == 3
        assert cfg.resolve_start_seed(cfg.seed) == 400000
        assert task_config["seed"] == 400000

    def test_get_task_config_applies_final_overrides(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            task_config_overrides=[
                ("data_type/rgb", True),
                ("head_camera_h", 720),
            ],
        )

        task_config = cfg.get_task_config()

        assert task_config["data_type"]["rgb"] is True
        assert task_config["data_type"]["depth"] is True
        assert task_config["head_camera_h"] == 720
        assert task_config["head_camera_w"] == 640
        assert task_config["task_name"] == "place_object_basket"

    def test_get_task_config_applies_endpose_override(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            task_config_overrides=[("data_type/endpose", False)],
        )

        task_config = cfg.get_task_config()

        assert task_config["data_type"]["endpose"] is False

    def test_get_task_config_rejects_reserved_or_missing_paths(
        self, robotwin_task_config_assets: Path
    ):
        reserved_cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            task_config_overrides=[("seed", 3)],
        )
        missing_cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            task_config_overrides=[("data_type/infrared", True)],
        )

        with pytest.raises(ValueError, match="seed"):
            reserved_cfg.get_task_config()
        with pytest.raises(KeyError, match="infrared"):
            missing_cfg.get_task_config()

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            ("left_robot_file", "/tmp/other"),
            ("right_robot_file", "/tmp/other"),
            ("left_embodiment_config/gripper_bias", 99.0),
            ("right_embodiment_config/gripper_scale", [0.0, 1.0]),
            ("dual_arm_embodied", True),
            ("embodiment_dis", -1.0),
            ("embodiment_name", "other"),
        ],
    )
    def test_get_task_config_rejects_embodiment_lowering_overrides(
        self,
        robotwin_task_config_assets: Path,
        path: str,
        value: object,
    ) -> None:
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = ["left_arm", "right_arm", 0.3]
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
            task_config_overrides=[(path, value)],
        )

        with pytest.raises(ValueError, match="derived"):
            cfg.get_task_config()

    def test_get_task_config_lowers_split_arm_embodiment_layout(
        self, robotwin_task_config_assets: Path
    ):
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = ["left_arm", "right_arm", 0.3]
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        resolved = cfg.get_task_config()

        robotwin_root = robotwin_task_config_assets.parents[1]
        assert resolved["left_robot_file"] == str(
            (robotwin_root / "robots" / "left_arm").resolve()
        )
        assert resolved["right_robot_file"] == str(
            (robotwin_root / "robots" / "right_arm").resolve()
        )
        assert resolved["embodiment_dis"] == 0.3
        assert resolved["dual_arm_embodied"] is False
        assert resolved["embodiment"] == ["left_arm", "right_arm", 0.3]

    @pytest.mark.parametrize("distance", [0, -0.1, float("inf"), "0.8"])
    def test_get_task_config_rejects_invalid_split_distance(
        self,
        robotwin_task_config_assets: Path,
        distance: object,
    ):
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = ["left_arm", "right_arm", distance]
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises(ValueError, match="distance"):
            cfg.get_task_config()

    @pytest.mark.parametrize(
        ("embodiment", "expected_error"),
        [
            (["left_arm"], "single-arm"),
            (["combined", "right_arm", 0.8], "combined"),
            (["missing"], "registered"),
        ],
    )
    def test_get_task_config_rejects_invalid_embodiment_topology(
        self,
        robotwin_task_config_assets: Path,
        embodiment: list[object],
        expected_error: str,
    ):
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = embodiment
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises((KeyError, ValueError), match=expected_error):
            cfg.get_task_config()

    def test_get_task_config_rejects_missing_upstream_gripper_field(
        self, robotwin_task_config_assets: Path
    ) -> None:
        combined_config_path = (
            robotwin_task_config_assets.parents[1]
            / "robots"
            / "combined"
            / "config.yml"
        )
        robot_config = yaml.safe_load(
            combined_config_path.read_text(encoding="utf-8")
        )
        robot_config.pop("gripper_bias")
        combined_config_path.write_text(
            yaml.safe_dump(robot_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises(ValueError, match="gripper_bias"):
            cfg.get_task_config()

    @pytest.mark.parametrize(
        ("embodiment", "robot_name", "planner_name"),
        [
            (["combined"], "combined", "curobo_left.yml"),
            (
                ["left_arm", "right_arm", 0.3],
                "left_arm",
                "curobo.yml",
            ),
        ],
    )
    def test_get_task_config_rejects_missing_topology_planner_config(
        self,
        robotwin_task_config_assets: Path,
        embodiment: list[object],
        robot_name: str,
        planner_name: str,
    ) -> None:
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = embodiment
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        planner_path = (
            robotwin_task_config_assets.parents[1]
            / "robots"
            / robot_name
            / planner_name
        )
        planner_path.unlink()
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises(FileNotFoundError, match="planner config"):
            cfg.get_task_config()

    def test_get_task_config_rejects_missing_curobo_reference(
        self, robotwin_task_config_assets: Path
    ) -> None:
        collision_path = (
            robotwin_task_config_assets.parents[1]
            / "robots"
            / "combined"
            / "collision.yml"
        )
        collision_path.unlink()
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises(FileNotFoundError, match="collision_spheres"):
            cfg.get_task_config()

    def test_task_config_snapshot_pins_bytes_after_source_changes(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )
        original_digest = cfg._task_config_content_sha256
        original_config = cfg.get_task_config_for_seed(7)

        robotwin_task_config_assets.write_text(
            "embodiment: [left_arm, right_arm, 0.8]\n",
            encoding="utf-8",
        )
        assert cfg.get_task_config_for_seed(8)["embodiment"] == ["combined"]
        robotwin_task_config_assets.unlink()

        assert cfg.get_task_config_for_seed(9)["embodiment"] == ["combined"]
        assert cfg._task_config_content_sha256 == original_digest
        assert original_config["seed"] == 7

    def test_task_config_path_replace_refreshes_pinned_snapshot(
        self, robotwin_task_config_assets: Path
    ) -> None:
        replacement_path = (
            robotwin_task_config_assets.parent / "demo_clean.yml"
        )
        replacement_config = yaml.safe_load(
            replacement_path.read_text(encoding="utf-8")
        )
        replacement_config["data_type"]["rgb"] = True
        replacement_path.write_text(
            yaml.safe_dump(replacement_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        replaced = cfg.replace(task_config_path="demo_clean")

        assert replaced.task_config_path == str(replacement_path.resolve())
        assert replaced._task_config_content_sha256 != (
            cfg._task_config_content_sha256
        )
        assert replaced.get_task_config_for_seed(7)["data_type"]["rgb"]
        assert not cfg.get_task_config_for_seed(7)["data_type"]["rgb"]
        restored = RoboTwinEnvCfg.from_str(
            replaced.to_str(format="json"), format="json"
        )
        assert restored._task_config_content_sha256 == (
            replaced._task_config_content_sha256
        )

    def test_task_config_path_model_copy_refreshes_pinned_snapshot(
        self, robotwin_task_config_assets: Path
    ) -> None:
        replacement_path = (
            robotwin_task_config_assets.parent / "demo_clean.yml"
        )
        replacement_config = yaml.safe_load(
            replacement_path.read_text(encoding="utf-8")
        )
        replacement_config["data_type"]["rgb"] = True
        replacement_path.write_text(
            yaml.safe_dump(replacement_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        copied = cfg.model_copy(update={"task_config_path": "demo_clean"})

        assert copied.task_config_path == str(replacement_path.resolve())
        assert copied._task_config_content_sha256 != (
            cfg._task_config_content_sha256
        )
        assert copied.get_task_config_for_seed(7)["data_type"]["rgb"]
        assert not cfg.get_task_config_for_seed(7)["data_type"]["rgb"]

    def test_class_config_path_override_refreshes_pinned_snapshot(
        self, robotwin_task_config_assets: Path
    ) -> None:
        replacement_path = (
            robotwin_task_config_assets.parent / "demo_clean.yml"
        )
        replacement_config = yaml.safe_load(
            replacement_path.read_text(encoding="utf-8")
        )
        replacement_config["data_type"]["rgb"] = True
        replacement_path.write_text(
            yaml.safe_dump(replacement_config), encoding="utf-8"
        )
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        class CaptureEnv:
            InitFromConfig = True

            def __init__(self, env_cfg: RoboTwinEnvCfg) -> None:
                self.env_cfg = env_cfg

        object.__setattr__(cfg, "class_type", CaptureEnv)
        captured = cfg.create_instance_by_cfg(task_config_path="demo_clean")

        assert isinstance(captured, CaptureEnv)
        assert captured.env_cfg.task_config_path == str(
            replacement_path.resolve()
        )
        assert captured.env_cfg._task_config_content_sha256 != (
            cfg._task_config_content_sha256
        )

    def test_task_config_path_rejects_direct_assignment(
        self, robotwin_task_config_assets: Path
    ) -> None:
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        with pytest.raises(AttributeError, match="is pinned"):
            cfg.task_config_path = "demo_clean"

    def test_task_config_snapshot_survives_copy_and_cloudpickle(
        self, robotwin_task_config_assets: Path
    ):
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        copied = copy.deepcopy(cfg)
        unpickled = cloudpickle.loads(cloudpickle.dumps(cfg))
        robotwin_task_config_assets.unlink()

        for restored in (copied, unpickled):
            assert restored._task_config_content_sha256 == (
                cfg._task_config_content_sha256
            )
            assert restored.get_task_config_for_seed(11)["seed"] == 11

    def test_task_config_snapshot_survives_json_after_source_deleted(
        self, robotwin_task_config_assets: Path
    ) -> None:
        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )
        serialized = cfg.to_str(format="json")
        original_digest = cfg._task_config_content_sha256
        robotwin_task_config_assets.unlink()

        restored = RoboTwinEnvCfg.from_str(serialized, format="json")

        assert restored._task_config_content_sha256 == original_digest
        assert restored.get_task_config_for_seed(12)["embodiment"] == [
            "combined"
        ]

    def test_same_path_new_cfg_gets_new_snapshot_digest(
        self, robotwin_task_config_assets: Path
    ):
        first = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )
        task_config = yaml.safe_load(
            robotwin_task_config_assets.read_text(encoding="utf-8")
        )
        task_config["data_type"]["rgb"] = True
        robotwin_task_config_assets.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        second = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(robotwin_task_config_assets),
        )

        assert first.task_config_path == second.task_config_path
        assert first._task_config_content_sha256 != (
            second._task_config_content_sha256
        )
        assert first != second

    @pytest.mark.parametrize(
        "embodiment",
        [
            ["aloha-agilex"],
            ["ur5-wsg", "ur5-wsg", 0.8],
            ["ARX-X5", "ARX-X5", 0.8],
            ["franka-panda", "franka-panda", 0.8],
            ["piper", "piper", 0.8],
            ["piper", "franka-panda", 0.8],
        ],
    )
    def test_official_embodiment_assets_pass_config_preflight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        embodiment: list[object],
    ) -> None:
        robotwin_path = os.environ.get("RoboTwin_PATH")
        if not robotwin_path:
            pytest.skip("RoboTwin_PATH is required for official asset checks.")
        robotwin_root = Path(robotwin_path)
        demo_clean_path = robotwin_root / "task_config" / "demo_clean.yml"
        if not demo_clean_path.is_file():
            pytest.skip("RoboTwin demo_clean.yml is unavailable.")

        task_config = yaml.safe_load(
            demo_clean_path.read_text(encoding="utf-8")
        )
        task_config["embodiment"] = embodiment
        task_config_path = tmp_path / "task.yml"
        task_config_path.write_text(
            yaml.safe_dump(task_config), encoding="utf-8"
        )
        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.config_robotwin_path",
            lambda: str(robotwin_root),
        )

        cfg = RoboTwinEnvCfg(
            task_name="place_empty_cup",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(task_config_path),
        )
        resolved = cfg.get_task_config_for_seed(0)

        assert Path(resolved["left_robot_file"]).is_absolute()
        assert Path(resolved["right_robot_file"]).is_absolute()
        assert resolved["dual_arm_embodied"] is (len(embodiment) == 1)
        if len(embodiment) == 3:
            assert resolved["embodiment_dis"] == 0.8
