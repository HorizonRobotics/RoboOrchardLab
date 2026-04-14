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

from pathlib import Path

import pytest
import yaml

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnvCfg


@pytest.fixture()
def robotwin_task_config_assets(tmp_path: Path, monkeypatch):
    robotwin_root = tmp_path / "robotwin"
    task_config_dir = robotwin_root / "task_config"
    robot_dir = robotwin_root / "robots" / "single_arm"
    task_config_dir.mkdir(parents=True)
    robot_dir.mkdir(parents=True)

    task_config_path = task_config_dir / "task.yml"
    task_config_path.write_text(
        yaml.safe_dump(
            {
                "data_type": {
                    "rgb": False,
                    "depth": True,
                    "endpose": False,
                },
                "camera": {"head_camera_type": "default_head"},
                "embodiment": ["single_arm"],
            }
        ),
        encoding="utf-8",
    )

    (task_config_dir / "_embodiment_config.yml").write_text(
        yaml.safe_dump(
            {
                "single_arm": {
                    "file_path": str(robot_dir),
                }
            }
        ),
        encoding="utf-8",
    )
    (robot_dir / "config.yml").write_text("robot_name: single_arm\n")

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

    def test_get_task_config_rejects_split_arm_embodiment_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        robotwin_root = tmp_path / "robotwin"
        task_config_dir = robotwin_root / "task_config"
        left_robot_dir = robotwin_root / "robots" / "left_arm"
        right_robot_dir = robotwin_root / "robots" / "right_arm"
        task_config_dir.mkdir(parents=True)
        left_robot_dir.mkdir(parents=True)
        right_robot_dir.mkdir(parents=True)

        task_config_path = task_config_dir / "task.yml"
        task_config_path.write_text(
            yaml.safe_dump(
                {
                    "data_type": {
                        "rgb": False,
                        "depth": True,
                        "endpose": False,
                    },
                    "camera": {"head_camera_type": "default_head"},
                    "embodiment": ["left_arm", "right_arm", "0.3"],
                }
            ),
            encoding="utf-8",
        )

        (task_config_dir / "_embodiment_config.yml").write_text(
            yaml.safe_dump(
                {
                    "left_arm": {"file_path": str(left_robot_dir)},
                    "right_arm": {"file_path": str(right_robot_dir)},
                }
            ),
            encoding="utf-8",
        )
        (left_robot_dir / "config.yml").write_text(
            "robot_name: left_arm\n", encoding="utf-8"
        )
        (right_robot_dir / "config.yml").write_text(
            "robot_name: right_arm\n", encoding="utf-8"
        )
        (task_config_dir / "_camera_config.yml").write_text(
            yaml.safe_dump({"default_head": {"h": 480, "w": 640}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.config_robotwin_path",
            lambda: str(robotwin_root),
        )

        cfg = RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(task_config_path),
        )

        with pytest.raises(
            NotImplementedError,
            match="combined dual-arm robot layout",
        ):
            cfg.get_task_config()
