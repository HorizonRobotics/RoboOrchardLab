# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

import ast
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[4]


def test_robotwin_default_dataset2mcap_config_maps_expected_topics() -> None:
    from robo_orchard_lab.dataset.robotwin.to_mcap import (
        dataset_to_mcap_config,
        default_dataset_to_mcap_config,
    )

    config = default_dataset_to_mcap_config()
    entry_point_config = dataset_to_mcap_config(dataset=object())

    assert sorted(config) == [
        "front_camera",
        "front_camera_depth",
        "head_camera",
        "head_camera_depth",
        "joints",
        "left_camera",
        "left_camera_depth",
        "right_camera",
        "right_camera_depth",
    ]
    assert sorted(entry_point_config) == sorted(config)
    assert config["joints"].target_topic == (
        "/observation/robot_state/joints"
    )
    assert config["front_camera"].image_topic == (
        "/observation/cameras/front_camera/image"
    )
    assert config["front_camera"].calib_topic == (
        "/observation/cameras/front_camera/calib"
    )
    assert config["front_camera"].tf_topic == (
        "/observation/cameras/front_camera/tf"
    )
    assert config["front_camera_depth"].image_topic == (
        "/observation/cameras/front_camera/depth"
    )


def test_libero_dataset2mcap_config_entry_point_accepts_dataset() -> None:
    from robo_orchard_lab.dataset.libero.to_mcap import (
        dataset_to_mcap_config,
        default_dataset_to_mcap_config,
    )

    config = default_dataset_to_mcap_config()
    entry_point_config = dataset_to_mcap_config(dataset=object())

    assert sorted(entry_point_config) == sorted(config)
    assert entry_point_config["action_goal_eef"].target_topic == (
        "/action/goal_eef"
    )
    assert entry_point_config["tf_world"].target_topic == "/tf"


def test_setup_declares_dataset2mcap_preset_entry_points() -> None:
    setup_py = (_LAB_ROOT / "setup.py").read_text(encoding="utf-8")
    module = ast.parse(setup_py)
    entry_points: dict[str, list[str]] | None = None
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "entry_points":
                entry_points = ast.literal_eval(keyword.value)

    assert entry_points is not None
    assert entry_points["robo_orchard_datasets.dataset2mcap_presets"] == [
        "robotwin=robo_orchard_lab.dataset.robotwin.to_mcap:"
        "dataset_to_mcap_config",
        "libero=robo_orchard_lab.dataset.libero.to_mcap:"
        "dataset_to_mcap_config",
    ]
