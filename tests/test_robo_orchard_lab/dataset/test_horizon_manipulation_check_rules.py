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
import importlib.util
import sys
from pathlib import Path

import numpy as np

TOOLS_MODULE = "robo_orchard_lab.dataset.horizon_manipulation.tools"
TOOLS_PATH = (
    Path(__file__).resolve().parents[3]
    / "robo_orchard_lab"
    / "dataset"
    / "horizon_manipulation"
    / "tools"
)


def _load_tools_module(module_name: str):
    name = f"{TOOLS_MODULE}.{module_name}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, TOOLS_PATH / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_episode_rules_handles_empty_topic_frequency():
    check_config = _load_tools_module("check_config")
    check_models = _load_tools_module("check_models")
    check_rules = _load_tools_module("check_rules")

    topic = "/observation/cameras/middle/depth_image/image_raw"
    episode_data = check_models.EpisodeData(
        uuid="task/user/episode",
        meta={},
        topic_counts={topic: 1},
        topic_frequencies={topic: np.array([], dtype=np.float64)},
        base_time=np.array([0.0, 0.1], dtype=np.float64),
        joint_positions=np.zeros((0, 14), dtype=np.float64),
        master_joint_positions=np.zeros((0, 14), dtype=np.float64),
        fk_ee_poses=np.zeros((0, 2, 7), dtype=np.float64),
        recorded_ee_poses=None,
        images={},
        depths={},
        required_topics=[topic],
        observed_topics={topic},
    )

    inspection = check_rules.evaluate_episode_rules(
        episode_data, check_config.build_default_inspect_config()
    )

    assert inspection.episode_status == "fail"
    fps_result = next(
        item
        for item in inspection.rule_results
        if item.rule_id == "fps_out_of_range"
    )
    assert fps_result.metrics[topic]["mean_fps"] == 0.0
    assert fps_result.metrics[topic]["min_fps"] == 0.0
    assert fps_result.metrics[topic]["max_interval_limit"] == 0.2
