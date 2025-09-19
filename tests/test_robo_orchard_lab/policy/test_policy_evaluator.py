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
import os

import pytest
from robo_orchard_core.utils.ray import RayRemoteClassConfig
from ut_help import (
    DummyEnvConfig,
    DummyPolicyConfig,
    DummySuccessRateMetricConfig,
)

from robo_orchard_lab.metrics import MetricDictConfig
from robo_orchard_lab.policy import (
    PolicyEvaluatorConfig,
    PolicyEvaluatorRemoteConfig,
)


class TestPolicyEvaluator:
    @pytest.mark.parametrize("use_remote", [False, True])
    def test_evaluate_init(self, use_remote: bool) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        eval_cfg = PolicyEvaluatorConfig(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            metrics=metric_cfg,
        )
        remote_class_config = RayRemoteClassConfig(
            num_cpus=0.3,
            num_gpus=0,
            runtime_env={
                "env_vars": {"PYTHONPATH": current_dir},
            },
        )
        if use_remote:
            eval_cfg = eval_cfg.as_remote(
                remote_class_config=remote_class_config
            )
            assert isinstance(eval_cfg, PolicyEvaluatorRemoteConfig)
        evaluator = eval_cfg.__call__()
        # print(evaluator.compute_metrics())
        metric_res = evaluator.compute_metrics()
        assert metric_res == {"success_rate": 0.0}

    @pytest.mark.parametrize(
        "use_remote,rollout_steps",
        [
            (False, 1),
            (False, 3),
            (True, 1),
            (True, 3),
        ],
    )
    def test_evaluate_eval(self, use_remote: bool, rollout_steps: int) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        eval_cfg = PolicyEvaluatorConfig(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            metrics=metric_cfg,
        )
        remote_class_config = RayRemoteClassConfig(
            num_cpus=0.3,
            num_gpus=0,
            runtime_env={
                "env_vars": {"PYTHONPATH": current_dir},
            },
        )
        if use_remote:
            eval_cfg = eval_cfg.as_remote(
                remote_class_config=remote_class_config
            )
            assert isinstance(eval_cfg, PolicyEvaluatorRemoteConfig)

        evaluator = eval_cfg.__call__()
        total_step = 0
        for step in evaluator.make_episode_evaluation(
            max_steps=20,
            rollout_steps=rollout_steps,
        ):
            print(step)
            # assert step == 1
            total_step += step

        assert total_step == 5
        assert evaluator.compute_metrics() == {"success_rate": 1.0}

        evaluator.reset_metrics()
        assert evaluator.compute_metrics() == {"success_rate": 0.0}
