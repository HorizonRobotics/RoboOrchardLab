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

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "projects"
    / "holobrain"
    / "scripts"
    / "robotwin_eval.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "holobrain_robotwin_eval_script", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSuccessRateMetric:
    def __init__(self):
        self.info = {}
        self.last_update_info = None

    def reset(self, **kwargs):
        self.info = {}
        self.last_update_info = None

    def update_episode(
        self,
        task_name: str,
        seed: int,
        success: bool,
        offset_seed: int | None = None,
    ):
        task_info = self.info.setdefault(
            task_name,
            {
                "task_name": task_name,
                "success_count": 0,
                "total_count": 0,
                "info_list": [],
            },
        )
        task_info["total_count"] += 1
        if success:
            task_info["success_count"] += 1
        info = {"seed": seed, "success": success}
        last_update_info = {
            "task_name": task_name,
            "seed": seed,
            "success": success,
        }
        if offset_seed is not None:
            info["offset_seed"] = offset_seed
            last_update_info["offset_seed"] = offset_seed
        task_info["info_list"].append(info)
        self.last_update_info = last_update_info

    def merge(self, metrics):
        for metric in metrics:
            for task_name, info in metric.info.items():
                current = self.info.setdefault(
                    task_name,
                    {
                        "task_name": task_name,
                        "success_count": 0,
                        "total_count": 0,
                        "info_list": [],
                    },
                )
                current["success_count"] += info["success_count"]
                current["total_count"] += info["total_count"]
                current["info_list"].extend(info["info_list"])
            self.last_update_info = metric.last_update_info

    def compute(self):
        tasks = []
        rates = []
        for info in self.info.values():
            success_rate = info["success_count"] / info["total_count"]
            tasks.append(
                {
                    "task_name": info["task_name"],
                    "success_count": info["success_count"],
                    "total_count": info["total_count"],
                    "success_rate": success_rate,
                }
            )
            rates.append(success_rate)
        average_success_rate = sum(rates) / len(rates) if rates else 0.0
        return {
            "tasks": tasks,
            "average_success_rate": average_success_rate,
            "last_update": self.last_update_info,
        }


class _FakeEvaluator:
    def __init__(self):
        self.metrics = None
        self.env_cfg = None
        self.setup_calls = []
        self.reset_calls = []
        self.evaluate_calls = []
        self.current_episode = None
        self.current_start_seed = None
        self.current_offset_seed = 0
        self.has_applied_initial_retry = False

    def setup(self, env_cfg, policy_or_cfg, metrics, device=None):
        self.setup_calls.append(
            {
                "env_cfg": env_cfg,
                "policy_or_cfg": policy_or_cfg,
                "device": device,
            }
        )
        self.env_cfg = env_cfg
        self.metrics = metrics

    def reset_env(self, **kwargs):
        self.reset_calls.append(kwargs)
        self.current_episode = kwargs
        assert isinstance(self.env_cfg, dict)
        if self.current_start_seed is None:
            self.current_start_seed = self.env_cfg["seed"]

        seed_request = kwargs.get("seed")
        if seed_request == "next":
            self.current_offset_seed += 1
        elif seed_request is not None:
            self.current_start_seed = seed_request
            self.current_offset_seed = 0

        if not self.has_applied_initial_retry:
            self.current_offset_seed += 2
            self.has_applied_initial_retry = True

        resolved_start_seed = (
            100000 * (1 + self.current_start_seed)
            if self.current_start_seed < 100000
            else self.current_start_seed
        )
        return None, {
            "seed": resolved_start_seed + self.current_offset_seed,
            "offset_seed": self.current_offset_seed,
        }

    def evaluate_episode(self, max_steps, env_reset_kwargs=None):
        self.evaluate_calls.append(
            {
                "max_steps": max_steps,
                "env_reset_kwargs": env_reset_kwargs,
            }
        )
        if env_reset_kwargs is not None:
            self.reset_env(**env_reset_kwargs)
        assert self.metrics is not None
        assert self.current_episode is not None
        resolved_start_seed = (
            100000 * (1 + self.current_start_seed)
            if self.current_start_seed < 100000
            else self.current_start_seed
        )
        seed = resolved_start_seed + self.current_offset_seed
        self.metrics.update_episode(
            task_name=self.current_episode["task_name"],
            seed=seed,
            success=True,
            offset_seed=self.current_offset_seed,
        )
        return {"last_update": self.metrics.last_update_info}

    def get_metrics(self):
        return self.metrics

    def reset_metrics(self):
        assert self.metrics is not None
        self.metrics.reset()


class _FakePolicyEvaluatorConfig:
    last_evaluator = None

    def __call__(self):
        evaluator = _FakeEvaluator()
        _FakePolicyEvaluatorConfig.last_evaluator = evaluator
        return evaluator


class _FakeRemoteEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = []

    def evaluate(self, policy_or_cfg, device=None):
        self.calls.append({"policy_or_cfg": policy_or_cfg, "device": device})
        return self.metrics


class _FakeRemoteEvaluatorCfg:
    last_kwargs = None
    evaluator = None

    def __init__(self, **kwargs):
        _FakeRemoteEvaluatorCfg.last_kwargs = kwargs

    def __call__(self):
        return _FakeRemoteEvaluatorCfg.evaluator


class RobotwinEvalScriptTest(unittest.TestCase):
    def test_parse_args_uses_pydantic_cli_style(self):
        module = _load_script_module()

        with patch.object(
            sys,
            "argv",
            [
                "robotwin_eval.py",
                "--model_dir",
                "/tmp/model",
                "--model_prefix",
                "checkpoint",
                "--task_names",
                '["task_a","task_b"]',
                "--save_video",
                "true",
                "--gpu_ids",
                "[0,2]",
                "--workers_per_gpu",
                "2",
            ],
        ):
            cfg = module.parse_args()

        self.assertEqual(cfg.model_dir, "/tmp/model")
        self.assertEqual(cfg.model_prefix, "checkpoint")
        self.assertEqual(cfg.task_names, ["task_a", "task_b"])
        self.assertTrue(cfg.save_video)
        self.assertEqual(cfg.gpu_ids, [0, 2])
        self.assertEqual(cfg.workers_per_gpu, 2)

    def test_evaluate_tasks_locally_aggregates_metrics(self):
        module = _load_script_module()

        with (
            patch.object(
                module,
                "SuccessRateMetric",
                _FakeSuccessRateMetric,
            ),
            patch.object(
                module,
                "PolicyEvaluatorConfig",
                _FakePolicyEvaluatorConfig,
            ),
            patch.object(
                module,
                "build_robotwin_env_cfg",
                side_effect=lambda task_name, config_type, seed: {
                    "task_name": task_name,
                    "config_type": config_type,
                    "seed": seed,
                },
            ) as mock_build_env,
        ):
            metrics = module.evaluate_tasks_locally(
                policy_or_cfg="policy",
                task_names=["task_a", "task_b"],
                episode_num=2,
                device="cpu",
                config_type="demo_clean",
                seed=3,
            )

        self.assertEqual(mock_build_env.call_count, 2)
        self.assertEqual(metrics["average_success_rate"], 1.0)
        self.assertEqual(len(metrics["tasks"]), 2)
        self.assertEqual(metrics["tasks"][0]["total_count"], 2)
        self.assertEqual(metrics["tasks"][1]["total_count"], 2)
        self.assertEqual(metrics["last_update"]["seed"], 400003)
        self.assertEqual(metrics["last_update"]["offset_seed"], 3)
        self.assertEqual(
            _FakePolicyEvaluatorConfig.last_evaluator.reset_calls,
            [
                {
                    "clear_cache": True,
                    "episode_id": 0,
                    "return_obs": True,
                    "seed": 3,
                    "task_name": "task_b",
                    "video_dir": None,
                },
                {
                    "clear_cache": True,
                    "episode_id": 1,
                    "return_obs": True,
                    "seed": "next",
                    "task_name": "task_b",
                    "video_dir": None,
                },
            ],
        )

    def test_evaluate_tasks_locally_passes_episode_id_when_enabled(self):
        module = _load_script_module()

        with (
            patch.object(
                module,
                "SuccessRateMetric",
                _FakeSuccessRateMetric,
            ),
            patch.object(
                module,
                "PolicyEvaluatorConfig",
                _FakePolicyEvaluatorConfig,
            ),
            patch.object(
                module,
                "build_robotwin_env_cfg",
                side_effect=lambda task_name, config_type, seed: {
                    "task_name": task_name,
                    "config_type": config_type,
                    "seed": seed,
                },
            ),
        ):
            module.evaluate_tasks_locally(
                policy_or_cfg="policy",
                task_names=["task_a"],
                episode_num=1,
                device="cpu",
                config_type="demo_clean",
                seed=0,
                save_video=True,
            )

        assert _FakePolicyEvaluatorConfig.last_evaluator.reset_calls == [
            {
                "clear_cache": True,
                "episode_id": 0,
                "return_obs": True,
                "seed": 0,
                "task_name": "task_a",
                "video_dir": os.path.join(
                    "eval_result",
                    "task_a",
                    "demo_clean",
                ),
            }
        ]

    def test_run_loads_pipeline_builds_policy_and_writes_metrics(self):
        module = _load_script_module()

        fake_pipeline = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
        )
        fake_policy = object()
        fake_metrics = {
            "tasks": [{"task_name": "task_a", "success_rate": 1.0}],
            "average_success_rate": 1.0,
            "last_update": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics" / "robotwin_eval.json"
            cfg = module.Config(
                model_dir="/tmp/model",
                inference_prefix="robotwin2_0",
                model_prefix="checkpoint",
                task_names=["task_a"],
                episode_num=1,
                device="cpu",
                output_path=str(output_path),
                save_video=True,
            )

            with (
                patch.object(
                    module.HoloBrainInferencePipeline,
                    "load_pipeline",
                    return_value=fake_pipeline,
                ) as mock_load_pipeline,
                patch.object(
                    module,
                    "HoloBrainRoboTwinPolicy",
                    return_value=fake_policy,
                ) as mock_policy_cls,
                patch.object(
                    module,
                    "evaluate_tasks_locally",
                    return_value=fake_metrics,
                ) as mock_evaluate,
            ):
                metrics = module.run(cfg)

            self.assertEqual(metrics, fake_metrics)
            mock_load_pipeline.assert_called_once_with(
                directory="/tmp/model",
                inference_prefix="robotwin2_0",
                device="cpu",
                load_weights=True,
                load_impl="native",
                model_prefix="checkpoint",
            )
            mock_policy_cls.assert_called_once()
            mock_evaluate.assert_called_once()
            self.assertTrue(mock_evaluate.call_args.kwargs["save_video"])
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text()), fake_metrics)

    def test_run_ray_builds_policy_cfg_and_uses_remote_evaluator(self):
        module = _load_script_module()

        fake_policy_cfg = object()
        fake_metrics = {
            "tasks": [{"task_name": "task_a", "success_rate": 1.0}],
            "average_success_rate": 1.0,
            "last_update": None,
        }
        _FakeRemoteEvaluatorCfg.evaluator = _FakeRemoteEvaluator(fake_metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics" / "robotwin_eval.json"
            cfg = module.Config(
                model_dir="/tmp/model",
                inference_prefix="robotwin2_0",
                model_prefix="checkpoint",
                task_names=["task_a", "task_b"],
                episode_num=2,
                device="cuda",
                output_path=str(output_path),
                mode="ray",
                save_video=True,
                gpu_ids=[0, 2],
                workers_per_gpu=2,
                ray_temp_dir="/tmp/ray-user",
            )

            with (
                patch.object(
                    module,
                    "HoloBrainRoboTwinPolicyCfg",
                    return_value=fake_policy_cfg,
                ) as mock_policy_cfg_cls,
                patch.object(
                    module,
                    "RoboTwinRemoteEvaluatorCfg",
                    _FakeRemoteEvaluatorCfg,
                ),
                patch.object(
                    module,
                    "RayRemoteClassConfig",
                    side_effect=lambda **kwargs: kwargs,
                ) as mock_remote_class_cfg,
                patch.object(
                    module,
                    "set_env",
                    side_effect=lambda **kwargs: nullcontext(),
                ) as mock_set_env,
                patch.object(
                    module,
                    "config_robotwin_path",
                    return_value="/tmp/robotwin",
                ),
            ):
                metrics = module.run(cfg)

            self.assertEqual(metrics, fake_metrics)
            mock_policy_cfg_cls.assert_called_once_with(
                model_dir="/tmp/model",
                inference_prefix="robotwin2_0",
                model_prefix="checkpoint",
                use_action_chunk_size=32,
            )
            mock_set_env.assert_called_once_with(CUDA_VISIBLE_DEVICES="0,2")
            mock_remote_class_cfg.assert_called_once()
            self.assertEqual(
                _FakeRemoteEvaluatorCfg.last_kwargs["num_parallel_envs"], 4
            )
            self.assertEqual(
                _FakeRemoteEvaluatorCfg.last_kwargs["ray_init_config"],
                {"_temp_dir": "/tmp/ray-user"},
            )
            self.assertEqual(
                _FakeRemoteEvaluatorCfg.last_kwargs["task_names"],
                ["task_a", "task_b"],
            )
            self.assertEqual(
                _FakeRemoteEvaluatorCfg.last_kwargs["artifact_root_dir"],
                "eval_result",
            )
            self.assertTrue(
                _FakeRemoteEvaluatorCfg.last_kwargs["format_datatypes"]
            )
            self.assertEqual(
                _FakeRemoteEvaluatorCfg.evaluator.calls,
                [{"policy_or_cfg": fake_policy_cfg, "device": "cuda"}],
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text()), fake_metrics)

    def test_run_ray_defaults_device_to_cuda(self):
        module = _load_script_module()

        fake_policy_cfg = object()
        fake_metrics = {
            "tasks": [{"task_name": "task_a", "success_rate": 1.0}],
            "average_success_rate": 1.0,
            "last_update": None,
        }
        _FakeRemoteEvaluatorCfg.evaluator = _FakeRemoteEvaluator(fake_metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics" / "robotwin_eval.json"
            cfg = module.Config(
                model_dir="/tmp/model",
                task_names=["task_a"],
                episode_num=1,
                output_path=str(output_path),
                mode="ray",
                gpu_ids=[0],
            )

            with (
                patch.object(
                    module,
                    "HoloBrainRoboTwinPolicyCfg",
                    return_value=fake_policy_cfg,
                ),
                patch.object(
                    module,
                    "RoboTwinRemoteEvaluatorCfg",
                    _FakeRemoteEvaluatorCfg,
                ),
                patch.object(
                    module,
                    "RayRemoteClassConfig",
                    side_effect=lambda **kwargs: kwargs,
                ),
                patch.object(
                    module,
                    "set_env",
                    side_effect=lambda **kwargs: nullcontext(),
                ),
                patch.object(
                    module,
                    "config_robotwin_path",
                    return_value="/tmp/robotwin",
                ),
            ):
                module.run(cfg)

            self.assertEqual(
                _FakeRemoteEvaluatorCfg.evaluator.calls,
                [{"policy_or_cfg": fake_policy_cfg, "device": "cuda"}],
            )

    def test_run_accepts_output_path_without_directory(self):
        module = _load_script_module()

        fake_pipeline = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
        )
        fake_metrics = {
            "tasks": [],
            "average_success_rate": 0.0,
            "last_update": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                cfg = module.Config(
                    model_dir="/tmp/model",
                    task_names=["task_a"],
                    episode_num=1,
                    device="cpu",
                    output_path="metrics.json",
                )

                with (
                    patch.object(
                        module.HoloBrainInferencePipeline,
                        "load_pipeline",
                        return_value=fake_pipeline,
                    ),
                    patch.object(
                        module,
                        "HoloBrainRoboTwinPolicy",
                        return_value=object(),
                    ),
                    patch.object(
                        module,
                        "evaluate_tasks_locally",
                        return_value=fake_metrics,
                    ),
                ):
                    metrics = module.run(cfg)
            finally:
                os.chdir(cwd)

            self.assertEqual(metrics, fake_metrics)
            self.assertTrue(Path(tmpdir, "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
