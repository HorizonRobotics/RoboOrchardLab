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


class _FakeLocalBenchmarkBackendCfg:
    instances = []

    def __init__(self):
        self.instances.append(self)


class _FakeRemoteBenchmarkBackendCfg:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)


class _FakeBenchmarkEvaluator:
    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluate_calls = []
        self.instances.append(self)

    def evaluate(self, policy_or_cfg, device=None):
        self.evaluate_calls.append(
            {
                "policy_or_cfg": policy_or_cfg,
                "device": device,
            }
        )
        return SimpleNamespace(
            metrics={
                "tasks": [{"task_name": "task_a", "success_rate": 1.0}],
                "average_success_rate": 1.0,
                "last_update": None,
            }
        )


class _FakeBenchmarkEvaluatorCfg:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def __call__(self):
        return _FakeBenchmarkEvaluator(self)


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

    def test_evaluate_tasks_locally_uses_local_benchmark_evaluator(self):
        module = _load_script_module()
        _FakeBenchmarkEvaluatorCfg.instances = []
        _FakeBenchmarkEvaluator.instances = []
        _FakeLocalBenchmarkBackendCfg.instances = []

        with (
            patch.object(
                module,
                "RoboTwinBenchmarkEvaluatorCfg",
                _FakeBenchmarkEvaluatorCfg,
            ),
            patch.object(
                module,
                "RoboTwinLocalBenchmarkBackendCfg",
                _FakeLocalBenchmarkBackendCfg,
            ),
            patch.object(
                module,
                "artifact_root_dir",
                return_value="/artifacts",
            ),
        ):
            metrics = module.evaluate_tasks_locally(
                policy_or_cfg="policy",
                task_names=["task_a", "task_b"],
                episode_num=2,
                device="cpu",
                config_type="demo_clean",
                seed=3,
                save_video=True,
            )

        self.assertEqual(metrics["average_success_rate"], 1.0)
        benchmark_cfg = _FakeBenchmarkEvaluatorCfg.instances[0]
        self.assertEqual(
            benchmark_cfg.kwargs["task_names"],
            ["task_a", "task_b"],
        )
        self.assertEqual(benchmark_cfg.kwargs["episode_num"], 2)
        self.assertEqual(benchmark_cfg.kwargs["config_type"], "demo_clean")
        self.assertEqual(benchmark_cfg.kwargs["start_seed"], 3)
        self.assertTrue(benchmark_cfg.kwargs["format_datatypes"])
        self.assertTrue(benchmark_cfg.kwargs["fail_fast"])
        self.assertEqual(
            benchmark_cfg.kwargs["artifact_root_dir"],
            "/artifacts",
        )
        self.assertIs(
            benchmark_cfg.kwargs["backend"],
            _FakeLocalBenchmarkBackendCfg.instances[0],
        )
        benchmark_evaluator = _FakeBenchmarkEvaluator.instances[0]
        self.assertIs(benchmark_evaluator.cfg, benchmark_cfg)
        self.assertEqual(
            benchmark_evaluator.evaluate_calls,
            [{"policy_or_cfg": "policy", "device": "cpu"}],
        )

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

    def test_run_ray_builds_policy_cfg_and_uses_remote_benchmark_backend(self):
        module = _load_script_module()

        fake_policy_cfg = object()
        fake_metrics = {
            "tasks": [{"task_name": "task_a", "success_rate": 1.0}],
            "average_success_rate": 1.0,
            "last_update": None,
        }
        _FakeBenchmarkEvaluatorCfg.instances = []
        _FakeBenchmarkEvaluator.instances = []
        _FakeRemoteBenchmarkBackendCfg.instances = []

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
                    "RoboTwinBenchmarkEvaluatorCfg",
                    _FakeBenchmarkEvaluatorCfg,
                ),
                patch.object(
                    module,
                    "RoboTwinRemoteBenchmarkBackendCfg",
                    _FakeRemoteBenchmarkBackendCfg,
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
            benchmark_cfg = _FakeBenchmarkEvaluatorCfg.instances[0]
            self.assertEqual(
                benchmark_cfg.kwargs["task_names"],
                ["task_a", "task_b"],
            )
            self.assertEqual(benchmark_cfg.kwargs["episode_num"], 2)
            self.assertEqual(benchmark_cfg.kwargs["config_type"], "demo_clean")
            self.assertEqual(benchmark_cfg.kwargs["start_seed"], 0)
            self.assertTrue(benchmark_cfg.kwargs["format_datatypes"])
            self.assertEqual(
                benchmark_cfg.kwargs["artifact_root_dir"],
                "eval_result",
            )
            self.assertIs(
                benchmark_cfg.kwargs["backend"],
                _FakeRemoteBenchmarkBackendCfg.instances[0],
            )
            remote_backend_cfg = _FakeRemoteBenchmarkBackendCfg.instances[0]
            self.assertEqual(
                remote_backend_cfg.kwargs["num_parallel_envs"],
                4,
            )
            self.assertEqual(
                remote_backend_cfg.kwargs["ray_init_config"],
                {"_temp_dir": "/tmp/ray-user"},
            )
            self.assertIn(
                "runtime_env",
                remote_backend_cfg.kwargs["remote_class_config"],
            )
            benchmark_evaluator = _FakeBenchmarkEvaluator.instances[0]
            self.assertIs(benchmark_evaluator.cfg, benchmark_cfg)
            self.assertEqual(
                benchmark_evaluator.evaluate_calls,
                [{"policy_or_cfg": fake_policy_cfg, "device": "cuda"}],
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text()), fake_metrics)

    def test_run_ray_defaults_device_to_cuda(self):
        module = _load_script_module()

        fake_policy_cfg = object()
        _FakeBenchmarkEvaluatorCfg.instances = []
        _FakeBenchmarkEvaluator.instances = []
        _FakeRemoteBenchmarkBackendCfg.instances = []

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
                    "RoboTwinBenchmarkEvaluatorCfg",
                    _FakeBenchmarkEvaluatorCfg,
                ),
                patch.object(
                    module,
                    "RoboTwinRemoteBenchmarkBackendCfg",
                    _FakeRemoteBenchmarkBackendCfg,
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

            benchmark_evaluator = _FakeBenchmarkEvaluator.instances[0]
            self.assertEqual(
                benchmark_evaluator.evaluate_calls,
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
