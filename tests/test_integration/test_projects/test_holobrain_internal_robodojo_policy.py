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

import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


@pytest.fixture
def robodojo_policy_modules():
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    paths = (repo_root, common_dir)
    for path in paths:
        sys.path.insert(0, str(path))
    module_names = (
        "holobrain_robodojo_policy.model",
        "holobrain_robodojo_policy.deploy_policy",
        "holobrain_robodojo_policy",
    )
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    try:
        deploy_policy = importlib.import_module(
            "holobrain_robodojo_policy.deploy_policy"
        )
        model = importlib.import_module("holobrain_robodojo_policy.model")
        yield deploy_policy, model
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for path in paths:
            sys.path.remove(str(path))


def _make_observation(*, plural_extrinsics=False):
    extrinsic_key = (
        "extrinsics_matrix" if plural_extrinsics else "extrinsic_matrix"
    )
    vision = {}
    camera_colors = {
        "cam_left_wrist": [10, 20, 30],
        "cam_right_wrist": [40, 50, 60],
        "cam_head": [70, 80, 90],
    }
    for camera_name, color in camera_colors.items():
        vision[camera_name] = {
            "color": np.tile(
                np.asarray(color, dtype=np.uint8),
                (2, 3, 1),
            ),
            "intrinsic_matrix": np.diag([100.0, 101.0, 1.0]),
            extrinsic_key: np.eye(4),
        }
    return {
        "instruction": "stack the bowls",
        "vision": vision,
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.asarray([0.25], dtype=np.float32),
            "right_arm_joint_state": np.arange(
                10,
                16,
                dtype=np.float32,
            ),
            "right_ee_joint_state": np.asarray([0.75], dtype=np.float32),
        },
    }


@pytest.mark.parametrize("plural_extrinsics", [False, True])
def test_robodojo_policy_builds_training_compatible_input(
    robodojo_policy_modules,
    plural_extrinsics,
):
    deploy_policy, _ = robodojo_policy_modules
    policy = deploy_policy.HoloBrainRoboDojoPolicy(
        deploy_policy.HoloBrainRoboDojoPolicyCfg(valid_action_step=1),
        pipeline=lambda data: data,
    )

    data = policy.data_preprocess(
        _make_observation(plural_extrinsics=plural_extrinsics)
    )

    assert tuple(data.image) == deploy_policy.ROBODOJO_CAMERAS
    np.testing.assert_array_equal(
        data.image["cam_left_wrist"][0][0, 0],
        [30, 20, 10],
    )
    assert data.depth["cam_head"][0].shape == (2, 3)
    assert data.depth["cam_head"][0].dtype == np.float32
    assert not data.depth["cam_head"][0].any()
    np.testing.assert_array_equal(
        data.history_joint_state[0],
        [0, 1, 2, 3, 4, 5, 0.25, 10, 11, 12, 13, 14, 15, 0.75],
    )
    np.testing.assert_array_equal(
        data.t_world2cam["cam_head"],
        np.diag([1.0, -1.0, -1.0, 1.0]),
    )
    assert data.intrinsic["cam_head"].shape == (4, 4)
    assert data.instruction == "stack the bowls"


def test_robodojo_policy_requires_camera_calibration(
    robodojo_policy_modules,
):
    deploy_policy, _ = robodojo_policy_modules
    policy = deploy_policy.HoloBrainRoboDojoPolicy(
        deploy_policy.HoloBrainRoboDojoPolicyCfg(valid_action_step=1),
        pipeline=lambda data: data,
    )
    observation = _make_observation()
    del observation["vision"]["cam_head"]["intrinsic_matrix"]

    with pytest.raises(ValueError, match="intrinsic_matrix"):
        policy.data_preprocess(observation)


def test_robodojo_policy_clips_chunk_grippers_only(
    robodojo_policy_modules,
):
    deploy_policy, _ = robodojo_policy_modules
    actions = np.arange(42, dtype=np.float32).reshape(3, 14)
    actions[:, 6] = [-1.0, 0.5, 2.0]
    actions[:, 13] = [2.0, 0.25, -1.0]

    class FakePipeline:
        def __call__(self, data):
            return SimpleNamespace(action=actions)

    policy = deploy_policy.HoloBrainRoboDojoPolicy(
        deploy_policy.HoloBrainRoboDojoPolicyCfg(valid_action_step=2),
        pipeline=FakePipeline(),
    )

    result = policy.predict_actions(_make_observation())

    assert result.shape == (2, 14)
    np.testing.assert_array_equal(result[:, 6], [0.0, 0.5])
    np.testing.assert_array_equal(result[:, 13], [1.0, 0.25])
    assert result[1, 0] == actions[1, 0]


def test_remote_model_cache_is_isolated_by_url(
    robodojo_policy_modules,
    tmp_path,
    monkeypatch,
):
    deploy_policy, _ = robodojo_policy_modules
    downloaded_paths = []

    def fake_download(url, output_path):
        downloaded_paths.append(output_path)

    monkeypatch.setattr(deploy_policy, "_download_file", fake_download)

    first_dir = deploy_policy.prepare_model_dir(
        "https://models.example/checkpoints/run-a/model",
        "robodojo_processor",
        "model",
        str(tmp_path),
    )
    second_dir = deploy_policy.prepare_model_dir(
        "https://models.example/checkpoints/run-b/model",
        "robodojo_processor",
        "model",
        str(tmp_path),
    )

    assert first_dir != second_dir
    assert {path.parent for path in downloaded_paths} == {
        Path(first_dir),
        Path(second_dir),
    }


class FakePolicy:
    def __init__(self):
        self.reset_count = 0

    def predict_actions(self, obs):
        offset = float(obs.get("offset", 0.0))
        return np.arange(14, dtype=np.float32)[None] + offset

    def reset(self):
        self.reset_count += 1


def test_xpolicylab_model_module_reexports_deploy_model(
    robodojo_policy_modules,
):
    deploy_policy, model_module = robodojo_policy_modules

    assert model_module.Model is deploy_policy.HoloBrainRoboDojoPolicy
    assert model_module.__all__ == ["Model"]


def test_xpolicylab_model_formats_single_and_batch_actions(
    robodojo_policy_modules,
):
    _, model_module = robodojo_policy_modules
    fake_policy = FakePolicy()
    model = model_module.Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "action_dim": 14,
            "model_dir": "/unused",
        },
        pipeline=fake_policy,
    )
    model.predict_actions = fake_policy.predict_actions

    with pytest.raises(RuntimeError, match="update_obs"):
        model.get_action()

    model.update_obs({"offset": 1})
    action = model.get_action()[0]
    np.testing.assert_array_equal(
        action["left_arm_joint_state"],
        [1, 2, 3, 4, 5, 6],
    )
    np.testing.assert_array_equal(action["left_ee_joint_state"], [7])
    np.testing.assert_array_equal(
        action["right_arm_joint_state"],
        [8, 9, 10, 11, 12, 13],
    )
    np.testing.assert_array_equal(action["right_ee_joint_state"], [14])

    model.update_obs_batch(
        [
            {"env_idx": 3, "offset": 3},
            {"env_idx": 1, "offset": 1},
        ]
    )
    batch = model.get_action_batch([1, 3])
    assert batch[0][0]["left_arm_joint_state"][0] == 1
    assert batch[1][0]["left_arm_joint_state"][0] == 3

    model.reset()
    assert fake_policy.reset_count == 1
    with pytest.raises(RuntimeError, match="update_obs"):
        model.get_action()


def test_xpolicylab_model_accepts_runtime_env_config(
    robodojo_policy_modules,
):
    deploy_policy, model_module = robodojo_policy_modules

    model = model_module.Model(
        {
            "action_type": "joint",
            "env_cfg_type": "custom_x5",
            "model_dir": "/unused",
        },
        pipeline=FakePolicy(),
    )

    assert isinstance(model, deploy_policy.HoloBrainRoboDojoPolicy)


def test_robodojo_deploy_config_and_adapter_files_are_complete():
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    policy_dir = common_dir / "holobrain_robodojo_policy"

    required_files = {
        "deploy.py",
        "deploy.yml",
        "eval.sh",
        "model.py",
        "setup_eval_env_client.sh",
        "setup_eval_policy_server.sh",
    }
    assert required_files <= {path.name for path in policy_dir.iterdir()}

    deploy_config = yaml.safe_load((policy_dir / "deploy.yml").read_text())
    assert deploy_config["policy_name"] == "holobrain_robodojo_policy"
    assert deploy_config["protocol"] == "ws"
    assert deploy_config["action_type"] == "joint"
    assert deploy_config["eval_batch"] is False
    assert deploy_config["env_cfg_type"] == "arx_x5"
    assert not (policy_dir / "env_cfg/arx_x5.yml").exists()


def test_robodojo_eval_distributes_tasks_across_gpus(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    spec = importlib.util.spec_from_file_location(
        "robodojo_eval",
        common_dir / "robodojo_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runtime_root = tmp_path / "working"
    policy_source = runtime_root / "holobrain_robodojo_policy"
    policy_source.mkdir(parents=True)

    robodojo_root = tmp_path / "robodojo"
    (robodojo_root / "XPolicyLab/policy").mkdir(parents=True)
    (robodojo_root / "third_party/IsaacLab/source/isaaclab").mkdir(
        parents=True
    )
    source_env_config = robodojo_root / "env_cfg/custom_x5.yml"
    source_env_config.parent.mkdir(parents=True)
    source_env_config.write_text(
        yaml.safe_dump(
            {
                "config_name": "custom_x5",
                "observation": {
                    "vision": {
                        "intrinsic_matrix": False,
                        "extrinsic_matrix": False,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for name in ("camera", "robot", "scene", "sim"):
        (robodojo_root / "env_cfg" / name).mkdir(parents=True)

    conda_root = tmp_path / "conda"
    policy_env = tmp_path / "policy-env"
    env_config_dir = tmp_path / "env-config"
    result_dir = tmp_path / "results"
    kit_root = tmp_path / "kit"
    monkeypatch.setenv("PYTHONPATH", "/caller/pythonpath")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    task_names = [
        "align_blocks",
        "build_tower",
        "push_T",
        "stack_bowls",
        "swap_T",
    ]
    monkeypatch.setattr(module, "_resolve_task_names", lambda args: task_names)
    calls = []
    logged_messages = []
    barrier = threading.Barrier(4)
    first_call_workers = set()
    call_lock = threading.Lock()

    def fake_run(command, *, cwd, env, check, stdout, stderr):
        run_id = env["ROBODOJO_RUN_ID"]
        worker_id = int(run_id.split("_worker_", 1)[1].split("_", 1)[0])
        with call_lock:
            is_first_call = worker_id not in first_call_workers
            first_call_workers.add(worker_id)
        if is_first_call:
            barrier.wait(timeout=5)

        task = command[8]
        result_path = module._task_result_path(
            result_dir,
            task,
            env_config="custom_x5",
            seed=0,
            run_id=run_id,
        )
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "success_rate": (task_names.index(task) + 1) / 10,
                    "eval_time": 10,
                }
            )
        )
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "check": check,
                "stderr": stderr,
                "log_path": Path(stdout.name),
                "task": task,
                "worker_id": worker_id,
            }
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda message, *args: logged_messages.append(message % args),
    )

    argv = [
        "--policy_source",
        str(policy_source),
        "--model_dir",
        "https://model.example/ckpt",
        "--model_processor",
        "robodojo_arx_x5a_processor",
        "--robodojo_root",
        str(robodojo_root),
        "--conda_root",
        str(conda_root),
        "--policy_env",
        str(policy_env),
        "--env_config_dir",
        str(env_config_dir),
        "--env_config",
        "custom_x5",
        "--eval_result_dir",
        str(result_dir),
        "--kit_root",
        str(kit_root),
        "--processes_per_gpu",
        "2",
    ]
    module.main(argv)

    policy_dir = robodojo_root / "XPolicyLab/policy/holobrain_robodojo_policy"
    assert policy_dir.resolve() == policy_source.resolve()
    runtime_env_config = env_config_dir / "custom_x5.yml"
    assert runtime_env_config.is_file()
    assert not runtime_env_config.is_symlink()
    source_config = yaml.safe_load(source_env_config.read_text())
    runtime_config = yaml.safe_load(runtime_env_config.read_text())
    assert source_config["observation"]["vision"] == {
        "intrinsic_matrix": False,
        "extrinsic_matrix": False,
    }
    assert runtime_config["observation"]["vision"] == {
        "intrinsic_matrix": True,
        "extrinsic_matrix": True,
    }
    for name in ("camera", "robot", "scene", "sim"):
        assert (env_config_dir / name).resolve() == (
            robodojo_root / "env_cfg" / name
        ).resolve()

    assert len(calls) == len(task_names)
    calls_by_worker = {
        worker_id: [call for call in calls if call["worker_id"] == worker_id]
        for worker_id in range(4)
    }
    expected_workers = {
        0: ("2", ["align_blocks", "swap_T"]),
        1: ("5", ["build_tower"]),
        2: ("2", ["push_T"]),
        3: ("5", ["stack_bowls"]),
    }
    assert set(calls_by_worker) == set(expected_workers)
    for worker_id, (gpu_id, expected_tasks) in expected_workers.items():
        worker_calls = calls_by_worker[worker_id]
        assert [call["task"] for call in worker_calls] == expected_tasks
        for call in worker_calls:
            task = call["task"]
            command = call["command"]
            assert call["cwd"] == robodojo_root
            assert call["check"] is False
            assert call["stderr"] is subprocess.STDOUT
            assert command == [
                str(conda_root / "bin/conda"),
                "run",
                "--no-capture-output",
                "-n",
                "RoboDojo",
                "bash",
                str(policy_dir / "eval.sh"),
                "RoboDojo",
                task,
                "holobrain",
                "custom_x5",
                "joint",
                "0",
                gpu_id,
                gpu_id,
                str(policy_env),
                "RoboDojo",
            ]
            assert call["env"]["ROBODOJO_RUN_ID"] == (
                f"aidi_seed_0_worker_{worker_id}_{task}"
            )
            assert call["env"]["EVAL_NUM"] == "10"
            assert call["env"]["ROBODOJO_FATAL_RESTART_COUNT"] == "0"
            assert call["log_path"] == (
                result_dir
                / f"aidi_seed_0_worker_{worker_id}"
                / "logs"
                / f"{task}.log"
            )
            assert (
                f"--portable-root {kit_root / f'worker_{worker_id}'}"
                in (call["env"]["ROBODOJO_KIT_ARGS"])
            )
            python_paths = call["env"]["PYTHONPATH"].split(os.pathsep)
            assert str(runtime_root) not in python_paths
            assert "/caller/pythonpath" in python_paths
            assert call["env"]["HOLOBRAIN_MODEL_DIR"] == (
                "https://model.example/ckpt"
            )
            assert call["env"]["HOLOBRAIN_MODEL_PROCESSOR"] == (
                "robodojo_arx_x5a_processor"
            )

    summary = json.loads((result_dir / "summary_seed_0.json").read_text())
    assert summary["average_success_rate"] == pytest.approx(0.3)
    assert summary["num_tasks"] == len(task_names)
    assert summary["eval_num_per_task"] == 10
    assert summary["task_success_rates"] == {
        task: (index + 1) / 10 for index, task in enumerate(task_names)
    }
    assert "missing_tasks" not in summary
    assert not (result_dir / "summary_seed_0.md").exists()
    benchmark_summary = json.loads(
        (result_dir / "benchmark_summary_seed_0.json").read_text()
    )
    assert benchmark_summary["complete"] is False
    assert benchmark_summary["completed_tasks"] == 0
    assert benchmark_summary["num_tasks"] == 42
    task_result_messages = {
        message
        for message in logged_messages
        if message.startswith("Task ") and " success_rate=" in message
    }
    assert task_result_messages == {
        (
            f"Task {task} finished: success_rate={(index + 1) * 10:.2f}% "
            "eval_time=10 exit_code=0"
        )
        for index, task in enumerate(task_names)
    }


def test_robodojo_eval_resolves_default_and_selected_tasks(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    spec = importlib.util.spec_from_file_location(
        "robodojo_eval_tasks",
        common_dir / "robodojo_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parsed_args = module._parse_args(
        [
            "--policy_source",
            str(tmp_path / "policy"),
            "--model_dir",
            "/model",
            "--model_processor",
            "processor",
        ]
    )
    assert parsed_args.env_config == "arx_x5"

    task_dir = tmp_path / "task/RoboDojo/tasks"
    config_dir = tmp_path / "task/RoboDojo/config"
    task_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    for task_name in ("align_blocks", "build_tower", "stack_bowls"):
        (task_dir / f"{task_name}.py").write_text(
            f"class {task_name}:\n    pass\n"
        )
        (config_dir / f"{task_name}.yml").touch()
    (task_dir / "missing_config.py").write_text(
        "class missing_config:\n    pass\n"
    )
    (task_dir / "missing_class.py").write_text("class Other:\n    pass\n")
    (config_dir / "missing_class.yml").touch()

    args = SimpleNamespace(robodojo_root=tmp_path, tasks=None)
    assert module._resolve_task_names(args) == [
        "align_blocks",
        "build_tower",
        "stack_bowls",
    ]

    args.tasks = "stack_bowls,align_blocks"
    assert module._resolve_task_names(args) == ["align_blocks", "stack_bowls"]
    args.tasks = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        module._resolve_task_names(args)

    logged_messages = []
    monkeypatch.setattr(module.logger, "info", logged_messages.append)
    missing_tasks = module._write_combined_summary(
        {}, ["stack_bowls"], tmp_path, seed=0, eval_num=10
    )
    assert missing_tasks == ["stack_bowls"]
    summary = json.loads(logged_messages[-1])
    assert summary["missing_tasks"] == ["stack_bowls"]
    assert (tmp_path / "summary_seed_0.json").is_file()


def test_robodojo_eval_writes_official_benchmark_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    spec = importlib.util.spec_from_file_location(
        "robodojo_eval_benchmark",
        common_dir / "robodojo_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result_paths = {}

    def write_result(task_name, successes, score, *, extra_entry=False):
        entries = [
            {"success": index < successes, "score": score}
            for index in range(50)
        ]
        if extra_entry:
            entries.append({"success": True, "score": 1.0})
        result_path = tmp_path / f"{task_name}.json"
        result_path.write_text(
            json.dumps(
                {
                    "details": {
                        str(index): entry
                        for index, entry in reversed(list(enumerate(entries)))
                    }
                }
            )
        )
        result_paths[task_name] = result_path

    for task_name in module.BENCHMARK_DIMENSIONS["Generalization"]:
        write_result(task_name, 25, 0.7, extra_entry=True)
        write_result(f"{task_name}_random", 0, 0.3, extra_entry=True)
    for task_name in module.BENCHMARK_DIMENSIONS["Precision"]:
        write_result(task_name, 50, 0.8)
    for task_name in module.BENCHMARK_DIMENSIONS["Long-Horizon"]:
        write_result(task_name, 0, 0.6)
    for task_name in module.BENCHMARK_DIMENSIONS["Memory"]:
        write_result(task_name, 25, 0.4)
    for task_name in module.BENCHMARK_DIMENSIONS["Open"]:
        write_result(task_name, 10, 0.2)

    module._write_benchmark_summary(result_paths, tmp_path, seed=0)

    summary = json.loads(
        (tmp_path / "benchmark_summary_seed_0.json").read_text()
    )
    assert summary["complete"] is True
    assert summary["metric_unit"] == "percent"
    assert summary["num_tasks"] == 42
    assert summary["completed_tasks"] == 42
    assert summary["num_run_configs"] == 54
    assert summary["expected_episodes"] == 2100
    assert summary["average_success_rate"] == pytest.approx(44.0)
    assert summary["average_score"] == pytest.approx(50.0)
    assert summary["dimension_metrics"]["Generalization"] == {
        "success_rate": pytest.approx(50.0),
        "score": pytest.approx(50.0),
        "num_tasks": 12,
        "completed_tasks": 12,
    }
    assert summary["task_metrics"]["stack_bowls"] == {
        "success_rate": pytest.approx(50.0),
        "score": pytest.approx(50.0),
    }
    assert summary["missing_tasks"] == []
    assert summary["incomplete_tasks"] == {}


def test_robodojo_benchmark_summary_reports_incomplete_tasks(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    common_dir = repo_root / "projects/holobrain_internal/common"
    spec = importlib.util.spec_from_file_location(
        "robodojo_eval_incomplete_benchmark",
        common_dir / "robodojo_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def result_file(task_name, num_episodes):
        result_path = tmp_path / f"{task_name}.json"
        result_path.write_text(
            json.dumps(
                {
                    "details": {
                        str(index): {"success": True, "score": 1.0}
                        for index in range(num_episodes)
                    }
                }
            )
        )
        return result_path

    result_paths = {
        "fasten_screws": result_file("fasten_screws", 49),
        "stack_bowls": result_file("stack_bowls", 25),
        "stack_bowls_random": result_file("stack_bowls_random", 24),
    }

    module._write_benchmark_summary(result_paths, tmp_path, seed=2)

    summary = json.loads(
        (tmp_path / "benchmark_summary_seed_2.json").read_text()
    )
    assert summary["complete"] is False
    assert summary["completed_tasks"] == 0
    assert summary["average_success_rate"] is None
    assert summary["average_score"] is None
    assert summary["incomplete_tasks"] == {
        "stack_bowls": {
            "stack_bowls": 25,
            "stack_bowls_random": 24,
        },
        "fasten_screws": {
            "fasten_screws": 49,
        },
    }
    assert "stack_bowls" not in summary["missing_tasks"]
    assert "fasten_screws" not in summary["missing_tasks"]


def test_policy_server_launcher_accepts_virtualenv_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    source_policy_dir = (
        repo_root
        / "projects/holobrain_internal/common/holobrain_robodojo_policy"
    )
    xpolicy_root = tmp_path / "XPolicyLab"
    policy_dir = xpolicy_root / "policy/holobrain_robodojo_policy"
    policy_dir.mkdir(parents=True)
    launcher = policy_dir / "setup_eval_policy_server.sh"
    launcher.write_bytes(
        (source_policy_dir / "setup_eval_policy_server.sh").read_bytes()
    )
    (policy_dir / "deploy.yml").write_text("protocol: ws\n")

    capture_path = tmp_path / "capture.json"
    (xpolicy_root / "setup_policy_server.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps({\n"
        "    'argv': sys.argv,\n"
        "    'python_path': os.environ.get('PYTHONPATH'),\n"
        "    'virtual_env': os.environ.get('VIRTUAL_ENV'),\n"
        "}))\n"
    )

    virtualenv = tmp_path / "holobrain"
    bin_dir = virtualenv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "activate").write_text(
        f"export VIRTUAL_ENV={virtualenv}\nexport PATH={bin_dir}:$PATH\n"
    )

    env = os.environ.copy()
    env["CAPTURE_PATH"] = str(capture_path)
    caller_pythonpath = tmp_path / "caller-pythonpath"
    ignored_root = tmp_path / "ignored-robo-orchard-root"
    env["PYTHONPATH"] = str(caller_pythonpath)
    env["ROBO_ORCHARD_ROOT"] = str(ignored_root)
    subprocess.run(
        [
            "bash",
            str(launcher),
            "RoboDojo",
            "stack_bowls",
            "holobrain",
            "arx_x5",
            "joint",
            "0",
            "0",
            str(virtualenv),
            "19000",
        ],
        check=True,
        env=env,
    )

    capture = json.loads(capture_path.read_text())
    assert capture["virtual_env"] == str(virtualenv)
    python_paths = capture["python_path"].split(os.pathsep)
    assert str(tmp_path) in python_paths
    assert str(xpolicy_root) in python_paths
    assert str(caller_pythonpath) in python_paths
    assert str(ignored_root) not in python_paths
    assert "port=19000" in capture["argv"]


def test_policy_eval_launcher_runs_server_from_policy_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    source_policy_dir = (
        repo_root
        / "projects/holobrain_internal/common/holobrain_robodojo_policy"
    )
    xpolicy_root = tmp_path / "XPolicyLab"
    policy_dir = xpolicy_root / "policy/holobrain_robodojo_policy"
    utils_dir = xpolicy_root / "utils"
    policy_dir.mkdir(parents=True)
    utils_dir.mkdir()
    (policy_dir / "eval.sh").write_bytes(
        (source_policy_dir / "eval.sh").read_bytes()
    )

    server_cwd_path = tmp_path / "server-cwd"
    (utils_dir / "get_free_port.sh").write_text("echo 19000\n")
    (utils_dir / "wait_for_policy_server.sh").write_text(
        "for _ in $(seq 1 100); do\n"
        '    [[ -s "${SERVER_CWD_PATH}" ]] && exit 0\n'
        "    sleep 0.01\n"
        "done\n"
        "exit 1\n"
    )
    (policy_dir / "setup_eval_policy_server.sh").write_text(
        'pwd > "${SERVER_CWD_PATH}"\nexec sleep 30\n'
    )
    (policy_dir / "setup_eval_env_client.sh").write_text("exit 0\n")

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    env = os.environ.copy()
    env["SERVER_CWD_PATH"] = str(server_cwd_path)
    subprocess.run(
        [
            "bash",
            str(policy_dir / "eval.sh"),
            "RoboDojo",
            "stack_bowls",
            "holobrain",
            "arx_x5",
            "joint",
            "0",
            "0",
            "0",
            "/policy-env",
            "RoboDojo",
        ],
        cwd=caller_dir,
        env=env,
        check=True,
        timeout=10,
    )

    assert Path(server_cwd_path.read_text().strip()) == policy_dir.resolve()
