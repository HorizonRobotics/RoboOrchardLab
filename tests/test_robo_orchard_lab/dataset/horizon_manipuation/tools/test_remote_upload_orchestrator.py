# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
from pathlib import Path

from robo_orchard_lab.dataset.horizon_manipulation.tools import (
    remote_upload_orchestrator as orchestrator,
)


def test_load_config_reads_yaml_object(tmp_path: Path):
    config_path = tmp_path / "hosts.json"
    config_path.write_text(
        (
            '{"defaults": {"remote_python": "python3"}, "hosts": ['
            '{"host": "192.168.1.101", "input_path": "/data/raw", '
            '"output_path": "/bucket/out", "user_names": "alice", '
            '"task_names": "pick"}]}'
        ),
        encoding="utf-8",
    )

    config = orchestrator.load_config(config_path)

    assert config["defaults"]["remote_python"] == "python3"
    assert config["hosts"][0]["host"] == "192.168.1.101"


def test_build_remote_host_configs_merges_defaults():
    config = {
        "defaults": {
            "remote_python": "python3",
            "output_path": "/bucket/out",
            "user_names": "alice,bob",
            "task_names": "pick,place",
            "date_prefix": "2026_03_17",
            "ssh_user": "robot",
        },
        "hosts": [
            {
                "name": "collector-01",
                "host": "192.168.1.101",
                "input_path": "/data/raw",
            }
        ],
    }

    host_configs = orchestrator.build_remote_host_configs(config)

    assert len(host_configs) == 1
    host = host_configs[0]
    assert host.name == "collector-01"
    assert host.ssh_user == "robot"
    assert host.remote_python == "python3"
    assert host.output_path == "/bucket/out"
    assert host.date_prefix == "2026_03_17"


def test_build_remote_command_includes_upload_args():
    host_config = orchestrator.build_remote_host_configs(
        {
            "defaults": {},
            "hosts": [
                {
                    "host": "192.168.1.101",
                    "remote_python": "python3",
                    "input_path": "/data/raw",
                    "output_path": "/bucket/out",
                    "user_names": "alice,bob",
                    "task_names": "pick,place",
                    "date_prefix": "2026_03_17",
                    "num_workers": 8,
                }
            ],
        }
    )[0]

    command = orchestrator.build_remote_command(
        host_config, "/tmp/remote_upload_data.py"
    )

    assert "python3" in command
    assert "/tmp/remote_upload_data.py" in command
    assert "--input_path" in command
    assert "--output_path" in command
    assert "--user_names" in command
    assert "--task_names" in command
    assert "--date_prefix" in command
    assert "--num_workers" in command


def test_build_ssh_command_includes_target_and_flags():
    host_config = orchestrator.build_remote_host_configs(
        {
            "defaults": {},
            "hosts": [
                {
                    "name": "collector-01",
                    "host": "192.168.1.101",
                    "ssh_user": "robot",
                    "ssh_key": "/home/server/.ssh/id_rsa",
                    "port": 2222,
                    "remote_python": "python3",
                    "input_path": "/data/raw",
                    "output_path": "/bucket/out",
                    "user_names": "alice",
                    "task_names": "pick",
                }
            ],
        }
    )[0]

    command = orchestrator.build_ssh_command(
        host_config, "/tmp/remote_upload_data.py"
    )

    assert command[:5] == ["ssh", "-p", "2222", "-o", "ConnectTimeout=10"]
    assert "-i" in command
    assert "robot@192.168.1.101" in command
    assert "/tmp/remote_upload_data.py" in command[-1]


def test_get_host_log_path_uses_host_name(tmp_path: Path):
    host_config = orchestrator.build_remote_host_configs(
        {
            "defaults": {},
            "hosts": [
                {
                    "name": "collector-01",
                    "host": "192.168.1.101",
                    "remote_python": "python3",
                    "input_path": "/data/raw",
                    "output_path": "/bucket/out",
                    "user_names": "alice",
                    "task_names": "pick",
                }
            ],
        }
    )[0]

    log_path = orchestrator.get_host_log_path(tmp_path, host_config)

    assert log_path == tmp_path / "collector-01.log"


def test_append_host_log_line_writes_message(tmp_path: Path):
    log_path = tmp_path / "collector-01.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        orchestrator.append_host_log_line(log_file, "hello world")

    content = log_path.read_text(encoding="utf-8")
    assert "hello world" in content
