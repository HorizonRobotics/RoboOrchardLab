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
import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__name__)

_ACTIVE_REMOTE_TASKS: dict[str, dict[str, Any]] = {}
_ACTIVE_REMOTE_TASKS_LOCK = Lock()
_INTERRUPT_EVENT = Event()
_INTERRUPT_CLEANUP_DONE = Event()


def _build_ssh_target(host_config: RemoteHostConfig) -> str:
    return (
        f"{host_config.ssh_user}@{host_config.host}"
        if host_config.ssh_user
        else host_config.host
    )


@dataclass(frozen=True)
class RemoteHostConfig:
    name: str
    host: str
    remote_python: str
    input_path: str
    output_path: str
    user_names: str | None = None
    task_names: str | None = None
    remote_venv_activate: str | None = None
    date_prefix: str | None = None
    token: str | None = None
    port: int = 22
    ssh_user: str | None = None
    ssh_key: str | None = None
    num_workers: int = 4
    connect_timeout: int = 10


def load_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config file must contain an object")
    return payload


def build_remote_host_configs(
    config: dict[str, Any],
) -> list[RemoteHostConfig]:
    defaults = config.get("defaults", {})
    hosts = config.get("hosts", [])
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    if not isinstance(hosts, list) or not hosts:
        raise ValueError("hosts must be a non-empty list")

    host_configs: list[RemoteHostConfig] = []
    for host_payload in hosts:
        if not isinstance(host_payload, dict):
            raise ValueError("each host item must be an object")
        merged = {**defaults, **host_payload}
        host_configs.append(
            RemoteHostConfig(
                name=str(merged.get("name") or merged["host"]),
                host=str(merged["host"]),
                remote_python=str(merged.get("remote_python", "python3")),
                remote_venv_activate=(
                    str(merged["remote_venv_activate"])
                    if merged.get("remote_venv_activate") is not None
                    else None
                ),
                input_path=str(merged["input_path"]),
                output_path=str(merged["output_path"]),
                user_names=(
                    str(merged["user_names"])
                    if merged.get("user_names") is not None
                    else None
                ),
                task_names=(
                    str(merged["task_names"])
                    if merged.get("task_names") is not None
                    else None
                ),
                date_prefix=(
                    str(merged["date_prefix"])
                    if merged.get("date_prefix") is not None
                    else None
                ),
                token=(
                    str(merged["token"])
                    if merged.get("token") is not None
                    else None
                ),
                port=int(merged.get("port", 22)),
                ssh_user=(
                    str(merged["ssh_user"])
                    if merged.get("ssh_user") is not None
                    else None
                ),
                ssh_key=(
                    str(merged["ssh_key"])
                    if merged.get("ssh_key") is not None
                    else None
                ),
                num_workers=int(merged.get("num_workers", 4)),
                connect_timeout=int(merged.get("connect_timeout", 10)),
            )
        )
    return host_configs


def build_remote_command(
    host_config: RemoteHostConfig, remote_script_path: str
) -> str:
    command_parts = [
        shlex.quote(host_config.remote_python),
        shlex.quote(remote_script_path),
        "--input_path",
        shlex.quote(host_config.input_path),
        "--output_path",
        shlex.quote(host_config.output_path),
        "--num_workers",
        shlex.quote(str(host_config.num_workers)),
    ]
    if host_config.user_names is not None:
        command_parts.extend(
            ["--user_names", shlex.quote(host_config.user_names)]
        )
    if host_config.task_names is not None:
        command_parts.extend(
            ["--task_names", shlex.quote(host_config.task_names)]
        )
    if host_config.date_prefix is not None:
        command_parts.extend(
            ["--date_prefix", shlex.quote(host_config.date_prefix)]
        )
    if host_config.token:
        command_parts.extend(["--token", shlex.quote(host_config.token)])
    remote_python_command = " ".join(command_parts)
    if host_config.remote_venv_activate:
        return " && ".join(
            [
                f"source {shlex.quote(host_config.remote_venv_activate)}",
                remote_python_command,
            ]
        )
    return remote_python_command


def build_ssh_command(
    host_config: RemoteHostConfig, remote_script_path: str
) -> list[str]:
    ssh_target = _build_ssh_target(host_config)
    command = [
        "ssh",
        "-p",
        str(host_config.port),
        "-o",
        f"ConnectTimeout={host_config.connect_timeout}",
        "-o",
        "BatchMode=yes",
    ]
    if host_config.ssh_key:
        command.extend(["-i", host_config.ssh_key])
    command.extend(
        [ssh_target, build_remote_command(host_config, remote_script_path)]
    )
    return command


def build_cleanup_ssh_command(
    host_config: RemoteHostConfig, remote_script_path: str
) -> list[str]:
    ssh_target = _build_ssh_target(host_config)
    cleanup_command = "sh -lc " + shlex.quote(
        " && ".join(
            [
                (
                    f"pkill -f {shlex.quote(remote_script_path)} "
                    ">/dev/null 2>&1 || true"
                ),
                (
                    f"rm -f {shlex.quote(remote_script_path)} "
                    ">/dev/null 2>&1 || true"
                ),
            ]
        )
    )
    command = [
        "ssh",
        "-p",
        str(host_config.port),
        "-o",
        f"ConnectTimeout={host_config.connect_timeout}",
        "-o",
        "BatchMode=yes",
    ]
    if host_config.ssh_key:
        command.extend(["-i", host_config.ssh_key])
    command.extend([ssh_target, cleanup_command])
    return command


def get_host_log_path(
    run_log_dir: Path, host_config: RemoteHostConfig
) -> Path:
    run_log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in host_config.name
    )
    return run_log_dir / f"{safe_name}.log"


def register_remote_task(
    host_config: RemoteHostConfig, remote_script_path: str, log_path: Path
) -> None:
    with _ACTIVE_REMOTE_TASKS_LOCK:
        _ACTIVE_REMOTE_TASKS[host_config.name] = {
            "host_config": host_config,
            "remote_script_path": remote_script_path,
            "log_path": log_path,
        }


def unregister_remote_task(host_name: str) -> None:
    with _ACTIVE_REMOTE_TASKS_LOCK:
        _ACTIVE_REMOTE_TASKS.pop(host_name, None)


def interrupt_remote_tasks() -> None:
    if _INTERRUPT_CLEANUP_DONE.is_set():
        return

    with _ACTIVE_REMOTE_TASKS_LOCK:
        active_tasks = list(_ACTIVE_REMOTE_TASKS.values())

    try:
        for task in active_tasks:
            host_config = task["host_config"]
            remote_script_path = task["remote_script_path"]
            log_path = task["log_path"]
            cleanup_command = build_cleanup_ssh_command(
                host_config, remote_script_path
            )
            try:
                with log_path.open("a", encoding="utf-8") as log_file:
                    append_host_log_line(
                        log_file,
                        (
                            "Interrupt received locally, cleaning remote "
                            "process and uploaded script."
                        ),
                    )
                    append_host_log_line(
                        log_file,
                        f"cleanup_ssh_command: {' '.join(cleanup_command)}",
                    )
                cleanup_proc = subprocess.run(
                    cleanup_command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                with log_path.open("a", encoding="utf-8") as log_file:
                    append_host_log_line(
                        log_file,
                        f"cleanup returncode={cleanup_proc.returncode}",
                    )
                    if cleanup_proc.stdout.strip():
                        append_host_log_line(
                            log_file,
                            f"cleanup stdout: {cleanup_proc.stdout.strip()}",
                        )
                    if cleanup_proc.stderr.strip():
                        append_host_log_line(
                            log_file,
                            f"cleanup stderr: {cleanup_proc.stderr.strip()}",
                        )
            except OSError as exc:
                logger.warning(
                    "Failed to clean remote task for %s: %s",
                    host_config.name,
                    exc,
                )
    finally:
        _INTERRUPT_CLEANUP_DONE.set()


def _handle_termination_signal(signum: int, _frame: Any) -> None:
    if _INTERRUPT_EVENT.is_set():
        return
    _INTERRUPT_EVENT.set()
    logger.warning(
        "Received signal %s, interrupting remote uploads...", signum
    )
    interrupt_remote_tasks()


def append_host_log_line(log_file: Any, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {message}\n")
    log_file.flush()


def _stream_process_output(
    stream: Any, prefix: str, lines: list[str], log_file: Any
) -> None:
    try:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            if not line:
                break
            clean_line = line.rstrip("\n")
            lines.append(line)
            append_host_log_line(log_file, f"{prefix}{clean_line}")
    finally:
        if stream is not None:
            stream.close()


def run_remote_upload(
    host_config: RemoteHostConfig, run_log_dir: Path
) -> dict[str, Any]:
    log_path = get_host_log_path(run_log_dir, host_config)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    if _INTERRUPT_CLEANUP_DONE.is_set():
        _INTERRUPT_CLEANUP_DONE.clear()

    if _INTERRUPT_EVENT.is_set():
        return {
            "name": host_config.name,
            "host": host_config.host,
            "returncode": 130,
            "stdout": "",
            "stderr": "Interrupted before start",
            "command": [],
            "log_path": str(log_path),
        }

    # 1. 上传本地 upload_data.py 到远端 /tmp/remote_upload_<name>_<pid>.py
    local_script = Path(__file__).parent / "upload_data.py"
    remote_script_path = "/tmp/remote_upload_{}_{}_{}.py".format(
        "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in host_config.name
        ),
        os.getpid(),
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )

    scp_target = (
        f"{host_config.ssh_user}@{host_config.host}:{remote_script_path}"
        if host_config.ssh_user
        else f"{host_config.host}:{remote_script_path}"
    )
    scp_command = [
        "scp",
        "-P",
        str(host_config.port),
        "-o",
        f"ConnectTimeout={host_config.connect_timeout}",
    ]
    if host_config.ssh_key:
        scp_command.extend(["-i", host_config.ssh_key])
    scp_command.extend([str(local_script), scp_target])

    register_remote_task(host_config, remote_script_path, log_path)

    with log_path.open("a", encoding="utf-8") as log_file:
        host_summary = (
            f"name={host_config.name}, host={host_config.host}, "
            f"ssh_user={host_config.ssh_user or '<default>'}, "
            f"port={host_config.port}, "
            f"remote_python={host_config.remote_python}, "
            "remote_venv_activate="
            f"{host_config.remote_venv_activate or '<none>'}, "
            f"input_path={host_config.input_path}, "
            f"output_path={host_config.output_path}, "
            f"user_names={host_config.user_names}, "
            f"task_names={host_config.task_names}, "
            f"date_prefix={host_config.date_prefix or '<none>'}, "
            f"num_workers={host_config.num_workers}"
        )
        logger.info("Preparing remote upload host: %s", host_summary)
        append_host_log_line(log_file, f"Host info: {host_summary}")
        append_host_log_line(
            log_file,
            (
                f"Uploading upload_data.py to {host_config.name} "
                f"({host_config.host}) as {remote_script_path}"
            ),
        )
        append_host_log_line(log_file, f"scp_command: {' '.join(scp_command)}")
        scp_proc = subprocess.run(scp_command, text=True, capture_output=True)
        if scp_proc.returncode != 0:
            append_host_log_line(log_file, f"SCP failed: {scp_proc.stderr}")
            unregister_remote_task(host_config.name)
            return {
                "name": host_config.name,
                "host": host_config.host,
                "returncode": scp_proc.returncode,
                "stdout": scp_proc.stdout,
                "stderr": scp_proc.stderr,
                "command": scp_command,
                "log_path": str(log_path),
            }

        # 2. SSH 执行远端脚本
        ssh_command = build_ssh_command(host_config, remote_script_path)
        append_host_log_line(log_file, f"ssh_command: {' '.join(ssh_command)}")

        process = subprocess.Popen(
            ssh_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        stdout_thread = Thread(
            target=_stream_process_output,
            args=(process.stdout, "STDOUT: ", stdout_lines, log_file),
            daemon=True,
            name=f"remote-upload-stdout-{host_config.name}",
        )
        stderr_thread = Thread(
            target=_stream_process_output,
            args=(process.stderr, "STDERR: ", stderr_lines, log_file),
            daemon=True,
            name=f"remote-upload-stderr-{host_config.name}",
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            _INTERRUPT_EVENT.set()
            append_host_log_line(
                log_file,
                "KeyboardInterrupt received, terminating local ssh process.",
            )
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            interrupt_remote_tasks()
            raise
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        append_host_log_line(
            log_file, f"Finished remote upload with returncode={returncode}"
        )

    unregister_remote_task(host_config.name)

    return {
        "name": host_config.name,
        "host": host_config.host,
        "returncode": returncode,
        "stdout": "".join(stdout_lines),
        "stderr": "".join(stderr_lines),
        "command": ssh_command,
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trigger upload_data.py on multiple remote collection "
            "machines via SSH."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument(
        "--log-dir", type=Path, default=Path(".remote_upload_logs")
    )
    args = parser.parse_args()

    log_basic_config(
        format="%(asctime)s %(levelname)s-%(lineno)d: %(message)s",
        level=logging.INFO,
    )

    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    config = load_config(args.config)
    host_configs = build_remote_host_configs(config)
    run_log_dir = args.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    max_parallel = (
        len(host_configs)
        if args.max_parallel is None
        else max(args.max_parallel, 1)
    )

    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_map = {
                executor.submit(
                    run_remote_upload, host_config, run_log_dir
                ): host_config
                for host_config in host_configs
            }
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                status = "OK" if result["returncode"] == 0 else "FAILED"
                log_fn = (
                    logger.info if result["returncode"] == 0 else logger.error
                )
                log_fn("[%s] %s (%s)", status, result["name"], result["host"])
    except KeyboardInterrupt:
        _INTERRUPT_EVENT.set()
        logger.warning("Interrupted locally, stopping remote tasks...")
        interrupt_remote_tasks()
        return 130
    finally:
        if _INTERRUPT_EVENT.is_set() and not _INTERRUPT_CLEANUP_DONE.is_set():
            interrupt_remote_tasks()

    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        logger.error("Remote upload failed on %d host(s).", len(failed))
        return 1
    logger.info("Remote upload completed successfully on all hosts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
