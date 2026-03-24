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


from __future__ import annotations
import io
import json
import os
import queue
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread
from typing import Any, cast
from urllib.parse import quote_plus

import yaml
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
)
from pydantic import BaseModel, ConfigDict


def load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            value
            and len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        os.environ[key] = value


PROJECT_ROOT = Path(__file__).parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent.parent.parent

ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

ENV_CONFIGURED = ENV_PATH.exists()

if ENV_CONFIGURED:
    load_dotenv_file(ENV_PATH)
elif ENV_EXAMPLE_PATH.exists():
    shutil.copy(ENV_EXAMPLE_PATH, ENV_PATH)
    load_dotenv_file(ENV_PATH)


def get_env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def get_env_path(name: str, default: Path) -> Path:
    raw_value = get_env_value(name, str(default)).strip()
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


BASE_DATA_PATH = Path(get_env_value("DATA_ROOT", "./data"))

app = Flask(__name__)

CACHE_LOCK = Lock()
EPISODE_CACHE: dict[str, dict[str, Any]] = {}
CACHE_DIR = get_env_path("CACHE_DIR", PROJECT_ROOT / ".cache")
SUBMIT_CONFIG_DIR = get_env_path(
    "SUBMIT_CONFIG_DIR", PROJECT_ROOT / ".submit_configs"
)
REMOTE_UPLOAD_CONFIG_PATH = get_env_path(
    "REMOTE_UPLOAD_CONFIG_PATH",
    PROJECT_ROOT / "remote_upload_hosts.example.json",
)
AUTO_REFRESH_STARTED = False
SCAN_TASK_LOCK = Lock()
SCAN_TASKS: dict[str, dict[str, Any]] = {}
SUBMIT_TASK_LOCK = Lock()
SUBMIT_TASKS: dict[str, dict[str, Any]] = {}
REMOTE_UPLOAD_TASK_LOCK = Lock()
REMOTE_UPLOAD_TASKS: dict[str, dict[str, Any]] = {}


def reinitialize_from_env() -> None:
    global \
        ENV_CONFIGURED, \
        BASE_DATA_PATH, \
        CACHE_DIR, \
        SUBMIT_CONFIG_DIR, \
        REMOTE_UPLOAD_CONFIG_PATH
    BASE_DATA_PATH = Path(get_env_value("DATA_ROOT", "./data"))
    CACHE_DIR = get_env_path("CACHE_DIR", PROJECT_ROOT / ".cache")
    SUBMIT_CONFIG_DIR = get_env_path(
        "SUBMIT_CONFIG_DIR", PROJECT_ROOT / ".submit_configs"
    )
    REMOTE_UPLOAD_CONFIG_PATH = get_env_path(
        "REMOTE_UPLOAD_CONFIG_PATH",
        PROJECT_ROOT / "remote_upload_hosts.example.json",
    )
    ENV_CONFIGURED = True


def parse_env_file_for_setup(content: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    pending_comments: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            pending_comments.append(line[1:].strip())
        elif "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {'"', "'"}
            ):
                value = value[1:-1]
            fields.append(
                {
                    "key": key,
                    "value": value,
                    "description": " ".join(pending_comments),
                    "needs_attention": "/path/to/your/" in value,
                }
            )
            pending_comments = []
        elif not line:
            pending_comments = []
    return fields


def rebuild_env_content(
    template_content: str, new_values: dict[str, str]
) -> str:
    lines = []
    for raw_line in template_content.splitlines():
        stripped = raw_line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in new_values:
                lines.append(f"{key}={new_values[key]}")
                continue
        lines.append(raw_line)
    return "\n".join(lines) + "\n"


SCAN_MAX_WORKERS = max(2, min(os.cpu_count() or 4, 8))
DEFAULT_EMBODIEDMENT = "piper"


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_name: str
    task_name: str
    embodiedment: str
    episode_id: str
    day: str
    path: str
    duration_hours: float


class ScanCancelledError(RuntimeError):
    pass


class FilterOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_name: str = ""
    task_name: str = ""
    embodiedment: str = ""
    date_prefix: str = ""
    data_root: str = ""
    refresh: bool = False
    page: int = 1
    page_size: int = 20


def normalize_date_prefixes(value: str) -> list[str]:
    return parse_filter_items(normalize_filter(value))


def record_matches_any_date_prefix(
    record: EpisodeRecord, date_prefixes: list[str]
) -> bool:
    if not date_prefixes:
        return False
    episode_time_prefix = extract_episode_time_prefix(record.episode_id)
    return any(
        episode_time_prefix.startswith(prefix) for prefix in date_prefixes
    )


def derive_submit_selection(
    records: list[EpisodeRecord], filters: FilterOptions
) -> dict[str, list[str] | str]:
    if not records:
        raise ValueError("No records selected for submit")

    user_names = sorted({record.user_name for record in records})
    task_names = sorted({record.task_name for record in records})
    requested_date_prefixes = parse_filter_items(filters.date_prefix)
    if requested_date_prefixes:
        date_prefixes = requested_date_prefixes
    else:
        date_prefixes = sorted(
            {
                extract_episode_time_prefix(record.episode_id).split("-")[0]
                for record in records
            }
        )

    all_embodiedments: list[str] = sorted(
        {
            e
            for r in records
            for e in parse_record_embodiedments(r.embodiedment)
        }
    )
    embodiedment = all_embodiedments[0] if all_embodiedments else ""

    return {
        "user_names": user_names,
        "task_names": task_names,
        "date_prefixes": date_prefixes,
        "user_name": user_names[0],
        "task_name": task_names[0],
        "date_prefix": date_prefixes[0],
        "embodiedment": embodiedment,
    }


def get_robo_orchard_lab_dir() -> str:
    value = get_env_value("ROBO_ORCHARD_LAB_DIR", "").strip()
    if not value:
        raise RuntimeError("ROBO_ORCHARD_LAB_DIR is not set")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def get_submit_template_path(source: str) -> Path:
    if source not in {"check", "pack"}:
        raise ValueError("source must be check or pack")
    base_dir = Path(get_robo_orchard_lab_dir())
    template_path = (
        base_dir
        / "dataset"
        / "horizon_manipulation"
        / "tools"
        / f"submit_{source}.json"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"Submit template not found: {template_path}")
    return template_path


def get_submit_config_path(config_id: str) -> Path:
    SUBMIT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return SUBMIT_CONFIG_DIR / f"{config_id}.json"


def get_submit_config_patch_path() -> Path:
    return get_env_path(
        "SUBMIT_CONFIG_PATCH_PATH", PROJECT_ROOT / "submit_config_patch.json"
    )


def build_submit_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(WORKSPACE_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath_entries = [
        entry for entry in existing_pythonpath.split(os.pathsep) if entry
    ]
    if repo_root not in pythonpath_entries:
        pythonpath_entries.insert(0, repo_root)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    if not get_env_bool("SUBMIT_JOB_CLEAR_PROXY", True):
        return env

    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    for key in proxy_keys:
        env.pop(key, None)
    return env


def deep_merge_dict(
    base: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_submit_config_patch() -> dict[str, Any]:
    patch_path = get_submit_config_patch_path()
    if not patch_path.exists():
        return {}

    try:
        payload = json.loads(patch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to load submit config patch: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "submit_config_patch.json must contain a JSON object"
        )
    return payload


def build_initial_submit_config(
    source: str, selection: dict[str, list[str] | str], base_path: Path
) -> dict[str, Any]:
    template = json.loads(
        get_submit_template_path(source).read_text(encoding="utf-8")
    )
    cmd = template.get("cmd", [])
    output_path = (
        "/horizon-bucket/robot_lab/users/xuewu.lin/self-collected-data"
    )
    input_path = str(base_path)
    joined_user_names = ",".join(selection["user_names"])
    joined_task_names = ",".join(selection["task_names"])
    joined_date_prefixes = ",".join(selection["date_prefixes"])
    embodiedment = str(selection.get("embodiedment", ""))

    for index, item in enumerate(cmd):
        if not isinstance(item, str):
            continue
        if item.startswith("date_prefix="):
            cmd[index] = f"date_prefix={joined_date_prefixes}"
        elif item.startswith("user_name="):
            cmd[index] = f"user_name={joined_user_names}"
        elif item.startswith("task_name="):
            cmd[index] = f"task_name={joined_task_names}"
        elif item.startswith("input_path="):
            cmd[index] = f"input_path={input_path}"
        elif item.startswith("    --input_path "):
            cmd[index] = f"    --input_path {input_path} \\"
        elif item.startswith("    --output_path ") and source == "pack":
            cmd[index] = (
                f"    --output_path {output_path}"
                + "/${user_name}-${task_name}-${date_prefix} \\"
            )
        elif item.startswith("    --embodiedment ") and embodiedment:
            suffix = " \\" if item.endswith(" \\") else ""
            cmd[index] = f"    --embodiedment {embodiedment}{suffix}"

    template["cmd"] = cmd
    template["job_name"] = "-".join(
        [
            f"data-{source}",
            *selection["user_names"],
            *selection["task_names"],
            *selection["date_prefixes"],
        ]
    )
    template["to_upload"] = [get_robo_orchard_lab_dir()]
    submit_config_patch = load_submit_config_patch()
    return deep_merge_dict(template, submit_config_patch)


def collect_submit_records(
    base_path: Path, filters: FilterOptions
) -> list[EpisodeRecord]:
    records, _ = get_cached_episode_records(base_path, refresh=False)
    filtered_records = filter_records(records, filters)
    if not filtered_records:
        raise ValueError(
            "Please search first and ensure current filters return at least one episode"  # noqa: E501
        )
    return filtered_records


def create_submit_config(
    source: str, base_path: Path, filters: FilterOptions
) -> dict[str, Any]:
    selected_records = collect_submit_records(base_path, filters)
    selection = derive_submit_selection(selected_records, filters)
    config = build_initial_submit_config(source, selection, base_path)
    config_id = uuid.uuid4().hex
    config_path = get_submit_config_path(config_id)
    config_path.write_text(
        json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "config_id": config_id,
        "config_path": str(config_path),
        "source": source,
        "selection": selection,
        "episode_paths": [record.path for record in selected_records],
        "config": config,
    }


def create_submit_task(
    config_id: str, config_path: Path, command: list[str]
) -> dict[str, Any]:
    task = {
        "task_id": uuid.uuid4().hex,
        "config_id": config_id,
        "config_path": str(config_path),
        "command": command,
        "command_text": " ".join(command),
        "status": "pending",
        "stdout": "",
        "stderr": "",
        "error_message": "",
        "returncode": None,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
    }
    with SUBMIT_TASK_LOCK:
        SUBMIT_TASKS[task["task_id"]] = task
    return dict(task)


def update_submit_task(task_id: str, **kwargs: Any) -> None:
    with SUBMIT_TASK_LOCK:
        if task_id in SUBMIT_TASKS:
            SUBMIT_TASKS[task_id].update(kwargs)


def append_submit_task_log(task_id: str, field: str, text: str) -> None:
    if not text:
        return
    with SUBMIT_TASK_LOCK:
        if task_id in SUBMIT_TASKS:
            SUBMIT_TASKS[task_id][field] = (
                str(SUBMIT_TASKS[task_id].get(field, "")) + text
            )


def get_submit_task(task_id: str) -> dict[str, Any] | None:
    with SUBMIT_TASK_LOCK:
        task = SUBMIT_TASKS.get(task_id)
        return dict(task) if task is not None else None


def get_remote_upload_command(config_path: Path) -> list[str]:
    return [
        "python3",
        str(PROJECT_ROOT / "remote_upload_orchestrator.py"),
        "--config",
        str(config_path),
        "--log-dir",
        str(WORKSPACE_ROOT / ".remote_upload_logs"),
    ]


def create_remote_upload_task(
    config_path: Path, config_text: str
) -> dict[str, Any]:
    task = {
        "task_id": uuid.uuid4().hex,
        "config_path": str(config_path),
        "config_text": config_text,
        "command": get_remote_upload_command(config_path),
        "command_text": " ".join(get_remote_upload_command(config_path)),
        "status": "pending",
        "message": "Waiting to start remote upload...",
        "error_message": "",
        "returncode": None,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "cancel_requested": False,
        "stdout": "",
        "stderr": "",
        "host_logs": {},
        "main_log": "",
        "log_dir": "",
        "log_file_offsets": {},
        "process_pid": None,
    }
    with REMOTE_UPLOAD_TASK_LOCK:
        REMOTE_UPLOAD_TASKS[task["task_id"]] = task
    return dict(task)


def update_remote_upload_task(task_id: str, **kwargs: Any) -> None:
    with REMOTE_UPLOAD_TASK_LOCK:
        if task_id in REMOTE_UPLOAD_TASKS:
            REMOTE_UPLOAD_TASKS[task_id].update(kwargs)


def append_remote_upload_task_log(task_id: str, field: str, text: str) -> None:
    if not text:
        return
    with REMOTE_UPLOAD_TASK_LOCK:
        if task_id in REMOTE_UPLOAD_TASKS:
            REMOTE_UPLOAD_TASKS[task_id][field] = (
                str(REMOTE_UPLOAD_TASKS[task_id].get(field, "")) + text
            )


def append_remote_upload_host_log(
    task_id: str, host_name: str, text: str
) -> None:
    if not text:
        return
    with REMOTE_UPLOAD_TASK_LOCK:
        if task_id not in REMOTE_UPLOAD_TASKS:
            return
        host_logs = cast(
            dict[str, str],
            REMOTE_UPLOAD_TASKS[task_id].setdefault("host_logs", {}),
        )
        host_logs[host_name] = str(host_logs.get(host_name, "")) + text


def get_remote_upload_task(task_id: str) -> dict[str, Any] | None:
    with REMOTE_UPLOAD_TASK_LOCK:
        task = REMOTE_UPLOAD_TASKS.get(task_id)
        if task is None:
            return None
        snapshot = dict(task)
        snapshot["host_logs"] = dict(
            cast(dict[str, str], task.get("host_logs", {}))
        )
        snapshot["log_file_offsets"] = dict(
            cast(dict[str, int], task.get("log_file_offsets", {}))
        )
        return snapshot


def is_remote_upload_cancel_requested(task_id: str) -> bool:
    task = get_remote_upload_task(task_id)
    return bool(task is not None and task.get("cancel_requested"))


def load_remote_upload_default_config() -> str:
    if not REMOTE_UPLOAD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Remote upload config not found: {REMOTE_UPLOAD_CONFIG_PATH}"
        )
    return REMOTE_UPLOAD_CONFIG_PATH.read_text(encoding="utf-8")


def build_remote_upload_default_config(filters: FilterOptions) -> str:
    config_text = load_remote_upload_default_config()
    payload = json.loads(config_text or "{}") or {}
    if not isinstance(payload, dict):
        raise ValueError("Remote upload config must contain a JSON object")

    defaults = payload.get("defaults")
    if defaults is None:
        defaults = {}
        payload["defaults"] = defaults
    if not isinstance(defaults, dict):
        raise ValueError("Remote upload config defaults must be a JSON object")

    user_names = ",".join(parse_filter_items(filters.user_name))
    task_names = ",".join(parse_filter_items(filters.task_name))
    embodiedments = ",".join(parse_filter_items(filters.embodiedment))
    date_prefixes = ",".join(parse_filter_items(filters.date_prefix))
    data_root = normalize_filter(filters.data_root)

    if user_names:
        defaults["user_names"] = user_names
    if task_names:
        defaults["task_names"] = task_names
    if embodiedments:
        defaults["embodiedment"] = embodiedments
    if date_prefixes:
        defaults["date_prefix"] = date_prefixes
    if data_root:
        defaults["output_path"] = data_root

    return json.dumps(payload, indent=2, ensure_ascii=False)


def detect_remote_upload_run_log_dir(text: str) -> Path | None:
    pattern = re.compile(r"\.remote_upload_logs/\d{8}_\d{6}")
    match = pattern.search(text)
    if match is None:
        return None
    candidate = WORKSPACE_ROOT / match.group(0)
    return candidate if candidate.exists() else None


def get_remote_upload_log_base_dir(task: dict[str, Any]) -> Path:
    command = [str(part) for part in cast(list[str], task.get("command", []))]
    for index, part in enumerate(command):
        if part != "--log-dir":
            continue
        if index + 1 < len(command):
            return Path(command[index + 1]).expanduser()
        break
    return WORKSPACE_ROOT / ".remote_upload_logs"


def detect_remote_upload_run_log_dir_from_disk(
    task: dict[str, Any],
) -> Path | None:
    base_dir = get_remote_upload_log_base_dir(task)
    if not base_dir.exists():
        return None

    started_at_raw = str(task.get("started_at", "")).strip()
    started_at: datetime | None = None
    if started_at_raw:
        try:
            started_at = datetime.strptime(started_at_raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            started_at = None

    candidates: list[Path] = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not re.fullmatch(r"\d{8}_\d{6}", child.name):
            continue
        if started_at is None:
            candidates.append(child)
            continue
        try:
            child_time = datetime.strptime(child.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if child_time >= started_at:
            candidates.append(child)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def sync_remote_upload_log_files(task_id: str) -> None:
    task = get_remote_upload_task(task_id)
    if task is None:
        return

    log_dir_value = str(task.get("log_dir", "")).strip()
    log_dir = Path(log_dir_value) if log_dir_value else None
    if log_dir is None or not log_dir.exists():
        detected_dir = detect_remote_upload_run_log_dir_from_disk(task)
        if detected_dir is None:
            combined_output = (
                f"{task.get('stdout', '')}\n{task.get('stderr', '')}"
            )
            detected_dir = detect_remote_upload_run_log_dir(combined_output)
        if detected_dir is None:
            return
        log_dir = detected_dir
        update_remote_upload_task(task_id, log_dir=str(log_dir))

    file_offsets = cast(dict[str, int], task.get("log_file_offsets", {}))
    for log_file in sorted(log_dir.glob("*.log")):
        file_key = str(log_file)
        offset = int(file_offsets.get(file_key, 0))
        try:
            content = log_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if offset >= len(content):
            continue
        new_chunk = content[offset:]
        host_name = log_file.stem
        append_remote_upload_host_log(task_id, host_name, new_chunk)
        update_remote_upload_task(
            task_id,
            log_file_offsets={
                **cast(
                    dict[str, int],
                    (get_remote_upload_task(task_id) or {}).get(
                        "log_file_offsets", {}
                    ),
                ),
                file_key: len(content),
            },
        )


def _pump_stream(
    stream: Any,
    field: str,
    log_queue: queue.Queue[tuple[str, str]],
) -> None:
    try:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            if not line:
                break
            log_queue.put((field, line))
    finally:
        if stream is not None:
            stream.close()


def run_remote_upload_task(task_id: str, command: list[str]) -> None:
    update_remote_upload_task(
        task_id, status="running", message="Remote upload is running..."
    )
    log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=build_submit_subprocess_env(),
            cwd=str(PROJECT_ROOT),
        )
    except OSError as exc:
        update_remote_upload_task(
            task_id,
            status="failed",
            error_message=str(exc),
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    update_remote_upload_task(task_id, process_pid=process.pid)

    stdout_thread = Thread(
        target=_pump_stream,
        args=(process.stdout, "stdout", log_queue),
        daemon=True,
        name=f"remote-upload-stdout-{task_id[:8]}",
    )
    stderr_thread = Thread(
        target=_pump_stream,
        args=(process.stderr, "stderr", log_queue),
        daemon=True,
        name=f"remote-upload-stderr-{task_id[:8]}",
    )
    stdout_thread.start()
    stderr_thread.start()

    while process.poll() is None:
        while True:
            try:
                field, chunk = log_queue.get_nowait()
            except queue.Empty:
                break
            append_remote_upload_task_log(task_id, field, chunk)
        sync_remote_upload_log_files(task_id)
        if is_remote_upload_cancel_requested(task_id):
            update_remote_upload_task(
                task_id,
                message=(
                    "Cancel requested. Waiting for the remote upload "
                    "process to stop..."
                ),
            )
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                append_remote_upload_task_log(
                    task_id,
                    "stderr",
                    (
                        "Local orchestrator did not exit after SIGTERM "
                        "within 20 seconds; sending SIGKILL.\n"
                    ),
                )
                process.kill()
                process.wait()
            break
        time.sleep(0.1)

    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    while True:
        try:
            field, chunk = log_queue.get_nowait()
        except queue.Empty:
            break
        append_remote_upload_task_log(task_id, field, chunk)
    sync_remote_upload_log_files(task_id)

    returncode = process.returncode
    cancelled = is_remote_upload_cancel_requested(task_id)
    task_snapshot = get_remote_upload_task(task_id) or {}
    stderr_text = str(task_snapshot.get("stderr", "")).strip()
    stdout_text = str(task_snapshot.get("stdout", "")).strip()
    update_remote_upload_task(
        task_id,
        returncode=returncode,
        process_pid=None,
        status="cancelled"
        if cancelled
        else ("completed" if returncode == 0 else "failed"),
        message=(
            "Remote upload cancelled"
            if cancelled
            else (
                "Remote upload completed successfully"
                if returncode == 0
                else (stderr_text or stdout_text or "Remote upload failed")
            )
        ),
        error_message=""
        if cancelled or returncode == 0
        else (stderr_text or stdout_text),
        finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def start_remote_upload_task(config_text: str) -> dict[str, Any]:
    json.loads(config_text)
    config_dir = CACHE_DIR / "remote_upload_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"remote_upload_{uuid.uuid4().hex}.json"
    config_path.write_text(config_text, encoding="utf-8")
    task = create_remote_upload_task(config_path, config_text)
    thread = Thread(
        target=run_remote_upload_task,
        args=(task["task_id"], task["command"]),
        daemon=True,
        name=f"remote-upload-task-{task['task_id'][:8]}",
    )
    thread.start()
    return get_remote_upload_task(task["task_id"]) or task


def _drain_submit_queue(
    task_id: str, log_queue: queue.Queue[tuple[str, str]]
) -> None:
    while True:
        try:
            stream_name, chunk = log_queue.get_nowait()
        except queue.Empty:
            break
        append_submit_task_log(task_id, stream_name, chunk)


def run_submit_task(task_id: str, command: list[str]) -> None:
    update_submit_task(task_id, status="running")
    log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=build_submit_subprocess_env(),
        )
    except OSError as exc:
        update_submit_task(
            task_id,
            status="failed",
            error_message=str(exc),
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    stdout_thread = Thread(
        target=_pump_stream,
        args=(process.stdout, "stdout", log_queue),
        daemon=True,
        name=f"submit-stdout-{task_id[:8]}",
    )
    stderr_thread = Thread(
        target=_pump_stream,
        args=(process.stderr, "stderr", log_queue),
        daemon=True,
        name=f"submit-stderr-{task_id[:8]}",
    )
    stdout_thread.start()
    stderr_thread.start()

    while process.poll() is None:
        _drain_submit_queue(task_id, log_queue)
        time.sleep(0.1)

    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    _drain_submit_queue(task_id, log_queue)

    returncode = process.returncode
    update_submit_task(
        task_id,
        returncode=returncode,
        status="submitted" if returncode == 0 else "failed",
        finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def start_submit_task(
    config_id: str, config_data: dict[str, Any]
) -> dict[str, Any]:
    config_path = get_submit_config_path(config_id)
    if not config_path.exists():
        raise FileNotFoundError("submit config not found")

    config_path.write_text(
        json.dumps(config_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    command = [
        "RoboOrchardJob-AIDISubmit",
        "submit_from_config",
        "--config",
        str(config_path),
    ]
    task = create_submit_task(config_id, config_path, command)
    thread = Thread(
        target=run_submit_task,
        args=(task["task_id"], command),
        daemon=True,
        name=f"submit-task-{task['task_id'][:8]}",
    )
    thread.start()
    return get_submit_task(task["task_id"]) or task


def infer_day_from_episode_dir(episode_dir: Path) -> str:
    """Infer a YYYY-MM-DD date from directory metadata.

    Prefer a date parsed from `episode_id` when the directory name follows
    `episode_YYYY_MM_DD...`; otherwise fall back to directory mtime because the
    required layout only guarantees `user_name/task_name/episode_id`.
    """
    episode_name = episode_dir.name
    if episode_name.startswith("episode_"):
        raw_prefix = episode_name[len("episode_") :].split("-", 1)[0]
        try:
            return datetime.strptime(raw_prefix, "%Y_%m_%d").strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            pass

    ts = episode_dir.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def round_hours(value: float) -> float:
    return round(value, 4)


def format_duration_hours(value: float) -> str:
    total_seconds = max(int(round(value * 3600)), 0)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours} hours")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes} mins")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} secs")
    return " ".join(parts)


def read_duration_hours(episode_dir: Path) -> float:
    metadata_path = episode_dir / "metadata.yaml"
    if not metadata_path.exists():
        return 0.0

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return 0.0

    bag_info = metadata.get("rosbag2_bagfile_information")
    if isinstance(bag_info, dict):
        duration = bag_info.get("duration")
    else:
        duration = metadata.get("duration")

    if not isinstance(duration, dict):
        return 0.0

    nanoseconds = duration.get("nanoseconds", 0)
    try:
        nanoseconds_value = float(nanoseconds)
    except (TypeError, ValueError):
        return 0.0

    return round_hours(nanoseconds_value / 1_000_000_000 / 3600)


def read_embodiedment_tag(episode_dir: Path) -> str:
    episode_meta_path = episode_dir / "episode_meta.json"
    if not episode_meta_path.exists():
        return ""

    try:
        payload = json.loads(episode_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    metas = payload.get("metas")
    if not isinstance(metas, dict):
        return ""

    embodiedment = metas.get("embodiedment")
    if not isinstance(embodiedment, list):
        return ""

    normalized = sorted(
        item.strip().lower()
        for item in embodiedment
        if isinstance(item, str) and item.strip()
    )
    return ",".join(normalized)


def parse_record_embodiedments(value: str) -> list[str]:
    items = parse_filter_items(value)
    return items or [DEFAULT_EMBODIEDMENT]


def resolve_embodiedment(value: str) -> str:
    return ",".join(parse_record_embodiedments(value))


def build_episode_record(
    user_name: str, task_name: str, episode_dir: Path
) -> EpisodeRecord:
    return EpisodeRecord(
        user_name=user_name,
        task_name=task_name,
        embodiedment=read_embodiedment_tag(episode_dir),
        episode_id=episode_dir.name,
        day=infer_day_from_episode_dir(episode_dir),
        path=str(episode_dir),
        duration_hours=read_duration_hours(episode_dir),
    )


def is_valid_episode_dir(episode_dir: Path) -> bool:
    required_files = [
        episode_dir / "episode_meta.json",
        episode_dir / "metadata.yaml",
    ]
    if any(not file_path.is_file() for file_path in required_files):
        return False

    return any(path.is_file() for path in episode_dir.glob("*.mcap"))


def iter_episode_dirs(
    base_path: Path,
    task_id: str | None = None,
    date_prefixes: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    if not base_path.exists():
        return []

    normalized_date_prefixes = [
        prefix for prefix in date_prefixes or [] if prefix
    ]
    user_dirs = sorted([p for p in base_path.iterdir() if p.is_dir()])
    total_task_dirs = sum(
        len([p for p in user_dir.iterdir() if p.is_dir()])
        for user_dir in user_dirs
    )
    scanned_task_dirs = 0
    episode_dirs: list[tuple[str, str, Path]] = []
    for user_dir in user_dirs:
        if task_id is not None and is_scan_cancel_requested(task_id):
            raise ScanCancelledError()
        for task_dir in sorted([p for p in user_dir.iterdir() if p.is_dir()]):
            if task_id is not None and is_scan_cancel_requested(task_id):
                raise ScanCancelledError()
            if task_id is not None:
                update_scan_task(
                    task_id,
                    status="running",
                    progress=max(
                        int(scanned_task_dirs * 100 / total_task_dirs), 1
                    )
                    if total_task_dirs
                    else 1,
                    processed=len(episode_dirs),
                    total=max(len(episode_dirs), 1),
                    message=(
                        "Listing episode directories: "
                        f"{task_dir.relative_to(base_path)}"
                    ),
                )

            task_episode_dirs = sorted(
                [p for p in task_dir.iterdir() if p.is_dir()]
            )
            task_episode_total = len(task_episode_dirs)

            for episode_index, episode_dir in enumerate(
                task_episode_dirs, start=1
            ):
                if task_id is not None and is_scan_cancel_requested(task_id):
                    raise ScanCancelledError()
                episode_time_prefix = extract_episode_time_prefix(
                    episode_dir.name
                )
                if task_id is not None:
                    update_scan_task(
                        task_id,
                        status="running",
                        progress=max(
                            int(scanned_task_dirs * 100 / total_task_dirs), 1
                        )
                        if total_task_dirs
                        else 1,
                        processed=episode_index,
                        total=max(task_episode_total, 1),
                        message=(
                            "Validating episodes in "
                            f"{task_dir.relative_to(base_path)}"
                        ),
                    )
                if normalized_date_prefixes and not any(
                    episode_time_prefix.startswith(prefix)
                    for prefix in normalized_date_prefixes
                ):
                    continue
                if not is_valid_episode_dir(episode_dir):
                    continue
                episode_dirs.append(
                    (user_dir.name, task_dir.name, episode_dir)
                )
            scanned_task_dirs += 1
            if task_id is not None:
                progress = (
                    int(scanned_task_dirs * 100 / total_task_dirs)
                    if total_task_dirs
                    else 100
                )
                update_scan_task(
                    task_id,
                    status="running",
                    progress=max(progress, 1),
                    processed=len(episode_dirs),
                    total=max(len(episode_dirs), 1),
                    message=(
                        "Scanning directory structure: "
                        f"{scanned_task_dirs}/{total_task_dirs} task directories checked, "  # noqa: E501
                        f"{len(episode_dirs)} valid episodes found..."
                    ),
                )
    return episode_dirs


def build_records_from_episode_dirs_with_progress(
    episode_dirs: list[tuple[str, str, Path]], task_id: str
) -> list[EpisodeRecord]:
    total = len(episode_dirs)
    if total == 0:
        update_scan_task(
            task_id,
            status="completed",
            progress=100,
            message="No matching episode dir were found during scanning",
            processed=0,
            total=0,
        )
        return []

    records: list[EpisodeRecord] = []

    def build_record(item: tuple[str, str, Path]) -> EpisodeRecord:
        if is_scan_cancel_requested(task_id):
            raise ScanCancelledError()
        user_name, task_name, episode_dir = item
        record = build_episode_record(user_name, task_name, episode_dir)
        if is_scan_cancel_requested(task_id):
            raise ScanCancelledError()
        return record

    processed = 0
    executor = ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS)
    try:
        pending = {
            executor.submit(build_record, item) for item in episode_dirs
        }

        while pending:
            if is_scan_cancel_requested(task_id):
                for future in pending:
                    future.cancel()
                update_scan_task(
                    task_id,
                    status="cancelled",
                    message="Scan cancelled",
                    progress=int(processed * 100 / total) if total else 0,
                    processed=processed,
                    total=total,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                return []

            done, pending = wait(
                pending, timeout=0.05, return_when=FIRST_COMPLETED
            )
            for future in done:
                try:
                    record = future.result()
                except ScanCancelledError:
                    if is_scan_cancel_requested(task_id):
                        for pending_future in pending:
                            pending_future.cancel()
                        update_scan_task(
                            task_id,
                            status="cancelled",
                            message="Scan cancelled",
                            progress=(
                                int(processed * 100 / total) if total else 0
                            ),
                            processed=processed,
                            total=total,
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        return []
                    continue
                records.append(record)
                processed += 1
                update_scan_task(
                    task_id,
                    status="running",
                    progress=int(processed * 100 / total),
                    processed=processed,
                    total=total,
                    message=f"Scanning episodes: {processed}/{total}...",
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=False)

    update_scan_task(
        task_id,
        status="completed",
        progress=100,
        processed=total,
        total=total,
        message="Scan completed",
    )
    return records


def merge_records_for_date_prefixes(
    existing_records: list[EpisodeRecord],
    refreshed_records: list[EpisodeRecord],
    date_prefixes: list[str],
) -> list[EpisodeRecord]:
    if not date_prefixes:
        return refreshed_records

    preserved_records = [
        record
        for record in existing_records
        if not record_matches_any_date_prefix(record, date_prefixes)
    ]
    return preserved_records + refreshed_records


def refresh_cached_episode_records_for_date_prefixes(
    base_path: Path,
    date_prefixes: list[str],
    task_id: str,
) -> list[EpisodeRecord]:
    if not date_prefixes:
        raise ValueError("date_prefix is required for partial refresh")

    cache_key = get_cache_key(base_path)
    existing_records, _ = get_cached_episode_records(base_path, refresh=False)

    try:
        filtered_episode_dirs = iter_episode_dirs(
            base_path,
            task_id=task_id,
            date_prefixes=date_prefixes,
        )
    except ScanCancelledError:
        update_scan_task(
            task_id,
            status="cancelled",
            message="Scan cancelled",
        )
        return []

    update_scan_task(
        task_id,
        status="running",
        progress=0,
        processed=0,
        total=len(filtered_episode_dirs),
        message=(
            "Refreshing cached records for date prefixes: "
            + ", ".join(date_prefixes)
        ),
    )

    refreshed_records = build_records_from_episode_dirs_with_progress(
        filtered_episode_dirs, task_id
    )
    task_state = get_scan_task(task_id)
    if task_state is not None and task_state.get("status") == "cancelled":
        return []

    merged_records = merge_records_for_date_prefixes(
        existing_records, refreshed_records, date_prefixes
    )
    cache_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_cache_entry(cache_key, merged_records, cache_created_at)
    update_scan_task(
        task_id,
        status="completed",
        progress=100,
        processed=len(refreshed_records),
        total=len(filtered_episode_dirs),
        message="Partial refresh completed, the page will refresh shortly",
    )
    return merged_records


def scan_episode_records_with_progress(
    base_path: Path, task_id: str
) -> list[EpisodeRecord]:
    try:
        episode_dirs = iter_episode_dirs(base_path, task_id=task_id)
    except ScanCancelledError:
        update_scan_task(
            task_id,
            status="cancelled",
            message="Scan cancelled",
        )
        return []
    return build_records_from_episode_dirs_with_progress(episode_dirs, task_id)


def scan_episode_records_parallel(base_path: Path) -> list[EpisodeRecord]:
    episode_dirs = iter_episode_dirs(base_path)
    if not episode_dirs:
        return []

    def build_record(item: tuple[str, str, Path]) -> EpisodeRecord:
        user_name, task_name, episode_dir = item
        return build_episode_record(user_name, task_name, episode_dir)

    records: list[EpisodeRecord] = []
    with ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS) as executor:
        pending = {
            executor.submit(build_record, item) for item in episode_dirs
        }
        while pending:
            done, pending = wait(
                pending, timeout=0.05, return_when=FIRST_COMPLETED
            )
            for future in done:
                records.append(future.result())
    return records


def normalize_filter(value: str | None) -> str:
    return (value or "").strip()


def parse_filter_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_base_paths(filters: FilterOptions) -> list[Path]:
    paths = [Path(item) for item in parse_filter_items(filters.data_root)]
    return paths or [BASE_DATA_PATH]


def format_base_paths(base_paths: list[Path]) -> str:
    return ",".join(str(path) for path in base_paths)


def extract_episode_time_prefix(episode_id: str) -> str:
    if episode_id.startswith("episode_"):
        return episode_id[len("episode_") :]
    return episode_id


def parse_filters(args: Any) -> FilterOptions:
    page_raw = normalize_filter(args.get("page"))
    page_size_raw = normalize_filter(args.get("page_size"))
    try:
        page = max(int(page_raw), 1) if page_raw else 1
    except ValueError:
        page = 1
    try:
        page_size = max(int(page_size_raw), 1) if page_size_raw else 20
    except ValueError:
        page_size = 20

    return FilterOptions(
        user_name=normalize_filter(args.get("user_name")),
        task_name=normalize_filter(args.get("task_name")),
        embodiedment=normalize_filter(args.get("embodiedment")),
        date_prefix=normalize_filter(args.get("date_prefix")),
        data_root=normalize_filter(args.get("data_root")),
        refresh=normalize_filter(args.get("refresh")).lower()
        in {"1", "true", "yes", "y"},
        page=page,
        page_size=page_size,
    )


def resolve_base_path(filters: FilterOptions) -> Path:
    return resolve_base_paths(filters)[0]


def get_cache_key(base_path: Path) -> str:
    try:
        normalized = str(base_path.resolve())
    except OSError:
        normalized = str(base_path)
    return normalized


def get_cache_file_path(cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = sha256(cache_key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def has_cached_episode_records(base_path: Path) -> bool:
    cache_key = get_cache_key(base_path)
    with CACHE_LOCK:
        if cache_key in EPISODE_CACHE:
            return True
    return get_cache_file_path(cache_key).exists()


def save_cache_to_disk(
    cache_key: str, records: list[EpisodeRecord], cache_created_at: str
) -> None:
    cache_file = get_cache_file_path(cache_key)
    payload = {
        "cache_key": cache_key,
        "cache_created_at": cache_created_at,
        "records": [record.model_dump(mode="json") for record in records],
    }
    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_cache_from_disk(cache_key: str) -> dict[str, Any] | None:
    cache_file = get_cache_file_path(cache_key)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    records_data = payload.get("records", [])
    if not isinstance(records_data, list):
        return None

    try:
        records = [EpisodeRecord.model_validate(item) for item in records_data]
    except Exception:
        return None

    return {
        "records": records,
        "cache_created_at": str(payload.get("cache_created_at", "")),
    }


def update_cache_entry(
    cache_key: str, records: list[EpisodeRecord], cache_created_at: str
) -> None:
    with CACHE_LOCK:
        EPISODE_CACHE[cache_key] = {
            "records": records,
            "cache_created_at": cache_created_at,
        }
    save_cache_to_disk(cache_key, records, cache_created_at)


def create_scan_task(base_path: Path) -> str:
    task_id = uuid.uuid4().hex
    with SCAN_TASK_LOCK:
        SCAN_TASKS[task_id] = {
            "task_id": task_id,
            "base_path": str(base_path),
            "status": "pending",
            "progress": 0,
            "processed": 0,
            "total": 0,
            "cancel_requested": False,
            "message": "Task created, waiting to start...",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    return task_id


def update_scan_task(task_id: str, **kwargs: Any) -> None:
    with SCAN_TASK_LOCK:
        if task_id in SCAN_TASKS:
            SCAN_TASKS[task_id].update(kwargs)


def get_scan_task(task_id: str) -> dict[str, Any] | None:
    with SCAN_TASK_LOCK:
        task = SCAN_TASKS.get(task_id)
        return dict(task) if task is not None else None


def is_scan_cancel_requested(task_id: str) -> bool:
    task_state = get_scan_task(task_id)
    return bool(task_state is not None and task_state.get("cancel_requested"))


def run_scan_task(task_id: str, base_path: Path) -> None:
    try:
        task_state = get_scan_task(task_id) or {}
        refresh_mode = str(task_state.get("refresh_mode", "full"))
        target_date_prefixes = [
            str(item)
            for item in task_state.get("target_date_prefixes", [])
            if str(item).strip()
        ]
        update_scan_task(
            task_id,
            status="running",
            progress=1,
            message=(
                "Preparing to incrementally refresh selected dates..."
                if refresh_mode == "partial"
                else "Preparing to scan directories..."
            ),
        )
        if refresh_mode == "partial":
            records = refresh_cached_episode_records_for_date_prefixes(
                base_path, target_date_prefixes, task_id
            )
        else:
            records = scan_episode_records_with_progress(base_path, task_id)
        task_state = get_scan_task(task_id)
        if task_state is not None and task_state.get("status") == "cancelled":
            return
        if refresh_mode != "partial":
            cache_key = get_cache_key(base_path)
            cache_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            update_cache_entry(cache_key, records, cache_created_at)
            update_scan_task(
                task_id,
                status="completed",
                progress=100,
                message="Scan completed, the page will refresh shortly",
            )
    except Exception as exc:  # noqa: BLE001
        update_scan_task(
            task_id, status="failed", message=f"Scan failed: {exc}"
        )


def start_scan_task(
    base_path: Path,
    refresh_mode: str = "full",
    target_date_prefixes: list[str] | None = None,
) -> str:
    task_id = create_scan_task(base_path)
    if refresh_mode == "partial":
        update_scan_task(
            task_id,
            refresh_mode="partial",
            target_date_prefixes=target_date_prefixes or [],
        )
    thread = Thread(
        target=run_scan_task,
        args=(task_id, base_path),
        daemon=True,
        name=f"scan-task-{task_id[:8]}",
    )
    thread.start()
    return task_id


def get_cached_episode_records(
    base_path: Path, refresh: bool = False
) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    cache_key = get_cache_key(base_path)

    with CACHE_LOCK:
        if not refresh and cache_key in EPISODE_CACHE:
            cached = EPISODE_CACHE[cache_key]
            return cached["records"], {
                "cache_hit": True,
                "cache_source": "memory",
                "cache_created_at": cached["cache_created_at"],
                "cache_entry_count": len(cached["records"]),
            }

    if not refresh:
        disk_cached = load_cache_from_disk(cache_key)
        if disk_cached is not None:
            with CACHE_LOCK:
                EPISODE_CACHE[cache_key] = disk_cached
            return disk_cached["records"], {
                "cache_hit": True,
                "cache_source": "disk",
                "cache_created_at": disk_cached["cache_created_at"],
                "cache_entry_count": len(disk_cached["records"]),
            }

    records = scan_episode_records_parallel(base_path)
    cache_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_cache_entry(cache_key, records, cache_created_at)

    return records, {
        "cache_hit": False,
        "cache_source": "refresh",
        "cache_created_at": cache_created_at,
        "cache_entry_count": len(records),
    }


def merge_cache_infos(cache_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not cache_infos:
        return {
            "cache_hit": False,
            "cache_source": "refresh",
            "cache_created_at": "",
            "cache_entry_count": 0,
        }

    cache_hit = all(bool(item.get("cache_hit")) for item in cache_infos)
    cache_sources = [
        str(item.get("cache_source", "")) for item in cache_infos if item
    ]
    cache_created_candidates = [
        str(item.get("cache_created_at", ""))
        for item in cache_infos
        if str(item.get("cache_created_at", "")).strip()
    ]
    return {
        "cache_hit": cache_hit,
        "cache_source": ",".join(cache_sources),
        "cache_created_at": max(cache_created_candidates, default=""),
        "cache_entry_count": sum(
            int(item.get("cache_entry_count", 0)) for item in cache_infos
        ),
    }


def get_cached_episode_records_for_paths(
    base_paths: list[Path], refresh: bool = False
) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    all_records: list[EpisodeRecord] = []
    cache_infos: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for base_path in base_paths:
        records, cache_info = get_cached_episode_records(
            base_path, refresh=refresh
        )
        cache_infos.append(cache_info)
        for record in records:
            if record.path in seen_paths:
                continue
            seen_paths.add(record.path)
            all_records.append(record)

    return all_records, merge_cache_infos(cache_infos)


def create_episode_zip_bytes(episode_path: Path) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file_path in sorted(episode_path.rglob("*")):
            if file_path.is_file():
                archive.write(
                    file_path, arcname=file_path.relative_to(episode_path)
                )
    buffer.seek(0)
    return buffer


def build_download_url(episode_path: str) -> str:
    return f"/api/download?episode_path={quote_plus(episode_path)}"


def count_episode_files(episode_path: Path) -> int:
    return sum(1 for path in episode_path.rglob("*") if path.is_file())


def refresh_all_cached_paths() -> None:
    with CACHE_LOCK:
        cache_keys = list(EPISODE_CACHE.keys())

    for cache_key in cache_keys:
        get_cached_episode_records(Path(cache_key), refresh=True)


def seconds_until_next_refresh(now: datetime | None = None) -> float:
    current = now or datetime.now()
    next_refresh = current.replace(hour=2, minute=0, second=0, microsecond=0)
    if current >= next_refresh:
        next_refresh += timedelta(days=1)
    return max((next_refresh - current).total_seconds(), 0.0)


def auto_refresh_worker() -> None:
    while True:
        time.sleep(seconds_until_next_refresh())
        refresh_all_cached_paths()


def ensure_auto_refresh_started() -> None:
    global AUTO_REFRESH_STARTED
    with CACHE_LOCK:
        if AUTO_REFRESH_STARTED:
            return
        thread = Thread(
            target=auto_refresh_worker, daemon=True, name="cache-auto-refresh"
        )
        thread.start()
        AUTO_REFRESH_STARTED = True


def filter_records(
    records: list[EpisodeRecord], filters: FilterOptions
) -> list[EpisodeRecord]:
    user_names = set(parse_filter_items(filters.user_name))
    task_names = set(parse_filter_items(filters.task_name))
    embodiedments = set(parse_filter_items(filters.embodiedment))
    date_prefixes = parse_filter_items(filters.date_prefix)

    def matched(record: EpisodeRecord) -> bool:
        if user_names and record.user_name not in user_names:
            return False
        if task_names and record.task_name not in task_names:
            return False
        if embodiedments and embodiedments.isdisjoint(
            parse_record_embodiedments(record.embodiedment)
        ):
            return False
        episode_time_prefix = extract_episode_time_prefix(record.episode_id)
        if date_prefixes and not any(
            episode_time_prefix.startswith(prefix) for prefix in date_prefixes
        ):
            return False
        return True

    return [record for record in records if matched(record)]


def build_summary(
    records: list[EpisodeRecord], base_path: Path
) -> dict[str, Any]:
    by_user_task_day: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    by_user_task_day_hours: dict[str, dict[str, dict[str, float]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    by_day_total: dict[str, int] = defaultdict(int)
    by_day_hours: dict[str, float] = defaultdict(float)
    by_user_total: dict[str, int] = defaultdict(int)
    by_user_hours: dict[str, float] = defaultdict(float)
    by_task_total: dict[str, int] = defaultdict(int)
    by_task_hours: dict[str, float] = defaultdict(float)
    by_embodiedment_total: dict[str, int] = defaultdict(int)
    by_embodiedment_hours: dict[str, float] = defaultdict(float)

    for record in records:
        by_user_task_day[record.user_name][record.task_name][record.day] += 1
        by_user_task_day_hours[record.user_name][record.task_name][
            record.day
        ] += record.duration_hours
        by_day_total[record.day] += 1
        by_day_hours[record.day] += record.duration_hours
        by_user_total[record.user_name] += 1
        by_user_hours[record.user_name] += record.duration_hours
        by_task_total[record.task_name] += 1
        by_task_hours[record.task_name] += record.duration_hours
        for resolved_embodiedment in parse_record_embodiedments(
            record.embodiedment
        ):
            by_embodiedment_total[resolved_embodiedment] += 1
            by_embodiedment_hours[resolved_embodiedment] += (
                record.duration_hours
            )

    all_days = sorted(by_day_total.keys())
    users: list[dict[str, Any]] = []
    for user_name in sorted(by_user_task_day.keys()):
        tasks = []
        for task_name in sorted(by_user_task_day[user_name].keys()):
            day_counts = dict(
                sorted(by_user_task_day[user_name][task_name].items())
            )
            day_hours = {
                day: round_hours(hours)
                for day, hours in sorted(
                    by_user_task_day_hours[user_name][task_name].items()
                )
            }
            tasks.append(
                {
                    "task_name": task_name,
                    "total": sum(day_counts.values()),
                    "day_counts": day_counts,
                    "total_hours": round_hours(sum(day_hours.values())),
                    "total_duration_text": format_duration_hours(
                        sum(day_hours.values())
                    ),
                    "day_hours": day_hours,
                    "day_duration_text": {
                        day: format_duration_hours(hours)
                        for day, hours in day_hours.items()
                    },
                }
            )
        users.append(
            {
                "user_name": user_name,
                "total": by_user_total[user_name],
                "total_hours": round_hours(by_user_hours[user_name]),
                "total_duration_text": format_duration_hours(
                    by_user_hours[user_name]
                ),
                "tasks": tasks,
            }
        )

    hours_by_day = {
        day: round_hours(hours) for day, hours in sorted(by_day_hours.items())
    }
    hours_by_user = {
        user_name: round_hours(hours)
        for user_name, hours in sorted(by_user_hours.items())
    }
    hours_by_task = {
        task_name: round_hours(hours)
        for task_name, hours in sorted(by_task_hours.items())
    }
    hours_by_embodiedment = {
        embodiedment: round_hours(hours)
        for embodiedment, hours in sorted(by_embodiedment_hours.items())
    }
    episodes = [
        {
            "episode_id": record.episode_id,
            "user_name": record.user_name,
            "task_name": record.task_name,
            "embodiedment": resolve_embodiedment(record.embodiedment),
            "day": record.day,
            "path": record.path,
            "duration_hours": record.duration_hours,
            "duration_text": format_duration_hours(record.duration_hours),
            "download_url": build_download_url(record.path),
        }
        for record in sorted(
            records,
            key=lambda item: (
                item.day,
                item.user_name,
                item.task_name,
                item.episode_id,
            ),
            reverse=True,
        )
    ]

    return {
        "base_path": str(base_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes": len(records),
        "total_hours": round_hours(
            sum(record.duration_hours for record in records)
        ),
        "total_duration_text": format_duration_hours(
            sum(record.duration_hours for record in records)
        ),
        "filters": {
            "user_name": "",
            "task_name": "",
            "embodiedment": "",
            "date_prefix": "",
            "data_root": str(base_path),
            "refresh": False,
        },
        "cache": {
            "cache_hit": False,
            "cache_source": "refresh",
            "cache_created_at": "",
            "cache_entry_count": len(records),
        },
        "days": all_days,
        "totals": {
            "by_day": dict(sorted(by_day_total.items())),
            "hours_by_day": hours_by_day,
            "duration_text_by_day": {
                day: format_duration_hours(hours)
                for day, hours in hours_by_day.items()
            },
            "by_user": dict(sorted(by_user_total.items())),
            "hours_by_user": hours_by_user,
            "duration_text_by_user": {
                user_name: format_duration_hours(hours)
                for user_name, hours in hours_by_user.items()
            },
            "by_task": dict(sorted(by_task_total.items())),
            "hours_by_task": hours_by_task,
            "duration_text_by_task": {
                task_name: format_duration_hours(hours)
                for task_name, hours in hours_by_task.items()
            },
            "by_embodiedment": dict(sorted(by_embodiedment_total.items())),
            "hours_by_embodiedment": hours_by_embodiedment,
            "duration_text_by_embodiedment": {
                embodiedment: format_duration_hours(hours)
                for embodiedment, hours in hours_by_embodiedment.items()
            },
        },
        "users": users,
        "episodes": episodes,
    }


def build_summary_with_filters(
    records: list[EpisodeRecord],
    base_path: Path,
    filters: FilterOptions,
    cache_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filtered_records = filter_records(records, filters)
    summary = build_summary(filtered_records, base_path)
    total_pages = max(
        (len(summary["episodes"]) + filters.page_size - 1)
        // filters.page_size,
        1,
    )
    current_page = min(max(filters.page, 1), total_pages)
    start_index = (current_page - 1) * filters.page_size
    end_index = start_index + filters.page_size
    summary["episodes"] = summary["episodes"][start_index:end_index]
    summary["filters"] = {
        "user_name": filters.user_name,
        "task_name": filters.task_name,
        "embodiedment": filters.embodiedment,
        "date_prefix": filters.date_prefix,
        "data_root": str(base_path),
        "refresh": filters.refresh,
        "page": current_page,
        "page_size": filters.page_size,
    }
    summary["pagination"] = {
        "page": current_page,
        "page_size": filters.page_size,
        "total_items": len(filtered_records),
        "total_pages": total_pages,
        "has_prev": current_page > 1,
        "has_next": current_page < total_pages,
        "start_index": start_index + 1 if filtered_records else 0,
        "end_index": min(end_index, len(filtered_records)),
    }
    if cache_info is not None:
        summary["cache"] = cache_info
    return summary


def build_summary_with_filters_for_paths(
    records: list[EpisodeRecord],
    base_paths: list[Path],
    filters: FilterOptions,
    cache_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_summary_with_filters(
        records, base_paths[0], filters, cache_info
    )
    joined_base_paths = format_base_paths(base_paths)
    summary["base_path"] = joined_base_paths
    summary["filters"]["data_root"] = joined_base_paths
    return summary


def load_summary_from_request_args(args: Any) -> dict[str, Any]:
    ensure_auto_refresh_started()
    filters = parse_filters(args)
    base_paths = resolve_base_paths(filters)
    records, cache_info = get_cached_episode_records_for_paths(
        base_paths, refresh=filters.refresh
    )
    return build_summary_with_filters_for_paths(
        records, base_paths, filters, cache_info
    )


def build_loading_summary(
    base_path: Path, filters: FilterOptions, task_id: str
) -> dict[str, Any]:
    return {
        "base_path": str(base_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes": 0,
        "total_hours": 0.0,
        "total_duration_text": format_duration_hours(0.0),
        "filters": {
            "user_name": filters.user_name,
            "task_name": filters.task_name,
            "embodiedment": filters.embodiedment,
            "date_prefix": filters.date_prefix,
            "data_root": str(base_path),
            "refresh": filters.refresh,
            "page": max(filters.page, 1),
            "page_size": filters.page_size,
        },
        "cache": {
            "cache_hit": False,
            "cache_source": "loading",
            "cache_created_at": "",
            "cache_entry_count": 0,
        },
        "days": [],
        "totals": {
            "by_day": {},
            "hours_by_day": {},
            "duration_text_by_day": {},
            "by_user": {},
            "hours_by_user": {},
            "duration_text_by_user": {},
            "by_task": {},
            "hours_by_task": {},
            "duration_text_by_task": {},
            "by_embodiedment": {},
            "hours_by_embodiedment": {},
            "duration_text_by_embodiedment": {},
        },
        "users": [],
        "episodes": [],
        "pagination": {
            "page": max(filters.page, 1),
            "page_size": filters.page_size,
            "total_items": 0,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False,
            "start_index": 0,
            "end_index": 0,
        },
        "loading": {
            "is_loading": True,
            "scan_task_id": task_id,
            "message": "No cache found yet. Starting initial scan...",
        },
    }


def build_loading_summary_for_paths(
    base_paths: list[Path], filters: FilterOptions, task_ids: list[str]
) -> dict[str, Any]:
    summary = build_loading_summary(base_paths[0], filters, task_ids[0])
    joined_base_paths = format_base_paths(base_paths)
    summary["base_path"] = joined_base_paths
    summary["filters"]["data_root"] = joined_base_paths
    summary["loading"]["scan_task_id"] = ",".join(task_ids)
    if len(base_paths) > 1:
        summary["loading"]["message"] = (
            "No cache found yet. Starting initial scans for multiple data roots..."  # noqa: E501
        )
    return summary


def maybe_build_initial_loading_summary(args: Any) -> dict[str, Any] | None:
    ensure_auto_refresh_started()
    filters = parse_filters(args)
    base_paths = resolve_base_paths(filters)
    if filters.refresh or all(
        has_cached_episode_records(base_path) for base_path in base_paths
    ):
        return None

    task_ids = [
        start_scan_task(base_path)
        for base_path in base_paths
        if not has_cached_episode_records(base_path)
    ]
    if not task_ids:
        return None
    return build_loading_summary_for_paths(base_paths, filters, task_ids)


@app.before_request
def require_setup() -> Any:
    if ENV_CONFIGURED:
        return None
    if request.endpoint in {"setup_page", "save_setup_api"}:
        return None
    return redirect("/setup")


@app.get("/setup")
def setup_page():
    env_content = (
        ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    )
    fields = parse_env_file_for_setup(env_content)
    return render_template("setup.html", fields=fields, env_path=str(ENV_PATH))


@app.post("/api/setup")
def save_setup_api():
    data = request.get_json(silent=True) or {}
    new_values = data.get("fields")
    if not isinstance(new_values, dict):
        abort(400, description="fields must be a JSON object")

    template_content = (
        ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        if ENV_EXAMPLE_PATH.exists()
        else ENV_PATH.read_text(encoding="utf-8")
        if ENV_PATH.exists()
        else ""
    )
    env_content = rebuild_env_content(
        template_content, cast(dict[str, str], new_values)
    )
    ENV_PATH.write_text(env_content, encoding="utf-8")
    load_dotenv_file(ENV_PATH)
    reinitialize_from_env()
    return jsonify({"status": "ok"})


@app.get("/")
def index():
    loading_summary = maybe_build_initial_loading_summary(request.args)
    if loading_summary is not None:
        return render_template("index.html", summary=loading_summary)
    summary = load_summary_from_request_args(request.args)
    return render_template("index.html", summary=summary)


@app.get("/api/summary")
def api_summary():
    summary = load_summary_from_request_args(request.args)
    return jsonify(summary)


@app.post("/api/scan-tasks")
def create_scan_task_api():
    data = request.get_json(silent=True) or {}
    data_root = str(data.get("data_root", "")).strip()
    refresh_mode = str(data.get("refresh_mode", "full")).strip() or "full"
    date_prefixes = normalize_date_prefixes(str(data.get("date_prefix", "")))
    if not data_root:
        data_root = str(BASE_DATA_PATH)

    base_paths = [Path(item) for item in parse_filter_items(data_root)] or [
        BASE_DATA_PATH
    ]
    if refresh_mode not in {"full", "partial"}:
        abort(400, description="refresh_mode must be full or partial")
    if refresh_mode == "partial" and not date_prefixes:
        abort(400, description="date_prefix is required for partial refresh")

    tasks = []
    for base_path in base_paths:
        task_id = start_scan_task(
            base_path,
            refresh_mode=refresh_mode,
            target_date_prefixes=date_prefixes,
        )
        task = get_scan_task(task_id)
        if task is not None:
            tasks.append(task)
    return jsonify({"tasks": tasks}), 202


@app.get("/api/scan-tasks/<task_id>")
def get_scan_task_api(task_id: str):
    task = get_scan_task(task_id)
    if task is None:
        abort(404, description="Scan task not found")
    return jsonify(task)


@app.post("/api/scan-tasks/<task_id>/cancel")
def cancel_scan_task_api(task_id: str):
    task = get_scan_task(task_id)
    if task is None:
        abort(404, description="Scan task not found")

    update_scan_task(
        task_id,
        cancel_requested=True,
        message="Cancelling scan task...",
    )
    return jsonify(get_scan_task(task_id))


@app.post("/api/submit-jobs/prepare")
def prepare_submit_job_api():
    data = request.get_json(silent=True) or {}
    source = str(data.get("source", "")).strip()
    data_root = str(data.get("data_root", "")).strip()
    filters_payload = (
        data.get("filters", {})
        if isinstance(data.get("filters"), dict)
        else {}
    )

    if source not in {"check", "pack"}:
        abort(400, description="source must be check or pack")

    base_paths = [Path(item) for item in parse_filter_items(data_root)] or [
        BASE_DATA_PATH
    ]
    base_path = base_paths[0]
    filters = FilterOptions(
        user_name=str(filters_payload.get("user_name", "")).strip(),
        task_name=str(filters_payload.get("task_name", "")).strip(),
        embodiedment=str(filters_payload.get("embodiedment", "")).strip(),
        date_prefix=str(filters_payload.get("date_prefix", "")).strip(),
        data_root=format_base_paths(base_paths),
    )
    if not (
        filters.user_name
        or filters.task_name
        or filters.embodiedment
        or filters.date_prefix
    ):
        abort(400, description="Please search first before submitting jobs")

    try:
        payload = create_submit_config(source, base_path, filters)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        abort(400, description=str(exc))

    return jsonify(payload), 201


@app.post("/api/submit-jobs/<config_id>/submit")
def submit_job_api(config_id: str):
    data = request.get_json(silent=True) or {}
    config = data.get("config")
    if not isinstance(config, dict):
        abort(400, description="config must be a JSON object")
    config_data = cast(dict[str, Any], config)

    try:
        result = start_submit_task(config_id, config_data)
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except RuntimeError as exc:
        abort(400, description=str(exc))

    return jsonify(result), 202


@app.get("/api/submit-jobs/tasks/<task_id>")
def get_submit_task_api(task_id: str):
    task = get_submit_task(task_id)
    if task is None:
        abort(404, description="Submit task not found")
    return jsonify(task)


@app.get("/api/remote-upload/config")
def get_remote_upload_config_api():
    filters = FilterOptions(
        user_name=normalize_filter(request.args.get("user_name")),
        task_name=normalize_filter(request.args.get("task_name")),
        embodiedment=normalize_filter(request.args.get("embodiedment")),
        date_prefix=normalize_filter(request.args.get("date_prefix")),
        data_root=normalize_filter(request.args.get("data_root")),
    )
    try:
        config_text = build_remote_upload_default_config(filters)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        abort(404, description=str(exc))

    return jsonify(
        {
            "config_path": str(REMOTE_UPLOAD_CONFIG_PATH),
            "config_text": config_text,
        }
    )


@app.post("/api/remote-upload/tasks")
def create_remote_upload_task_api():
    data = request.get_json(silent=True) or {}
    config_text = str(data.get("config_text", "")).strip()
    if not config_text:
        abort(400, description="config_text is required")

    try:
        task = start_remote_upload_task(config_text)
    except json.JSONDecodeError as exc:
        abort(400, description=f"Invalid remote upload config: {exc}")

    return jsonify(task), 202


@app.get("/api/remote-upload/tasks/<task_id>")
def get_remote_upload_task_api(task_id: str):
    task = get_remote_upload_task(task_id)
    if task is None:
        abort(404, description="Remote upload task not found")
    return jsonify(task)


@app.post("/api/remote-upload/tasks/<task_id>/cancel")
def cancel_remote_upload_task_api(task_id: str):
    task = get_remote_upload_task(task_id)
    if task is None:
        abort(404, description="Remote upload task not found")

    update_remote_upload_task(
        task_id,
        cancel_requested=True,
        message=(
            "Cancel requested. Waiting for the remote upload process "
            "to stop..."
        ),
    )
    return jsonify(get_remote_upload_task(task_id))


@app.get("/api/download")
def download_episode_zip():
    episode_path_raw = request.args.get("episode_path", "").strip()
    if not episode_path_raw:
        abort(400, description="Missing episode_path")

    episode_path = Path(episode_path_raw)
    if not episode_path.exists() or not episode_path.is_dir():
        abort(404, description="Episode path not found")

    zip_bytes = create_episode_zip_bytes(episode_path)
    download_name = f"{episode_path.name}.zip"
    return send_file(
        zip_bytes,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@app.get("/api/download-status")
def download_episode_status():
    episode_path_raw = request.args.get("episode_path", "").strip()
    if not episode_path_raw:
        abort(400, description="Missing episode_path")

    episode_path = Path(episode_path_raw)
    if not episode_path.exists() or not episode_path.is_dir():
        abort(404, description="Episode path not found")

    file_count = count_episode_files(episode_path)
    return jsonify(
        {
            "episode_path": str(episode_path),
            "file_count": file_count,
            "status": "preparing",
            "message": (
                "Packaging in progress. "
                f"Detected {file_count} files. "
                "Please wait..."
            ),
        }
    )


if __name__ == "__main__":
    app.run(
        host=get_env_value("HOST", "0.0.0.0"),
        port=int(get_env_value("PORT", "8000")),
        debug=get_env_bool("FLASK_DEBUG", False),
    )
