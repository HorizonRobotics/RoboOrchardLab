# ruff: noqa: E501
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
import json
import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_DOCKER_IMAGE = "docker.hobot.cc/dlp/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch241-20250506"
DEFAULT_URDF_PATH = "/horizon-bucket/robot_lab/users/xuewu.lin/urdf/piper_x_description_dualarm.urdf"
DEFAULT_WORKSPACE_FOLDER = "workspace_to_submit"


def split_json_records(
    input_json: str | Path,
    chunk_dir: str | Path,
    chunk_size: int,
) -> list[Path]:
    input_json = Path(input_json)
    chunk_dir = Path(chunk_dir)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old_file in chunk_dir.glob("chunk_*.json"):
        old_file.unlink()

    chunk_paths = []
    for chunk_index, start_idx in enumerate(range(0, len(data), chunk_size)):
        chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.json"
        with chunk_path.open("w", encoding="utf-8") as f:
            json.dump(
                data[start_idx : start_idx + chunk_size],
                f,
                ensure_ascii=False,
                indent=2,
            )
        chunk_paths.append(chunk_path)

    return chunk_paths


def build_submit_json(
    input_path: str,
    output_path: str,
    task_name: str,
    urdf_path: str = DEFAULT_URDF_PATH,
    chunk_idx: str | None = None,
    force_overwrite: bool = True,
    workspace_folder: str = DEFAULT_WORKSPACE_FOLDER,
    job_name_prefix: str | None = None,
) -> dict:
    input_stem = input_path.split("/")[-1].split(".")[0]
    job_name = f"pack_{input_stem}"
    if job_name_prefix is not None:
        job_name = f"pack_{job_name_prefix}_{input_stem}"
    remote_output_path = output_path

    to_upload = [
        "robo_orchard_lab",
        f"{input_path}",
    ]

    input_path = os.path.basename(input_path)
    if chunk_idx is not None:
        remote_output_path = f"{output_path}_{chunk_idx}"
        workspace_folder = f"{workspace_folder}/chunk_{chunk_idx}"

    run_args = [
        "python3",
        "robo_orchard_lab/dataset/horizon_manipulation/packer/sim_mcap_arrow_packer.py",
        "--input_path",
        input_path,
        "--output_path",
        remote_output_path,
        "--task_name",
        task_name,
        "--urdf_path",
        urdf_path,
    ]
    if force_overwrite:
        run_args.append("--force_overwrite")

    run_cmd = " ".join(shlex.quote(arg) for arg in run_args)

    return {
        "job_name": job_name,
        "docker_image": DEFAULT_DOCKER_IMAGE,
        "input_bucket": ["robot_lab", "robot_lab2"],
        "output_bucket": ["robot_lab", "robot_lab2"],
        "num_workers": 1,
        "cpu_per_worker": 4,
        "cpu_mem_ratio": 16,
        "wall_time": 7200,
        "workspace_folder": workspace_folder,
        "cmd": [
            "pip3 install mcap-protobuf-support datasets foxglove_schemas_protobuf sqlalchemy sortedcontainers duckdb-engine duckdb pyzstd foxglove-sdk foxglove-schemas-protobuf",
            "export https_proxy='10.9.0.31:8838'",
            "export http_proxy='10.9.0.31:8838'",
            "export no_proxy='localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,horizon.ai,horizon.cc,hobot.cc,horizon.auto,hogpu.cc,gua.com,guasemi.com'",
            "pip3 install robo_orchard_core@git+https://github.com/HorizonRobotics/robo_orchard_core.git@7cfd9e8758cf79c0265730b315e1c905f4466058",
            "pip3 install robo_orchard_schemas",
            "export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH",
            "export HF_DATASETS_CACHE=./",
            f"echo '[EXECUTED] {run_cmd}'",
            run_cmd,
        ],
        "to_upload": to_upload,
        "job_password": "1227",
        "job_type": "packing",
        "queue_name": "project-cpu-horizon-labs-bcloud",
        "project_id": "horizon-labs",
    }


def build_chunk_submit_jsons(
    input_path: str,
    output_path: str,
    task_name: str,
    chunk_size: int,
    chunk_json_dir: str | Path,
    urdf_path: str = DEFAULT_URDF_PATH,
) -> list[dict]:
    input_stem = Path(input_path).stem
    chunk_paths = split_json_records(
        input_json=input_path,
        chunk_dir=chunk_json_dir,
        chunk_size=chunk_size,
    )
    submit_jsons = []
    for chunk_path in chunk_paths:
        chunk_idx = chunk_path.stem.removeprefix("chunk_")
        submit_jsons.append(
            build_submit_json(
                input_path=str(chunk_path),
                output_path=output_path,
                task_name=task_name,
                urdf_path=urdf_path,
                chunk_idx=chunk_idx,
                workspace_folder=f"{DEFAULT_WORKSPACE_FOLDER}/{input_stem}",
                job_name_prefix=input_stem,
            )
        )
    return submit_jsons


if __name__ == "__main__":
    tasks = [
        # input_json_path, output_path, task_name
        (
            # "/home/users/mengao.zhao-labs/docker_env/orchard_sim_dev/robo_orchard_sim/instructions/pick_category0522_005/selected_records_instructions_seen_only.json",
            # "/horizon-bucket/robot_lab/users/mengao.zhao-labs/dataset/anymove_arrow/pick_category0522_005_seen_only",
            # "anymove_pick_category"
            "/home/users/mengao.zhao-labs/docker_env/orchard_sim_dev/robo_orchard_sim/instructions/pick_attribute_0522_2124/selected_records_instructions_seen_only.json",
            "/horizon-bucket/robot_lab/users/mengao.zhao-labs/dataset/anymove_arrow/pick_attribute_0522_2124_seen_only",
            "anymove_pick_attribute",
        )
    ]

    chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
    chunk_json_root = Path(
        os.environ.get("CHUNK_JSON_DIR", "tmp/pack_submit_chunks")
    )
    urdf_path = os.environ.get("URDF_PATH", DEFAULT_URDF_PATH)

    save_dir = Path("submit_jsons")
    save_dir.mkdir(exist_ok=True)

    for input_path, output_path, task_name in tasks:
        task_chunk_dir = chunk_json_root / task_name / Path(input_path).stem
        submit_jsons = build_chunk_submit_jsons(
            input_path=input_path,
            output_path=output_path,
            task_name=task_name,
            chunk_size=chunk_size,
            chunk_json_dir=task_chunk_dir,
            urdf_path=urdf_path,
        )
        if not submit_jsons:
            raise RuntimeError(
                f"No chunk files were created from {input_path}"
            )

        for submit_json in submit_jsons:
            json_path = save_dir / f"{submit_json['job_name']}.json"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(submit_json, f, indent=4)

            print(f"saved: {json_path}")

            command = [
                "RoboOrchardJob-AIDISubmit",
                "submit_from_config",
                "--config",
                str(json_path),
            ]
            subprocess.run(command, check=True)
