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

import glob
import json
import os
import subprocess


def update_submit_json(
    config_file="submit_pack.json",
    script_file="run_pack.sh",
    dataset_name=None,
    start_idx=None,
    end_idx=None,
):
    """Updates the submission configuration JSON file with job details.

    Args:
        config_file (str, optional): Path to the output JSON config file.
            Defaults to "submit_pack.json".
        script_file (str, optional): Path to the script file to be executed.
            Defaults to "run_pack.sh".
        dataset_name (str, optional): Name of the dataset for the job name.
            Defaults to None.
        start_idx (int, optional): Start index of episodes. Defaults to None.
        end_idx (int, optional): End index of episodes. Defaults to None.
    """
    config = {
        "job_name": "",
        "docker_image": "docker.hobot.cc/dlp/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch241-20250506",  # noqa: E501
        "input_bucket": ["robot_lab", "robot_lab2"],
        "output_bucket": ["robot_lab", "robot_lab2"],
        "num_workers": 1,
        "gpu_per_worker": 0,
        "cpu_per_worker": 4,
        "cpu_mem_ratio": 8,
        "wall_time": 7200,
        "workspace_folder": "workspace_to_submit",
        "clear_workspace": True,
        "python_launcher": "python3",
        "cmd": [
            "pip3 install mcap-protobuf-support foxglove_schemas_protobuf robo_orchard_schemas",  # noqa: E501
            "pip3 install imageio[ffmpeg]",
            "pip3 install robo_orchard_core==0.3.0.dev20250919171359 --index-url https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple",  # noqa: E501
            "pip3 install --upgrade scipy --index-url https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple",  # noqa: E501
            "pip3 install datasets==4.0.0 sqlalchemy==2.0.38 duckdb==1.3.2 duckdb-engine sortedcontainers lerobot --index-url https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple",  # noqa: E501
            "export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH",
            "cat run_pack.sh",
            "pip3 list",
            "export HF_DATASETS_CACHE=./",
            "ulimit -n 1024000",
            "bash run_pack.sh",
        ],
        "project_id": "Robot-lab",
        # "queue_name": "share-idc-newage-cpu",
        "queue_name": "share-cpu-bcloud",
        "job_password": "aidi_job_passwd",
        "priority": 5,
        "to_upload": [],
    }  # noqa: E501

    config["job_name"] = (
        f"lmdb_dataset_pack_{dataset_name}_startid_{start_idx}_endid_{end_idx}_InternData_A1_data"  # noqa: E501
    )
    config["to_upload"] = [
        "robo_orchard_lab",
        "projects/sem/common/configs/config_interna1_dataset.py",
        "projects/sem/common/configs/config_sem_common.py",
        "robo_orchard_lab/dataset/interna1/packer/lmdb_packer/lmdb_pack_InternA1.py",
        "robo_orchard_lab/dataset/interna1/packer/lmdb_packer/viz_lmdb_InternA1.py",
        script_file,
    ]

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)


def update_script_shell(
    script_file="run_pack.sh",
    lmdb_output_path=None,
    task_dir=None,
    start_idx=None,
    end_idx=None,
):
    """Updates the shell script for packing the dataset.

    Args:
        script_file (str, optional): Path to the shell script file.
            Defaults to "run_pack.sh".
        lmdb_output_path (str, optional): Output path for the LMDB dataset.
            Defaults to None.
        task_dir (str, optional): Input directory containing task data.
            Defaults to None.
        start_idx (int, optional): Start index of episodes. Defaults to None.
        end_idx (int, optional): End index of episodes. Defaults to None.
    """
    script_content = f"""#!/bin/bash
        # 这个脚本用于处理 Env 为 {dataset_name} 的数据
        python3 lmdb_pack_InternA1.py \
        --input_path {task_dir} \
        --output_path {lmdb_output_path} \
        --start_idx {start_idx} \
        --end_idx {end_idx} \

        python3 viz_lmdb_InternA1.py \
        --lmdb_dataset_path {lmdb_output_path} \
        --output_path {os.path.join(lmdb_output_path, "viz")}

        """  # noqa: E501

    # 构建文件名

    with open(script_file, "w") as f:
        f.write(script_content)


if __name__ == "__main__":
    import json
    from glob import glob

    input_dirs = "/horizon-bucket/robot_lab2/datasets/InternData-A1/sim_updated_lerobotv30/"  # noqa: E501

    lmdb_output_root = "/horizon-bucket/robot_lab2/users/yun01.du/data/InternData_A1_lmdb_0125"  # noqa: E501

    task_dirs = sorted(glob(f"{input_dirs}*"))
    print(f"total task dir is {len(task_dirs)}")

    total_task_dir = []
    script_output_dir = "./"

    for task_idx, task_dir in enumerate(task_dirs):
        try:
            info_json = os.path.join(task_dir, "meta/info.json")
            with open(info_json, "r") as f:
                info = json.load(f)
                print(
                    f"{task_idx} robot: {info['robot_type']}, "
                    f"total episodes: {info['total_episodes']}, "
                    f"total_frames: {info.get('total_frames')}"
                )
                task_info = {
                    "task_dir": task_dir,
                    "robot_type": info["robot_type"],
                    "total_episodes": info["total_episodes"],
                    "total_frames": info["total_frames"],
                }
                total_task_dir.append(task_info)
        except Exception:
            subtask_dirs = sorted(glob(f"{task_dir}/*"))
            try:
                for subtask_idx, subtask_dir in enumerate(subtask_dirs):
                    info_json = os.path.join(subtask_dir, "meta/info.json")
                    with open(info_json, "r") as f:
                        info = json.load(f)
                        print(
                            f"{task_idx} {subtask_idx} robot: "
                            f"{info['robot_type']}, "
                            f"total episodes: {info['total_episodes']}, "
                            f"total_frames: {info.get('total_frames')}"
                        )

                        task_info = {
                            "task_dir": subtask_dir,
                            "robot_type": info["robot_type"],
                            "total_episodes": info["total_episodes"],
                            "total_frames": info["total_frames"],
                        }
                        total_task_dir.append(task_info)
            except Exception as e:
                print(f"{task_idx}, exception is {e}")

    print(f"total task dir is {len(total_task_dir)}")

    for task_idx, task_info in enumerate(total_task_dir):
        task_dir = task_info["task_dir"]
        robot_type = task_info["robot_type"]

        if (
            robot_type != "ARX Lift-2" and robot_type != "AgileX Split Aloha"
            # and robot_type != "Genie-1"
        ):
            # if robot_type != "ARX Lift-2":
            # if robot_type != "AgileX Split Aloha":
            # if robot_type != "Genie-1":
            continue

        robot_type = robot_type.replace(" ", "_").replace("-", "")

        total_episodes = task_info["total_episodes"]
        total_frames = task_info["total_frames"]

        each_size = 500
        for i in range(0, total_episodes, each_size):
            start_idx = i
            end_idx = min(i + each_size, total_episodes)

            processing_info = (
                f"Processing task_dir: {task_dir}, "
                f"robot_type: {robot_type}, "
                f"total_episodes: {total_episodes}, "
                f"total_frames: {total_frames}"
            )
            print(processing_info)

            dataset_name = (
                robot_type
                + "_"
                + task_dir.replace(input_dirs, "").replace("/", "_")
            )
            lmdb_output_path = os.path.join(
                lmdb_output_root,
                f"{robot_type}/lmdb_dataset_{dataset_name}_ep_startid_{start_idx}_ep_endid_{end_idx}",
            )
            print(lmdb_output_path)

            # 构建 pack.sh 的内容
            print(f"{task_idx} {dataset_name} {robot_type} start packing...")

            # update run_pack.sh
            script_file = os.path.join(script_output_dir, "run_pack.sh")
            update_script_shell(
                script_file=script_file,
                lmdb_output_path=lmdb_output_path,
                task_dir=task_dir,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            # update submit_pack.json
            config_file = os.path.join(script_output_dir, "submit_pack.json")
            update_submit_json(
                config_file=config_file,
                script_file=script_file,
                start_idx=start_idx,
                end_idx=end_idx,
                dataset_name=dataset_name,
            )

            submission_command = [
                "RoboOrchardJob-AIDISubmit",
                "submit_from_config",
                "--config",
                config_file,
            ]
            cmd_str = " ".join(submission_command)
            print(f"command: {cmd_str}")
            result = subprocess.run(
                submission_command,
                check=True,
                capture_output=True,
                text=True,
            )
            print("\n命令执行成功！")
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            # 清理配置以防止污染下一次循环
            delete_commands = [
                f"rm {script_file}",
                f"rm {config_file}",
                "rm aidi_job_submit.json",
            ]
            for cmd in delete_commands:
                subprocess.run(cmd, shell=True)

        #     break

        # break
