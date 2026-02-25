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

import argparse
import logging
import os
import random
import sys

import numpy as np
from robo_orchard_core.utils.logging import LoggerManager

sys.path.append("projects/sem/common/configs")
from config_interna1_dataset import build_lmdb_transforms
from config_sem_common import config

from robo_orchard_lab.dataset.interna1.packer.arrow_packer.viz_arrow_interna1 import (  # noqa: E501
    export_video,
)

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)


def export_lmdb_video(
    dataset, episode_index, video_path, num_workers=0, prefetch_factor=None
):
    # Export video to viz
    fps = 30
    end = dataset.cumsum_steps[episode_index]
    if episode_index != 0:
        begin = dataset.cumsum_steps[episode_index - 1]
    else:
        begin = 0

    export_video(
        dataset,
        begin,
        end,
        video_path,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        fps=fps,
    )


def build_lmdb_dataset(config):
    """Builds the InternA1 LMDB dataset based on the provided configuration.

    Args:
        config (dict): A dictionary containing configuration parameters such as
            'data_path', 'urdf', 'robot_type', 'cam_names', and 'task_names'.

    Returns:
        InternA1LmdbDataset: An instance of the InternA1LmdbDataset initialized
            with the specified configuration.
    """
    from robo_orchard_lab.dataset.interna1 import InternA1LmdbDataset

    train_transforms = build_lmdb_transforms(
        config=config,
        mode="training",
        urdf=config["urdf"],
        robot_type=config["robot_type"],
    )
    dataset = InternA1LmdbDataset(
        paths=config["data_path"],
        lazy_init=False,
        transforms=train_transforms,
        dataset_name="test",
        cam_names=config["cam_names"],
        task_names=config["task_names"],
        load_extrinsic=True,
        load_calibration=False,
        load_depth=True,
        load_ee_state=True,
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MCAP files directly to LMDB format."
    )
    parser.add_argument(
        "--lmdb_dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    logger.info(f"LMDB dataset path: {args.lmdb_dataset_path}")

    output_path = args.output_path

    if "ARX_Lift2" in args.lmdb_dataset_path:
        urdf = "/horizon-bucket/robot_lab2/datasets/InternData-A1/urdf/ARX_Lift2_fix/lift.urdf"  # noqa: E501
        link_topics = [
            "L_arm_link11",
            "L_arm_link12",
            "L_arm_link13",
            "L_arm_link14",
            "L_arm_link15",
            "L_arm_link16",
            "L_arm_link11",
            "R_arm_link21",
            "R_arm_link22",
            "R_arm_link23",
            "R_arm_link24",
            "R_arm_link25",
            "R_arm_link26",
            # "left_arm_left_catch",
            # "right_arm_left_catch",
            "left_arm_tcp_link",
            "right_arm_tcp_link",
        ]
        robot_type = "ARX Lift-2"
    elif "AgileX_Split_Aloha" in args.lmdb_dataset_path:
        urdf = "/horizon-bucket/robot_lab2/datasets/InternData-A1/urdf/AgileX_Split_Aloha_piper100/split_aloha_mid_360_with_piper.urdf"  # noqa: E501
        link_topics = [
            "left/link1",
            "left/link2",
            "left/link3",
            "left/link4",
            "left/link5",
            "left/link6",
            "right/link1",
            "right/link2",
            "right/link3",
            "right/link4",
            "right/link5",
            "right/link6",
            "left/link7",
            "right/link7",
        ]
        robot_type = "AgileX Split Aloha"
    elif "Genie" in args.lmdb_dataset_path:
        urdf = "/horizon-bucket/robot_lab2/datasets/InternData-A1/urdf/G1_120s/G1_120s.urdf"  # noqa: E501
        link_topics = [
            # Left arm joints
            "arm_l_link1",
            "arm_l_link2",
            "arm_l_link3",
            "arm_l_link4",
            "arm_l_link5",
            "arm_l_link6",
            "arm_l_end_link",
            # Left gripper outer joint
            "gripper_l_center_link",
            # Right arm joints
            "arm_r_link1",
            "arm_r_link2",
            "arm_r_link3",
            "arm_r_link4",
            "arm_r_link5",
            "arm_r_link6",
            "arm_r_end_link",
            # Right gripper outer joint
            "gripper_r_center_link",
        ]
        robot_type = "Genie-1"

    config.update(
        data_path=args.lmdb_dataset_path,
        cam_names=["head", "hand_left", "hand_right"],
        urdf=urdf,
        T_base2world=np.eye(4),
        task_names=None,
        with_depth=False,
        robot_type=robot_type,
    )

    # Build dataset
    dataset = build_lmdb_dataset(config)
    total_episodes = len(dataset.episode_indices)
    episodes_list = list(range(total_episodes))
    logger.info(f"Total episodes packed in dataset: {total_episodes}")
    logger.info(f"Total frames packed in dataset: {len(dataset)}")

    viz_num = 5
    if total_episodes <= viz_num:
        episode_index_list = list(range(total_episodes))
    else:
        episode_index_list = random.sample(range(total_episodes), viz_num)

    if len(episode_index_list) != 0:
        os.makedirs(output_path, exist_ok=True)
        episode_index_list.sort()
        logger.info(
            f"Randomly selected {len(episode_index_list)} episodes for visualization."  # noqa: E501
        )
        logger.info(
            f"Selected episode indices for visualization: {episode_index_list}"
        )
    else:
        logger.warning("No episodes selected for visualization.")

    # Initialize HTML content
    html_content = """<html>
<head><title>HViz Links for Selected Episodes</title></head>
<body>
<h1>HViz Links for Selected Episodes</h1>
<p>This page contains links to visualize the original and dataset MCAP files in HViz. Please use Chrome.</p>
<table border="1" cellpadding="5" cellspacing="0">
    <tr>
        <th>Episode Index</th>
        <th>Video (After Transform)</th>
    </tr>
"""  # noqa: E501
    html_path = os.path.join(output_path, "viz_all_episodes_hviz_links.html")
    cluster_dir = "viz_results"

    lmdb_dataset_dir = os.path.basename(args.lmdb_dataset_path)

    for episode_index in episode_index_list:
        logger.info(f"Processing episode index: {episode_index}")

        # Export video to viz data after transform
        save_name = f"viz_{lmdb_dataset_dir}_episodeidx_{episode_index}"
        video_name = f"{save_name}.mp4"
        video_path = os.path.join(output_path, video_name)

        # Set Transform
        export_lmdb_video(
            dataset=dataset,
            episode_index=episode_index,
            video_path=video_path,
            num_workers=16,
            prefetch_factor=4,
        )

        if os.path.exists("/job_data"):
            # Append row to HTML content
            html_content += f"""
    <tr>
        <td>{episode_index}</td>
        <td><a href="{video_name}" target="_blank">View Video</a></td>
    </tr>
"""  # noqa: E501

        # if running on cluster, cp video to working root
        if os.path.exists("/job_data"):
            working_root = os.path.join("/job_data", cluster_dir)
            os.makedirs(working_root, exist_ok=True)
            os.system(f"cp {video_path} {working_root}")
            logger.info(f"Copied viz video to working root: {working_root}")

            os.system(f"cp {html_path} {working_root}")
            logger.info(
                f"Copied combined hviz links html to working root: {working_root}"  # noqa: E501
            )

    html_content += """
</table>
</body>
</html>"""
    html_path = os.path.join(output_path, "viz_all_episodes_hviz_links.html")
    with open(html_path, "w") as f:
        f.write(html_content)
        logger.info(f"Saved combined hviz links to {html_path}")

    # Finalize and save HTML
    if os.path.exists("/job_data"):
        os.system(f"cp {html_path} {working_root}")
        logger.info(
            f"Copied combined hviz links html to working root: {working_root}"
        )
