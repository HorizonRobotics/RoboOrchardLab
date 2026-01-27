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

import imageio
import numpy as np
import torch
from robo_orchard_core.utils.logging import LoggerManager
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("projects/sem/common/configs")
from config_interna1_dataset import build_arrow_transforms
from config_sem_common import config

from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.camera import (  # noqa: E501
    McapBatchFromBatchCameraDataEncodedConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.joint_state import (  # noqa: E501
    McapBatchFromBatchJointStateConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.tf import (
    McapBatchFromBatchFrameTransformConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.writer import (
    Dataset2Mcap,
    McapBatchEncoderConfig,
)
from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robotwin.transforms import EpisodeSamplerConfig

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)


def export_video(
    dataset,
    begin,
    end,
    video_path,
    num_workers=0,
    prefetch_factor=None,
    fps=30,
):
    viz_dataset = VizDataset(dataset, begin, end)
    loader = DataLoader(
        viz_dataset,
        batch_size=None,  # Disable automatic batching to get single items
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )

    print("\n" + "*" * 16)
    print(
        f"Exporting video to {video_path} with {len(viz_dataset)} frames \
        using {num_workers} workers..."
    )

    # Use imageio.get_writer for streaming write to save memory
    with imageio.get_writer(
        video_path,
        fps=fps,
        macro_block_size=None,
        output_params=["-preset", "ultrafast", "-crf", "23"],
    ) as writer:
        for _, vis_imgs in tqdm(enumerate(loader), total=len(viz_dataset)):
            writer.append_data(vis_imgs.numpy()[..., ::-1])
    print(f"Export video to {video_path} with {len(viz_dataset)} frames Done")
    print("*" * 16 + "\n")


def export_arrow_video(
    dataset, episode_index, video_path, num_workers=0, prefetch_factor=None
):
    # Export video to viz
    fps = 30
    episode_info = dataset.get_meta(Episode, episode_index)
    begin = episode_info.dataset_begin_index
    end = begin + episode_info.frame_num
    export_video(
        dataset=dataset,
        begin=begin,
        end=end,
        video_path=video_path,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        fps=fps,
    )


def export_mcap(dataset, episode_index, target_path, link_topics, cam_names):
    """Export the specified episode to an MCAP file."""

    dataset2mcap_cfg: dict[str, McapBatchEncoderConfig] = {
        "joints": McapBatchFromBatchJointStateConfig(
            target_topic="/observation/robot_state/joints"
        ),
    }
    dataset2mcap_cfg["actions"] = McapBatchFromBatchJointStateConfig(
        target_topic="/action/robot_state/joints"
    )

    dataset2mcap_cfg["left_ee"] = McapBatchFromBatchFrameTransformConfig(
        target_topic="/observation/robot_state/left_ee"
    )
    dataset2mcap_cfg["right_ee"] = McapBatchFromBatchFrameTransformConfig(
        target_topic="/observation/robot_state/right_ee"
    )
    dataset2mcap_cfg["left_tcp"] = McapBatchFromBatchFrameTransformConfig(
        target_topic="/observation/robot_state/left_tcp"
    )
    dataset2mcap_cfg["right_tcp"] = McapBatchFromBatchFrameTransformConfig(
        target_topic="/observation/robot_state/right_tcp"
    )

    for link_topic in link_topics:
        dataset2mcap_cfg[link_topic] = McapBatchFromBatchFrameTransformConfig(  # noqa: E501
            target_topic=f"/observation/robot_state/{link_topic}"
        )

    topic_map = {"head": "middle", "hand_left": "left", "hand_right": "right"}
    for camera_name in cam_names:
        dataset2mcap_cfg[camera_name] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                calib_topic=f"/observation/cameras/{topic_map[camera_name]}/calib",
                image_topic=f"/observation/cameras/{topic_map[camera_name]}/image",
                tf_topic=f"/observation/cameras/{topic_map[camera_name]}/tf",
            )
        )

    to_mcap = Dataset2Mcap(dataset=dataset)
    print("\n" + "*" * 16)
    print(f"Exporting episode {episode_index} to {target_path}")
    to_mcap.save_episode(
        target_path=target_path,
        episode_index=episode_index,
        encoder_cfg=dataset2mcap_cfg,
    )
    print(f"Export episode {episode_index} to {target_path} Done")
    print("*" * 16 + "\n")


def build_arrow_dataset(config):
    """Builds the Arrow dataset based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing dataset parameters
            like 'data_path'.

    Returns:
        ROMultiRowDataset: An initialized instance of the ROMultiRowDataset.
    """
    joint_sampler = EpisodeSamplerConfig(
        # target_columns=["joints", "actions", "left_ee", "right_ee"]
        target_columns=["joints", "actions"]
        # target_columns=["joints", "actions"] + link_topics
    )
    dataset = ROMultiRowDataset(
        dataset_path=config["data_path"], row_sampler=joint_sampler
    )
    return dataset


def bucket_path_to_hviz_url(
    bucket_path, layout="3cb26b62-e6da-4604-8645-43abeddae3b2"
):
    """Convert a bucket path to an hviz URL."""
    import urllib.parse

    # Replace bucket prefix with dmpv2 scheme
    if "/horizon-bucket/" in bucket_path:
        dmp_path = bucket_path.replace("/horizon-bucket/", "dmpv2://")
    else:
        dmp_path = bucket_path

    # Construct the inner URL
    inner_url = f"https://hviz.aidi.hobot.cc/rosbag.mcap?filePath={dmp_path}"

    # Construct parameters for the outer URL
    params = {
        "ds": "remote-file",
        "ds.url": inner_url,
        "layout": layout,
    }

    # Encode parameters
    query_string = urllib.parse.urlencode(params)

    return f"https://hviz.aidi.hobot.cc/?{query_string}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MCAP files directly to Arrow format."
    )
    parser.add_argument(
        "--arrow_dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    logger.info(f"Arrow dataset path: {args.arrow_dataset_path}")

    output_path = args.output_path

    if "ARX_Lift2" in args.arrow_dataset_path:
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
    elif "AgileX_Split_Aloha" in args.arrow_dataset_path:
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
    elif "Genie" in args.arrow_dataset_path:
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
        data_path=args.arrow_dataset_path,
        cam_names=["head", "hand_left", "hand_right"],
        urdf=urdf,
        T_base2world=np.eye(4),
        task_names=None,
        with_depth=False,
        robot_type=robot_type,
    )
    dataset_config = dict(
        data_path=args.arrow_dataset_path,
        cam_names=["head", "hand_left", "hand_right"],
        urdf=urdf,
        robot_type=robot_type,
    )

    # Build dataset
    dataset = build_arrow_dataset(config)
    episodes_list = list(dataset.iterate_meta(Episode))
    total_episodes = len(episodes_list)
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
<p>This page contains links to visualize the original and dataset MCAP files in HViz. Please use Chrome.</p>    # noqa: E501
<table border="1" cellpadding="5" cellspacing="0">
    <tr>
        <th>Episode Index</th>
        <th>Dataset MCAP (After Pack & Before Transform) </th>
        <th>Video (After Transform)</th>
    </tr>
"""  # noqa: E501
    html_path = os.path.join(output_path, "viz_all_episodes_hviz_links.html")
    cluster_dir = "viz_results"

    arrow_dataset_dir = os.path.basename(args.arrow_dataset_path)

    for episode_index in episode_index_list:
        logger.info(f"Processing episode index: {episode_index}")

        save_name = f"viz_{arrow_dataset_dir}_episodeidx_{episode_index}"

        # Export mcap to viz data before transform
        target_path = os.path.join(output_path, f"{save_name}.mcap")
        export_mcap(
            dataset=dataset,
            episode_index=episode_index,
            target_path=target_path,
            link_topics=link_topics,
            cam_names=config["cam_names"],
        )

        # Export video to viz data after transform
        video_name = f"{save_name}.mp4"
        video_path = os.path.join(output_path, video_name)

        # Set Transform

        transforms = build_arrow_transforms(config, dataset_config)
        dataset.set_transform(transforms)
        export_arrow_video(
            dataset=dataset,
            episode_index=episode_index,
            video_path=video_path,
            num_workers=16,
            prefetch_factor=4,
        )

        if os.path.exists("/job_data"):
            dataset_url = bucket_path_to_hviz_url(
                target_path,
                layout="6a11ed09-f9a7-492d-961d-397f407db1ef",
            )

            logger.info(f"Dataset viz mcap dmpv2 url: {dataset_url}")

            # Append row to HTML content
            html_content += f"""
    <tr>
        <td>{episode_index}</td>
        <td><a href="{dataset_url}" target="_blank">View Dataset Mcap</a></td>
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


class VizDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, start_idx, end_idx):
        self.source_dataset = source_dataset
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, i):
        idx = self.start_idx + i
        data = self.source_dataset[idx]

        from robo_orchard_lab.dataset.interna1 import InternA1LmdbDataset

        vis_imgs = InternA1LmdbDataset.get_vis_imgs(
            data["imgs"],
            data.get("projection_mat"),
            data.get("hist_robot_state", [None])[-1],
        )
        return vis_imgs
