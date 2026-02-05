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

from robo_orchard_lab.dataset.robot.dataset_visualizer import (
    RODatasetVisualizer,
)


def build_ro_dataset(data_path, urdf_path, hist_steps=1, pred_steps=64):
    import sys

    from config_agibot_digit_dataset import (
        build_transforms,
        cam_names,
        scale_shift,
    )
    from torchvision.transforms import Compose

    from robo_orchard_lab.dataset.robot.dataset import RODataset
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    sys.path.append("projects/sem/common/configs")
    config = dict(hist_steps=hist_steps, pred_steps=pred_steps)
    g1_kinematics_config = dict(urdf=urdf_path)

    transforms = build_transforms(
        config,
        mode="training",
        cam_names=cam_names,
        kinematics_config=g1_kinematics_config,
        scale_shift=scale_shift,
    )
    transforms = Compose([build(x) for x in as_sequence(transforms)])

    ro = RODataset(dataset_path=data_path, meta_index2meta=False)
    ro.frame_dataset = ro.frame_dataset.with_format(
        type="numpy",
        columns=[
            "joint_state",
            "joint_action",
            "joint_raw_frame_index",
            "camera_intrinsics/middle",
            "camera_intrinsics/left",
            "camera_intrinsics/right",
        ],
        output_all_columns=True,
    )
    ro.set_transform(transforms)

    return ro


if __name__ == "__main__":

    import argparse
    import logging

    logger = logging.getLogger(__file__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d - %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="./urdf/G1_omnipicker.urdf",
    )
    args = parser.parse_args()

    dataset = build_ro_dataset(args.input_dir, args.urdf_path)

    vis = RODatasetVisualizer(dataset, ee_indices=(7, 15))
    vis.visualize_episode(
        episode_index=args.episode_index,
        output_dir=args.output_dir,
        fps=args.fps,
        interval=10,
        with_frame_idx=True,
        with_valid_mask=True,
    )
