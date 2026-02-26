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
import logging
import os

from huggingface_hub import snapshot_download

from robo_orchard_lab.inference import InferencePipelineMixin

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def build_input():
    # prepare input data
    import numpy as np
    import scipy.io as scio
    from PIL import Image

    from robo_orchard_lab.models.finegrasp.processor import GraspInput

    data_dir = snapshot_download(
        repo_id="HorizonRobotics/FineGrasp",
        allow_patterns="data_example/**",
    )
    rgb_image_path = os.path.join(data_dir, "data_example/0000_rgb.png")
    depth_image_path = os.path.join(data_dir, "data_example/0000_depth.png")
    intrinsic_file = os.path.join(data_dir, "data_example/0000.mat")

    depth_image = np.array(Image.open(depth_image_path), dtype=np.float32)
    rgb_image = np.array(Image.open(rgb_image_path), dtype=np.float32)
    intrinsic_matrix = scio.loadmat(intrinsic_file)["intrinsic_matrix"]

    # Grasp workspace limits [xmin, xmax, ymin, ymax, zmin, zmax].
    grasp_workspace = [-1, 1, -1, 1, 0.0, 2.0]

    # depth_image is in mm, depth_scale=1000.0.
    depth_scale = 1000.0

    input_data = GraspInput(
        rgb_image=rgb_image,
        depth_image=depth_image,
        depth_scale=depth_scale,
        intrinsic_matrix=intrinsic_matrix,
        grasp_workspace=grasp_workspace,
    )
    return input_data


def build_pipeline():
    pipeline_dir = snapshot_download(
        repo_id="HorizonRobotics/FineGrasp",
        allow_patterns="finegrasp_pipeline/**",
    )

    pipeline = InferencePipelineMixin.load(
        os.path.join(pipeline_dir, "finegrasp_pipeline")
    )
    pipeline.to("cuda")
    pipeline.model.eval()
    logger.info(f"Pipeline loaded: {pipeline}")

    return pipeline


if __name__ == "__main__":
    input_data = build_input()
    pipeline = build_pipeline()
    output = pipeline(input_data)
    logger.info(f"Best grasp pose: {output.grasp_poses[0]}")
