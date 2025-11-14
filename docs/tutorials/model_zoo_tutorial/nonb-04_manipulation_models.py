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

# ruff: noqa: E501 D415 D205 E402

"""Model Zoo: Loading Pre-trained Manipulation Models
=================================================================

This tutorial demonstrates how to load and use the pre-trained
State-of-the-Art (SOTA) manipulation models provided by the
**RoboOrchardLab**.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# FineGrasp: Towards Robust Grasping for Delicate Objects
# ---------------------------------------------------------------------------
#
# `Click here to visit the homepage. <https://horizonrobotics.github.io/robot_lab/finegrasp/index.html>`__
#
# Loading Pretrained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import os
#   import torch
#   from huggingface_hub import snapshot_download
#   from robo_orchard_lab.models import ModelMixin
#
#   file_path = snapshot_download(
#       repo_id="HorizonRobotics/FineGrasp",
#       allow_patterns=["finegrasp_pipeline/**"],
#   )
#   model_path = os.path.join(file_path, "finegrasp_pipeline")
#   model: torch.nn.Module = ModelMixin.load_model(model_path)
#
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import os
#   import numpy as np
#   import scipy.io as scio
#   from PIL import Image
#   from robo_orchard_lab.models.finegrasp.processor import GraspInput
#   from huggingface_hub import snapshot_download
#   from robo_orchard_lab.inference import InferencePipelineMixin
#
#   file_path = snapshot_download(
#       repo_id="HorizonRobotics/FineGrasp",
#       allow_patterns=[
#           "finegrasp_pipeline/**",
#           "data_example/**"
#       ],
#   )

#   pipeline = InferencePipelineMixin.load(
#       os.path.join(file_path, "finegrasp_pipeline")
#   )
#   pipeline.to("cuda")
#   pipeline.model.eval()

#   rgb_image_path = os.path.join(file_path, "data_example/0000_rgb.png")
#   depth_image_path = os.path.join(file_path, "data_example/0000_depth.png")
#   intrinsic_file = os.path.join(file_path, "data_example/0000.mat")

#   depth_image = np.array(Image.open(depth_image_path), dtype=np.float32)
#   rgb_image = np.array(Image.open(rgb_image_path), dtype=np.float32)
#   intrinsic_matrix = scio.loadmat(intrinsic_file)["intrinsic_matrix"]

#   # Grasp workspace limits [xmin, xmax, ymin, ymax, zmin, zmax].
#   grasp_workspace = [-1, 1, -1, 1, 0.0, 2.0]

#   # depth_image is in mm, depth_scale=1000.0.
#   depth_scale = 1000.0

#   input_data = GraspInput(
#       rgb_image=rgb_image,
#       depth_image=depth_image,
#       depth_scale=depth_scale,
#       intrinsic_matrix=intrinsic_matrix,
#       grasp_workspace=grasp_workspace,
#   )

#   output = pipeline(input_data)
#   print(f"Best grasp pose: {output.grasp_poses[0]}")


# %%
# SEM: Enhancing Spatial Understanding for Robust Robot Manipulation
# ---------------------------------------------------------------------------
#
# `Click here to visit the homepage. <https://arxiv.org/abs/2505.16196>`__
#
# Loading Pretrained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TBD
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TBD.
