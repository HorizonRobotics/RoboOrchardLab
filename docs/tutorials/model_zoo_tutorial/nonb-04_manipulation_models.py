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
#   import torch
#   from robo_orchard_lab.models import ModelMixin
#
#   model: torch.nn.Module = ModelMixin.load_model("hf://HorizonRobotics/FineGrasp/finegrasp_pipeline")
#
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   from robo_orchard_lab.models.finegrasp.processor import GraspInput
#   from robo_orchard_lab.inference import InferencePipelineMixin
#
#   pipeline = InferencePipelineMixin.load("hf://HorizonRobotics/FineGrasp/finegrasp_pipeline")
#   pipeline.to("cuda")
#   pipeline.model.eval()
#
#   # Grasp workspace limits, [xmin, xmax, ymin, ymax, zmin, zmax].
#   grasp_workspace = [-1, 1, -1, 1, 0.0, 2.0]
#
#   # depth_image is in mm, depth_scale=1000.0.
#   depth_scale = 1000.0
#
#   input_data = GraspInput(
#       rgb_image="hf://HorizonRobotics/FineGrasp/data_example/0000_rgb.png",
#       depth_image="hf://HorizonRobotics/FineGrasp/data_example/0000_depth.png",
#       depth_scale=depth_scale,
#       intrinsic_matrix="hf://HorizonRobotics/FineGrasp/data_example/0000.mat",
#       grasp_workspace=grasp_workspace,
#   )
#
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
