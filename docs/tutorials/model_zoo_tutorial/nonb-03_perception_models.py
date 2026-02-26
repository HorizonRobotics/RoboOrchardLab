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

"""Model Zoo: Loading Pre-trained Perception Models
=================================================================

This tutorial demonstrates how to load and use the pre-trained
State-of-the-Art (SOTA) perception models provided by the
**RoboOrchardLab**.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence
# ---------------------------------------------------------------------------
#
# `Click here to visit the homepage. <https://horizonrobotics.github.io/robot_lab/bip3d/index.html>`__
#
# Loading Pretrained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import torch
#   from robo_orchard_lab.models import TorchModelMixin
#
#   model: torch.nn.Module = TorchModelMixin.load_model("hf://HorizonRobotics/BIP3D_Tiny_Det")
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TBD.
