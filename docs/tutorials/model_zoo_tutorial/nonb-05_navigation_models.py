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

"""Model Zoo: Loading Pre-trained Navigation Models
=================================================================

This tutorial demonstrates how to load and use the pre-trained
State-of-the-Art (SOTA) navigation models provided by the
**RoboOrchardLab**.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation
# --------------------------------------------------------------------------------------------
#
# `Click here to visit the homepage. <https://horizonrobotics.github.io/robot_lab/aux-think/index.html>`__
#
# Loading Pretrained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import torch
#   from robo_orchard_lab.models import TorchModelMixin
#
#   model: torch.nn.Module = TorchModelMixin.load_model("hf://HorizonRobotics/Aux-Think")
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import torch
#   from robo_orchard_lab.inference import InferencePipelineMixin
#   from robo_orchard_lab.models.aux_think.processor import AuxThinkInput
#
#   # -----------------------------
#   # Step 1. Load a saved pipeline
#   # -----------------------------
#   pipeline = InferencePipelineMixin.load("hf://HorizonRobotics/Aux-Think")
#   pipeline.model.eval()
#
#   # -----------------------------
#   # Step 2. Prepare raw input
#   # -----------------------------
#   data = AuxThinkInput(
#       image_paths=[
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_0.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_1.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_2.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_3.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_4.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_5.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_6.png",
#           "hf://HorizonRobotics/Aux-Think/data_example/rgb_7.png",
#       ],
#       instruction="Walk down the hallway to the right of the billiards table. Stop at the top of the staircase."
#   )
#
#   # -----------------------------
#   # Step 3. Run inference
#   # (pre_process → collate → model → post_process)
#   # -----------------------------
#   result = pipeline(data)
#   print(result.text)
#
#   # Example Output:
#   # "The next action is turn right 15 degrees, move forward 50 cm, turn right 15 degrees."
#
#   # -----------------------------
#   # Step 4. Batch inference (optional)
#   # -----------------------------
#   batch_data = [data, data]
#   batch_results = list(pipeline(batch_data))
#   for r in batch_results:
#       print(r.text)

# %%
# MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming
# --------------------------------------------------------------------------------------------
#
# `Click here to visit the homepage. <https://horizonrobotics.github.io/robot_lab/monodream/index.html>`__
#
# Loading Pretrained Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import torch
#   from robo_orchard_lab.models import TorchModelMixin
#
#   model: torch.nn.Module = TorchModelMixin.load_model("hf://HorizonRobotics/MonoDream")
#
# Inference Pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#   import torch
#   from robo_orchard_lab.inference import InferencePipelineMixin
#   from robo_orchard_lab.models.monodream.processor import MonoDreamInput
#
#   # -----------------------------
#   # Step 1. Load a saved pipeline
#   # -----------------------------
#   directory = "hf://HorizonRobotics/MonoDream"
#   pipeline = InferencePipelineMixin.load(directory)
#   pipeline.model.init_components(directory)
#   pipeline.model.eval()
#
#   # -----------------------------
#   # Step 2. Prepare raw input
#   # -----------------------------
#   data = MonoDreamInput(
#       image_paths=[
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_0.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_1.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_2.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_3.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_4.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_5.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_6.png",
#           "hf://HorizonRobotics/MonoDream/data_example/rgb_7.png",
#       ],
#       instruction="Walk down the hallway to the right of the billiards table. Stop at the top of the staircase."
#   )
#
#   # -----------------------------
#   # Step 3. Run inference
#   # (pre_process → collate → model → post_process)
#   # -----------------------------
#   result = pipeline(data)
#   print(result.text)
#
#   # Example Output:
#   # "The next action is turn right 15 degrees, move forward 25 cm, turn right 45 degrees."
#
#   # -----------------------------
#   # Step 4. Batch inference (optional)
#   # -----------------------------
#   batch_data = [data, data]
#   batch_results = list(pipeline(batch_data))
#   for r in batch_results:
#       print(r.text)
