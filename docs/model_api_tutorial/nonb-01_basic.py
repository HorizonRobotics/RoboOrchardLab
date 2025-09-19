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

"""Creating, Saving, and Loading Models
==========================================================
"""

# %%
# Core concepts
# ---------------------------------------------------------
# Before we begin, let's understand the two key classes:
#
# 1. :py:class:`~robo_orchard_lab.models.mixin.TorchModuleCfg`: This is a specialized configuration class designed to define all the necessary parameters for a PyTorch ``nn.Module``,
# such as layer counts, dimensions, activation functions, etc. Your model will be instantiated from it.
#
# 2. :py:class:`~robo_orchard_lab.models.mixin.ModelMixin`: This is a mixin class. Your custom model class should inherit from it.
# It provides the essential ``save_model`` and ``load_model`` methods.
#

# %%
# Step 1: Define Your Model and Configuration
# ---------------------------------------------------------
# First, we need to define the model's architecture and its corresponding configuration.
#
# Let's create a simple fully-connected network called ``SimpleNet``.
#
# 1.  Create the config class ``SimpleNetCfg``: This class must inherit from :py:class:`~robo_orchard_lab.models.mixin.TorchModuleCfg` and should define the parameters your model needs (e.g., `input_size`, `hidden_size`, `output_size`).
#
# 2.  Create the model class ``SimpleNet``: This class must inherit from :py:class:`~robo_orchard_lab.models.mixin.ModelMixin`. In its ``__init__`` method, it accepts a config object ``cfg`` and calls ``super().__init__(cfg)`` to complete the setup.
#

import torch.nn as nn

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)


# 1. Define the model class, inheriting from ModelMixin
class SimpleNet(ModelMixin):
    def __init__(self, cfg: "SimpleNetCfg"):
        # It's crucial to call super().__init__ and pass the cfg
        super().__init__(cfg)

        self.fc1 = nn.Linear(cfg.input_size, cfg.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. Define the configuration class for the model
class SimpleNetCfg(TorchModuleCfg[SimpleNet]):
    class_type: ClassType_co[SimpleNet] = SimpleNet
    input_size: int = 784
    hidden_size: int = 128
    output_size: int = 10


# %%
# Step 2: Instantiate and Save the Model
# ---------------------------------------------------------
# The :py:class:`~robo_orchard_lab.models.mixin.ModelMixin` provides the ``save_model`` method,
# which automatically performs two actions:
#
# 1.  Saves the model's configuration ``cfg`` to ``model.config.json``.
# 2.  Saves the model's weights ``state_dict`` to ``model.safetensors``.
#

import os
import shutil

config = SimpleNetCfg(hidden_size=256)

# %%
# 2. Instantiate the model by calling the config object. This leverages the functionality of ClassInitFromConfigMixin
model = config()
print("Model created:", model)

# %%
# 3. Call the save_model method
output_dir = "./checkpoint"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

model.save_model(output_dir)

import subprocess

print(f"Model has been saved to the `{output_dir}` directory.")
print(subprocess.check_output(["tree", output_dir]).decode())

# %%
# Step 3: Load the Model
# ---------------------------------------------------------
# Loading the model is just as easy. The ``load_model`` method automatically reads ``model.config.json`` to build the model architecture and then loads the weights from ``model.safetensors``.
#

loaded_model = ModelMixin.load_model(output_dir)
print("Model loaded:", loaded_model)
