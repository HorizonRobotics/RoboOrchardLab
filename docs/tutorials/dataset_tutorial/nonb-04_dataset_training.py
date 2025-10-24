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

"""Integrating with PyTorch DataLoader
==========================================================

.. note::

    This tutorial assumes you have already downloaded the datasets as
    instructed in the :ref:`previous tutorial <sphx_glr_build_tutorials_dataset_tutorial_nonb-03_dataset_access.py>`.

This tutorial demonstrates how to bridge the gap between **RoboOrchard Dataset**
and PyTorch's ``DataLoader``, which is essential for model training.

The core challenge is that our datasets return rich, custom objects
(like ``BatchCameraDataEncoded``), but ``DataLoader``'s default collation
function only understands tensors, NumPy arrays, and standard Python types.

The solution is to use :py:meth:`~robo_orchard_lab.dataset.robot.dataset.RODataset.set_transform()` to convert the
complex sample into a simple dictionary of tensors before it gets batched.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Setup and Imports
# --------------------------------
#

import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
    BatchImageData,
    ImageMode,
)
from torch.utils.data import DataLoader

from robo_orchard_lab.dataset.robot import (
    DeltaTimestampSamplerConfig,
    RODataset,
    ROMultiRowDataset,
)

# %%
# Part 1: Transform sample
# --------------------------------
# We define a transform that takes the **raw sample dict** and
# converts it into a **collatable dict** containing only tensors.
# This transform will:
#
# 1. Decode the 'middle' camera image.
#
# 2. Extract its tensor data (squeezing the batch dim).
#
# 3. Extract the 'joints' position tensor.
#
# 4. Extract the 'actions' position tensor.
#
# 5. Return a new, simple dictionary.
#


def simple_decoder(image_bytes: bytes, format: str) -> BatchImageData:
    pil_img = Image.open(io.BytesIO(image_bytes))
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(np.array(pil_img)).unsqueeze(0)
    return BatchImageData(sensor_data=img_tensor, pix_fmt=ImageMode.RGB)


# required input type dict is sample and output type is dict
def transform(sample: dict) -> dict[str, torch.Tensor]:
    # 1. Decode camera
    #   (BatchCameraDataEncoded -> BatchCameraData)
    env_camera_data: BatchCameraData = sample["middle"].decode(
        decoder=simple_decoder
    )

    # 2. Extract image tensor (Shape: [1, H, W, C] -> [H, W, C])
    # We squeeze(0) to remove the batch dim, as the DataLoader will add it back
    obs_camera = env_camera_data.sensor_data.squeeze(0)

    # 3. Extract joints tensor (Type: BatchJointsState)
    # We squeeze(0) to remove the batch dim, as the DataLoader will add it back
    obs_joints = sample["joints"].position[0]

    # 4. Extract actions tensor (Type: BatchJointsState)
    # We squeeze(0) to remove the batch dim, as the DataLoader will add it back
    action = sample["actions"].position[0]

    # 5. Return the simple, collatable dictionary
    return {
        "obs_camera": obs_camera,
        "obs_joints": obs_joints,
        "action": action,
    }


dataset = RODataset("data1", meta_index2meta=True)
# Apply the transform
dataset.set_transform(transform)

# %%
# Let's test the transform on a single sample
transformed_sample = dataset[0]
print("--- Transformed Sample (Ready for Batching) ---")
print(f"Keys: {transformed_sample.keys()}")
print(
    f"Camera type: {type(transformed_sample['obs_camera'])} and shape {transformed_sample['obs_camera'].shape}"
)
print(
    f"Joints type: {type(transformed_sample['obs_joints'])} and shape: {transformed_sample['obs_joints'].shape}"
)
print(
    f"Actions type: {type(transformed_sample['action'])} and shape: {transformed_sample['action'].shape}"
)
# TODO: combine with dataloader

# %%
# Part 2: Working with PyTorch DataLoader
# -----------------------------------------------------
# Now that our dataset returns a simple ``dict`` of tensors, we can
# use the standard PyTorch ``DataLoader`` with its default collation fn.

data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,  # Set to > 0 for parallel loading
)
batch = next(iter(data_loader))

print(f"Batch keys: {batch.keys()}")
print(f"Batched 'obs_camera' shape: {batch['obs_camera'].shape}")
print(f"Batched 'obs_joints' shape: {batch['obs_joints'].shape}")
print(f"Batched 'action' shape: {batch['action'].shape}")

# %%
# Visualize the batch
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Batch of 'obs_camera' from DataLoader", fontsize=16)
for i in range(4):
    # Note: .numpy() is needed as data is now a Torch Tensor
    axes[i].imshow(batch["obs_camera"][i].numpy())  # type: ignore
    axes[i].set_title(f"Sample {i}")  # type: ignore
    axes[i].axis("off")  # type: ignore
plt.show()

# %%
# Part 3: DataLoader with ROMultiRowDataset
# ---------------------------------------------
# This is common for Imitation Learning. The dataset returns a **list**
# of states/actions. We need a transform to stack them into a tensor.
#


def multi_row_transform(sample: dict) -> dict[str, torch.Tensor]:
    """Converts a multi-row sample (with lists of data) into a collatable dictionary of stacked tensors."""

    # Decode camera
    env_camera_data: BatchCameraData = sample["middle"].decode(
        decoder=simple_decoder
    )
    obs_camera = env_camera_data.sensor_data.squeeze(0)

    # 'joints' is a list[BatchJointsState] of length 8
    # We stack their .position attributes
    obs_joints_history = torch.stack(
        [js.position[0] for js in sample["joints"]]
    )  # Shape: [8, N_JOINTS]

    action_history = torch.stack(
        [ac.position[0] for ac in sample["actions"]]
    )  # Shape: [8, N_ACTIONS]

    return {
        "obs_camera": obs_camera,
        "obs_joints_history": obs_joints_history,
        "action_history": action_history,
    }


timestamps = [0.0 + 1.0 / 25 * i for i in range(8)]
multi_row_config = DeltaTimestampSamplerConfig(
    column_delta_ts={
        "joints": timestamps,
        "actions": timestamps,
    },
    tolerance=0.01,
)

# Load the dataset and set the transform
multi_row_dataset = ROMultiRowDataset(
    "data1", row_sampler=multi_row_config, meta_index2meta=True
)
multi_row_dataset.set_transform(multi_row_transform)

# Create the DataLoader
multi_row_loader = DataLoader(
    dataset=multi_row_dataset, batch_size=4, shuffle=False, num_workers=2
)

# Get one batch
multi_row_batch = next(iter(multi_row_loader))

print(f"Multi-row batch keys: {multi_row_batch.keys()}")
print(f"Batched 'obs_camera' shape: {multi_row_batch['obs_camera'].shape}")
print(
    f"Batched 'obs_joints_history' shape (Batch, Context, Features): "
    f"{multi_row_batch['obs_joints_history'].shape}"
)
print(
    f"Batched 'action_history' shape (Batch, Context, Features): "
    f"{multi_row_batch['action_history'].shape}"
)
