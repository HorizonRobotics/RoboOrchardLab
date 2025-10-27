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

"""Accessing and Visualizing Datasets
==========================================================

This tutorial covers the two primary ways to access data using **RoboOrchard Dataset**

1.  **Row-level Access** (:py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset`): Accessing single, complete timesteps,
where metadata (from DuckDB) is automatically joined with frame data
(from Arrow).

2.  **Multi-Row Access** (:py:class:`~robo_orchard_lab.dataset.robot.dataset.ROMultiRowDataset`): Accessing a *context* of
data around a specific timestep, crucial for training policies that
require observation or action history.

We will also demonstrate how to visualize image modalities.

Before moving to this tutorial, it's helpful to understand the
specific core data types (like :py:class:`~robo_orchard_lab.dataset.datatypes.joint_state.BatchJointsStateFeature` or
:py:class:`~robo_orchard_lab.dataset.datatypes.camera.BatchCameraDataEncodedFeature`)
that we use. These are defined in our `core data types tutorial <https://horizonrobotics.github.io/robot_lab/robo_orchard/core/master/build/gallery/tutorials/datatypes/index.html>`__.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Prerequisite: Download the Sample Dataset
# ----------------------------------------------------------------
# Before running this tutorial, you must download the
# sample dataset (`dataset 1 (about 9GB) <https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/tree/main/arrow_dataset_place_shoe_2025_08_27>`__
# and `dataset 2 (about 14BG) <https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/tree/main/arrow_dataset_place_shoe_2025_09_11>`__)
# from Hugging Face.
#
# We recommend using the ``huggingface-cli`` tool, which is part of the ``huggingface_hub`` library.
# Below is the example to download dataset 1.
#
# .. code-block:: bash
#
#   dataset_name=arrow_dataset_place_shoe_2025_08_27
#   target_path=data1
#
#   # 1. Install the Hugging Face client
#   pip install huggingface_hub
#
#   # 2. Download the specific dataset sub-folder from the repo
#   # This downloads into a temporary directory 'hf_temp_download'
#   huggingface-cli download \
#       HorizonRobotics/Real-World-Dataset \
#       --repo-type dataset \
#       --include "${dataset_name}/*" \
#       --local-dir hf_temp_download \
#       --local-dir-use-symlinks False
#
#   # 3. Create the target directory and move the dataset files into it
#   # Our tutorial code expects the files to be directly in '${target_path}/'.
#   mkdir ${target_path}
#   mv hf_temp_download/${dataset_name}/* ${target_path}/
#
#   # 4. Clean up the temporary folder
#   rm -rf hf_temp_download
#
# After running these commands, your `data/` directory should have the
# following structure (you can verify with ``tree -L 1 ${target_path}``):
#
# .. code-block:: text
#
#   data1/
#       |---- data-00000-of-00004.arrow
#       |---- data-00001-of-00004.arrow
#       |---- data-00002-of-00004.arrow
#       |---- data-00003-of-00004.arrow
#       |---- dataset_info.json
#       |---- meta_db.duckdb
#       |---- state.json
#

# %%
# Setup and Imports
# --------------------------------
# We'll need `pprint` for nicely formatting metadata and `matplotlib` for
# visualization. We also import Image and io to handle image decoding,
# as BatchCameraDataEncoded typically stores compressed bytes.
#
import io
import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchImageData,
    ImageMode,
)

from robo_orchard_lab.dataset.robot import (
    ConcatRODataset,
    DeltaTimestampSamplerConfig,
    RODataset,
    ROMultiRowDataset,
)

dataset1_path = "data1"
dataset2_path = "data2"

# %%
# 1. Row-level Access with `RODataset`
# ---------------------------------------------
#
# This is the main entry point for most users. :py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset`
# provides a standard PyTorch-style `Dataset` interface.
#
# We set ``meta_index2meta=True``, which is a key feature. This tells the
# dataset to **automatically** resolve the integer indices (like `task_index`)
# in the Arrow file into the full, rich metadata objects (like the task
# description) from the DuckDB database.

dataset = RODataset(dataset1_path, meta_index2meta=True)

print(f"Dataset has {len(dataset)} total timesteps (frames).")
print("Dataset features (schema defined in dataset_info.json):")
pprint.pprint(dataset.frame_dataset.features)

# %%
# Accessing a Single Sample
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When you index the :py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset`,
# you get a single dictionary representing one complete timestep.

sample = dataset[100]  # Let's grab the 100th frame

print(f"Type of a single sample: {type(sample)}")
print(f"Keys in the sample: {sample.keys()}")

# %%
# Because we set ``meta_index2meta=True``, the metadata keys (`instruction`,
# `task`, `episode`) contain the actual data from DuckDB, not just indices.
print("--- Example of Resolved Metadata ---")
print(f"Instruction: {sample['instruction']}")
print(f"Task Info: {sample['task']}")
print(f"Episode Info: {sample['episode']}")

# %%
# The sample also contains the corresponding frame data for that timestep.
print("--- Example of Frame Data ---")
print(f"Joints data type: {type(sample['joints'])}")
print(f"Joints: {sample['joints']}")

# %%
# Visualizing Image Modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The data for `left` (the left wrist camera) is loaded according to the
# :py:class:`~robo_orchard_lab.dataset.datatypes.camera.BatchCameraDataEncodedFeature` type.
# This feature typically decodes the raw (e.g., JPEG) bytes from the Arrow file into a `PIL.Image` object
# or a raw `bytes` buffer.
#
# Let's visualize it.
#


def simple_decoder(image_bytes: bytes, format: str) -> BatchImageData:
    pil_img = Image.open(io.BytesIO(image_bytes))
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(np.array(pil_img)).unsqueeze(0)
    return BatchImageData(sensor_data=img_tensor, pix_fmt=ImageMode.RGB)


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("Camera Feeds", fontsize=16)

for ax, col_name in zip(axes, ["left", "middle", "right"], strict=True):  # type: ignore
    camera_data: BatchCameraDataEncoded = sample[col_name]
    decode_camera_data: BatchCameraData = camera_data.decode(
        decoder=simple_decoder
    )
    img_to_plot = decode_camera_data.sensor_data[0].numpy()
    # Plot the image on the correct axis
    ax.imshow(img_to_plot)
    ax.set_title(col_name.capitalize())
    ax.axis("off")  # Hide axis ticks and labels

plt.tight_layout()
plt.show()

# %%
# Accessing Entire Columns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can also access an entire column efficiently. This leverages the
# columnar nature of Apache Arrow.
all_joints = dataset["joints"]
print("--- Columnar Access Example ---")
print(f"Type of samples: {type(all_joints)}")
print(f"Total joint entries: {len(all_joints)}")
print(f"Type of first joint entry: {type(all_joints[0])}")

# %%
# Part 2: Multi-Row (Context) Access
# ---------------------------------------------
# For tasks like Imitation Learning, a model needs to see not just the
# current state, but also a *history* of observations and a *future*
# of actions. :py:class:`~robo_orchard_lab.dataset.robot.dataset.ROMultiRowDataset`
# handles this.
#
# We must provide a ``row_sampler`` configuration. Here, we use
# :py:class:`~robo_orchard_lab.dataset.robot.row_sampler.DeltaTimestampSamplerConfig`
# to sample data based on time.

# %%
# This configuration asks for 32 `joints` samples.
# The timestamps are relative to the anchor timestep (t=0).
#
# range(32) -> i = 0 to 31
#
# i=0: t + (0-1)/25 = t - 0.04s (1 step *before* anchor)
#
# i=1: t + (1-1)/25 = t + 0.0s  (the anchor step)
#
# i=31: t + (31-1)/25 = t + 1.2s (30 steps *after* anchor)
#
# This setup is common for behavior cloning (e.g., 1 observation, 31 actions).
timestamps = [0.0 + 1.0 / 25 * i for i in range(32)]
delta_timestamps_config = DeltaTimestampSamplerConfig(
    column_delta_ts={"joints": timestamps},
    tolerance=0.01,  # Allow 10ms tolerance in timestamp matching
)

multi_row_dataset = ROMultiRowDataset(
    dataset1_path, row_sampler=delta_timestamps_config, meta_index2meta=True
)

# %%
# Accessing a Multi-Row Sample
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, indexing the dataset returns a sample where "joints" is a *list*
# of 32 states, sampled according to our config.
#
delta_ts_row = multi_row_dataset[100]  # Get context around the 100th frame

print("--- Multi-Row Sample ---")
print(f"Type of multi-row sample: {type(delta_ts_row)}")
print(f"Keys in the sample: {delta_ts_row.keys()}")

# %%
# The "joints" key now contains a list of 32 elements, as requested.
print(f"Number of 'joints' samples retrieved: {len(delta_ts_row['joints'])}")

print(delta_ts_row["joints"])  # Uncomment to see the full data

# %%
# Part 3: Concatenating Datasets
# -----------------------------------------------------
# :py:class:`~robo_orchard_lab.dataset.robot.dataset.ConcatRODataset` combines data from different
# dataset into a single dataset.
#

dataset1 = RODataset(dataset1_path, meta_index2meta=True)
dataset2 = RODataset(dataset1_path, meta_index2meta=True)

concat_dataset = ConcatRODataset([dataset1, dataset2])

total_len = len(concat_dataset)
print(f"Total concatenated length: {total_len}")
assert total_len == len(dataset1) + len(dataset2)

# %%
# Indexing the Concatenated Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :py:class:`~robo_orchard_lab.dataset.robot.dataset.ConcatRODataset` automatically maps a global index to the correct
# dataset and injects a `dataset_index` key into the sample.

# %%
# Sample 1: From the first dataset (index < len(dataset_stack))
#
sample_a = concat_dataset[100]

# The `dataset_index` should be 0
print("Sample at index 100:")
print(f"  Task: {sample_a['task'].name}")
print(f"  Injected 'dataset_index': {sample_a['dataset_index']}")

# %%
# Sample 2: From the second dataset (index >= len(dataset_stack))
#
sample_b_index = len(dataset1) + 50  # 50th sample from the second dataset
sample_b = concat_dataset[sample_b_index]

# The `dataset_index` should be 1
print(f"Sample at index {sample_b_index}:")
print(f"  Task: {sample_b['task'].name}")
print(f"  Injected 'dataset_index': {sample_b['dataset_index']}")
