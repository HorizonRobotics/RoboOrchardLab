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


# %%
# Setup and Imports
# --------------------------------
#

import matplotlib.pyplot as plt
import torch
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
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


# required input type dict is sample and output type is dict
def transform(sample: dict) -> dict[str, torch.Tensor]:
    # 1. Decode camera
    #   (BatchCameraDataEncoded -> BatchCameraData)
    env_camera_data: BatchCameraData = sample["middle"].decode()

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
# Bounded readers for dataset mixtures
# ------------------------------------
# ``DictIterableDataset`` can rotate unfinished sources after a target number
# of batches. This lets another source enter the active pool while keeping
# ``max_dataset_concurrency`` readers per DataLoader worker. For example:
#
# .. code-block:: python
#
#     from robo_orchard_lab.dataset.robot import (
#         DataLoader as RODataLoader,
#         DictIterableDataset,
#         RODatasetItem,
#     )
#
#     mixture = DictIterableDataset(
#         datasets=[
#             RODatasetItem(dataset_path="data1", transform=transform),
#             RODatasetItem(dataset_path="data2", transform=transform),
#         ],
#         shuffle=True,
#         max_dataset_concurrency=1,
#         offload_num_batches=100,
#     )
#     loader = RODataLoader(
#         mixture,
#         batch_size=8,
#         num_workers=4,
#         use_dataset_side_batching=True,
#     )
#
# Each worker prepares its complete shuffled/resampled index plan before
# dividing it into finite reader segments. A small remainder merges into the
# last segment, so 100 batches is a target, not a strict maximum. Existing
# ``drop_last`` rules apply only to the original final tail.
#
# Offload also works with ``shuffle=False`` and with sample-level loading
# (``use_dataset_side_batching=False``). With this repository's DataLoader,
# sample-level segment size uses its actual batch size; direct iteration or
# an external loader without this context counts one sample per unit.
#
# Shuffled mixing uses each worker's fixed resampled output target T. Pending
# readers are selected with weight T/B, where B is their next residence's
# output rows; active sources use T/u, where u is the next yield's row count
# (one sample or an inner batch, including its short tail). These compensate
# the two different selection units. Inner ``drop_last`` tails do not count
# toward output weights; the underlying index plan remains unchanged.
# A completed reader closes before it becomes eligible again, including for
# immediate reselection. The concurrency bound still applies. Finite epoch
# initialization, changing tails and source exhaustion cause prefix variation:
# the complete epoch retains exact quotas, not IID finite-window proportions.
# Ordered iteration retains its source order.
#
# ``offload_num_batches=None`` keeps ordinary iteration. Its full-source
# residences naturally give uniform admission (T/B=1); without finite rotation,
# short-window mixture proportions are not guaranteed.
# When enabled,
# ``offload_dir`` optionally selects the cache parent; its default is the
# system temporary directory. Immutable base indices are reused across local
# workers and epochs. Private plans are removed after normal exhaustion;
# interrupted iterations retain them for cleanup with the run directory.
# Keep source data and split files immutable while reusing the cache, or use
# a new versioned source/cache directory. Enabling rotation changes delivery
# order and does not provide an exact training-checkpoint resume.
#
# With offload enabled, construction snapshots the declared configuration of
# input items and groups, excluding their runtime private caches. Caller
# objects stay unchanged; later changes to them do not update the mixture.
#
# Custom ``DatasetItem`` subclasses can override ``get_index_cache_key()``
# to return a stable string describing the reader/source version and complete
# unsharded row selection, including its order. This method must not create a
# reader or materialize indices. The default hashes JSON configuration;
# custom implementations can use another deterministic scheme for opaque
# configuration values. ``RODatasetItem`` excludes post-read transforms from
# this key so equivalent selections share their base file. Shuffle plans and
# consumption cursors remain private to each item and worker.

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
    env_camera_data: BatchCameraData = sample["middle"].decode()
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
