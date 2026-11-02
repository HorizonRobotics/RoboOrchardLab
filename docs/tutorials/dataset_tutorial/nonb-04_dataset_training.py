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
# Framework sampling seeds
# ------------------------
# ``DictIterableDataset`` defaults to ``seed=42`` when neither ``seed`` nor
# ``generator`` is supplied. It owns a private sampling stream; initialization
# does not draw from the
# default Torch RNG. Model startup seeding and user transforms/collators remain
# separate responsibilities. Prefer ``seed`` for
# framework-owned sampling and reproducible complete epochs. It is a base seed
# for dataset sampling, not a model or worker initialization seed.
#
# An explicit ``generator`` remains available for callers controlling a
# stateful sampling stream. Without ``seed``, that stream keeps advancing;
# ``set_epoch`` records the epoch but does not reseed the Generator or promise
# same-epoch replay. ``seed`` and ``generator`` cannot be combined.
# ``IterableWithLenDataset`` also accepts keyword-only ``seed`` for its private
# index-sampling generator. Omitting both inputs uses ``DEFAULT_SAMPLING_SEED``
# from ``robo_orchard_lab.dataset.sampler``; an explicit generator is retained.
#
# ``HookBasedTrainer`` and ``utils.accelerate.prepare_data_loader`` bind the
# final prepared dataset's sampling scope automatically. With a combined raw
# ``accelerator.prepare(loader, model, ...)``, bind after prepare, before
# iterating:
#
# .. code-block:: python
#
#     from robo_orchard_lab.utils.accelerate import (
#         bind_prepared_data_loader_sampling,
#     )
#     bind_prepared_data_loader_sampling(accelerator, loader)
#
# The binding takes global rank from the existing Accelerator in the parent
# process before workers start; project builders only supply the training seed.
#
# Native shards use rank-specific streams; Accelerate's ``IterableDatasetShard`` fallback uses the same stream
# on each rank before position-based assignment. Fallback requires matching
# source/worker configuration and sample order across ranks. Dispatcher retains
# its existing reading/distribution behavior; binding does not alter its dataset
# or introduce a dispatcher-specific RNG guarantee.
#
# For managed native/fallback sampling, before a new full traversal call
# ``prepared_loader.set_epoch(epoch)``; ``HookBasedTrainer`` does this
# automatically. For an unprepared loader, use ``loader.dataset.set_epoch``.
# Seeded complete traversals replay from that epoch's start, including
# persistent workers. Explicit generators retain their advancing state.
# Offload reader rotation never restarts the stream. Dispatcher and unmanaged
# epoch notifications retain Accelerate's behavior. Managed training gives
# missing loader and map-sampler generators independent data RNGs. Sampling
# defaults to 42; worker initialization uses a separate stream derived from
# ``torch.initial_seed()`` and global rank. Thus the run seed can change worker
# randomness without consuming the model's default RNG. An explicit loader
# generator keeps its PyTorch semantics. This differs from PyTorch's default
# of drawing worker seeds from the current global RNG state and does not
# guarantee exact replay of random transforms. Changing workers, batching or offload settings
# does not promise the same sample sequence.
#
# %%
# Resuming within an epoch
# ------------------------
# The Lab ``DataLoader`` exposes ``state_dict()`` and ``load_state_dict(state)``.
# Capture after a batch has been processed successfully, and load into an
# inactive loader before creating its iterator. The state records delivery
# positions and the information needed to reconstruct the data plan; it does
# not save readers, full permutations or prefetched batches. A retained
# iterator must be closed before loading state. Concurrent iterators have no
# unique recovery position.
#
# With ``HookBasedTrainer``, use ``SaveCheckpointConfig`` and
# ``ResumeCheckpointConfig``. In supported recovery configurations, a successful
# save at a completed optimizer-step boundary includes processed batch progress
# and the final prepared loader's recovery information, excluding unconsumed
# lookahead and prefetch. The Lab loader owns its worker positions. Trainer/data
# recovery excludes DeepSpeed/FSDP, node-local saves and external stateful
# loaders. Restoring model state alone or saving an arbitrary base-loader
# position does not establish that boundary.
#
# ``ResumeCheckpointConfig.load_state`` owns training capability checks and
# the post-load accumulation phase reset. The generic
# ``utils.huggingface.accelerator_load_state`` only loads native state and
# preserves its phase, even when Trainer progress is registered.
#
# For a seed-managed ``DictIterableDataset`` in the Lab ``DataLoader``, each
# worker can skip its consumed prefix without reading old samples when the
# data and iteration configuration can be reconstructed. Offload, sample
# prefetch, explicit dataset generators and other unsupported configurations
# instead replay data; recovery may repeat or omit samples. Both
# ``in_order=True`` and ``False`` are supported, but unordered delivery does
# not preserve the original cross-worker order. Keep the source data and
# rank/worker layout fixed. Random transforms, custom generator sharing and
# persistent workers' random streams are best effort.
#
# Other loaders use Accelerate's batch skipping during trainer resume. For map
# datasets this can skip sampler indices without reading old samples; iterable
# datasets may read and discard the skipped prefix. A regular PyTorch
# ``DataLoader`` wrapping ``DictIterableDataset`` follows this iterable replay
# path; use the Lab ``DataLoader`` for worker-local fast forward. Framework-owned
# map sampler RNG is restored from the epoch start, while user-provided
# generators and custom samplers are best effort.
# During generic loader recovery, the prepared loader and Accelerate's skip
# wrapper can both notify ``set_epoch``. Custom dataset and sampler methods
# should therefore tolerate repeated calls for the same epoch.
# Arbitrary dataset decisions made by an external ``set_epoch`` call are not
# serialized; the dataset must persist those decisions itself when they are
# needed for recovery.
#
# For direct dataset iteration, ``create_iterator(skip_n_items=N)`` counts native
# yields: samples, or dataset-side batches when batching is enabled. It does not
# count optimizer steps. The separate ``iter(batch_size)`` batching interface
# remains available.
#

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
