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
# For shuffled mixtures, native sampling scopes the private stream to both
# global rank and worker identity. ``IterableDatasetShard`` fallback omits rank
# from corresponding workers' private sampling identities before positional
# assignment; different workers still have distinct scopes. Fallback requires
# matching source/worker configuration and sample order across ranks, including
# matching seed/epoch or corresponding explicit-generator states. Binding does
# not synchronize unequal starting states. Sampling scope is separate from
# PyTorch worker initialization and transform/collator randomness.
# Ordered mixing preserves source order without random scheduling choices.
# Dispatcher keeps its existing distribution behavior without a new
# dispatcher-specific sampling guarantee.
#
# For managed native/fallback sampling, before a new full traversal call
# ``prepared_loader.set_epoch(epoch)``; ``HookBasedTrainer`` does this
# automatically. For an unprepared loader, use ``loader.dataset.set_epoch``.
# With the default mixture strategy, seeded complete traversals replay from
# that epoch's start, including persistent workers. Explicit generators retain
# their advancing state.
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
# Custom mixture strategies also require an explicit epoch-replay declaration
# and matching serialized strategy configuration before worker-prefix recovery
# is supported; see the strategy section below.
# Shuffled data-position states record the strategy identity and scheduling
# revision for compatibility checks. Legacy shuffled states without this
# layout are rejected. Ordered states retain their earlier layout compatibility.
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
# Choosing a mixture sampling strategy
# ------------------------------------
# ``DictIterableDataset(..., sampling_strategy=None)`` uses
# ``DefaultMixtureSamplingStrategyConfig``. For shuffled mixtures, the default
# uses remaining-quota admissions and randomized budgets for the initial finite
# reader residencies. This changes shuffled output order and RNG consumption
# from the earlier default, including for the same seed and loading settings.
# Seeded complete-epoch replay applies to the selected policy and configuration;
# it does not reproduce an earlier scheduling revision. To select the default
# explicitly:
#
# .. code-block:: python
#
#     from robo_orchard_lab.dataset.robot import DictIterableDataset, RODatasetItem
#     from robo_orchard_lab.dataset.robot.sampling_strategy import (
#         DefaultMixtureSamplingStrategyConfig,
#     )
#
#     mixture = DictIterableDataset(
#         datasets=[RODatasetItem(dataset_path="data1", transform=transform)],
#         shuffle=True,
#         sampling_strategy=DefaultMixtureSamplingStrategyConfig(),
#     )
#
# A strategy selects which pending physical source enters the reader pool and
# which active source yields next. The dataset engine still owns row quotas,
# reader admission and cleanup, the concurrency limit, and EOF checks. Source
# shuffle and resampling stay with each ``DatasetItem``. A different strategy
# can change source order without changing full-epoch row quotas.
#
# Pass a strategy configuration to the dataset. Each worker-local mixture
# iteration creates a fresh runtime with its existing scoped generator; a
# strategy does not introduce another seed. Do not share an active runtime
# between iterations or workers. A custom strategy with ``shuffle=False`` is
# rejected during construction. Ordered iteration keeps its existing order
# and makes no random scheduling choices.
#
# To customize source selection, subclass the runtime and its typed config.
# This minimal example always chooses the lowest available physical ID; it is
# an extension example, not a recommendation for balanced training mixtures.
# Put both classes at the top level of an importable module such as
# ``my_project.mixture_strategy``:
#
# .. code-block:: python
#
#     from robo_orchard_core.utils.config import ClassType
#     from robo_orchard_lab.dataset.robot.sampling_strategy import (
#         MixtureSamplingStrategy,
#         MixtureSamplingStrategyConfig,
#         MixtureSamplingView,
#     )
#
#     class FirstSourceStrategy(MixtureSamplingStrategy):
#         def choose_admission(self, view: MixtureSamplingView) -> int:
#             return min(view.pending_ids)
#
#         def choose_active(self, view: MixtureSamplingView) -> int:
#             return min(view.active_ids)
#
#         def residency_completed(self, item_id: int, *, requeue: bool) -> None:
#             pass  # This strategy has no per-residency state to update.
#
#     class FirstSourceStrategyConfig(MixtureSamplingStrategyConfig):
#         class_type: ClassType[FirstSourceStrategy] = FirstSourceStrategy
#
# The view contains read-only candidates, logical membership and output-row
# counts. ``targets`` and ``logical_indices`` use physical IDs as indices;
# ``logical_names`` uses logical IDs. ``remaining_targets`` and
# ``segment_remaining`` are live mappings: copy values if you need a historical
# observation. ``residence_rows`` refreshes with admission metadata. Always
# return an ID from the current method's candidate tuple.
# ``residency_completed`` runs after successful EOF, reader cleanup and progress
# commit; ``requeue=True`` means the unfinished source will become pending again.
# Failure and early close do not send this completion notification.
#
# The optional ``initial_residency_rows(item_id, *, rows, batch_size)`` hook
# requests a raw row budget for an initial finite residency. Its inherited
# implementation returns ``rows`` unchanged, so a custom strategy can keep the
# three methods in the example above. ``rows`` is the original cursor budget,
# including any original final tail, rather than a new full-epoch quota.
# A non-identity request must be a positive integer no larger than ``rows``;
# when ``rows >= batch_size``, it must contain whole batches. If the original
# ``rows`` is smaller than one batch, it stays unchanged. Returning the original
# ``rows`` also retains its existing tail behavior. The framework validates
# and applies the request; strategies do not own cursors or commit progress.
#
# ``view.initial_residency_slots`` exposes the initial pool size while this
# phase is available and is zero afterward. Only shuffled native offload uses
# the hook, for initial candidates with positive output. Ordered iteration and
# mixtures without offload do not request initial budgets.
#
# Use the existing ``ClassConfig`` JSON path to save the typed configuration.
# Both classes must remain importable when loading it, including in worker
# processes. For example, in your training code:
#
# .. code-block:: python
#
#     from my_project.mixture_strategy import FirstSourceStrategyConfig
#
#     cfg = FirstSourceStrategyConfig()
#     cfg_json = cfg.to_str(format="json")
#     restored_cfg = FirstSourceStrategyConfig.from_str(cfg_json, format="json")
#     mixture = DictIterableDataset(
#         datasets=[
#             RODatasetItem(dataset_path="data1", transform=transform),
#             RODatasetItem(dataset_path="data2", transform=transform),
#         ],
#         shuffle=True,
#         sampling_strategy=restored_cfg,
#     )
#
# Custom configs inherit ``supports_epoch_replay=False``, so their recovery
# capability is unknown by default. Set it to ``True`` only when rebuilding the
# runtime from the serialized config, injected generator and view reproduces
# every source decision. Include every decision-affecting option in the config;
# external state or additional random streams are not saved by this flag.
# The declaration permits the existing worker-prefix recovery checks, which
# also compare strategy identity and configuration. It does not make offload,
# sample prefetch, explicit dataset generators or other unsupported recovery
# modes exact. The default strategy config already declares replay support.
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
# dividing it into finite reader segments. With the default shuffled policy,
# initial resident readers can receive shorter first segments. The runtime
# reserves one set of uniform draws for the initial pool. Sample mode chooses
# a budget from one row through the original first-segment budget; dataset-side
# batching chooses complete native batch yields and retains a budget smaller
# than one batch. Later residencies use the existing cap and tail merging.
# A small remainder can merge into the last segment, so 100 batches is a
# target, not a strict maximum. Existing ``drop_last`` rules apply only to the
# original final tail; reader rotation does not introduce extra dropped rows.
#
# Offload also works with ``shuffle=False`` and with sample-level loading
# (``use_dataset_side_batching=False``). With this repository's DataLoader,
# sample-level segment size uses its actual batch size; direct iteration or
# an external loader without this context counts one sample per unit.
#
# The default strategy treats each input item or group as one logical source.
# For each worker, T is a physical item's fixed resampled output-row target,
# and Q is the sum of T over its logical group. The first initial reader's
# logical source is sampled in proportion to Q; subsequent initial slots
# sample uniformly among logical sources not yet represented when available.
# Normal admissions also prefer logical sources absent from the active pool
# when possible. For an eligible logical source, R includes all remaining
# output rows of its active and pending members. Let r_i be a pending member's
# remaining rows and B_i its predicted next-residency output rows. The mean
# residency budget is Bbar = sum(r_i * B_i) / sum(r_i), over pending members.
# Logical admission mass is R / Bbar; a physical member is then selected in
# proportion to its remaining rows. The engine remains the owner of progress.
#
# Active physical weight is R*r/(M*u): R is remaining logical output, r is
# the member's remaining output, M sums r over active members of that logical
# source, and u is the next yield's row count. Weights refresh on admission or
# when the next yield size changes. Inner ``drop_last`` tails do not count
# toward the remaining output; the underlying index plan remains unchanged.
# A completed reader closes before it becomes eligible again, including for
# immediate reselection. The concurrency bound still applies. Finite epoch
# initialization, changing tails and source exhaustion cause prefix variation:
# the complete epoch retains exact quotas, not IID finite-window proportions.
# Some tail windows can have larger source-proportion errors under this
# scheduling choice. The policy provides no exact per-batch ratio or training
# quality, throughput or memory improvement guarantee.
# Ordered iteration retains its existing admission and delivery order.
# Grouping changes the fairness unit and can change shuffled delivery order.
# Changing strategy or scheduling revision can change shuffled output order
# and RNG consumption.
#
# ``offload_num_batches=None`` keeps ordinary iteration. Its full-source
# residences use the logical admission policy without finite reader rotation
# or randomized initial budgets.
# Missing logical sources in the active pool can still cause skewed windows;
# the scheduler does not evict readers or guarantee finite-window proportions.
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
