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

"""Ecosystem: MCAP and LeRobot Interoperability
==========================================================

.. note::

    This tutorial assumes you have already finish the :ref:`previous tutorial <sphx_glr_build_tutorials_dataset_tutorial_nonb-05_dataset_create.py>`.

A dataset is only as useful as its ability to interact with the wider ecosystem.

Here, we demonstrate how the **RoboOrchard Dataset** handles this:

1.  **Part 1: Exporting the RoboOrchard Dataset to MCAP**: We show how to
    convert an episode from our dataset back into an `.mcap <https://mcap.dev/>`__ file
    for high-fidelity visualization in tools like `Foxglove <https://docs.foxglove.dev/>`__.
    This is a runnable example.

2.  **Part 2: Ingesting MCAP to RoboOrchard Dataset**: We show a
    template script for the most common use case: converting a
    large collection of `.mcap` files (raw data) into our optimized
    dataset format for training.

3.  **Part 3: LeRobot Conversion**: A note on interoperability
    with other dataset formats.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Setup and Imports
# --------------------------------
#

import os
import pprint
from typing import Any, Generator

import datasets as hg_datasets
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchJointsState,
    ImageMode,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_decoder import (
    McapBatch2BatchJointStateConfig,
    McapBatchDecoderConfig,
    McapBatchDecoders,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchFromBatchJointStateConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import McapReader
from robo_orchard_lab.dataset.experimental.mcap.writer import Dataset2Mcap
from robo_orchard_lab.dataset.robot import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    RobotData,
    RODataset,
    TaskData,
)
from robo_orchard_lab.dataset.robot.conversion.lerobot_dataset import (
    LerobotDatasetEpisodePackaging,
    get_hg_features,
)

# Define paths
SOURCE_DATASET_PATH = ".workspace/dummy_dataset/"
OUTPUT_MCAP_PATH = ".workspace/conversion/mcap/dummy_dataset_episode0.mcap"
MCAP_INGEST_DATASET_PATH = ".workspace/conversion/mcap/dataset/"

# %%
# Mcap conversion
# --------------------------------
#

# %%
# Part 1: Exporting dataset to MCAP (for Visualization)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This is the "Dataset -> MCAP" workflow. It's useful for
# debugging, visualization, and sharing.
#
# As the architecture diagram below illustrates, this entire
# process is orchestrated by the :py:class:`~robo_orchard_lab.dataset.experimental.mcap.writer.Dataset2Mcap` class.
#
# .. figure:: ../../../_static/dataset/ro_dataset_2_mcap.png
#    :align: center
#    :alt: Data flow diagram for exporting to MCAP.
#    :width: 100%
#
#    The :py:class:`~robo_orchard_lab.dataset.experimental.mcap.writer.Dataset2Mcap` class reads the dataset's native
#    stream and uses a configurable :py:class:`~robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base.McapBatchEncoder`
#    to adapt the schema and write to an MCAP file.
#
# The :py:class:`~robo_orchard_lab.dataset.experimental.mcap.writer.Dataset2Mcap`
# class reads the `Timeline Message Streams` (with its native `dataset schema`) and feeds it into an
# :py:class:`~robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base.McapBatchEncoder`,
# which acts as the **Schema Adaptor**.
#
# **Our job is to provide the configuration for this adaptor.**
# This configuration is a Python `dict` (`encoder_cfg`) that maps our
# dataset's feature names (e.g., `"joints"`) to a specific
# :py:class:`~robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base.McapBatchEncoderConfig` class (e.g.,
# :py:class:`~robo_orchard_lab.dataset.experimental.mcap.batch_encoder.joint_state.McapBatchFromBatchJointStateConfig`).
# This config tells the adaptor which MCAP topic (e.g., `"/joints"`) to
# write the data to.
#
# Let's walk through this step-by-step.
#

# 1. Load the dataset we created in the previous tutorial
print(f"Loading source dataset from: {SOURCE_DATASET_PATH}")
dataset = RODataset(SOURCE_DATASET_PATH)

# 2. Define the Encoder Configuration (`encoder_cfg`)
#    This dict is the configuration for the `McapBatchEncoder`.
#    The keys MUST match the feature names in our dataset.
#    For this demo, we will only export the "joints" feature.
encoder_cfg: dict[str, McapBatchEncoderConfig] = {
    "joints": McapBatchFromBatchJointStateConfig(target_topic="/joints"),
}

# 3. Instantiate the main conversion utility: `Dataset2Mcap`
dataset2mcap_writer = Dataset2Mcap(dataset=dataset)

# 4. Save the first episode (index 0) to an .mcap file
#    We pass our `encoder_cfg` to the `save_episode` method.
#    The `Dataset2Mcap` instance will use this config to
#    properly initialize its internal `McapBatchEncoder`.
print(f"\nSaving episode 0 to {OUTPUT_MCAP_PATH}...")
dataset2mcap_writer.save_episode(
    target_path=OUTPUT_MCAP_PATH, episode_index=0, encoder_cfg=encoder_cfg
)

print("\n--- Export Complete! ---")
print(f"You can now open '{OUTPUT_MCAP_PATH}' in Foxglove.")

# %%
# Part 2: Ingesting MCAP to RoboOrchard Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This is the "MCAP -> Dataset" workflow, it is the same packaging pattern we learned in the
# :ref:`previous tutorial <sphx_glr_build_tutorials_dataset_tutorial_nonb-05_dataset_create.py>`.
#

# define the schema for our data
DATASET_FEATURES = hg_datasets.Features(
    {
        "joints": BatchJointsState.dataset_feature(),  # type: ignore
    }
)

# define the mcap schema decoder config
decoder_cfg: dict[str, McapBatchDecoderConfig] = {
    "joints": McapBatch2BatchJointStateConfig(source_topic="/joints")
}


# Implement EpisodePackaging
class McapEpisodePackager(EpisodePackaging):
    """An `EpisodePackaging` implementation that pre-processes a full MCAP file before generating frames."""

    def __init__(
        self,
        mcap_path: str,
        decoder_config: dict[str, McapBatchDecoderConfig],
        monitor_topic: str,
    ):
        self.mcap_path = mcap_path
        self.decoder_config = decoder_config
        self.monitor_topic = monitor_topic

        self.mock_robot = RobotData("robot_from_mcap", "<urdf>...")
        self.mock_task = TaskData("task_from_mcap", "...")

    def generate_episode_meta(self) -> EpisodeMeta:
        """Returns the static metadata for this entire episode.

        .. note::

            In a real implementation, you might open the MCAP
            briefly here to read a metadata topic).
        """
        print(f"\n  [{self.mcap_path}] Generating episode metadata...")
        return EpisodeMeta(
            episode=EpisodeData(), robot=self.mock_robot, task=self.mock_task
        )

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """A streaming generator that reads, decodes, and yields frames from the MCAP file one batch at a time."""
        print(f"  [{self.mcap_path}] Starting frame generation stream...")

        # 1. Instantiate the decoders
        decoders = McapBatchDecoders(self.decoder_config)

        # 2. Instantiate the MCAP reader
        with open(self.mcap_path, "rb") as f:
            reader = McapReader.make_reader(f)

            # 3. Create a message batch iterator
            #    We'll batch by 1 message on our monitor topic
            batch_splitter = SplitBatchByTopics(
                SplitBatchByTopicArgs(
                    monitor_topic=self.monitor_topic,
                    min_messages_per_topic=1,
                    max_messages_per_topic=1,
                )
            )

            # 4. Iterate, Decode, and Yield
            #    This loop reads one batch at a time, processes it,
            #    yields it, and then discards it from memory
            #    before reading the next one.
            with McapDecoderContext() as msg_decoder_ctx:
                for batch in iter_messages_batch(
                    reader, batch_split=batch_splitter
                ):
                    if batch.is_last_batch and len(batch) == 0:
                        continue

                    # This call is the "Schema Adaptor" in action
                    # It converts a batch of MCAP messages into
                    # a dict of features (e.g., {"robot_joints": ...})
                    decoded_features = decoders(
                        batch, msg_decoder_ctx=msg_decoder_ctx
                    )

                    # 5. Yield the `DataFrame`
                    yield DataFrame(
                        features=decoded_features,
                        instruction=None,
                        timestamp_ns_min=batch.min_log_time,
                        timestamp_ns_max=batch.max_log_time,
                    )

        print(f"\n  [{self.mcap_path}] Finished generating frames.")


print("\n--- Starting MCAP Ingestion Process ---")

# 1. Create an iterable of our `McapEpisodePackager` instances
episodes_to_package = [
    McapEpisodePackager(
        mcap_path=OUTPUT_MCAP_PATH,
        decoder_config=decoder_cfg,
        monitor_topic="/joints",
    )
]

# 2. Initialize the main packager with our *target* schema
packager = DatasetPackaging(
    features=DATASET_FEATURES, database_driver="duckdb", check_timestamp=True
)

# 3. Run the packaging process!
packager.packaging(
    episodes=episodes_to_package,
    dataset_path=MCAP_INGEST_DATASET_PATH,
    force_overwrite=True,
)

print("--- Ingestion Complete! ---")

# %%
# Lerobot Dataset Conversion
# --------------------------------
#
# This section details our interoperability strategy with `LeRobot Dataset <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>`__.
#
# 1. We **provide** a clear path for **ingesting** data from LeRobot Dataset to RoboOrchard Dataset.
#
# 2. We **do not** provide a utility for **exporting** data from RoboOrchard Dataset to LeRobot Dataset.
#
# This one-way conversion is a deliberate design choice based on the
# fundamental limitations of the LeRobot format, which we discussed
# in the :ref:`Community Solutions Overview <sphx_glr_build_tutorials_dataset_tutorial_nonb-01_dataset_introduce.py>`.
#
# Exporting our high-fidelity format back to LeRobot Dataset would force
# us to accept data degradation and loss of flexibility.
#

# %%
# Part 1: Ingesting Lerobot Dataset to RoboOrchard Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The process for ingesting a `LeRobot Dataset <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>`__
# follows the exact same packaging pattern as our MCAP ingestion.
#
# 1.  Define a **Target Schema**: We define what we want our
#     new RoboOrchard dataset to look like (e.g., using
#     `BatchJointsStateFeature`).
#
# 2.  Define a **Transform Function**: This function will read the
#     *raw* LeRobot frame (a dict of Tensors or string) and "transform" it
#     to match our target schema (e.g., merging columns into a
#     :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState` object).
#
# 3.  Run the :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging` engine.
#
# .. note::
#
#    This example requires lerobot to be installed. (``pip install lerobot``)
#

# %%
# Example A: Using default conversion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This example uses the default behavior. We pass ``transform=None``.
# The :py:class:`~robo_orchard_lab.dataset.robot.conversion.lerobot_dataset.LerobotDatasetEpisodePackaging`
# will use the :py:class:`~robo_orchard_lab.dataset.robot.conversion.lerobot_dataset.DefaultTransform`,
# and the :py:func:`~robo_orchard_lab.dataset.robot.conversion.lerobot_dataset.get_hg_features` function will
# define the schema using `hg_datasets.features.Image()`.
#
# This creates a standard, simple Hugging Face dataset.
#

TARGET_LEROBOT_CONVERT_PATH = (
    ".workspace/conversion/lerobot/default_ts/dataset"
)

EPISODE_IDX = list(range(2))

os.environ["HF_DATASETS_CACHE"] = os.path.join(
    TARGET_LEROBOT_CONVERT_PATH, ".cache"
)

lerobot_dataset = LeRobotDataset(
    "yaak-ai/L2D-v3",
    episodes=EPISODE_IDX,
    video_backend="pyav",  # for compatible purpose
)

target_features = get_hg_features(lerobot_dataset)
print("Target Schema (features):\n{}".format(pprint.pformat(target_features)))

# %%
# Create the list of packagers
episodes_to_package = []
for ep_idx in EPISODE_IDX:
    # We access the metadata by index
    episode_meta = lerobot_dataset.meta.episodes[ep_idx]
    episodes_to_package.append(
        LerobotDatasetEpisodePackaging(
            lerobot_dataset,
            episode_meta,
            max_frames=2,  # for demo purpose
        )
    )

ro_dataset_packer = DatasetPackaging(
    features=target_features, check_timestamp=True
)
ro_dataset_packer.packaging(
    episodes=episodes_to_package,
    dataset_path=TARGET_LEROBOT_CONVERT_PATH,
    max_shard_size="100MB",
    force_overwrite=True,
)

# %%
# Example B: Using custom conversion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This example uses the `transform` argument to upgrade the
# data structure during ingestion. We will convert LeRobot's
# image tensors into our custom :py:class:`~robo_orchard_core.datatypes.camera.BatchCameraData` objects
#


TARGET_LEROBOT_CONVERT_PATH = ".workspace/conversion/lerobot/custom/dataset"

EPISODE_IDX = list(range(2))

os.environ["HF_DATASETS_CACHE"] = os.path.join(
    TARGET_LEROBOT_CONVERT_PATH, ".cache"
)

lerobot_dataset = LeRobotDataset(
    "yaak-ai/L2D-v3",
    episodes=EPISODE_IDX,
    video_backend="pyav",  # for compatible purpose
)

target_features = get_hg_features(lerobot_dataset)
# upgrade all camera data features to BatchCameraDataFeature
for key in lerobot_dataset.meta.camera_keys:
    if key in target_features:
        target_features[key] = BatchCameraData.dataset_feature()
print("Target Schema (features):\n{}".format(pprint.pformat(target_features)))


# %%
#
class FrameDataAdaptor:
    def __init__(self, camera_keys: list[str]):
        self.camera_keys = camera_keys

    def __call__(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        new_frame_data = dict()

        for key, value in frame_data.items():
            if key in self.camera_keys:
                # Note: Lerobot dataset will normalize the image data
                value = (value * 255).type(torch.uint8)
                new_frame_data[key] = BatchCameraData(
                    sensor_data=value[None, ...], pix_fmt=ImageMode.RGB
                )
            else:
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                new_frame_data[key] = value

        return new_frame_data


# Create the list of packagers
episodes_to_package = []
for ep_idx in EPISODE_IDX:
    # We access the metadata by index
    episode_meta = lerobot_dataset.meta.episodes[ep_idx]
    episodes_to_package.append(
        LerobotDatasetEpisodePackaging(
            lerobot_dataset,
            episode_meta,
            max_frames=2,  # for demo purpose
            transform=FrameDataAdaptor(
                camera_keys=lerobot_dataset.meta.camera_keys
            ),
        )
    )

ro_dataset_packer = DatasetPackaging(
    features=target_features, check_timestamp=True
)
ro_dataset_packer.packaging(
    episodes=episodes_to_package,
    dataset_path=TARGET_LEROBOT_CONVERT_PATH,
    max_shard_size="100MB",
    force_overwrite=True,
)

# %%
#
ro_dataset = RODataset(TARGET_LEROBOT_CONVERT_PATH)
sample = ro_dataset[0]
for key, value in sample.items():
    if key in lerobot_dataset.meta.camera_keys:
        print("key = {}, value type = {}".format(key, type(value)))
