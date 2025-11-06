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

"""Creating a RoboOrchard Dataset (Data Packaging)
==========================================================

This tutorial covers how to package your own raw robotics data into the
**RoboOrchard Dataset** (Arrow + DuckDB) format.

This process is essential for converting data from sources like MCAP files,
ROS bags, or other custom logs into a format optimized for training.

We will cover:

1.  **The Core Concepts**: Understanding :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging` and the
    :py:class:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging` abstract class.

2.  **Step 1: Defining your Schema**: Using the `huggingface datasets features <https://huggingface.co/docs/datasets/about_dataset_features>`__
    to define your data schema.

3.  **Step 2: Implementing EpisodePackaging**: Creating a class that
    knows how to generate your episode metadata and frame data.

4.  **Step 3: Running the Packager**: Using :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging` to
    write the dataset to disk.

5.  **Step 4: Verifying the Output**: Using :py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset`
    to load and inspect the dataset we just created.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Core Concepts and Architecture
# --------------------------------
# The packaging API is designed around two main classes. Your code
# implements **what to write**, and our engine handles how to write it.
#
# * :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging`:
#   The main orchestrator class. You initialize
#   it with your dataset's `schema (a.k.a features) <https://huggingface.co/docs/datasets/about_dataset_features>`__,
#   and it handles all the complexities of writing to Arrow files, de-duplicating metadata,
#   and building the DuckDB database.
#
# * :py:class:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging`:
#   **An abstract base class that you must subclass.**
#   An instance of your subclass represents **one episode of data**. You must implement two methods:
#
#   1.  :py:meth:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging.generate_episode_meta()`:
#       Returns static metadata for the entire episode (e.g., which robot was used, what task was
#       performed).
#
#   2.  :py:meth:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging.generate_frames()`:
#       A Python **generator** that `yields` one :py:class:`~robo_orchard_lab.dataset.robot.packaging.DataFrame`
#       object at a time for the entire episode.
#
# The :py:meth:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging.packaging()`
# method takes an **iterable** (like a list) of your :py:class:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging`
# instances and processes them.
#
# The entire process is visualized below.
#
# .. figure:: ../../../_static/dataset/ro_dataset_packaging_arch.png
#    :align: center
#    :alt: Data packaging flow diagram
#    :width: 100%
#
#    The DatasetPackaging engine's data flow, ignoring how to packaging the `state.json` and `dataset_info.json`
#
# As the diagram illustrates, when your :py:class:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging`
# instance is processed, the engine automatically performs the core data separation:
#
# * **Metadata** (from :py:meth:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging.generate_episode_meta`) is processed,
#   de-duplicated, and written to the **DuckDB** database.
#
# * **DataFrame** (from :py:meth:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging.generate_frames`) is serialized, injected with
#   the correct indices, and written to the **Shard Arrow Files**.
#
# Finally, the engine writes the ``dataset_info.json`` and ``state.json``
# files to complete the dataset.
#
# **Your only job is to implement the two generator methods.**
#

# %%
# Setup and Imports
# --------------------------------
import os
import pprint
from typing import Generator

import datasets as hg_datasets
import torch

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransformGraph,
    BatchJointsState,
)
from robo_orchard_lab.dataset.datatypes.geometry import BatchFrameTransform
from robo_orchard_lab.dataset.robot import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    RODataset,
    TaskData,
)

OUTPUT_DATASET_PATH = ".workspace/dummy_dataset/"


# %%
# Step 1: Define the Data Schema (Features)
# ---------------------------------------------------------
# First, we must define the schema for our data using the
# `datasets.Features <https://huggingface.co/docs/datasets/about_dataset_features>`__ class.
# This defines all the columns that will be written to the Apache Arrow files.
#
# .. note::
#
#   You do not need to define the reserved index columns
#   (like `episode_index`, `task_index`, etc.). The :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging`
#   class will add those automatically.
#
# For this example, we'll create a simple dataset with joints and a TF graph.
#
DATASET_FEATURES = hg_datasets.Features(
    {
        "joints": BatchJointsState.dataset_feature(),  # type: ignore
        "tf_graph": BatchFrameTransformGraph.dataset_feature(),  # type: ignore
        # You can also include simple types
        "some_string_data": hg_datasets.Value("string"),
    }
)

print("--- Defined Features ---")
pprint.pprint(DATASET_FEATURES)

# %%
# Step 2: Implement EpisodePackaging
# ------------------------------------
# This is the core of the work. We create a class that inherits from
# :py:class:`~robo_orchard_lab.dataset.robot.packaging.EpisodePackaging`
# and teaches it how to generate our specific data.
#
# Let's define some re-usable metadata objects.
# The packager is smart and will de-duplicate these in the database.
# You can re-use the same :py:class:`~robo_orchard_lab.dataset.robot.packaging.RobotData`
# object across many episodes.
#

MOCK_ROBOT = RobotData(name="my_robot_v1", urdf_content="<xml>...</xml>")
MOCK_TASK = TaskData(name="stack_blocks", description="Stack red on blue")

# We can also define instructions that might change frame-by-frame
INST_1 = InstructionData(name="pickup_red", json_content={"goal": "red_block"})
INST_2 = InstructionData(
    name="place_on_blue", json_content={"goal": "blue_block"}
)


class MyCustomEpisode(EpisodePackaging):
    """A custom packager for one episode of our mock data."""

    def __init__(self, episode_id: str, num_frames: int):
        self.episode_id = episode_id
        self.num_frames = num_frames
        print(f"Initialized packager for {self.episode_id}")

    def generate_episode_meta(self) -> EpisodeMeta:
        """Returns the static metadata for this entire episode."""
        print(f"\n  [{self.episode_id}] Generating episode metadata...")
        return EpisodeMeta(
            episode=EpisodeData(),  # Use defaults for a simple episode
            robot=MOCK_ROBOT,
            task=MOCK_TASK,
        )

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """A generator that yields one `DataFrame` at a time."""
        print(f"  [{self.episode_id}] Starting frame generation...")
        for i in range(self.num_frames):
            # 1. Assign a timestamp (optional but highly recommended)
            # Timestamps are in nanoseconds
            current_time_ns = i * 40_000_000  # 40ms = 25Hz

            # 2. Create the features (the actual sensor data)
            # This must match the `DATASET_FEATURES` schema
            features_dict = {
                "joints": BatchJointsState(
                    position=torch.rand(size=(1, 6)),  # 1x6 joint pos
                    timestamps=[current_time_ns],
                ),
                "tf_graph": BatchFrameTransformGraph(
                    tf_list=[
                        BatchFrameTransform(
                            xyz=torch.rand(size=(1, 3)),
                            quat=torch.rand(size=(1, 4)),
                            parent_frame_id="world",
                            child_frame_id="tcp",
                            timestamps=[current_time_ns],
                        ),
                    ],
                ),
                "some_string_data": f"frame_{i}",
            }

            # 3. Assign an instruction (optional)
            # We can change the instruction mid-episode
            current_instruction = (
                INST_1 if i < (self.num_frames / 2) else INST_2
            )

            # 4. Yield the final `DataFrame`
            yield DataFrame(
                features=features_dict,
                instruction=current_instruction,
                timestamp_ns_min=current_time_ns,
                timestamp_ns_max=current_time_ns,
            )
        print(
            f"\n  [{self.episode_id}] Finished generating {self.num_frames} frames."
        )


# %%
# Step 3: Run the Packager
# ------------------------
# Now we use the :py:class:`~robo_orchard_lab.dataset.robot.packaging.DatasetPackaging`
# orchestrator to do the work.

print("--- Starting Packaging Process ---")

# 1. Initialize the main packager with our schema
packager = DatasetPackaging(
    features=DATASET_FEATURES, database_driver="duckdb", check_timestamp=True
)

# 2. Create an iterable of our `EpisodePackaging` instances
#    We will create two episodes for this dataset.
episodes_to_package = [
    MyCustomEpisode(episode_id="ep_001", num_frames=5),
    MyCustomEpisode(episode_id="ep_002", num_frames=3),
]

# 3. Run the packaging process!
#    This will create the `OUTPUT_DATASET_PATH` directory,
#    write the `.arrow` files, and create the `meta_db.duckdb`.
packager.packaging(
    episodes=episodes_to_package,
    dataset_path=OUTPUT_DATASET_PATH,
    max_shard_size="100MB",  # Create new arrow files every 100MB
    force_overwrite=True,
)

print("--- Packaging Complete! ---")

# %%
# Step 4: Verify the Output
# -------------------------
# Let's check the directory structure and then load the dataset
# using :py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset` to prove it worked.
#

print(f"--- Verifying directory tree at {OUTPUT_DATASET_PATH} ---")
# Using os.listdir for a simple tree-like print
for root, _, files in os.walk(OUTPUT_DATASET_PATH):
    level = root.replace(OUTPUT_DATASET_PATH, "").count(os.sep)
    indent = " " * 4 * (level)
    print(f"{indent}{OUTPUT_DATASET_PATH}")
    sub_indent = " " * 4 * (level + 1)
    for f in files:
        print(f"{sub_indent}{f}")

# %%
# Now, let's load it just like any other :py:class:`~robo_orchard_lab.dataset.robot.dataset.RODataset`
print("--- Loading the new dataset with RODataset ---")

new_dataset = RODataset(OUTPUT_DATASET_PATH, meta_index2meta=True)

print("Successfully loaded dataset.")
print(f"Total frames (5 + 3): {len(new_dataset)}")

sample = new_dataset[5]

print("\n--- Inspecting Sample 5 (First frame of 2nd episode) ---")

# Check frame data
print(f"String data: {sample['some_string_data']}")
assert sample["some_string_data"] == "frame_0"  # 0th frame of 2nd ep

# Check resolved metadata
print(f"Robot: {sample['robot'].name}")
assert sample["robot"].name == "my_robot_v1"

print(f"Task: {sample['task'].name}")
assert sample["task"].name == "stack_blocks"

# Check the instruction. Frame 0 of ep_002 (3 frames total)
# is in the first half (i < 1.5), so it should be INST_1.
print(f"Instruction: {sample['instruction'].name}")
assert sample["instruction"].name == "pickup_red"

print("\n--- Verification Successful! ---")
