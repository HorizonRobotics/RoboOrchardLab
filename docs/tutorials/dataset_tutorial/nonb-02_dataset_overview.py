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

"""Dataset Overview
==========================================================

**RoboOrchard Dataset** is not a single file but a directory with a specific,
standardized structure. This design is the key to its performance,
separating high-frequency sensor data from low-frequency, queryable metadata.

A typical **RoboOrchard Dataset** dataset is organized on disk as follows:

.. code-block:: text

    <dataset_name>/
    |---- data-0000-of-00xx.arrow
    |---- ...
    |---- meta_db.duckdb
    |---- dataset_info.json
    |---- state.json

This structure is visually explained by our core architecture diagram.

.. figure:: ../../../_static/dataset/ro_dataset_arch.png
   :align: center
   :alt: RoboOrchard Dataset Architecture Diagram
   :width: 100%

   The high-level architecture of the RoboOrchard dataset format, showing the relationship between data components and the surrounding ecosystem.

As the diagram illustrates, the format is built on two primary components

1. Frame Dataset / HF Dataset (The ``data-*.arrow`` files)
----------------------------------------------------------------------------------------

This is where the high-frequency, time-series sensor data lives.

We use a set of Apache Arrow files (sharded as ``data-*.arrow``) to store all
frame-level data. This makes the dataset natively compatible with
`Hugging Face dataset <https://huggingface.co/docs/datasets/index>`__ ,
enabling zero-copy reads and extreme I/O efficiency for training.

As shown in the "Table" view, this is a large, 2D table where each row can be
thought of as a "timestep". It contains lightweight indices
(like episode_index, task_index, robot_index) and the actual sensor data payloads
(like image_data_0, joint_data_1).

Our design natively supports multi-frequency data.
A single "row" in this table can contain a chunk of high-frequency data
(e.g., 10 joint states) that occurred between two lower-frequency events
(e.g., two camera frames).

The ``dataset_info.json`` and ``state.json`` files are simple configuration
files that store the dataset's schema, features, version, and other static information.

2. Meta Database (The ``meta_db.duckdb`` file)
----------------------------------------------------------------------------------------

This file managing all non-frame metadata.

Instead of slow, hard-to-parse JSON files, we use a single embedded DuckDB database.
This database contains normalized tables for all relational metadata.
As shown in the diagram, this includes an Episode Table, Task Table, Robot Table, Instruction Table, etc.

The Frame Dataset does not duplicate this information.
It only holds lightweight indices (e.g., task_uuid_0).
As the dotted arrows show, these indices are foreign keys that point
to the corresponding entries in the Meta Database.

3. Why this design is powerful
----------------------------------------------------------------------------------------

This separation allows you to perform incredibly fast and complex queries on the
small ``meta_db.duckdb`` file first, without ever touching the massive ``Arrow`` files.

For example, you can use a SQL query to find "all episodes from robot_uuid_1 that completed the task_uuid_0 and lasted longer than 10 seconds."
This query runs almost instantly. The result gives you the exact episode_index values you need, allowing you to load only the specific data required for training from the ``data-*.arrow`` files.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'
