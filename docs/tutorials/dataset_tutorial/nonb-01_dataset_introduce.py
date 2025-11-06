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

"""Why RoboOrchard Dataset
==========================================================

The Embodied AI Data Challenge
----------------------------------------

A core challenge in Embodied AI is the data itself.
Unlike simple images or text, robotics data is a complex, time-series data of **multi-modal** and **multi-frequency** signals.
A single "timestep" might involve multiple camera feeds, low-frequency task instructions,
high-frequency joint states, and asynchronous force-torque readings.

To understand this, look at the following panel from the `DROID dataset <https://droid-dataset.github.io/>`__,
visualized in Foxglove. You can click `here <https://app.foxglove.dev/~/view?ds=foxglove-sample-stream&ds.recordingId=rec_0dtkuuK43PadKny8&ds.overrideLayoutId=f1366b1a-0e21-4c96-95f8-570a7325cb1f>`__
to visualize the full data.

.. figure:: ../../../_static/dataset/droid_sample.png
   :align: center
   :alt: Sample of the Droid Dataset
   :width: 100%

   A single timestamp of the droid dataset.

The image above isn't just a video, it's a recording of many
distinct data streams. Let's define the key concepts you are seeing

Metadata: Episode, Task, Robot and Instruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the **high-level**, **static** and **low-frequency** context for
the recording:

* **Episode**: The entire recording itself, representing one
  complete attempt at a task. A dataset is a collection of
  thousands of episodes.

* **Robot**: The 3D model in the center. This is the physical
  embodiment (e.g., ``Franka Emika Panda``) and its configuration (e.g. `URDF <https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html>`__).
  This metadata is complex but rarely changes.

* **Task**: The high-level goal (e.g., ``stack_block``).

* **Instruction**: A language command for this episode (e.g., ``Place the red block on the blue one.``).

Frame Data: Time-Series Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the continuous stream of
sensor readings and actions with highly varied characteristics:

* **High-Frequency, Low-Bandwidth Data**:
  Look at the **time-series plots on the right** (e.g., "Torque 1",
  "Torque 2", etc.). This is data from `JointState` messages,
  streaming at **100Hz or even 1000Hz**. Each timestep
  is small (a few floats), but there are many of them. The **TF
  transforms** (colorful axes in the 3D view) are also
  high-frequency.

* **Low-Frequency, High-Bandwidth Data**: Look at the **camera feeds at the bottom**.
  These are depth or RGB images, streaming at a **lower frequency** (e.g., 10-30Hz).
  Each frame is small in number but **massive in storage size**.
  This high-bandwidth data (video) typically dominates the
  dataset's total size.

The challenge, as demonstrated by the image, is that this data is a mix of:

1. Queryable static **Metadata** (Robot, Task).

2. **High-Bandwidth**, large-storage streams (Images).

3. **High-Frequency**, small-payload streams (Joints, Torques).

Community Solutions
----------------------------------------

The data you just saw is collection data. It is typically stored in formats like
`ROS bags <https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html>`__ or
`MCAP <https://mcap.dev/>`__,
optimized for high-fidelity, streaming recording.

However, for model training, we need training data that is optimized for high-speed **random access**, **batching**,
and **seamless integration with deep learning frameworks** like `PyTorch <https://pytorch.org/>`__
and `Hugging Face datasets <https://huggingface.co/docs/datasets/index>`__.

This creates a critical gap. The community has tried to bridge this,
most notably with the `LeRobot Dataset <https://docs.phospho.ai/learn/lerobot-dataset>`__ format.
While a valuable standard, it introduces several key limitations when faced with complex data like the stream above:

1. **Lossy Compression for Visual Data**: LeRobot encodes dense visual
data like RGB images and **depth maps** into standard video files
(e.g., `.mp4`). Common video codecs are inherently **lossy**, meaning
pixel-level information is lost during compression. This can be
problematic for tasks requiring high data fidelity, especially
with depth or segmentation data.

2. **Limited Sensor Flexibility**: The format is heavily optimized for
**cameras** and standard robot states. Its reliance on video encoding
makes it difficult to natively integrate other crucial sensor
modalities common in robotics, such as **LiDAR point clouds** or
**tactile sensor arrays**, which don't fit the video paradigm well.
(See community discussion in `LeRobot GitHub Issue #1144 <https://github.com/huggingface/lerobot/issues/1144>`__).

3. **No Native Multi-frequency Support**: Its flat-table structure cannot natively store multi-frequency signals,
forcing users to downsample (losing data) or create sparse, bloated tables.

4. **Crude Metadata**: Using flat JSON files for metadata is slow and scales poorly.
It is impossible to query or filter large datasets without loading them entirely.

5. **Performance Overhead**: LeRobot uses **Parquet** for tabular data.
While efficient, the Hugging Face `datasets` library requires
converting Parquet to **Apache Arrow** format upon loading,
introducing significant time and disk space overhead, especially
for large datasets.

6. **Loss of Visualization Fidelity**: The conversion process often breaks the round-trip back to powerful
visualizers like Foxglove.

This is why we created the **RoboOrchard Dataset**. It is designed specifically to solve these
challenges and bridge the gap without compromise.

Introduce the RoboOrchard Dataset format
----------------------------------------------------

Our format is built on two key components:

* Native `Apache Arrow <https://arrow.apache.org/docs/>`__ for **high-performance**,
  **zero-copy** and **customize data types** frame data (fully `Hugging Face dataset <https://huggingface.co/docs/datasets/index>`__ compatible).

* An embedded `DuckDB <https://duckdb.org/docs/stable/>`__ database that replaces crude JSON files with a
  powerful SQL engine for all **metadata** (episodes, tasks, instructions, etc.).

This design provides:

1. **High Performance**: Native Arrow ensures maximum I/O speed for training with zero conversion delay.

2. **Data Fidelity & Flexibility**: Preserves original sensor data (lossless options, multi-frequency support) and allows for easier integration of diverse sensor types like LiDAR or tactile sensors.

3. **Powerful Querying**: Use SQL to filter your dataset before loading.

4. **Full Visualization Fidelity**: Seamlessly export any episode back to `MCAP <https://mcap.dev/>`__ for debugging in `Foxglove <https://docs.foxglove.dev/>`__, preserving the rich experience you see above.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'
