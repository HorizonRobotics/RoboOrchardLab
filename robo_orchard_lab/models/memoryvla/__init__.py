# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""MemoryVLA perceptual-cognitive memory, ported for HoloBrain.

Source: https://github.com/shihao1895/MemoryVLA @ 0eef5c3 (MIT),
arXiv:2508.19236v2. See ``docs_analysis/memoryvla/`` for the anatomy, the
interface differences against this host, and the reference values the port is
checked against.
"""

from robo_orchard_lab.models.memoryvla.memory_bank import (
    BottleneckSE,
    CogMemBank,
    CrossTransformerBlock,
    GateFusion,
    PerMemBank,
    TimestepEmbedder,
)
from robo_orchard_lab.models.memoryvla.sampler import (
    MemoryVLAEpisodeStreamBatchSampler,
    assert_episode_stream_wired,
)
from robo_orchard_lab.models.memoryvla.wrapper import MemoryVLAMemory

__all__ = [
    "BottleneckSE",
    "CogMemBank",
    "CrossTransformerBlock",
    "GateFusion",
    "PerMemBank",
    "TimestepEmbedder",
    "MemoryVLAMemory",
    "MemoryVLAEpisodeStreamBatchSampler",
    "assert_episode_stream_wired",
]
