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

from collections.abc import Iterable
from dataclasses import dataclass

from datasets import Features, Value

from robo_orchard_lab.dataset.robot import RODataset
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
)


@dataclass(frozen=True, slots=True)
class _EmptyEpisode(EpisodePackaging):
    def generate_episode_meta(self) -> EpisodeMeta:
        return EpisodeMeta(episode=EpisodeData())

    def generate_frames(self) -> Iterable[DataFrame]:
        return iter(())


def test_dataset_packaging_publishes_an_empty_typed_dataset(tmp_path):
    target = tmp_path / "dataset"
    features = Features({"value": Value("int64")})

    DatasetPackaging(features).packaging([_EmptyEpisode()], target)

    dataset = RODataset(target)
    assert len(dataset) == 0
    assert dataset.features["value"] == features["value"]
