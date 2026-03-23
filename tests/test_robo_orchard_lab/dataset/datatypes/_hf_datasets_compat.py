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

from __future__ import annotations
from typing import Any

import datasets as hg_datasets
from packaging.version import Version

HF_DATASETS_REENCODES_STREAMING_FEATURES = Version(
    hg_datasets.__version__
) >= Version("4.8.3")


def get_generator_example(
    features: Any, example: dict[str, Any]
) -> dict[str, Any]:
    """Return the right generator payload for the installed datasets version.

    For `datasets>=4.8.3`, streaming iteration applies feature encoding again,
    so generator tests must yield the raw example. Older versions expect the
    generator to yield the pre-encoded payload.

    Args:
        features (Any): Hugging Face features object used by the dataset.
        example (dict[str, Any]): Example payload yielded by the generator.

    Returns:
        dict[str, Any]: Raw or pre-encoded example, depending on the installed
        `datasets` version.
    """

    if HF_DATASETS_REENCODES_STREAMING_FEATURES:
        return example
    return features.encode_example(example)
