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

import json


def load_episode_records_from_json(json_path: str) -> list[dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")

    records = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {idx}.")

        mcap_path = item.get("mcap_path")
        instruction = item.get("instruction")
        if not isinstance(mcap_path, str) or not mcap_path:
            raise ValueError(
                f"Missing non-empty string mcap_path at index {idx}."
            )
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(
                f"Missing non-empty string instruction at index {idx}."
            )

        records.append(
            {
                "mcap_path": mcap_path,
                "instruction": instruction,
            }
        )

    return records
