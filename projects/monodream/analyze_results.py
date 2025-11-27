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

# This file was originally copied from the [NaVid-VLN-CE] repository:
# https://github.com/jzhzhang/NaVid-VLN-CE
# Modifications have been made to fit the needs of this project.

import argparse
import json
import math
import os


def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
)
args = parser.parse_args()

jsons = os.listdir(os.path.join(args.path, "log"))
succ = 0
spl = 0
distance_to_goal = 0
path_length = 0
oracle_succ = 0
for j in jsons:
    with open(os.path.join(args.path, "log", j)) as f:
        try:
            data = json.load(f)
            succ += check_inf_nan(int(data["success"]))
            spl += check_inf_nan(data["spl"])
            distance_to_goal += check_inf_nan(data["distance_to_goal"])
            oracle_succ += check_inf_nan(int(data["oracle_success"]))
            path_length += data["path_length"]

        except Exception as e:
            print(e, j)
print(f"Success rate: {succ}/{len(jsons)} ({succ / len(jsons):.3f})")
print(
    f"Oracle success rate: {oracle_succ}/{len(jsons)} "
    f"({oracle_succ / len(jsons):.3f})"
)
print(f"SPL: {spl:.3f}/{len(jsons)} ({spl / len(jsons):.3f})")
print(f"Distance to goal: {distance_to_goal / len(jsons):.3f}")
print(f"Path length: {path_length / len(jsons):.3f}")
