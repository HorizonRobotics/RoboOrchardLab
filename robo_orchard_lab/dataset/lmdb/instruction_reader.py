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
import logging
from typing import Optional

import numpy as np

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb

logger = logging.getLogger(__name__)


class InstructionReader:
    def __init__(
        self,
        lmdb_path: str = None,
        instruction_path: Optional[str] = None,
        encoding_mode: str = "utf-8",
    ):
        if lmdb_path is not None:
            self.lmdb = Lmdb(
                lmdb_path, writable=False, encoding_mode=encoding_mode
            )
        else:
            self.lmdb = None
        if instruction_path is None:
            self.task2instruction = {}
        else:
            self.task2instruction = json.load(
                open(instruction_path, "r", encoding=encoding_mode)
            )

    def get_instruction(
        self, uuid, task_name=None, step_index=None, return_subtask=True
    ):
        if self.lmdb is not None:
            task_info = self.lmdb[f"{uuid}/task_info"]
        else:
            task_info = None

        if task_name is None:
            task_name = uuid.split("/")[0]

        output = {}
        if task_info is not None and "instruction" in task_info:
            instructions = task_info["instruction"]
            if instructions is None:
                instructions = task_info["task_name"]
        elif task_name in self.task2instruction:
            instructions = self.task2instruction[task_name]
        else:
            logger.warning(f"No instruction - uuid: {uuid}, task: {task_name}")
            instructions = ""

        if instructions is None:
            logger.warning(
                f"Instruction is None - uuid: {uuid}, task: {task_name}"
            )
            instructions = ""

        if isinstance(instructions, str):
            text = instructions
        elif len(instructions) == 0:
            text = ""
        else:
            idx = np.random.randint(len(instructions))
            text = instructions[idx]
        output["text"] = text

        if return_subtask:
            output["subtask"] = self.get_subtask_by_frame_idx(
                task_info, step_index
            )
        return output

    def get_subtask_by_frame_idx(self, task_info, step_index):
        if task_info is None:
            return ""

        subtasks = task_info["subtasks"]
        if len(subtasks) == 0:
            return ""

        if step_index >= subtasks[-1]["end_frame"]:
            return subtasks[-1]["description"]

        if step_index < subtasks[0]["start_frame"]:
            return subtasks[0]["description"]

        for subtask in subtasks:
            if subtask["start_frame"] <= step_index < subtask["end_frame"]:
                return subtask["description"]

        return ""
