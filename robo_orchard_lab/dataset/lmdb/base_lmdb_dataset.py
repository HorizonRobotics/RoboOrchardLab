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

import bisect
import logging
import os
from collections import deque
from typing import Callable, ClassVar, List, Optional, Tuple, Union

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.distributed.utils import get_dist_info
from robo_orchard_lab.utils.build import build
from robo_orchard_lab.utils.misc import as_sequence

logger = logging.getLogger(__name__)


MAX_VIRTUAL_MEM = 1024**3 * 64  # 64 TB


def get_virtual_mem():
    with open(f"/proc/{os.getpid()}/statm", "r") as f:
        statm = f.read().split()
    return int(statm[0]) * 4


class BaseIndexData(BaseModel):
    """Base data structure for indexing simulation or task-related information."""  # noqa: E501

    uuid: str
    num_steps: int
    task_name: str = Field(
        validation_alias=AliasChoices("task_name", "task"), default=None
    )
    user: Optional[str] = None
    embodiment: Optional[str] = None
    date: Optional[str] = None
    simulation: bool = False
    error: bool = False

    model_config = ConfigDict(extra="allow")


StepTag = Union[str, List[str]]


class StepLevelTags(BaseModel):
    """Base data structure for step level tags, such as subtask and skill.

    The subtask data is a list of (end_step, tag) tuples, where:
    - end_step: The last step index of this subtask (exclusive)
    - tag: Text tag or a list of text tag variants for that step range

    Data Structure Examples:
        [(100, "Grasp the red block."), (200, "Place the red block.")]
        [(100, ["Grasp the red block.", "Pick up the red block."])]
        [(100, "Pick"), (200, "Place")]
    """

    data: List[Tuple[int, StepTag]]
    BLANK_VALUES: ClassVar[frozenset[str]] = frozenset(
        ["none", "null", "None", ""]
    )

    def get_step_tag(self, step_index):
        end_steps = [end_step for end_step, _ in self.data]
        idx = bisect.bisect_left(end_steps, step_index)
        if idx >= len(self.data):
            return None
        end_index = self.data[idx][0]
        subtask = self.data[idx][1]
        if isinstance(subtask, str):
            if subtask in self.BLANK_VALUES:
                return None
            return subtask, end_index

        subtask = [item for item in subtask if item not in self.BLANK_VALUES]
        if len(subtask) == 0:
            return None
        return subtask, end_index


class InstructionReader:
    """A reader for instruction data stored in LMDB format.

    This class provides efficient read-only access to instruction and subtask
    data stored across multiple LMDB databases. It supports fallback lookup
    across multiple database paths.

    Data Structure Example:
        Key: "{prefix}/instruction"
        Value: "Pick up the pen cap and the pen with both arms."

        Key: "{prefix}/subtask"
        Value: [(100, "Grasp the red block."), (200, "Place the red block.")]

    Args:
        paths: A single path or list of paths to LMDB databases
        encoding_mode: Character encoding for the data (default: "utf-8")
    """

    def __init__(self, paths: List[str] | str, encoding_mode: str = "utf-8"):
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        self.paths = paths
        self.encoding_mode = encoding_mode
        self.initialized = False

    def init_lmdb(self):
        if self.initialized:
            return
        self.lmdbs = [
            Lmdb(
                uri=path,
                writable=False,
                encoding_mode=self.encoding_mode,
            )
            for path in self.paths
        ]
        self.initialized = True

    def get(self, prefix, step_index=None):
        """Retrieve instruction and/or subtask data.

        Three retrieval modes:
        1. Task-level instruction: `prefix` = task_name, `step_index` = None
        2. Episode-level instruction: `prefix` = uuid, `step_index` = None
        3. Step-level subtask: `prefix` = uuid, `step_index` = step_index

        Args:
            prefix: task_name (mode 1) or episode uuid (modes 2-3)
            step_index: step index for subtask retrieval (mode 3 only)
        """
        instruction = None
        for lmdb in self.lmdbs:
            instruction = lmdb[f"{prefix}/instruction"]
            if instruction is not None:
                break

        if instruction is None:
            return None

        if step_index is not None:
            subtask = None
            subtask_end_index = None
            for lmdb in self.lmdbs:
                subtask = lmdb[f"{prefix}/subtask"]
                if subtask is not None:
                    break
            if subtask is not None:
                subtask_info = StepLevelTags(data=subtask).get_step_tag(
                    step_index
                )
                if subtask_info is None:
                    subtask = None
                else:
                    subtask, subtask_end_index = subtask_info
        else:
            subtask = None
            subtask_end_index = None
        return {
            "instruction": instruction,
            "subtask": subtask,
            "subtask_end_index": subtask_end_index,
        }


class BaseLmdbManipulationDataset(Dataset):
    """A dataset class for manipulation tasks stored in LMDB format.

    The dataset is structured into four fundamental components:
    `index`, `meta`, `depth`, and `image`.

    .. note::

        **index** and **meta** are organized by episode as the basic unit.

        **depth** and **image** are stored by step as the basic unit.

    An example:

    .. code-block:: text

        - index:
            - `episode_id`: `BaseIndexData`.
        - meta:
            - `{uuid}/meta_data`: General metadata about the task.
            - `{uuid}/camera_names`: List of camera names used in the task.
            - `{uuid}/observation/joint_positions`: [num_steps * num_joint]
        - image:
            - `{uuid}/{cam_name}/{step_idx}`: image_buffer
        - depth:
            - `{uuid}/{cam_name}/{step_idx}`: depth_buffer

    Args:
        paths (Union[str, List[str]]): Path(s) to the LMDB database(s). Can be
            a single path or a list of paths.
        transforms (Optional[List[Callable]]): A function/transform to apply to
            the data samples. Can also be a sequence of transforms.
            Default: None.
        interval (Optional[int]): Interval between steps to sample.
            Default: None
        load_image (bool): Whether to load image data. Default: True.
        load_depth (bool): Whether to load depth data. Default: True.
        task_names (Optional[Union[str, List[str]]]): List of task names to
            filter by. Default: None.
        num_episode_per_task (Optional[int]): Maximum num of episodes per task.
            Default: None.
        lazy_init (bool): If True, initialization is deferred until first
            access. Default: False.
        encoding_mode (str): Encoding mode of keys from LMDB.
            Default: "utf-8".
        lru_queue_length (Optional[int]): The length of the queue that tracks
            recently accessed LMDB indexes. This queue is used to close the
            least recently accessed LMDB to prevent virtual memory from
            exceeding the limit. Default: None.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        transforms: Optional[List[Callable]] = None,
        interval: Optional[int] = None,
        load_image: bool = True,
        load_depth: bool = True,
        task_names: Optional[Union[str, List[str]]] = None,
        num_episode_per_task: Optional[int] = None,
        lazy_init: bool = False,
        encoding_mode: str = "utf-8",
        dataset_name: str = "",
        reset_step: int = 10000,
        lmdb_kwargs: Optional[dict] = None,
        flag: Optional[int] = None,
        lru_queue_length: Optional[int] = None,
        instruction_reader: Optional[InstructionReader] = None,
        pred_steps: Optional[int] = None,
        hist_steps: Optional[int] = None,
    ):
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        self.paths = paths
        self.transforms = [build(x) for x in as_sequence(transforms)]
        self.interval = interval
        self.load_image = load_image
        self.load_depth = load_depth
        self.task_names = task_names
        self.num_episode_per_task = num_episode_per_task
        self.encoding_mode = encoding_mode
        self.dataset_name = dataset_name
        self.reset_step = reset_step
        self.lmdb_kwargs = lmdb_kwargs if lmdb_kwargs is not None else {}
        self.flag = flag
        self.lru_queue_length = lru_queue_length
        self.instruction_reader = instruction_reader
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps
        self.initialized = False
        if not lazy_init:
            self._init_lmdb()

    def _check_valid(self, index_data):
        if index_data.error:
            return False
        if (self.task_names is not None) and (
            index_data.task_name not in self.task_names
        ):
            return False
        return True

    def _get_task_name(self, index_data):
        return str(index_data.task_name)

    def _init_lmdb(self):
        if self.initialized:
            return
        if self.instruction_reader is not None:
            self.instruction_reader.init_lmdb()
        self.meta_lmdbs = [None for path in self.paths]
        self.img_lmdbs = [None for path in self.paths]
        self.depth_lmdbs = [None for path in self.paths]
        self.read_times = [0 for _ in self.paths]
        if self.lru_queue_length is not None:
            self.lru_queue = deque()
        self.idx_lmdbs = [
            Lmdb(
                uri=os.path.join(path, "index"),
                writable=False,
                encoding_mode=self.encoding_mode,
            )
            for path in self.paths
        ]

        lmdb_indices = []
        episode_indices = []
        num_steps = []
        task_names = []
        task_statistics = {
            "num_episode": {},
            "num_steps": {},
        }
        current_num_episode = 0
        for i, idx_lmdb in enumerate(self.idx_lmdbs):
            for episode_idx in idx_lmdb.keys():
                if episode_idx == "__len__":
                    continue
                data = BaseIndexData.model_validate(idx_lmdb.get(episode_idx))
                task_name = self._get_task_name(data)

                if (self._check_valid(data)) and (
                    self.num_episode_per_task is None
                    or task_statistics["num_episode"].get(task_name, 0)
                    < self.num_episode_per_task
                ):
                    lmdb_indices.append(i)
                    episode_indices.append(episode_idx)
                    num_steps.append(data.num_steps)
                    task_names.append(task_name)
                    current_num_episode += 1

                    if task_name not in task_statistics["num_episode"]:
                        task_statistics["num_episode"][task_name] = 0
                        task_statistics["num_steps"][task_name] = 0
                    task_statistics["num_episode"][task_name] += 1
                    task_statistics["num_steps"][task_name] += data.num_steps

        (task_names, lmdb_indices, episode_indices, num_steps) = map(
            list,
            zip(
                *sorted(
                    zip(
                        task_names,
                        lmdb_indices,
                        episode_indices,
                        num_steps,
                        strict=True,
                    ),
                    key=lambda x: x[0],
                ),
                strict=True,
            ),
        )

        self.lmdb_indices = lmdb_indices
        self.episode_indices = episode_indices
        self.num_steps = np.array(num_steps)
        self.cumsum_steps = np.cumsum(num_steps)
        self.num_episode = len(num_steps)
        self.task_statistics = task_statistics
        self.initialized = True
        dist_info = get_dist_info()
        if dist_info.rank == 0:
            logger.info(
                f"{self.dataset_name} dataset length: {self.__len__()}, "
                f"number of episode: {self.num_episode}, "
                f"task_statistics: {self.task_statistics}"
            )

    def __len__(self):
        if not self.initialized:
            self._init_lmdb()
        if len(self.cumsum_steps) == 0:
            return 0
        if self.interval is None:
            return self.cumsum_steps[-1]
        else:
            return self.cumsum_steps[-1] // self.interval

    def _mem_manager(self, lmdb_index):
        self.read_times[lmdb_index] += 1
        if self.lru_queue_length is not None:
            while self.lru_queue_length <= len(self.lru_queue):
                earliest_index = self.lru_queue.pop()
                self.read_times[earliest_index] -= 1
            self.lru_queue.append(lmdb_index)

        for lmdb_index_to_close in np.argsort(self.read_times):
            if get_virtual_mem() < MAX_VIRTUAL_MEM:
                break
            if (
                self.load_image
                and self.img_lmdbs[lmdb_index_to_close] is not None
            ):
                self.img_lmdbs[lmdb_index_to_close].close()
            if (
                self.load_depth
                and self.depth_lmdbs[lmdb_index_to_close] is not None
            ):
                self.depth_lmdbs[lmdb_index_to_close].close()

    def _get_indices(self, index):
        if not self.initialized:
            self._init_lmdb()

        if self.interval is not None:
            index *= self.interval
        episode_index = np.searchsorted(self.cumsum_steps, index, side="right")
        lmdb_index = self.lmdb_indices[episode_index]
        if episode_index == 0:
            step_index = index
        else:
            step_index = index - self.cumsum_steps[episode_index - 1]
        episode_index = self.episode_indices[episode_index]

        self._mem_manager(lmdb_index)
        if self.meta_lmdbs[lmdb_index] is None:
            self.meta_lmdbs[lmdb_index] = Lmdb(
                uri=os.path.join(self.paths[lmdb_index], "meta"),
                writable=False,
                encoding_mode=self.encoding_mode,
                reset_step=-1,
                **self.lmdb_kwargs,
            )
            if self.load_image:
                self.img_lmdbs[lmdb_index] = Lmdb(
                    uri=os.path.join(self.paths[lmdb_index], "image"),
                    writable=False,
                    encoding_mode=self.encoding_mode,
                    reset_step=self.reset_step,
                    **self.lmdb_kwargs,
                )
            if self.load_depth:
                self.depth_lmdbs[lmdb_index] = Lmdb(
                    uri=os.path.join(self.paths[lmdb_index], "depth"),
                    writable=False,
                    encoding_mode=self.encoding_mode,
                    reset_step=self.reset_step,
                    **self.lmdb_kwargs,
                )
        return lmdb_index, episode_index, step_index

    def _concat_shards(self, *shards):
        shards = [x for x in shards if x is not None]
        if len(shards) == 0:
            return None
        elif isinstance(shards[0], np.ndarray):
            return np.concatenate(shards, axis=0)
        elif isinstance(shards[0], list):
            results = []
            for x in shards:
                results.extend(x)
        elif isinstance(shards[0], dict):
            keys = set(shards[0])
            if any(set(x) != keys for x in shards[1:]):
                raise ValueError("All dict shards must contain the same keys.")
            results = {}
            for key in shards[0].keys():
                results[key] = self._concat_shards(*(x[key] for x in shards))
        return results

    def _get_step_index_in_shard(
        self, step_index, num_steps_per_shard=None, retrival_index=None
    ):
        if num_steps_per_shard is None:
            return step_index
        self._require_shard_steps()
        shard_index = step_index // num_steps_per_shard
        step_index_in_shard = step_index % num_steps_per_shard
        if (
            self.hist_steps is not None
            and step_index_in_shard < self.hist_steps - 1
            and shard_index != 0
        ):
            step_index_in_shard += num_steps_per_shard
        if retrival_index is not None:
            step_index_in_shard += retrival_index - step_index
        return step_index_in_shard

    def _require_shard_steps(self):
        if self.hist_steps is None or self.pred_steps is None:
            raise ValueError(
                "hist_steps and pred_steps must be specified when reading "
                "sharded LMDB metadata."
            )

    def _get_meta(
        self, lmdb_index, uuid, key, step_index=None, num_steps_per_shard=None
    ):
        if num_steps_per_shard is None:
            return self.meta_lmdbs[lmdb_index][f"{uuid}/{key}"]

        self._require_shard_steps()
        shard_index = step_index // num_steps_per_shard
        current_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/{shard_index}/{key}"
        ]
        step_index_in_shard = step_index % num_steps_per_shard
        if (
            self.hist_steps is not None
            and step_index_in_shard < self.hist_steps - 1
            and shard_index != 0
        ):
            pre_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index - 1}/{key}"
            ]
        else:
            pre_shard = None

        if (
            self.pred_steps is not None
            and num_steps_per_shard - step_index_in_shard < self.pred_steps
        ):
            next_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index + 1}/{key}"
            ]
        else:
            next_shard = None
        data = self._concat_shards(pre_shard, current_shard, next_shard)
        return data

    def __getitem__(self, index):
        """Get data dict by index.

        Obtain the hierarchical indices (lmdb_index, episode_index, step_index)
        by first calling:
            lmdb_index, episode_index, step_index = self._get_indices(index)
        """
        raise NotImplementedError

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        end = int(self.cumsum_steps[ep_idx])
        start = int(self.cumsum_steps[ep_idx - 1]) if ep_idx > 0 else 0
        return start, end

    def visualize(self, episode_index, output_path="./vis_data"):
        raise NotImplementedError


class BaseLmdbManipulationDataPacker(object):
    def __init__(self, input_path, output_path, commit_step=500, **kwargs):
        self.input_path = input_path
        self.output_path = output_path
        self.commit_step = commit_step
        self.lmdb_kwargs = kwargs

    def _init_lmdbs(self):
        for f in ["index", "meta", "image", "depth"]:
            uri = os.path.join(self.output_path, f)
            if not os.path.exists(uri):
                os.makedirs(uri)
            setattr(
                self,
                f"{f}_pack_file",
                Lmdb(
                    uri=uri,
                    writable=True,
                    commit_step=self.commit_step,
                    **self.lmdb_kwargs,
                ),
            )

    def close(self):
        self.index_pack_file.close()
        self.meta_pack_file.close()
        self.image_pack_file.close()
        self.depth_pack_file.close()

    def write_index(self, index: Union[int, str], index_data: dict):
        self.index_pack_file.write(
            index, BaseIndexData.model_validate(index_data).model_dump()
        )

    def __call__(self):
        self._init_lmdbs()
        self._pack()

    def _pack(self):
        raise NotImplementedError
