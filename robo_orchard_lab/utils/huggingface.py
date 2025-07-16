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

import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import fsspec
from accelerate import Accelerator
from huggingface_hub import snapshot_download

from robo_orchard_lab.utils.env import set_env

__all__ = [
    "get_accelerate_project_last_checkpoint_id",
    "accelerator_load_state",
    "AcceleratorState",
    "download_repo",
]


def get_accelerate_project_last_checkpoint_id(project_dir: str) -> int:
    """Helper function to get last checkpoint id.

    Retrieves the ID of the last checkpoint in the specified project directory.

    This function specifically handles checkpoints saved using the
    `Accelerator.save_state` method from the Hugging Face `accelerate`
    library, which follows an automatic checkpoint naming convention.
    It searches the specified `project_dir/checkpoints` directory,
    extracts numerical IDs from folder names, and returns the highest ID,
    representing the most recent checkpoint.

    Args:
        project_dir (str): Path to the project directory containing the
            `checkpoints` folder. This directory should contain only
            checkpoints saved by `Accelerator.save_state`.

    Returns:
        int: The ID of the last (most recent) checkpoint found in the
            project directory. Returns `-1` if the `checkpoints` directory
            does not exist or is empty.

    Raises:
        ValueError: If no valid checkpoint IDs are found in the `checkpoints`
            directory.

    Example:
        >>> get_accelerate_project_last_checkpoint_id("/path/to/project")
        42

    Note:
        This function assumes that all entries in the `checkpoints` directory
        follow the automatic checkpoint naming pattern used by
        `Accelerator.save_state`. Checkpoints not saved with
        `Accelerator.save_state` may cause this function to fail.
    """
    input_dir = os.path.join(project_dir, "checkpoints")

    if not os.path.exists(input_dir):
        return -1

    iter_ids = []
    for folder_i in os.listdir(input_dir):
        iter_ids.append(
            int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder_i)[0])
        )

    iter_ids.sort()

    return iter_ids[-1]


def accelerator_load_state(
    accelerator: Accelerator,
    input_dir: str,
    cache_dir: str | None = None,
    safe_serialization: bool = True,
    **kwargs,
) -> None:
    """Load the state of the accelerator from a checkpoint.

    This function extends the functionality of `accelerator.load_state` to
    support loading checkpoints from remote filesystems (e.g., S3, GCS).

    It first checks if the `input_dir` is a local path or a remote path.
    If it's a local path, it directly calls `accelerator.load_state`. If it's
    a remote path, it synchronizes the checkpoint files to a local cache
    directory before loading the state.

    Args:
        accelerator (Accelerator): The `Accelerator` instance to load the
            state into.
        input_dir (str): The path to the checkpoint directory or file.
            This can be a local path or a remote path (e.g., S3, GCS).
        cache_dir (str | None): The local directory to cache the checkpoint
            files. This is required if `input_dir` is a remote path.
        safe_serialization (bool): Whether to use safe serialization when
            loading the state. This is used when input_dir is a remote
            path. The names of checkpoint files depend on whether
            `safe_serialization` is set to `True` or `False`. Users should
            ensure that the checkpoint files in the remote directory are
            compatible with the specified `safe_serialization` option.
        **kwargs: Additional arguments passed to `accelerator.load_state`.
    """

    def get_fs_protocol(path: str) -> str:
        """Get the filesystem protocol from a path."""
        path_splits = path.split("://")
        if len(path_splits) == 1:
            protocol = "file"
        else:
            protocol = path_splits[0]
        return protocol

    def sync_remote_checkpoints(
        accelerator: Accelerator,
        remote_dir: str,
        cache_dir: str,
        safe_serialization: bool = True,
    ) -> None:
        """Sync remote checkpoints to local cache."""
        if not accelerator.is_local_main_process:
            raise RuntimeError(
                "sync_remote_checkpoints should only be called "
                "on the main process."
            )
        pj_config = accelerator.project_configuration
        old_v = pj_config.automatic_checkpoint_naming
        # disable automatic checkpoint naming to use given checkpoint
        # directory!
        pj_config.automatic_checkpoint_naming = False
        accelerator.save_state(
            cache_dir, safe_serialization=safe_serialization
        )
        file_names = list(os.listdir(cache_dir))
        for file_name in file_names:
            try:
                with (
                    fsspec.open(
                        os.path.join(remote_dir, file_name), "rb"
                    ) as remote_file,
                    open(
                        os.path.join(cache_dir, file_name), "wb"
                    ) as local_file,
                ):
                    # chunk read and write with 32MB
                    while True:
                        data = remote_file.read(1024 * 1024 * 32)  # type: ignore
                        if not data:
                            break
                        local_file.write(data)
            except FileNotFoundError:
                warnings.warn(
                    f"File {file_name} not found in {cache_dir}. Skipping."
                )
        pj_config.automatic_checkpoint_naming = old_v

    input_dir_fs_protocol = get_fs_protocol(input_dir)
    if input_dir_fs_protocol == "file":
        if not os.path.exists(input_dir):
            raise ValueError(
                f"Checkpoint directory {input_dir} does not exist."
            )
        return accelerator.load_state(input_dir, **kwargs)
    else:
        if cache_dir is None:
            raise ValueError(
                "cache_dir should be specified when input_dir is "
                "not a local path."
            )
        if not os.path.exists(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        if accelerator.is_local_main_process:
            sync_remote_checkpoints(
                accelerator,
                input_dir,
                cache_dir,
                safe_serialization=safe_serialization,
            )
        accelerator.wait_for_everyone()
        accelerator.load_state(cache_dir, **kwargs)


@dataclass
class AcceleratorState:
    """A data class for storing the state of the Accelerator.

    This class implements the `state_dict` and `load_state_dict` methods to
    save and load the state of the Accelerator. Any dataclass that is used by
    Accelerator.load_state should inherit from this class.
    """

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the training progress as a dictionary.

        Returns:
            dict: A dictionary containing the current epoch, step, and
                global step IDs.
        """
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads the state of the training progress from a dictionary.

        Args:
            state (dict): A dictionary containing the state to be loaded.
        """
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Key {key} not found in TrainerProgressState.")


def download_repo(url: str, repo_type: str) -> str:
    if not url.startswith("hf://"):
        raise ValueError("url should starts with hf://")
    parsed_url = urlparse(url)
    token_from_url = parsed_url.username

    repo_id = parsed_url.hostname

    if not repo_id:
        raise ValueError(f"Invalid Hugging Face Hub URI: {url}")

    if parsed_url.path:
        repo_id += parsed_url.path

    with set_env(HF_HUB_DISABLE_PROGRESS_BARS="1"):
        directory = snapshot_download(
            repo_id=repo_id,
            # Pass the token extracted from the URL.
            # If no token was in the URL, this will be None, and
            # huggingface_hub will fall back to env variables/cache.
            token=token_from_url,
            repo_type=repo_type,
        )
        return directory
