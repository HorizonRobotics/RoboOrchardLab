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

import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import fsspec
from accelerate import Accelerator
from accelerate.utils import DistributedType
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile

from robo_orchard_lab.utils.env import set_env

__all__ = [
    "get_accelerate_project_last_checkpoint_id",
    "accelerator_load_state",
    "AcceleratorState",
    "download_hf_resource",
    "auto_add_repo_type",
    "resolve_hf_compatible_path",
]


def _should_write_like_accelerate_save(accelerator: Accelerator) -> bool:
    if accelerator.project_configuration.save_on_each_node:
        return accelerator.is_local_main_process
    return accelerator.is_main_process


def _has_model_save_artifact(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False

    try:
        with os.scandir(directory) as entries:
            return any(
                entry.is_file()
                and _is_accelerate_model_artifact_name(entry.name)
                for entry in entries
            )
    except FileNotFoundError:
        return False


def _is_accelerate_model_artifact_name(filename: str) -> bool:
    if filename in {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }:
        return True

    return (
        filename.startswith("model-") and filename.endswith(".safetensors")
    ) or (filename.startswith("pytorch_model-") and filename.endswith(".bin"))


def get_accelerate_project_last_checkpoint_id(project_dir: str) -> int:
    """Helper function to get last checkpoint id.

    Return the highest occupied automatic checkpoint ID in a project.

    This function specifically handles checkpoints saved using the
    `Accelerator.save_state` method from the Hugging Face `accelerate`
    library, which follows an automatic checkpoint naming convention.
    It searches the specified `project_dir/checkpoints` directory,
    recognizes entries named ``checkpoint_<nonnegative integer>`` and returns
    the highest ID. Other entries are ignored.

    Args:
        project_dir (str): Path to the project directory containing the
            `checkpoints` folder.

    Returns:
        int: The largest occupied checkpoint ID, or ``-1`` if no matching
        entry exists. An empty checkpoint directory still occupies its ID.

    Example:
        >>> get_accelerate_project_last_checkpoint_id("/path/to/project")
        42

    Note:
        The result is suitable for selecting the next unused ID; it does not
        establish that the matching checkpoint is complete or loadable.
    """
    input_dir = os.path.join(project_dir, "checkpoints")

    if not os.path.exists(input_dir):
        return -1

    return max(
        (
            int(match.group(1))
            for name in os.listdir(input_dir)
            if (match := re.fullmatch(r"checkpoint_([0-9]+)", name))
            is not None
        ),
        default=-1,
    )


def _supports_checkpoint_backend(accelerator: Accelerator) -> bool:
    """Identify backends supporting trainer/data recovery."""
    return accelerator.distributed_type in {
        DistributedType.NO,
        DistributedType.MULTI_CPU,
        DistributedType.MULTI_GPU,
    }


def _checkpoint_capability_reason(accelerator: Accelerator) -> str | None:
    """Return a current trainer/data recovery limitation, if any."""
    if not _supports_checkpoint_backend(accelerator):
        return (
            "Trainer/data checkpoint recovery is unsupported for this backend."
        )
    if accelerator.device.type not in {"cpu", "cuda"}:
        return "Trainer/data recovery RNG is unsupported for this device."
    if accelerator.project_configuration.save_on_each_node:
        return "Trainer/data recovery with node-local saves is unsupported."
    for loader in accelerator._dataloaders:
        if getattr(loader, "use_stateful_dataloader", False):
            return (
                "Trainer/data recovery of an external stateful loader "
                "is unsupported."
            )
    return None


def _copy_remote_checkpoint(input_dir: str, stage: str) -> None:
    """Copy all remote members into an owned staging directory."""
    fs, remote_path = fsspec.core.url_to_fs(input_dir)
    members = []
    for source in fs.find(remote_path):
        relative = posixpath.relpath(source, remote_path)
        if (
            relative in {".", ".."}
            or relative.startswith("../")
            or posixpath.isabs(relative)
        ):
            raise ValueError("Remote checkpoint member escapes its root.")
        members.append((source, relative))
    if not members:
        raise FileNotFoundError("Remote checkpoint contains no files.")
    for source, relative in members:
        target = os.path.join(stage, *relative.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with fs.open(source, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 32)


def accelerator_load_state(
    accelerator: Accelerator,
    input_dir: str,
    cache_dir: str | None = None,
    safe_serialization: bool = True,
    **kwargs,
) -> None:
    """Load local or staged remote state through Accelerate.

    Accelerate owns model, optimizer, custom-object and global RNG loading.
    This helper does not inspect registered object types or reset the native
    accumulation phase. Training-specific capability checks and phase handling
    belong to ``ResumeCheckpointConfig.load_state``; prepared data restoration
    remains separate.

    Args:
        accelerator: Existing Accelerator containing the runtime objects.
        input_dir: Local checkpoint directory, file:// path or remote URL.
        cache_dir: Existing local directory used for remote staging.
        safe_serialization: Retained for callers; Accelerate detects format.
        **kwargs: Passed to Accelerator.load_state.
    """

    del safe_serialization
    remote = "://" in input_dir and not input_dir.startswith("file://")
    directory = input_dir.removeprefix("file://")
    stage: str | None = None
    try:
        if remote:
            if cache_dir is None or not os.path.isdir(cache_dir):
                raise ValueError("Remote recovery requires a cache directory.")
            stage = tempfile.mkdtemp(prefix="checkpoint-load-", dir=cache_dir)
            _copy_remote_checkpoint(input_dir, stage)
            directory = stage
        accelerator.load_state(directory, **kwargs)
    finally:
        if stage is not None:
            primary_active = sys.exc_info()[0] is not None
            try:
                shutil.rmtree(stage)
            except BaseException as cleanup_error:
                if not primary_active:
                    raise
                try:
                    logging.getLogger(__name__).warning(
                        "Checkpoint staging cleanup failed: %r", cleanup_error
                    )
                except Exception:
                    pass


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


VALID_REPO_TYPES = {"model", "dataset", "space"}


def resolve_hf_compatible_path(path: str, repo_type: str = "model") -> str:
    """Resolve the repository's ``hf://`` URI form into a local resource.

    Non-``hf://`` inputs are returned unchanged so callers can layer their own
    path semantics, such as ``abspath`` or local-existence checks, on top.

    Args:
        path (str): Local path, standard Hugging Face identifier, or
            repository-specific ``hf://`` URI.
        repo_type (str, optional): Default repo type to inject for bare
            ``hf://`` URIs that omit it. Default is ``"model"``.

    Returns:
        str: Local downloaded path for ``hf://`` URIs, otherwise the original
            input path.
    """

    if path.startswith("hf://"):
        return download_hf_resource(auto_add_repo_type(path, repo_type))
    return path


def download_hf_resource(url: str) -> str:
    """Downloads a resource (repo, dir, or file) from the Hugging Face Hub using a custom HF URI scheme.

    The URI scheme is defined as:
    ``hf://[<token>@]<repo_type>/<repo_id>[/<path>][@<revision>]``

    - <token>@: (Optional) User token

    - repo_type: (Optional) 'model', 'dataset', or 'space'

    - repo_id: The repository ID (e.g., 'gpt2' or 'meta-llama/Llama-2-7b-chat-hf')

    - /<path>: (Optional) Download a single file or a specific subdirectory

    - @<revision>: (Optional) A git revision

    Args:
        url (str): The HF URI string.

    Returns:
        str: The local directory path (for repo/dir) or file path.

    Raises:
        ValueError: If the URL format is invalid or missing required components.
    """  # noqa: E501
    if not url.startswith("hf://"):
        raise ValueError("URL must start with hf://")

    parsed_uri = urlparse(url)

    # Extract Token (Priority: Userinfo > Query Param)
    token = parsed_uri.username
    if token:
        token = unquote(token)

    # Handle revision at the end of the path (e.g., .../file.txt@v1.0)
    raw_path = parsed_uri.path
    revision = None
    if "@" in raw_path:
        raw_path, revision = raw_path.rsplit("@", 1)

    # Path reconstruction
    netloc = parsed_uri.netloc

    # remove token part
    if "@" in netloc:
        host_part = netloc.rsplit("@", 1)[-1]
    else:
        host_part = netloc

    path_part = raw_path.lstrip("/")
    full_path_str = (
        f"{host_part}/{path_part}"
        if host_part and path_part
        else (host_part or path_part)
    )
    parts = [p for p in full_path_str.split("/") if p]

    if not parts:
        raise ValueError(f"Invalid URI: Empty path in '{url}'")

    if parts[0] not in VALID_REPO_TYPES:
        raise ValueError(f"Invalid repo type {parts[0]}")
    repo_type = parts[0]
    segments = parts[1:]

    if len(segments) < 2:
        raise ValueError(
            f"Invalid repo_id structure. Expected 'username/repo_name', "
            f"but got: {'/'.join(segments)}"
        )

    repo_id = f"{segments[0]}/{segments[1]}"
    path_inside_repo = "/".join(segments[2:]) if len(segments) > 2 else None

    # download
    with set_env(HF_HUB_DISABLE_PROGRESS_BARS="1"):
        if not path_inside_repo:
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )

        api = HfApi(token=token)
        try:
            paths_info = api.get_paths_info(
                repo_id=repo_id,
                repo_type=repo_type,
                paths=[path_inside_repo],
                revision=revision,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to resolve path '{path_inside_repo}' in repo: {e}"
            )

        if not paths_info:
            raise ValueError(
                f"Path '{path_inside_repo}' does not exist in {repo_id}"
            )

        if isinstance(paths_info[0], RepoFile):
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=path_inside_repo,
                revision=revision,
                token=token,
            )
        else:
            allow_pattern = f"{path_inside_repo.rstrip('/')}/*"
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                allow_patterns=[allow_pattern],
            )
            return os.path.join(path, path_inside_repo)


def auto_add_repo_type(url: str, repo_type: str = "model") -> str:
    """Automatically add repo type to url."""

    if not url.startswith("hf://"):
        raise ValueError("URL must start with hf://")

    if repo_type not in VALID_REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}")

    parsed_uri = urlparse(url)

    url_repo_type = parsed_uri.netloc.split("@")[-1]

    if url_repo_type in VALID_REPO_TYPES:
        if url_repo_type != repo_type:
            raise ValueError(
                f"url already has repo type {url_repo_type} "
                f"but not matched required repo type {repo_type}"
            )

        return url

    if parsed_uri.username:
        token, organize = parsed_uri.netloc.split("@")
        netloc = f"{token}@{repo_type}/{organize}/"
    else:
        netloc = f"{repo_type}/{parsed_uri.netloc}"

    return f"{parsed_uri.scheme}://{netloc}/{parsed_uri.path}"
