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
import fsspec
from datasets.arrow_dataset import (
    Dataset,
    DatasetInfo,
    InMemoryTable,
    MemoryMappedTable,
    Optional,
    Path,
    PathLike,
    Split,
    concat_tables,
    estimate_dataset_size,
    hf_tqdm,
    is_remote_filesystem,
    is_small_dataset,
    json,
    thread_map,
)
from fsspec import url_to_fs

__all__ = ["load_from_disk"]


def load_from_disk(
    dataset_path: PathLike,
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
) -> Dataset:
    """A wrapper around `datasets.load_from_disk`.

    This method fix the issue when loading `Dataset` object when info.feature
    does not match the arrow table schema exactly. It is a bug in `Dataset`
    implementation, and we provide this wrapper to fix the issue.
    """
    import posixpath

    fs: fsspec.AbstractFileSystem
    fs, dataset_path = url_to_fs(dataset_path, **(storage_options or {}))
    import datasets.config as config

    dest_dataset_path = dataset_path
    dataset_dict_json_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASETDICT_JSON_FILENAME,  # type: ignore
    )
    dataset_state_json_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASET_STATE_JSON_FILENAME,  # type: ignore
    )
    dataset_info_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASET_INFO_FILENAME,  # type: ignore
    )

    dataset_dict_is_file = fs.isfile(dataset_dict_json_path)
    dataset_info_is_file = fs.isfile(dataset_info_path)
    dataset_state_is_file = fs.isfile(dataset_state_json_path)
    if not dataset_info_is_file and not dataset_state_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )
    if not dataset_info_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such file: '{dataset_info_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such file: '{dataset_info_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )
    if not dataset_state_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such file: '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such file: '{dataset_state_json_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )

    # copies file from filesystem if it is remote filesystem to local
    # filesystem and modifies dataset_path to temp directory
    # containing local copies
    if is_remote_filesystem(fs):
        src_dataset_path = dest_dataset_path
        dest_dataset_path = Dataset._build_local_temp_path(src_dataset_path)  # type: ignore
        fs.download(
            src_dataset_path, dest_dataset_path.as_posix(), recursive=True
        )
        dataset_state_json_path = posixpath.join(
            dest_dataset_path, config.DATASET_STATE_JSON_FILENAME
        )
        dataset_info_path = posixpath.join(
            dest_dataset_path, config.DATASET_INFO_FILENAME
        )

    with open(dataset_state_json_path, encoding="utf-8") as state_file:
        state = json.load(state_file)
    with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
        dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_size = estimate_dataset_size(
        Path(dest_dataset_path, data_file["filename"])  # type: ignore
        for data_file in state["_data_files"]
    )
    keep_in_memory = (
        keep_in_memory
        if keep_in_memory is not None
        else is_small_dataset(dataset_size)
    )
    table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable

    arrow_table = concat_tables(
        thread_map(
            table_cls.from_file,
            [
                posixpath.join(dest_dataset_path, data_file["filename"])
                for data_file in state["_data_files"]
            ],
            tqdm_class=hf_tqdm,
            desc="Loading dataset from disk",
            # set `disable=None` rather than `disable=False` by default
            # to disable progress bar when no TTY attached
            disable=len(state["_data_files"]) <= 16 or None,
        )
    )

    split = state["_split"]
    split = Split(split) if split is not None else split

    if arrow_table.schema != dataset_info.features.arrow_schema:  # type: ignore # noqa: E501
        arrow_table = arrow_table.cast(dataset_info.features.arrow_schema)  # type: ignore # noqa: E501

    dataset = Dataset(
        arrow_table=arrow_table,
        info=dataset_info,
        split=split,
        fingerprint=state["_fingerprint"],
    )

    format = {
        "type": state["_format_type"],
        "format_kwargs": state["_format_kwargs"],
        "columns": state["_format_columns"],
        "output_all_columns": state["_output_all_columns"],
    }
    dataset = dataset.with_format(**format)

    return dataset
