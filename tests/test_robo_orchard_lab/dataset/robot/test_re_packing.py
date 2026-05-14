# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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
from dataclasses import replace
from pathlib import Path

import datasets as hg_datasets
import pytest

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode, Instruction, Task
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    TaskData,
)
from robo_orchard_lab.dataset.robot.re_packing import (
    DefaultRePackingEpisodeHelper,
    IdentityRODatasetRepackTransform,
    RODatasetEpisodeContext,
    RODatasetEpisodeSelection,
    RODatasetFrameContext,
    RODatasetRepackContext,
    repack_dataset,
)


class _SimpleRepackEpisode(EpisodePackaging):
    features = hg_datasets.Features(
        {
            "value": hg_datasets.Value("int64"),
            "text": hg_datasets.Value("string"),
        }
    )

    def __init__(
        self,
        episode_id: int,
        *,
        frame_count: int = 2,
        prev_episode_index: int | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.frame_count = frame_count
        self.prev_episode_index = prev_episode_index

    def generate_episode_meta(self) -> EpisodeMeta:
        return EpisodeMeta(
            episode=EpisodeData(
                index=(
                    self.episode_id
                    if self.prev_episode_index is not None
                    else None
                ),
                prev_episode_index=self.prev_episode_index,
                truncated=bool(self.episode_id),
                success=not bool(self.episode_id),
                info={"episode": self.episode_id},
            ),
            robot=RobotData(
                name=f"robot-{self.episode_id}",
                content=None,
                content_format=None,
            ),
            task=TaskData(
                name=f"task-{self.episode_id}",
                description=f"task {self.episode_id}",
                info={"task": self.episode_id},
            ),
        )

    def generate_frames(self):
        for frame_index in range(self.frame_count):
            value = self.episode_id * 10 + frame_index
            yield DataFrame(
                features={
                    "value": value,
                    "text": f"episode-{self.episode_id}-frame-{frame_index}",
                },
                instruction=InstructionData(
                    name=f"instruction-{self.episode_id}-{frame_index}",
                    json_content={"frame": frame_index},
                ),
                timestamp_ns_min=value,
                timestamp_ns_max=value,
            )


def _make_source_dataset(tmp_path: Path) -> RODataset:
    source_path = tmp_path / "source_ro_dataset"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[_SimpleRepackEpisode(0), _SimpleRepackEpisode(1)],
        dataset_path=str(source_path),
        writer_batch_size=1,
        force_overwrite=True,
    )
    return RODataset(str(source_path))


def _make_linked_source_dataset(tmp_path: Path) -> RODataset:
    source_path = tmp_path / "linked_source_ro_dataset"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[
            _SimpleRepackEpisode(0),
            _SimpleRepackEpisode(1, prev_episode_index=0),
        ],
        dataset_path=str(source_path),
        writer_batch_size=1,
        force_overwrite=True,
    )
    return RODataset(str(source_path))


def _make_linked_source_dataset_with_four_episodes(
    tmp_path: Path,
) -> RODataset:
    source_path = tmp_path / "linked_source_ro_dataset_with_four_episodes"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[
            _SimpleRepackEpisode(0),
            _SimpleRepackEpisode(1, prev_episode_index=0),
            _SimpleRepackEpisode(2, prev_episode_index=1),
            _SimpleRepackEpisode(3, prev_episode_index=2),
        ],
        dataset_path=str(source_path),
        writer_batch_size=1,
        force_overwrite=True,
    )
    return RODataset(str(source_path))


def _make_branching_source_dataset(tmp_path: Path) -> RODataset:
    source_path = tmp_path / "branching_source_ro_dataset"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[
            _SimpleRepackEpisode(0),
            _SimpleRepackEpisode(1, prev_episode_index=0),
            _SimpleRepackEpisode(2, prev_episode_index=0),
        ],
        dataset_path=str(source_path),
        writer_batch_size=1,
        force_overwrite=True,
    )
    return RODataset(str(source_path))


def _episode_prev_indices(dataset: RODataset) -> list[int | None]:
    prev_indices: list[int | None] = []
    for episode_index in range(dataset.episode_num):
        episode = dataset.get_meta(Episode, episode_index)
        assert episode is not None
        prev_indices.append(episode.prev_episode_index)
    return prev_indices


class _CustomRePackingEpisodeHelper(DefaultRePackingEpisodeHelper):
    pass


def test_transform_identity_repack_preserves_complete_episode_metadata(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_ro_dataset"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert len(target_dataset) == len(source_dataset)
    frame0 = target_dataset[0]
    assert frame0["value"] == 0
    assert frame0["text"] == "episode-0-frame-0"
    assert frame0["timestamp_min"] == 0

    episode = target_dataset.get_meta(Episode, int(frame0["episode_index"]))
    assert episode is not None
    assert episode.info == {"episode": 0}
    assert episode.truncated is False
    assert episode.success is True


def test_default_helper_path_keeps_compatibility_behavior(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_helper_path"

    repack_dataset(
        source_dataset,
        str(target_path),
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert len(target_dataset) == len(source_dataset)
    frame0 = target_dataset[0]
    assert frame0["value"] == 0
    episode = target_dataset.get_meta(Episode, int(frame0["episode_index"]))
    assert episode is not None
    assert episode.info is None


def test_transform_repack_preserves_adjacent_complete_episode_links(
    tmp_path: Path,
) -> None:
    source_dataset = _make_linked_source_dataset(tmp_path)
    target_path = tmp_path / "target_linked_ro_dataset"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert _episode_prev_indices(target_dataset) == [None, 0]


@pytest.mark.parametrize(
    ("frame_indices", "expected_prev_indices"),
    [
        ([2, 3], [None]),
        ([0, 2, 3], [None, None]),
    ],
)
def test_transform_repack_clears_links_without_adjacent_complete_output(
    tmp_path: Path,
    frame_indices: list[int],
    expected_prev_indices: list[int | None],
) -> None:
    source_dataset = _make_linked_source_dataset(tmp_path)
    target_path = tmp_path / "target_unlinked_ro_dataset"

    repack_dataset(
        source_dataset,
        str(target_path),
        frame_indices=frame_indices,
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert _episode_prev_indices(target_dataset) == expected_prev_indices


def test_transform_repack_resumes_links_after_skipped_middle_episode(
    tmp_path: Path,
) -> None:
    source_dataset = _make_linked_source_dataset_with_four_episodes(tmp_path)
    target_path = tmp_path / "target_skip_middle_linked_ro_dataset"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[_SkipEpisodeTransform({1})],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert _episode_prev_indices(target_dataset) == [None, None, 1]


def test_transform_repack_uses_source_to_target_episode_index_map(
    tmp_path: Path,
) -> None:
    source_dataset = _make_branching_source_dataset(tmp_path)
    target_path = tmp_path / "target_branching_linked_ro_dataset"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert _episode_prev_indices(target_dataset) == [None, 0, 0]


def test_repacking_dataset_helper_remains_importable() -> None:
    from robo_orchard_lab.dataset.robot.re_packing import (
        RePackingDatasetHelper,
    )

    assert RePackingDatasetHelper is not None


def test_transform_contract_types_stay_off_robot_root_namespace() -> None:
    import robo_orchard_lab.dataset.robot as robot_dataset

    assert hasattr(robot_dataset, "repack_dataset")
    assert not hasattr(robot_dataset, "RODatasetRepackTransform")


def test_transform_mode_rejects_custom_packing_impl_and_fail_fast_false(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="packing_impl"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_custom_helper"),
            transforms=[],
            packing_impl=_CustomRePackingEpisodeHelper,
        )

    with pytest.raises(ValueError, match="fail-fast"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_fail_fast_false"),
            transforms=[],
            fail_fast=False,
        )


def test_transform_mode_rejects_duplicate_and_split_frame_indices(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_duplicate"),
            frame_indices=[0, 0],
            transforms=[],
        )

    with pytest.raises(ValueError, match="multiple chunks"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_split"),
            frame_indices=[0, 2, 1],
            transforms=[],
        )


def test_transform_mode_preserves_existing_target_on_late_failure(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_preserved_on_failure"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[_SimpleRepackEpisode(99)],
        dataset_path=str(target_path),
        writer_batch_size=1,
        force_overwrite=True,
    )

    with pytest.raises(ValueError, match="multiple chunks"):
        repack_dataset(
            source_dataset,
            str(target_path),
            frame_indices=[0, 2, 1],
            transforms=[],
            writer_batch_size=1,
            force_overwrite=True,
        )

    target_dataset = RODataset(str(target_path))
    assert len(target_dataset) == 2
    assert target_dataset[0]["value"] == 990
    assert target_dataset[0]["text"] == "episode-99-frame-0"


def test_transform_mode_replaces_existing_target_after_success(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_replaced_after_success"
    DatasetPackaging(
        features=_SimpleRepackEpisode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[_SimpleRepackEpisode(99)],
        dataset_path=str(target_path),
        writer_batch_size=1,
        force_overwrite=True,
    )

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert len(target_dataset) == len(source_dataset)
    assert target_dataset[0]["value"] == 0
    assert target_dataset[0]["text"] == "episode-0-frame-0"


class _RequireValueTransform(IdentityRODatasetRepackTransform):
    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> list[str]:
        return ["value"]


class _RequireReservedColumnTransform(IdentityRODatasetRepackTransform):
    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> list[str]:
        return ["index"]


def test_required_columns_respect_columns_projection(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="missing column 'value'"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_missing_required_column"),
            columns=["text"],
            transforms=[_RequireValueTransform()],
        )

    with pytest.raises(ValueError, match="reserved column 'index'"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_reserved_required_column"),
            transforms=[_RequireReservedColumnTransform()],
        )


def test_helper_exposes_episode_selection(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    partial_helper = DefaultRePackingEpisodeHelper(source_dataset, [0])
    assert partial_helper.episode_selection.selected_frame_indices == (0,)
    assert partial_helper.episode_selection.source_episode_frame_indices == (
        0,
        1,
    )
    assert partial_helper.episode_selection.selected_frame_count == 1
    assert partial_helper.episode_selection.source_episode_frame_count == 2
    assert partial_helper.episode_selection.is_complete_source_episode is False
    assert partial_helper.episode_selection.is_contiguous_source_slice is True

    complete_helper = DefaultRePackingEpisodeHelper(source_dataset, [1, 0])
    assert complete_helper.episode_selection.selected_frame_indices == (0, 1)
    assert complete_helper.episode_selection.is_complete_source_episode is True


class _AddValueCopyTransform(IdentityRODatasetRepackTransform):
    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        updated = hg_datasets.Features(features.copy())
        updated["value_copy"] = hg_datasets.Value("int64")
        return updated

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        frame.features["value_copy"] = frame.features["value"]
        return frame


class _ObserveValueCopyTransform(IdentityRODatasetRepackTransform):
    def __init__(self) -> None:
        self.prepare_saw_value_copy = False

    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> list[str]:
        return ["value_copy"]

    def prepare(self, context: RODatasetRepackContext) -> None:
        self.prepare_saw_value_copy = "value_copy" in context.target_features


def test_transform_feature_order_and_required_columns(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    observer = _ObserveValueCopyTransform()
    target_path = tmp_path / "target_feature_order"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[_AddValueCopyTransform(), observer],
        writer_batch_size=1,
        force_overwrite=True,
    )

    assert observer.prepare_saw_value_copy is True
    target_dataset = RODataset(str(target_path))
    assert target_dataset[0]["value_copy"] == target_dataset[0]["value"]


class _MutateMetadataTransform(IdentityRODatasetRepackTransform):
    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        if episode_meta.episode.info is not None:
            episode_meta.episode.info["episode"] = "mutated"
        if (
            episode_meta.task is not None
            and episode_meta.task.info is not None
        ):
            episode_meta.task.info["task"] = "mutated"
        return episode_meta

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        if (
            frame.instruction is not None
            and frame.instruction.json_content is not None
        ):
            frame.instruction.json_content["frame"] = "mutated"
        return frame


def test_transform_metadata_copy_does_not_mutate_source(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    repack_dataset(
        source_dataset,
        str(tmp_path / "target_mutated_metadata"),
        transforms=[_MutateMetadataTransform()],
        writer_batch_size=1,
        force_overwrite=True,
    )

    source_episode = source_dataset.get_meta(Episode, 0)
    source_task = source_dataset.get_meta(Task, 0)
    source_instruction = source_dataset.get_meta(Instruction, 0)
    assert source_episode is not None
    assert source_task is not None
    assert source_instruction is not None
    assert source_episode.info == {"episode": 0}
    assert source_task.info == {"task": 0}
    assert source_instruction.json_content == {"frame": 0}


class _InvalidFeatureReturnTransform(IdentityRODatasetRepackTransform):
    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        return {"value": hg_datasets.Value("int64")}  # type: ignore[return-value]


class _ReservedFeatureTransform(IdentityRODatasetRepackTransform):
    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        updated = hg_datasets.Features(features.copy())
        updated["index"] = hg_datasets.Value("int64")
        return updated


def test_transform_feature_contract_is_validated(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(TypeError, match="must return datasets.Features"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_invalid_features"),
            transforms=[_InvalidFeatureReturnTransform()],
        )

    with pytest.raises(ValueError, match="reserved columns"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_reserved_features"),
            transforms=[_ReservedFeatureTransform()],
        )


class _ReturnNoneFrameTransform(IdentityRODatasetRepackTransform):
    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        return None  # type: ignore[return-value]


def test_transform_frame_returning_none_is_rejected(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(TypeError, match="returned None"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_none_frame"),
            transforms=[_ReturnNoneFrameTransform()],
            writer_batch_size=1,
            force_overwrite=True,
        )


class _DropFrameFeatureTransform(IdentityRODatasetRepackTransform):
    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        features = frame.features.copy()
        features.pop("text")
        return replace(frame, features=features)


def test_transform_frame_features_must_match_target(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="missing=\\['text'\\]"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_missing_frame_feature"),
            transforms=[_DropFrameFeatureTransform()],
            writer_batch_size=1,
            force_overwrite=True,
        )


class _SelectionRecorderTransform(IdentityRODatasetRepackTransform):
    def __init__(self) -> None:
        self.selections: list[RODatasetEpisodeSelection] = []

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        self.selections.append(context.selection)
        return episode_meta


def test_partial_selection_clears_episode_metadata_and_exposes_selection(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    recorder = _SelectionRecorderTransform()
    target_path = tmp_path / "target_partial_selection"

    repack_dataset(
        source_dataset,
        str(target_path),
        frame_indices=[0],
        transforms=[recorder],
        writer_batch_size=1,
        force_overwrite=True,
    )

    assert len(recorder.selections) == 1
    selection = recorder.selections[0]
    assert selection.selected_frame_indices == (0,)
    assert selection.source_episode_frame_indices == (0, 1)
    assert selection.is_complete_source_episode is False
    assert selection.is_contiguous_source_slice is True

    target_dataset = RODataset(str(target_path))
    frame0 = target_dataset[0]
    episode = target_dataset.get_meta(Episode, int(frame0["episode_index"]))
    assert episode is not None
    assert episode.info is None
    assert episode.truncated is None
    assert episode.success is None


class _SkipEpisodeTransform(IdentityRODatasetRepackTransform):
    def __init__(self, skip_episode_indices: set[int]) -> None:
        self.skip_episode_indices = skip_episode_indices

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        if context.episode_index in self.skip_episode_indices:
            return None
        return episode_meta


def test_transform_episode_meta_can_skip_episode(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_skip_episode"

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[_SkipEpisodeTransform({0})],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert len(target_dataset) == 2
    frame0 = target_dataset[0]
    assert frame0["value"] == 10
    episode = target_dataset.get_meta(Episode, int(frame0["episode_index"]))
    assert episode is not None
    assert episode.info == {"episode": 1}


def test_transform_mode_rejects_all_episodes_skipped(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="produced no episodes"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_all_skipped"),
            transforms=[_SkipEpisodeTransform({0, 1})],
            writer_batch_size=1,
            force_overwrite=True,
        )


class _StoreFrameTransform(IdentityRODatasetRepackTransform):
    def __init__(self) -> None:
        self.frames: list[DataFrame] = []

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        self.frames.append(frame)
        return frame


def test_cached_repack_episode_does_not_pollute_cached_frames(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    transform = _StoreFrameTransform()

    repack_dataset(
        source_dataset,
        str(tmp_path / "target_cached_frames"),
        transforms=[transform],
        writer_batch_size=1,
        force_overwrite=True,
    )

    assert transform.frames
    assert "index" not in transform.frames[0].features
    assert "episode_index" not in transform.frames[0].features
