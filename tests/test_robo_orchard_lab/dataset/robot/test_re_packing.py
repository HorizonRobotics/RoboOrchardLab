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
import importlib.util
from collections.abc import Iterable
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
    IdentityEpisodePackagingTransform,
    InstructionData,
    RobotData,
    TaskData,
)
from robo_orchard_lab.dataset.robot.re_packing import repack_dataset
from robo_orchard_lab.dataset.robot.re_packing._errors import (
    RepackFrameTransformError,
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


class _TestEpisodePackaging(EpisodePackaging):
    def __init__(
        self,
        episode_meta: EpisodeMeta | None,
        frames: Iterable[DataFrame],
    ) -> None:
        self._episode_meta = episode_meta
        self._frames = frames

    def generate_episode_meta(self) -> EpisodeMeta | None:
        return self._episode_meta

    def generate_frames(self):
        yield from self._frames


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


def test_default_repack_uses_canonical_runner_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_default_canonical"

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
    assert episode.info == {"episode": 0}
    assert episode.truncated is False
    assert episode.success is True


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


def test_default_repack_with_all_frames_preserves_episode_links(
    tmp_path: Path,
) -> None:
    source_dataset = _make_linked_source_dataset(tmp_path)
    target_path = tmp_path / "target_default_all_frames_links"

    repack_dataset(
        source_dataset,
        str(target_path),
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
    episode_indices = []
    for episode_index in range(target_dataset.episode_num):
        episode = target_dataset.get_meta(Episode, episode_index)
        assert episode is not None
        episode_indices.append(episode.index)
    assert episode_indices == [0, 1, 2]
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


def test_transform_contract_types_stay_off_robot_root_namespace() -> None:
    import robo_orchard_lab.dataset.robot as robot_dataset

    assert hasattr(robot_dataset, "repack_dataset")
    assert not hasattr(robot_dataset, "RODatasetRepackTransform")


def test_repacking_does_not_export_old_transform_contracts() -> None:
    import robo_orchard_lab.dataset.robot as robot_dataset
    import robo_orchard_lab.dataset.robot.re_packing as repacking

    assert not hasattr(repacking, "RODatasetRepackTransform")
    assert not hasattr(repacking, "IdentityRODatasetRepackTransform")
    assert not hasattr(repacking, "RODatasetRepackEpisode")
    assert not hasattr(repacking, "RODatasetRepackFrame")
    assert not hasattr(repacking, "EpisodePackagingTransform")
    assert not hasattr(repacking, "DefaultRePackingEpisodeHelper")
    assert not hasattr(repacking, "RePackingDatasetHelper")
    assert not hasattr(repacking, "RODatasetEpisodeRepackTransform")
    assert not hasattr(repacking, "IdentityRODatasetEpisodeRepackTransform")
    assert not hasattr(repacking, "RepackFrameTransformError")
    assert not hasattr(robot_dataset, "DefaultRePackingEpisodeHelper")
    assert not hasattr(robot_dataset, "RODatasetEpisodeRepackTransform")


def test_repack_runner_uses_unified_internal_names() -> None:
    from robo_orchard_lab.dataset.robot.packaging import (
        _episode as packaging_episode_module,
    )
    from robo_orchard_lab.dataset.robot.re_packing import (
        _runner as repack_runner_module,
        _source as repack_source_module,
    )

    assert (
        importlib.util.find_spec(
            "robo_orchard_lab.dataset.robot._packaging_transform"
        )
        is None
    )
    assert not hasattr(packaging_episode_module, "_MappedEpisodePackaging")
    assert not hasattr(packaging_episode_module, "_CachedEpisodePackaging")
    assert not hasattr(
        packaging_episode_module,
        "_EpisodePackagingTransformPipeline",
    )
    assert not hasattr(packaging_episode_module, "_EpisodePackagingView")
    assert hasattr(packaging_episode_module, "EpisodePackagingView")
    assert packaging_episode_module.__all__ == [
        "ComposedEpisodePackagingTransform",
        "DataFrame",
        "EpisodeMeta",
        "EpisodePackaging",
        "EpisodePackagingTransform",
        "EpisodePackagingView",
        "IdentityEpisodePackagingTransform",
    ]
    assert (
        importlib.util.find_spec(
            "robo_orchard_lab.dataset.robot.re_packing.runner"
        )
        is None
    )
    assert (
        importlib.util.find_spec(
            "robo_orchard_lab.dataset.robot.re_packing.source"
        )
        is None
    )
    assert not hasattr(repack_runner_module, "_SourceRepackEpisode")
    assert not hasattr(repack_runner_module, "TransformRepackRunner")
    assert not hasattr(repack_runner_module, "transform_repack_dataset")
    assert not hasattr(repack_runner_module, "_RepackEpisodeRunner")
    assert not hasattr(repack_runner_module, "_StagedDatasetOutput")
    assert hasattr(repack_runner_module, "RepackEpisodeRunner")
    assert hasattr(repack_runner_module, "_StagedDatasetWriteSession")
    assert hasattr(repack_runner_module, "repack_dataset")
    assert not hasattr(repack_runner_module, "_run_repack_dataset")
    assert hasattr(repack_source_module, "SourceReader")
    assert hasattr(repack_source_module, "SourceEpisodeChunk")


def test_identity_transform_has_no_dispatch_mode_flag() -> None:
    assert not hasattr(
        IdentityEpisodePackagingTransform(),
        "is_frame_level_transform",
    )


def test_repack_dataset_rejects_uri_target_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_orchard_lab.dataset.robot.re_packing import (
        _runner as repack_runner_module,
    )

    class _MinimalSourceDataset:
        features = hg_datasets.Features({"value": hg_datasets.Value("int64")})

    class _UnexpectedDatasetPackaging:
        def __init__(self, features: hg_datasets.Features) -> None:
            del features

        def packaging(self, *args, **kwargs) -> None:
            del args, kwargs
            raise AssertionError("URI target path reached dataset writing.")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        repack_runner_module,
        "DatasetPackaging",
        _UnexpectedDatasetPackaging,
    )

    with pytest.raises(ValueError, match="URI paths are not supported"):
        repack_dataset(
            _MinimalSourceDataset(),  # type: ignore[arg-type]
            "s3://bucket/target_ro_dataset",
            transforms=[],
            force_overwrite=True,
        )


def test_repack_dataset_rejects_removed_legacy_keywords(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(TypeError, match="packing_impl"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_custom_helper"),
            packing_impl=object,
        )

    with pytest.raises(TypeError, match="fail_fast"):
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


class _RequireValueTransform(IdentityEpisodePackagingTransform):
    def prepare_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        if "value" not in features:
            raise ValueError(f"{self!r} requires missing columns: ['value'].")
        return features


class _RequireReservedColumnTransform(IdentityEpisodePackagingTransform):
    def prepare_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        del features
        raise ValueError(f"{self!r} requires reserved columns: ['index'].")


def test_prepare_features_respects_columns_projection(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="requires missing columns"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_missing_required_column"),
            columns=["text"],
            transforms=[_RequireValueTransform()],
        )

    with pytest.raises(ValueError, match="requires reserved columns"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_reserved_required_column"),
            transforms=[_RequireReservedColumnTransform()],
        )


class _AddValueCopyTransform(IdentityEpisodePackagingTransform):
    def prepare_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        updated = hg_datasets.Features(features.copy())
        updated["value_copy"] = hg_datasets.Value("int64")
        return updated

    def transform_frame(
        self,
        frame: DataFrame,
    ) -> DataFrame:
        features = frame.features.copy()
        features["value_copy"] = features["value"]
        return replace(frame, features=features)


class _ObserveValueCopyTransform(IdentityEpisodePackagingTransform):
    def __init__(self) -> None:
        self.prepare_saw_value_copy = False

    def prepare_features(
        self, features: hg_datasets.Features
    ) -> hg_datasets.Features:
        self.prepare_saw_value_copy = "value_copy" in features
        if "value_copy" not in features:
            raise ValueError(
                f"{self!r} requires missing columns: ['value_copy']."
            )
        return features


def test_transform_feature_order_and_prepare_features_contract(
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


class _AppendTextFrameTransform(IdentityEpisodePackagingTransform):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def transform_frame(
        self,
        frame: DataFrame,
    ) -> DataFrame:
        features = frame.features.copy()
        features["text"] = features["text"] + self.suffix
        return replace(frame, features=features)


def test_mixed_frame_and_episode_transforms_keep_user_order(
    tmp_path: Path,
) -> None:
    class _AppendTextEpisodeTransform(IdentityEpisodePackagingTransform):
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        def transform_episode(
            self,
            episode: EpisodePackaging,
        ) -> EpisodePackaging | None:
            episode_meta = episode.generate_episode_meta()
            if episode_meta is None:
                return None

            def frames():
                for frame in episode.generate_frames():
                    features = frame.features.copy()
                    features["text"] = features["text"] + self.suffix
                    yield replace(frame, features=features)

            return _TestEpisodePackaging(episode_meta, frames())

    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_mixed_transform_order"
    frame_transform = _AppendTextFrameTransform("-frame")
    episode_transform = _AppendTextEpisodeTransform("-episode")

    repack_dataset(
        source_dataset,
        str(target_path),
        transforms=[
            frame_transform,
            episode_transform,
            _AppendTextFrameTransform("-last"),
        ],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    assert target_dataset[0]["text"] == (
        "episode-0-frame-0-frame-episode-last"
    )


class _MutateMetadataTransform(IdentityEpisodePackagingTransform):
    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta | None,
    ) -> EpisodeMeta | None:
        if episode_meta is None:
            return None
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


class _MutateTargetLinkTransform(IdentityEpisodePackagingTransform):
    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta | None,
    ) -> EpisodeMeta | None:
        if episode_meta is None:
            return None
        episode_meta.episode.index = 999
        return episode_meta


def test_transform_cannot_mutate_target_episode_linkage(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="target episode linkage"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_mutated_episode_linkage"),
            transforms=[_MutateTargetLinkTransform()],
            writer_batch_size=1,
            force_overwrite=True,
        )


class _InvalidFeatureReturnTransform(IdentityEpisodePackagingTransform):
    def prepare_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        del features
        return {"value": hg_datasets.Value("int64")}  # type: ignore[return-value]


class _ReservedFeatureTransform(IdentityEpisodePackagingTransform):
    def prepare_features(
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

    with pytest.raises(ValueError, match="reserved frame-table columns"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_reserved_features"),
            transforms=[_ReservedFeatureTransform()],
        )


class _ReturnNoneFrameTransform(IdentityEpisodePackagingTransform):
    def transform_frame(
        self,
        frame: DataFrame,
    ) -> DataFrame:
        del frame
        return None  # type: ignore[return-value]


def _assert_repack_frame_transform_error(
    exc: BaseException,
) -> None:
    assert isinstance(exc, RepackFrameTransformError)


def test_transform_frame_returning_none_is_rejected(tmp_path: Path) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(Exception) as exc_info:
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_none_frame"),
            transforms=[_ReturnNoneFrameTransform()],
            writer_batch_size=1,
            force_overwrite=True,
        )

    exc = exc_info.value
    _assert_repack_frame_transform_error(exc)
    assert isinstance(exc.__cause__, TypeError)
    assert exc.original_error is exc.__cause__
    assert "must return DataFrame, got None" in str(exc)
    assert "source_episode_index=0" in str(exc)
    assert "frame_offset=0" in str(exc)
    assert "source_frame_index=0" in str(exc)


class _FailSecondSelectedFrameTransform(IdentityEpisodePackagingTransform):
    def __init__(self) -> None:
        self._frame_count = 0
        self.original_error: ValueError | None = None

    def transform_frame(
        self,
        frame: DataFrame,
    ) -> DataFrame:
        self._frame_count += 1
        if self._frame_count == 2:
            self.original_error = ValueError("bad frame")
            raise self.original_error
        return frame


def test_transform_frame_failure_has_source_frame_context(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    transform = _FailSecondSelectedFrameTransform()

    with pytest.raises(Exception) as exc_info:
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_bad_transform_frame"),
            frame_indices=[2, 3],
            transforms=[transform],
            writer_batch_size=1,
            force_overwrite=True,
        )

    exc = exc_info.value
    _assert_repack_frame_transform_error(exc)
    assert transform.original_error is not None
    assert exc.__cause__ is transform.original_error
    assert exc.original_error is transform.original_error
    assert exc.source_episode_index == 1
    assert exc.frame_offset == 1
    assert exc.source_frame_index == 3
    message = str(exc)
    assert "ValueError: bad frame" in message
    assert "source_episode_index=1" in message
    assert "frame_offset=1" in message
    assert "source_frame_index=3" in message


class _DropFrameFeatureTransform(IdentityEpisodePackagingTransform):
    def transform_frame(
        self,
        frame: DataFrame,
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


class _DropOneEpisodeFrameTransform(IdentityEpisodePackagingTransform):
    def transform_episode(
        self,
        episode: EpisodePackaging,
    ) -> EpisodePackaging | None:
        episode_meta = episode.generate_episode_meta()
        if episode_meta is None:
            return None

        def frames():
            iterator = iter(episode.generate_frames())
            next(iterator)
            yield from iterator

        return _TestEpisodePackaging(episode_meta, frames())


def test_episode_transform_cannot_change_row_count(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="row count"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_drop_frame"),
            transforms=[_DropOneEpisodeFrameTransform()],
            writer_batch_size=1,
            force_overwrite=True,
        )


def test_partial_selection_clears_episode_metadata(
    tmp_path: Path,
) -> None:
    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target_partial_selection"

    repack_dataset(
        source_dataset,
        str(target_path),
        frame_indices=[0],
        transforms=[],
        writer_batch_size=1,
        force_overwrite=True,
    )

    target_dataset = RODataset(str(target_path))
    frame0 = target_dataset[0]
    episode = target_dataset.get_meta(Episode, int(frame0["episode_index"]))
    assert episode is not None
    assert episode.info is None
    assert episode.truncated is None
    assert episode.success is None


class _SkipEpisodeTransform(IdentityEpisodePackagingTransform):
    def __init__(self, skip_episode_indices: set[int]) -> None:
        self.skip_episode_indices = skip_episode_indices

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta | None,
    ) -> EpisodeMeta | None:
        if episode_meta is None:
            return None
        episode_id = (
            episode_meta.episode.info.get("episode")
            if episode_meta.episode.info is not None
            else None
        )
        if episode_id in self.skip_episode_indices:
            return None
        return episode_meta


class _SkipEpisodeAndCountFramesTransform(IdentityEpisodePackagingTransform):
    def __init__(self, skip_episode_indices: set[int]) -> None:
        self.skip_episode_indices = skip_episode_indices
        self.frame_transform_count = 0

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta | None,
    ) -> EpisodeMeta | None:
        if episode_meta is None:
            return None
        episode_id = (
            episode_meta.episode.info.get("episode")
            if episode_meta.episode.info is not None
            else None
        )
        if episode_id in self.skip_episode_indices:
            return None
        return episode_meta

    def transform_frame(
        self,
        frame: DataFrame,
    ) -> DataFrame:
        self.frame_transform_count += 1
        return frame


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


def test_skipped_episode_does_not_read_source_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_orchard_lab.dataset.robot.re_packing import (
        _runner as runner_module,
    )

    source_dataset = _make_source_dataset(tmp_path)
    transform = _SkipEpisodeAndCountFramesTransform({0, 1})
    read_count = 0
    original_iter_packaging_frames = (
        runner_module.SourceReader.iter_packaging_frames
    )

    def count_iter_packaging_frames(self, frame_indices):
        nonlocal read_count
        read_count += 1
        yield from original_iter_packaging_frames(self, frame_indices)

    monkeypatch.setattr(
        runner_module.SourceReader,
        "iter_packaging_frames",
        count_iter_packaging_frames,
    )

    with pytest.raises(ValueError, match="produced no episodes"):
        repack_dataset(
            source_dataset,
            str(tmp_path / "target_skip_without_reading_frames"),
            transforms=[transform],
            writer_batch_size=1,
            force_overwrite=True,
        )

    assert transform.frame_transform_count == 0
    assert read_count == 0


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


class _StoreFrameTransform(IdentityEpisodePackagingTransform):
    def __init__(self) -> None:
        self.frames: list[DataFrame] = []

    def transform_frame(
        self,
        frame: DataFrame,
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
