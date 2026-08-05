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
import os
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
    StagedDatasetWriteSession,
    TaskData,
)
from robo_orchard_lab.dataset.robot.re_packing import repack_dataset
from robo_orchard_lab.dataset.robot.re_packing._errors import (
    RepackFrameTransformError,
)


@pytest.fixture(autouse=True)
def _use_test_packaging_lock_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XDG_CACHE_HOME",
        str(tmp_path.parent / f".{tmp_path.name}-cache"),
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
    assert hasattr(repack_runner_module, "StagedDatasetWriteSession")
    assert not hasattr(repack_runner_module, "_StagedDatasetWriteSession")
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


def test_staging_cleanup_failure_does_not_mask_repack_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active repack error keeps priority over staging cleanup failure."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    session = StagedDatasetWriteSession(
        target_path=str(tmp_path / "target"),
        force_overwrite=False,
    )
    original_remove_path = staging_module.remove_path
    with pytest.raises(RuntimeError, match="repack failed") as exc_info:
        with session as output:
            workspace_path = DatasetPackagingPaths.resolve(
                output.path
            ).workspace_dir
            Path(workspace_path).mkdir(parents=True, exist_ok=True)

            def fail_workspace_cleanup(
                path: str,
                *,
                missing_ok: bool = True,
            ) -> None:
                if path == workspace_path:
                    raise PermissionError("workspace cleanup failed")
                original_remove_path(path, missing_ok=missing_ok)

            monkeypatch.setattr(
                staging_module,
                "remove_path",
                fail_workspace_cleanup,
            )
            raise RuntimeError("repack failed")

    assert exc_info.value.__notes__ is not None
    assert "workspace cleanup failed" in exc_info.value.__notes__[0]


@pytest.mark.parametrize(
    ("body_error", "expected_type", "expected_message"),
    [
        (None, PermissionError, "workspace cleanup failed"),
        (RuntimeError("write failed"), RuntimeError, "write failed"),
    ],
)
def test_staged_session_exit_preserves_first_error_across_teardown_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_error: RuntimeError | None,
    expected_type: type[BaseException],
    expected_message: str,
) -> None:
    """Later teardown failures are notes, not replacement primary errors."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    class _ReleaseFailureLock:
        def release(self) -> None:
            raise OSError("release failed")

    session = StagedDatasetWriteSession(
        target_path=str(tmp_path / "target"),
        force_overwrite=False,
    )
    original_remove_path = staging_module.remove_path
    original_close_target_identity = (
        staging_module.StagedDatasetWriteSession._close_target_identity_fd
    )

    def fail_workspace_cleanup(
        path: str,
        *,
        missing_ok: bool = True,
    ) -> None:
        if path == workspace_path:
            raise PermissionError("workspace cleanup failed")
        original_remove_path(path, missing_ok=missing_ok)

    def close_then_fail(
        self: StagedDatasetWriteSession,
    ) -> None:
        original_close_target_identity(self)
        raise OSError("identity handle close failed")

    with pytest.raises(expected_type, match=expected_message) as exc_info:
        with session as output:
            workspace_path = DatasetPackagingPaths.resolve(
                output.path
            ).workspace_dir
            Path(workspace_path).mkdir(parents=True, exist_ok=True)
            original_lock = session._coordination_lock
            assert original_lock is not None
            original_lock.release()
            session._coordination_lock = _ReleaseFailureLock()
            monkeypatch.setattr(
                staging_module,
                "remove_path",
                fail_workspace_cleanup,
            )
            monkeypatch.setattr(
                staging_module.StagedDatasetWriteSession,
                "_close_target_identity_fd",
                close_then_fail,
            )
            if body_error is not None:
                raise body_error

    notes = exc_info.value.__notes__ or []
    if body_error is not None:
        assert any("clean staged dataset paths" in note for note in notes)
    assert any(
        "release the dataset target coordination lock" in note
        for note in notes
    )
    assert any(
        "close the staged dataset target identity handle" in note
        for note in notes
    )


def test_staged_session_setup_rollback_preserves_setup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup rollback keeps its triggering error over later teardown errors."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    class _ReleaseFailureLock:
        def acquire(self) -> None:
            pass

        def release(self) -> None:
            raise OSError("release failed")

    target_path = tmp_path / "target"
    target_path.mkdir()
    session = StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=True,
    )
    original_close_target_identity = (
        staging_module.StagedDatasetWriteSession._close_target_identity_fd
    )

    def fail_staging_path(self: StagedDatasetWriteSession) -> str:
        raise RuntimeError("setup failed")

    def close_then_fail(
        self: StagedDatasetWriteSession,
    ) -> None:
        original_close_target_identity(self)
        raise OSError("identity handle close failed")

    monkeypatch.setattr(
        staging_module,
        "_create_coordination_lock",
        lambda _lock_path: _ReleaseFailureLock(),
    )
    monkeypatch.setattr(
        staging_module.StagedDatasetWriteSession,
        "_make_staging_path",
        fail_staging_path,
    )
    monkeypatch.setattr(
        staging_module.StagedDatasetWriteSession,
        "_close_target_identity_fd",
        close_then_fail,
    )

    with pytest.raises(RuntimeError, match="setup failed") as exc_info:
        session.__enter__()

    notes = exc_info.value.__notes__ or []
    assert any(
        "identity handle during setup rollback" in note for note in notes
    )
    assert any(
        "coordination lock during setup rollback" in note for note in notes
    )


def test_staged_session_holds_final_target_coordination_lock(
    tmp_path: Path,
) -> None:
    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_paths = DatasetPackagingPaths.resolve(target_path)

    with StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=False,
    ):
        with pytest.raises(staging_module.filelock.Timeout):
            staging_module.filelock.FileLock(
                target_paths.coordination_lock_path,
                timeout=0,
            ).acquire()


def test_staged_session_preserves_target_that_appears_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    observed_target_inode: int | None = None
    original_publish = staging_module.rename_noreplace

    def inject_external_target(src: str, dst: str) -> None:
        nonlocal observed_target_inode
        if dst == str(target_path):
            target_path.mkdir()
            observed_target_inode = target_path.stat().st_ino
        original_publish(src, dst)

    monkeypatch.setattr(
        staging_module,
        "rename_noreplace",
        inject_external_target,
    )

    with StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=False,
    ) as output:
        Path(output.path).mkdir()

        with pytest.raises(FileExistsError):
            output.commit()

    assert target_path.is_dir()
    assert not list(target_path.iterdir())
    assert target_path.stat().st_ino == observed_target_inode
    assert not list(tmp_path.rglob("*.lock"))


def test_staged_session_preserves_replacement_target_with_force(
    tmp_path: Path,
) -> None:
    """Force only replaces the target identity present when staging began."""

    target_path = tmp_path / "target"
    target_path.mkdir()
    original_marker = target_path / "original"
    original_marker.write_text("original", encoding="utf-8")

    with StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=True,
    ) as output:
        original_marker.unlink()
        target_path.rmdir()
        target_path.mkdir()
        external_marker = target_path / "external"
        external_marker.write_text("external", encoding="utf-8")
        Path(output.path).mkdir()

        with pytest.raises(RuntimeError, match="target changed"):
            output.commit()

    assert external_marker.read_text(encoding="utf-8") == "external"


def test_staged_session_preserves_new_target_with_force(
    tmp_path: Path,
) -> None:
    """Force does not authorize a target created after staging begins."""

    target_path = tmp_path / "target"
    with StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=True,
    ) as output:
        target_path.mkdir()
        external_marker = target_path / "external"
        external_marker.write_text("external", encoding="utf-8")
        Path(output.path).mkdir()

        with pytest.raises(RuntimeError, match="target changed"):
            output.commit()

    assert external_marker.read_text(encoding="utf-8") == "external"


def test_staged_session_restores_target_after_publish_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt after backup creation restores the prior target."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    original_marker = target_path / "original"
    original_marker.write_text("original", encoding="utf-8")
    original_publish = staging_module.rename_noreplace
    staged_path: str | None = None

    def interrupt_staged_publish(src: str, dst: str) -> None:
        if src == staged_path and dst == str(target_path):
            raise KeyboardInterrupt
        original_publish(src, dst)

    with pytest.raises(KeyboardInterrupt):
        with StagedDatasetWriteSession(
            target_path=str(target_path),
            force_overwrite=True,
        ) as output:
            staged_path = output.path
            Path(staged_path).mkdir()
            monkeypatch.setattr(
                staging_module,
                "rename_noreplace",
                interrupt_staged_publish,
            )
            output.commit()

    assert original_marker.read_text(encoding="utf-8") == "original"
    assert staged_path is not None
    assert not os.path.lexists(staged_path)
    assert not list(tmp_path.glob(".target.backup-*"))


def test_staged_session_restores_target_after_backup_rename_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt after moving the target to backup restores that target."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    original_marker = target_path / "original"
    original_marker.write_text("original", encoding="utf-8")
    original_rename = staging_module.os.rename

    def interrupt_after_backup_rename(source: str, destination: str) -> None:
        original_rename(source, destination)
        if source == str(target_path):
            raise KeyboardInterrupt("backup rename interrupted")

    monkeypatch.setattr(
        staging_module.os,
        "rename",
        interrupt_after_backup_rename,
    )
    with pytest.raises(KeyboardInterrupt, match="backup rename interrupted"):
        with StagedDatasetWriteSession(
            target_path=str(target_path),
            force_overwrite=True,
        ) as output:
            Path(output.path).mkdir()
            output.commit()

    assert original_marker.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".target.backup-*"))


@pytest.mark.skipif(
    os.name != "nt",
    reason="requires Windows directory handles",
)
def test_staged_session_overwrites_existing_directory_on_windows(
    tmp_path: Path,
) -> None:
    """Windows identity handles permit the owned target replacement."""

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")

    with StagedDatasetWriteSession(
        target_path=str(target_path),
        force_overwrite=True,
    ) as output:
        staged_path = Path(output.path)
        staged_path.mkdir()
        (staged_path / "new").write_text("new", encoding="utf-8")
        output.commit()

    assert (target_path / "new").read_text(encoding="utf-8") == "new"
    assert not (target_path / "old").exists()


def test_staged_session_surfaces_backup_cleanup_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed backup cleanup reports a post-publication cleanup failure."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")
    original_remove_path = staging_module.remove_path

    def fail_published_backup_cleanup(
        path: str,
        *,
        missing_ok: bool = True,
    ) -> None:
        if (
            Path(path).name.startswith(".target.backup-")
            and (Path(path) / "old").exists()
        ):
            raise PermissionError("backup cleanup failed")
        original_remove_path(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        staging_module,
        "remove_path",
        fail_published_backup_cleanup,
    )
    with pytest.raises(PermissionError, match="backup cleanup failed"):
        with StagedDatasetWriteSession(
            target_path=str(target_path),
            force_overwrite=True,
        ) as output:
            staged_path = Path(output.path)
            staged_path.mkdir()
            (staged_path / "new").write_text("new", encoding="utf-8")
            output.commit()

    assert (target_path / "new").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".target.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old").read_text(encoding="utf-8") == "old"


def test_staged_session_preserves_publish_error_when_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore diagnostics do not mask the publication failure."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")

    def fail_publish_and_restore(src: str, dst: str) -> None:
        if Path(src).name.startswith(".target.tmp-"):
            raise RuntimeError("publish failed")
        raise PermissionError(f"restore failed: {src} -> {dst}")

    monkeypatch.setattr(
        staging_module,
        "rename_noreplace",
        fail_publish_and_restore,
    )

    with pytest.raises(RuntimeError, match="publish failed") as exc_info:
        with StagedDatasetWriteSession(
            target_path=str(target_path),
            force_overwrite=True,
        ) as output:
            Path(output.path).mkdir()
            output.commit()

    assert exc_info.value.__notes__ is not None
    assert (
        "Failed to restore the prior dataset target"
        in (exc_info.value.__notes__[0])
    )
    assert "PermissionError" in exc_info.value.__notes__[0]
    assert not target_path.exists()
    backups = list(tmp_path.glob(".target.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old").read_text(encoding="utf-8") == "old"


def test_directory_cleanup_errors_reach_exception_priority_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle staging cleanup surfaces its first removal error."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    staging_dir = tmp_path / "staging-workspace"
    staging_dir.mkdir()
    session = StagedDatasetWriteSession(
        target_path=str(tmp_path / "target"),
        force_overwrite=False,
    )
    session._staging_path = str(staging_dir)
    session._staging_paths = DatasetPackagingPaths.resolve(staging_dir)
    original_remove_path = staging_module.remove_path

    def fail_directory_cleanup(
        path: str,
        *,
        missing_ok: bool = True,
    ) -> None:
        if path == str(staging_dir):
            raise PermissionError(f"directory cleanup failed: {path}")
        original_remove_path(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        staging_module,
        "remove_path",
        fail_directory_cleanup,
    )

    with pytest.raises(PermissionError, match="directory cleanup failed"):
        session._cleanup_staging_paths()


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
    assert not list(tmp_path.rglob("*.lock"))


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
    assert not list(tmp_path.rglob("*.lock"))


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


def test_staged_repack_holds_the_final_target_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repack serialization uses the final target instead of the stage."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    monkeypatch.setenv(
        "XDG_CACHE_HOME", str(tmp_path.parent / f".{tmp_path.name}-cache")
    )
    target_path = tmp_path / "target"
    paths = DatasetPackagingPaths.resolve(target_path)
    with staging_module.StagedDatasetWriteSession(
        target_path=target_path,
        force_overwrite=False,
    ):
        with pytest.raises(staging_module.filelock.Timeout):
            staging_module.filelock.FileLock(
                paths.coordination_lock_path,
                timeout=0,
            ).acquire()


def test_repack_cleans_private_stage_coordination_locks(
    tmp_path: Path,
) -> None:
    """Repeated repacks leave no newly created private-stage locks."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
    )

    source_dataset = _make_source_dataset(tmp_path)
    target_path = tmp_path / "target"
    target_paths = DatasetPackagingPaths.resolve(target_path)
    lock_dir = Path(target_paths.coordination_lock_path).parent
    initial_locks = set(lock_dir.glob("*.lock"))

    for _ in range(2):
        repack_dataset(
            source_dataset,
            str(target_path),
            writer_batch_size=1,
            force_overwrite=True,
        )

    remaining_locks = set(lock_dir.glob("*.lock"))
    assert remaining_locks <= initial_locks | {
        Path(target_paths.coordination_lock_path)
    }


def test_staged_repack_cleans_private_stage_lock_after_failed_write(
    tmp_path: Path,
) -> None:
    """An aborted repack retires its released direct-writer stage lock."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
        _create_coordination_lock,
    )

    stage_lock_path: Path | None = None
    with pytest.raises(RuntimeError, match="write failed"):
        with StagedDatasetWriteSession(
            target_path=tmp_path / "target",
            force_overwrite=False,
        ) as stage:
            stage_paths = DatasetPackagingPaths.resolve(stage.path)
            stage_lock_path = Path(stage_paths.coordination_lock_path)
            stage_lock = _create_coordination_lock(
                stage_paths.coordination_lock_path
            )
            stage_lock.acquire(timeout=0)
            stage_lock.release()
            stage_lock_path.touch(exist_ok=True)
            assert stage_lock_path.is_file()
            raise RuntimeError("write failed")

    assert stage_lock_path is not None
    assert not stage_lock_path.exists()


def test_staged_repack_retires_private_lock_after_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace cleanup error does not strand the released stage lock."""

    from robo_orchard_lab.dataset.packaging_paths import (
        DatasetPackagingPaths,
        _create_coordination_lock,
    )
    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    stage_lock_path: Path | None = None
    with pytest.raises(RuntimeError, match="write failed"):
        with StagedDatasetWriteSession(
            target_path=tmp_path / "target",
            force_overwrite=False,
        ) as stage:
            stage_paths = DatasetPackagingPaths.resolve(stage.path)
            Path(stage_paths.workspace_dir).mkdir()
            stage_lock_path = Path(stage_paths.coordination_lock_path)
            stage_lock = _create_coordination_lock(
                stage_paths.coordination_lock_path
            )
            stage_lock.acquire(timeout=0)
            stage_lock.release()
            stage_lock_path.touch(exist_ok=True)
            original_remove_path = staging_module.remove_path

            def fail_workspace_cleanup(
                path: str,
                *,
                missing_ok: bool = True,
            ) -> None:
                if path == stage_paths.workspace_dir:
                    raise OSError("workspace cleanup failed")
                original_remove_path(path, missing_ok=missing_ok)

            monkeypatch.setattr(
                staging_module,
                "remove_path",
                fail_workspace_cleanup,
            )
            raise RuntimeError("write failed")

    assert stage_lock_path is not None
    assert not stage_lock_path.exists()


def test_staged_repack_restores_the_prior_target_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed staged publication restores the replaced target generation."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")
    original_rename = staging_module.rename_noreplace
    staging_path: str | None = None

    def fail_final_publish(source: str, target: str) -> None:
        if source == staging_path:
            raise RuntimeError("publish failed")
        original_rename(source, target)

    with pytest.raises(RuntimeError, match="publish failed"):
        with staging_module.StagedDatasetWriteSession(
            target_path=target_path,
            force_overwrite=True,
        ) as stage:
            staging_path = stage.path
            Path(staging_path).mkdir()
            monkeypatch.setattr(
                staging_module,
                "rename_noreplace",
                fail_final_publish,
            )
            stage.commit()

    assert (target_path / "old").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.backup-*"))


@pytest.mark.parametrize(
    ("initial_target", "mutate_target"),
    [
        (
            False,
            lambda target: (
                target.mkdir(),
                (target / "unowned").write_text("unowned", encoding="utf-8"),
            ),
        ),
        (
            True,
            lambda target: (
                (target / "old").unlink(),
                target.rmdir(),
            ),
        ),
        (
            True,
            lambda target: (
                target.rename(target.parent / "original"),
                target.mkdir(),
                (target / "replacement").write_text(
                    "replacement", encoding="utf-8"
                ),
            ),
        ),
    ],
    ids=("appears", "disappears", "is-replaced"),
)
def test_staged_repack_rejects_target_identity_changes(
    tmp_path: Path,
    initial_target: bool,
    mutate_target,
) -> None:
    """Publication never removes a target changed during repacking."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    if initial_target:
        target_path.mkdir()
        (target_path / "old").write_text("old", encoding="utf-8")

    with staging_module.StagedDatasetWriteSession(
        target_path=target_path,
        force_overwrite=True,
    ) as stage:
        Path(stage.path).mkdir()
        mutate_target(target_path)
        with pytest.raises(RuntimeError, match="target changed"):
            stage.commit()

    if initial_target and not target_path.exists():
        assert not target_path.exists()
    else:
        assert target_path.exists()
    if initial_target and (tmp_path / "original").exists():
        assert (tmp_path / "original" / "old").read_text(
            encoding="utf-8"
        ) == "old"


def test_staged_repack_rechecks_identity_after_backup_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target replacement at backup rename is restored and preserved."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")
    replaced_original = tmp_path / "replaced_original"
    original_rename = staging_module.os.rename
    replaced = False

    def replace_before_backup(source: str, target: str) -> None:
        nonlocal replaced
        if source == str(target_path) and not replaced:
            target_path.rename(replaced_original)
            target_path.mkdir()
            (target_path / "replacement").write_text(
                "replacement", encoding="utf-8"
            )
            replaced = True
        original_rename(source, target)

    monkeypatch.setattr(staging_module.os, "rename", replace_before_backup)
    with pytest.raises(RuntimeError, match="target changed"):
        with staging_module.StagedDatasetWriteSession(
            target_path=target_path,
            force_overwrite=True,
        ) as stage:
            Path(stage.path).mkdir()
            stage.commit()

    assert (
        replaced_original.joinpath("old").read_text(encoding="utf-8") == "old"
    )
    assert (
        target_path.joinpath("replacement").read_text(encoding="utf-8")
        == "replacement"
    )
    assert not list(tmp_path.glob(".target.backup-*"))


def test_staged_repack_restores_target_after_backup_identity_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-backup identity-read failure restores the prior target."""

    from robo_orchard_lab.dataset.robot.packaging import (
        _staging as staging_module,
    )

    target_path = tmp_path / "target"
    target_path.mkdir()
    (target_path / "old").write_text("old", encoding="utf-8")
    original_read_target_identity = staging_module._read_target_identity

    def fail_backup_identity_read(path: str) -> tuple[int, int, int] | None:
        if Path(path).name.startswith(".target.backup-"):
            raise OSError("identity read failed")
        return original_read_target_identity(path)

    monkeypatch.setattr(
        staging_module,
        "_read_target_identity",
        fail_backup_identity_read,
    )
    with pytest.raises(OSError, match="identity read failed"):
        with staging_module.StagedDatasetWriteSession(
            target_path=target_path,
            force_overwrite=True,
        ) as stage:
            Path(stage.path).mkdir()
            stage.commit()

    assert (target_path / "old").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.backup-*"))
