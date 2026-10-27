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

"""LMDB dataset resource ownership on close, reopen and failed construction."""

import pytest
from robo_orchard_core.utils.config import ClassType

import robo_orchard_lab.dataset.lmdb.base_lmdb_dataset as base_module
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataset,
)
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DatasetItem,
    DictIterableDataset,
)

__all__ = []


class TinyLmdbDataset(BaseLmdbManipulationDataset):
    def __getitem__(self, index):
        _, _, step = self._get_indices(index)
        return int(step)


class TinyLmdbItem(DatasetItem[TinyLmdbDataset]):
    class_type: ClassType[TinyLmdbDataset] = TinyLmdbDataset
    path: str

    def get_dataset_row_num(self):
        return 4

    def _create_dataset(self):
        return TinyLmdbDataset(self.path)


@pytest.fixture()
def tiny_lmdb(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("index", "meta", "image", "depth"):
        writer = Lmdb(str(root / name), map_size=1024**2)
        try:
            if name == "index":
                writer.write(0, dict(uuid="episode", num_steps=4, task="test"))
                writer.write("__len__", 1)
            else:
                writer.write("fixture", b"value")
        finally:
            writer.close()
    return str(root)


class TestLmdbReaderClose:
    @pytest.mark.parametrize("lazy", [False, True])
    def test_all_handles_close_and_reopen(self, tiny_lmdb, lazy):
        dataset = TinyLmdbDataset(tiny_lmdb, lazy_init=lazy)
        dataset.close()
        assert not dataset.initialized
        assert dataset[2] == 2
        wrappers = [
            group[0]
            for group in (
                dataset.idx_lmdbs,
                dataset.meta_lmdbs,
                dataset.img_lmdbs,
                dataset.depth_lmdbs,
            )
        ]
        assert all(wrapper.env is not None for wrapper in wrappers)
        dataset.close()
        dataset.close()
        assert all(
            wrapper.env is None and wrapper.txn is None for wrapper in wrappers
        )
        assert dataset[3] == 3
        dataset.close()

    def test_initialization_failure_closes_opened_index(
        self, tiny_lmdb, monkeypatch
    ):
        writer = Lmdb(tiny_lmdb + "/index")
        try:
            writer.write(0, dict(uuid="invalid-without-num-steps"))
        finally:
            writer.close()
        wrappers = []
        original = base_module.Lmdb

        def track_wrapper(*args, **kwargs):
            wrapper = original(*args, **kwargs)
            wrappers.append(wrapper)
            return wrapper

        monkeypatch.setattr(base_module, "Lmdb", track_wrapper)
        with pytest.raises(ValueError):
            TinyLmdbDataset(tiny_lmdb)
        assert wrappers
        assert all(wrapper.env is None for wrapper in wrappers)

    def test_close_attempts_other_groups_and_can_retry(self, tiny_lmdb):
        class Wrapper:
            def __init__(self, fail=False):
                self.fail = fail
                self.calls = 0

            def close(self):
                self.calls += 1
                if self.fail and self.calls == 1:
                    raise RuntimeError("fixture close failure")

        dataset = TinyLmdbDataset(tiny_lmdb, lazy_init=True)
        wrappers = [Wrapper(fail=True), Wrapper(), Wrapper(), Wrapper()]
        for name, wrapper in zip(
            ("idx_lmdbs", "meta_lmdbs", "img_lmdbs", "depth_lmdbs"),
            wrappers,
            strict=True,
        ):
            setattr(dataset, name, [wrapper])
        with pytest.raises(RuntimeError, match="fixture close failure"):
            dataset.close()
        assert [wrapper.calls for wrapper in wrappers] == [1, 1, 1, 1]
        dataset.close()
        assert [wrapper.calls for wrapper in wrappers] == [2, 1, 1, 1]

    def test_finite_item_releases_and_reopens_real_wrappers(
        self, tiny_lmdb, tmp_path, monkeypatch
    ):
        wrappers = []
        original = base_module.Lmdb

        def track_wrapper(*args, **kwargs):
            wrapper = original(*args, **kwargs)
            wrappers.append(wrapper)
            return wrapper

        monkeypatch.setattr(base_module, "Lmdb", track_wrapper)
        dataset = DictIterableDataset(
            [TinyLmdbItem(path=tiny_lmdb)],
            offload_num_batches=1,
            offload_dir=str(tmp_path / "offload"),
        )
        assert list(dataset) == [0, 1, 2, 3]
        assert len(wrappers) == 16
        assert all(
            wrapper.env is None and wrapper.txn is None for wrapper in wrappers
        )
