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

"""Tests for basic length-aware iterable datasets and batching views."""

import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    BatchedArrayDataset,
    TestIterableDatasetMixin,
    _get_dataloader_multiprocessing_context,
)
from torch.utils.data import (
    Dataset,
    IterableDataset as TorchIterableDataset,
)

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    DatasetWithIndices,
    IterableWithLenDataset,
    ShardConfig,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE,
)
from robo_orchard_lab.dataset.sampler import (
    DEFAULT_SAMPLING_SEED,
)


class DelayedBatchedArrayDataset(BatchedArrayDataset):
    def __init__(self, data: list, delay: float) -> None:
        super().__init__(data)
        self._delay = delay
        self._getitems_call_count = 0

    def __getitems__(self, indices: list[int]) -> list:
        if self._delay > 0 and self._getitems_call_count % 2 == 0:
            time.sleep(self._delay)
        self._getitems_call_count += 1
        return super().__getitems__(indices)


class ArrayIterableDataset(TorchIterableDataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)


@pytest.fixture()
def dummy_array_dataset():
    return ArrayDataset(data=list(range(0, 10)))


class TestNonIterableDatasetMixinDataLoader:
    def test_map_dataset_accepts_shuffle_config(self):
        dataset = ArrayDataset(data=list(range(10)))

        with pytest.warns(UserWarning) as warning_records:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=ShuffleConfig(shuffle=True, chunk_size=4),
                num_workers=0,
            )

        flattened_items: list[int] = []
        for batch in dataloader:
            flattened_items.extend(cast(torch.Tensor, batch).tolist())

        assert sorted(flattened_items) == dataset.data
        assert any(
            "ShuffleConfig.chunk_size" in str(record.message)
            for record in warning_records
        )

    def test_iterable_dataset_accepts_shuffle_config_without_error(self):
        dataset = ArrayIterableDataset(data=list(range(10)))

        with pytest.warns(UserWarning) as warning_records:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=ShuffleConfig(shuffle=True, chunk_size=4),
                num_workers=0,
            )

        flattened_items: list[int] = []
        for batch in dataloader:
            flattened_items.extend(cast(torch.Tensor, batch).tolist())

        assert flattened_items == dataset.data
        warning_messages = [str(record.message) for record in warning_records]
        assert any(
            "ShuffleConfig.chunk_size" in message
            for message in warning_messages
        )
        assert any(
            "Resetting `shuffle=False`" in message
            for message in warning_messages
        )


class TestIterableWithLenDataset(TestIterableDatasetMixin):
    @pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
    def test_sampling_seed_checkpoint_restores_consumed_prefix(self, seed):
        def loader():
            return DataLoader(
                IterableWithLenDataset(
                    ArrayDataset(list(range(32))), shuffle=True, seed=seed
                ),
                batch_size=2,
            )

        original = loader()
        iterator = iter(original)
        next(iterator)
        next(iterator)
        state = original.state_dict()
        remaining = [batch.tolist() for batch in iterator]
        resumed = loader()
        resumed.load_state_dict(state)
        assert resumed.state_dict()["seed"] == seed
        with pytest.warns(UserWarning, match="replaying the batch prefix"):
            assert [batch.tolist() for batch in resumed] == remaining

    def test_sampling_seed_controls_private_stream_and_survives_views(self):
        source = ArrayDataset(list(range(32)))
        default = IterableWithLenDataset(source, shuffle=True)
        explicit_default = IterableWithLenDataset(
            source, shuffle=True, seed=DEFAULT_SAMPLING_SEED
        )
        custom = IterableWithLenDataset(source, shuffle=True, seed=17)
        reference = IterableWithLenDataset(
            source, shuffle=True, generator=torch.Generator().manual_seed(17)
        )
        assert list(default) == list(explicit_default)
        custom_rows = list(custom)
        assert custom_rows == list(reference)
        assert custom_rows != list(
            IterableWithLenDataset(source, shuffle=True)
        )
        stream = custom.indice_sampler.generator
        before = stream.get_state().clone()
        views = [custom.take(slice(0, 16)), custom.shard(2, 0)]
        loader = DataLoader(
            custom, batch_size=4, use_dataset_side_batching=True
        )
        views.append(loader.dataset)
        assert torch.equal(stream.get_state(), before)
        for view in views:
            assert view._sampling_seed == 17
            assert view.indice_sampler.generator is stream

    @pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5, "42"])
    def test_sampling_seed_rejects_invalid_values(self, seed):
        with pytest.raises(ValueError, match="seed must be"):
            IterableWithLenDataset(ArrayDataset([0]), seed=seed)

    @pytest.mark.parametrize("numpy_generator", [False, True])
    def test_sampling_seed_and_explicit_generator_are_exclusive(
        self, numpy_generator
    ):
        generator = (
            np.random.default_rng(7)
            if numpy_generator
            else torch.Generator().manual_seed(7)
        )
        dataset = IterableWithLenDataset(
            ArrayDataset([0]), generator=generator
        )
        assert dataset.indice_sampler.generator is generator
        assert dataset._sampling_seed is None
        with pytest.raises(ValueError, match="cannot be combined"):
            IterableWithLenDataset(
                ArrayDataset([0]), generator=generator, seed=7
            )

    @pytest.fixture(params=["dummy_array_dataset"])
    def total_batch_consistency_test_dataset(self, request):
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize(
        "resample_ratio,expected_length",
        [
            (0.5, 5),
            (1.0, 10),
            (2.5, 25),
        ],
    )
    def test_resample_ratio_controls_exact_logical_row_count(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: float,
        expected_length: int,
    ):
        dataset = DatasetWithIndices(dummy_array_dataset).to_iterable_dataset(
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=resample_ratio,
        )

        items = list(dataset)

        assert len(items) == dataset.total_iterator_length == expected_length
        if resample_ratio < 1:
            assert len(set(items)) == expected_length
        else:
            for cycle_start in range(0, expected_length - 9, 10):
                assert sorted(items[cycle_start : cycle_start + 10]) == list(
                    range(10)
                )

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_oversampling_prefetch_is_independent_of_source_latency(
        self,
        generator_kind: str,
    ):
        """Seeded source cycles and reservoir shuffle use separate streams."""

        def collect(delay: float) -> list[int]:
            generator: torch.Generator | np.random.Generator
            if generator_kind == "torch":
                generator = torch.Generator().manual_seed(7)
            else:
                generator = np.random.default_rng(7)
            dataset = DatasetWithIndices(
                DelayedBatchedArrayDataset(list(range(12)), delay)
            ).to_iterable_dataset(
                shuffle=ShuffleConfig(
                    shuffle=True,
                    chunk_size=2,
                    prefetch_factor=2,
                ),
                generator=generator,
                resample_ratio=10.0,
            )
            return list(dataset)

        fast_items = collect(delay=0.0)
        delayed_items = collect(delay=0.001)

        assert fast_items == delayed_items
        assert len(fast_items) == 120
        assert all(fast_items.count(value) == 10 for value in range(12))

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_size_one_prefetch_preserves_sampling_generator_stream(
        self,
        generator_kind: str,
    ):
        """Size-one prefetch is a pass-through without RNG splitting."""

        if generator_kind == "torch":
            prefetch_generator: torch.Generator | np.random.Generator = (
                torch.Generator().manual_seed(7)
            )
            direct_generator: torch.Generator | np.random.Generator = (
                torch.Generator().manual_seed(7)
            )
        else:
            prefetch_generator = np.random.default_rng(7)
            direct_generator = np.random.default_rng(7)

        shuffle_config = ShuffleConfig(
            shuffle=True,
            chunk_size=1,
            prefetch_factor=1,
        )
        prefetch_items = list(
            DatasetWithIndices(
                ArrayDataset(list(range(12)))
            ).to_iterable_dataset(
                shuffle=shuffle_config,
                generator=prefetch_generator,
                resample_ratio=2.0,
            )
        )
        direct_dataset = DatasetWithIndices(
            ArrayDataset(list(range(12)))
        ).to_iterable_dataset(
            shuffle=shuffle_config,
            generator=direct_generator,
            resample_ratio=2.0,
        )
        direct_items = list(
            direct_dataset._iter_impl(direct_dataset.indice_sampler)
        )

        assert prefetch_items == direct_items
        if isinstance(prefetch_generator, torch.Generator):
            assert isinstance(direct_generator, torch.Generator)
            assert torch.equal(
                prefetch_generator.get_state(),
                direct_generator.get_state(),
            )
        else:
            assert isinstance(direct_generator, np.random.Generator)
            assert (
                prefetch_generator.bit_generator.state
                == direct_generator.bit_generator.state
            )

    @pytest.mark.parametrize("prefetch_factor", [0, -1])
    def test_nonpositive_sample_prefetch_size_still_raises(
        self,
        dummy_array_dataset: ArrayDataset,
        prefetch_factor: int,
    ):
        dataset = DatasetWithIndices(dummy_array_dataset).to_iterable_dataset(
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=1,
                prefetch_factor=prefetch_factor,
            ),
        )

        with pytest.raises(
            ValueError,
            match="prefetch_size must be greater than 0",
        ):
            list(dataset)

    @pytest.mark.parametrize(
        "resample_ratio,exception",
        [
            (0.0, ValueError),
            (-1.0, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
            ("1.0", TypeError),
        ],
    )
    def test_resample_ratio_validates_public_contract(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: Any,
        exception: type[Exception],
    ):
        with pytest.raises(exception):
            IterableWithLenDataset(
                dummy_array_dataset,
                shuffle=True,
                resample_ratio=resample_ratio,
            )

    @pytest.mark.parametrize("resample_ratio", [0.5, 2.0])
    def test_resample_ratio_requires_shuffle(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: float,
    ):
        with pytest.raises(ValueError, match="require shuffle=True"):
            IterableWithLenDataset(
                dummy_array_dataset,
                shuffle=False,
                resample_ratio=resample_ratio,
            )

    def test_resample_ratio_batches_the_complete_stream_once(self):
        dataset = IterableWithLenDataset(
            ArrayDataset([0, 1]),
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=True,
            ),
        )

        batches = list(dataset)

        assert len(batches) == 1
        assert sorted(batches[0].tolist()) == [0, 0, 1, 1]

    @pytest.mark.parametrize("drop_last_batch", [False, True])
    def test_explicit_batch_iteration(self, drop_last_batch):
        dataset = IterableWithLenDataset(BatchedArrayDataset(list(range(10))))

        batches = list(dataset.iter(4, drop_last_batch=drop_last_batch))

        expected = [list(range(4)), list(range(4, 8))]
        if not drop_last_batch:
            expected.append([8, 9])
        assert [batch.tolist() for batch in batches] == expected
        assert dataset.batch_loader_kwargs is None

    def test_explicit_batch_iteration_overrides_only_batch_policy(self):
        def collate_fn(rows):
            return {"rows": tuple(rows)}

        config = BatchLoaderConfig(
            batch_size=7, drop_last=True, collate_fn=collate_fn
        )
        dataset = IterableWithLenDataset(
            ArrayDataset(list(range(10))), batch_loader_kwargs=config
        )

        assert list(dataset.iter(batch_size=4)) == [
            {"rows": (0, 1, 2, 3)},
            {"rows": (4, 5, 6, 7)},
            {"rows": (8, 9)},
        ]
        assert dataset.batch_loader_kwargs is config
        assert config.batch_size == 7
        assert config.drop_last is True
        assert config.collate_fn is collate_fn
        assert list(dataset) == [{"rows": tuple(range(7))}]

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    @pytest.mark.parametrize("resample_ratio", [0.5, 2.5])
    @pytest.mark.parametrize("drop_last_batch", [False, True])
    def test_explicit_batch_iteration_preserves_sampling(
        self, generator_kind, resample_ratio, drop_last_batch
    ):
        def collect(explicit):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(31)
                generator = (
                    torch.Generator().manual_seed(17)
                    if generator_kind == "torch"
                    else np.random.default_rng(17)
                )
                dataset = IterableWithLenDataset(
                    ArrayDataset(list(range(10))),
                    shuffle=ShuffleConfig(
                        shuffle=True, chunk_size=3, prefetch_factor=3
                    ),
                    generator=generator,
                    resample_ratio=resample_ratio,
                    batch_loader_kwargs=BatchLoaderConfig(
                        batch_size=7 if explicit else 4,
                        drop_last=not drop_last_batch
                        if explicit
                        else drop_last_batch,
                    ),
                )
                iterator = (
                    dataset.iter(4, drop_last_batch)
                    if explicit
                    else iter(dataset)
                )
                batches = [batch.tolist() for batch in iterator]
                state = (
                    generator.get_state()
                    if isinstance(generator, torch.Generator)
                    else generator.bit_generator.state
                )
                return batches, state, torch.get_rng_state()

        actual, actual_rng, actual_global_rng = collect(True)
        expected, expected_rng, expected_global_rng = collect(False)

        assert actual == expected
        if generator_kind == "torch":
            assert torch.equal(actual_rng, expected_rng)
        else:
            assert actual_rng == expected_rng
        assert torch.equal(actual_global_rng, expected_global_rng)
        rows = round(10 * resample_ratio)
        if drop_last_batch:
            rows -= rows % 4
        assert sum(map(len, actual)) == rows

    @pytest.mark.parametrize("worker_id", [0, 1])
    @pytest.mark.parametrize("drop_last_batch", [False, True])
    def test_explicit_batch_iteration_uses_worker_local_rows(
        self, monkeypatch, worker_id, drop_last_batch
    ):
        worker = SimpleNamespace(num_workers=2, id=worker_id)
        monkeypatch.setattr(
            torch.utils.data, "get_worker_info", lambda: worker
        )
        dataset = IterableWithLenDataset(
            ArrayDataset(list(range(10))),
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
            resample_ratio=2.5,
        )

        batches = list(dataset.iter(4, drop_last_batch))
        rows = [value for batch in batches for value in batch.tolist()]

        assert set(rows) == set(range(worker_id * 5, worker_id * 5 + 5))
        assert len(rows) == (12 if drop_last_batch or worker_id == 1 else 13)

    @pytest.mark.parametrize(
        "batch_size,drop_last_batch,error",
        [
            (0, False, "batch_size"),
            (-1, False, "batch_size"),
            (True, False, "batch_size"),
            (1.5, False, "batch_size"),
            ("4", False, "batch_size"),
            (None, False, "batch_size"),
            (4, None, "drop_last_batch"),
            (4, 1, "drop_last_batch"),
            (4, "False", "drop_last_batch"),
        ],
    )
    def test_explicit_batch_iteration_validates_arguments(
        self, batch_size, drop_last_batch, error
    ):
        dataset = IterableWithLenDataset(ArrayDataset([0, 1]))

        with pytest.raises(ValueError, match=error):
            list(dataset.iter(batch_size, drop_last_batch))

    def test_resample_ratio_survives_views_and_dataloader_clone(
        self,
        dummy_array_dataset: ArrayDataset,
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=2.0,
        )
        sharded = dataset.shard(num_shards=2, index=0)
        taken = dataset.take(slice(0, 3))
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            use_dataset_side_batching=True,
        )

        assert sharded.resample_ratio == 2.0
        assert sharded.total_iterator_length == 10
        assert taken.resample_ratio == 2.0
        assert taken.total_iterator_length == 6
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.resample_ratio == 2.0

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_worker_sharding_precedes_oversampling_for_tiny_dataset(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset([7]),
            shuffle=True,
            resample_ratio=2.0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        batches = list(dataloader)

        assert len(batches) == len(dataloader) == 1
        assert batches[0].tolist() == [7, 7]

    def test_worker_targets_preserve_global_downsample_length(self):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset([0, 1, 2]),
            shuffle=True,
            resample_ratio=0.5,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        items = list(dataloader)

        assert len(items) == dataset.total_iterator_length == 2
        assert len(set(items)) == 2

    @pytest.mark.parametrize("resample_ratio", [0.75, 2.0])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_shuffled_multi_worker_resampling_uses_disjoint_worker_shards(
        self,
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset(list(range(8))),
            shuffle=True,
            resample_ratio=resample_ratio,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2 if use_dataset_side_batching else None,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(0),
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        dataloader_items = list(dataloader)
        items = (
            [int(item) for batch in dataloader_items for item in batch]
            if use_dataset_side_batching
            else [int(item) for item in dataloader_items]
        )

        if resample_ratio < 1:
            assert len(items) == 6
            assert len(set(items)) == 6
        else:
            assert sorted(items) == sorted([*range(8), *range(8)])

    @pytest.mark.parametrize(
        "batch_size, num_workers, drop_last",
        [
            (3, 0, False),
            (4, 0, False),
            (6, 0, False),
            (3, 0, True),
            (4, 0, True),
            (6, 0, True),
            (3, 3, False),
            (4, 3, False),
            (6, 3, False),
            (3, 3, True),
            (4, 3, True),
            (6, 3, True),
        ],
    )
    def test_total_batch_consistency(
        self,
        total_batch_consistency_test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ):
        dataset = IterableWithLenDataset(total_batch_consistency_test_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        # check batched reader
        dataset = IterableWithLenDataset(
            total_batch_consistency_test_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def test_dataloader_item_consistency(self, dummy_array_dataset: Dataset):
        dataset = IterableWithLenDataset(dummy_array_dataset)
        batch_size = 3
        num_workers = 0
        drop_last = False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

        # check batched reader
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

    def test_unbatched_iterable_len(self, dummy_array_dataset: ArrayDataset):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        dataloader_items = list(dataloader)

        assert dataloader_items == dummy_array_dataset.data
        assert len(dataloader) == len(dummy_array_dataset)
        assert len(dataloader_items) == len(dataloader)

    def test_iterable_with_len_self_batched_overrides_dataset_config(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=False,
            ),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=True,
            num_workers=0,
        )

        assert dataloader.dataset is not dataset
        assert dataset.batch_loader_kwargs is not None
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataset.batch_loader_kwargs.batch_size == 4
        assert dataset.batch_loader_kwargs.drop_last is False
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True
        assert len(list(dataloader)) == len(dataloader)

    def test_use_dataset_side_batching_supports_iterable_with_len(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=False,
            num_workers=0,
            use_dataset_side_batching=True,
        )

        assert dataloader.dataset is not dataset
        assert dataset.batch_loader_kwargs is None
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is False
        assert len(list(dataloader)) == len(dataloader)

    def test_use_dataset_side_batching_aligns_user_collate_fn(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)
        collate_inputs: list[list[int]] = []

        def collate_fn(batch: list[int]) -> dict[str, Any]:
            batch_list = list(batch)
            collate_inputs.append(batch_list)
            return {"values": batch_list, "size": len(batch_list)}

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_fn,
            use_dataset_side_batching=True,
        )

        assert list(dataloader) == [
            {"values": [0, 1, 2, 3], "size": 4},
            {"values": [4, 5, 6, 7], "size": 4},
        ]
        assert collate_inputs == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]

    def test_iterable_with_len_accepts_dataloader_shuffle_true(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset, shuffle=False)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle_config.shuffle is False
        assert dataloader.dataset._shuffle_config.shuffle is True
        assert len(list(dataloader)) == len(dataloader)

    def test_iterable_with_len_self_batched_aligns_shuffle_config(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shuffle=ShuffleConfig(
                shuffle=False,
                chunk_size=4,
                prefetch_factor=3,
            ),
            batch_loader_kwargs=BatchLoaderConfig(batch_size=2),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle_config.shuffle is False
        assert dataset._shuffle_config.chunk_size == 4
        assert dataset._shuffle_config.prefetch_factor == 3
        assert dataloader.dataset._shuffle_config.shuffle is True
        assert dataloader.dataset._shuffle_config.chunk_size == 4
        assert dataloader.dataset._shuffle_config.prefetch_factor == 3
        assert len(list(dataloader)) == len(dataloader)

    def test_iterable_with_len_uses_virtual_batch_getitems(self):
        dataset = BatchedArrayDataset(data=list(range(10)))

        iterable_dataset = IterableWithLenDataset(dataset)

        assert list(iterable_dataset) == list(range(10))
        assert dataset.getitem_calls == []
        assert dataset.getitems_calls
        assert sum(len(batch) for batch in dataset.getitems_calls) == len(
            dataset
        )
        assert all(
            len(batch) <= _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE
            for batch in dataset.getitems_calls
        )

    def test_iterable_with_len_shard_uses_accelerate_signature(
        self, dummy_array_dataset: ArrayDataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        sharded_dataset = dataset.shard(num_shards=3, index=1)

        assert list(sharded_dataset) == [4, 5, 6]

    def test_iterable_with_len_shard_preserves_shard_kwargs(
        self, dummy_array_dataset: ArrayDataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shard_kwargs=ShardConfig(
                contiguous=False,
                shard_strategy="pad_last",
            ),
        )

        sharded_dataset = dataset.shard(num_shards=3, index=1)

        assert list(sharded_dataset) == [1, 4, 7]
        assert sharded_dataset.shard_kwargs.contiguous is False
        assert sharded_dataset.shard_kwargs.shard_strategy == "pad_last"
