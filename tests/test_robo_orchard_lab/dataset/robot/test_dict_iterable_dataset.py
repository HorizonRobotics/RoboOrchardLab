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

"""Tests for dictionary dataset mixtures, resampling, and summaries."""

import multiprocessing as mp
import pickle
from typing import Any, cast

import pytest
import torch
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    ArrayDatasetItem,
    TestIterableDatasetMixin,
    _get_dataloader_multiprocessing_context,
)
from torch.utils.data import (
    DataLoader as TorchDataLoader,
)

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    DatasetItem,
    DatasetItemGroup,
    IterableDatasetMixin,
    ShardConfig,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
    _build_path_tree,
    _format_path_tree,
)
from robo_orchard_lab.dataset.sampler import (
    ShardStrategy,
)


class ArrayDatasetItemGroup(DatasetItemGroup):
    """Test-only logical group backed by independent in-memory leaves."""

    data_groups: list[list[int]]

    def iter_dataset_items(self):
        for data in self.data_groups:
            yield ArrayDatasetItem(data=data)


class TestDictIterableDataset(TestIterableDatasetMixin):
    @pytest.fixture()
    def dummy_dataset_items(self):
        return [
            ArrayDatasetItem(
                data=list(range(0, 10)),
            ),
            ArrayDatasetItem(data=list(range(100, 110))),
        ]

    def test_resample_ratios_control_exact_row_counts(self):
        downsampled = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10))),
                ArrayDatasetItem(data=list(range(100, 110))),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5, 0.5],
        )
        oversampled = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(10)))],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.5,
        )

        downsampled_items = list(downsampled)
        first = [item for item in downsampled_items if item < 100]
        second = [item for item in downsampled_items if item >= 100]
        oversampled_items = list(oversampled)

        assert len(first) == len(set(first)) == 5
        assert len(second) == len(set(second)) == 5
        assert downsampled.total_iterator_length == 10
        assert sorted(oversampled_items[:10]) == list(range(10))
        assert sorted(oversampled_items[10:20]) == list(range(10))
        assert len(set(oversampled_items[20:])) == 5
        assert oversampled.total_iterator_length == 25

    def test_resample_ratio_input_forms_preserve_scale_one_behavior(
        self,
        dummy_dataset_items: list[DatasetItem],
    ):
        outputs = [
            list(
                DictIterableDataset(
                    dummy_dataset_items,
                    shuffle=False,
                    resample_ratios=ratios,
                )
            )
            for ratios in (None, 1.0, [1.0, 1.0])
        ]

        assert outputs[0] == outputs[1] == outputs[2]

    def test_group_expands_to_its_physical_members(self) -> None:
        group = ArrayDatasetItemGroup(
            name="source/group",
            data_groups=[[0, 1], [10, 11, 12]],
        )
        dataset = DictIterableDataset([group])

        assert group.get_dataset_row_num() == 5
        assert group.get_dataset_row_num() == 5
        assert list(dataset) == [0, 1, 10, 11, 12]
        assert len(dataset.dataset_items) == 2
        assert dataset.logical_dataset_items == [group]
        assert dataset.total_dataset_length == 5
        assert dataset.total_iterator_length == 5

    def test_group_reflects_config_changes(self) -> None:
        group = ArrayDatasetItemGroup(data_groups=[[0, 1]])

        assert group.get_dataset_row_num() == 2
        group.data_groups[0].append(2)
        assert group.get_dataset_row_num() == 3

        copied = group.copy()
        copied.data_groups = [[10, 11, 12, 13]]
        assert copied.get_dataset_row_num() == 4
        dataset = DictIterableDataset([copied])
        assert dataset.total_iterator_length == 4
        assert list(dataset) == [10, 11, 12, 13]

    @pytest.mark.parametrize(
        "shard_kwargs,expected",
        [
            (ShardConfig(contiguous=True), [2, 5, 6]),
            (ShardConfig(contiguous=False), [1, 4, 6]),
        ],
    )
    def test_group_propagates_static_sharding_per_member(
        self,
        shard_kwargs: ShardConfig,
        expected: list[int],
    ) -> None:
        group = ArrayDatasetItemGroup(
            data_groups=[[0, 1, 2], [3, 4, 5, 6]],
        )
        dataset = DictIterableDataset(
            [group],
            shard_kwargs=shard_kwargs,
        ).shard(num_shards=2, index=1)

        assert list(dataset) == expected
        assert dataset.total_dataset_length == 7
        assert dataset.total_iterator_length == len(expected)
        assert group.shard(num_shards=2, index=1).get_sharded_row_num(
            shard_kwargs
        ) == len(expected)

    def test_group_rejects_nested_or_statically_sharded_members(self) -> None:
        class _InvalidDatasetItemGroup(DatasetItemGroup):
            mode: str

            def iter_dataset_items(self):
                if self.mode == "nested":
                    yield ArrayDatasetItemGroup(data_groups=[[0]])
                else:
                    yield ArrayDatasetItem(data=[0]).shard(2, 0)

        with pytest.raises(TypeError, match="not nested DatasetItemGroup"):
            DictIterableDataset([_InvalidDatasetItemGroup(mode="nested")])
        with pytest.raises(ValueError, match="must be unsharded"):
            DictIterableDataset([_InvalidDatasetItemGroup(mode="sharded")])

    def test_group_ratio_is_independently_rounded_per_member(self) -> None:
        dataset = DictIterableDataset(
            [ArrayDatasetItemGroup(data_groups=[[0], [10, 11]])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=0.5,
        )

        values = list(dataset)

        assert values in ([10], [11])
        assert dataset.total_iterator_length == 1

    def test_group_members_keep_existing_batch_boundaries(self) -> None:
        dataset = DictIterableDataset(
            [ArrayDatasetItemGroup(data_groups=[[0, 1], [10, 11]])],
            batch_loader_kwargs=BatchLoaderConfig(batch_size=3),
        )

        assert [batch.tolist() for batch in dataset] == [[0, 1], [10, 11]]
        assert (
            dataset.get_total_batch_num(
                num_workers=0,
                batch_size=3,
                drop_last=False,
            )
            == 2
        )

    def test_group_bounds_and_closes_member_readers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _CloseTrackingArrayDataset(ArrayDataset):
            def __init__(self, data: list[int]) -> None:
                super().__init__(data)
                self.closed = False

            def close(self) -> None:
                self.closed = True

        created: list[_CloseTrackingArrayDataset] = []

        def create_closeable_dataset(
            item: ArrayDatasetItem,
        ) -> _CloseTrackingArrayDataset:
            dataset = _CloseTrackingArrayDataset(item.data)
            created.append(dataset)
            return dataset

        monkeypatch.setattr(
            ArrayDatasetItem,
            "_create_dataset",
            create_closeable_dataset,
        )
        dataset = DictIterableDataset(
            [ArrayDatasetItemGroup(data_groups=[[0], [1], [2]])],
            max_dataset_concurrency=1,
        )

        assert list(dataset) == [0, 1, 2]
        assert len(created) == 3
        assert all(member.closed for member in created)

    def test_group_is_a_named_resample_leaf_and_summary_node(self) -> None:
        dataset = DictIterableDataset(
            [
                ArrayDatasetItemGroup(
                    name="a/b",
                    data_groups=[[0], [1]],
                ),
                ArrayDatasetItem(data=[10, 11], name="a/c"),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios={"a": 0.5, "a/b": 2.0},
        )

        values = list(dataset)

        assert dataset._logical_resample_ratios == [2.0, 0.5]
        assert dataset._resample_ratios == [2.0, 2.0, 0.5]
        assert len([value for value in values if value < 10]) == 4
        assert len([value for value in values if value >= 10]) == 1
        assert "├─ b" in dataset.summary()

    def test_group_worker_shards_preserve_exact_global_target(
        self,
    ) -> None:
        dataset = DictIterableDataset(
            [ArrayDatasetItemGroup(data_groups=[list(range(7))])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            multiprocessing_context=_get_dataloader_multiprocessing_context(2),
        )

        values = [int(batch.item()) for batch in dataloader]

        assert len(values) == dataset.total_iterator_length == 14
        assert sorted(values) == sorted([*range(7), *range(7)])

    @pytest.mark.parametrize(
        "rules,expected",
        [
            ({"sim": 0.5, "sim/*": 0.8, "sim/ur5": 1.0}, [1.0, 0.8, 0.5, 1.0]),
            ({"sim/**": 0.8}, [0.8, 0.8, 0.8, 1.0]),
            ({"**/ur?": 0.7}, [0.7, 1.0, 0.7, 1.0]),
            ({"sim/[au]*": 0.7}, [0.7, 0.7, 1.0, 1.0]),
            ({"sim/*": 0.7, "sim/**": 0.7}, [0.7, 0.7, 0.7, 1.0]),
            ({"sim/**/ur5": 0.7}, [0.7, 1.0, 0.7, 1.0]),
        ],
    )
    def test_named_globs_apply_only_to_complete_logical_names(
        self, rules, expected
    ):
        mixture = DictIterableDataset(
            [
                ArrayDatasetItem(data=[0], name="sim/ur5"),
                ArrayDatasetItem(data=[1], name="sim/arx"),
                ArrayDatasetItem(data=[2], name="sim/sub/ur5"),
                ArrayDatasetItem(data=[3], name="real/horizon"),
            ],
            shuffle=True,
            resample_ratios=rules,
        )
        assert mixture._logical_resample_ratios == expected

    def test_glob_conflicts_require_exact_name_override(self):
        item = ArrayDatasetItem(data=[0], name="sim/ur5")
        rules = {"sim/*": 0.5, "**/ur5": 0.8}
        with pytest.raises(ValueError, match="Conflicting.*sim/ur5"):
            DictIterableDataset([item], shuffle=True, resample_ratios=rules)
        rules["sim/ur5"] = 1.0
        mixture = DictIterableDataset([item], resample_ratios=rules)
        assert mixture._logical_resample_ratios == [1.0]

    @pytest.mark.parametrize("pattern", ["real/*", "SIM/*", "sim/*/ur5"])
    def test_glob_rules_must_match_a_logical_name(self, pattern):
        with pytest.raises(
            ValueError, match="does not match any dataset item"
        ):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0], name="sim/ur5")],
                shuffle=True,
                resample_ratios={pattern: 0.5},
            )

    @pytest.mark.parametrize(
        "pattern", ["sim//**", "sim/ur**", "/sim/*", "sim/*/"]
    )
    def test_glob_rules_reject_empty_or_partial_recursive_segments(
        self, pattern
    ):
        with pytest.raises(ValueError, match="nonempty segments"):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0], name="sim/ur5")],
                shuffle=True,
                resample_ratios={pattern: 0.5},
            )

    def test_group_glob_ratio_keeps_per_member_rounding_and_clone_metadata(
        self,
    ):
        mixture = DictIterableDataset(
            [
                ArrayDatasetItemGroup(
                    name="sim/source", data_groups=[[0], [1, 2]]
                )
            ],
            shuffle=True,
            resample_ratios={"sim/*": 0.5},
        )
        assert mixture._logical_resample_ratios == [0.5]
        assert mixture._resample_ratios == [0.5, 0.5]
        assert len(list(mixture)) == 1
        cloned = DataLoader(mixture, batch_size=2, shuffle=True).dataset
        assert cloned._logical_resample_ratios == [0.5]
        assert cloned._resample_ratios == [0.5, 0.5]

    def test_dict_resample_ratios_resolve_path_prefixes(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10)), name="a/b"),
                ArrayDatasetItem(data=list(range(100, 110)), name="a/c"),
                ArrayDatasetItem(data=list(range(200, 210)), name="d/e"),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios={"a": 0.5, "a/b": 2.0},
        )

        values = list(dataset)

        assert dataset._resample_ratios == [2.0, 0.5, 1.0]
        assert len([value for value in values if value < 100]) == 20
        assert len([value for value in values if 100 <= value < 200]) == 5
        assert len([value for value in values if value >= 200]) == 10

    @pytest.mark.parametrize(
        "item_names",
        [
            ("a", "a/b"),
            ("a/b", "a"),
        ],
    )
    def test_dict_resample_ratios_reject_non_leaf_item_names(
        self,
        item_names: tuple[str, str],
    ):
        with pytest.raises(
            ValueError,
            match="Dataset item names must be unique leaf paths",
        ):
            DictIterableDataset(
                [
                    ArrayDatasetItem(data=[0], name=item_names[0]),
                    ArrayDatasetItem(data=[1], name=item_names[1]),
                ],
                shuffle=True,
                resample_ratios={"a": 1.0},
            )

    def test_dict_resample_ratios_reject_duplicate_item_names(self):
        with pytest.raises(
            ValueError, match="Dataset item names must be unique"
        ):
            DictIterableDataset(
                [
                    ArrayDatasetItem(data=[0], name="a/b"),
                    ArrayDatasetItem(data=[1], name="a/b"),
                ],
                shuffle=True,
                resample_ratios={"a": 1.0},
            )

    @pytest.mark.parametrize(
        "name,resample_ratios,match",
        [
            (None, {"a": 1.0}, "dataset_items\\[0\\]\\.name"),
            ("a/B", {"a": 1.0}, "dataset_items\\[0\\]\\.name"),
            ("a/b", {"a/B": 1.0}, "resample_ratios key"),
            ("a/b", {"a/bb": 1.0}, "does not match any dataset item"),
        ],
    )
    def test_dict_resample_ratios_validate_names_and_rule_coverage(
        self,
        name: str | None,
        resample_ratios: dict[str, float],
        match: str,
    ):
        with pytest.raises(ValueError, match=match):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0], name=name)],
                shuffle=True,
                resample_ratios=resample_ratios,
            )

    def test_dict_resample_ratios_validate_rule_ratios(self):
        with pytest.raises(ValueError, match="resample_ratios\\['a'\\]"):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0], name="a/b")],
                shuffle=True,
                resample_ratios={"a": 0.0},
            )

    def test_dict_resample_ratios_require_shuffle_for_non_unit_ratio(self):
        with pytest.raises(ValueError, match="require shuffle=True"):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0], name="a/b")],
                shuffle=False,
                resample_ratios={"a": 2.0},
            )

    def test_dict_resample_ratios_survive_shard_and_clone(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(10)), name="a/b")],
            shuffle=True,
            resample_ratios={"a": 2.0},
        )

        sharded = dataset.shard(num_shards=2, index=1)
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            use_dataset_side_batching=True,
        )

        assert sharded._resample_ratios == [2.0]
        assert dataloader.dataset._resample_ratios == [2.0]

    @pytest.mark.parametrize(
        "resample_ratios,exception",
        [
            ([], ValueError),
            ([0.0], ValueError),
            ([-1.0], ValueError),
            ([float("nan")], ValueError),
            ([float("inf")], ValueError),
            ([True], TypeError),
            (["1.0"], TypeError),
        ],
    )
    def test_resample_ratios_validate_public_contract(
        self,
        resample_ratios: Any,
        exception: type[Exception],
    ):
        with pytest.raises(exception):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0])],
                shuffle=True,
                resample_ratios=resample_ratios,
            )

    @pytest.mark.parametrize("resample_ratios", [0.5, 2.0, [1.0, 2.0]])
    def test_resample_ratios_require_shuffle(
        self,
        resample_ratios: float | list[float],
    ):
        datasets = [ArrayDatasetItem(data=[0])]
        if isinstance(resample_ratios, list):
            datasets.append(ArrayDatasetItem(data=[1]))

        with pytest.raises(ValueError, match="require shuffle=True"):
            DictIterableDataset(
                datasets,
                shuffle=False,
                resample_ratios=resample_ratios,
            )

    def test_zero_target_items_are_skipped_without_zero_weight_division(self):
        mixed = DictIterableDataset(
            [
                ArrayDatasetItem(data=[0]),
                ArrayDatasetItem(data=[100, 101]),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5, 1.0],
        )
        all_zero = DictIterableDataset(
            [ArrayDatasetItem(data=[0])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5],
        )

        assert sorted(mixed) == [100, 101]
        assert mixed._total_indices_length == [0, 2]
        assert list(all_zero) == []
        assert all_zero.total_iterator_length == 0

    def test_resampling_batches_the_complete_target_row_stream_once(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(10)))],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=False,
            ),
        )

        batches = list(dataset)
        flattened = [int(item) for batch in batches for item in batch]

        assert len(batches) == dataset.get_total_batch_num(
            num_workers=0,
            batch_size=4,
            drop_last=False,
        )
        assert sorted(flattened) == sorted([*range(10), *range(10)])

    def test_resampling_batches_across_natural_cycles_before_drop_last(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[0, 1])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=True,
            ),
        )

        batches = list(dataset)

        assert len(batches) == 1
        assert sorted(batches[0].tolist()) == [0, 0, 1, 1]
        assert (
            dataset.get_total_batch_num(
                num_workers=0,
                batch_size=4,
                drop_last=True,
            )
            == 1
        )

    def test_resampling_fails_when_positive_metadata_yields_no_rows(self):
        class _EmptyDespiteMetadataItem(ArrayDatasetItem):
            def get_dataset_row_num(self) -> int:
                return 1

        dataset = DictIterableDataset(
            [_EmptyDespiteMetadataItem(data=[])],
            shuffle=True,
            resample_ratios=2.0,
        )

        with pytest.raises(RuntimeError, match="produced no rows"):
            list(dataset)

    def test_resample_metadata_and_item_names_survive_shard_and_clone(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10)), name="first"),
                ArrayDatasetItem(data=list(range(100, 110)), name="second"),
            ],
            shuffle=True,
            resample_ratios=[0.5, 2.0],
        )

        sharded = dataset.shard(num_shards=2, index=1)
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            use_dataset_side_batching=True,
        )

        assert sharded._resample_ratios == [0.5, 2.0]
        assert [item.name for item in sharded.dataset_items] == [
            "first",
            "second",
        ]
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset._resample_ratios == [0.5, 2.0]
        assert [item.name for item in dataloader.dataset.dataset_items] == [
            "first",
            "second",
        ]

    def test_summary_reports_tree_aggregates_and_leaf_resample_ratios(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10)), name="a/b"),
                ArrayDatasetItem(data=list(range(10, 20)), name="a/c"),
                ArrayDatasetItem(data=list(range(20, 30)), name="d/e"),
            ],
            shuffle=True,
            resample_ratios=[2.0, 0.5, 1.0],
        )

        assert dataset.summary() == "\n".join(
            [
                "name                           sample_ratio [frame_ratio] "
                "[length] resample_ratio [resampled_length]",
                "├─ a                                 71.43% [     66.67%] "
                "[    20]              — [              25]",
                "│  ├─ b                              57.14% [     33.33%] "
                "[    10]             ×2 [              20]",
                "│  └─ c                              14.29% [     33.33%] "
                "[    10]           ×0.5 [               5]",
                "├─ d/e                               28.57% [     33.33%] "
                "[    10]             ×1 [              10]",
                "└─ total                            100.00% [    100.00%] "
                "[    30]              — [              35]",
            ]
        )

    def test_summary_displays_raw_and_resampled_lengths(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(128)), name=name)
                for name in ("libero", "robotwin", "robodojo", "horizon")
            ],
            shuffle=True,
            resample_ratios={"horizon": 0.1734178109542776},
        )

        lines = dataset.summary().splitlines()

        assert lines[-1].startswith("└─ total")
        assert "[   512]" in lines[-1]
        assert lines[-1].endswith("[             406]")
        horizon = next(line for line in lines if "horizon" in line)
        assert "[   128]" in horizon
        assert horizon.endswith("[              22]")
        assert "×0.17342" in horizon
        assert "0.1734178109542776" not in horizon
        assert dataset.total_iterator_length == 406

    def test_summary_resampled_length_uses_physical_member_rounding(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItemGroup(name="group", data_groups=[[0], [1], [2]])],
            shuffle=True,
            resample_ratios={"group": 0.5},
        )

        assert "[     3]" in dataset.summary().splitlines()[-1]
        assert (
            dataset.summary().splitlines()[-1].endswith("[               0]")
        )
        assert dataset.total_iterator_length == 0

    def test_summary_displays_overlapping_and_duplicate_item_names(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(8)), name="a"),
                ArrayDatasetItem(data=list(range(8, 16)), name="a/b"),
                ArrayDatasetItem(data=list(range(16, 24)), name="a/b"),
            ],
            shuffle=True,
            resample_ratios=[1.0, 2.0, 0.5],
        )

        assert dataset.summary() == "\n".join(
            [
                "name                           sample_ratio [frame_ratio] "
                "[length] resample_ratio [resampled_length]",
                "├─ a                                100.00% [    100.00%] "
                "[    24]              — [              28]",
                "│  ├─ (self)                         28.57% [     33.33%] "
                "[     8]             ×1 [               8]",
                "│  └─ b                              71.43% [     66.67%] "
                "[    16]              — [              20]",
                "│     ├─ (self #0)                   57.14% [     33.33%] "
                "[     8]             ×2 [              16]",
                "│     └─ (self #1)                   14.29% [     33.33%] "
                "[     8]           ×0.5 [               4]",
                "└─ total                            100.00% [    100.00%] "
                "[    24]              — [              28]",
            ]
        )

    def test_summary_escapes_and_displays_malformed_legacy_names(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=[0]),
                ArrayDatasetItem(data=[1], name="root/"),
                ArrayDatasetItem(data=[2], name="root//child"),
                ArrayDatasetItem(data=[3], name="robot\narm"),
            ]
        )

        assert dataset.summary() == "\n".join(
            [
                "name                           sample_ratio [frame_ratio] "
                "[length] resample_ratio [resampled_length]",
                "├─ (unnamed #0)                      25.00% [     25.00%] "
                "[     1]             ×1 [               1]",
                "├─ root/(empty)                      50.00% [     50.00%] "
                "[     2]              — [               2]",
                "│  ├─ (self)                         25.00% [     25.00%] "
                "[     1]             ×1 [               1]",
                "│  └─ child                          25.00% [     25.00%] "
                "[     1]             ×1 [               1]",
                "├─ robot\\narm                        25.00% [     25.00%] "
                "[     1]             ×1 [               1]",
                "└─ total                            100.00% [    100.00%] "
                "[     4]              — [               4]",
            ]
        )

    def test_path_tree_helper_preserves_order_and_compacts_single_paths(self):
        root, root_item_labels = _build_path_tree(
            [None, "b", "b/c", "b/c", "a", "d/e"]
        )

        assert _format_path_tree(root, root_item_labels) == [
            ("├─ (unnamed #0)", (0,), 0),
            ("├─ b", (1, 2, 3), None),
            ("│  ├─ (self)", (1,), 1),
            ("│  └─ c", (2, 3), None),
            ("│     ├─ (self #0)", (2,), 2),
            ("│     └─ (self #1)", (3,), 3),
            ("├─ a", (4,), 4),
            ("├─ d/e", (5,), 5),
        ]

    def test_summary_handles_an_empty_shard(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[0])],
            shard_kwargs=ShardConfig(shard_strategy="drop_last"),
        ).shard(num_shards=2, index=1)

        summary = dataset.summary()

        assert "(unnamed #0)" in summary
        assert summary.count("0.00%") == 4
        assert "[     0]" in summary.splitlines()[-1]
        assert "— [               0]" in summary.splitlines()[-1]

    def test_dataloader_item_consistency(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)
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
        dataset = DictIterableDataset(
            dummy_dataset_items,
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

    def test_repr_does_not_require_hf_internal_info(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        repr_text = repr(dataset)

        assert "DictIterableDataset(" in repr_text
        assert "dataset_items=2" in repr_text

    @pytest.mark.skipif(
        (
            "spawn" not in mp.get_all_start_methods()
            or "file_descriptor"
            not in torch.multiprocessing.get_all_sharing_strategies()
        ),
        reason="requires spawn workers with file-descriptor Torch IPC",
    )
    def test_torch_generator_is_safe_for_spawn_workers(
        self,
        dummy_dataset_items: list[DatasetItem],
    ) -> None:
        """Spawn workers restore Torch RNG state without shared storage FDs."""

        generator = torch.Generator().manual_seed(17)
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=False,
            generator=generator,
        )
        # The project DataLoader clones iterable datasets before Torch starts
        # workers. Use Torch's loader directly to exercise the serialization
        # boundary used by the profiling runner.
        dataloader = TorchDataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
        )

        sharing_strategy = torch.multiprocessing.get_sharing_strategy()
        try:
            # The test suite normally selects ``file_system`` IPC globally.
            # Production uses Torch's Linux ``file_descriptor`` default, whose
            # generator-state transport is the boundary under test here.
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
            batches = list(dataloader)
        finally:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)

        assert batches
        assert sorted(
            item
            for batch in batches
            for item in cast(torch.Tensor, batch).tolist()
        ) == [*range(10), *range(100, 110)]

    def test_torch_generator_pickle_preserves_rng_state(
        self,
        dummy_dataset_items: list[DatasetItem],
    ) -> None:
        """Dataset serialization must not change Torch shuffle sequences."""

        generator = torch.Generator().manual_seed(23)
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=True,
            generator=generator,
        )
        expected_state = generator.get_state().clone()

        restored = pickle.loads(pickle.dumps(dataset))

        assert isinstance(restored._generator, torch.Generator)
        assert restored._generator is not generator
        assert torch.equal(restored._generator.get_state(), expected_state)
        assert "_serialized_torch_generator" not in restored.__dict__

    def test_use_dataset_side_batching_option(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=0,
            drop_last=False,
            use_dataset_side_batching=True,
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is False

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert [9, 100, 101] not in batches
        for batch in batches:
            assert batch, "Batch should not be empty."
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_close_closes_active_child_iterators(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch,
    ):
        class _CloseTrackingIterator:
            def __init__(self, items: list[int]) -> None:
                self._items = iter(items)
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return next(self._items)

            def close(self) -> None:
                self.closed = True

        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)
        dataset._total_indices_length = [2, 2]
        child_iters = [
            _CloseTrackingIterator([0, 1]),
            _CloseTrackingIterator([100, 101]),
        ]

        def fake_prepare_dataset_for_iter(
            cur_dataset_iters: list[tuple[int, Any]],
            remaining_dataset_indices: list[int],
            make_iter,
            targets,
            segment_remaining,
            offload_cursors,
            **logical_accounting,
        ) -> list[float]:
            del remaining_dataset_indices, make_iter
            del targets, offload_cursors, logical_accounting
            segment_remaining.update({0: 2, 1: 2})
            if not cur_dataset_iters:
                cur_dataset_iters.extend(
                    [(0, child_iters[0]), (1, child_iters[1])]
                )
            return [0.5, 0.5]

        monkeypatch.setattr(
            dataset,
            "_prepare_dataset_for_iter",
            fake_prepare_dataset_for_iter,
        )

        dataset_iter = iter(dataset)

        assert next(dataset_iter) == 0

        cast(Any, dataset_iter).close()

        assert child_iters[0].closed is True
        assert child_iters[1].closed is True

    @pytest.mark.parametrize("stop_early", [False, True])
    @pytest.mark.parametrize("resample_ratio", [1.0, 2.5])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_iteration_closes_datasets_created_by_items(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch: pytest.MonkeyPatch,
        stop_early: bool,
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ) -> None:
        """Close item-created datasets on exhaustion and early exit."""

        class _CloseTrackingArrayDataset(ArrayDataset):
            def __init__(self, data: list) -> None:
                super().__init__(data)
                self.closed = False

            def close(self) -> None:
                self.closed = True

        created_datasets: list[_CloseTrackingArrayDataset] = []

        def create_closeable_dataset(
            item: ArrayDatasetItem,
        ) -> _CloseTrackingArrayDataset:
            created = _CloseTrackingArrayDataset(item.data)
            created_datasets.append(created)
            return created

        monkeypatch.setattr(
            ArrayDatasetItem,
            "_create_dataset",
            create_closeable_dataset,
        )
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=resample_ratio != 1.0,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=resample_ratio,
            batch_loader_kwargs=(
                BatchLoaderConfig(batch_size=3)
                if use_dataset_side_batching
                else None
            ),
        )
        dataset_iter = iter(dataset)

        if stop_early:
            first_item = next(dataset_iter)
            if use_dataset_side_batching:
                assert len(first_item) == 3
            else:
                assert first_item in {*range(10), *range(100, 110)}
            cast(Any, dataset_iter).close()
        else:
            list(dataset_iter)

        assert created_datasets
        assert all(created.closed for created in created_datasets)

    def test_iteration_failure_closes_dataset_created_by_item(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Close the item-created dataset when row iteration fails."""

        class _FailingArrayDataset(ArrayDataset):
            def __init__(self, data: list) -> None:
                super().__init__(data)
                self.closed = False

            def __getitem__(self, index: int) -> Any:
                raise RuntimeError(f"row {index} failed")

            def close(self) -> None:
                self.closed = True

        created_datasets: list[_FailingArrayDataset] = []

        def create_failing_dataset(
            item: ArrayDatasetItem,
        ) -> _FailingArrayDataset:
            created = _FailingArrayDataset(item.data)
            created_datasets.append(created)
            return created

        monkeypatch.setattr(
            ArrayDatasetItem,
            "_create_dataset",
            create_failing_dataset,
        )
        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)

        with pytest.raises(RuntimeError, match="row 0 failed"):
            list(dataset)

        assert len(created_datasets) == 1
        assert created_datasets[0].closed is True

    def test_use_dataset_side_batching_aligns_batch_loader_kwargs(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            drop_last=True,
            use_dataset_side_batching=True,
        )

        assert dataset.batch_loader_kwargs is None
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 4
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert len(batches) == len(dataloader)
        assert all(len(batch) == 4 for batch in batches)
        for batch in batches:
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_use_dataset_side_batching_accepts_shuffle_config(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=ShuffleConfig(
                shuffle=False,
                chunk_size=5,
                prefetch_factor=2,
            ),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=0,
            use_dataset_side_batching=True,
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=3,
                prefetch_factor=4,
            ),
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle.shuffle is False
        assert dataset._shuffle.chunk_size == 5
        assert dataset._shuffle.prefetch_factor == 2
        assert dataloader.dataset._shuffle.shuffle is True
        assert dataloader.dataset._shuffle.chunk_size == 3
        assert dataloader.dataset._shuffle.prefetch_factor == 4

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert batches
        for batch in batches:
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_dict_iterable_accepts_dataloader_shuffle_true(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle.shuffle is False
        assert dataloader.dataset._shuffle.shuffle is True
        assert len(list(dataloader)) == len(dataloader)

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
        dummy_dataset_items: list[DatasetItem],
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ):
        dataset = DictIterableDataset(dummy_dataset_items)
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
        dataset = DictIterableDataset(
            dummy_dataset_items,
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

    @pytest.mark.parametrize("resample_ratio", [0.5, 2.0])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_resampled_total_batch_consistency_across_workers(
        self,
        dummy_dataset_items: list[DatasetItem],
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ):
        batch_size = 4
        num_workers = 2
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=resample_ratio,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=(
                cast(IterableDatasetMixin, dataloader.dataset)
                if use_dataset_side_batching
                else dataset
            ),
            batch_size=batch_size,
            drop_last=False,
        )

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_worker_sharding_precedes_oversampling_for_tiny_dataset(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[7])],
            shuffle=True,
            resample_ratios=2.0,
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

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_shuffled_multi_worker_resampling_uses_each_item_worker_shard(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(8))),
                ArrayDatasetItem(data=list(range(100, 108))),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=0.75,
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
        first = [item for item in items if item < 100]
        second = [item for item in items if item >= 100]

        assert len(first) == len(set(first)) == 6
        assert len(second) == len(set(second)) == 6

    def test_dict_iterable_self_batched_overrides_dataset_config(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(
            dummy_dataset_items,
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
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataset.batch_loader_kwargs.batch_size == 4
        assert dataset.batch_loader_kwargs.drop_last is False
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True
        assert len(list(dataloader)) == len(dataloader)

    def test_dataset_item_name_is_optional_and_survives_sharding(self):
        unnamed_item = ArrayDatasetItem(data=[0])
        item = ArrayDatasetItem(data=list(range(10)), name="source/action")

        sharded_item = item.shard(num_shards=3, index=1)
        restored_item = ArrayDatasetItem.model_validate(item.model_dump())

        assert unnamed_item.name is None
        assert item.model_dump()["name"] == "source/action"
        assert restored_item.name == "source/action"
        assert sharded_item.name == "source/action"
        assert sharded_item.num_shards == 3
        assert sharded_item.shard_id == 1

    def test_dict_iterable_summary_reads_current_dataset_item_names(self):
        named_item = ArrayDatasetItem(data=[0], name="named")
        dataset = DictIterableDataset(
            [
                named_item,
                ArrayDatasetItem(data=[1]),
            ]
        )
        named_item.name = "renamed"

        summary = dataset.summary()

        assert "renamed" in summary
        assert "(unnamed #1)" in summary

    def test_dict_iterable_shard_uses_accelerate_signature(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        sharded_dataset = dataset.shard(num_shards=2, index=1)

        assert isinstance(sharded_dataset, DictIterableDataset)
        assert len(sharded_dataset.dataset_items) == len(dataset.dataset_items)
        assert [item.shard_id for item in sharded_dataset.dataset_items] == [
            1,
            1,
        ]
        assert [item.num_shards for item in sharded_dataset.dataset_items] == [
            2,
            2,
        ]

    @staticmethod
    def _collect_rank_local_dataset_trace(
        rank: int,
        shard_strategy: ShardStrategy,
    ) -> tuple[list[str], list[int]]:
        generator = torch.Generator()
        generator.manual_seed(123)
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(21))),
                ArrayDatasetItem(data=list(range(100, 105))),
            ],
            shuffle=True,
            generator=generator,
            shard_kwargs=ShardConfig(
                contiguous=True,
                shard_strategy=shard_strategy,
            ),
        ).shard(num_shards=2, index=rank)

        trace = ["A" if item < 100 else "B" for item in dataset]
        assert dataset._total_indices_length is not None
        return trace, dataset._total_indices_length

    def test_rank_local_schedule_can_diverge_without_even_shards(self):
        # Multi-process iterable training relies on each rank observing the
        # same per-dataset mixture weights. Without an even shard strategy,
        # local shard lengths can differ and the rank-local dataset schedule
        # will drift even when the same RNG seed is used.
        rank0_trace, rank0_lengths = self._collect_rank_local_dataset_trace(
            rank=0,
            shard_strategy=None,
        )
        rank1_trace, rank1_lengths = self._collect_rank_local_dataset_trace(
            rank=1,
            shard_strategy=None,
        )

        assert rank0_lengths != rank1_lengths
        assert rank0_trace != rank1_trace, (
            "Expected the rank-local dataset schedule to diverge when "
            "sharded lengths are uneven, but both ranks produced the same "
            f"trace: {''.join(rank0_trace)}"
        )

    def test_rank_local_schedule_stays_aligned_with_pad_last(self):
        rank0_trace, rank0_lengths = self._collect_rank_local_dataset_trace(
            rank=0,
            shard_strategy="pad_last",
        )
        rank1_trace, rank1_lengths = self._collect_rank_local_dataset_trace(
            rank=1,
            shard_strategy="pad_last",
        )

        assert rank0_lengths == rank1_lengths
        assert rank0_trace == rank1_trace
