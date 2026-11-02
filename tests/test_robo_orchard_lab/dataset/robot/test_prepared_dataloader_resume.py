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

"""Tests for prepared DataLoader checkpoint ownership and recovery."""

import copy

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    ArrayDatasetItem,
    isolated_accelerate_state as isolated_accelerate_state,
)
from torch.utils.data import (
    DataLoader as TorchDataLoader,
)

from robo_orchard_lab.dataset.robot import (
    DataLoader,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    close_dataloader_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
)


class TestLoaderResume:
    @pytest.mark.parametrize("seed,epoch", [(123, 0), (-1, 1)])
    def test_prepared_seedable_sampler_seed_restores_actual_owner(
        self, seed, epoch, isolated_accelerate_state
    ):
        from accelerate.data_loader import SeedableRandomSampler

        from robo_orchard_lab.utils.accelerate import _identified_data_seed

        def build(data_seed):
            accelerator = Accelerator(
                cpu=True,
                dataloader_config=DataLoaderConfiguration(
                    use_seedable_sampler=True, data_seed=data_seed
                ),
            )
            source = DataLoader(
                ArrayDataset(list(range(24))), batch_size=2, shuffle=True
            )
            wrapper = accelerator.prepare(source)
            base = DataLoader._from_prepared_base(
                wrapper.base_dataloader, source_loader=source
            )
            wrapper.base_dataloader = base
            base._notify_epoch(epoch)
            wrapper.set_epoch(epoch)
            wrapper.dl_state_dict = base.state_dict()
            return wrapper, base

        reference, _ = build(seed)
        expected = [batch.tolist() for batch in reference]
        wrapper, base = build(seed)
        assert (
            type(base._resume_runtime._sampler_owner(base))
            is SeedableRandomSampler
        )
        assert base._get_sampling_seed() == seed
        assert _identified_data_seed(wrapper) == seed
        iterator = iter(wrapper)
        prefix = [next(iterator).tolist() for _ in range(3)]
        state = copy.deepcopy(wrapper.state_dict())
        assert state["seed"] == seed
        close_dataloader_resources(
            wrapper, iterator, reason=DataloaderCloseReason.EARLY_BREAK
        )
        restored, restored_base = build(999)
        restored_base.load_state_dict(state)
        restored.dl_state_dict = copy.deepcopy(state)
        assert (
            restored_base._resume_runtime._sampler_owner(
                restored_base
            ).initial_seed
            == seed
        )
        assert restored_base._get_sampling_seed() == seed
        suffix = [batch.tolist() for batch in restored]
        assert prefix + suffix == expected

    def test_seedable_sampler_checkpoint_records_actual_rank_seed(
        self, tmp_path, isolated_accelerate_state
    ):
        from robo_orchard_lab.pipeline.hooks.checkpoint import (
            SaveCheckpointConfig,
        )
        from robo_orchard_lab.pipeline.training.hook_based_trainer import (
            HookBasedTrainer,
        )
        from robo_orchard_lab.utils.accelerate import _identified_data_seed

        accelerator = Accelerator(
            cpu=True,
            dataloader_config=DataLoaderConfiguration(
                use_seedable_sampler=True, data_seed=123
            ),
        )
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loader = DataLoader(
            torch.arange(1.0, 9.0).view(-1, 1), batch_size=1, shuffle=True
        )

        def processor(*, pipeline_hooks, on_batch_hook_args, model):
            del pipeline_hooks
            accelerator.backward(model(on_batch_hook_args.batch).sum())

        root = tmp_path / "checkpoints"
        trainer = HookBasedTrainer(
            accelerator=accelerator,
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            batch_processor=processor,
            max_step=1,
            hooks=[
                SaveCheckpointConfig(
                    save_root=str(root),
                    save_epoch_freq=None,
                    save_step_freq=1,
                    save_when_loop_end=False,
                    save_model=False,
                )
            ],
        )
        assert _identified_data_seed(trainer.dataloader) == 123
        trainer()
        payload = torch.load(
            root / "custom_checkpoint_0.pkl",
            map_location="cpu",
            weights_only=False,
        )
        assert payload["rank_states"][0]["loader_state"] is None
        assert payload["rank_states"][0]["seed"] == 123

    def test_prepared_loader_inherits_only_a_matching_default_seed_owner(
        self, isolated_accelerate_state
    ):
        accelerator = Accelerator(cpu=True)
        source = DataLoader(ArrayDataset(list(range(8))), shuffle=True)
        wrapper = accelerator.prepare(source)
        base = DataLoader._from_prepared_base(
            wrapper.base_dataloader, source_loader=source
        )
        assert base._get_sampling_seed() == 42
        assert base.state_dict()["seed"] == 42
        base._resume_runtime._sampler_owner(
            base
        ).generator = torch.Generator().manual_seed(42)
        assert base._get_sampling_seed() is None
        assert base.state_dict()["seed"] is None

    def test_prepared_seeded_subclass_keeps_untrusted_source_capability(
        self, isolated_accelerate_state
    ):
        class CustomDict(DictIterableDataset):
            pass

        accelerator = Accelerator(cpu=True)
        for dataset_type, expected in (
            (DictIterableDataset, "worker_local"),
            (CustomDict, "unknown"),
        ):
            source = DataLoader(
                dataset_type(
                    [ArrayDatasetItem(data=list(range(16)))], seed=42
                ),
                batch_size=2,
            )
            assert type(source.dataset) is DictIterableDataset
            assert source.state_dict()["recovery"]["accuracy"] == expected
            prepared = accelerator.prepare(source)
            base = DataLoader._from_prepared_base(
                prepared.base_dataloader, source_loader=source
            )
            assert type(base.dataset) is DictIterableDataset
            assert base.state_dict()["recovery"]["accuracy"] == expected
            shard = source.dataset.shard(num_shards=2, index=0)
            sharded = DataLoader._from_prepared_base(
                TorchDataLoader(shard, batch_size=2), source_loader=source
            )
            assert type(sharded.dataset) is DictIterableDataset
            assert sharded.state_dict()["recovery"]["accuracy"] == expected

    @pytest.mark.parametrize("num_workers", [0, 2])
    @pytest.mark.parametrize("kind", ["iterable", "map"])
    def test_real_prepared_wrapper_caches_delivery_and_load(
        self, num_workers, kind, isolated_accelerate_state
    ):
        accelerator = Accelerator(cpu=True)

        def build():
            dataset = (
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(24)))],
                    shuffle=True,
                    seed=42,
                )
                if kind == "iterable"
                else ArrayDataset(list(range(24)))
            )
            source = DataLoader(
                dataset,
                batch_size=2,
                num_workers=num_workers,
                **({"shuffle": True} if kind == "map" else {}),
            )
            wrapper = accelerator.prepare(source)
            base = DataLoader._from_prepared_base(
                wrapper.base_dataloader, source_loader=source
            )
            wrapper.base_dataloader = base
            wrapper.dl_state_dict = base.state_dict()
            return wrapper, base

        full, _ = build()
        reference = [value for batch in full for value in batch.tolist()]

        wrapper, base = build()
        iterator = iter(wrapper)
        prefix = [next(iterator) for _ in range(3)]
        state = copy.deepcopy(wrapper.state_dict())
        assert state["delivered_batches"] == 3
        assert base.state_dict()["delivered_batches"] >= 3
        close_dataloader_resources(
            wrapper, iterator, reason=DataloaderCloseReason.EARLY_BREAK
        )

        restored, restored_base = build()
        restored_base.load_state_dict(state)
        restored.dl_state_dict = copy.deepcopy(state)
        assert restored.state_dict()["delivered_batches"] == 3
        suffix = list(restored)
        actual = [
            value for batch in prefix + suffix for value in batch.tolist()
        ]
        assert sorted(actual) == sorted(reference)
