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

"""Trainer checkpoint save and resume policy integration tests."""

import random
from pathlib import Path

import fsspec
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration
from torch.utils.data import DataLoader

from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
    ResumeCheckpointConfig,
)


@pytest.fixture(autouse=True)
def _isolate_accelerate_state(monkeypatch):
    """Keep explicit CPU tests independent of other tests' device choice."""
    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})


class _UnmanagedTestDataLoader(DataLoader):
    """Keep these checkpoint tests independent of data-owner helpers."""


def _trainer(
    root: Path,
    *,
    max_step: int,
    resume: str | None = None,
    accumulation=1,
    sync_with_dataloader=True,
    configure_accelerator=None,
    standard_loader=False,
):
    accelerator = Accelerator(
        cpu=True,
        project_config=ProjectConfiguration(
            project_dir=str(root.parent), automatic_checkpoint_naming=True
        ),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation,
            sync_with_dataloader=sync_with_dataloader,
        ),
    )
    if configure_accelerator is not None:
        configure_accelerator(accelerator)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader_class = DataLoader if standard_loader else _UnmanagedTestDataLoader
    loader = loader_class(torch.arange(1.0, 9.0).view(-1, 1), batch_size=1)
    processed = []

    def processor(*, pipeline_hooks, on_batch_hook_args, model):
        del pipeline_hooks
        batch = on_batch_hook_args.batch
        processed.append(batch.item())
        accelerator.backward(model(batch).sum())

    trainer = HookBasedTrainer(
        accelerator=accelerator,
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        batch_processor=processor,
        max_step=max_step,
        hooks=[
            SaveCheckpointConfig(
                save_root=None,
                save_epoch_freq=None,
                save_step_freq=1,
                save_when_loop_end=False,
                save_model=False,
            )
        ],
        resume_from=(
            ResumeCheckpointConfig(resume_from=resume) if resume else None
        ),
    )
    return trainer, processed


def test_sequential_map_does_not_record_worker_seed_as_sampling_seed(tmp_path):
    from robo_orchard_lab.utils.accelerate import (
        _capture_prepared_loader_recovery,
    )

    trainer, _ = _trainer(
        tmp_path / "checkpoints", max_step=1, standard_loader=True
    )
    assert (
        _capture_prepared_loader_recovery(
            trainer.dataloader, processed_batches=0
        )["seed"]
        is None
    )
    assert trainer.dataloader.base_dataloader.generator is not None
    trainer()
    state = torch.load(
        tmp_path / "checkpoints" / "checkpoint_0" / "custom_checkpoint_0.pkl",
        weights_only=False,
    )
    assert state["rank_states"][0]["seed"] is None


def test_fallback_checkpoint_reads_sampling_identity_from_loader_owner(
    tmp_path,
):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    payload = torch.load(
        root / "checkpoint_0" / "custom_checkpoint_0.pkl",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["rank_states"][0]["loader_state"] is None
    assert payload["rank_states"][0]["seed"] is None


def test_native_only_checkpoint_backend_does_not_block_new_training(
    tmp_path, monkeypatch
):
    from robo_orchard_lab.pipeline.hooks import checkpoint

    monkeypatch.setattr(
        checkpoint,
        "_checkpoint_capability_reason",
        lambda accelerator: "backend capability unavailable",
    )
    trainer, _ = _trainer(tmp_path / "checkpoints", max_step=1)
    assert not hasattr(trainer, "_full_checkpoint_supported")


@pytest.mark.parametrize("capability", ["node_local", "stateful_loader"])
def test_native_only_capability_preserves_save_and_rejects_resume(
    tmp_path, capability, caplog
):
    def configure(accelerator):
        if capability == "node_local":
            accelerator.project_configuration.save_on_each_node = True
        else:
            original_prepare = accelerator.prepare

            def prepare_stateful(*args):
                prepared = original_prepare(*args)
                # Emulate the external adapter's contract without torchdata.
                prepared[0].use_stateful_dataloader = True
                prepared[0].dl_state_dict = {}
                return prepared

            accelerator.prepare = prepare_stateful

    root = tmp_path / "checkpoints"
    trainer, batches = _trainer(
        root, max_step=1, configure_accelerator=configure
    )
    assert not hasattr(trainer, "_full_checkpoint_supported")
    trainer()
    assert batches == [1.0]
    payload = torch.load(
        root / "checkpoint_0" / "custom_checkpoint_0.pkl",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["rank_states"] == []
    assert "without trainer/data" in caplog.text
    with pytest.raises(RuntimeError, match="unsupported"):
        _trainer(
            root,
            max_step=2,
            resume=str(root / "checkpoint_0"),
            configure_accelerator=configure,
        )


def test_custom_model_hooks_preserve_trainer_resume(tmp_path):
    root = tmp_path / "checkpoints"
    events = []

    def configure(accelerator):
        accelerator.register_save_state_pre_hook(
            lambda models, weights, output_dir: events.append("save")
        )
        accelerator.register_load_state_pre_hook(
            lambda models, input_dir: events.append("load")
        )

    first, _ = _trainer(root, max_step=1, configure_accelerator=configure)
    first()
    resumed, batches = _trainer(
        root,
        max_step=2,
        resume=str(root / "checkpoint_0"),
        configure_accelerator=configure,
    )
    resumed()
    assert batches == [2.0]
    assert events == ["save", "load", "save"]


def test_cpu_checkpoint_replay_and_rng_restore(tmp_path):
    """A real CPU checkpoint reloads progress and replays the prefix once."""
    root = tmp_path / "checkpoints"
    first, first_batches = _trainer(root, max_step=1)
    first()
    checkpoint = root / "checkpoint_0"
    payload = torch.load(
        checkpoint / "custom_checkpoint_0.pkl",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema_version"] == 3
    assert "checkpoint_complete" not in payload
    assert first_batches == [1.0]

    saved_rng = torch.load(
        checkpoint / "random_states_0.pkl",
        map_location="cpu",
        weights_only=False,
    )
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    restored, resumed_batches = _trainer(
        root, max_step=2, resume=str(checkpoint)
    )
    assert random.getstate() == saved_rng["random_state"]
    np.testing.assert_equal(
        np.random.get_state(), saved_rng["numpy_random_seed"]
    )
    assert torch.equal(torch.get_rng_state(), saved_rng["torch_manual_seed"])
    restored()
    assert resumed_batches == [2.0]
    assert restored.trainer_progress_state.global_step_id == 2


@pytest.mark.parametrize(
    "position", [0, 2, 4], ids=["epoch-start", "batch", "eof"]
)
def test_trainer_replay_restores_rng_after_all_checkpoint_objects_load(
    tmp_path, position
):
    """First resumed processing sees the post-load RNG, after data replay."""
    from robo_orchard_lab.pipeline.hooks.mixin import (
        HookContext,
        PipelineHooks,
    )

    def snapshot():
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    def draws_from(state):
        py = random.Random()
        py.setstate(state[0])
        numpy = np.random.RandomState()
        numpy.set_state(state[1])
        generator = torch.Generator().set_state(state[2])
        return (
            py.random(),
            numpy.random_sample(),
            torch.rand((), generator=generator).item(),
        )

    class RandomLoadObject:
        after_load = None

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}
            random.random()
            np.random.random()
            torch.rand(())
            self.after_load = snapshot()

    class RandomIterable(torch.utils.data.IterableDataset):
        def __init__(self):
            self.read_states = []

        def __iter__(self):
            for value in range(4):
                random.random()
                np.random.random()
                torch.rand(())
                self.read_states.append(snapshot())
                yield torch.tensor([float(value)])

    root = tmp_path / "checkpoints"

    def make_trainer(*, resume=None):
        accelerator = Accelerator(
            cpu=True,
            project_config=ProjectConfiguration(
                project_dir=str(root.parent), automatic_checkpoint_naming=True
            ),
        )
        checkpoint_object = RandomLoadObject()
        accelerator.register_for_checkpointing(checkpoint_object)
        dataset = RandomIterable()
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        observations = []

        def observe():
            observations.append(
                (random.random(), np.random.random(), torch.rand(()).item())
            )

        def processor(*, pipeline_hooks, on_batch_hook_args, model):
            del pipeline_hooks
            if not observations:
                # Accelerate can prefetch before delivering the first batch.
                observations.append(("before", dataset.read_states[-1]))
                observe()
            accelerator.backward(model(on_batch_hook_args.batch).sum())

        observer = PipelineHooks()
        observer.register_hook(
            "on_epoch", HookContext.from_callable(after=lambda args: observe())
        )
        trainer = HookBasedTrainer(
            accelerator=accelerator,
            model=model,
            dataloader=DataLoader(dataset, batch_size=1),
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1
            ),
            batch_processor=processor,
            max_epoch=(
                1 if not resume and position == 0 else (2 if resume else None)
            ),
            max_step=None if resume or position == 0 else position,
            hooks=(
                [
                    SaveCheckpointConfig(
                        save_root=None,
                        save_epoch_freq=(
                            1 if not resume and position == 0 else None
                        ),
                        save_step_freq=(
                            position if not resume and position else None
                        ),
                        save_when_loop_end=False,
                        save_model=False,
                    ),
                ]
                if not resume
                else []
            )
            + [observer],
            resume_from=(
                ResumeCheckpointConfig(resume_from=str(resume))
                if resume
                else None
            ),
        )
        return trainer, checkpoint_object, observations

    original, _, _ = make_trainer()
    original()
    resumed, loaded_object, observations = make_trainer(
        resume=root / "checkpoint_0"
    )
    assert loaded_object.after_load is not None
    resumed()
    if position == 4:
        # The restored epoch has no remaining batches; the epoch hook runs
        # after the replay reaches normal EOF.
        actual = observations[0]
        expected = draws_from(loaded_object.after_load)
    else:
        assert observations[0][0] == "before"
        actual = observations[1]
        expected = draws_from(
            loaded_object.after_load if position else observations[0][1]
        )
    assert actual == pytest.approx(expected)


def test_resume_then_save_uses_next_native_automatic_id(tmp_path):
    root = tmp_path / "checkpoints"
    first, _ = _trainer(root, max_step=1)
    first()
    resumed, batches = _trainer(
        root, max_step=2, resume=str(root / "checkpoint_0")
    )
    resumed()
    assert batches == [2.0]
    assert (root / "checkpoint_1" / "custom_checkpoint_0.pkl").is_file()
    assert (root / "checkpoint_0" / "custom_checkpoint_0.pkl").is_file()


def test_resume_keeps_higher_requested_native_iteration(tmp_path):
    root = tmp_path / "checkpoints"
    first, _ = _trainer(root, max_step=1)
    first()
    resumed, _ = _trainer(
        root,
        max_step=2,
        resume=str(root / "checkpoint_0"),
        configure_accelerator=lambda accelerator: setattr(
            accelerator.project_configuration, "iteration", 9
        ),
    )
    resumed()
    assert (root / "checkpoint_9" / "custom_checkpoint_0.pkl").is_file()


def test_native_automatic_retention_is_owned_by_accelerate(tmp_path):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(
        root,
        max_step=2,
        configure_accelerator=lambda accelerator: setattr(
            accelerator.project_configuration, "total_limit", 1
        ),
    )
    trainer()
    assert not (root / "checkpoint_0").exists()
    assert (root / "checkpoint_1" / "custom_checkpoint_0.pkl").is_file()


def test_failed_save_does_not_remove_partial_native_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    hook = SaveCheckpointConfig(
        save_root=None,
        save_epoch_freq=None,
        save_step_freq=1,
        save_model=False,
    )()
    partial = root / "checkpoint_1"

    def fail_save(directory):
        partial.mkdir()
        (partial / "partial.bin").write_bytes(b"partial")
        raise OSError("injected save failure")

    monkeypatch.setattr(trainer.accelerator, "save_state", fail_save)
    with pytest.raises(OSError, match="injected save failure"):
        hook._save_state(trainer.accelerator)
    assert (partial / "partial.bin").read_bytes() == b"partial"


def test_remote_checkpoint_uses_owned_staging(tmp_path):
    """Remote recovery leaves caller-owned cache files untouched."""
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    checkpoint = root / "checkpoint_0"
    fs = fsspec.filesystem("memory")
    remote = "memory://checkpoint-resume-test/generation"
    for path in checkpoint.iterdir():
        with fs.open(f"{remote}/{path.name}", "wb") as target:
            target.write(path.read_bytes())
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "caller-file"
    sentinel.write_text("keep")
    restored, _ = _trainer(root, max_step=2)
    ResumeCheckpointConfig(
        resume_from=remote, cache_dir=str(cache)
    ).load_state(restored.accelerator)
    assert restored.trainer_progress_state.global_step_id == 1
    assert sentinel.read_text() == "keep"
    assert list(cache.iterdir()) == [sentinel]


def test_remote_missing_model_member_is_reported_by_accelerate(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    fs = fsspec.filesystem("memory")
    remote = "memory://checkpoint-resume-missing/generation"
    for path in (root / "checkpoint_0").iterdir():
        if path.name != "optimizer.bin":
            with fs.open(f"{remote}/{path.name}", "wb") as target:
                target.write(path.read_bytes())
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "caller-file"
    sentinel.write_text("keep")
    restored, _ = _trainer(root, max_step=2)
    load_calls = []
    original_load = restored.accelerator.load_state

    def observe_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(restored.accelerator, "load_state", observe_load)
    with pytest.raises(FileNotFoundError, match="optimizer.bin"):
        ResumeCheckpointConfig(
            resume_from=remote, cache_dir=str(cache)
        ).load_state(restored.accelerator)
    assert len(load_calls) == 1
    assert list(cache.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_resume_rejects_old_progress_schema(tmp_path):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    checkpoint = root / "checkpoint_0"
    filename = checkpoint / "custom_checkpoint_0.pkl"
    state = torch.load(filename, map_location="cpu", weights_only=False)
    state["schema_version"] = 2
    torch.save(state, filename)
    with pytest.raises(RuntimeError, match="Unsupported.*schema_version=2"):
        _trainer(root, max_step=2, resume=str(checkpoint))


def test_resume_changed_accumulation_starts_full_window(tmp_path):
    root = tmp_path / "checkpoints"
    trainer, first_batches = _trainer(root, max_step=1, accumulation=4)
    trainer()
    assert first_batches == [1.0, 2.0, 3.0, 4.0]
    saved_rng = torch.load(
        root / "checkpoint_0" / "random_states_0.pkl", weights_only=False
    )
    assert saved_rng["step"] == 4
    resumed, resumed_batches = _trainer(
        root, max_step=2, resume=str(root / "checkpoint_0"), accumulation=3
    )
    assert resumed.accelerator.step == 0
    resumed()
    assert resumed_batches == [5.0, 6.0, 7.0]


def test_resume_with_changed_dataloader_sync(tmp_path):
    root = tmp_path / "checkpoints"
    trainer, _ = _trainer(root, max_step=1)
    trainer()
    resumed, batches = _trainer(
        root,
        max_step=2,
        resume=str(root / "checkpoint_0"),
        sync_with_dataloader=False,
    )
    resumed()
    assert batches == [2.0]
