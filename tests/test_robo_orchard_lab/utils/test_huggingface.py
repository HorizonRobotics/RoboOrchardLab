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

"""Generic native checkpoint loading, staging, and naming tests."""

import builtins
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fsspec
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import DistributedType

from robo_orchard_lab.pipeline.training._progress import TrainerProgressState
from robo_orchard_lab.utils.huggingface import (
    accelerator_load_state,
    get_accelerate_project_last_checkpoint_id,
)


@pytest.fixture(autouse=True)
def _isolate_accelerate_state(monkeypatch):
    """Keep explicit CPU tests independent of other tests' device choice."""
    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})


def _generic_accelerator():
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    model, optimizer = accelerator.prepare(model, optimizer)
    with torch.no_grad():
        model.weight.fill_(3.5)
    return accelerator, model, optimizer


def test_last_checkpoint_id_ignores_unrelated_and_counts_occupied_names(
    tmp_path,
):
    """Occupied checkpoint names determine the next automatic ID."""
    checkpoint_root = tmp_path / "checkpoints"
    assert get_accelerate_project_last_checkpoint_id(str(tmp_path)) == -1
    checkpoint_root.mkdir()
    (checkpoint_root / "notes").mkdir()
    (checkpoint_root / "checkpoint_999999.tmp").mkdir()
    assert get_accelerate_project_last_checkpoint_id(str(tmp_path)) == -1

    native = checkpoint_root / "checkpoint_2"
    native.mkdir()
    (native / "model.safetensors").write_bytes(b"native")
    reserved = checkpoint_root / "checkpoint_12"
    reserved.mkdir()
    (checkpoint_root / "checkpoint_123.tmp").mkdir()
    assert get_accelerate_project_last_checkpoint_id(str(tmp_path)) == 12

    next_id = get_accelerate_project_last_checkpoint_id(str(tmp_path)) + 1
    (checkpoint_root / f"checkpoint_{next_id}").mkdir()
    assert native.joinpath("model.safetensors").read_bytes() == b"native"
    assert list(reserved.iterdir()) == []


def test_generic_accelerator_loads_local_state(tmp_path):
    accelerator, model, optimizer = _generic_accelerator()
    checkpoint = tmp_path / "generic-local"
    accelerator.save_state(str(checkpoint))
    with torch.no_grad():
        model.weight.fill_(-4)
    optimizer.param_groups[0]["lr"] = 0.75

    accelerator_load_state(accelerator, str(checkpoint))

    assert model.weight.item() == 3.5
    assert optimizer.param_groups[0]["lr"] == 0.25


@pytest.mark.parametrize("registered_progress", [False, True])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_xla_generic_load_delegates_without_object_collective(
    tmp_path, monkeypatch, remote, registered_progress
):
    from accelerate.utils import operations

    monkeypatch.setattr(
        operations,
        "PartialState",
        lambda: SimpleNamespace(distributed_type=DistributedType.XLA),
    )
    with pytest.raises(NotImplementedError, match="TPU"):
        operations.gather_object([{"probe": True}])
    accelerator = SimpleNamespace(
        _custom_objects=(
            [TrainerProgressState()] if registered_progress else []
        ),
        distributed_type=DistributedType.XLA,
        step=19,
        load_state=MagicMock(),
    )
    if remote:
        remote_path = "memory://generic-xla-native-load/state"
        fs = fsspec.filesystem("memory")
        with fs.open(f"{remote_path}/model.bin", "wb") as stream:
            stream.write(b"model")
        cache = tmp_path / "cache"
        cache.mkdir()

        def load(path):
            assert (Path(path) / "model.bin").read_bytes() == b"model"

        accelerator.load_state.side_effect = load
        accelerator_load_state(accelerator, remote_path, cache_dir=str(cache))
        assert list(cache.iterdir()) == []
    else:
        accelerator_load_state(accelerator, str(tmp_path))
        accelerator.load_state.assert_called_once_with(str(tmp_path))
    assert accelerator.step == 19


def test_generic_remote_load_copies_members_to_owned_staging(tmp_path):
    accelerator, model, optimizer = _generic_accelerator()
    checkpoint = tmp_path / "generic-source"
    accelerator.save_state(str(checkpoint))
    nested = checkpoint / "backend" / "metadata.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"remote backend member")
    fs = fsspec.filesystem("memory")
    remote = "memory://generic-checkpoint-test/generation"
    for file in checkpoint.rglob("*"):
        if file.is_file():
            with fs.open(
                f"{remote}/{file.relative_to(checkpoint)}", "wb"
            ) as out:
                out.write(file.read_bytes())

    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "model.safetensors"
    sentinel.write_bytes(b"caller cache")
    seen = []

    def inspect_staging(models, directory):
        seen.append(Path(directory))
        assert (Path(directory) / "backend" / "metadata.bin").read_bytes() == (
            b"remote backend member"
        )

    accelerator.register_load_state_pre_hook(inspect_staging)
    with torch.no_grad():
        model.weight.fill_(-4)
    optimizer.param_groups[0]["lr"] = 0.75
    accelerator_load_state(accelerator, remote, cache_dir=str(cache))

    assert model.weight.item() == 3.5
    assert optimizer.param_groups[0]["lr"] == 0.25
    assert len(seen) == 1 and seen[0].parent == cache
    assert sentinel.read_bytes() == b"caller cache"
    assert list(cache.iterdir()) == [sentinel]
    with pytest.raises(FileNotFoundError, match="contains no files"):
        accelerator_load_state(
            accelerator, remote + "-missing", cache_dir=str(cache)
        )
    assert list(cache.iterdir()) == [sentinel]


@pytest.mark.parametrize("phase", ["preflight", "load"])
@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_remote_cleanup_preserves_primary_error_under_strict_warnings(
    tmp_path, monkeypatch, caplog, phase, diagnostic_failure
):
    """Cleanup and its reporting must preserve an existing recovery error."""
    import logging

    from robo_orchard_lab.utils import huggingface as hf

    accelerator, _, _ = _generic_accelerator()
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "caller-file"
    sentinel.write_bytes(b"preserve caller cache")
    original_rmtree = hf.shutil.rmtree

    def copy_checkpoint(*args, **kwargs):
        if phase == "preflight":
            raise FileNotFoundError("primary missing member")

    def load_checkpoint(*args, **kwargs):
        raise ValueError("primary load failure")

    def fail_cleanup(*args, **kwargs):
        raise OSError("secondary cleanup failure")

    def fail_diagnostic(*args, **kwargs):
        raise RuntimeError("secondary diagnostic failure")

    monkeypatch.setattr(hf, "_copy_remote_checkpoint", copy_checkpoint)
    monkeypatch.setattr(accelerator, "load_state", load_checkpoint)
    monkeypatch.setattr(hf.shutil, "rmtree", fail_cleanup)
    if diagnostic_failure:
        monkeypatch.setattr(
            logging.getLogger(hf.__name__), "warning", fail_diagnostic
        )
    expected = FileNotFoundError if phase == "preflight" else ValueError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(expected, match="primary"):
                accelerator_load_state(
                    accelerator, "memory://checkpoint", cache_dir=str(cache)
                )
        assert sentinel.read_bytes() == b"preserve caller cache"
        if not diagnostic_failure:
            assert "Checkpoint staging cleanup failed" in caplog.text
    finally:
        # The injected failure intentionally leaves only our temporary stage.
        for path in cache.iterdir():
            if path.is_dir():
                original_rmtree(path)


def test_generic_load_does_not_import_training_pipeline(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("robo_orchard_lab.pipeline"):
            raise AssertionError("generic loading imported training policy")
        return original_import(name, *args, **kwargs)

    accelerator = SimpleNamespace(load_state=MagicMock())
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded_import)
        accelerator_load_state(accelerator, str(tmp_path), map_location="cpu")
    accelerator.load_state.assert_called_once_with(
        str(tmp_path), map_location="cpu"
    )


def test_generic_load_keeps_native_phase_with_registered_progress(tmp_path):
    accelerator, _, _ = _generic_accelerator()
    accelerator.register_for_checkpointing(TrainerProgressState())
    accelerator.step = 7
    checkpoint = tmp_path / "native-phase"
    accelerator.save_state(str(checkpoint))
    accelerator.step = 99

    accelerator_load_state(accelerator, str(checkpoint))

    assert accelerator.step == 7
