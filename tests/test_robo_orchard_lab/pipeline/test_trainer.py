# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import importlib
import warnings
from dataclasses import dataclass
from typing import Optional, cast
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.pipeline as pipeline_module
from robo_orchard_lab.dataset.robot import (
    DatasetItem,
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from robo_orchard_lab.processing.io_processor import (
    EnvelopeIOProcessor,
    EnvelopeIOProcessorCfg,
    PipelineEnvelope,
    compose_envelope,
)
from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.envelope import (
    ModelIOProcessorEnvelopeAdapter,
)
from robo_orchard_lab.processing.step_processor import (
    DeprecatedError as StepProcessorDeprecatedError,
    SimpleStepProcessor,
    StepProcessorFromCallable,
)


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0


class DummyPipelineHook(PipelineHooks):
    def __init__(self):
        super().__init__()

        self._on_loop_begin_cnt = 0
        self._on_loop_end_cnt = 0
        self._on_epoch_begin_cnt = 0
        self._on_epoch_end_cnt = 0
        self._on_step_begin_cnt = 0
        self._on_step_end_cnt = 0

        self.register_hook(
            "on_loop",
            HookContext.from_callable(
                before=self.on_loop_begin, after=self.on_loop_end
            ),
        )
        self.register_hook(
            "on_epoch",
            HookContext.from_callable(
                before=self.on_epoch_begin, after=self.on_epoch_end
            ),
        )
        self.register_hook(
            "on_step",
            HookContext.from_callable(
                before=self.on_step_begin, after=self.on_step_end
            ),
        )
        self._on_epoch_begin_state: Optional[TrainerState] = None
        self._on_epoch_end_state: Optional[TrainerState] = None

        self._on_step_begin_state: Optional[TrainerState] = None
        self._on_step_end_state: Optional[TrainerState] = None

    def on_loop_begin(self, args: PipelineHookArgs):
        self._on_loop_begin_cnt += 1

    def on_loop_end(self, args: PipelineHookArgs):
        self._on_loop_end_cnt += 1

    def on_epoch_begin(self, args: PipelineHookArgs):
        self._on_epoch_begin_cnt += 1
        self._on_epoch_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )

    def on_epoch_end(self, args: PipelineHookArgs):
        self._on_epoch_end_cnt += 1
        self._on_epoch_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        assert self._on_epoch_begin_state is not None
        assert (
            self._on_epoch_end_state.epoch == self._on_epoch_begin_state.epoch
        )
        assert (
            self._on_epoch_end_state.global_step
            != self._on_epoch_begin_state.global_step
        )
        assert self._on_epoch_end_state.step != self._on_epoch_begin_state.step

    def on_step_begin(self, args: PipelineHookArgs):
        self._on_step_begin_cnt += 1
        self._on_step_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )

    def on_step_end(self, args: PipelineHookArgs):
        self._on_step_end_cnt += 1
        self._on_step_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        assert self._on_step_begin_state is not None
        assert self._on_step_end_state.step == self._on_step_begin_state.step
        assert (
            self._on_step_end_state.global_step
            == self._on_step_begin_state.global_step
        )
        assert self._on_step_end_state.epoch == self._on_step_begin_state.epoch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class DummyBatchProcessor(SimpleStepProcessor):
    def forward(self, model: torch.nn.Module, batch: torch.Tensor):
        if (
            self.accelerator is not None
            and batch.device != self.accelerator.device
        ):
            batch = batch.to(self.accelerator.device)

        outputs = model(batch)
        loss = torch.mean((outputs - 1) ** 2)
        return outputs, loss


class AddTensorIOProcessor(ModelIOProcessor):
    cfg: "AddTensorIOProcessorCfg"

    def pre_process(self, data: torch.Tensor) -> torch.Tensor:
        return data + self.cfg.pre_add

    def post_process(
        self,
        model_outputs: torch.Tensor,
        model_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del model_input
        return model_outputs + self.cfg.post_add


class AddTensorIOProcessorCfg(ModelIOProcessorCfg[AddTensorIOProcessor]):
    class_type: ClassType[AddTensorIOProcessor] = AddTensorIOProcessor
    pre_add: float = 0.0
    post_add: float = 0.0


class ModuleBackedIOProcessor(ModelIOProcessor, torch.nn.Module):
    cfg: "ModuleBackedIOProcessorCfg"

    def __init__(self, cfg: "ModuleBackedIOProcessorCfg"):
        torch.nn.Module.__init__(self)
        ModelIOProcessor.__init__(self, cfg)
        self.linear = torch.nn.Linear(2, 2)

    def pre_process(self, data: torch.Tensor) -> torch.Tensor:
        return self.linear(data)

    def post_process(
        self,
        model_outputs: torch.Tensor,
        model_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del model_input
        return model_outputs


class ModuleBackedIOProcessorCfg(ModelIOProcessorCfg[ModuleBackedIOProcessor]):
    class_type: ClassType[ModuleBackedIOProcessor] = ModuleBackedIOProcessor


class FailingPreProcessIOProcessor(ModelIOProcessor):
    cfg: "FailingPreProcessIOProcessorCfg"

    def __init__(self, cfg: "FailingPreProcessIOProcessorCfg"):
        super().__init__(cfg)
        self.pre_process_calls = 0
        self.post_process_calls = 0

    def pre_process(self, data):
        del data
        self.pre_process_calls += 1
        raise RuntimeError("pre_process failed")

    def post_process(self, model_outputs, model_input=None):
        del model_outputs, model_input
        self.post_process_calls += 1
        return None


class FailingPreProcessIOProcessorCfg(
    ModelIOProcessorCfg[FailingPreProcessIOProcessor]
):
    class_type: ClassType[FailingPreProcessIOProcessor] = (
        FailingPreProcessIOProcessor
    )


class RecordingBoundaryIOProcessor(ModelIOProcessor):
    cfg: "RecordingBoundaryIOProcessorCfg"

    def __init__(self, cfg: "RecordingBoundaryIOProcessorCfg"):
        super().__init__(cfg)
        self.pre_inputs: list[object] = []
        self.post_inputs: list[object] = []
        self.post_outputs: list[object] = []

    def pre_process(self, data):
        self.pre_inputs.append(data)
        return data

    def post_process(self, model_outputs, model_input=None):
        self.post_outputs.append(model_outputs)
        self.post_inputs.append(model_input)
        return {
            "model_outputs": model_outputs,
            "model_input": model_input,
        }


class RecordingBoundaryIOProcessorCfg(
    ModelIOProcessorCfg[RecordingBoundaryIOProcessor]
):
    class_type: ClassType[RecordingBoundaryIOProcessor] = (
        RecordingBoundaryIOProcessor
    )


class EnvelopeRecordingIOProcessor(EnvelopeIOProcessor):
    cfg: "EnvelopeRecordingIOProcessorCfg"

    def __init__(self, cfg: "EnvelopeRecordingIOProcessorCfg"):
        super().__init__(cfg)
        self.pre_contexts: list[dict[str, torch.Tensor]] = []
        self.post_inputs: list[torch.Tensor] = []
        self.post_context_refs: list[dict[str, torch.Tensor]] = []
        self.post_contexts: list[dict[str, torch.Tensor]] = []

    def pre_process(self, data: PipelineEnvelope) -> PipelineEnvelope:
        raw_batch = cast(torch.Tensor, data.model_input)
        processor_context = {
            "raw_batch": raw_batch.clone(),
            "marker": torch.tensor(self.cfg.context_marker),
        }
        self.pre_contexts.append(processor_context)
        return PipelineEnvelope(
            model_input=raw_batch + self.cfg.pre_add,
            processor_context=processor_context,
        )

    def post_process(
        self,
        model_outputs,
        *,
        model_input=None,
        processor_context=None,
    ):
        self.post_inputs.append(cast(torch.Tensor, model_input).clone())
        context = cast(dict[str, torch.Tensor], processor_context)
        self.post_context_refs.append(context)
        self.post_contexts.append(
            {
                "raw_batch": context["raw_batch"].clone(),
                "marker": context["marker"].clone(),
            }
        )
        return {
            "model_outputs": model_outputs,
            "model_input": model_input,
            "processor_context": processor_context,
        }


class EnvelopeRecordingIOProcessorCfg(
    EnvelopeIOProcessorCfg[EnvelopeRecordingIOProcessor]
):
    class_type: ClassType[EnvelopeRecordingIOProcessor] = (
        EnvelopeRecordingIOProcessor
    )
    pre_add: float = 0.0
    context_marker: float = 0.0


class ForwardOnlyStepProcessor(SimpleStepProcessor):
    def __init__(self, **kwargs):
        super().__init__(need_backward=False, **kwargs)
        self.forward_batches: list[torch.Tensor] = []

    def forward(self, model, batch):
        self.forward_batches.append(batch.clone())
        return model(batch), None


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data[idx]
        return torch.tensor([value, value], dtype=torch.float32)


class ArrayDatasetItem(DatasetItem[ArrayDataset]):
    class_type: ClassType[ArrayDataset] = ArrayDataset

    data: list[int]

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> ArrayDataset:
        return ArrayDataset(self.data)


def test_simple_step_processor_with_io_processor():
    io_processor = AddTensorIOProcessor(
        AddTensorIOProcessorCfg(pre_add=1.0, post_add=3.0)
    )
    step_processor = ForwardOnlyStepProcessor(
        io_processor=io_processor,
        apply_post_process=True,
    )
    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch * 2,
    )

    expected_model_input = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    expected_outputs = torch.tensor([[7.0, 9.0]], dtype=torch.float32)

    assert len(step_processor.forward_batches) == 1
    assert torch.equal(step_processor.forward_batches[0], expected_model_input)
    assert isinstance(hook_args.batch, torch.Tensor)
    assert isinstance(hook_args.model_outputs, torch.Tensor)
    assert torch.equal(hook_args.batch, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(hook_args.model_outputs, expected_outputs)
    assert hook_args.reduce_loss is None


def test_simple_step_processor_with_envelope_io_processor():
    io_processor = EnvelopeRecordingIOProcessor(
        EnvelopeRecordingIOProcessorCfg(pre_add=1.0, context_marker=7.0)
    )
    step_processor = ForwardOnlyStepProcessor(
        io_processor=io_processor,
        apply_post_process=True,
    )
    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch * 2,
    )

    expected_model_input = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    assert step_processor.io_processor is io_processor
    assert step_processor.resolved_envelope_processor is io_processor
    assert len(step_processor.forward_batches) == 1
    assert torch.equal(step_processor.forward_batches[0], expected_model_input)
    assert torch.equal(hook_args.batch, torch.tensor([[1.0, 2.0]]))
    assert isinstance(hook_args.model_outputs, dict)
    model_outputs = cast(dict[str, object], hook_args.model_outputs)
    assert torch.equal(
        model_outputs["model_outputs"], expected_model_input * 2
    )
    assert torch.equal(model_outputs["model_input"], expected_model_input)
    processor_context = cast(
        dict[str, torch.Tensor], model_outputs["processor_context"]
    )
    assert len(io_processor.pre_contexts) == 1
    assert len(io_processor.post_context_refs) == 1
    assert processor_context is io_processor.pre_contexts[0]
    assert processor_context is io_processor.post_context_refs[0]
    assert torch.equal(
        processor_context["raw_batch"],
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    assert torch.equal(processor_context["marker"], torch.tensor(7.0))
    assert processor_context["raw_batch"] is not hook_args.batch
    assert len(io_processor.post_inputs) == 1
    assert torch.equal(io_processor.post_inputs[0], expected_model_input)
    assert torch.equal(
        io_processor.post_contexts[0]["raw_batch"],
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )


def test_simple_step_processor_keeps_legacy_io_processor_unprepared():
    io_processor = ModuleBackedIOProcessor(ModuleBackedIOProcessorCfg())
    step_processor = ForwardOnlyStepProcessor(io_processor=io_processor)
    accelerator = Accelerator(device_placement=True)
    accelerator.prepare = MagicMock(
        side_effect=AssertionError(
            "SimpleStepProcessor must not call accelerator.prepare."
        )
    )
    hook_args = PipelineHookArgs(
        accelerator=accelerator,
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch,
    )

    accelerator.prepare.assert_not_called()
    assert step_processor.io_processor is io_processor
    assert step_processor.resolved_envelope_processor is not None
    assert step_processor.resolved_envelope_processor.legacy is io_processor


def test_simple_step_processor_keeps_composed_legacy_modules_unprepared():
    first_io_processor = ModuleBackedIOProcessor(ModuleBackedIOProcessorCfg())
    second_io_processor = ModuleBackedIOProcessor(ModuleBackedIOProcessorCfg())
    composed = compose_envelope(first_io_processor, second_io_processor)
    step_processor = ForwardOnlyStepProcessor(io_processor=composed)
    accelerator = Accelerator(device_placement=True)
    accelerator.prepare = MagicMock(
        side_effect=AssertionError(
            "SimpleStepProcessor must not call accelerator.prepare."
        )
    )
    hook_args = PipelineHookArgs(
        accelerator=accelerator,
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch,
    )

    accelerator.prepare.assert_not_called()
    assert step_processor.io_processor is composed
    assert step_processor.resolved_envelope_processor is composed
    first_adapter = cast(
        ModelIOProcessorEnvelopeAdapter,
        composed.processors[0],
    )
    second_adapter = cast(
        ModelIOProcessorEnvelopeAdapter,
        composed.processors[1],
    )
    assert first_adapter.legacy is first_io_processor
    assert second_adapter.legacy is second_io_processor


def test_simple_step_processor_io_processor_reassignment_updates_runtime():
    step_processor = ForwardOnlyStepProcessor(apply_post_process=True)
    io_processor = EnvelopeRecordingIOProcessor(
        EnvelopeRecordingIOProcessorCfg(pre_add=1.0, context_marker=7.0)
    )
    step_processor.io_processor = io_processor
    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch * 2,
    )

    expected_model_input = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    assert step_processor.io_processor is io_processor
    assert step_processor.resolved_envelope_processor is io_processor
    assert len(step_processor.forward_batches) == 1
    assert torch.equal(step_processor.forward_batches[0], expected_model_input)
    assert isinstance(hook_args.model_outputs, dict)
    model_outputs = cast(dict[str, object], hook_args.model_outputs)
    assert torch.equal(
        model_outputs["model_outputs"],
        expected_model_input * 2,
    )
    assert torch.equal(model_outputs["model_input"], expected_model_input)
    processor_context = cast(
        dict[str, torch.Tensor], model_outputs["processor_context"]
    )
    assert len(io_processor.pre_contexts) == 1
    assert len(io_processor.post_context_refs) == 1
    assert processor_context is io_processor.pre_contexts[0]
    assert processor_context is io_processor.post_context_refs[0]
    assert processor_context["raw_batch"] is not hook_args.batch


def test_simple_step_processor_pre_process_error_keeps_hook_args_clean():
    io_processor = FailingPreProcessIOProcessor(
        FailingPreProcessIOProcessorCfg()
    )
    forward_state = {"calls": 0}

    def forward_fn(model, batch):
        del model, batch
        forward_state["calls"] += 1
        return None, None

    step_processor = SimpleStepProcessor.from_callable(
        forward_fn,
        need_backward=False,
        io_processor=io_processor,
        apply_post_process=True,
    )
    original_batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        batch=original_batch.clone(),
    )

    with pytest.raises(RuntimeError, match="pre_process failed"):
        step_processor(
            pipeline_hooks=PipelineHooks(),
            on_batch_hook_args=hook_args,
            model=lambda batch: batch,
        )

    assert io_processor.pre_process_calls == 1
    assert io_processor.post_process_calls == 0
    assert forward_state["calls"] == 0
    assert isinstance(hook_args.batch, torch.Tensor)
    assert torch.equal(hook_args.batch, original_batch)
    assert hook_args.model_outputs is None
    assert hook_args.reduce_loss is None


@pytest.mark.parametrize(
    ("batch", "is_tensor"),
    [
        pytest.param(None, False, id="none-batch"),
        pytest.param(
            torch.empty((0, 2), dtype=torch.float32),
            True,
            id="empty-tensor-batch",
        ),
    ],
)
def test_simple_step_processor_post_process_handles_boundary_batches(
    batch,
    is_tensor: bool,
):
    io_processor = RecordingBoundaryIOProcessor(
        RecordingBoundaryIOProcessorCfg()
    )
    step_processor = SimpleStepProcessor.from_callable(
        lambda model, model_batch: (model(model_batch), None),
        need_backward=False,
        io_processor=io_processor,
        apply_post_process=True,
    )
    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        batch=batch,
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda model_batch: model_batch,
    )

    assert len(io_processor.pre_inputs) == 1
    assert len(io_processor.post_inputs) == 1
    assert len(io_processor.post_outputs) == 1
    assert isinstance(hook_args.model_outputs, dict)
    model_outputs = cast(dict[str, object], hook_args.model_outputs)
    if is_tensor:
        assert isinstance(io_processor.pre_inputs[0], torch.Tensor)
        assert isinstance(io_processor.post_inputs[0], torch.Tensor)
        assert isinstance(io_processor.post_outputs[0], torch.Tensor)
        assert torch.equal(io_processor.pre_inputs[0], batch)
        assert torch.equal(io_processor.post_inputs[0], batch)
        assert torch.equal(io_processor.post_outputs[0], batch)
        assert isinstance(model_outputs["model_input"], torch.Tensor)
        assert isinstance(model_outputs["model_outputs"], torch.Tensor)
        assert torch.equal(model_outputs["model_input"], batch)
        assert torch.equal(model_outputs["model_outputs"], batch)
    else:
        assert io_processor.pre_inputs[0] is None
        assert io_processor.post_inputs[0] is None
        assert io_processor.post_outputs[0] is None
        assert model_outputs["model_input"] is None
        assert model_outputs["model_outputs"] is None
    assert hook_args.reduce_loss is None


def test_simple_step_processor_reduces_loss_in_multi_process_path(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(device_placement=True)
    monkeypatch.setattr(accelerator.state, "num_processes", 2)
    reduce_calls: list[tuple[torch.Tensor, str]] = []

    def fake_reduce(loss: torch.Tensor, reduction: str = "mean"):
        reduce_calls.append((loss.clone(), reduction))
        return loss / 2

    monkeypatch.setattr(accelerator, "reduce", fake_reduce)

    step_processor = SimpleStepProcessor.from_callable(
        lambda model, batch: (model(batch), torch.tensor(4.0)),
        need_backward=False,
        io_processor=AddTensorIOProcessor(
            AddTensorIOProcessorCfg(pre_add=1.0, post_add=3.0)
        ),
        apply_post_process=True,
    )
    hook_args = PipelineHookArgs(
        accelerator=accelerator,
        batch=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    step_processor(
        pipeline_hooks=PipelineHooks(),
        on_batch_hook_args=hook_args,
        model=lambda batch: batch * 2,
    )

    assert len(reduce_calls) == 1
    assert reduce_calls[0][1] == "mean"
    assert torch.equal(reduce_calls[0][0], torch.tensor(4.0))
    assert isinstance(hook_args.model_outputs, torch.Tensor)
    assert hook_args.reduce_loss is not None
    assert torch.equal(
        hook_args.model_outputs,
        torch.tensor([[7.0, 9.0]], dtype=torch.float32),
    )
    assert torch.equal(hook_args.reduce_loss, torch.tensor(2.0))


def test_legacy_batch_processor_facade_only_exports_legacy_names():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_batch_processor = importlib.import_module(
            "robo_orchard_lab.pipeline.batch_processor"
        )
    with pytest.warns(DeprecationWarning):
        reloaded_legacy_batch_processor = importlib.reload(
            legacy_batch_processor
        )
    assert issubclass(
        reloaded_legacy_batch_processor.SimpleBatchProcessor,
        SimpleStepProcessor,
    )
    assert not hasattr(legacy_batch_processor, "SimpleStepProcessor")
    assert not hasattr(legacy_batch_processor, "BatchStepProcessorMixin")


def test_legacy_batch_processor_from_callable_preserves_facade_type():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_batch_processor = importlib.import_module(
            "robo_orchard_lab.pipeline.batch_processor"
        )
        legacy_batch_processor_simple = importlib.import_module(
            "robo_orchard_lab.pipeline.batch_processor.simple"
        )

    processor = legacy_batch_processor.SimpleBatchProcessor.from_callable(
        lambda model, batch: (batch, None),
        need_backward=False,
    )

    assert isinstance(processor, StepProcessorFromCallable)
    assert isinstance(
        processor, legacy_batch_processor_simple.BatchProcessorFromCallable
    )


def test_legacy_batch_processor_transforms_still_raise_deprecated_error():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_batch_processor_simple = importlib.import_module(
            "robo_orchard_lab.pipeline.batch_processor.simple"
        )

    class LegacyDummyBatchProcessor(
        legacy_batch_processor_simple.SimpleBatchProcessor
    ):
        def forward(self, model, batch):
            del model
            return batch, None

    with pytest.raises(StepProcessorDeprecatedError):
        LegacyDummyBatchProcessor(transforms=[])

    with pytest.raises(StepProcessorDeprecatedError):
        legacy_batch_processor_simple.BatchProcessorFromCallable(
            lambda model, batch: (batch, None),
            need_backward=False,
            transforms=[],
        )

    with pytest.raises(StepProcessorDeprecatedError):
        legacy_batch_processor_simple.SimpleBatchProcessor.from_callable(
            lambda model, batch: (batch, None),
            need_backward=False,
            transforms=[],
        )


def test_legacy_trainer_facades_export_runtime_types():
    from robo_orchard_lab.pipeline.training.trainer import SimpleTrainer

    with pytest.warns(DeprecationWarning):
        legacy_hook_based_trainer = importlib.import_module(
            "robo_orchard_lab.pipeline.hook_based_trainer"
        )
        legacy_hook_based_trainer = importlib.reload(legacy_hook_based_trainer)

    with pytest.warns(DeprecationWarning):
        legacy_trainer_module = importlib.import_module(
            "robo_orchard_lab.pipeline.trainer"
        )
        legacy_trainer_module = importlib.reload(legacy_trainer_module)

    assert legacy_hook_based_trainer.HookBasedTrainer is HookBasedTrainer
    assert legacy_trainer_module.SimpleTrainer is SimpleTrainer
    assert pipeline_module.HookBasedTrainer is HookBasedTrainer
    assert pipeline_module.SimpleTrainer is SimpleTrainer


@pytest.fixture(scope="function")
def dummy_trainer():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor([[0.5, 0.5], [0.1, 0.2]], dtype=torch.float32)
        ),
    )

    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(device_placement=True)
    batch_processor = DummyBatchProcessor()

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=batch_processor,
        max_epoch=1,
    )

    return trainer


def test_trainer_initialization(dummy_trainer):
    assert dummy_trainer.max_epoch == 1


def test_hook_based_trainer_prepares_iterable_dataloader_once(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        ),
    )
    monkeypatch.setattr(accelerator.state, "num_processes", 4)
    monkeypatch.setattr(accelerator.state, "process_index", 0)
    monkeypatch.setattr(accelerator.state, "local_process_index", 0)

    dataset = DictIterableDataset(
        [ArrayDatasetItem(data=list(range(32)))],
    )
    dataloader = RODataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
    )
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epoch=1,
    )

    assert isinstance(trainer.dataloader, DataLoaderShard)
    base_dataset = trainer.dataloader.base_dataloader.dataset
    assert isinstance(base_dataset, DictIterableDataset)
    assert base_dataset.shard_kwargs.shard_strategy == "pad_last"
    assert base_dataset.dataset_items[0].num_shards == 4
    assert base_dataset.dataset_items[0].shard_id == 0
    assert base_dataset.total_iterator_length == 8


def test_hook_based_trainer_early_break_cleans_accelerate_dataloader_state(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        ),
    )
    monkeypatch.setattr(accelerator.state, "num_processes", 4)
    monkeypatch.setattr(accelerator.state, "process_index", 0)
    monkeypatch.setattr(accelerator.state, "local_process_index", 0)

    dataset = DictIterableDataset(
        [ArrayDatasetItem(data=list(range(32)))],
    )
    dataloader = RODataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
    )
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_step=1,
    )

    trainer()

    assert not accelerator.gradient_state.in_dataloader
    assert (
        sum(
            ref is not None
            for ref in accelerator.gradient_state.dataloader_references
        )
        == 0
    )


def test_training_loop(dummy_trainer: HookBasedTrainer):
    hook = DummyPipelineHook()
    dummy_trainer.hooks = hook
    dummy_trainer()

    assert dummy_trainer.dataloader is not None

    assert hook._on_loop_begin_cnt == 1
    assert hook._on_epoch_begin_cnt == 1
    assert hook._on_step_begin_cnt == len(dummy_trainer.dataloader)
    assert hook._on_step_end_cnt == len(dummy_trainer.dataloader)
    assert hook._on_epoch_end_cnt == 1
    assert hook._on_loop_end_cnt == 1


def test_optimizer_and_scheduler(dummy_trainer):
    initial_lr = dummy_trainer.optimizer.param_groups[0]["lr"]
    dummy_trainer()
    final_lr = dummy_trainer.optimizer.param_groups[0]["lr"]

    assert final_lr < initial_lr


if __name__ == "__main__":
    pytest.main(["-s", __file__])
