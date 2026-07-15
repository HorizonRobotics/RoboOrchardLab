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
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
    DummyOptim,
    DummyScheduler,
)
from robo_orchard_core.utils.config import ClassType
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.pipeline as pipeline_module
import robo_orchard_lab.pipeline.training as training_module
import robo_orchard_lab.pipeline.training.hook_based_trainer as trainer_module
from robo_orchard_lab.dataset.robot import (
    DatasetItem,
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHookConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    MicroStepProgressState,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.hooks.optimizer import OptimizerHookConfig
from robo_orchard_lab.pipeline.training import _deepspeed as deepspeed_training
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from robo_orchard_lab.pipeline.training.trainer import SimpleTrainer
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

    def pre_process(
        self,
        data: PipelineEnvelope[torch.Tensor, None],
    ) -> PipelineEnvelope[torch.Tensor, dict[str, torch.Tensor]]:
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
        model_input: torch.Tensor | None = None,
        processor_context: (
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
        ) = None,
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


class FakeDeepSpeedEngine:
    def __init__(self) -> None:
        self.global_steps = 0
        self.skipped_steps = 0


class FakeDeepSpeedEngineWrapper:
    def __init__(self, engine: FakeDeepSpeedEngine) -> None:
        self.engine = engine


class FakeDeepSpeedAccelerator:
    distributed_type = DistributedType.DEEPSPEED

    def __init__(self, engine: FakeDeepSpeedEngine) -> None:
        self.deepspeed_engine_wrapped = FakeDeepSpeedEngineWrapper(engine)


class PrepareOnlyDeepSpeedAccelerator:
    distributed_type = DistributedType.DEEPSPEED

    def __init__(self, deepspeed_config: dict[str, object]) -> None:
        self._models: list[torch.nn.Module] = []
        self.deepspeed_plugin = SimpleNamespace(
            deepspeed_config=deepspeed_config,
            gradient_clipping=deepspeed_config.get("gradient_clipping", None),
        )
        self.prepare_calls = 0
        self.prepared_args: tuple[object, ...] = ()
        self.registered_for_checkpointing: list[object] = []

    def prepare(self, *args):
        self.prepare_calls += 1
        self.prepared_args = args
        return args

    def register_for_checkpointing(self, obj: object) -> None:
        self.registered_for_checkpointing.append(obj)


def _adamw_deepspeed_config() -> dict[str, Any]:
    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
    }


def _deepspeed_topology_config() -> dict[str, Any]:
    return {"zero_optimization": {"stage": 0}}


def build_deepspeed_observer_trainer(
    engine: FakeDeepSpeedEngine,
) -> HookBasedTrainer:
    trainer = HookBasedTrainer.__new__(HookBasedTrainer)
    trainer.accelerator = cast(Accelerator, FakeDeepSpeedAccelerator(engine))
    trainer.trainer_progress_state = trainer_module.TrainerProgressState()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    return trainer


def build_micro_step_hook_args(
    trainer: HookBasedTrainer,
    *,
    is_optimizer_step_boundary: bool,
) -> PipelineHookArgs:
    current_micro_step = (
        trainer.trainer_progress_state.preview_next_micro_step()
    )
    trainer.trainer_progress_state.commit_micro_step(current_micro_step)
    hook_args = PipelineHookArgs(
        accelerator=trainer.accelerator,
        micro_step=trainer.trainer_progress_state.micro_step,
        is_optimizer_step_boundary=is_optimizer_step_boundary,
    )
    trainer.trainer_progress_state.sync_pipeline_hook_arg(hook_args)
    hook_args.is_optimizer_step_boundary = is_optimizer_step_boundary
    return hook_args


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
    assert hook_args.reduced_backward_loss is None


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
    assert hook_args.reduced_backward_loss is None


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
    assert hook_args.reduced_backward_loss is None


def test_simple_step_processor_reduces_backward_loss_after_backward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Backward loss reduction runs after backward with a detached tensor."""

    accelerator = Accelerator(device_placement=True)
    monkeypatch.setattr(accelerator.state, "num_processes", 2)
    event_order: list[str] = []
    backward_calls: list[torch.Tensor] = []
    reduce_calls: list[tuple[torch.Tensor, str, bool]] = []

    def fake_backward(loss: torch.Tensor):
        event_order.append("backward")
        backward_calls.append(loss)

    def fake_reduce(loss: torch.Tensor, reduction: str = "mean"):
        event_order.append("reduce")
        reduce_calls.append((loss.clone(), reduction, loss.requires_grad))
        return loss / 2

    monkeypatch.setattr(accelerator, "backward", fake_backward)
    monkeypatch.setattr(accelerator, "reduce", fake_reduce)

    step_processor = SimpleStepProcessor.from_callable(
        lambda model, batch: (
            model(batch),
            torch.tensor(4.0, requires_grad=True),
        ),
        need_backward=True,
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

    assert event_order == ["backward", "reduce"]
    assert len(backward_calls) == 1
    assert len(reduce_calls) == 1
    assert reduce_calls[0][1] == "mean"
    assert torch.equal(reduce_calls[0][0], torch.tensor(4.0))
    assert reduce_calls[0][2] is False
    assert isinstance(hook_args.model_outputs, torch.Tensor)
    assert hook_args.reduced_backward_loss is not None
    assert torch.equal(
        hook_args.model_outputs,
        torch.tensor([[7.0, 9.0]], dtype=torch.float32),
    )
    assert torch.equal(hook_args.reduced_backward_loss, torch.tensor(2.0))


def test_pipeline_hook_args_reduce_loss_is_deprecated_read_only():
    """The legacy reduce_loss alias warns on read and cannot be assigned."""

    hook_args = PipelineHookArgs(
        accelerator=Accelerator(device_placement=True),
        reduced_backward_loss=torch.tensor(2.0),
    )

    with pytest.warns(DeprecationWarning, match="reduced_backward_loss"):
        assert torch.equal(hook_args.reduce_loss, torch.tensor(2.0))

    with pytest.raises(AttributeError):
        hook_args.reduce_loss = torch.tensor(3.0)


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
    assert (
        training_module.LRSchedulerFactory is trainer_module.LRSchedulerFactory
    )


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
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
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


def test_hook_based_trainer_supports_scheduler_coupled_accelerator():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(device_placement=True)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_hook_based_trainer_supports_manual_scheduler_with_accumulation():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
                dtype=torch.float32,
            )
        ),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=2,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)


def test_trainer_progress_state_tracks_micro_and_optimizer_steps():
    """Progress state separates micro steps from optimizer steps."""
    progress = trainer_module.TrainerProgressState()

    first_micro_step = progress.preview_next_micro_step()
    assert first_micro_step == MicroStepProgressState(
        epoch_step_id=0,
        global_step_id=0,
        index_in_optimizer_step=1,
        last_optimizer_step_size=0,
    )

    progress.commit_micro_step(first_micro_step)
    assert progress.step_id == 0
    assert progress.global_step_id == 0
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=1,
        last_optimizer_step_size=0,
    )

    progress.commit_optimizer_step(micro_steps=1)
    assert progress.step_id == 1
    assert progress.global_step_id == 1
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=1,
    )

    state_dict = progress.state_dict()
    assert state_dict["schema_version"] == 2
    restored = trainer_module.TrainerProgressState()
    restored.load_state_dict(state_dict)

    assert restored == progress
    assert isinstance(restored.micro_step, MicroStepProgressState)

    progress.update_epoch()
    assert progress.epoch_id == 1
    assert progress.step_id == 0
    assert progress.global_step_id == 1
    assert progress.micro_step.epoch_step_id == 0
    assert progress.micro_step.global_step_id == 1


def test_trainer_progress_state_rejects_legacy_checkpoint_payload():
    """Legacy dataloader-step checkpoints should fail with a clear error."""

    progress = trainer_module.TrainerProgressState()

    with pytest.raises(RuntimeError, match="Legacy TrainerProgressState"):
        progress.load_state_dict(
            {
                "epoch_id": 0,
                "step_id": 3,
                "global_step_id": 3,
            }
        )


def test_trainer_progress_state_rejects_unsupported_schema_version():
    """Unknown progress schemas should fail before partial state mutation."""

    progress = trainer_module.TrainerProgressState()

    with pytest.raises(RuntimeError, match="Unsupported.*schema_version"):
        progress.load_state_dict(
            {
                "schema_version": 999,
                "epoch_id": 0,
                "step_id": 0,
                "global_step_id": 0,
                "micro_step": {
                    "epoch_step_id": 0,
                    "global_step_id": 0,
                    "index_in_optimizer_step": 0,
                    "last_optimizer_step_size": 0,
                },
            }
        )

    assert progress == trainer_module.TrainerProgressState()


def test_trainer_progress_state_resets_optimizer_step_window():
    """Progress state can discard an uncommitted accumulation window."""
    progress = trainer_module.TrainerProgressState()
    current_micro_step = progress.preview_next_micro_step()
    progress.commit_micro_step(current_micro_step)

    progress.reset_optimizer_step_window()

    assert progress.global_step_id == 0
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=0,
    )


def test_hook_based_trainer_observes_deepspeed_accumulation_boundary():
    """DeepSpeed progress is committed from engine counters only."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)

    first_step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    first_hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=False,
    )

    committed = trainer._run_optimizer_step(
        first_hook_args,
        is_optimizer_step_boundary=False,
        micro_steps=first_hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=first_step_before,
    )

    assert committed is False
    assert trainer.trainer_progress_state.global_step_id == 0
    assert (
        trainer.trainer_progress_state.micro_step.index_in_optimizer_step == 1
    )

    second_step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    second_hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )
    engine.global_steps = 1

    committed = trainer._run_optimizer_step(
        second_hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=second_hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=second_step_before,
    )

    assert committed is True
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=2,
        global_step_id=2,
        index_in_optimizer_step=0,
        last_optimizer_step_size=2,
    )
    trainer.optimizer.step.assert_not_called()
    trainer.lr_scheduler.step.assert_not_called()


def test_hook_based_trainer_observes_deepspeed_skipped_step():
    """DeepSpeed overflow skip resets the accumulation window only."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )
    engine.global_steps = 1
    engine.skipped_steps = 1

    committed = trainer._run_optimizer_step(
        hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=step_before,
    )

    assert committed is False
    assert trainer.trainer_progress_state.global_step_id == 0
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=0,
    )


def test_hook_based_trainer_requires_deepspeed_boundary_step():
    """DeepSpeed boundary without engine progress fails fast."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )

    with pytest.raises(RuntimeError, match="did not advance"):
        trainer._run_optimizer_step(
            hook_args,
            is_optimizer_step_boundary=True,
            micro_steps=hook_args.micro_step.index_in_optimizer_step,
            deepspeed_step_before=step_before,
        )


def test_deepspeed_progress_observer_requires_micro_step():
    """The DeepSpeed observer fails fast without trainer micro-step state."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    assert step_before is not None

    with pytest.raises(RuntimeError, match="PipelineHookArgs.micro_step"):
        deepspeed_training.commit_deepspeed_optimizer_progress(
            accelerator=trainer.accelerator,
            progress_state=trainer.trainer_progress_state,
            hook_args=PipelineHookArgs(
                accelerator=trainer.accelerator,
                is_optimizer_step_boundary=True,
            ),
            is_optimizer_step_boundary=True,
            micro_steps=1,
            step_before=step_before,
        )


def test_hook_based_trainer_ignores_deprecated_user_optimizer_hook():
    """Deprecated user OptimizerHook config is a no-op compatibility shim."""
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.warns(DeprecationWarning, match="OptimizerHook is deprecated"):
        trainer = HookBasedTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            batch_processor=DummyBatchProcessor(),
            hooks=[OptimizerHookConfig()],
            max_epoch=1,
        )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_hook_based_trainer_runs_internal_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
):
    """HookBasedTrainer applies configured grad clipping internally."""
    clip_calls: list[tuple[int, float, float]] = []
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        clip_calls.append((len(list(params)), max_norm, norm_type))

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
            norm_type=2.0,
        ),
        max_step=1,
    )

    trainer()

    assert clip_calls == [(2, 1.0, 2.0)]


def test_hook_based_trainer_clips_only_accumulation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    """Trainer-owned grad clipping waits for optimizer-step boundaries."""

    events: list[str] = []
    clip_calls: list[tuple[int, float, float]] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")
        clip_calls.append((len(list(params)), max_norm, norm_type))

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
            norm_type=2.0,
        ),
        max_step=1,
    )

    trainer()

    assert events == ["clip", "optimizer_step"]
    assert clip_calls == [(2, 1.0, 2.0)]
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


def test_hook_based_trainer_clips_skipped_boundary_without_commit():
    """Skipped optimizer boundaries may clip but must not commit progress."""

    events: list[str] = []

    class SkippedStepAccelerator:
        distributed_type = DistributedType.NO
        optimizer_step_was_skipped = True

        def clip_grad_norm_(self, params, max_norm, norm_type):
            del max_norm, norm_type
            events.append(f"clip:{len(list(params))}")

    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])
    optimizer = MagicMock()
    optimizer.param_groups = [{"params": [param]}]
    trainer = HookBasedTrainer.__new__(HookBasedTrainer)
    trainer.accelerator = cast(Accelerator, SkippedStepAccelerator())
    trainer.trainer_progress_state = trainer_module.TrainerProgressState()
    trainer.optimizer = optimizer
    trainer.lr_scheduler = MagicMock()
    trainer._grad_clip = GradientClippingHookConfig(
        clip_mode="norm",
        max_norm=1.0,
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )

    committed = trainer._run_optimizer_step(
        hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=hook_args.micro_step.index_in_optimizer_step,
    )

    assert committed is False
    assert events == ["clip:1"]
    optimizer.step.assert_called_once()
    trainer.lr_scheduler.step.assert_not_called()
    optimizer.zero_grad.assert_called_once()
    assert trainer.trainer_progress_state.global_step_id == 0
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=0,
    )


def test_hook_based_trainer_rejects_duplicate_gradient_clipping_config():
    """Logical grad clipping has a single trainer-owned source of truth."""

    with pytest.raises(ValueError, match="Only one gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=[
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=1.0,
                )
            ],
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            ),
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_direct_hook_with_logical_grad_clip():
    """A directly constructed legacy hook cannot share trainer ownership."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="already constructed"):
        trainer_module._split_gradient_clipping_config(
            hooks=[legacy_hook],
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            ),
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_direct_grad_clip_hook_for_deepspeed():
    """DeepSpeed cannot use a late GradientClippingHook instance."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="DeepSpeed gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=[legacy_hook],
            grad_clip=None,
            is_deepspeed=True,
        )


def test_hook_based_trainer_allows_opaque_deepspeed_hooks():
    """Opaque PipelineHooks are not rejected without a grad-clip marker."""

    opaque_hooks = PipelineHooks()

    remaining_hooks, grad_clip = (
        trainer_module._split_gradient_clipping_config(
            hooks=[opaque_hooks],
            grad_clip=None,
            is_deepspeed=True,
        )
    )

    assert remaining_hooks == [opaque_hooks]
    assert grad_clip is None


def test_hook_based_trainer_rejects_direct_hook_before_grad_clip_config():
    """Direct hook before logical config should not double clip."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="already constructed"):
        trainer_module._split_gradient_clipping_config(
            hooks=[
                legacy_hook,
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=0.5,
                ),
            ],
            grad_clip=None,
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_multiple_direct_grad_clip_hooks():
    """Multiple direct legacy hooks would clip the same gradients twice."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hooks = [
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            )(),
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.5,
            )(),
        ]

    with pytest.raises(ValueError, match="Only one gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=legacy_hooks,
            grad_clip=None,
            is_deepspeed=False,
        )


def test_hook_based_trainer_injects_missing_deepspeed_gradient_clipping():
    """DeepSpeed config receives trainer max_norm before prepare."""

    deepspeed_config: dict[str, object] = {}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25


def test_hook_based_trainer_overwrites_deepspeed_auto_with_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    """Trainer grad_clip owns DeepSpeed 'auto' when both are provided."""

    warnings_: list[str] = []

    def record_warning(message: str, *args, **kwargs) -> None:
        del kwargs
        warnings_.append(message % args if args else message)

    monkeypatch.setattr(deepspeed_training.logger, "warning", record_warning)
    deepspeed_config = _adamw_deepspeed_config()
    deepspeed_config["gradient_clipping"] = "auto"
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25
    assert len(warnings_) == 1
    assert "overriding it with grad_clip.max_norm=0.25" in warnings_[0]


def test_hook_based_trainer_accepts_matching_deepspeed_gradient_clipping():
    """Matching config and trainer grad_clip values share one numeric value."""

    deepspeed_config: dict[str, object] = {"gradient_clipping": "0.25"}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25


def test_hook_based_trainer_materializes_scheduler_factory_before_prepare():
    """A regular backend binds the factory to its code-owned optimizer."""

    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    factory_calls: list[torch.optim.Optimizer] = []

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        factory_calls.append(actual_optimizer)
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=DataLoader(
            TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
        ),
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=Accelerator(step_scheduler_with_optimizer=False),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert factory_calls == [optimizer]
    assert (
        trainer.lr_scheduler.scheduler.optimizer is trainer.optimizer.optimizer
    )


def test_hook_based_trainer_converts_adamw_factory_before_deepspeed_prepare():
    """Trainer routes a real AdamW and factory to one DeepSpeed Dummy pair."""

    deepspeed_config = _deepspeed_topology_config()
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [model.linear.weight],
                "lr": 0.01,
                "weight_decay": 0.001,
            },
            {
                "params": [model.linear.bias],
                "lr": 0.02,
                "betas": (0.8, 0.95),
                "eps": 1e-6,
            },
        ],
        lr=0.1,
        weight_decay=0.0005,
    )
    factory_calls: list[torch.optim.Optimizer] = []

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        factory_calls.append(actual_optimizer)
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    HookBasedTrainer(
        model=model,
        dataloader=DataLoader(
            TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
        ),
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert accelerator.prepare_calls == 1
    prepared_optimizer = accelerator.prepared_args[2]
    prepared_scheduler = accelerator.prepared_args[3]
    assert isinstance(prepared_optimizer, DummyOptim)
    assert isinstance(prepared_scheduler, DummyScheduler)
    assert prepared_scheduler.optimizer is prepared_optimizer
    assert factory_calls == []
    assert [group["lr"] for group in prepared_optimizer.params] == [0.01, 0.02]
    assert prepared_optimizer.params[0]["weight_decay"] == 0.001
    assert prepared_optimizer.params[1]["betas"] == (0.8, 0.95)
    assert prepared_optimizer.params[1]["eps"] == 1e-6
    assert all(
        set(group) == {"params", "lr", "weight_decay", "betas", "eps"}
        for group in prepared_optimizer.params
    )
    assert deepspeed_config["optimizer"] == {
        "type": "AdamW",
        "params": {
            "lr": 0.1,
            "weight_decay": 0.0005,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    }

    actual_optimizer = torch.optim.AdamW(
        prepared_optimizer.params,
        lr=0.1,
        weight_decay=0.0005,
    )
    actual_scheduler = prepared_scheduler.lr_scheduler_callable(
        actual_optimizer
    )
    assert actual_scheduler.optimizer is actual_optimizer
    assert factory_calls == [actual_optimizer]


@pytest.mark.parametrize("dummy_kind", ["optimizer", "scheduler"])
def test_hook_based_trainer_rejects_dummy_inputs_without_deepspeed(
    dummy_kind: str,
):
    """Dummy prepare inputs have no runtime owner on regular backends."""

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = SGD([parameter], lr=0.01)
    if dummy_kind == "optimizer":
        dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)
        optimizer_input = dummy_optimizer
        scheduler_input = DummyScheduler(
            optimizer=dummy_optimizer,
            lr_scheduler_callable=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )
    else:
        optimizer_input = optimizer
        scheduler_input = DummyScheduler(optimizer=optimizer)

    with pytest.raises(ValueError, match="only with a DeepSpeed accelerator"):
        trainer_module._resolve_optimizer_scheduler_before_prepare(
            accelerator=Accelerator(),
            optimizer=optimizer_input,
            lr_scheduler=scheduler_input,
        )


def test_deepspeed_prepare_accepts_existing_dummy_pair_and_factory():
    """Advanced callers can pass a valid dummy pair or a factory."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_adamw_deepspeed_config())
    parameter = torch.nn.Parameter(torch.ones(1))
    dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1)

    resolved_optimizer, resolved_scheduler = (
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=dummy_optimizer,
            lr_scheduler=build_scheduler,
        )
    )
    assert resolved_optimizer is dummy_optimizer
    assert isinstance(resolved_scheduler, DummyScheduler)
    assert resolved_scheduler.optimizer is dummy_optimizer
    assert resolved_scheduler.lr_scheduler_callable is build_scheduler

    assert deepspeed_training.prepare_deepspeed_optimizer_scheduler(
        accelerator=cast(Accelerator, accelerator),
        optimizer=dummy_optimizer,
        lr_scheduler=resolved_scheduler,
    ) == (dummy_optimizer, resolved_scheduler)


def test_deepspeed_prepare_accepts_fully_config_owned_dummy_pair():
    """A config-owned optimizer and scheduler use explicit placeholders."""

    config = _adamw_deepspeed_config()
    config["scheduler"] = {"type": "WarmupLR"}
    accelerator = PrepareOnlyDeepSpeedAccelerator(config)
    parameter = torch.nn.Parameter(torch.ones(1))
    dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)
    dummy_scheduler = DummyScheduler(optimizer=dummy_optimizer)

    assert deepspeed_training.prepare_deepspeed_optimizer_scheduler(
        accelerator=cast(Accelerator, accelerator),
        optimizer=dummy_optimizer,
        lr_scheduler=dummy_scheduler,
    ) == (dummy_optimizer, dummy_scheduler)


def test_deepspeed_prepare_rejects_dummy_optimizer_without_config_owner():
    """DeepSpeed must have a config optimizer before accepting DummyOptim."""

    accelerator = PrepareOnlyDeepSpeedAccelerator({})
    dummy_optimizer = DummyOptim(
        params=[torch.nn.Parameter(torch.ones(1))],
        lr=0.01,
    )
    dummy_scheduler = DummyScheduler(
        optimizer=dummy_optimizer,
        lr_scheduler_callable=lambda actual_optimizer: StepLR(
            actual_optimizer,
            step_size=1,
        ),
    )

    with pytest.raises(ValueError, match="explicit DummyOptim requires"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=dummy_optimizer,
            lr_scheduler=dummy_scheduler,
        )


def test_deepspeed_prepare_rejects_concrete_scheduler_for_real_optimizer():
    """A concrete scheduler cannot bind the later DeepSpeed optimizer."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="requires an LRSchedulerFactory"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=StepLR(optimizer, step_size=1),
        )


@pytest.mark.parametrize(
    ("optimizer", "match"),
    [
        (
            torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4),
            "only exact torch.optim.AdamW",
        ),
        (
            torch.optim.AdamW(
                [torch.nn.Parameter(torch.ones(1))],
                lr=1e-4,
                amsgrad=True,
            ),
            "amsgrad",
        ),
        (
            torch.optim.AdamW(
                [
                    {
                        "params": [torch.nn.Parameter(torch.ones(1))],
                        "amsgrad": True,
                    }
                ],
                lr=1e-4,
            ),
            "param group 0.*amsgrad",
        ),
    ],
)
def test_deepspeed_prepare_rejects_unsupported_optimizer_recipe(
    optimizer: torch.optim.Optimizer,
    match: str,
):
    """Runtime lowering accepts only a supported exact AdamW recipe."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())

    with pytest.raises((TypeError, ValueError), match=match):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_accepts_adamw_without_newer_defaults():
    """Older exact AdamW shapes have implicit decoupled weight decay."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )
    optimizer.defaults.pop("decoupled_weight_decay", None)
    for param_group in optimizer.param_groups:
        param_group.pop("decoupled_weight_decay", None)

    resolved_optimizer, _ = (
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )
    )

    assert isinstance(resolved_optimizer, DummyOptim)


def test_deepspeed_prepare_rejects_adamw_with_existing_state():
    """Lowering must not silently discard already-restored Adam moments."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    optimizer.state[parameter]["step"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match="pristine AdamW"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_rejects_config_and_factory_scheduler_owners():
    """Python factory and config scheduler cannot both own the schedule."""

    config = {
        **_deepspeed_topology_config(),
        "scheduler": {"type": "WarmupLR"},
    }
    accelerator = PrepareOnlyDeepSpeedAccelerator(config)
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="fully config-owned pair"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_rejects_duplicate_real_optimizer_source():
    """Real optimizer lowering cannot coexist with a config optimizer."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_adamw_deepspeed_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="must not also define optimizer"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_hook_based_trainer_rejects_conflicting_deepspeed_gradient_clipping():
    """Conflicting DeepSpeed config and trainer grad_clip fail fast."""

    deepspeed_config: dict[str, object] = {"gradient_clipping": 0.5}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    with pytest.raises(ValueError, match="Conflicting gradient clipping"):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
            ),
        )


@pytest.mark.parametrize(
    ("grad_clip", "match"),
    [
        (
            GradientClippingHookConfig(
                clip_mode="value",
                clip_value=0.1,
            ),
            "supports only norm clipping",
        ),
        (
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
                norm_type=1.0,
            ),
            "norm_type must be 2.0",
        ),
    ],
)
def test_hook_based_trainer_rejects_non_equivalent_deepspeed_grad_clip(
    grad_clip: GradientClippingHookConfig,
    match: str,
):
    """DeepSpeed config injection accepts only equivalent L2 norm clipping."""

    accelerator = PrepareOnlyDeepSpeedAccelerator({})

    with pytest.raises(ValueError, match=match):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=grad_clip,
        )


def test_hook_based_trainer_rejects_deepspeed_norm_without_max_norm():
    """DeepSpeed clipping cannot be configured without a max norm."""

    grad_clip = GradientClippingHookConfig(
        clip_mode="norm",
        max_norm=0.25,
    )
    grad_clip.max_norm = None
    accelerator = PrepareOnlyDeepSpeedAccelerator({})

    with pytest.raises(ValueError, match="requires max_norm"):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=grad_clip,
        )


def test_hook_based_trainer_leaves_deepspeed_auto_without_grad_clip():
    """Without trainer grad_clip, DeepSpeed auto stays owned by Accelerate."""

    deepspeed_config = _deepspeed_topology_config()
    deepspeed_config["gradient_clipping"] = "auto"
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert deepspeed_config["gradient_clipping"] == "auto"
    assert accelerator.prepare_calls == 1


def test_hook_based_trainer_injects_deepspeed_grad_clip_from_raw_hooks():
    """Raw GradientClippingHookConfig is routed before DeepSpeed prepare."""

    deepspeed_config = _deepspeed_topology_config()
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        hooks=[
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
            )
        ],
        max_step=1,
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25
    assert trainer._grad_clip is None
    assert accelerator.prepare_calls == 1


def test_hook_based_trainer_preserves_user_hook_order_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
):
    """User on_step after-hooks observe state before optimizer finalization.

    Post-update user hooks should use ``on_optimizer_step`` instead.
    """
    events: list[str] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    class RecordingStepHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            events.append(self.name)

    class RecordingOptimizerStepBeforeHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(before=self.on_optimizer_step_begin),
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs):
            events.append(self.name)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[
            RecordingStepHook("first_user_hook"),
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
                norm_type=2.0,
            ),
            RecordingOptimizerStepBeforeHook("user_optimizer_before"),
            RecordingStepHook("last_user_hook"),
        ],
        max_step=1,
    )

    trainer()

    assert events == [
        "first_user_hook",
        "last_user_hook",
        "user_optimizer_before",
        "clip",
        "optimizer_step",
    ]


def test_simple_trainer_passes_raw_gradient_clipping_hook_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """SimpleTrainer must not instantiate grad clip configs before parent."""

    events: list[str] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    class RecordingStepHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            events.append(self.name)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    with pytest.warns(DeprecationWarning):
        trainer = SimpleTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            batch_processor=DummyBatchProcessor(),
            hooks=[
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=1.0,
                    norm_type=2.0,
                ),
                RecordingStepHook("user_hook"),
            ],
            max_step=1,
        )

    trainer()

    assert events == [
        "user_hook",
        "clip",
        "optimizer_step",
    ]


def test_simple_trainer_rejects_duplicate_legacy_and_raw_grad_clip():
    """SimpleTrainer forwards duplicate logical grad clip sources.

    The parent trainer should then reject the duplicate logical ownership.
    """

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Only one gradient clipping"):
            SimpleTrainer(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                batch_processor=DummyBatchProcessor(),
                grad_clip_mode="norm",
                grad_max_norm=1.0,
                hooks=[
                    GradientClippingHookConfig(
                        clip_mode="norm",
                        max_norm=1.0,
                    )
                ],
                max_step=1,
            )


def test_hook_based_trainer_prepares_iterable_dataloader_once(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
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
        step_scheduler_with_optimizer=False,
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
    dummy_trainer.hooks += hook
    dummy_trainer()

    assert dummy_trainer.dataloader is not None

    assert hook._on_loop_begin_cnt == 1
    assert hook._on_epoch_begin_cnt == 1
    assert hook._on_step_begin_cnt == len(dummy_trainer.dataloader)
    assert hook._on_step_end_cnt == len(dummy_trainer.dataloader)
    assert hook._on_epoch_end_cnt == 1
    assert hook._on_loop_end_cnt == 1


def test_hook_based_trainer_counts_optimizer_steps_with_accumulation():
    """With accumulation, optimizer progress advances only on boundaries."""
    records: list[tuple[int, int, MicroStepProgressState, bool]] = []

    class RecordingStepHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            assert args.micro_step is not None
            records.append(
                (
                    args.step_id,
                    args.global_step_id,
                    args.micro_step,
                    args.is_optimizer_step_boundary,
                )
            )

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[RecordingStepHook()],
        max_step=2,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 2
    assert trainer.trainer_progress_state.step_id == 0
    assert trainer.trainer_progress_state.micro_step.global_step_id == 4
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )
    assert records == [
        (
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=1,
                global_step_id=1,
                index_in_optimizer_step=1,
                last_optimizer_step_size=0,
            ),
            False,
        ),
        (
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=2,
                last_optimizer_step_size=0,
            ),
            True,
        ),
        (
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=3,
                global_step_id=3,
                index_in_optimizer_step=1,
                last_optimizer_step_size=2,
            ),
            False,
        ),
        (
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=2,
                last_optimizer_step_size=2,
            ),
            True,
        ),
    ]


def test_hook_based_trainer_runs_optimizer_step_hook_on_boundaries():
    """on_optimizer_step wraps only optimizer-step boundaries."""
    records: list[
        tuple[
            str,
            int,
            int,
            MicroStepProgressState,
            bool,
            bool,
        ]
    ] = []

    class RecordingOptimizerStepHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(
                    before=self.on_optimizer_step_begin,
                    after=self.on_optimizer_step_end,
                ),
            )

        def _record(self, phase: str, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            records.append(
                (
                    phase,
                    args.step_id,
                    args.global_step_id,
                    args.micro_step,
                    args.is_optimizer_step_boundary,
                    args.is_optimizer_step_committed,
                )
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs) -> None:
            self._record("before", args)

        def on_optimizer_step_end(self, args: PipelineHookArgs) -> None:
            self._record("after", args)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[RecordingOptimizerStepHook()],
        max_step=2,
    )

    trainer()

    assert records == [
        (
            "before",
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=2,
                last_optimizer_step_size=0,
            ),
            True,
            False,
        ),
        (
            "after",
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=0,
                last_optimizer_step_size=2,
            ),
            True,
            True,
        ),
        (
            "before",
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=2,
                last_optimizer_step_size=2,
            ),
            True,
            False,
        ),
        (
            "after",
            2,
            2,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=0,
                last_optimizer_step_size=2,
            ),
            True,
            True,
        ),
    ]


def test_hook_based_trainer_ignores_user_boundary_enable_mutation():
    """User hook mutation cannot turn a non-boundary into a commit."""

    observed_step_events: list[tuple[int, bool]] = []

    class BoundaryEnablingHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(
                    before=None,
                    after=self.on_step_end,
                ),
            )

        def on_step_end(self, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            observed_step_events.append(
                (
                    args.micro_step.global_step_id,
                    args.is_optimizer_step_boundary,
                )
            )
            if args.micro_step.global_step_id == 1:
                args.is_optimizer_step_boundary = True

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25]],
                dtype=torch.float32,
            )
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[BoundaryEnablingHook()],
        max_step=1,
    )

    trainer()

    assert observed_step_events == [(1, False), (2, True)]
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step.global_step_id == 2
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


def test_hook_based_trainer_restores_boundary_snapshot_for_optimizer_hooks():
    """User hook mutation cannot corrupt optimizer boundary snapshots."""

    optimizer_events: list[tuple[str, bool, int, bool]] = []

    class BoundaryCorruptingHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(
                    before=None,
                    after=self.on_step_end,
                ),
            )
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(
                    before=self.on_optimizer_step_begin,
                    after=self.on_optimizer_step_end,
                ),
            )

        def on_step_end(self, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            if args.micro_step.global_step_id == 2:
                args.is_optimizer_step_boundary = False
                args.micro_step.index_in_optimizer_step = 99

        def _record(self, phase: str, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            optimizer_events.append(
                (
                    phase,
                    args.is_optimizer_step_boundary,
                    args.micro_step.index_in_optimizer_step,
                    args.is_optimizer_step_committed,
                )
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs) -> None:
            self._record("before", args)

        def on_optimizer_step_end(self, args: PipelineHookArgs) -> None:
            self._record("after", args)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25]],
                dtype=torch.float32,
            )
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[BoundaryCorruptingHook()],
        max_epoch=1,
    )

    trainer()

    assert optimizer_events == [
        ("before", True, 2, False),
        ("after", True, 0, True),
    ]
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step.global_step_id == 2
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


def test_hook_based_trainer_honors_peer_epoch_stop_before_step(
    monkeypatch: pytest.MonkeyPatch,
):
    event_order: list[str] = []
    close_reasons: list[trainer_module.DataloaderCloseReason] = []

    class RecordingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            event_order.append("step")
            return super().forward(model=model, batch=batch)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=RecordingBatchProcessor(),
        max_epoch=1,
    )
    check_results = iter([True, True])

    def record_sync() -> None:
        event_order.append("sync")

    def record_set_trigger() -> None:
        event_order.append("set")

    def record_check_trigger() -> bool:
        event_order.append("check")
        return next(check_results, False)

    monkeypatch.setattr(accelerator, "wait_for_everyone", record_sync)
    monkeypatch.setattr(accelerator, "set_trigger", record_set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", record_check_trigger)
    original_close = trainer_module.close_dataloader_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "close_dataloader_resources",
        record_close,
    )

    trainer()

    assert event_order == ["check", "set", "check"]
    assert trainer.trainer_progress_state.global_step_id == 0
    assert close_reasons == [
        trainer_module.DataloaderCloseReason.COORDINATED_EPOCH_END
    ]


def test_hook_based_trainer_sets_trigger_when_dataloader_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
):
    event_order: list[str] = []
    close_reasons: list[trainer_module.DataloaderCloseReason] = []

    class RecordingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            event_order.append("step")
            return super().forward(model=model, batch=batch)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.empty((0, 2), dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=RecordingBatchProcessor(),
        max_epoch=1,
    )
    check_results = iter([True, True])

    def record_sync() -> None:
        event_order.append("sync")

    def record_set_trigger() -> None:
        event_order.append("set")

    def record_check_trigger() -> bool:
        event_order.append("check")
        return next(check_results, False)

    monkeypatch.setattr(accelerator, "wait_for_everyone", record_sync)
    monkeypatch.setattr(accelerator, "set_trigger", record_set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", record_check_trigger)
    original_close = trainer_module.close_dataloader_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "close_dataloader_resources",
        record_close,
    )

    trainer()

    assert event_order == ["set", "check", "set", "check"]
    assert trainer.trainer_progress_state.global_step_id == 0
    assert close_reasons == [
        trainer_module.DataloaderCloseReason.EPOCH_EXHAUSTED
    ]


def test_hook_based_trainer_closes_max_step_with_max_step_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    """A local max-step stop must tear down the active iterator."""
    close_reasons: list[trainer_module.DataloaderCloseReason] = []
    teardown_primary_excs: list[BaseException | None] = []

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )
    original_close = trainer_module.close_dataloader_resources
    original_teardown = trainer_module._close_dataloader_owner_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    def record_teardown(*args, **kwargs):
        teardown_primary_excs.append(kwargs.get("primary_exc"))
        return original_teardown(*args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "close_dataloader_resources",
        record_close,
    )
    monkeypatch.setattr(
        trainer_module,
        "_close_dataloader_owner_resources",
        record_teardown,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert close_reasons == [trainer_module.DataloaderCloseReason.MAX_STEP_END]
    assert teardown_primary_excs == [None]


def test_hook_based_trainer_closes_exception_with_exception_abort_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    """An exception in the step body must preserve the primary failure."""
    close_reasons: list[trainer_module.DataloaderCloseReason] = []
    teardown_primary_excs: list[BaseException | None] = []

    class FailingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            raise RuntimeError("step failed")

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=FailingBatchProcessor(),
        max_epoch=1,
    )
    original_close = trainer_module.close_dataloader_resources
    original_teardown = trainer_module._close_dataloader_owner_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    def record_teardown(*args, **kwargs):
        teardown_primary_excs.append(kwargs.get("primary_exc"))
        return original_teardown(*args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "close_dataloader_resources",
        record_close,
    )
    monkeypatch.setattr(
        trainer_module,
        "_close_dataloader_owner_resources",
        record_teardown,
    )

    with pytest.raises(RuntimeError, match="step failed"):
        trainer()

    assert close_reasons == [
        trainer_module.DataloaderCloseReason.EXCEPTION_ABORT
    ]
    assert len(teardown_primary_excs) == 1
    assert isinstance(teardown_primary_excs[0], RuntimeError)
    assert trainer.trainer_progress_state.global_step_id == 0
    assert trainer.trainer_progress_state.micro_step.global_step_id == 0


def test_optimizer_and_scheduler(dummy_trainer):
    initial_lr = dummy_trainer.optimizer.param_groups[0]["lr"]
    dummy_trainer()
    final_lr = dummy_trainer.optimizer.param_groups[0]["lr"]

    assert final_lr < initial_lr


if __name__ == "__main__":
    pytest.main(["-s", __file__])
