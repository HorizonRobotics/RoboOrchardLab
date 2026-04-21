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
import copy
import importlib
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, cast

import pytest
import torch
import torch.nn as nn
from robo_orchard_core.utils.config import ClassType_co

import robo_orchard_lab.processing.io_processor as io_processor_module
from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.models.mixin import (
    ModelMixin,
    TorchModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.pipeline.inference import (
    InferencePipeline,
    InferencePipelineCfg,
    InferencePipelineMixin,
)
from robo_orchard_lab.processing.io_processor import (
    ComposedEnvelopeIOProcessor,
    ComposedEnvelopeIOProcessorCfg,
    EnvelopeIOProcessor,
    EnvelopeIOProcessorCfg,
    PipelineEnvelope,
    ProcessorContextStack,
    compose_envelope,
    compose_envelope_cfg,
)
from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.compose import (
    ComposedIOProcessor,
    ComposedIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.envelope import (
    ModelIOProcessorEnvelopeAdapter,
    ModelIOProcessorEnvelopeAdapterCfg,
    PostProcessContext,
    resolve_envelope_processor,
    resolve_envelope_processor_cfg,
)
from robo_orchard_lab.utils.path import DirectoryNotEmptyError

InputDict = Dict[str, torch.Tensor]
OutputDict = Dict[str, torch.Tensor]


class DummyModel(ModelMixin):
    def __init__(self, cfg: "DummyModelCfg"):
        super().__init__(cfg)
        self.linear = nn.Linear(10, 5)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {"output_data": self.linear(batch["input_data"]) * 2}


class DummyModelCfg(TorchModuleCfg[DummyModel]):
    class_type: ClassType_co[DummyModel] = DummyModel


class DummyProcessor(ModelIOProcessor):
    def pre_process(self, data: InputDict) -> InputDict:
        data["input_data"] = data["input_data"] + 1
        return data

    def post_process(self, model_outputs: OutputDict, batch) -> OutputDict:
        model_outputs["output_data"] = model_outputs["output_data"] + 10
        return model_outputs


class DummyProcessorCfg(ModelIOProcessorCfg[DummyProcessor]):
    class_type: ClassType_co[DummyProcessor] = DummyProcessor


class StatefulProcessor(ModelIOProcessor):
    cfg: "StatefulProcessorCfg"

    def __init__(self, cfg: "StatefulProcessorCfg"):
        super().__init__(cfg)
        self.counter = 0

    def pre_process(self, data: InputDict) -> InputDict:
        return data

    def post_process(self, model_outputs: OutputDict, batch) -> OutputDict:
        del batch
        return model_outputs


class StatefulProcessorCfg(ModelIOProcessorCfg[StatefulProcessor]):
    class_type: ClassType_co[StatefulProcessor] = StatefulProcessor


class AddProcessor(ModelIOProcessor):
    cfg: "AddProcessorCfg"

    def pre_process(self, data: InputDict) -> InputDict:
        data = data.copy()
        data["input_data"] = data["input_data"] + self.cfg.pre_add
        return data

    def post_process(self, model_outputs: OutputDict, batch) -> OutputDict:
        del batch
        model_outputs = model_outputs.copy()
        model_outputs["output_data"] = (
            model_outputs["output_data"] + self.cfg.post_add
        )
        return model_outputs


class AddProcessorCfg(ModelIOProcessorCfg[AddProcessor]):
    class_type: ClassType_co[AddProcessor] = AddProcessor
    pre_add: float = 0.0
    post_add: float = 0.0


class MultiplyProcessor(ModelIOProcessor):
    cfg: "MultiplyProcessorCfg"

    def pre_process(self, data: InputDict) -> InputDict:
        data = data.copy()
        data["input_data"] = data["input_data"] * self.cfg.pre_scale
        return data

    def post_process(self, model_outputs: OutputDict, batch) -> OutputDict:
        del batch
        model_outputs = model_outputs.copy()
        model_outputs["output_data"] = (
            model_outputs["output_data"] * self.cfg.post_scale
        )
        return model_outputs


class MultiplyProcessorCfg(ModelIOProcessorCfg[MultiplyProcessor]):
    class_type: ClassType_co[MultiplyProcessor] = MultiplyProcessor
    pre_scale: float = 1.0
    post_scale: float = 1.0


class EnvelopeContextProcessor(EnvelopeIOProcessor):
    cfg: "EnvelopeContextProcessorCfg"

    def __init__(self, cfg: "EnvelopeContextProcessorCfg"):
        super().__init__(cfg)
        self.post_calls: list[tuple[InputDict | None, object]] = []

    def pre_process(
        self,
        data: PipelineEnvelope[InputDict, object | None],
    ) -> PipelineEnvelope[InputDict, dict[str, torch.Tensor]]:
        model_input = copy.deepcopy(cast(InputDict, data.model_input))
        model_input["input_data"] = (
            model_input["input_data"] + self.cfg.pre_add
        )
        return PipelineEnvelope(
            model_input=model_input,
            processor_context={
                "offset": torch.tensor(self.cfg.context_offset),
                "envelope_input": model_input["input_data"].clone(),
            },
        )

    def post_process(
        self,
        model_outputs: OutputDict,
        *,
        model_input: InputDict | None = None,
        processor_context: (
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
        ) = None,
    ) -> OutputDict:
        self.post_calls.append(
            (
                copy.deepcopy(cast(InputDict | None, model_input)),
                copy.deepcopy(processor_context),
            )
        )
        model_outputs = model_outputs.copy()
        model_outputs["output_data"] = (
            model_outputs["output_data"] + self.cfg.post_add
        )
        if isinstance(processor_context, list):
            context_list = cast(
                list[dict[str, torch.Tensor]], processor_context
            )
            model_outputs["context_offset"] = torch.stack(
                [item["offset"] for item in context_list], dim=0
            )
            model_outputs["envelope_input"] = torch.stack(
                [item["envelope_input"] for item in context_list], dim=0
            )
            return model_outputs

        context = cast(dict[str, torch.Tensor], processor_context)
        model_outputs["context_offset"] = context["offset"]
        model_outputs["envelope_input"] = context["envelope_input"]
        return model_outputs


class EnvelopeContextProcessorCfg(
    EnvelopeIOProcessorCfg[EnvelopeContextProcessor]
):
    class_type: ClassType_co[EnvelopeContextProcessor] = (
        EnvelopeContextProcessor
    )
    pre_add: float = 0.0
    post_add: float = 0.0
    context_offset: float = 0.0


class ContextMutatingEnvelopeProcessor(EnvelopeIOProcessor):
    cfg: "ContextMutatingEnvelopeProcessorCfg"

    def __init__(self, cfg: "ContextMutatingEnvelopeProcessorCfg"):
        super().__init__(cfg)
        self.post_calls: list[tuple[InputDict | None, object]] = []
        self.raw_post_calls: list[tuple[InputDict | None, object]] = []

    def pre_process(
        self,
        data: PipelineEnvelope[InputDict, dict[str, list[int]] | None],
    ) -> PipelineEnvelope[InputDict, dict[str, list[int]]]:
        model_input = cast(InputDict, data.model_input)
        model_input["input_data"] = (
            model_input["input_data"] + self.cfg.pre_add
        )
        processor_context = cast(
            dict[str, list[int]] | None, data.processor_context
        )
        if processor_context is None:
            processor_context = {"history": [self.cfg.marker]}
        else:
            # Intentionally reuse the same context object so compose tests
            # cover the no-copy shared-reference contract.
            processor_context["history"].append(self.cfg.marker)
        return PipelineEnvelope(
            model_input=model_input,
            processor_context=processor_context,
        )

    def post_process(
        self,
        model_outputs: OutputDict,
        *,
        model_input: InputDict | None = None,
        processor_context: (
            dict[str, list[int]] | list[dict[str, list[int]]] | None
        ) = None,
    ) -> OutputDict:
        # Keep raw references so compose tests can assert the no-copy replay
        # contract directly.
        self.raw_post_calls.append(
            (cast(InputDict | None, model_input), processor_context)
        )
        self.post_calls.append(
            (
                copy.deepcopy(cast(InputDict | None, model_input)),
                copy.deepcopy(processor_context),
            )
        )
        return model_outputs


class ContextMutatingEnvelopeProcessorCfg(
    EnvelopeIOProcessorCfg[ContextMutatingEnvelopeProcessor]
):
    class_type: ClassType_co[ContextMutatingEnvelopeProcessor] = (
        ContextMutatingEnvelopeProcessor
    )
    pre_add: float = 0.0
    marker: int = 0


class ModelInputAwareProcessor(ModelIOProcessor):
    cfg: "ModelInputAwareProcessorCfg"

    def pre_process(self, data: InputDict) -> InputDict:
        data = data.copy()
        data["input_data"] = data["input_data"] + self.cfg.pre_add
        return data

    def post_process(
        self, model_outputs: OutputDict, batch: InputDict
    ) -> OutputDict:
        model_outputs = model_outputs.copy()
        model_outputs["output_data"] = (
            model_outputs["output_data"] + batch["input_data"]
        )
        return model_outputs


class ModelInputAwareProcessorCfg(
    ModelIOProcessorCfg[ModelInputAwareProcessor]
):
    class_type: ClassType_co[ModelInputAwareProcessor] = (
        ModelInputAwareProcessor
    )
    pre_add: float = 0.0


class MyTestPipeline(InferencePipeline[InputDict, OutputDict]):
    pass


class MyTestPipelineCfg(InferencePipelineCfg[MyTestPipeline]):
    class_type: ClassType_co[MyTestPipeline] = MyTestPipeline
    processor: DummyProcessorCfg = DummyProcessorCfg()


class HookedRuntimeInferencePipeline(InferencePipeline[InputDict, OutputDict]):
    forwarded_inputs: list[object]
    forwarded_processor_contexts: list[object]

    def __init__(
        self,
        cfg: "HookedRuntimeInferencePipelineCfg",
        model: TorchModelMixin | None = None,
    ):
        self.forwarded_inputs = []
        self.forwarded_processor_contexts = []
        super().__init__(cfg=cfg, model=model)

    def _model_forward_with_envelope(
        self,
        data: InputDict,
        *,
        processor_context: PostProcessContext[None] = None,
    ) -> OutputDict:
        self.forwarded_inputs.append(copy.deepcopy(data))
        self.forwarded_processor_contexts.append(
            copy.deepcopy(processor_context)
        )
        return super()._model_forward_with_envelope(
            data,
            processor_context=processor_context,
        )


class HookedRuntimeInferencePipelineCfg(
    InferencePipelineCfg[HookedRuntimeInferencePipeline]
):
    class_type: ClassType_co[HookedRuntimeInferencePipeline] = (
        HookedRuntimeInferencePipeline
    )
    processor: DummyProcessorCfg = DummyProcessorCfg()


class DirectMixinPipeline(InferencePipelineMixin[InputDict, OutputDict]):
    cfg: "DirectMixinPipelineCfg"

    def __call__(self, data: InputDict) -> OutputDict:
        return {"output_data": data["input_data"]}


class DirectMixinPipelineCfg(InferencePipelineCfg[DirectMixinPipeline]):
    class_type: ClassType_co[DirectMixinPipeline] = DirectMixinPipeline


@pytest.fixture(scope="module")
def deterministic_setup():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="function")
def test_pipeline_cfg() -> MyTestPipelineCfg:
    return MyTestPipelineCfg(model_cfg=DummyModelCfg())


@pytest.fixture(scope="function")
def test_pipeline(test_pipeline_cfg: MyTestPipelineCfg) -> MyTestPipeline:
    return MyTestPipeline(cfg=test_pipeline_cfg)


def test_pipeline_initialization(test_pipeline: MyTestPipeline):
    assert isinstance(test_pipeline, MyTestPipeline)
    assert isinstance(test_pipeline.model, DummyModel)
    assert isinstance(test_pipeline.cfg, MyTestPipelineCfg)
    assert isinstance(
        test_pipeline.envelope_processor, ModelIOProcessorEnvelopeAdapter
    )
    assert isinstance(test_pipeline.processor, DummyProcessor)


def test_direct_mixin_pipeline_reset_requires_concrete_implementation():
    pipeline = DirectMixinPipeline(
        cfg=DirectMixinPipelineCfg(model_cfg=DummyModelCfg())
    )

    with pytest.raises(NotImplementedError, match="must be implemented"):
        pipeline.reset(episode_id=1)


@pytest.mark.parametrize("with_collator", [True, False])
def test_pipeline_call_with_collator(with_collator: bool):
    test_pipeline = MyTestPipeline(
        cfg=MyTestPipelineCfg(
            model_cfg=DummyModelCfg(),
            batch_size=1,
            collate_fn=collate_batch_dict if with_collator else None,
        )
    )
    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()
    output = test_pipeline(raw_input)
    pre_processed_data = original_data + 1
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = model.linear(pre_processed_data) * 2
    expected_final_output = expected_model_output + 10

    assert "output_data" in output
    assert torch.allclose(output["output_data"], expected_final_output)
    if with_collator:
        expected_final_output = expected_final_output.unsqueeze(0)
        assert output["output_data"].shape == expected_final_output.shape


@pytest.mark.parametrize("with_collator", [True, False])
def test_runtime_pipeline_model_forward_with_envelope_hook_single(
    with_collator: bool,
):
    pipeline = HookedRuntimeInferencePipeline(
        cfg=HookedRuntimeInferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            batch_size=1,
            collate_fn=collate_batch_dict if with_collator else None,
        )
    )
    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    pipeline(raw_input)

    assert len(pipeline.forwarded_inputs) == 1
    assert len(pipeline.forwarded_processor_contexts) == 1
    forwarded_input = cast(
        dict[str, torch.Tensor], pipeline.forwarded_inputs[0]
    )
    assert set(forwarded_input.keys()) == set(raw_input.keys())
    expected_input = original_data + 1
    if with_collator:
        expected_input = expected_input.unsqueeze(0)
    assert torch.equal(forwarded_input["input_data"], expected_input)
    expected_processor_context = [None] if with_collator else None
    assert (
        pipeline.forwarded_processor_contexts[0] == expected_processor_context
    )


def test_runtime_pipeline_model_forward_with_envelope_hook_batch():
    pipeline = HookedRuntimeInferencePipeline(
        cfg=HookedRuntimeInferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            batch_size=2,
        )
    )
    raw_input = [{"input_data": torch.randn(1, 10)} for _ in range(3)]
    original_batches = [item["input_data"].clone() for item in raw_input]

    list(pipeline(raw_input))

    assert len(pipeline.forwarded_inputs) == 2
    assert pipeline.forwarded_processor_contexts == [[None, None], [None]]
    first_batch = cast(dict[str, torch.Tensor], pipeline.forwarded_inputs[0])
    second_batch = cast(dict[str, torch.Tensor], pipeline.forwarded_inputs[1])
    assert torch.equal(
        first_batch["input_data"],
        torch.stack([original_batches[0] + 1, original_batches[1] + 1], dim=0),
    )
    assert torch.equal(
        second_batch["input_data"],
        torch.stack([original_batches[2] + 1], dim=0),
    )


@pytest.mark.parametrize("with_collator", [True, False])
def test_runtime_warns_and_never_calls_unsupported_private_processor_hook_name(
    with_collator: bool,
):
    class UnsupportedPrivateHookPipeline(
        InferencePipeline[InputDict, OutputDict]
    ):
        forwarded_inputs: list[object]

        def __init__(
            self,
            cfg: "UnsupportedPrivateHookPipelineCfg",
            model: TorchModelMixin | None = None,
        ):
            self.forwarded_inputs = []
            super().__init__(cfg=cfg, model=model)

        def _model_forward_with_processor(self, data: object) -> OutputDict:
            self.forwarded_inputs.append(copy.deepcopy(data))
            raise AssertionError(
                "_model_forward_with_processor is unsupported and should "
                "never be called by runtime"
            )

    class UnsupportedPrivateHookPipelineCfg(
        InferencePipelineCfg[UnsupportedPrivateHookPipeline]
    ):
        class_type: ClassType_co[UnsupportedPrivateHookPipeline] = (
            UnsupportedPrivateHookPipeline
        )
        processor: DummyProcessorCfg = DummyProcessorCfg()

    with pytest.warns(UserWarning, match="unsupported private"):
        pipeline = UnsupportedPrivateHookPipeline(
            cfg=UnsupportedPrivateHookPipelineCfg(
                model_cfg=DummyModelCfg(),
                batch_size=1,
                collate_fn=collate_batch_dict if with_collator else None,
            )
        )
    assert not hasattr(InferencePipeline, "_model_forward_with_processor")

    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    output = cast(OutputDict, pipeline(raw_input))
    assert pipeline.forwarded_inputs == []
    expected_model_output = pipeline.model.linear(original_data + 1) * 2
    expected_final_output = expected_model_output + 10
    if with_collator:
        expected_final_output = expected_final_output.unsqueeze(0)
    assert "output_data" in output
    assert torch.allclose(output["output_data"], expected_final_output)


def test_runtime_warns_when_inheriting_private_processor_hook_name():
    class UnsupportedPrivateHookPipeline(
        InferencePipeline[InputDict, OutputDict]
    ):
        forwarded_inputs: list[object]

        def __init__(
            self,
            cfg: "UnsupportedPrivateHookPipelineCfg",
            model: TorchModelMixin | None = None,
        ):
            self.forwarded_inputs = []
            super().__init__(cfg=cfg, model=model)

        def _model_forward_with_processor(self, data: object) -> OutputDict:
            self.forwarded_inputs.append(copy.deepcopy(data))
            raise AssertionError(
                "_model_forward_with_processor is unsupported and should "
                "never be called by runtime"
            )

    class UnsupportedPrivateHookPipelineCfg(
        InferencePipelineCfg[UnsupportedPrivateHookPipeline]
    ):
        class_type: ClassType_co[UnsupportedPrivateHookPipeline] = (
            UnsupportedPrivateHookPipeline
        )
        processor: DummyProcessorCfg = DummyProcessorCfg()

    class InheritedUnsupportedPrivateHookPipeline(
        UnsupportedPrivateHookPipeline
    ):
        pass

    class InheritedUnsupportedPrivateHookPipelineCfg(
        InferencePipelineCfg[InheritedUnsupportedPrivateHookPipeline]
    ):
        class_type: ClassType_co[InheritedUnsupportedPrivateHookPipeline] = (
            InheritedUnsupportedPrivateHookPipeline
        )
        processor: DummyProcessorCfg = DummyProcessorCfg()

    with pytest.warns(UserWarning, match="unsupported private"):
        pipeline = InheritedUnsupportedPrivateHookPipeline(
            cfg=InheritedUnsupportedPrivateHookPipelineCfg(
                model_cfg=DummyModelCfg(),
            )
        )

    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    output = cast(OutputDict, pipeline(raw_input))

    assert pipeline.forwarded_inputs == []
    expected_model_output = pipeline.model.linear(original_data + 1) * 2
    expected_final_output = expected_model_output + 10
    assert "output_data" in output
    assert torch.allclose(output["output_data"], expected_final_output)


def test_runtime_does_not_warn_when_envelope_hook_is_overridden():
    class DualHookPipeline(InferencePipeline[InputDict, OutputDict]):
        forwarded_inputs: list[object]

        def __init__(
            self,
            cfg: "DualHookPipelineCfg",
            model: TorchModelMixin | None = None,
        ):
            self.forwarded_inputs = []
            super().__init__(cfg=cfg, model=model)

        def _model_forward_with_processor(self, data: object) -> OutputDict:
            self.forwarded_inputs.append(copy.deepcopy(data))
            raise AssertionError(
                "_model_forward_with_processor is unsupported and should "
                "never be called by runtime"
            )

        def _model_forward_with_envelope(
            self,
            data: InputDict,
            *,
            processor_context: PostProcessContext[None] = None,
        ) -> OutputDict:
            return super()._model_forward_with_envelope(
                data,
                processor_context=processor_context,
            )

    class DualHookPipelineCfg(InferencePipelineCfg[DualHookPipeline]):
        class_type: ClassType_co[DualHookPipeline] = DualHookPipeline
        processor: DummyProcessorCfg = DummyProcessorCfg()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pipeline = DualHookPipeline(
            cfg=DualHookPipelineCfg(
                model_cfg=DummyModelCfg(),
            )
        )

    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    output = cast(OutputDict, pipeline(raw_input))

    assert pipeline.forwarded_inputs == []
    expected_model_output = pipeline.model.linear(original_data + 1) * 2
    expected_final_output = expected_model_output + 10
    assert "output_data" in output
    assert torch.allclose(output["output_data"], expected_final_output)


def test_pipeline_processor_reassignment_updates_runtime():
    pipeline = InferencePipeline(
        cfg=InferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            processor=DummyProcessorCfg(),
        )
    )
    envelope_processor = EnvelopeContextProcessor(
        EnvelopeContextProcessorCfg(
            pre_add=1.0,
            post_add=10.0,
            context_offset=7.0,
        )
    )
    legacy_processor = DummyProcessor(DummyProcessorCfg())

    pipeline.processor = envelope_processor
    assert pipeline.processor is envelope_processor
    assert pipeline.envelope_processor is envelope_processor

    pipeline.processor = legacy_processor
    assert pipeline.processor is legacy_processor
    assert isinstance(
        pipeline.envelope_processor, ModelIOProcessorEnvelopeAdapter
    )

    pipeline.processor = None
    assert pipeline.processor is None
    assert pipeline.envelope_processor is None


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_pipeline_batched(batch_size):
    batch_size = batch_size
    test_pipeline = MyTestPipeline(
        cfg=MyTestPipelineCfg(model_cfg=DummyModelCfg(), batch_size=batch_size)
    )
    raw_input = [{"input_data": torch.randn(1, 10)} for _ in range(3)]
    batched_input = []
    for i in range(0, len(raw_input), batch_size):
        batch = torch.stack(
            [item["input_data"] for item in raw_input[i : i + batch_size]],
            dim=0,
        )
        batched_input.append(batch)
    output = list(test_pipeline(raw_input))
    pre_processed_data = [i + 1 for i in batched_input]
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = [
            model.linear(i) * 2 for i in pre_processed_data
        ]
    expected_final_output = [i + 10 for i in expected_model_output]

    for o, e_o in zip(output, expected_final_output, strict=True):
        assert "output_data" in o
        assert torch.allclose(o["output_data"], e_o)


def test_pipeline_call_dataset_like(test_pipeline: MyTestPipeline):
    raw_input = [{"input_data": torch.randn(1, 10)} for _ in range(3)]
    original_data = [i["input_data"].clone() for i in raw_input]
    output = list(test_pipeline(raw_input))
    pre_processed_data = [i + 1 for i in original_data]
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = [
            model.linear(i) * 2 for i in pre_processed_data
        ]
    expected_final_output = [i + 10 for i in expected_model_output]

    for o, e_o in zip(output, expected_final_output, strict=True):
        assert "output_data" in o
        assert torch.allclose(o["output_data"], e_o)


def test_processor_cfg_add_and_iadd():
    add_cfg = AddProcessorCfg(pre_add=1.0, post_add=10.0)
    scale_cfg = MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0)

    combined_cfg = add_cfg + scale_cfg
    assert isinstance(combined_cfg, ComposedIOProcessorCfg)
    assert len(combined_cfg.processors) == 2
    assert combined_cfg[0].class_type is AddProcessor
    assert combined_cfg[1].class_type is MultiplyProcessor

    extended_cfg = combined_cfg + AddProcessorCfg(pre_add=5.0, post_add=7.0)
    assert len(combined_cfg.processors) == 2
    assert len(extended_cfg.processors) == 3

    prepended_cfg = AddProcessorCfg(pre_add=8.0, post_add=9.0) + combined_cfg
    assert isinstance(prepended_cfg, ComposedIOProcessorCfg)
    assert len(prepended_cfg.processors) == 3
    assert cast(AddProcessorCfg, prepended_cfg[0]).pre_add == 8.0
    assert cast(AddProcessorCfg, prepended_cfg[1]).pre_add == 1.0
    assert cast(MultiplyProcessorCfg, prepended_cfg[2]).pre_scale == 2.0

    combined_cfg += AddProcessorCfg(pre_add=4.0, post_add=6.0)
    assert len(combined_cfg.processors) == 3
    assert isinstance(combined_cfg(), ComposedIOProcessor)


def test_processor_cfg_accepts_canonical_composed_cfg():
    cfg = AddProcessorCfg(
        pre_add=1.0,
        post_add=10.0,
    ) + MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0)
    canonical_cfg = ComposedIOProcessorCfg(
        processors=[AddProcessorCfg(pre_add=5.0, post_add=7.0)]
    )

    mixed_cfg = cfg + canonical_cfg

    assert isinstance(mixed_cfg, ComposedIOProcessorCfg)
    assert len(cfg.processors) == 2
    assert len(mixed_cfg.processors) == 3
    assert cast(AddProcessorCfg, mixed_cfg[2]).pre_add == 5.0


def test_deprecated_runtime_imports_are_compatible():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_inference_module = importlib.import_module(
            "robo_orchard_lab.inference"
        )
        legacy_processor_module = importlib.import_module(
            "robo_orchard_lab.inference.processor"
        )

    assert legacy_inference_module.ClassType_co is ClassType_co
    assert issubclass(
        legacy_inference_module.InferencePipeline,
        InferencePipeline,
    )
    assert issubclass(
        legacy_inference_module.InferencePipelineCfg, InferencePipelineCfg
    )
    assert issubclass(legacy_processor_module.ProcessorMixin, ModelIOProcessor)
    assert issubclass(
        legacy_processor_module.ComposeProcessor, ComposedIOProcessor
    )
    assert issubclass(
        legacy_processor_module.ComposeProcessorCfg,
        ComposedIOProcessorCfg,
    )
    with pytest.warns(DeprecationWarning):
        reloaded_legacy_processor_module = importlib.reload(
            legacy_processor_module
        )
        assert (
            reloaded_legacy_processor_module.ComposeProcessor
            is legacy_processor_module.ComposeProcessor
        )
        assert reloaded_legacy_processor_module.ClassType_co is ClassType_co
    assert not hasattr(legacy_processor_module, "ModelIOProcessor")
    assert not hasattr(legacy_processor_module, "IOProcessorMixin")
    assert not hasattr(legacy_processor_module, "ComposedIOProcessor")


def test_lazy_package_imports_are_stable_under_concurrent_access():
    import robo_orchard_lab

    top_level_barrier = threading.Barrier(8)

    def load_pipeline(_):
        top_level_barrier.wait()
        return robo_orchard_lab.pipeline

    with ThreadPoolExecutor(max_workers=8) as executor:
        pipeline_results = list(executor.map(load_pipeline, range(8)))

    assert all(module is pipeline_results[0] for module in pipeline_results)
    assert pipeline_results[0].__name__ == "robo_orchard_lab.pipeline"

    pipeline_package = pipeline_results[0]
    training_barrier = threading.Barrier(8)

    def load_training(_):
        training_barrier.wait()
        return pipeline_package.training

    with ThreadPoolExecutor(max_workers=8) as executor:
        training_results = list(executor.map(load_training, range(8)))

    assert all(module is training_results[0] for module in training_results)
    assert training_results[0].__name__ == "robo_orchard_lab.pipeline.training"


def test_canonical_imports_stay_quiet_under_strict_deprecation_filter():
    import robo_orchard_lab

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pipeline_package = robo_orchard_lab.pipeline
        training_package = pipeline_package.training
        inference_package = importlib.import_module(
            "robo_orchard_lab.pipeline.inference"
        )

    assert pipeline_package.__name__ == "robo_orchard_lab.pipeline"
    assert training_package.__name__ == "robo_orchard_lab.pipeline.training"
    assert inference_package.__name__ == "robo_orchard_lab.pipeline.inference"


@pytest.mark.parametrize(
    ("module_name", "expected_message"),
    [
        pytest.param(
            "robo_orchard_lab.inference",
            "`robo_orchard_lab.inference` is deprecated. Use "
            "`robo_orchard_lab.pipeline.inference` and "
            "`robo_orchard_lab.processing.io_processor` instead.",
            id="legacy-inference-package",
        ),
        pytest.param(
            "robo_orchard_lab.inference.processor",
            "`robo_orchard_lab.inference.processor` is deprecated. Use "
            "`robo_orchard_lab.processing.io_processor` instead.",
            id="legacy-inference-processor-package",
        ),
        pytest.param(
            "robo_orchard_lab.pipeline.batch_processor",
            "`robo_orchard_lab.pipeline.batch_processor` is deprecated. Use "
            "`robo_orchard_lab.processing.step_processor` instead.",
            id="legacy-batch-processor-package",
        ),
    ],
)
def test_deprecated_package_reload_emits_single_warning(
    module_name: str,
    expected_message: str,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(module_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        reloaded_module = importlib.reload(module)

    assert reloaded_module is module
    assert [str(item.message) for item in caught] == [expected_message]


def test_processor_add_and_iadd():
    add_processor = AddProcessor(AddProcessorCfg(pre_add=1.0, post_add=10.0))
    scale_processor = MultiplyProcessor(
        MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0)
    )

    combined = add_processor + scale_processor
    assert isinstance(combined, ComposedIOProcessor)
    assert len(combined.processors) == 2
    assert combined[0] is add_processor
    assert combined[1] is scale_processor

    prepended = (
        AddProcessor(AddProcessorCfg(pre_add=5.0, post_add=7.0)) + combined
    )
    assert isinstance(prepended, ComposedIOProcessor)
    assert len(prepended.processors) == 3
    assert cast(AddProcessor, prepended[0]).cfg.pre_add == 5.0
    assert prepended[1] is add_processor
    assert prepended[2] is scale_processor

    pre_processed = combined.pre_process({"input_data": torch.tensor([2.0])})
    assert torch.equal(pre_processed["input_data"], torch.tensor([6.0]))

    prepended_processed = prepended.pre_process(
        {"input_data": torch.tensor([2.0])}
    )
    assert torch.equal(prepended_processed["input_data"], torch.tensor([16.0]))

    post_processed = combined.post_process(
        {"output_data": torch.tensor([4.0])},
        model_input=None,
    )
    assert torch.equal(post_processed["output_data"], torch.tensor([22.0]))

    extended = combined + AddProcessor(
        AddProcessorCfg(pre_add=5.0, post_add=7.0)
    )
    assert len(combined.processors) == 2
    assert len(extended.processors) == 3

    combined += AddProcessor(AddProcessorCfg(pre_add=4.0, post_add=6.0))
    assert len(combined.processors) == 3

    pre_processed = combined.pre_process({"input_data": torch.tensor([2.0])})
    assert torch.equal(pre_processed["input_data"], torch.tensor([10.0]))


def test_processor_accepts_canonical_composed_processor():
    composed = AddProcessor(
        AddProcessorCfg(pre_add=1.0, post_add=10.0)
    ) + MultiplyProcessor(MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0))
    canonical_composed = ComposedIOProcessorCfg(
        processors=[AddProcessorCfg(pre_add=5.0, post_add=7.0)]
    )()

    mixed = composed + canonical_composed

    assert isinstance(mixed, ComposedIOProcessor)
    assert len(composed.processors) == 2
    assert len(mixed.processors) == 3
    assert isinstance(mixed[2], AddProcessor)


def test_pipeline_resolves_legacy_processor_into_envelope_runtime():
    pipeline = MyTestPipeline(cfg=MyTestPipelineCfg(model_cfg=DummyModelCfg()))

    assert isinstance(
        pipeline.envelope_processor, ModelIOProcessorEnvelopeAdapter
    )
    assert isinstance(pipeline.processor, DummyProcessor)


def test_pipeline_envelope_processor_preserves_context_with_collator():
    pipeline = InferencePipeline(
        cfg=InferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            processor=EnvelopeContextProcessorCfg(
                pre_add=1.0,
                post_add=10.0,
                context_offset=7.0,
            ),
            collate_fn=collate_batch_dict,
            batch_size=1,
        )
    )
    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    output = cast(OutputDict, pipeline(raw_input))

    assert isinstance(pipeline.envelope_processor, EnvelopeContextProcessor)
    assert pipeline.processor is pipeline.envelope_processor
    processor = cast(EnvelopeContextProcessor, pipeline.envelope_processor)
    assert len(processor.post_calls) == 1
    post_model_input, post_context = processor.post_calls[0]
    assert post_model_input is not None
    assert isinstance(post_context, list)
    assert len(post_context) == 1
    assert isinstance(post_context[0], dict)

    expected_input = (original_data + 1).unsqueeze(0)
    assert torch.equal(post_model_input["input_data"], expected_input)
    assert torch.equal(output["envelope_input"], expected_input)
    assert torch.equal(output["context_offset"], torch.tensor([7.0]))


def test_pipeline_batch_keeps_processor_context_per_sample():
    pipeline = InferencePipeline(
        cfg=InferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            processor=ContextMutatingEnvelopeProcessorCfg(
                pre_add=1.0,
                marker=7,
            ),
            collate_fn=collate_batch_dict,
            batch_size=2,
        )
    )
    raw_input = [
        {"input_data": torch.randn(1, 10)},
        {"input_data": torch.randn(1, 10)},
    ]

    list(pipeline(raw_input))

    processor = cast(
        ContextMutatingEnvelopeProcessor, pipeline.envelope_processor
    )
    _, post_context = processor.post_calls[0]
    assert isinstance(post_context, list)
    assert post_context == [
        {"history": [7]},
        {"history": [7]},
    ]


def test_io_processor_all_focuses_common_surface():
    exported_names = set(io_processor_module.__all__)

    assert {
        "EnvelopeIOProcessor",
        "EnvelopeIOProcessorCfg",
        "PipelineEnvelope",
        "ComposedEnvelopeIOProcessor",
        "ComposedEnvelopeIOProcessorCfg",
        "compose_envelope",
        "compose_envelope_cfg",
        "IdentityIOProcessor",
        "IdentityIOProcessorCfg",
    } <= exported_names
    assert {
        "ClassType_co",
        "ModelIOProcessor",
        "ModelIOProcessorCfg",
        "ComposedIOProcessor",
        "ComposedIOProcessorCfg",
        "ModelIOProcessorEnvelopeAdapter",
        "ModelIOProcessorEnvelopeAdapterCfg",
        "ModelIOProcessorType_co",
        "ModelIOProcessorCfgType_co",
        "EnvelopeIOProcessorType_co",
        "EnvelopeIOProcessorCfgType_co",
        "adapt_model_io_processor_to_envelope",
        "normalize_pipeline_envelope",
        "resolve_envelope_processor",
        "resolve_envelope_processor_cfg",
    }.isdisjoint(exported_names)


def test_io_processor_legacy_reexports_warn_but_keep_working():
    legacy_exports = {
        "ModelIOProcessor": ModelIOProcessor,
        "ModelIOProcessorCfg": ModelIOProcessorCfg,
        "ComposedIOProcessor": ComposedIOProcessor,
        "ComposedIOProcessorCfg": ComposedIOProcessorCfg,
    }

    for name, expected in legacy_exports.items():
        with pytest.deprecated_call(
            match="deprecated compatibility re-export"
        ):
            resolved = getattr(io_processor_module, name)
        assert resolved is expected

        namespace = {}
        with pytest.deprecated_call(
            match="deprecated compatibility re-export"
        ):
            exec(
                f"from robo_orchard_lab.processing.io_processor import {name}",
                {},
                namespace,
            )
        assert namespace[name] is expected


def test_legacy_compose_auto_upgrade_uses_single_adapter():
    composed = ModelInputAwareProcessor(
        ModelInputAwareProcessorCfg(pre_add=1.0)
    ) + ModelInputAwareProcessor(ModelInputAwareProcessorCfg(pre_add=2.0))

    resolved = resolve_envelope_processor(composed)
    original_context = {"marker": torch.tensor(3.0)}

    assert isinstance(resolved, ModelIOProcessorEnvelopeAdapter)
    assert resolved.legacy is composed
    envelope = resolved.pre_process(
        PipelineEnvelope(
            model_input={"input_data": torch.tensor([2.0])},
            processor_context=original_context,
        )
    )
    assert isinstance(envelope.processor_context, dict)
    assert envelope.processor_context is original_context
    output = resolved.post_process(
        {"output_data": torch.tensor([1.0])},
        model_input=envelope.model_input,
        processor_context=envelope.processor_context,
    )

    assert torch.equal(envelope.model_input["input_data"], torch.tensor([5.0]))
    assert torch.equal(envelope.processor_context["marker"], torch.tensor(3.0))
    assert torch.equal(output["output_data"], torch.tensor([11.0]))


def test_resolve_envelope_processor_cfg_wraps_legacy_compose_once():
    cfg = ModelInputAwareProcessorCfg(
        pre_add=1.0
    ) + ModelInputAwareProcessorCfg(pre_add=2.0)

    resolved_cfg = resolve_envelope_processor_cfg(cfg)

    assert isinstance(resolved_cfg, ModelIOProcessorEnvelopeAdapterCfg)
    assert resolved_cfg.legacy_processor == cfg


def test_model_io_processor_envelope_adapter_load_preserves_legacy_state(
    tmp_path,
):
    legacy = StatefulProcessor(StatefulProcessorCfg())
    legacy.counter = 7
    adapter = ModelIOProcessorEnvelopeAdapter.from_legacy(legacy)
    save_dir = tmp_path / "adapter_state"

    adapter.save(str(save_dir))
    restored = ModelIOProcessorEnvelopeAdapter.load(str(save_dir))

    assert isinstance(restored.legacy, StatefulProcessor)
    assert restored.legacy.counter == 7
    assert restored.legacy is not legacy


def test_composed_envelope_load_preserves_mixed_legacy_child(tmp_path):
    composed = compose_envelope(
        EnvelopeContextProcessorCfg(
            pre_add=2.0,
            post_add=1.0,
            context_offset=4.0,
        )(),
        AddProcessorCfg(pre_add=3.0, post_add=5.0)(),
    )
    save_dir = tmp_path / "composed_envelope_state"

    composed.save(str(save_dir))
    restored = ComposedEnvelopeIOProcessor.load(str(save_dir))

    assert isinstance(restored, ComposedEnvelopeIOProcessor)
    assert isinstance(restored.processors[0], EnvelopeContextProcessor)
    assert isinstance(restored.processors[1], ModelIOProcessorEnvelopeAdapter)
    assert isinstance(
        restored.cfg.processors[1],
        ModelIOProcessorEnvelopeAdapterCfg,
    )
    assert isinstance(
        restored.cfg.processors[1].legacy_processor,
        AddProcessorCfg,
    )
    assert restored.cfg.processors[1].legacy_processor.pre_add == 3.0
    assert restored.cfg.processors[1].legacy_processor.post_add == 5.0


def test_envelope_authoring_surface_accepts_legacy_inputs():
    envelope_cfg = EnvelopeContextProcessorCfg(pre_add=1.0, post_add=2.0)
    legacy_cfg = AddProcessorCfg(pre_add=3.0, post_add=4.0)
    runtime_processor = EnvelopeContextProcessor(envelope_cfg)
    legacy_runtime = AddProcessor(legacy_cfg)

    composed_runtime = compose_envelope(runtime_processor, legacy_runtime)
    composed_cfg = compose_envelope_cfg(envelope_cfg, legacy_cfg)
    chained_runtime = runtime_processor + legacy_runtime
    chained_cfg = envelope_cfg + legacy_cfg

    assert isinstance(composed_runtime, ComposedEnvelopeIOProcessor)
    assert len(composed_runtime.processors) == 2
    assert isinstance(composed_cfg, ComposedEnvelopeIOProcessorCfg)
    assert len(composed_cfg.processors) == 2
    assert isinstance(chained_runtime, ComposedEnvelopeIOProcessor)
    assert len(chained_runtime.processors) == 2
    assert isinstance(chained_cfg, ComposedEnvelopeIOProcessorCfg)
    assert len(chained_cfg.processors) == 2


def test_compose_envelope_ignores_none_processors():
    first = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=1.0, marker=1)
    )
    second = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=2.0, marker=2)
    )

    composed = compose_envelope(first, None, second)

    assert isinstance(composed, ComposedEnvelopeIOProcessor)
    assert len(composed.processors) == 2
    assert composed.processors[0] is first
    assert composed.processors[1] is second


def test_composed_envelope_post_process_degrades_gracefully_on_none_context():
    first = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=1.0, marker=1)
    )
    second = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=2.0, marker=2)
    )
    composed = compose_envelope(first, second)
    final_model_input = {"input_data": torch.tensor([4.0])}

    output = composed.post_process(
        {"output_data": torch.tensor([0.0])},
        model_input=final_model_input,
        processor_context=None,
    )

    assert torch.equal(output["output_data"], torch.tensor([0.0]))
    first_model_input, first_context = first.post_calls[0]
    second_model_input, second_context = second.post_calls[0]
    assert first_model_input is not None
    assert second_model_input is not None
    assert torch.equal(first_model_input["input_data"], torch.tensor([4.0]))
    assert torch.equal(second_model_input["input_data"], torch.tensor([4.0]))
    assert first_context is None
    assert second_context is None


def test_composed_envelope_uses_final_model_input_and_context_stack():
    first = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=1.0, marker=1)
    )
    second = ContextMutatingEnvelopeProcessor(
        ContextMutatingEnvelopeProcessorCfg(pre_add=2.0, marker=2)
    )
    composed = compose_envelope(first, second)

    envelope = composed.pre_process(
        PipelineEnvelope(
            model_input={"input_data": torch.tensor([1.0])},
        )
    )
    compose_context = envelope.processor_context

    assert isinstance(compose_context, ProcessorContextStack)
    assert len(compose_context.processor_context_stack) == 2
    assert (
        compose_context.processor_context_stack[0]
        is compose_context.processor_context_stack[1]
    )
    assert compose_context.processor_context_stack == [
        {"history": [1, 2]},
        {"history": [1, 2]},
    ]

    final_model_input = cast(InputDict, envelope.model_input)
    assert torch.equal(final_model_input["input_data"], torch.tensor([4.0]))

    composed.post_process(
        {"output_data": torch.tensor([0.0])},
        model_input=final_model_input,
        processor_context=compose_context,
    )

    first_model_input, first_context = first.post_calls[0]
    second_model_input, second_context = second.post_calls[0]
    first_raw_model_input, first_raw_context = first.raw_post_calls[0]
    second_raw_model_input, second_raw_context = second.raw_post_calls[0]
    assert first_model_input is not None
    assert second_model_input is not None
    assert torch.equal(first_model_input["input_data"], torch.tensor([4.0]))
    assert torch.equal(second_model_input["input_data"], torch.tensor([4.0]))
    assert first_context == {"history": [1, 2]}
    assert second_context == {"history": [1, 2]}
    assert first_raw_model_input is final_model_input
    assert second_raw_model_input is final_model_input
    assert first_raw_model_input is second_raw_model_input
    assert first_raw_context is compose_context.processor_context_stack[0]
    assert second_raw_context is compose_context.processor_context_stack[1]


def test_pipeline_composed_envelope_preserves_batched_context_with_collator():
    pipeline = InferencePipeline(
        cfg=InferencePipelineCfg(
            model_cfg=DummyModelCfg(),
            processor=compose_envelope_cfg(
                ContextMutatingEnvelopeProcessorCfg(pre_add=1.0, marker=1),
                ContextMutatingEnvelopeProcessorCfg(pre_add=2.0, marker=2),
            ),
            collate_fn=collate_batch_dict,
            batch_size=1,
        )
    )
    raw_input = {"input_data": torch.randn(1, 10)}
    original_data = raw_input["input_data"].clone()

    pipeline(raw_input)

    composed = cast(ComposedEnvelopeIOProcessor, pipeline.envelope_processor)
    first = cast(ContextMutatingEnvelopeProcessor, composed.processors[0])
    second = cast(ContextMutatingEnvelopeProcessor, composed.processors[1])
    expected_model_input = (original_data + 3).unsqueeze(0)
    expected_processor_context = [{"history": [1, 2]}]

    assert len(first.post_calls) == 1
    assert len(second.post_calls) == 1
    first_model_input, first_context = first.post_calls[0]
    second_model_input, second_context = second.post_calls[0]
    assert first_model_input is not None
    assert second_model_input is not None
    assert torch.allclose(
        first_model_input["input_data"],
        expected_model_input,
    )
    assert torch.allclose(
        second_model_input["input_data"],
        expected_model_input,
    )
    assert first_context == expected_processor_context
    assert second_context == expected_processor_context


def test_pipeline_save_and_load(test_pipeline: MyTestPipeline, tmp_path):
    save_dir = tmp_path / "saved_pipeline"
    test_pipeline.save_pipeline(str(save_dir))

    assert (save_dir / "inference.config.json").is_file()
    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "model.config.json").is_file()

    loaded_pipeline = InferencePipelineMixin.load_pipeline(str(save_dir))

    assert isinstance(loaded_pipeline, MyTestPipeline)
    assert isinstance(loaded_pipeline.model, DummyModel)
    assert torch.equal(
        test_pipeline.model.linear.weight, loaded_pipeline.model.linear.weight
    )

    assert loaded_pipeline.device == torch.device("cpu")
    assert test_pipeline.device == torch.device("cpu")

    test_input = {"input_data": torch.randn(1, 10)}
    original_output = test_pipeline(copy.deepcopy(test_input))
    loaded_output = loaded_pipeline(copy.deepcopy(test_input))

    assert torch.allclose(
        original_output["output_data"], loaded_output["output_data"]
    )


def test_save_raises_error_if_dir_not_empty(
    test_pipeline: MyTestPipeline, tmp_path
):
    save_dir = tmp_path / "not_empty_dir"
    save_dir.mkdir()
    (save_dir / "some_file.txt").touch()

    with pytest.raises(DirectoryNotEmptyError):
        test_pipeline.save_pipeline(str(save_dir), required_empty=True)

    try:
        test_pipeline.save_pipeline(str(save_dir), required_empty=False)
    except DirectoryNotEmptyError:
        pytest.fail("save() raised DirectoryNotEmptyError unexpectedly.")
