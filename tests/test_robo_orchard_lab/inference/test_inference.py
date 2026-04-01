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

import robo_orchard_lab.inference.processor as legacy_processor_module
from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.inference import (
    ClassType_co,
    InferencePipeline,
    InferencePipelineCfg,
    InferencePipelineMixin,
)
from robo_orchard_lab.inference.processor import (
    ClassType_co as LegacyProcessorClassType_co,
    ComposeProcessor,
    ComposeProcessorCfg,
    ProcessorMixin,
    ProcessorMixinCfg,
)
from robo_orchard_lab.models.mixin import (
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.pipeline.inference import (
    InferencePipeline as RuntimeInferencePipeline,
    InferencePipelineCfg as RuntimeInferencePipelineCfg,
)
from robo_orchard_lab.processing.io_processor import (
    ComposedIOProcessor,
    ComposedIOProcessorCfg,
    ModelIOProcessor,
)
from robo_orchard_lab.utils.path import DirectoryNotEmptyError

# ---- 1. Test Mocks and Dummy Implementations ----
# We need concrete implementations of the abstract classes to test them.

# Dummy Input/Output types for testing
InputDict = Dict[str, torch.Tensor]
OutputDict = Dict[str, torch.Tensor]


class DummyModel(ModelMixin):
    """A simple dummy model for testing purposes."""

    def __init__(self, cfg: "DummyModelCfg"):
        super().__init__(cfg)
        self.linear = nn.Linear(10, 5)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Simple transformation for verification
        return {"output_data": self.linear(batch["input_data"]) * 2}


class DummyModelCfg(TorchModuleCfg[DummyModel]):
    """Config for DummyModel."""

    # This associates the config with the model class
    class_type: ClassType_co[DummyModel] = DummyModel


class DummyProcessor(ProcessorMixin):
    """A simple dummy processor for testing."""

    def pre_process(self, data: InputDict) -> InputDict:
        # Add a value during pre-processing
        data["input_data"] = data["input_data"] + 1
        return data

    def post_process(self, model_outputs: OutputDict, batch) -> OutputDict:
        # Add a value during post-processing
        model_outputs["output_data"] = model_outputs["output_data"] + 10
        return model_outputs


class DummyProcessorCfg(ProcessorMixinCfg[DummyProcessor]):
    """Config for DummyProcessor."""

    class_type: ClassType_co[DummyProcessor] = DummyProcessor


class AddProcessor(ProcessorMixin):
    """A processor that adds constants in pre- and post-processing."""

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


class AddProcessorCfg(ProcessorMixinCfg[AddProcessor]):
    """Config for AddProcessor."""

    class_type: ClassType_co[AddProcessor] = AddProcessor
    pre_add: float = 0.0
    post_add: float = 0.0


class MultiplyProcessor(ProcessorMixin):
    """A processor that scales tensors in pre- and post-processing."""

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


class MultiplyProcessorCfg(ProcessorMixinCfg[MultiplyProcessor]):
    """Config for MultiplyProcessor."""

    class_type: ClassType_co[MultiplyProcessor] = MultiplyProcessor
    pre_scale: float = 1.0
    post_scale: float = 1.0


class MyTestPipeline(InferencePipeline[InputDict, OutputDict]):
    """A concrete pipeline class for testing."""

    pass


class MyTestPipelineCfg(InferencePipelineCfg[MyTestPipeline]):
    """Config for MyTestPipeline."""

    class_type: ClassType_co[MyTestPipeline] = MyTestPipeline
    processor: DummyProcessorCfg = DummyProcessorCfg()


# ---- 2. Pytest Fixtures ----
# Fixtures provide a fixed baseline upon which tests can reliably
# and repeatedly execute.


@pytest.fixture(scope="module")
def deterministic_setup():
    """Fixture to set random seed and enable deterministic algorithms for reproducibility.

    This runs once per module.
    """  # noqa: E501
    torch.manual_seed(42)
    # The following line is crucial for reproducible results on GPU.
    # It may impact performance, but it's essential for testing.
    torch.use_deterministic_algorithms(True)
    # If using cuDNN
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


# @pytest.fixture(scope="function")
# def dummy_model() -> DummyModel:
#     """Provides an instance of DummyModel."""
#     return DummyModel(cfg=DummyModelCfg())


@pytest.fixture(scope="function")
def test_pipeline_cfg() -> MyTestPipelineCfg:
    """Provides an instance of MyTestPipelineCfg."""
    return MyTestPipelineCfg(model_cfg=DummyModelCfg())


@pytest.fixture(scope="function")
def test_pipeline(test_pipeline_cfg: MyTestPipelineCfg) -> MyTestPipeline:
    """Provides a fully initialized MyTestPipeline instance."""
    return MyTestPipeline(cfg=test_pipeline_cfg)


# ---- 3. Test Cases ----


def test_pipeline_initialization(test_pipeline: MyTestPipeline):
    """Tests if the pipeline and its components are initialized correctly."""  # noqa: E501
    # assert test_pipeline.model is dummy_model
    assert isinstance(test_pipeline, MyTestPipeline)
    assert isinstance(test_pipeline.model, DummyModel)
    assert isinstance(test_pipeline.cfg, MyTestPipelineCfg)
    assert isinstance(test_pipeline.processor, DummyProcessor)


@pytest.mark.parametrize("with_collator", [True, False])
def test_pipeline_call_with_collator(with_collator: bool):
    test_pipeline = MyTestPipeline(
        cfg=MyTestPipelineCfg(
            model_cfg=DummyModelCfg(),
            batch_size=1,
            collate_fn=collate_batch_dict if with_collator else None,
        )
    )
    # 1. Create raw input data
    raw_input = {"input_data": torch.randn(1, 10)}

    # 2. Get a reference to the original data to check transformations
    original_data = raw_input["input_data"].clone()

    # 3. Perform the call
    output = test_pipeline(raw_input)

    # 4. Verify the steps
    # 4.1. pre_process: should add 1
    pre_processed_data = original_data + 1

    # 4.2. model forward: linear transform and multiply by 2
    # We need the model's weight to calculate the expected output
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = model.linear(pre_processed_data) * 2

    # 4.3. post_process: should add 10
    expected_final_output = expected_model_output + 10

    assert "output_data" in output
    assert torch.allclose(output["output_data"], expected_final_output)
    if with_collator:
        expected_final_output = expected_final_output.unsqueeze(0)
        assert output["output_data"].shape == expected_final_output.shape


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
    # 3. Perform the call
    output = list(test_pipeline(raw_input))

    # 4. Verify the steps
    # 4.1. pre_process: should add 1
    pre_processed_data = [i + 1 for i in batched_input]

    # 4.2. model forward: linear transform and multiply by 2
    # We need the model's weight to calculate the expected output
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = [
            model.linear(i) * 2 for i in pre_processed_data
        ]

    # 4.3. post_process: should add 10
    expected_final_output = [i + 10 for i in expected_model_output]

    for o, e_o in zip(output, expected_final_output, strict=True):
        assert "output_data" in o
        assert torch.allclose(o["output_data"], e_o)


def test_pipeline_call_dataset_like(test_pipeline: MyTestPipeline):
    # 1. Create raw input data
    raw_input = [{"input_data": torch.randn(1, 10)} for _ in range(3)]

    # 2. Get a reference to the original data to check transformations
    original_data = [i["input_data"].clone() for i in raw_input]

    # 3. Perform the call
    output = list(test_pipeline(raw_input))

    # 4. Verify the steps
    # 4.1. pre_process: should add 1
    pre_processed_data = [i + 1 for i in original_data]

    # 4.2. model forward: linear transform and multiply by 2
    # We need the model's weight to calculate the expected output
    model = test_pipeline.model
    with torch.no_grad():
        expected_model_output = [
            model.linear(i) * 2 for i in pre_processed_data
        ]

    # 4.3. post_process: should add 10
    expected_final_output = [i + 10 for i in expected_model_output]

    for o, e_o in zip(output, expected_final_output, strict=True):
        assert "output_data" in o
        assert torch.allclose(o["output_data"], e_o)


def test_processor_cfg_add_and_iadd():
    add_cfg = AddProcessorCfg(pre_add=1.0, post_add=10.0)
    scale_cfg = MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0)

    combined_cfg = add_cfg + scale_cfg
    assert isinstance(combined_cfg, ComposeProcessorCfg)
    assert len(combined_cfg.processors) == 2
    assert combined_cfg[0].class_type is AddProcessor
    assert combined_cfg[1].class_type is MultiplyProcessor

    extended_cfg = combined_cfg + AddProcessorCfg(pre_add=5.0, post_add=7.0)
    assert len(combined_cfg.processors) == 2
    assert len(extended_cfg.processors) == 3

    prepended_cfg = AddProcessorCfg(pre_add=8.0, post_add=9.0) + combined_cfg
    assert isinstance(prepended_cfg, ComposeProcessorCfg)
    assert len(prepended_cfg.processors) == 3
    assert cast(AddProcessorCfg, prepended_cfg[0]).pre_add == 8.0
    assert cast(AddProcessorCfg, prepended_cfg[1]).pre_add == 1.0
    assert cast(MultiplyProcessorCfg, prepended_cfg[2]).pre_scale == 2.0

    combined_cfg += AddProcessorCfg(pre_add=4.0, post_add=6.0)
    assert len(combined_cfg.processors) == 3
    assert isinstance(combined_cfg(), ComposeProcessor)


def test_legacy_processor_cfg_accepts_canonical_composed_cfg():
    legacy_cfg = AddProcessorCfg(
        pre_add=1.0,
        post_add=10.0,
    ) + MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0)
    canonical_cfg = ComposedIOProcessorCfg(
        processors=[AddProcessorCfg(pre_add=5.0, post_add=7.0)]
    )

    mixed_cfg = legacy_cfg + canonical_cfg

    assert isinstance(mixed_cfg, ComposeProcessorCfg)
    assert len(legacy_cfg.processors) == 2
    assert len(mixed_cfg.processors) == 3
    assert cast(AddProcessorCfg, mixed_cfg[2]).pre_add == 5.0


def test_runtime_imports_are_compatible():
    assert LegacyProcessorClassType_co is ClassType_co
    assert issubclass(InferencePipeline, RuntimeInferencePipeline)
    assert issubclass(InferencePipelineCfg, RuntimeInferencePipelineCfg)
    assert issubclass(ProcessorMixin, ModelIOProcessor)
    assert issubclass(ComposeProcessor, ComposedIOProcessor)
    assert issubclass(ComposeProcessorCfg, ComposedIOProcessorCfg)
    with pytest.warns(DeprecationWarning):
        reloaded_legacy_processor_module = importlib.reload(
            legacy_processor_module
        )
        assert (
            reloaded_legacy_processor_module.ComposeProcessor
            is ComposeProcessor
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
    assert isinstance(combined, ComposeProcessor)
    assert len(combined.processors) == 2
    assert combined[0] is add_processor
    assert combined[1] is scale_processor

    prepended = (
        AddProcessor(AddProcessorCfg(pre_add=5.0, post_add=7.0)) + combined
    )
    assert isinstance(prepended, ComposeProcessor)
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


def test_legacy_processor_accepts_canonical_composed_processor():
    legacy_composed = AddProcessor(
        AddProcessorCfg(pre_add=1.0, post_add=10.0)
    ) + MultiplyProcessor(MultiplyProcessorCfg(pre_scale=2.0, post_scale=3.0))
    canonical_composed = ComposedIOProcessorCfg(
        processors=[AddProcessorCfg(pre_add=5.0, post_add=7.0)]
    )()

    mixed = legacy_composed + canonical_composed

    assert isinstance(mixed, ComposeProcessor)
    assert len(legacy_composed.processors) == 2
    assert len(mixed.processors) == 3
    assert isinstance(mixed[2], AddProcessor)


def test_pipeline_save_and_load(test_pipeline: MyTestPipeline, tmp_path):
    """Tests the save and load functionality.

    This is a critical integration test.
    """
    save_dir = tmp_path / "saved_pipeline"

    # 1. Save the pipeline
    test_pipeline.save_pipeline(str(save_dir))

    # 2. Check if files were created
    assert (save_dir / "inference.config.json").is_file()
    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "model.config.json").is_file()

    loaded_pipeline = InferencePipelineMixin.load_pipeline(str(save_dir))

    # 4. Verify the loaded pipeline
    assert isinstance(loaded_pipeline, MyTestPipeline)
    assert isinstance(loaded_pipeline.model, DummyModel)
    assert torch.equal(
        test_pipeline.model.linear.weight, loaded_pipeline.model.linear.weight
    )

    assert loaded_pipeline.device == torch.device("cpu")
    assert test_pipeline.device == torch.device("cpu")

    # 5. Verify it produces the same output
    test_input = {"input_data": torch.randn(1, 10)}
    original_output = test_pipeline(copy.deepcopy(test_input))
    loaded_output = loaded_pipeline(copy.deepcopy(test_input))

    assert torch.allclose(
        original_output["output_data"], loaded_output["output_data"]
    )


def test_save_raises_error_if_dir_not_empty(
    test_pipeline: MyTestPipeline, tmp_path
):
    """Tests that save() raises DirectoryNotEmptyError if the directory is not empty and required_empty is True."""  # noqa: E501
    save_dir = tmp_path / "not_empty_dir"
    save_dir.mkdir()
    (save_dir / "some_file.txt").touch()  # Make the directory non-empty

    with pytest.raises(DirectoryNotEmptyError):
        test_pipeline.save_pipeline(str(save_dir), required_empty=True)

    # Should not raise error if required_empty is False
    try:
        test_pipeline.save_pipeline(str(save_dir), required_empty=False)
    except DirectoryNotEmptyError:
        pytest.fail("save() raised DirectoryNotEmptyError unexpectedly.")
