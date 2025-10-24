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
from typing import Dict

import pytest
import torch
import torch.nn as nn

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.inference import (
    ClassType_co,
    InferencePipeline,
    InferencePipelineCfg,
    InferencePipelineMixin,
)
from robo_orchard_lab.inference.processor import (
    ProcessorMixin,
    ProcessorMixinCfg,
)
from robo_orchard_lab.models.mixin import (
    ModelMixin,
    TorchModuleCfg,
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
    return MyTestPipelineCfg(model=DummyModelCfg())


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
            model=DummyModelCfg(),
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
        cfg=MyTestPipelineCfg(model=DummyModelCfg(), batch_size=batch_size)
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


def test_pipeline_save_and_load(test_pipeline: MyTestPipeline, tmp_path):
    """Tests the save and load functionality.

    This is a critical integration test.
    """
    save_dir = tmp_path / "saved_pipeline"

    # 1. Save the pipeline
    test_pipeline.save(str(save_dir))

    # 2. Check if files were created
    assert (save_dir / "inference.config.json").is_file()
    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "model.config.json").is_file()

    loaded_pipeline = InferencePipelineMixin.load(str(save_dir))

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
        test_pipeline.save(str(save_dir), required_empty=True)

    # Should not raise error if required_empty is False
    try:
        test_pipeline.save(str(save_dir), required_empty=False)
    except DirectoryNotEmptyError:
        pytest.fail("save() raised DirectoryNotEmptyError unexpectedly.")
