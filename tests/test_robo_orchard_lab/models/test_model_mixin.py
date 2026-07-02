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

import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.utils.path import DirectoryNotEmptyError
from robo_orchard_lab.utils.state import StateSaveLoadMixin


# 1. without shared tensor
class SimpleModel(ModelMixin):
    def __init__(self, cfg: "SimpleModelCfg"):
        super().__init__(cfg)
        self.layer1 = nn.Linear(cfg.input_size, 20)
        self.layer2 = nn.Linear(20, cfg.output_size)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


class SimpleModelCfg(TorchModuleCfg[SimpleModel]):
    class_type: ClassType_co[SimpleModel] = SimpleModel
    input_size: int = 10
    output_size: int = 5


# 2. with shared tensor
class TiedWeightModel(ModelMixin):
    def __init__(self, cfg: "TiedWeightModelCfg"):
        super().__init__(cfg)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.projection = nn.Linear(
            cfg.embedding_dim, cfg.vocab_size, bias=False
        )
        self.projection.weight = self.embedding.weight

    def forward(self, x):
        embedded = self.embedding(x)
        return self.projection(embedded)


class TiedWeightModelCfg(TorchModuleCfg[TiedWeightModel]):
    class_type: ClassType_co[TiedWeightModel] = TiedWeightModel
    vocab_size: int = 50
    embedding_dim: int = 10


# 3. Nested Tied-Weight Model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocab_size, bias=False)


class NestedTiedWeightModel(ModelMixin):
    def __init__(self, cfg: "NestedTiedWeightModelCfg"):
        super().__init__(cfg)
        self.encoder = Encoder(cfg.vocab_size, cfg.embedding_dim)
        self.decoder = Decoder(cfg.vocab_size, cfg.embedding_dim)

        # Key: Tie weights between nested submodules
        self.decoder.projection.weight = self.encoder.embedding.weight


class NestedTiedWeightModelCfg(TorchModuleCfg[NestedTiedWeightModel]):
    class_type: ClassType_co[NestedTiedWeightModel] = NestedTiedWeightModel
    vocab_size: int = 30
    embedding_dim: int = 8


class FakeAccelerator:
    def __init__(
        self,
        *,
        is_main_process: bool = True,
        is_local_main_process: bool = True,
        save_on_each_node: bool = False,
        create_model_artifact: bool = True,
    ) -> None:
        self.is_main_process = is_main_process
        self.is_local_main_process = is_local_main_process
        self.project_configuration = SimpleNamespace(
            save_on_each_node=save_on_each_node
        )
        self.create_model_artifact = create_model_artifact
        self._save_model_state_pre_hook = OrderedDict()
        self.save_model_calls = []

    def register_save_state_pre_hook(self, hook):
        handle = RemovableHandle(self._save_model_state_pre_hook)
        self._save_model_state_pre_hook[handle.id] = hook
        return handle

    def save_model(self, model, save_directory):
        self.save_model_calls.append((model, save_directory))
        if self.create_model_artifact:
            os.makedirs(save_directory, exist_ok=True)
            with open(
                os.path.join(save_directory, "model.safetensors"), "wb"
            ) as f:
                f.write(b"")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(dir=os.path.abspath("./")) as temp_dir:
        yield temp_dir


class TestModelMixin:
    def test_save_and_load_simple_model(self, temp_dir):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        model.eval()
        save_path = os.path.join(temp_dir, "simple_model")

        model.save_model(save_path)

        assert os.path.exists(os.path.join(save_path, "model.config.json"))
        assert os.path.exists(os.path.join(save_path, "model.safetensors"))

        loaded_model = ModelMixin.load_model(save_path, device="cpu")
        loaded_model.eval()

        assert isinstance(loaded_model, SimpleModel)
        assert model.cfg.model_dump() == loaded_model.cfg.model_dump()

        for key, param in model.state_dict().items():
            assert torch.equal(param, loaded_model.state_dict()[key])

    def test_save_and_load_state_api_simple_model(self, temp_dir):
        with tempfile.TemporaryDirectory(dir=temp_dir) as save_path:
            state = StateSaveLoadMixin()

            cfg = SimpleModelCfg()
            model = SimpleModel(cfg)
            model.eval()

            # model.save(save_path)
            state.model = model
            state.save(save_path)
            print("save_path: ", save_path)
            # import ipdb

            # ipdb.set_trace()
            assert os.path.exists(
                os.path.join(save_path, "state", "model", "model.config.json")
            )
            assert os.path.exists(
                os.path.join(save_path, "state", "model", "model.safetensors")
            )
            new_state = StateSaveLoadMixin.load(save_path)
            loaded_model = new_state.model

            # loaded_model = ModelMixin.load_model(save_path, device="cpu")
            # loaded_model.eval()

            assert isinstance(loaded_model, SimpleModel)
            assert model.cfg.model_dump() == loaded_model.cfg.model_dump()

            for key, param in model.state_dict().items():
                assert torch.equal(param, loaded_model.state_dict()[key])

    def test_save_and_load_tied_weight_model(self, temp_dir):
        cfg = TiedWeightModelCfg()
        model = TiedWeightModel(cfg)
        model.eval()

        assert model.embedding.weight is model.projection.weight

        save_path = os.path.join(temp_dir, "tied_model")

        model.save_model(save_path)

        loaded_model = ModelMixin.load_model(save_path, device="cpu")
        loaded_model.eval()

        assert isinstance(loaded_model, TiedWeightModel)

        assert loaded_model.embedding.weight is loaded_model.projection.weight

        for key, param in model.state_dict().items():
            assert torch.equal(param, loaded_model.state_dict()[key])

    def test_save_and_load_nested_tied_weight_model(self, temp_dir):
        """Tests saving and loading for a model where shared tensors are located within nested submodules."""  # noqa: E501
        # 1. Setup the model and verify the pre-condition
        cfg = NestedTiedWeightModelCfg()
        model = NestedTiedWeightModel(cfg)
        model.eval()

        # Pre-condition: Assert that weights are the same object before saving
        assert (
            model.encoder.embedding.weight is model.decoder.projection.weight
        ), "Weights must be the same object before saving."

        # 2. Save the model with the shared tensor handling enabled
        save_path = os.path.join(temp_dir, "nested_tied_model")
        model.save_model(save_path)

        # 4. Load the model using the standard method
        loaded_model = ModelMixin.load_model(
            save_path, device="cpu", strict=True
        )
        loaded_model.eval()

        # 5. Perform the required post-load verifications
        assert isinstance(loaded_model, NestedTiedWeightModel)

        # 5a. Numerical Consistency Test:
        # Verify that all corresponding weights are numerically identical.
        original_sd = model.state_dict()
        loaded_sd = loaded_model.state_dict()
        assert original_sd.keys() == loaded_sd.keys(), (
            "State dict keys do not match."
        )
        for key, param in original_sd.items():
            assert torch.equal(param, loaded_sd[key]), (
                f"Tensor for key '{key}' does not match numerically."
            )

        # 5b. 'is' Test (Memory Identity Test):
        # Verify that the weight-tying relationship (memory sharing) was perfectly restored.  # noqa: E501
        assert (
            loaded_model.encoder.embedding.weight
            is loaded_model.decoder.projection.weight
        ), "Weights are not the same object after loading."

    def test_load_config_only(self, temp_dir):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "config_only")
        model.save_model(save_path)

        initialized_model = ModelMixin.load_model(
            save_path, load_weights=False
        )

        assert isinstance(initialized_model, SimpleModel)
        assert model.cfg.model_dump() == initialized_model.cfg.model_dump()

        assert not torch.equal(
            model.state_dict()["layer1.weight"],
            initialized_model.state_dict()["layer1.weight"],
        )

    def test_directory_not_empty_error(self, temp_dir):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "not_empty")
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "dummy.txt"), "w") as f:
            f.write("I am not empty")

        with pytest.raises(DirectoryNotEmptyError):
            model.save_model(save_path)

    def test_load_from_nonexistent_directory(self, temp_dir):
        non_existent_path = os.path.join(temp_dir, "non_existent")

        with pytest.raises(FileNotFoundError):
            ModelMixin.load_model(non_existent_path)

    def test_accelerator_register_all_hooks_registers_local_main_node(self):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        model.accelerate_model_id = 0
        accelerator = FakeAccelerator(
            is_main_process=False,
            is_local_main_process=True,
            save_on_each_node=True,
        )

        handles = model.accelerator_register_all_hooks(accelerator)

        assert len(handles) == 1
        assert list(accelerator._save_model_state_pre_hook.values()) == [
            model.accelerator_save_state_pre_hook
        ]

    def test_accelerator_register_all_hooks_non_writer_skips_model_id(self):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        accelerator = FakeAccelerator(
            is_main_process=False,
            is_local_main_process=False,
            save_on_each_node=True,
        )

        handles = model.accelerator_register_all_hooks(accelerator)

        assert handles == []
        assert accelerator._save_model_state_pre_hook == {}

    def test_accelerator_register_all_hooks_deduplicates_active_handle(
        self, mocker
    ):
        mock_logger = mocker.patch(
            "robo_orchard_lab.models.torch_model.logger"
        )
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        model.accelerate_model_id = 0
        accelerator = FakeAccelerator()

        first_handles = model.accelerator_register_all_hooks(accelerator)
        second_handles = model.accelerator_register_all_hooks(accelerator)
        first_handles[0].remove()
        third_handles = model.accelerator_register_all_hooks(accelerator)

        assert len(first_handles) == 1
        assert second_handles == []
        assert len(third_handles) == 1
        assert len(accelerator._save_model_state_pre_hook) == 1
        mock_logger.warning.assert_called_once()

    def test_save_model_with_accelerator_non_writer_skips_config(
        self, temp_dir
    ):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "accelerate_non_writer")
        accelerator = FakeAccelerator(
            is_main_process=False,
            is_local_main_process=False,
        )

        model.save_model(save_path, accelerator=accelerator)

        assert len(accelerator.save_model_calls) == 1
        assert not os.path.exists(os.path.join(save_path, "model.config.json"))

    def test_save_model_with_accelerator_writer_writes_config_after_artifact(
        self, temp_dir
    ):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "accelerate_writer")
        accelerator = FakeAccelerator()

        model.save_model(save_path, accelerator=accelerator)

        assert len(accelerator.save_model_calls) == 1
        assert os.path.exists(os.path.join(save_path, "model.config.json"))

    def test_save_model_with_accelerator_skips_config_without_artifact(
        self, temp_dir
    ):
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "accelerate_no_artifact")
        accelerator = FakeAccelerator(create_model_artifact=False)

        model.save_model(save_path, accelerator=accelerator)

        assert len(accelerator.save_model_calls) == 1
        assert not os.path.exists(os.path.join(save_path, "model.config.json"))

    def test_save_model_with_accelerator_warnings_only_on_global_main(
        self, mocker, temp_dir
    ):
        mock_logger = mocker.patch(
            "robo_orchard_lab.models.torch_model.logger"
        )
        cfg = SimpleModelCfg()
        model = SimpleModel(cfg)
        save_path = os.path.join(temp_dir, "accelerate_non_writer_warning")
        accelerator = FakeAccelerator(
            is_main_process=False,
            is_local_main_process=False,
        )

        model.save_model(
            save_path,
            model_prefix="custom",
            allow_shared_tensor=False,
            accelerator=accelerator,
        )

        mock_logger.warning.assert_not_called()
