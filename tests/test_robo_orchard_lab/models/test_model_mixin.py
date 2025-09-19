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

import pytest
import torch
import torch.nn as nn

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.utils.path import DirectoryNotEmptyError


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


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_directory:
        yield temp_directory


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

        initialized_model = ModelMixin.load_model(save_path, load_weight=False)

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
