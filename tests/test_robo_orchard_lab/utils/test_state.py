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

import glob
import os
import random
import string
import tempfile

import numpy as np
import pytest
import torch
import transformers
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from robo_orchard_lab.utils.state import State, StateList, StateSaveLoadMixin


class TestSateAndStateList:
    @pytest.mark.parametrize("hierarchical_save", [True, False])
    def test_save_load_np_parameters(
        self, tmp_local_folder: str, hierarchical_save: bool
    ):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={"key": "value"},
            config=None,
            parameters={"key": np.array([1, 2, 3])},
            hierarchical_save=hierarchical_save,
        )
        protocol = "pickle"
        state.save(save_path, protocol=protocol)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        recovered_state = State.load(save_path, protocol=protocol)
        assert state.parameters is not None
        assert recovered_state.parameters is not None
        assert recovered_state.parameters.keys() == state.parameters.keys()
        for k in state.parameters:
            assert np.array_equal(
                recovered_state.parameters[k], state.parameters[k]
            ), f"Parameter {k} does not match after save/load."

        assert os.path.exists(
            os.path.join(save_path, "parameters.safetensors.np")
        )

    @pytest.mark.parametrize("hierarchical_save", [True, False])
    def test_save_load_tensor_parameters(
        self, tmp_local_folder: str, hierarchical_save: bool
    ):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={"key": "value"},
            config=None,
            parameters={"key": torch.asarray([1, 2, 3])},
            hierarchical_save=hierarchical_save,
        )
        protocol = "pickle"

        state.save(save_path, protocol=protocol)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        recovered_state = State.load(save_path, protocol=protocol)
        assert state.parameters is not None
        assert recovered_state.parameters is not None
        assert recovered_state.parameters.keys() == state.parameters.keys()
        for k in state.parameters:
            src = recovered_state.parameters[k]
            dst = state.parameters[k]
            assert isinstance(src, torch.Tensor), (
                f"Parameter {k} is not a torch.Tensor after save/load."
            )
            assert isinstance(dst, torch.Tensor), (
                f"Parameter {k} is not a torch.Tensor before save/load."
            )
            assert torch.equal(src, dst), (
                f"Parameter {k} does not match after save/load."
            )
        assert os.path.exists(
            os.path.join(save_path, "parameters.safetensors.pt")
        )

    @pytest.mark.parametrize("hierarchical_save", [True, False])
    def test_save_load_str(
        self, tmp_local_folder: str, hierarchical_save: bool
    ):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={"key": "value"},
            config=None,
            parameters=None,
            hierarchical_save=hierarchical_save,
        )
        protocol = "pickle"
        state.save(save_path, protocol=protocol)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.hierarchical_save in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path, protocol=protocol)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters

    @pytest.mark.parametrize("hierarchical_save", [True, False])
    def test_save_load_str_recursive(
        self, tmp_local_folder: str, hierarchical_save: bool
    ):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={
                "key": State(
                    state={"nested_key": "nested_value"},
                    config=None,
                    parameters=None,
                    hierarchical_save=None,
                )
            },
            config=None,
            parameters=None,
            hierarchical_save=hierarchical_save,
        )
        protocol = "pickle"
        state.save(save_path, protocol=protocol)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.hierarchical_save in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path, protocol=protocol)
        print("recovered_state: ", recovered_state)
        print("recovered_state.state: ", recovered_state.state)
        print("state.state: ", state.state)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters

    @pytest.mark.parametrize("hierarchical_save", [True, False])
    def test_save_load_state_list(
        self, tmp_local_folder: str, hierarchical_save: bool
    ):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={
                "key": StateList(
                    [
                        "value1",
                        State(
                            state={"nested_key": "nested_value"},
                            config=None,
                            parameters=None,
                            hierarchical_save=None,
                        ),
                    ]
                )
            },
            config=None,
            parameters=None,
            hierarchical_save=hierarchical_save,
        )
        protocol = "pickle"
        state.save(save_path, protocol=protocol)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.hierarchical_save in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path, protocol=protocol)
        print("recovered_state: ", recovered_state)
        print("recovered_state.state: ", recovered_state.state)
        print("state.state: ", state.state)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters


class TestStateSaveLoadMixin:
    def test_load_save_hf_processor(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        tmp_local_folder: str,
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/huggingface/hub/models--google--paligemma2-3b-pt-224/",
            "snapshots/96eeb174da13ca1a2b247e4d0867436296c36420/",
        )
        processor = AutoProcessor.from_pretrained(path)
        state_mixin = StateSaveLoadMixin()
        state_mixin.processor = processor

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_mixin.save(save_path)
            assert os.path.exists(
                os.path.join(
                    save_path, "state", "processor", "preprocessor_config.json"
                )
            )
            recovered_mixin = StateSaveLoadMixin.load(save_path)
            assert hasattr(recovered_mixin, "processor")
            assert type(recovered_mixin.processor) is type(processor)

    def test_load_save_hf_tokenizer(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        tmp_local_folder: str,
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/huggingface/hub/",
            "models--google--gemma-3-270m/snapshots/"
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        state_mixin = StateSaveLoadMixin()
        state_mixin.tokenizer = tokenizer

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_mixin.save(save_path)
            assert os.path.exists(
                os.path.join(save_path, "state", "tokenizer", "tokenizer.json")
            )
            recovered_mixin = StateSaveLoadMixin.load(save_path)
            assert hasattr(recovered_mixin, "tokenizer")
            assert type(recovered_mixin.tokenizer) is type(tokenizer)

    @pytest.mark.skipif(
        condition=Version(transformers.__version__) <= Version("4.49.0"),
        reason="gemma3 model is not compatible with transformers <= 4.49.0",
    )  # type: ignore
    def test_load_save_hf_models(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        tmp_local_folder: str,
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/huggingface/hub/",
            "models--google--gemma-3-270m/snapshots/"
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        )
        model = AutoModelForCausalLM.from_pretrained(path)
        state_mixin = StateSaveLoadMixin()
        state_mixin.model = model

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_mixin.save(save_path)
            assert os.path.exists(
                os.path.join(save_path, "state", "model", "model.safetensors")
            )
            recovered_mixin = StateSaveLoadMixin.load(save_path)
            assert hasattr(recovered_mixin, "model")
            assert type(recovered_mixin.model) is type(model)
