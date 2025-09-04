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

import numpy as np
import pytest
import torch

from robo_orchard_lab.utils.state import State, StateList


class TestSateAndStateList:
    @pytest.mark.parametrize("save_to_path", [True, False])
    def test_save_load_np_parameters(
        self, tmp_local_folder: str, save_to_path: bool
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
            save_to_path=save_to_path,
        )
        state.save(save_path)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        recovered_state = State.load(save_path)
        assert state.parameters is not None
        assert recovered_state.parameters is not None
        assert recovered_state.parameters.keys() == state.parameters.keys()
        for k in state.parameters:
            assert np.array_equal(
                recovered_state.parameters[k], state.parameters[k]
            ), f"Parameter {k} does not match after save/load."
        if state.save_to_path in [None, False]:
            assert os.path.exists(os.path.join(save_path, "parameters.pkl"))
        else:
            assert not os.path.exists(
                os.path.join(save_path, "parameters.pkl")
            )

    @pytest.mark.parametrize("save_to_path", [True, False])
    def test_save_load_tensor_parameters(
        self, tmp_local_folder: str, save_to_path: bool
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
            save_to_path=save_to_path,
        )
        state.save(save_path)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        recovered_state = State.load(save_path)
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
        if state.save_to_path in [None, False]:
            assert os.path.exists(os.path.join(save_path, "parameters.pkl"))
        else:
            assert not os.path.exists(
                os.path.join(save_path, "parameters.pkl")
            )

    @pytest.mark.parametrize("save_to_path", [True, False])
    def test_save_load_str(self, tmp_local_folder: str, save_to_path: bool):
        save_path = os.path.join(
            tmp_local_folder,
            "test_save_"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        state = State(
            state={"key": "value"},
            config=None,
            parameters=None,
            save_to_path=save_to_path,
        )
        state.save(save_path)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.save_to_path in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters

    @pytest.mark.parametrize("save_to_path", [True, False])
    def test_save_load_str_recursive(
        self, tmp_local_folder: str, save_to_path: bool
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
                    save_to_path=None,
                )
            },
            config=None,
            parameters=None,
            save_to_path=save_to_path,
        )
        state.save(save_path)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.save_to_path in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path)
        print("recovered_state: ", recovered_state)
        print("recovered_state.state: ", recovered_state.state)
        print("state.state: ", state.state)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters

    @pytest.mark.parametrize("save_to_path", [True, False])
    def test_save_load_state_list(
        self, tmp_local_folder: str, save_to_path: bool
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
                            save_to_path=None,
                        ),
                    ]
                )
            },
            config=None,
            parameters=None,
            save_to_path=save_to_path,
        )
        state.save(save_path)

        # print all files in the save path
        files = glob.glob(os.path.join(save_path, "**"), recursive=True)
        print("Files in save path:", files)
        # Check if the config file is created
        if state.config is not None:
            config_path = save_path + "/config.json"
            assert os.path.exists(config_path)
        # Check if the state file is created
        if state.save_to_path in [None, False]:
            state_path = save_path + "/state.pkl"
            assert os.path.exists(state_path)

        recovered_state = State.load(save_path)
        print("recovered_state: ", recovered_state)
        print("recovered_state.state: ", recovered_state.state)
        print("state.state: ", state.state)
        assert recovered_state.state == state.state
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters
