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
from __future__ import annotations
import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    NumpyTensor,
    TorchTensor,
    load_config_class,
)
from typing_extensions import Self

__all__ = [
    "State",
    "StateList",
    "StateSaveLoadMixin",
    "obj2state",
    "state2obj",
]


META_FILE_NAME = "meta.json"
CONFIG_NAME = "config.json"


class State(BaseModel):
    """The state dataclass when pickling a object.

    This class provides a way to save and load the state of an object
    in a structured way.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    class_type: ClassType[Any] | None = None
    """The class type of the state. This is used to identify which class
    the state belongs to when loading it back."""

    state: dict[str, Any]
    """The state of the object. It should be picklable."""

    config: ClassConfig | None
    """The configuration of the object if it has one."""

    parameters: dict[str, TorchTensor | NumpyTensor] | None = None
    """The parameters of the state. It should be picklable.

    Different from the state which including runtime information,
    the parameters are the static information such as NN parameters.

    The parameters should be a dictionary mapping parameter names
    to tensors. If the transform does not have any parameters,
    this can be None.

    """

    save_to_path: bool | None = None
    """If this is set to True, the state will be saved to the given path as
    a separate folder. Otherwise, the state will be a part of the parent
    object state.

    This is useful for transforms that need to save their state to structured
    files not just a single file.

    If None, this flag will inherit from the parent object.
    """

    def save(self, path: str, protocol: Literal["pickle"] = "pickle") -> None:
        """Save the state of the transform to a file.

        The structure of the saved files will be:
            - path/
                - meta.json
                - state.pkl (if self.save_to_path is not True)
                - parameters.pkl (if self.save_to_path is not True)
                - config.json
                - parameters/ (if self.save_to_path is True)
                    - <parameter_name>.pkl
                - state/ (if self.save_to_path is True)
                    - <state_name>.pkl (for each state if not
                        State nor StateList)
                    - <state_name> (for State or StateList type)

        Args:
            folder (str): The folder path to save the state.
            protocol (Literal["pickle"]): The protocol to use for saving.
        """
        assert protocol == "pickle", (
            f"Only pickle protocol is supported, but got {protocol}."
        )

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    state_class_type=State, class_type=self.class_type
                ).to_str(format="json")
            )

        if self.config is not None:
            with open(os.path.join(path, CONFIG_NAME), "w") as f:
                f.write(
                    self.config.to_str(format="json", include_config_type=True)
                )

        # save parameters
        if self.parameters is not None:
            if self.save_to_path is True:
                parameters_path = os.path.join(path, "parameters")
                if not os.path.exists(parameters_path):
                    os.makedirs(parameters_path)
                for name, tensor in self.parameters.items():
                    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
                        raise ValueError(
                            f"Parameter {name} is not a tensor or numpy array."
                        )
                    if isinstance(tensor, torch.Tensor):
                        torch.save(
                            tensor, os.path.join(parameters_path, f"{name}.pt")
                        )
                    else:
                        np.save(
                            os.path.join(parameters_path, f"{name}.npy"),
                            tensor,
                        )
            else:
                parameters_path = os.path.join(path, "parameters.pkl")
                with open(parameters_path, "wb") as f:
                    pickle.dump(self.parameters, f)

        state_copy = self.state.copy()
        # process if state item need to be saved to a separate folder
        state_folder = os.path.join(path, "state")
        if not os.path.exists(state_folder):
            os.makedirs(state_folder)

        # TODO: Refactor to be more generic
        for k in list(state_copy.keys()):
            v = state_copy[k]
            if isinstance(v, (State, StateList)):
                if v.save_to_path is True or (
                    v.save_to_path is None and self.save_to_path is True
                ):
                    v = v.model_copy()
                    v.save_to_path = True
                    v.save(
                        os.path.join(state_folder, f"{k}"), protocol=protocol
                    )
                    del state_copy[k]

        # save the rest of the state
        if self.save_to_path is True:
            for k, item in state_copy.items():
                if isinstance(item, (State, StateList)):
                    item.save(
                        os.path.join(state_folder, f"{k}"), protocol=protocol
                    )
                else:
                    item_path = os.path.join(state_folder, f"{k}.pkl")
                    with open(item_path, "wb") as f:
                        pickle.dump(item, f)
        else:
            # save the state as a single file
            state_path = os.path.join(path, "state.pkl")
            with open(state_path, "wb") as f:
                pickle.dump(state_copy, f)

    @classmethod
    def load(cls, path: str) -> State:
        """Load the state of the transform from a file."""
        # process parameters.
        # parameters can only be saved to a folder or a single file.
        if os.path.exists(os.path.join(path, "parameters")):
            # if parameters are saved to a separate folder
            parameters = {}
            for file in os.listdir(os.path.join(path, "parameters")):
                if file.endswith(".pt"):
                    name = file[:-3]
                    tensor = torch.load(
                        os.path.join(path, "parameters", file),
                        weights_only=True,
                    )
                elif file.endswith(".npy"):
                    name = file[:-4]
                    tensor = np.load(os.path.join(path, "parameters", file))
                else:
                    raise ValueError(f"Unknown parameter file format: {file}")
                parameters[name] = tensor
        else:
            # if parameters are saved to a single file
            parameters_path = os.path.join(path, "parameters.pkl")
            if not os.path.exists(parameters_path):
                parameters = None
            else:
                with open(parameters_path, "rb") as f:
                    parameters = pickle.load(f)

        # process config
        config = None
        if os.path.exists(os.path.join(path, CONFIG_NAME)):
            with open(os.path.join(path, CONFIG_NAME), "r") as f:
                config = load_config_class(f.read(), format="json")

        # process state
        state = {}
        if os.path.exists(os.path.join(path, "state.pkl")):
            with open(os.path.join(path, "state.pkl"), "rb") as f:
                state.update(pickle.load(f))

        state_folder = os.path.join(path, "state")
        if os.path.exists(state_folder):
            for file in os.listdir(state_folder):
                if file.endswith(".pkl"):
                    name = file[:-4]
                    with open(os.path.join(state_folder, file), "rb") as f:
                        state[name] = pickle.load(f)
                elif os.path.isdir(os.path.join(state_folder, file)):
                    name = file
                    state[name] = load_state_from_path(
                        os.path.join(state_folder, file)
                    )
                else:
                    raise ValueError(
                        f"Unknown state file format: {file}. "
                        "Only .pkl files and folders are supported."
                    )
        return State(
            state=state,
            config=config,  # type: ignore
            parameters=parameters,
        )


class StateList(list):
    """A dataclass to hold a list of TransformState objects."""

    save_to_path: bool | None = None
    """Whether to save the state of the transforms to a separate path."""

    def __init__(self, *args, save_to_path: bool | None = None) -> None:
        """Initialize the TransformStateList with a list of TransformState."""
        super().__init__(*args)
        self.save_to_path = save_to_path
        """The path to save the state of the transform as independent folder.
        If this is set, the state will be saved to this path. Otherwise,
        the state will be a part of the parent object state.
        """

    def copy(self) -> StateList:
        return StateList(
            [data for data in self],
            save_to_path=self.save_to_path,
        )

    def model_copy(self) -> StateList:
        return self.copy()

    def save(self, path: str, protocol: Literal["pickle"] = "pickle") -> None:
        """Save the state of the transforms to a file.

        The structure of the saved files will be:
            - path/
                - meta.json
                - all.pkl (if self.save_to_path is not True)
                - 0/ (for State or StateList type if their save_to_path is True)
                    - ...
                - 1.pkl (for other types if self.save_to_path is True)


        Args:
            path (str): The folder path to save the state.
            protocol (Literal["pickle"]): The protocol to use for saving.
                Defaults to "pickle".

        """  # noqa: E501
        assert protocol == "pickle", (
            f"Only pickle protocol is supported, but got {protocol}."
        )

        # convert list to dict
        dict_data = {i: item for i, item in enumerate(self)}

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    state_class_type=StateList, class_type=None
                ).to_str(format="json")
            )

        # TODO: Refactor to be more generic
        # save the State and StateList objects to separate folders
        # if their save_to_path is True or inherited from parent
        # to be True.
        for k in list(dict_data.keys()):
            v = dict_data[k]
            if isinstance(v, (State, StateList)):
                if v.save_to_path is True or (
                    v.save_to_path is None and self.save_to_path is True
                ):
                    d = v.model_copy()
                    d.save_to_path = True
                    d.save(os.path.join(path, f"{k}"), protocol=protocol)
                    # remove the state because it is saved.
                    del dict_data[k]

        # save the rest of the state
        if self.save_to_path is True:
            for i, item in enumerate(dict_data.values()):
                if isinstance(item, (State, StateList)):
                    item.save(os.path.join(path, f"{i}"), protocol=protocol)
                else:
                    item_path = os.path.join(path, f"{i}.pkl")
                    pickle.dump(item, open(item_path, "wb"))
        else:
            # save the state as a single file
            state_path = os.path.join(path, "all.pkl")
            with open(state_path, "wb") as f:
                pickle.dump(dict_data, f)

    @staticmethod
    def load(path: str) -> StateList:
        """Load the state of the transforms from a file.

        Args:
            path (str): The folder path to load the state from.
        """
        # load the state
        data_dict: dict[int, Any] = {}

        # find all pkl files in the path
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                if file != "all.pkl":
                    idx = int(file.split(".")[0])
                    with open(os.path.join(path, file), "rb") as f:
                        data_dict[idx] = pickle.load(f)
                else:
                    with open(os.path.join(path, file), "rb") as f:
                        data_dict.update(pickle.load(f))
            elif os.path.isdir(os.path.join(path, file)):
                idx = int(file)
                data_dict[idx] = load_state_from_path(os.path.join(path, file))
            elif file == META_FILE_NAME:
                continue
            else:
                raise ValueError(
                    f"Unknown file format in the state folder: {file}. "
                    "Only .pkl files and folders are supported."
                )

        # check that no missing keys
        key_list = sorted(list(data_dict.keys()))
        if key_list[-1] != len(key_list) - 1:
            raise ValueError(
                f"Missing keys in the state list: {key_list}. "
                "The keys should be continuous from 0 to n-1."
            )
        # convert to StateList
        return StateList(
            [data_dict[i] for i in range(len(data_dict))],
            save_to_path=None,
        )


def load_state_from_path(path: str) -> State | StateList:
    """Load the state from a given path."""
    # check if the meta.json file exists
    meta_path = os.path.join(path, META_FILE_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file {meta_path} does not exist.")
    # load the meta.json file
    with open(meta_path, "r") as f:
        meta = f.read()
    type_config = StateConfig.from_str(meta, format="json")
    assert isinstance(type_config, StateConfig), (
        f"Invalid type config: {type_config}"
    )
    return type_config.state_class_type.load(path)


class StateSaveLoadMixin(metaclass=ABCMeta):
    @abstractmethod
    def __getstate__(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state: State) -> None:
        raise NotImplementedError

    def save(self, path: str, protocol: Literal["pickle"] = "pickle") -> None:
        """Save the state of the object to a file."""
        state = self.__getstate__()
        state.save(path, protocol=protocol)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load the state of the object from a file."""
        state = State.load(path)
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj


class StateConfig(ClassConfig[State | StateList]):
    class_type: ClassType[Any] | None
    """The class type to generate the state from.

    For StateList, this should be None.
    """

    state_class_type: ClassType[State | StateList]
    """The class type of the state to be saved and loaded."""


def obj2state(obj: Any) -> Any:
    """Convert an object to a State if applicable.

    This method recursively converts objects to State if they are
    instances of StateSaveLoadMixin.

    Args:
        obj (Any): The object to convert.

    """
    if isinstance(obj, State):
        return obj
    elif isinstance(obj, StateList):
        return StateList(
            [obj2state(item) for item in obj],
            save_to_path=obj.save_to_path,
        )
    elif isinstance(obj, (list, tuple)):
        return list([obj2state(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: obj2state(v) for k, v in obj.items()}
    elif isinstance(obj, StateSaveLoadMixin):
        state = obj.__getstate__()
        return state
    else:
        return obj


def state2obj(obj: Any) -> Any:
    """Convert a State back to its original object."""
    if isinstance(obj, State):
        new_obj = obj.class_type.__new__(obj.class_type)  # type: ignore
        new_obj.__setstate__(obj)
        return new_obj

    elif isinstance(obj, StateList):
        return StateList(
            [state2obj(item) for item in obj],
            save_to_path=obj.save_to_path,
        )
    elif isinstance(obj, list):
        return [state2obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: state2obj(v) for k, v in obj.items()}
    else:
        return obj
