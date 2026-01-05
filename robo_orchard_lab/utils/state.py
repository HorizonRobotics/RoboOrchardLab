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

"""State saving and loading utilities.

The State API provides a structured way to save and load objects
in a hierarchical manner. Different from traditional pickling,
the State API allows objects to store their state in separate folders,
enabling flexible and organized serialization, such as only loading
specific parts of a complex object.

"""

from __future__ import annotations
import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Literal

import cloudpickle
import numpy as np
import safetensors
import torch
from pydantic import BaseModel, ConfigDict, Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
)
from safetensors import (
    numpy as safetensors_numpy,
    torch as safetensors_pytorch,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin as HuggingFaceProcessorMixin,
)
from typing_extensions import Self

from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    is_empty_directory,
)

__all__ = [
    "State",
    "StateList",
    "StateSaveLoadMixin",
    "CustomizedSaveLoadMixin",
    "load",
    "obj2state",
    "state2obj",
]


META_FILE_NAME = "meta.json"


_protocol2module = {
    "pickle": pickle,
    "cloudpickle": cloudpickle,
}


class CustomizedSaveLoadMixin(metaclass=ABCMeta):
    """A mixin class for customized saving and loading of an object.

    In such cases where the serialization logic is already defined by
    the user, this mixin allows users to adapt their custom save/load methods
    to be compatible with the State API.

    User should implement the `_save_impl` and `load` methods to define how to
    save and load the object.

    """

    @abstractmethod
    def _save_impl(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> dict | None: ...

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Save the state of the object to a file."""
        load_kwargs: dict | None = self._save_impl(path, protocol)
        if load_kwargs is None:
            load_kwargs = {}
        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    class_type=type(self),
                    load_kwargs=load_kwargs,
                ).to_str(format="json", indent=2)
            )

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Self:
        """Load the state of the object from a file."""
        ...


class StateSaveLoadMixin:
    """A mixin class for saving and loading the state of an object.

    User should implement the `_get_state` and `_set_state` methods
    to define how to get and set the state of the object, or override
    the `save` and `load` methods directly.

    """

    def _get_ignore_save_attributes(self) -> list[str]:
        return []

    def _get_state(self) -> State:
        # raise NotImplementedError
        # if has __getstate__ method, use it to get the state
        if hasattr(self, "__getstate__"):
            state_dict = self.__getstate__()  # type: ignore
        else:
            state_dict = self.__dict__.copy()
        for key in self._get_ignore_save_attributes():
            state_dict.pop(key, None)
        return State(
            state=state_dict,
            class_type=type(self),
            config=None,
            hierarchical_save=None,
        )

    def _set_state(self, state: State) -> None:
        # raise NotImplementedError
        state_dict = state.state
        if hasattr(self, "__setstate__"):
            self.__setstate__(state_dict)  # type: ignore
        else:
            self.__dict__.update(state_dict)

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Save the state of the object to a file using State API."""
        state = self._get_state()
        state.state = obj2state(state.state)
        if hierarchical_save is not None:
            state.hierarchical_save = hierarchical_save
        state.save(path, protocol=protocol)

    def load_state(self, path_or_state: str | State):
        """Load the state from a file or State using State API.

        Args:
            path_or_state (str | State): The file path to load the state from,
                or the State object to load the state from.
        """
        if isinstance(path_or_state, str):
            state = _load_state_from_path(path_or_state)
        else:
            state = path_or_state
        assert isinstance(state, State), (
            f"Loaded state is not of type State: {type(state)}"
        )
        state.state = state2obj(state.state)
        self._set_state(state)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load the state of the object from a file using State API.

        Different from `load_state`, this is a class method that creates
        a new instance of the class and loads the state into it.

        Args:
            path (str): The file path to load the state from.
        """
        obj = cls.__new__(cls)
        obj.load_state(path)
        return obj


HuggingFacePreTrainedObj = (
    PreTrainedModel | PreTrainedTokenizerBase | HuggingFaceProcessorMixin
)


class WrappedHuggingFaceObj(CustomizedSaveLoadMixin):
    unwrapped_obj: HuggingFacePreTrainedObj

    def __init__(self, unwrapped_obj: HuggingFacePreTrainedObj) -> None:
        if not self.is_huggingface_pretrained(unwrapped_obj):
            raise TypeError(
                f"Expected a HuggingFace PreTrainedModel, "
                f"PreTrainedTokenizerBase or ProcessorMixin, "
                f"but got {type(unwrapped_obj)}."
            )

        self.unwrapped_obj = unwrapped_obj

    def _save_impl(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ):
        self.unwrapped_obj.save_pretrained(path)

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Save the state of the object to a file."""
        load_kwargs: dict | None = self._save_impl(path, protocol)
        if load_kwargs is None:
            load_kwargs = {}
        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    class_type=type(self),
                    load_kwargs=load_kwargs,
                    state_class_type=type(self.unwrapped_obj),
                ).to_str(format="json", indent=2)
            )

    @classmethod
    def load(cls, path: str) -> Any:
        meta_path = os.path.join(path, META_FILE_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} does not exist.")
        # load the meta.json file
        with open(meta_path, "r") as f:
            meta = f.read()
        type_config = StateConfig.from_str(meta, format="json")
        assert type_config.state_class_type is not None
        assert type_config.class_type == cls, (
            f"Type config class type {type_config.class_type} "
            f"does not match {cls}."
        )
        origin_type: HuggingFacePreTrainedObj = type_config.state_class_type  # type: ignore # noqa

        return origin_type.from_pretrained(path)  # type: ignore

    @staticmethod
    def is_huggingface_pretrained(obj: Any) -> bool:
        return isinstance(
            obj,
            (
                PreTrainedModel,
                PreTrainedTokenizerBase,
                HuggingFaceProcessorMixin,
            ),
        )


def _save_state_dict(
    path: str,
    name: str,
    states: dict[str, Any],
    protocol: Literal["pickle", "cloudpickle"],
    hierarchical_save: bool | None,
) -> None:
    pickle_module = _protocol2module[protocol]
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # TODO: Refactor to be more generic
    for k in list(states.keys()):
        v = states[k]
        if isinstance(v, (State, StateList)):
            # special case to handle hierarchical_save
            # flag inheritance
            if v.hierarchical_save is True or (
                v.hierarchical_save is None and hierarchical_save is True
            ):
                v = v.model_copy()
                v.hierarchical_save = True
                v.save(os.path.join(folder, f"{k}"), protocol=protocol)
                del states[k]
        elif isinstance(v, CustomizedSaveLoadMixin):
            # special case to handle CustomizedSaveLoadMixin
            v.save(os.path.join(folder, f"{k}"), protocol=protocol)
            del states[k]

    # save the rest of the state
    if hierarchical_save is True:
        for k, item in states.items():
            if isinstance(item, (State, StateList)):
                item.save(os.path.join(folder, f"{k}"), protocol=protocol)
            else:
                item_path = os.path.join(folder, f"{k}.pkl")
                with open(item_path, "wb") as f:
                    pickle_module.dump(item, f)
    else:
        if len(states) == 0:
            return
        # save the state as a single file
        state_path = os.path.join(path, f"{name}.pkl")
        with open(state_path, "wb") as f:
            pickle_module.dump(states, f)


def _load_state_dict(
    path: str,
    name: str,
    protocol: Literal["pickle", "cloudpickle"],
) -> dict[str, Any]:
    state = {}
    protocol_module = _protocol2module[protocol]
    if os.path.exists(os.path.join(path, f"{name}.pkl")):
        with open(os.path.join(path, f"{name}.pkl"), "rb") as f:
            state.update(protocol_module.load(f))

    state_folder = os.path.join(path, name)
    if os.path.exists(state_folder):
        for file in os.listdir(state_folder):
            if file.endswith(".pkl"):
                n = file[:-4]
                with open(os.path.join(state_folder, file), "rb") as f:
                    state[n] = protocol_module.load(f)
            elif os.path.isdir(os.path.join(state_folder, file)):
                n = file
                state[n] = _load_state_from_path(
                    os.path.join(state_folder, file)
                )
            else:
                raise ValueError(
                    f"Unknown state file format: {file}. "
                    "Only .pkl files and folders are supported."
                )
    return state


class State(BaseModel):
    """The state dataclass for saving and loading object.

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

    config: ConfigInstanceOf[Config] | None
    """The configuration of the object if it has one."""

    parameters: dict[str, Any] | None = None
    """The parameters of the state. It should be picklable.

    Different from the state which including runtime information,
    the parameters are the static information such as NN parameters.

    The parameters should be a dictionary mapping parameter names
    to tensors. If the transform does not have any parameters,
    this can be None.

    """

    hierarchical_save: bool | None = None
    """If this is set to True, the state will be saved to the given path as
    a separate folder. Otherwise, the state will be a part of the parent
    object state.

    This is useful for transforms that need to save their state to structured
    files not just a single file.

    If None, this flag will inherit from the parent object.
    """

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> None:
        """Save the state of the transform to a file.

        The structure of the saved files will be:
            - path/
                - meta.json (with StateConfig information, containing
                  deserialization info like class type.)
                - state.pkl (if self.hierarchical_save is not True)
                - parameters.safetensors.xx
                - config.json
                - state/ (if self.hierarchical_save is True)
                    - <state_name>.pkl (for each state if not
                      State nor StateList)
                    - <state_name> (for State or StateList type)

        Args:
            folder (str): The folder path to save the state.
            protocol (Literal["pickle", , "cloudpickle"]): The protocol to use
                for saving.
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not is_empty_directory(path):
            raise DirectoryNotEmptyError(
                f"Path {path} already exists and is not empty."
            )

        state_config = StateConfig(
            class_type=State,
            load_kwargs={
                "protocol": protocol,
            },
            state_class_type=self.class_type,
            state_class_config=self.config,
        )
        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(state_config.to_str(format="json", indent=2))

        if self.parameters is not None:
            np_tensors = {}
            pt_tensors = {}
            for k, v in self.parameters.items():
                if isinstance(v, np.ndarray):
                    np_tensors[k] = v
                elif isinstance(v, torch.Tensor):
                    pt_tensors[k] = v
                else:
                    raise ValueError(
                        f"Unsupported tensor type for parameter {k}: {type(v)}"
                    )
            if len(np_tensors) > 0:
                safetensors_numpy.save_file(
                    np_tensors,
                    os.path.join(path, "parameters.safetensors.np"),
                )
            if len(pt_tensors) > 0:
                safetensors_pytorch.save_file(
                    pt_tensors,
                    os.path.join(path, "parameters.safetensors.pt"),
                )

        state_copy = self.state.copy()
        # process if state item need to be saved to a separate folder

        _save_state_dict(
            path=path,
            name="state",
            states=state_copy,
            protocol=protocol,
            hierarchical_save=self.hierarchical_save,
        )

    @classmethod
    def load(
        cls, path: str, protocol: Literal["pickle", "cloudpickle"]
    ) -> State:
        """Load the state of the transform from a file."""
        # process parameters.
        # parameters can only be saved to a folder or a single file.
        meta_path = os.path.join(path, META_FILE_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} does not exist.")
        with open(meta_path, "r") as f:
            meta = f.read()
        type_config = StateConfig.from_str(meta, format="json")
        # process parameters
        parameters = {}
        np_tensors_path = os.path.join(path, "parameters.safetensors.np")
        pt_tensors_path = os.path.join(path, "parameters.safetensors.pt")
        if os.path.exists(np_tensors_path):
            np_tensors = safetensors.safe_open(
                np_tensors_path, framework="numpy"
            )
            for key in np_tensors.keys():
                parameters[key] = np_tensors.get_tensor(key)
        if os.path.exists(pt_tensors_path):
            pt_tensors = safetensors.safe_open(
                pt_tensors_path, framework="pt", device="cpu"
            )
            for key in pt_tensors.keys():
                parameters[key] = pt_tensors.get_tensor(key)
        if len(parameters) == 0:
            parameters = None

        # process state

        state = _load_state_dict(path=path, name="state", protocol=protocol)

        return State(
            state=state,
            config=type_config.state_class_config,  # type: ignore
            parameters=parameters,
            class_type=type_config.state_class_type,
        )


class StateList(list):
    """A dataclass to hold a list of objects as states."""

    hierarchical_save: bool | None = None
    """Whether to save the state of the transforms to a separate path."""

    def __init__(self, *args, hierarchical_save: bool | None = None) -> None:
        """Initialize the TransformStateList with a list of TransformState."""
        super().__init__(*args)
        self.hierarchical_save = hierarchical_save
        """The path to save the state of the transform as independent folder.
        If this is set, the state will be saved to this path. Otherwise,
        the state will be a part of the parent object state.
        """

    def copy(self) -> StateList:
        return StateList(
            [data for data in self],
            hierarchical_save=self.hierarchical_save,
        )

    def model_copy(self) -> StateList:
        return self.copy()

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> None:
        """Save the state of the transforms to a file.

        The structure of the saved files will be:
            - path/
                - meta.json
                - all.pkl (if self.hierarchical_save is not True)
                - all/ (if self.hierarchical_save is True)
                    - 0/ (for State or StateList type if their hierarchical_save is True)
                        - ...
                    - 1.pkl (for other types, if hierarchical_save is True)

        Args:
            path (str): The folder path to save the state.
            protocol (Literal["pickle"]): The protocol to use for saving.
                Defaults to "pickle".

        """  # noqa: E501

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not is_empty_directory(path):
            raise DirectoryNotEmptyError(
                f"Path {path} already exists and is not empty."
            )

        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    class_type=StateList,
                    load_kwargs={
                        "protocol": protocol,
                    },
                ).to_str(format="json", indent=2)
            )
        # convert list to dict
        dict_data = {str(i): item for i, item in enumerate(self)}

        _save_state_dict(
            path=path,
            name="all",
            states=dict_data,
            protocol=protocol,
            hierarchical_save=self.hierarchical_save,
        )

    @staticmethod
    def load(
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> StateList:
        """Load the state of the transforms from a file.

        Args:
            path (str): The folder path to load the state from.
        """
        data_dict = _load_state_dict(
            path=path,
            name="all",
            protocol=protocol,
        )
        # map keys to int
        data_dict = {int(k): v for k, v in data_dict.items()}
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
            hierarchical_save=None,
        )


def _load_state_from_path(
    path: str,
) -> State | StateList | StateSaveLoadMixin | CustomizedSaveLoadMixin:
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
    return type_config.class_type.load(path, **type_config.load_kwargs)


class StateConfig(
    ClassConfig[
        State | StateList | StateSaveLoadMixin | CustomizedSaveLoadMixin
    ]
):
    class_type: ClassType[
        State | StateList | StateSaveLoadMixin | CustomizedSaveLoadMixin
    ]
    """The class type of the state to be saved and loaded."""

    load_kwargs: dict[str, Any] = Field(
        default_factory=dict,
    )
    """The keyword arguments for loading the state."""

    state_class_type: ClassType[Any] | None = None
    """The class type of the state object belonging to.

    It is the class type that generate the state object.
    """
    state_class_config: ConfigInstanceOf[Config] | None = None


def load(path: str) -> Any:
    """Load instance from a given path."""
    state = _load_state_from_path(path)
    if isinstance(state, (State, StateList)):
        return state2obj(state)
    else:
        return state


def obj2state(obj: Any) -> Any:
    """Convert an object to a State if applicable.

    This method recursively converts objects to State if they are
    instances of StateSaveLoadMixin.

    Args:
        obj (Any): The object to convert.

    """
    if isinstance(obj, State):
        obj.state = obj2state(obj.state)
        return obj
    elif isinstance(obj, StateList):
        return StateList(
            [obj2state(item) for item in obj],
            hierarchical_save=obj.hierarchical_save,
        )
    elif isinstance(obj, (list, tuple)):
        return StateList([obj2state(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: obj2state(v) for k, v in obj.items()}
    elif isinstance(obj, StateSaveLoadMixin):
        state = obj._get_state()
        return obj2state(state)
    elif WrappedHuggingFaceObj.is_huggingface_pretrained(obj):
        wrapped_obj = WrappedHuggingFaceObj(unwrapped_obj=obj)
        return wrapped_obj
    else:
        return obj


def state2obj(obj: Any) -> Any:
    """Convert a State back to its original object."""
    if isinstance(obj, State):
        new_obj = obj.class_type.__new__(obj.class_type)  # type: ignore
        assert isinstance(new_obj, StateSaveLoadMixin), (
            f"Class type {obj.class_type} is not a subclass of "
            "StateSaveLoadMixin"
        )
        new_obj._set_state(obj)
        return new_obj
    elif isinstance(obj, (StateList, list)):
        return [state2obj(item) for item in obj]
    elif isinstance(obj, tuple):
        return (state2obj(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: state2obj(v) for k, v in obj.items()}
    elif isinstance(obj, WrappedHuggingFaceObj):
        return obj.unwrapped_obj
    else:
        return obj
