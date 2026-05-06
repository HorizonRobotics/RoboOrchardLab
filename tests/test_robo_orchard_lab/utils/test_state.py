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
import json
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

import robo_orchard_lab.utils.state as state_api
from robo_orchard_lab.utils.state import (
    ConstructableStateApplyMode,
    CustomizedSaveLoadMixin,
    State,
    StateList,
    StateMaterializeMixin,
    StatePersistenceMixin,
    StateRuntimeMixin,
    StateRuntimeProtocol,
    StateSaveLoadMixin,
    StateSequence,
    load,
    obj2state,
    state2obj,
    validate_recovery_state,
)
from robo_orchard_lab.utils.state.conversion import (
    decode_state_payload_for_apply,
)
from robo_orchard_lab.utils.state.mixin import StateMaterializeProtocol
from robo_orchard_lab.utils.state.save_profile import (
    register_save_profile,
    resolve_save_profile,
)
from robo_orchard_lab.utils.state.save_profile.graph import (
    GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION,
    GRAPH_PROFILE,
    GraphEntryKind,
    GraphLoadScope,
    GraphNodeType,
    GraphStateUnsupportedFormatVersionError,
    GraphStorageKind,
    NonSelfContainedEntryError,
    _read_graph_entry,
    _read_graph_manifest,
)


class _NestedStateMixin(StateSaveLoadMixin):
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _ParentStateMixin(StateSaveLoadMixin):
    def __init__(self, value: int = 0, child_value: int = 0) -> None:
        self.value = value
        self.child = _NestedStateMixin(child_value)


class _ParentStateMixinPreserveConstructableChild(StateSaveLoadMixin):
    def __init__(self, value: int = 0, child_value: int = 0) -> None:
        self.value = value
        self.child = _NestedStateMixin(child_value)

    def _set_state(self, state: State) -> None:
        payload = state.state
        if not isinstance(payload, dict):
            raise TypeError(
                "_ParentStateMixinPreserveConstructableChild expects a dict "
                f"payload. Got {type(payload).__name__}."
            )
        child_state = payload.get("child")
        if not isinstance(child_state, State):
            raise TypeError(
                "_ParentStateMixinPreserveConstructableChild expects child "
                "to stay a State payload during apply."
            )
        self.value = payload["value"]
        self.child.load_state(child_state)


class _LegacyDictStateMixin(StateSaveLoadMixin):
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def _get_state(self) -> dict[str, int]:
        return {"value": self.value}


class _DefaultDictBackedStateMixin(dict, StateSaveLoadMixin):
    pass


class _HookedDictBackedStateMixin(dict, StateSaveLoadMixin):
    def __getstate__(self) -> dict[str, dict[str, object]]:
        return {"items": dict(self)}

    def __setstate__(self, state: dict[str, dict[str, object]]) -> None:
        self.clear()
        self.update(state["items"])


class _NeverLoadSaveProfile:
    def __init__(self, name: str) -> None:
        self.name = name
        self.load_priority = -1000

    def save(self, state, *, path: str, protocol: str) -> None:
        raise AssertionError("test profile should not save")

    def load(self, path: str, *, protocol: str):
        raise AssertionError("test profile should not load")

    def has_manifest(self, path: str) -> bool:
        return False

    def has_artifact(self, path: str) -> bool:
        return False

    def load_path(self, path: str):
        raise AssertionError("test profile should not load")


class _SequenceStateMixin(StateSaveLoadMixin):
    def __init__(self) -> None:
        self.items = [1, {"nested": 2}]
        self.pair = ("left", "right")


class _SequenceWithStateMixin(StateSaveLoadMixin):
    def __init__(self) -> None:
        self.items = [_NestedStateMixin(4)]


class _TwoPhaseMaterializeState:
    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def allocate_state_instance(cls) -> "_TwoPhaseMaterializeState":
        obj = cls.__new__(cls)
        obj.value = -1
        return obj

    def apply_decoded_state(self, state: State) -> None:
        payload = state.state
        if not isinstance(payload, dict):
            raise TypeError(
                "_TwoPhaseMaterializeState expects a dict payload. "
                f"Got {type(payload).__name__}."
            )
        self.value = int(payload["value"])


class _CustomLeafState(CustomizedSaveLoadMixin):
    def __init__(self, value: int) -> None:
        self.value = value

    def _save_impl(
        self, path: str, protocol: str = "cloudpickle"
    ) -> dict | None:
        del protocol
        with open(os.path.join(path, "value.json"), "w") as f:
            json.dump({"value": self.value}, f)
        return None

    @classmethod
    def load(cls, path: str) -> "_CustomLeafState":
        with open(os.path.join(path, "value.json")) as f:
            payload = json.load(f)
        return cls(payload["value"])


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
        recovered_child = recovered_state.state["key"]
        expected_child = state.state["key"]
        assert isinstance(recovered_child, State)
        assert isinstance(expected_child, State)
        assert recovered_child.state == expected_child.state
        assert recovered_child.config == expected_child.config
        assert recovered_child.parameters == expected_child.parameters
        if hierarchical_save:
            assert recovered_child.save_profile == "tree"
        else:
            assert recovered_child.save_profile is None
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
        recovered_list = recovered_state.state["key"]
        expected_list = state.state["key"]
        assert isinstance(recovered_list, StateList)
        assert isinstance(expected_list, StateList)
        assert recovered_list[0] == expected_list[0]
        assert isinstance(recovered_list[1], State)
        assert isinstance(expected_list[1], State)
        assert recovered_list[1].state == expected_list[1].state
        if hierarchical_save:
            assert recovered_list.save_profile == "tree"
            assert recovered_list[1].save_profile == "tree"
        else:
            assert recovered_list.save_profile is None
            assert recovered_list[1].save_profile is None
        assert recovered_state.config == state.config
        assert recovered_state.parameters == state.parameters

    def test_legacy_state_list_loads_empty_list(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            StateList([]).save(save_path)
            recovered_state_list = StateList.load(save_path)

        assert recovered_state_list == []
        assert isinstance(recovered_state_list, StateList)


class TestStateSaveLoadMixin:
    def test_state_defaults_optional_metadata(self) -> None:
        state = State(state={})

        assert state.class_type is None
        assert state.config is None
        assert state.parameters is None

    def test_state_package_exports_only_public_state_api(self) -> None:
        assert set(state_api.__all__) == {
            "State",
            "ConstructableStateApplyMode",
            "StateRuntimeProtocol",
            "StateRuntimeMixin",
            "StatePersistenceMixin",
            "StateMaterializeMixin",
            "StateSaveLoadMixin",
            "CustomizedSaveLoadMixin",
            "load",
            "obj2state",
            "state2obj",
            "validate_recovery_state",
        }
        assert state_api.StateSequence is StateSequence
        assert state_api.StateList is StateList

    def test_state_mixin_layers_are_exported(self) -> None:
        assert issubclass(StateSaveLoadMixin, StateRuntimeMixin)
        assert issubclass(StateSaveLoadMixin, StatePersistenceMixin)
        assert issubclass(StateSaveLoadMixin, StateMaterializeMixin)
        assert isinstance(_ParentStateMixin(), StateRuntimeProtocol)

    def test_load_state_from_path_applies_persisted_state(
        self,
        tmp_local_folder: str,
    ) -> None:
        source = _ParentStateMixin(value=3, child_value=7)
        target = _ParentStateMixin(value=0, child_value=0)

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            source.save(save_path)
            target.load_state_from_path(save_path)

        assert target.value == 3
        assert target.child.value == 7

    def test_save_profile_registry_accepts_new_profile(self) -> None:
        profile = _NeverLoadSaveProfile(
            "unit_test_profile_registry_accepts_new_profile"
        )

        assert register_save_profile(profile) is profile
        assert resolve_save_profile(profile.name) is profile

    def test_save_profile_none_uses_tree_as_root_default(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            State(state={"value": 1}, save_profile=None).save(save_path)

            assert not os.path.exists(
                os.path.join(save_path, "graph_manifest.json")
            )
            recovered = State.load(save_path, protocol="cloudpickle")

        assert resolve_save_profile(None).name == "tree"
        assert recovered.save_profile == "tree"

    def test_nested_save_profile_can_override_tree_parent(
        self,
        tmp_local_folder: str,
    ) -> None:
        child = State(
            state={"value": 1},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )
        parent = State(
            state={"child": child},
            hierarchical_save=True,
            save_profile="tree",
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            parent.save(save_path)

            assert os.path.exists(
                os.path.join(
                    save_path,
                    "state",
                    "child",
                    "graph_manifest.json",
                )
            )

    def test_nested_save_profile_none_inherits_tree_parent(
        self,
        tmp_local_folder: str,
    ) -> None:
        child = State(
            state={"value": 1},
            hierarchical_save=True,
            save_profile=None,
        )
        parent = State(
            state={"child": child},
            hierarchical_save=True,
            save_profile="tree",
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            parent.save(save_path)

            assert not os.path.exists(
                os.path.join(
                    save_path,
                    "state",
                    "child",
                    "graph_manifest.json",
                )
            )

    def test_tree_profile_load_path_dispatches_via_meta_json(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state = State(state={"value": 1})
            state.save(save_path, protocol="pickle")

            recovered = resolve_save_profile("tree").load_path(save_path)

        assert isinstance(recovered, State)
        assert recovered.state == {"value": 1}
        assert recovered.save_profile == "tree"

    def test_validate_recovery_state_public_helper(self) -> None:
        state = State(state={})

        validate_recovery_state(state)

        with pytest.raises(ValueError, match="class_type"):
            validate_recovery_state(state, require_class_type=True)
        with pytest.raises(ValueError, match="config"):
            validate_recovery_state(state, require_config=True)

    def test_validate_recovery_state_rejects_non_state_payload(self) -> None:
        with pytest.raises(TypeError, match="expects a State payload"):
            validate_recovery_state({"state": {}})  # type: ignore[arg-type]

    def test_get_state_returns_canonical_state_payload(self) -> None:
        state_mixin = _ParentStateMixin(value=3, child_value=7)

        state = state_mixin.get_state()

        assert isinstance(state, State)
        assert state.class_type is _ParentStateMixin
        assert state.state["value"] == 3
        assert isinstance(state.state["child"], State)
        assert state.state["child"].class_type is _NestedStateMixin
        assert state.state["child"].state["value"] == 7

    def test_get_state_canonicalizes_legacy_dict_hook(self) -> None:
        state_mixin = _LegacyDictStateMixin(value=5)

        state = state_mixin.get_state()

        assert isinstance(state, State)
        assert state.class_type is _LegacyDictStateMixin
        assert state.config is None
        assert state.state == {"value": 5}

    def test_get_state_rejects_default_container_backed_objects(self) -> None:
        state_mixin = _DefaultDictBackedStateMixin({"value": 5})

        with pytest.raises(TypeError, match="attribute-backed objects"):
            state_mixin.get_state()

    def test_load_state_rejects_default_container_backed_objects(self) -> None:
        state_mixin = _DefaultDictBackedStateMixin()

        with pytest.raises(TypeError, match="attribute-backed objects"):
            state_mixin.load_state(
                State(
                    state={"value": 5},
                    class_type=_DefaultDictBackedStateMixin,
                    config=None,
                )
            )

    def test_container_backed_objects_can_define_custom_state_hooks(
        self,
    ) -> None:
        state_mixin = _HookedDictBackedStateMixin({"value": 5})

        state = state_mixin.get_state()

        assert state.class_type is _HookedDictBackedStateMixin
        assert state.state == {"items": {"value": 5}}

        state_mixin.clear()
        state_mixin.load_state(state)

        assert dict(state_mixin) == {"value": 5}

    def test_load_state_accepts_live_state_payload(self) -> None:
        state_mixin = _ParentStateMixin(value=3, child_value=7)
        state = state_mixin.get_state()
        original_child = state_mixin.child

        state_mixin.value = 0
        state_mixin.child.value = 0
        state_mixin.load_state(state)

        assert state_mixin.value == 3
        assert state_mixin.child is not original_child
        assert state_mixin.child.value == 7
        assert isinstance(state.state["child"], State)

    def test_two_phase_state_materialize_protocol_is_runtime_checkable(
        self,
    ) -> None:
        materialized = _TwoPhaseMaterializeState.allocate_state_instance()

        assert isinstance(materialized, StateMaterializeProtocol)

    def test_load_state_rejects_invalid_input_type(self) -> None:
        state_mixin = _ParentStateMixin()

        with pytest.raises(TypeError, match="str path or State payload"):
            state_mixin.load_state(123)  # type: ignore[arg-type]

    def test_obj2state_state2obj_preserves_sequence_kinds(self) -> None:
        payload = {
            "items": [1, {"nested": 2}],
            "pair": ("left", "right"),
        }

        encoded = obj2state(payload)
        decoded = state2obj(encoded)

        assert isinstance(encoded["items"], StateSequence)
        assert encoded["items"].kind == "list"
        assert isinstance(encoded["pair"], StateSequence)
        assert encoded["pair"].kind == "tuple"
        assert decoded == payload
        assert isinstance(decoded["items"], list)
        assert isinstance(decoded["pair"], tuple)

    def test_obj2state_converts_legacy_state_list_to_state_sequence(
        self,
    ) -> None:
        encoded = obj2state(StateList([1, {"nested": 2}]))

        assert isinstance(encoded, StateSequence)
        assert encoded.kind == "list"
        assert state2obj(encoded) == [1, {"nested": 2}]

    @pytest.mark.parametrize(
        ("payload", "expected_type"),
        [
            ([1, {"nested": 2}], list),
            (("left", "right"), tuple),
        ],
    )
    def test_load_top_level_sequence_payload(
        self,
        tmp_local_folder: str,
        payload: list[object] | tuple[object, ...],
        expected_type: type[list] | type[tuple],
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            obj2state(payload).save(save_path)
            recovered = load(save_path)

        assert recovered == payload
        assert isinstance(recovered, expected_type)

    def test_obj2state_preserves_user_dict_with_container_like_key(
        self,
    ) -> None:
        payload = {
            "container": {
                "__state_container__": "sequence",
                "kind": "list",
                "items": [1, 2],
            }
        }

        decoded = state2obj(obj2state(payload))

        assert decoded == payload
        assert isinstance(decoded["container"], dict)

    def test_load_state_preserves_sequence_kinds(self) -> None:
        state_mixin = _SequenceStateMixin()
        state = state_mixin.get_state()

        state_mixin.items = []
        state_mixin.pair = ()
        state_mixin.load_state(state)

        assert state_mixin.items == [1, {"nested": 2}]
        assert isinstance(state_mixin.items, list)
        assert state_mixin.pair == ("left", "right")
        assert isinstance(state_mixin.pair, tuple)

    def test_save_load_preserves_sequence_kinds(
        self,
        tmp_local_folder: str,
    ) -> None:
        state_mixin = _SequenceStateMixin()

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_mixin.save(save_path)
            recovered_mixin = _SequenceStateMixin.load(save_path)

        assert recovered_mixin.items == [1, {"nested": 2}]
        assert isinstance(recovered_mixin.items, list)
        assert recovered_mixin.pair == ("left", "right")
        assert isinstance(recovered_mixin.pair, tuple)

    def test_hierarchical_save_recurses_into_sequence_items(
        self,
        tmp_local_folder: str,
    ) -> None:
        state_mixin = _SequenceWithStateMixin()

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_mixin.save(save_path, hierarchical_save=True)

            assert os.path.isdir(
                os.path.join(save_path, "state", "items", "items", "0")
            )
            recovered_mixin = _SequenceWithStateMixin.load(save_path)

        assert isinstance(recovered_mixin.items, list)
        assert len(recovered_mixin.items) == 1
        assert isinstance(recovered_mixin.items[0], _NestedStateMixin)
        assert recovered_mixin.items[0].value == 4

    def test_shared_reference_identity_is_not_preserved(self) -> None:
        shared = [1, 2]
        payload = {"first": shared, "second": shared}

        decoded = state2obj(obj2state(payload))

        assert decoded == payload
        assert decoded["first"] is not decoded["second"]

    def test_obj2state_rejects_circular_dict(self) -> None:
        payload = {}
        payload["self"] = payload

        with pytest.raises(ValueError, match="Circular reference"):
            obj2state(payload)

    def test_get_state_rejects_circular_mixin_state(self) -> None:
        state_mixin = _ParentStateMixin()
        state_mixin.self = state_mixin

        with pytest.raises(ValueError, match="Circular reference"):
            state_mixin.get_state()

    def test_load_state_rejects_circular_state_payload(self) -> None:
        state_mixin = StateSaveLoadMixin()
        state = State(state={}, class_type=None, config=None)
        state.state["self"] = state

        with pytest.raises(ValueError, match="Circular reference"):
            state_mixin.load_state(state)

    def test_state2obj_rejects_circular_state_payload(self) -> None:
        state = State(state={}, class_type=_ParentStateMixin, config=None)
        state.state["self"] = state

        with pytest.raises(ValueError, match="Circular reference"):
            state2obj(state)

    def test_decode_state_payload_for_apply_keeps_nested_apply_only_state(
        self,
    ) -> None:
        nested = State(state={"value": 3}, class_type=None, config=None)

        decoded = decode_state_payload_for_apply({"nested": nested})

        assert isinstance(decoded["nested"], State)
        assert decoded["nested"].class_type is None
        assert decoded["nested"].state == {"value": 3}

    def test_decode_state_payload_for_apply_materializes_constructable_state(
        self,
    ) -> None:
        nested = State(
            state={"value": 4},
            class_type=_NestedStateMixin,
            config=None,
        )

        decoded = decode_state_payload_for_apply({"nested": nested})

        assert isinstance(decoded["nested"], _NestedStateMixin)
        assert decoded["nested"].value == 4

    def test_decode_state_payload_for_apply_materializes_two_phase_state(
        self,
    ) -> None:
        nested = State(
            state={"value": 9},
            class_type=_TwoPhaseMaterializeState,
            config=None,
        )

        decoded = decode_state_payload_for_apply({"nested": nested})

        assert isinstance(decoded["nested"], _TwoPhaseMaterializeState)
        assert decoded["nested"].value == 9

    def test_decode_state_payload_for_apply_can_preserve_constructable_state(
        self,
    ) -> None:
        nested = State(
            state={"value": 4},
            class_type=_NestedStateMixin,
            config=None,
        )

        decoded = decode_state_payload_for_apply(
            {"nested": nested},
            constructable_state_apply_mode=(
                ConstructableStateApplyMode.PRESERVE_STATE
            ),
        )

        assert isinstance(decoded["nested"], State)
        assert decoded["nested"].class_type is _NestedStateMixin
        assert decoded["nested"].state == {"value": 4}

    def test_decode_state_payload_for_apply_graph_profile_preserves_aliases(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared: dict[str, int] = {"value": 1}
        state = State(
            state={"left": shared, "right": shared},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            loaded_state = State.load(save_path, protocol="pickle")

        decoded = decode_state_payload_for_apply(
            loaded_state.state,
            save_profile=loaded_state.save_profile,
        )

        assert decoded["left"] == {"value": 1}
        assert decoded["left"] is decoded["right"]

    def test_decode_state_payload_for_apply_tree_profile_does_not_preserve_aliases(  # noqa: E501
        self,
    ) -> None:
        shared = {"value": 1}

        decoded = decode_state_payload_for_apply(
            {"left": shared, "right": shared},
            save_profile="tree",
        )

        assert decoded["left"] == {"value": 1}
        assert decoded["left"] is not decoded["right"]

    def test_decode_state_payload_for_apply_nested_graph_state_preserves_aliases(  # noqa: E501
        self,
    ) -> None:
        shared = {"value": 1}
        nested = State(
            state={"left": shared, "right": shared},
            class_type=None,
            config=None,
            save_profile=GRAPH_PROFILE,
        )

        decoded = decode_state_payload_for_apply(
            {"nested": nested},
            save_profile="tree",
        )

        assert isinstance(decoded["nested"], State)
        assert decoded["nested"].state["left"] == {"value": 1}
        assert (
            decoded["nested"].state["left"] is decoded["nested"].state["right"]
        )

    def test_load_state_can_preserve_constructable_nested_state_for_apply(
        self,
    ) -> None:
        state_mixin = _ParentStateMixinPreserveConstructableChild(
            value=3,
            child_value=7,
        )
        state = state_mixin.get_state()
        original_child = state_mixin.child

        state_mixin.value = 0
        state_mixin.child.value = 0
        state_mixin.load_state(
            state,
            constructable_state_apply_mode=(
                ConstructableStateApplyMode.PRESERVE_STATE
            ),
        )

        assert state_mixin.value == 3
        assert state_mixin.child is original_child
        assert state_mixin.child.value == 7

    def test_load_state_keeps_nested_apply_only_state_payload(self) -> None:
        nested = State(state={"value": 3}, class_type=None, config=None)
        state_mixin = StateSaveLoadMixin()

        state_mixin.load_state(
            State(
                state={"nested": nested},
                class_type=StateSaveLoadMixin,
                config=None,
            )
        )

        assert isinstance(state_mixin.nested, State)
        assert state_mixin.nested.state == {"value": 3}
        assert state_mixin.nested.class_type is None

    def test_load_state_decodes_nested_apply_only_state_sequences(
        self,
    ) -> None:
        nested = obj2state(
            State(
                state={
                    "items": [1, {"nested": 2}],
                    "pair": ("left", "right"),
                },
                class_type=None,
                config=None,
            )
        )
        state_mixin = StateSaveLoadMixin()

        state_mixin.load_state(
            State(
                state={"nested": nested},
                class_type=StateSaveLoadMixin,
                config=None,
            )
        )

        assert isinstance(state_mixin.nested, State)
        assert state_mixin.nested.state["items"] == [1, {"nested": 2}]
        assert isinstance(state_mixin.nested.state["items"], list)
        assert state_mixin.nested.state["pair"] == ("left", "right")
        assert isinstance(state_mixin.nested.state["pair"], tuple)

    def test_state2obj_rejects_nested_apply_only_state_payload(self) -> None:
        nested = State(state={"value": 3}, class_type=None, config=None)
        state = State(
            state={"nested": nested},
            class_type=StateSaveLoadMixin,
            config=None,
        )

        with pytest.raises(ValueError, match="class_type"):
            state2obj(state)

    def test_state2obj_materializes_two_phase_state(self) -> None:
        state = State(
            state={"value": 8},
            class_type=_TwoPhaseMaterializeState,
            config=None,
        )

        decoded = state2obj(state)

        assert isinstance(decoded, _TwoPhaseMaterializeState)
        assert decoded.value == 8

    def test_state2obj_rejects_top_level_apply_only_state_payload(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="class_type"):
            state2obj(State(state={}))

    def test_module_load_rejects_apply_only_state_payload(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            State(state={}).save(save_path)

            with pytest.raises(ValueError, match="class_type"):
                load(save_path)

    def test_state_load_accepts_legacy_public_module_metadata_path(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            State(state={"value": 1}).save(save_path, protocol="pickle")

            meta_path = os.path.join(save_path, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            meta["__config_type__"] = (
                "robo_orchard_lab.utils.state:StateConfig"
            )
            meta["class_type"] = "robo_orchard_lab.utils.state:State"
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state == {"value": 1}
        assert recovered.save_profile == "tree"

    def test_module_load_accepts_legacy_state_list_root(
        self,
        tmp_local_folder: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            StateList([1, {"nested": 2}]).save(save_path)
            recovered = load(save_path)

        assert recovered == [1, {"nested": 2}]

    def test_graph_profile_save_preserves_shared_identity(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared: dict[str, int] = {"value": 1}
        state = State(
            state={},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )
        state.state["left"] = shared
        state.state["right"] = shared

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")

            assert os.path.exists(
                os.path.join(save_path, "graph_manifest.json")
            )
            assert os.path.exists(os.path.join(save_path, "entry.json"))
            assert os.path.isdir(os.path.join(save_path, "state", "left"))
            assert os.path.isdir(os.path.join(save_path, "state", "right"))

            with pytest.raises(
                NonSelfContainedEntryError,
                match="reference_entry",
            ):
                load(os.path.join(save_path, "state", "right"))
            assert load(os.path.join(save_path, "state", "left")) == {
                "value": 1
            }

            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state["left"] == {"value": 1}
        assert recovered.state["left"] is recovered.state["right"]

    def test_state_sequence_root_uses_graph_aware_save_profile(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared: dict[str, int] = {"value": 1}
        sequence = StateSequence(
            kind="list",
            items=[shared, shared],
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            sequence.save(save_path, protocol="pickle")

            assert os.path.exists(
                os.path.join(save_path, "graph_manifest.json")
            )
            assert os.path.exists(os.path.join(save_path, "entry.json"))

            recovered = StateSequence.load(save_path, protocol="pickle")

        assert recovered.kind == "list"
        assert recovered.save_profile == GRAPH_PROFILE
        assert recovered.items[0] == {"value": 1}
        assert recovered.items[0] is recovered.items[1]

    def test_state_list_root_uses_graph_aware_save_profile(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared: dict[str, int] = {"value": 1}
        state_list = StateList(
            [shared, shared],
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state_list.save(save_path, protocol="pickle")

            assert os.path.exists(
                os.path.join(save_path, "graph_manifest.json")
            )
            assert os.path.exists(os.path.join(save_path, "entry.json"))

            recovered = StateList.load(save_path, protocol="pickle")

        assert recovered.save_profile == GRAPH_PROFILE
        assert recovered[0] == {"value": 1}
        assert recovered[0] is recovered[1]

    def test_graph_profile_manifest_records_versioned_contract(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"value": 1},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")

            with open(os.path.join(save_path, "graph_manifest.json")) as f:
                manifest = json.load(f)

        assert manifest["profile"] == GRAPH_PROFILE
        assert (
            manifest["format_version"]
            == GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION
        )
        assert manifest["root_node_id"] in manifest["nodes"]

    def test_graph_profile_parser_uses_enum_contract_records(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"value": 1},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")

            with open(os.path.join(save_path, "entry.json")) as f:
                entry_payload = json.load(f)
            with open(os.path.join(save_path, "graph_manifest.json")) as f:
                manifest_payload = json.load(f)

            entry = _read_graph_entry(os.path.join(save_path, "entry.json"))
            manifest = _read_graph_manifest(save_path)

        assert entry_payload["kind"] == "owner"
        assert entry_payload["node_type"] == "state"
        assert manifest_payload["nodes"]["n0"]["node_type"] == "state"
        assert entry.kind is GraphEntryKind.OWNER
        assert entry.node_type is GraphNodeType.STATE
        assert entry.storage_kind is GraphStorageKind.GRAPH_CONTAINER
        assert entry.load_scope is GraphLoadScope.SELF_CONTAINED
        assert manifest.nodes["n0"].node_type is GraphNodeType.STATE

    def test_graph_profile_save_respects_compact_state_layout(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"value": 1, "nested": {"child": 2}},
            hierarchical_save=False,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")

            assert os.path.exists(os.path.join(save_path, "state.pkl"))
            assert not os.path.exists(os.path.join(save_path, "state"))

            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state == {"value": 1, "nested": {"child": 2}}
        assert recovered.hierarchical_save is False

    def test_graph_profile_save_respects_split_state_layout(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"value": 1, "nested": {"child": 2}},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")

            assert not os.path.exists(os.path.join(save_path, "state.pkl"))
            assert os.path.exists(
                os.path.join(save_path, "state", "value", "entry.json")
            )
            assert os.path.exists(
                os.path.join(save_path, "state", "nested", "entry.json")
            )

            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state == {"value": 1, "nested": {"child": 2}}
        assert recovered.hierarchical_save is True

    def test_graph_profile_load_rejects_unknown_format_version(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"value": 1},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            manifest_path = os.path.join(save_path, "graph_manifest.json")
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["format_version"] = (
                GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION + 1
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            with pytest.raises(
                GraphStateUnsupportedFormatVersionError,
                match="Unsupported graph-aware State format version",
            ):
                State.load(save_path, protocol="pickle")

    def test_graph_profile_save_preserves_circular_dict(
        self,
        tmp_local_folder: str,
    ) -> None:
        circular: dict[str, object] = {"value": 1}
        circular["self"] = circular
        state = State(
            state={"root": circular},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state["root"]["value"] == 1
        assert recovered.state["root"]["self"] is recovered.state["root"]

    def test_graph_profile_save_preserves_circular_list(
        self,
        tmp_local_folder: str,
    ) -> None:
        circular: list[object] = ["value"]
        circular.append(circular)
        state = State(
            state={"root": circular},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = State.load(save_path, protocol="pickle")

        assert recovered.state["root"][0] == "value"
        assert recovered.state["root"][1] is recovered.state["root"]

    def test_graph_profile_load_state_applies_cyclic_payload(
        self,
        tmp_local_folder: str,
    ) -> None:
        circular: dict[str, object] = {"value": 1}
        circular["self"] = circular
        state = State(
            state={"root": circular},
            class_type=StateSaveLoadMixin,
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = StateSaveLoadMixin.load(save_path)
            materialized = load(save_path)

        assert recovered.root["self"] is recovered.root
        assert materialized.root["self"] is materialized.root

    def test_graph_profile_load_state_applies_custom_leaf_payload(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"leaf": _CustomLeafState(3)},
            class_type=StateSaveLoadMixin,
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = StateSaveLoadMixin.load(save_path)

        assert isinstance(recovered.leaf, _CustomLeafState)
        assert recovered.leaf.value == 3

    def test_graph_profile_custom_leaf_subentry_loads_directly(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"leaf": _CustomLeafState(3)},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = load(os.path.join(save_path, "state", "leaf"))

        assert isinstance(recovered, _CustomLeafState)
        assert recovered.value == 3

    def test_graph_profile_list_subentry_loads_directly(
        self,
        tmp_local_folder: str,
    ) -> None:
        state = State(
            state={"items": [{"value": 1}, {"value": 2}]},
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = load(os.path.join(save_path, "state", "items"))

        assert recovered == [{"value": 1}, {"value": 2}]

    def test_graph_profile_load_materializes_nested_stateful_payload(
        self,
        tmp_local_folder: str,
    ) -> None:
        nested_state = _NestedStateMixin(4).get_state()
        state = State(
            state={"child": nested_state},
            class_type=StateSaveLoadMixin,
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = StateSaveLoadMixin.load(save_path)
            materialized = load(save_path)

        assert isinstance(recovered.child, _NestedStateMixin)
        assert recovered.child.value == 4
        assert isinstance(materialized.child, _NestedStateMixin)
        assert materialized.child.value == 4

    def test_graph_profile_load_preserves_shared_stateful_identity(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared_state = _NestedStateMixin(7).get_state()
        state = State(
            state={"left": shared_state, "right": shared_state},
            class_type=StateSaveLoadMixin,
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = StateSaveLoadMixin.load(save_path)

        assert isinstance(recovered.left, _NestedStateMixin)
        assert recovered.left is recovered.right
        assert recovered.left.value == 7

    def test_graph_profile_load_preserves_shared_two_phase_stateful_identity(
        self,
        tmp_local_folder: str,
    ) -> None:
        shared_state = State(
            state={"value": 11},
            class_type=_TwoPhaseMaterializeState,
            config=None,
        )
        state = State(
            state={"left": shared_state, "right": shared_state},
            class_type=StateSaveLoadMixin,
            hierarchical_save=True,
            save_profile=GRAPH_PROFILE,
        )

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            state.save(save_path, protocol="pickle")
            recovered = StateSaveLoadMixin.load(save_path)

        assert isinstance(recovered.left, _TwoPhaseMaterializeState)
        assert recovered.left is recovered.right
        assert recovered.left.value == 11

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
