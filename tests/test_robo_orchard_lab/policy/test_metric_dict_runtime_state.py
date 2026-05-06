# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
import copy
import pickle
from typing import cast

import pytest

from robo_orchard_lab.metrics.base import MetricDict
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    capture_metric_state,
    load_metric_state,
)
from robo_orchard_lab.utils.state import (
    ConstructableStateApplyMode,
    State,
    StateSaveLoadMixin,
    obj2state,
)


class _RecoverableMetric(StateSaveLoadMixin):
    def __init__(self) -> None:
        self.value = 0

    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret

    def compute(self) -> int:
        return self.value

    def to(self, *args, **kwargs) -> None:
        del args, kwargs

    def _get_state(self) -> dict[str, object]:
        return {"value": self.value}


class _SequenceRecoverableMetric(_RecoverableMetric):
    def __init__(self) -> None:
        super().__init__()
        self.runtime_state = {}

    def _get_state(self) -> dict[str, object]:
        return {"runtime_state": copy.deepcopy(self.runtime_state)}

    def _set_state(self, state: State) -> None:
        payload = state.state
        if not isinstance(payload, dict):
            raise TypeError(
                "_SequenceRecoverableMetric state payload must be a dict. "
                f"Got {type(payload).__name__}."
            )
        runtime_state = payload.get("runtime_state")
        if not isinstance(runtime_state, dict):
            raise TypeError(
                "_SequenceRecoverableMetric state field `runtime_state` "
                f"must be a dict. Got {type(runtime_state).__name__}."
            )
        self.runtime_state = runtime_state


class _ConstructableRecoverableMetric(StateSaveLoadMixin):
    def __init__(self) -> None:
        self.value = 0

    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret

    def compute(self) -> int:
        return self.value

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _PlainSnapshotMetric:
    def __init__(self) -> None:
        self.value = 0

    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret

    def compute(self) -> int:
        return self.value

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


def test_capture_metric_state_returns_canonical_state() -> None:
    metrics = MetricDict({"score": _RecoverableMetric()})
    score_metric = cast(_RecoverableMetric, metrics["score"])
    score_metric.value = 7

    canonical_state = capture_metric_state(metrics)

    assert isinstance(canonical_state, State)
    assert canonical_state.class_type is MetricDict
    assert canonical_state == metrics.get_state()

    member_state = canonical_state.state["score"]
    assert isinstance(member_state, State)
    assert member_state.class_type is _RecoverableMetric
    assert member_state.state == {"value": 7}

    score_metric.value = 0
    metrics.load_state(canonical_state)
    assert metrics["score"] is not score_metric
    assert cast(_RecoverableMetric, metrics["score"]).value == 7


def test_metric_dict_state_api_round_trips_runtime_state() -> None:
    metrics = MetricDict({"score": _RecoverableMetric()})
    score_metric = cast(_RecoverableMetric, metrics["score"])
    score_metric.value = 7

    state = metrics.get_state()

    assert isinstance(state, State)
    assert state.class_type is MetricDict
    assert isinstance(state.state["score"], State)
    assert state.state["score"].class_type is _RecoverableMetric
    assert state.state["score"].state == {"value": 7}

    score_metric.value = 0
    metrics.load_state(state)
    assert metrics["score"] is not score_metric
    assert cast(_RecoverableMetric, metrics["score"]).value == 7


def test_metric_dict_get_state_can_snapshot_plain_metric_member() -> None:
    metrics = MetricDict({"score": _PlainSnapshotMetric()})
    score_metric = cast(_PlainSnapshotMetric, metrics["score"])
    score_metric.value = 7

    state = metrics.get_state()

    member_state = state.state["score"]
    assert isinstance(member_state, _PlainSnapshotMetric)
    assert member_state is not score_metric
    assert member_state.value == 7

    score_metric.value = 0
    metrics.load_state(state)

    assert metrics["score"] is not score_metric
    assert isinstance(metrics["score"], _PlainSnapshotMetric)
    assert cast(_PlainSnapshotMetric, metrics["score"]).value == 7


def test_metric_dict_get_state_preserves_constructable_member_metadata() -> (
    None
):
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    score_metric = cast(_ConstructableRecoverableMetric, metrics["score"])
    score_metric.value = 7

    state = metrics.get_state()

    member_state = state.state["score"]
    assert isinstance(member_state, State)
    assert member_state.class_type is _ConstructableRecoverableMetric
    assert member_state.state == {"value": 7}


def test_metric_dict_load_state_replaces_constructable_members() -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    score_metric = cast(_ConstructableRecoverableMetric, metrics["score"])
    score_metric.value = 7
    state = metrics.get_state()

    score_metric.value = 0
    metrics.load_state(state)

    assert metrics["score"] is not score_metric
    assert isinstance(metrics["score"], _ConstructableRecoverableMetric)
    assert cast(_ConstructableRecoverableMetric, metrics["score"]).value == 7


def test_load_metric_state_replaces_constructable_metric_instances() -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    score_metric = cast(_ConstructableRecoverableMetric, metrics["score"])
    score_metric.value = 7
    state = capture_metric_state(metrics)

    score_metric.value = 0
    load_metric_state(metrics, state)

    assert metrics["score"] is not score_metric
    assert cast(_ConstructableRecoverableMetric, metrics["score"]).value == 7


def test_metric_dict_load_state_does_not_depend_on_container_setstate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    score_metric = cast(_ConstructableRecoverableMetric, metrics["score"])
    score_metric.value = 7
    state = metrics.get_state()

    def _forbid_container_setstate(
        self: MetricDict,
        state_dict: dict[str, object],
    ) -> None:
        del self, state_dict
        raise AssertionError(
            "MetricDict._set_state(...) should install members directly."
        )

    monkeypatch.setattr(
        MetricDict,
        "__setstate__",
        _forbid_container_setstate,
        raising=False,
    )

    metrics.load_state(state)

    assert isinstance(metrics["score"], _ConstructableRecoverableMetric)
    assert cast(_ConstructableRecoverableMetric, metrics["score"]).value == 7


def test_metric_dict_load_state_replaces_all_recoverable_members() -> None:
    metrics = MetricDict(
        {
            "score": _RecoverableMetric(),
            "constructable": _ConstructableRecoverableMetric(),
        }
    )
    score_metric = cast(_RecoverableMetric, metrics["score"])
    constructable_metric = cast(
        _ConstructableRecoverableMetric, metrics["constructable"]
    )
    score_metric.value = 3
    constructable_metric.value = 7

    state = metrics.get_state()

    score_metric.value = 0
    constructable_metric.value = 0
    metrics.load_state(state)

    assert metrics["score"] is not score_metric
    assert cast(_RecoverableMetric, metrics["score"]).value == 3
    assert metrics["constructable"] is not constructable_metric
    assert isinstance(
        metrics["constructable"], _ConstructableRecoverableMetric
    )
    assert (
        cast(_ConstructableRecoverableMetric, metrics["constructable"]).value
        == 7
    )


def test_metric_dict_pickle_round_trip_preserves_members() -> None:
    metrics = MetricDict({"score": _RecoverableMetric()})
    score_metric = cast(_RecoverableMetric, metrics["score"])
    score_metric.value = 7

    round_tripped = pickle.loads(pickle.dumps(metrics))

    assert isinstance(round_tripped, MetricDict)
    assert isinstance(round_tripped["score"], _RecoverableMetric)
    assert cast(_RecoverableMetric, round_tripped["score"]).value == 7


def test_metric_dict_state_wrapper_materializes_constructable_payload() -> (
    None
):
    metrics = MetricDict()
    state = State(
        state=cast(
            dict[str, object],
            obj2state(
                {
                    "sequence": State(
                        state={
                            "runtime_state": {
                                "items": [1, {"nested": 2}],
                                "pair": ("left", "right"),
                            }
                        },
                        class_type=_SequenceRecoverableMetric,
                    )
                }
            ),
        ),
        class_type=MetricDict,
    )

    metrics.load_state(state)

    metric = cast(_SequenceRecoverableMetric, metrics["sequence"])
    assert metric.runtime_state["items"] == [1, {"nested": 2}]
    assert isinstance(metric.runtime_state["items"], list)
    assert metric.runtime_state["pair"] == ("left", "right")
    assert isinstance(metric.runtime_state["pair"], tuple)


def test_metric_dict_load_state_replaces_container_keys() -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    replacement = MetricDict({"wrong": _ConstructableRecoverableMetric()})
    cast(_ConstructableRecoverableMetric, replacement["wrong"]).value = 7

    metrics.load_state(replacement.get_state())

    assert list(metrics.keys()) == ["wrong"]
    assert isinstance(metrics["wrong"], _ConstructableRecoverableMetric)
    assert cast(_ConstructableRecoverableMetric, metrics["wrong"]).value == 7


def test_metric_dict_load_rejects_apply_only_member_states(
    tmp_path,
) -> None:
    save_path = tmp_path / "metric-state"
    State(
        state={"score": State(state={"value": 7})},
        class_type=MetricDict,
    ).save(str(save_path))

    with pytest.raises(TypeError, match="replace-only"):
        MetricDict.load(str(save_path))


def test_metric_dict_load_state_rejects_preserve_state_mode() -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})

    with pytest.raises(ValueError, match="constructable_state_apply_mode"):
        metrics.load_state(
            metrics.get_state(),
            constructable_state_apply_mode=(
                ConstructableStateApplyMode.PRESERVE_STATE
            ),
        )


def test_metric_dict_load_materializes_constructable_member_metrics(
    tmp_path,
) -> None:
    metrics = MetricDict({"score": _ConstructableRecoverableMetric()})
    cast(_ConstructableRecoverableMetric, metrics["score"]).value = 7
    save_path = tmp_path / "metric-state"

    metrics.save(str(save_path))

    loaded = MetricDict.load(str(save_path))

    assert isinstance(loaded, MetricDict)
    assert isinstance(loaded["score"], _ConstructableRecoverableMetric)
    assert cast(_ConstructableRecoverableMetric, loaded["score"]).value == 7


def test_metric_dict_load_restores_plain_snapshot_member_metrics(
    tmp_path,
) -> None:
    metrics = MetricDict({"score": _PlainSnapshotMetric()})
    cast(_PlainSnapshotMetric, metrics["score"]).value = 7
    save_path = tmp_path / "metric-state"

    metrics.save(str(save_path))

    loaded = MetricDict.load(str(save_path))

    assert isinstance(loaded, MetricDict)
    assert isinstance(loaded["score"], _PlainSnapshotMetric)
    assert cast(_PlainSnapshotMetric, loaded["score"]).value == 7
