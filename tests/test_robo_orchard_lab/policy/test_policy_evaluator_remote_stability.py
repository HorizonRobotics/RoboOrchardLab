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
import concurrent.futures
from collections.abc import Callable, Sequence
from typing import Any

import pytest
import ray
from ray.exceptions import ActorDiedError, GetTimeoutError, RayTaskError
from robo_orchard_core.utils.ray import (
    RayRemoteClassConfig,
    RayRemoteInstance,
)

import robo_orchard_lab.policy.evaluator.remote as remote_module
from robo_orchard_lab.policy import (
    PolicyEvaluationError,
    PolicyEvaluationExecutionError,
    PolicyEvaluationRemoteTimeoutError,
    PolicyEvaluationWorkerLostError,
    PolicyEvaluatorConfig,
    PolicyEvaluatorRemote,
)
from robo_orchard_lab.policy.evaluator.base import (
    PolicyEvaluatorRecoverySnapshot,
    evaluate_rollout_stop_condition,
)
from robo_orchard_lab.policy.evaluator.contracts import (
    EpisodeResult,
    EvaluationStatus,
    TerminalReason,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.utils.state import State

_ENV_CFG: Any = "env"
_POLICY_CFG: Any = "policy"
_DEFAULT = object()


class _ScriptedFuture(concurrent.futures.Future[Any]):
    @classmethod
    def resolved(cls, value: Any) -> _ScriptedFuture:
        future = cls()
        future.set_result(value)
        return future

    @classmethod
    def failed(cls, exception: BaseException) -> _ScriptedFuture:
        future = cls()
        future.set_exception(exception)
        return future


class _ScriptedObjectRef:
    def __init__(self, future: concurrent.futures.Future[Any]) -> None:
        self._future = future

    def future(self) -> concurrent.futures.Future[Any]:
        return self._future

    def get(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


class _TimeoutThenResolvedRef:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def get(self, timeout: float | None = None) -> Any:
        del timeout
        self.calls += 1
        if self.calls == 1:
            raise GetTimeoutError("first wait timed out")
        return self.value


def _resolved_ref(value: Any) -> _ScriptedObjectRef:
    return _ScriptedObjectRef(_ScriptedFuture.resolved(value))


def _failed_ref(exception: BaseException) -> _ScriptedObjectRef:
    return _ScriptedObjectRef(_ScriptedFuture.failed(exception))


def _coerce_ref(value: Any) -> _ScriptedObjectRef | _TimeoutThenResolvedRef:
    if isinstance(value, (_ScriptedObjectRef, _TimeoutThenResolvedRef)):
        return value
    if isinstance(value, BaseException):
        return _failed_ref(value)
    return _resolved_ref(value)


class _RemoteMethod:
    def __init__(self, handler: Callable[..., Any]) -> None:
        self._handler = handler
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._handler(*args, **kwargs)


class _UnsupportedCloseGenerator:
    def __init__(self, refs: Sequence[Any]) -> None:
        self._refs = list(refs)
        self.close_calls = 0

    def __iter__(self) -> Any:
        return iter(self._refs)

    def close(self) -> None:
        self.close_calls += 1
        raise NotImplementedError("`gen.close` is not supported.")


class _RayWaitStreamRefGenerator:
    def __init__(
        self,
        refs: Sequence[Any],
        *,
        ready_script: Sequence[bool] = (),
        complete_when_empty: bool = True,
    ) -> None:
        self._refs = list(refs)
        self._ready_script = list(ready_script)
        self._complete_when_empty = complete_when_empty
        self.wait_timeouts: list[float | None] = []
        self.close_calls = 0

    def wait(self, timeout: float | None) -> bool:
        self.wait_timeouts.append(timeout)
        if self._ready_script:
            return self._ready_script.pop(0)
        return bool(self._refs) or self._complete_when_empty

    def __iter__(self) -> _RayWaitStreamRefGenerator:
        return self

    def __next__(self) -> Any:
        if not self._refs:
            raise StopIteration
        return self._refs.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class _MetricHandle:
    def reset(self, **kwargs: Any) -> None:
        del kwargs

    def update(self, action: Any, step_ret: Any) -> None:
        del action, step_ret

    def compute(self) -> float:
        return 1.0

    def to(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _ScriptedActor:
    def __init__(
        self,
        *,
        evaluate_response: Any = None,
        stream_response: Sequence[Any] = (),
        compute_response: Any | None = None,
        metric_state_response: Any | None = None,
        snapshot_response: Any = _DEFAULT,
    ) -> None:
        self.evaluate_response = evaluate_response
        self.stream_response = list(stream_response)
        self.compute_response = (
            {} if compute_response is None else compute_response
        )
        self.metric_state_response = (
            State(state={"metric": "checkpoint"})
            if metric_state_response is None
            else metric_state_response
        )
        self.snapshot_response = snapshot_response
        self.setup_kwargs: dict[str, Any] | None = None
        self.restored_metric_states: list[Any] = []
        self.restored_snapshots: list[PolicyEvaluatorRecoverySnapshot] = []

        def _setup(**kwargs: Any) -> _ScriptedObjectRef:
            self.setup_kwargs = kwargs
            return _resolved_ref(None)

        def _export_snapshot() -> _ScriptedObjectRef:
            if self.snapshot_response is not _DEFAULT:
                return _coerce_ref(self.snapshot_response)
            kwargs = self.setup_kwargs or {}
            return _resolved_ref(
                PolicyEvaluatorRecoverySnapshot(
                    env_cfg=kwargs.get("env_cfg"),
                    policy_recovery_input=kwargs.get("policy_or_cfg"),
                    metrics_recovery_input=kwargs.get("metrics"),
                    device=kwargs.get("device"),
                )
            )

        def _restore_snapshot(
            snapshot: PolicyEvaluatorRecoverySnapshot,
        ) -> _ScriptedObjectRef:
            self.restored_snapshots.append(snapshot)
            return _resolved_ref(None)

        def _restore_metric_state(state: State) -> _ScriptedObjectRef:
            self.restored_metric_states.append(state)
            return _resolved_ref(None)

        self.setup = _RemoteMethod(_setup)
        self.reconfigure_env = _RemoteMethod(
            lambda env_cfg, force_recreate=None: _resolved_ref(None)
        )
        self.reconfigure_policy = _RemoteMethod(
            lambda policy_or_cfg, device=None: _resolved_ref(None)
        )
        self.reconfigure_metrics = _RemoteMethod(
            lambda metrics: _resolved_ref(None)
        )
        self.evaluate_episode = _RemoteMethod(
            lambda **kwargs: _coerce_ref(self.evaluate_response)
        )
        self.make_episode_evaluation = _RemoteMethod(
            lambda **kwargs: [
                _coerce_ref(item) for item in self.stream_response
            ]
        )
        self.reset_env = _RemoteMethod(lambda **kwargs: _resolved_ref(kwargs))
        self.reset_policy = _RemoteMethod(lambda **kwargs: _resolved_ref(None))
        self.reset_metrics = _RemoteMethod(
            lambda **kwargs: _resolved_ref(None)
        )
        self.get_metrics = _RemoteMethod(lambda: _resolved_ref(None))
        self.compute_metrics = _RemoteMethod(
            lambda: _coerce_ref(self.compute_response)
        )
        self._export_metric_runtime_state = _RemoteMethod(
            lambda: _coerce_ref(self.metric_state_response)
        )
        self._restore_metric_runtime_state = _RemoteMethod(
            _restore_metric_state
        )
        self._export_recovery_snapshot = _RemoteMethod(_export_snapshot)
        self._restore_recovery_snapshot = _RemoteMethod(_restore_snapshot)


class _ScriptedRemoteClass:
    def __init__(
        self,
        actor: _ScriptedActor | Sequence[_ScriptedActor],
    ) -> None:
        self.actors = [actor] if isinstance(actor, _ScriptedActor) else actor
        self.actor = self.actors[0]
        self.creation_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.get_timeouts: list[float | None] = []
        self.kill_calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def remote(self, *args: Any, **kwargs: Any) -> _ScriptedActor:
        actor_idx = min(len(self.creation_calls), len(self.actors) - 1)
        actor = self.actors[actor_idx]
        self.actor = actor
        self.creation_calls.append((args, kwargs))
        return actor


def _fake_ray_get(ref: Any, *args: Any, **kwargs: Any) -> Any:
    timeout = kwargs.get("timeout")
    if args:
        timeout = args[0]
    if isinstance(ref, list):
        return [_fake_ray_get(item, timeout=timeout) for item in ref]
    if isinstance(ref, (_ScriptedObjectRef, _TimeoutThenResolvedRef)):
        return ref.get(timeout=timeout)
    return ref


def _make_remote_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    actor: _ScriptedActor | Sequence[_ScriptedActor],
    *,
    rollout_timeout_s: float | None = 120.0,
    reset_timeout_s: float | None = 1200.0,
    timeout_grace_retries: int = 1,
) -> tuple[PolicyEvaluatorRemote, _ScriptedRemoteClass]:
    remote_cls = _ScriptedRemoteClass(actor)

    def _fake_remote_instance_init(self, cfg, **kwargs) -> None:
        del kwargs
        self.cfg = cfg
        self.remote_cls = remote_cls
        self._remote = remote_cls.remote(cfg.instance_config)
        self._remote_checked = True

    monkeypatch.setattr(
        RayRemoteInstance,
        "__init__",
        _fake_remote_instance_init,
    )

    def _recording_ray_get(ref: Any, *args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if args:
            timeout = args[0]
        remote_cls.get_timeouts.append(timeout)
        return _fake_ray_get(ref, *args, **kwargs)

    def _recording_ray_kill(
        actor: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        remote_cls.kill_calls.append((actor, args, kwargs))

    monkeypatch.setattr(ray, "get", _recording_ray_get)
    monkeypatch.setattr(ray, "kill", _recording_ray_kill)

    cfg = PolicyEvaluatorConfig().as_remote(
        remote_class_config=RayRemoteClassConfig(
            num_cpus=0,
            num_gpus=0,
        ),
        rollout_timeout_s=rollout_timeout_s,
        reset_timeout_s=reset_timeout_s,
        timeout_grace_retries=timeout_grace_retries,
    )
    return PolicyEvaluatorRemote(cfg), remote_cls


def _install_fake_stream_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_module,
        "ObjectRefGenerator",
        _RayWaitStreamRefGenerator,
    )

    def _ray_wait(
        waitables: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any]]:
        del args
        assert len(waitables) == 1
        waitable = waitables[0]
        if not isinstance(waitable, _RayWaitStreamRefGenerator):
            raise TypeError(f"Unsupported waitable: {type(waitable)}")
        timeout = kwargs.get("timeout")
        if waitable.wait(timeout):
            return [waitable], []
        return [], [waitable]

    monkeypatch.setattr(remote_module.ray, "wait", _ray_wait)


def _evaluator_metrics() -> EvaluatorMetrics:
    return EvaluatorMetrics.from_metric(_MetricHandle())


def _execution_error() -> RayTaskError:
    return RayTaskError(
        "evaluate_episode",
        "episode execution failed",
        PolicyEvaluationExecutionError(
            "episode execution failed",
            result=EpisodeResult(
                status=EvaluationStatus.FAILED,
                terminal_reason=TerminalReason.ERROR,
                episode_steps=3,
                metrics={},
            ),
            cause_type="ValueError",
            cause_message="policy act failed",
        ),
    )


class TestPolicyEvaluatorRemoteStability:
    def test_setup_and_evaluate_episode_forward_to_remote_actor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response={"success_rate": 1.0},
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)
        metrics = _evaluator_metrics()

        evaluator.setup(
            env_cfg=_ENV_CFG,
            policy_or_cfg=_POLICY_CFG,
            metrics=metrics,
            device="cpu",
        )

        assert actor.setup.calls == [
            (
                (),
                {
                    "env_cfg": _ENV_CFG,
                    "policy_or_cfg": _POLICY_CFG,
                    "metrics": metrics,
                    "device": "cpu",
                },
            )
        ]
        assert actor._export_recovery_snapshot.calls == []
        assert not hasattr(evaluator, "_remote_actor_snapshot")
        assert remote_cls.get_timeouts[:1] == [1200.0]
        assert evaluator.evaluate_episode(
            max_steps=5,
            env_reset_input={"seed": 7},
            policy_reset_input={"temperature": 0.2},
        ) == {"success_rate": 1.0}
        assert actor.evaluate_episode.calls == []
        _, kwargs = actor.make_episode_evaluation.calls[0]
        assert kwargs == {
            "max_steps": 5,
            "env_reset_input": {"seed": 7},
            "policy_reset_input": {"temperature": 0.2},
            "rollout_steps": 5,
            "rollout_stop_condition": evaluate_rollout_stop_condition,
        }
        assert len(actor.compute_metrics.calls) == 1
        assert remote_cls.get_timeouts[-3:] == [1200.0, 120.0, 1200.0]

    def test_make_episode_evaluation_yields_remote_rollout_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[2, 3])
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        assert list(
            evaluator.make_episode_evaluation(
                max_steps=5,
                rollout_steps=2,
            )
        ) == [2, 3]

        _, kwargs = actor.make_episode_evaluation.calls[0]
        assert kwargs == {
            "max_steps": 5,
            "env_reset_input": None,
            "policy_reset_input": None,
            "rollout_steps": 2,
            "rollout_stop_condition": evaluate_rollout_stop_condition,
        }

    def test_episode_stream_ignores_unsupported_ray_generator_close(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)
        remote_gen = _UnsupportedCloseGenerator([_resolved_ref(2)])
        actor.make_episode_evaluation = _RemoteMethod(
            lambda **kwargs: remote_gen
        )

        assert list(evaluator.make_episode_evaluation(max_steps=5)) == [2]

        assert remote_gen.close_calls == 1
        evaluator.reset_policy()

    def test_episode_stream_rejects_other_operations_until_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[2, 3])
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        stream = evaluator.make_episode_evaluation(max_steps=5)

        assert actor.make_episode_evaluation.calls == []
        evaluator.reset_policy()

        assert next(stream) == 2
        with pytest.raises(PolicyEvaluationError):
            evaluator.reset_policy()

        stream.close()
        evaluator.reset_policy()

    def test_episode_stream_releases_guard_after_exhaustion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[2])
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        assert list(evaluator.make_episode_evaluation(max_steps=5)) == [2]

        evaluator.reset_policy()

    def test_close_is_idempotent_and_blocks_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        assert evaluator._remote_finalizer is not None
        assert evaluator._remote_finalizer.alive

        evaluator.close()
        evaluator.close()

        assert remote_cls.kill_calls == [(actor, (), {"no_restart": True})]
        assert getattr(evaluator, "_remote", None) is None
        assert evaluator._remote_finalizer is None
        with pytest.raises(PolicyEvaluationError, match="closed"):
            evaluator.reset_policy()

    def test_context_manager_closes_remote_actor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with evaluator as active:
            assert active is evaluator
            active.reset_policy()

        assert remote_cls.kill_calls == [(actor, (), {"no_restart": True})]

    def test_async_methods_are_not_exposed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        assert not hasattr(evaluator, "async_setup")
        assert not hasattr(evaluator, "async_evaluate_episode")
        assert not hasattr(evaluator, "async_reset_env")

    def test_public_execution_error_from_ray_task_is_unwrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[_execution_error()])
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert exc_info.value.result.episode_steps == 3
        assert exc_info.value.cause_type == "ValueError"
        assert exc_info.value.cause_message == "policy act failed"
        assert actor.restored_metric_states == [actor.metric_state_response]
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [1200.0, 120.0, 1200.0]

    def test_unsupported_generator_close_does_not_hide_episode_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)
        remote_gen = _UnsupportedCloseGenerator(
            [_failed_ref(ValueError("rollout failed"))]
        )
        actor.make_episode_evaluation = _RemoteMethod(
            lambda **kwargs: remote_gen
        )

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert exc_info.value.cause_type == "ValueError"
        assert exc_info.value.cause_message == "rollout failed"
        assert remote_gen.close_calls == 1
        assert actor.restored_metric_states == [actor.metric_state_response]
        assert remote_cls.get_timeouts == [1200.0, 120.0, 1200.0]

    def test_compute_failure_rolls_back_metric_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response=ValueError("compute failed"),
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert str(exc_info.value) == (
            "Failed to compute policy evaluation metrics. "
            "Cause: ValueError: compute failed"
        )
        assert exc_info.value.result.episode_steps == 5
        assert exc_info.value.cause_type == "ValueError"
        assert exc_info.value.cause_message == "compute failed"
        assert actor.restored_metric_states == [actor.metric_state_response]
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [
            1200.0,
            120.0,
            1200.0,
            1200.0,
        ]

    def test_compute_timeout_does_not_roll_back_metric_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response=GetTimeoutError("metrics timed out"),
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationRemoteTimeoutError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert (
            str(exc_info.value)
            == "Remote policy evaluator metrics compute timed out."
        )
        assert actor.restored_metric_states == []
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [
            1200.0,
            120.0,
            1200.0,
            1200.0,
        ]

    def test_metric_state_export_timeout_does_not_start_episode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response={"success_rate": 1.0},
            metric_state_response=GetTimeoutError("metric export timed out"),
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationRemoteTimeoutError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert (
            str(exc_info.value)
            == "Remote policy evaluator metric state export timed out."
        )
        assert actor.make_episode_evaluation.calls == []
        assert actor.restored_metric_states == []
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [1200.0, 1200.0]

    def test_keyboard_interrupt_closes_actor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[KeyboardInterrupt()])
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(KeyboardInterrupt):
            evaluator.evaluate_episode(max_steps=5)

        assert remote_cls.kill_calls == [(actor, (), {"no_restart": True})]
        assert getattr(evaluator, "_remote", None) is None
        with pytest.raises(PolicyEvaluationError, match="closed"):
            evaluator.evaluate_episode(max_steps=5)

    def test_stream_keyboard_interrupt_releases_guard_and_closes_actor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[KeyboardInterrupt()])
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(KeyboardInterrupt):
            list(evaluator.make_episode_evaluation(max_steps=5))

        assert evaluator._stream_active is False
        assert remote_cls.kill_calls == [(actor, (), {"no_restart": True})]
        with pytest.raises(PolicyEvaluationError, match="closed"):
            evaluator.reset_policy()

    def test_evaluate_episode_uses_rollout_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response={"success_rate": 1.0},
        )
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            rollout_timeout_s=12.0,
        )

        evaluator.evaluate_episode(max_steps=5)

        assert actor.evaluate_episode.calls == []
        assert len(actor.make_episode_evaluation.calls) == 1
        assert remote_cls.get_timeouts == [1200.0, 12.0, 1200.0]

    def test_public_methods_accept_per_call_timeout_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[2],
            compute_response={"success_rate": 1.0},
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        evaluator.evaluate_episode(max_steps=5, timeout_s=1.5)
        assert list(
            evaluator.make_episode_evaluation(max_steps=5, timeout_s=2.5)
        ) == [2]
        evaluator.reset_env(timeout_s=3.5)
        evaluator.reset_policy(timeout_s=4.5)
        evaluator.reset_metrics(timeout_s=5.5)
        evaluator.get_metrics(timeout_s=6.5)
        evaluator.compute_metrics(timeout_s=7.5)

        assert remote_cls.get_timeouts == [
            1200.0,
            1.5,
            1200.0,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            7.5,
        ]

    def test_invalid_per_call_timeout_is_rejected_before_remote_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response={"success_rate": 1.0},
        )
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(ValueError, match="timeout_s"):
            evaluator.evaluate_episode(max_steps=5, timeout_s=0)
        with pytest.raises(ValueError, match="timeout_s"):
            evaluator.reset_policy(timeout_s=-1.0)
        with pytest.raises(ValueError, match="timeout_s"):
            evaluator.compute_metrics(timeout_s=float("nan"))

        assert actor.evaluate_episode.calls == []
        assert actor.make_episode_evaluation.calls == []
        assert actor.reset_policy.calls == []
        assert actor.compute_metrics.calls == []

    def test_reset_methods_use_reset_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            reset_timeout_s=3.0,
        )

        evaluator.reset_env(seed=1)
        evaluator.reset_policy(mode="eval")
        evaluator.reset_metrics(clear=True)

        assert remote_cls.get_timeouts == [3.0, 3.0, 3.0]

    def test_reset_env_normalizes_legacy_kwargs_to_env_reset_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)
        reset_state = State(state={"env": "post-reset"})

        assert evaluator.reset_env(seed=1) == {"env_reset_input": {"seed": 1}}
        assert evaluator.reset_env(env_reset_input={"seed": 2}) == {
            "env_reset_input": {"seed": 2}
        }
        assert evaluator.reset_env(env_reset_input=reset_state) == {
            "env_reset_input": reset_state
        }
        assert evaluator.reset_env() == {"env_reset_input": None}

        assert [kwargs for _, kwargs in actor.reset_env.calls] == [
            {"env_reset_input": {"seed": 1}},
            {"env_reset_input": {"seed": 2}},
            {"env_reset_input": reset_state},
            {"env_reset_input": None},
        ]

    def test_reset_env_rejects_mixed_input_shapes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(ValueError, match="either env_reset_input"):
            evaluator.reset_env(env_reset_input={"seed": 1}, seed=2)

        assert actor.reset_env.calls == []

    def test_reset_timeout_maps_to_public_timeout_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        actor.reset_policy = _RemoteMethod(
            lambda **kwargs: _failed_ref(GetTimeoutError("reset timed out"))
        )
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            reset_timeout_s=3.0,
        )

        with pytest.raises(PolicyEvaluationRemoteTimeoutError):
            evaluator.reset_policy()

        assert remote_cls.get_timeouts == [3.0, 3.0]

    def test_successful_operations_do_not_auto_refresh_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[5],
            compute_response={"success_rate": 1.0},
        )
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        evaluator.setup(_ENV_CFG, _POLICY_CFG, _evaluator_metrics())
        actor._export_recovery_snapshot.calls.clear()

        evaluator.reconfigure_env("env-2")
        evaluator.reconfigure_policy("policy-2")
        evaluator.reconfigure_metrics(_evaluator_metrics())
        evaluator.reset_env(seed=1)
        evaluator.reset_policy()
        evaluator.reset_metrics()
        evaluator.evaluate_episode(max_steps=5)

        assert actor._export_recovery_snapshot.calls == []

    def test_reconfigure_env_forwards_force_recreate_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            reset_timeout_s=3.0,
        )

        evaluator.reconfigure_env(
            "env-2",
            force_recreate=False,
            timeout_s=1.5,
        )

        assert actor.reconfigure_env.calls == [
            (
                ("env-2",),
                {"force_recreate": False},
            )
        ]
        assert remote_cls.get_timeouts == [1.5]

    def test_actor_snapshot_public_methods_are_not_exposed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor()
        evaluator, _ = _make_remote_evaluator(monkeypatch, actor)

        assert not hasattr(evaluator, "save_remote_actor_snapshot")
        assert not hasattr(evaluator, "restore_remote_actor_snapshot")

    def test_timeout_maps_to_public_timeout_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[GetTimeoutError("episode timed out")]
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationRemoteTimeoutError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert (
            str(exc_info.value)
            == "Remote policy evaluator evaluate_episode timed out after "
            "2 consecutive 120.0s waits."
        )
        assert actor.restored_metric_states == []
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [1200.0, 120.0, 120.0]
        assert getattr(evaluator, "_remote", None) is actor

    def test_stream_next_ref_timeout_maps_to_public_timeout_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_stream_wait(monkeypatch)
        stream = _RayWaitStreamRefGenerator(
            [],
            complete_when_empty=False,
        )
        actor = _ScriptedActor(compute_response={"success_rate": 1.0})
        actor.make_episode_evaluation = _RemoteMethod(lambda **kwargs: stream)
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            rollout_timeout_s=0.01,
        )

        with pytest.raises(PolicyEvaluationRemoteTimeoutError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert (
            str(exc_info.value)
            == "Remote policy evaluator evaluate_episode timed out after "
            "2 consecutive 0.01s waits."
        )
        assert stream.wait_timeouts == [0.01, 0.01]
        assert stream.close_calls == 1
        assert actor.compute_metrics.calls == []
        assert actor.restored_metric_states == []
        assert actor.restored_snapshots == []
        assert remote_cls.get_timeouts == [1200.0]
        assert getattr(evaluator, "_remote", None) is actor

    def test_stream_next_ref_grace_wait_returns_completed_operation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_stream_wait(monkeypatch)
        stream = _RayWaitStreamRefGenerator(
            [_resolved_ref(5)],
            ready_script=[False, True],
        )
        actor = _ScriptedActor(compute_response={"success_rate": 1.0})
        actor.make_episode_evaluation = _RemoteMethod(lambda **kwargs: stream)
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            rollout_timeout_s=0.01,
        )

        assert evaluator.evaluate_episode(max_steps=5) == {"success_rate": 1.0}

        assert stream.wait_timeouts == [0.01, 0.01, 0.01]
        assert stream.close_calls == 1
        assert actor.evaluate_episode.calls == []
        assert len(actor.make_episode_evaluation.calls) == 1
        assert remote_cls.get_timeouts == [1200.0, 0.01, 1200.0]
        assert getattr(evaluator, "_remote", None) is actor

    def test_timeout_grace_wait_returns_completed_operation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[_TimeoutThenResolvedRef(5)],
            compute_response={"success_rate": 1.0},
        )
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        assert evaluator.evaluate_episode(max_steps=5) == {"success_rate": 1.0}

        assert actor.evaluate_episode.calls == []
        assert len(actor.make_episode_evaluation.calls) == 1
        assert remote_cls.get_timeouts == [1200.0, 120.0, 120.0, 1200.0]
        assert getattr(evaluator, "_remote", None) is actor

    def test_timeout_grace_wait_can_be_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(
            stream_response=[GetTimeoutError("episode timed out")]
        )
        evaluator, remote_cls = _make_remote_evaluator(
            monkeypatch,
            actor,
            timeout_grace_retries=0,
        )

        with pytest.raises(PolicyEvaluationRemoteTimeoutError):
            evaluator.evaluate_episode(max_steps=5)

        assert remote_cls.get_timeouts == [1200.0, 120.0]

    def test_worker_lost_does_not_clear_actor_handle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actor = _ScriptedActor(stream_response=[ActorDiedError()])
        evaluator, remote_cls = _make_remote_evaluator(monkeypatch, actor)

        with pytest.raises(PolicyEvaluationWorkerLostError):
            evaluator.evaluate_episode(max_steps=5)

        assert getattr(evaluator, "_remote", None) is actor
        assert remote_cls.kill_calls == []

        actor.stream_response = [5]
        actor.compute_response = {"success_rate": 1.0}
        assert evaluator.evaluate_episode(max_steps=5) == {"success_rate": 1.0}

    def test_as_remote_does_not_expose_transport_rollout_steps(self) -> None:
        cfg = PolicyEvaluatorConfig().as_remote()

        assert not hasattr(cfg, "transport_rollout_steps")

    def test_as_remote_uses_remote_timeout_defaults(self) -> None:
        cfg = PolicyEvaluatorConfig().as_remote()

        assert cfg.rollout_timeout_s == 120.0
        assert cfg.reset_timeout_s == 1200.0

    def test_as_remote_accepts_separate_reset_timeout(self) -> None:
        cfg = PolicyEvaluatorConfig().as_remote(
            rollout_timeout_s=11.0,
            reset_timeout_s=2.0,
            timeout_grace_retries=3,
        )

        assert cfg.rollout_timeout_s == 11.0
        assert cfg.reset_timeout_s == 2.0
        assert cfg.timeout_grace_retries == 3
