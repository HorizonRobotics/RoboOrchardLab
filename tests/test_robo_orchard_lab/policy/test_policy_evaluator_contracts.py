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

import pickle
from typing import Any, cast

import pytest
import torch
from ut_help import (
    DummyEnvConfig,
    DummyPolicy,
    DummyPolicyConfig,
    DummySuccessRateMetricConfig,
)

from robo_orchard_lab.envs.base import EnvStepReturn
from robo_orchard_lab.envs.state import ENV_STATE_SCOPE_KEY, EnvStateScope
from robo_orchard_lab.metrics import MetricDict, MetricDictConfig
from robo_orchard_lab.policy import (
    PolicyEvaluationExecutionError,
    evaluator as evaluator_pkg,
)
from robo_orchard_lab.policy.evaluator.base import (
    PolicyEvaluatorConfig,
    evaluate_rollout_stop_condition,
)
from robo_orchard_lab.policy.evaluator.contracts import (
    EpisodeResult,
    EvaluationRequest,
    EvaluationStatus,
    PreparedEnvStart,
    TerminalReason,
)
from robo_orchard_lab.policy.evaluator.episode import (
    _run_episode_loop as run_policy_episode_loop,
    evaluate_episode as evaluate_policy_episode,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.policy.evaluator.recovery import (
    PolicyEvaluatorRecoverySnapshot,
    _detach_runtime_state,
)
from robo_orchard_lab.utils.state import State, StateSaveLoadMixin


class _ConstantMetric:
    def __init__(self, value: Any) -> None:
        self.value = value

    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret

    def compute(self) -> Any:
        return self.value

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _RecoverableMetric(StateSaveLoadMixin):
    def __init__(self) -> None:
        self.last_reward = -1.0

    def reset(self, **kwargs) -> None:
        del kwargs
        self.last_reward = -1.0

    def update(self, action, step_ret) -> None:
        del action
        self.last_reward = step_ret[1]

    def compute(self) -> float:
        return self.last_reward

    def to(self, *args, **kwargs) -> None:
        del args, kwargs

    def _get_state(self) -> dict[str, object]:
        return {"last_reward": self.last_reward}


class _TaggedDummyPolicyConfig(DummyPolicyConfig):
    tag: int = 1


class _StatefulDummyEnv:
    supported_state_scopes = frozenset({EnvStateScope.POST_RESET})

    def __init__(self, success_step_count: int = 3) -> None:
        self.success_step_count = success_step_count
        self.step_count = 0
        self.reset_calls = 0
        self.reset_from_state_calls: list[State] = []
        self.actions: list[Any] = []

    def reset(self, **kwargs):
        del kwargs
        self.reset_calls += 1
        self.step_count = 0
        return {"obs": 0}, {"step": 0}

    def reset_from_state(self, state: State):
        self.reset_from_state_calls.append(state)
        self.step_count = int(state.state["step_count"])
        return {"obs": self.step_count}, {"step": self.step_count}

    def get_state(self) -> State:
        return State(
            state={
                ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value,
                "step_count": self.step_count,
            }
        )

    def load_state(self, state: State) -> None:
        self.step_count = int(state.state["step_count"])

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        reward = 1.0 if self.step_count >= self.success_step_count else 0.0
        return (
            {"obs": self.step_count, "act": action},
            reward,
            self.step_count >= self.success_step_count,
            False,
            {"step": self.step_count},
        )

    def close(self):
        pass

    @property
    def num_envs(self) -> int:
        return 1

    def rollout(self, *args, **kwargs):
        from robo_orchard_core.envs.rollout import rollout

        return rollout(self, *args, **kwargs)


class TestPolicyEvaluatorContracts:
    def test_runtime_state_helper_detaches_state_payload(self) -> None:
        state = State(state={"items": [1, {"nested": 2}]})

        detached = _detach_runtime_state(
            state,
            context="metric runtime state",
        )
        items = cast(list[object], cast(dict[str, Any], state.state)["items"])
        cast(dict[str, int], items[1])["nested"] = 3

        assert detached.state == {"items": [1, {"nested": 2}]}
        assert detached is not state

    def test_request_defaults_preserve_v1_contract(self) -> None:
        request = EvaluationRequest(max_steps=8)

        assert request.max_steps == 8
        assert request.rollout_steps == 5
        assert request.env_reset_input is None
        assert request.policy_reset_input is None
        assert (
            request.rollout_stop_condition is evaluate_rollout_stop_condition
        )

    def test_evaluation_request_accepts_reset_input(self) -> None:
        kwargs = {"seed": 3}

        request = EvaluationRequest(max_steps=8, env_reset_input=kwargs)
        kwargs["seed"] = 4

        assert request.env_reset_input == {"seed": 3}
        assert request.env_reset_input is not kwargs

    def test_evaluation_request_accepts_state(self) -> None:
        state = State(state={ENV_STATE_SCOPE_KEY: "post_reset"})

        request = EvaluationRequest(max_steps=8, env_reset_input=state)

        assert request.env_reset_input is state

    def test_prepared_env_start_wraps_reset_return(self) -> None:
        info = {"seed": 3}

        prepared = PreparedEnvStart.from_reset_return(({"obs": 7}, info))
        info["seed"] = 4

        assert prepared.observations == {"obs": 7}
        assert prepared.info == {"seed": 3}

    def test_evaluation_request_accepts_prepared_env_start(self) -> None:
        prepared = PreparedEnvStart(
            observations={"obs": 7},
            info={"seed": 3},
        )

        request = EvaluationRequest(max_steps=8, env_reset_input=prepared)

        assert request.env_reset_input is prepared

    def test_prepared_env_start_rejects_invalid_info(self) -> None:
        with pytest.raises(TypeError, match="PreparedEnvStart.info"):
            PreparedEnvStart(observations={}, info=object())  # type: ignore[arg-type]

    def test_evaluation_request_rejects_invalid_env_reset_input(self) -> None:
        with pytest.raises(TypeError, match="env_reset_input"):
            EvaluationRequest(max_steps=8, env_reset_input=object())  # type: ignore[arg-type]

    def test_evaluation_request_accepts_policy_reset_input(self) -> None:
        policy_reset_input = {"episode_id": 3}

        request = EvaluationRequest(
            max_steps=8,
            policy_reset_input=policy_reset_input,
        )
        policy_reset_input["episode_id"] = 4

        assert request.policy_reset_input == {"episode_id": 3}
        assert request.policy_reset_input is not policy_reset_input

    def test_evaluation_request_rejects_invalid_policy_reset_input(
        self,
    ) -> None:
        with pytest.raises(TypeError, match="policy_reset_input"):
            EvaluationRequest(max_steps=8, policy_reset_input=object())  # type: ignore[arg-type]

    def test_episode_result_captures_minimal_terminal_contract(self) -> None:
        result = EpisodeResult(
            status=EvaluationStatus.SUCCEEDED,
            terminal_reason=TerminalReason.TERMINATED,
            episode_steps=5,
            metrics={"success_rate": 1.0},
        )

        assert result.metrics == {"success_rate": 1.0}
        assert result.status is EvaluationStatus.SUCCEEDED
        assert result.terminal_reason is TerminalReason.TERMINATED
        assert result.episode_steps == 5

    def test_contracts_remain_internal_to_package_root(self) -> None:
        assert not hasattr(evaluator_pkg, "EvaluationRequest")
        assert not hasattr(evaluator_pkg, "EpisodeResult")
        assert not hasattr(evaluator_pkg, "PreparedEnvStart")

    def test_execution_error_round_trips_through_pickle(self) -> None:
        result = EpisodeResult(
            status=EvaluationStatus.FAILED,
            terminal_reason=TerminalReason.ERROR,
            episode_steps=3,
            metrics={},
        )
        error = PolicyEvaluationExecutionError(
            "episode failed",
            result=result,
            cause_type="ValueError",
            cause_message="bad action",
        )

        round_tripped = pickle.loads(pickle.dumps(error))

        assert isinstance(round_tripped, PolicyEvaluationExecutionError)
        assert str(round_tripped) == (
            "episode failed Cause: ValueError: bad action"
        )
        assert round_tripped.result == result
        assert round_tripped.cause_type == "ValueError"
        assert round_tripped.cause_message == "bad action"

    def test_runtime_evaluate_episode_preserves_episode_result_contract(
        self,
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric_dict(
                MetricDictConfig(
                    {"success_rate": DummySuccessRateMetricConfig()}
                )()
            ),
        )
        result = evaluate_policy_episode(
            evaluator,
            EvaluationRequest(max_steps=20),
        )

        assert result == EpisodeResult(
            status=EvaluationStatus.SUCCEEDED,
            terminal_reason=TerminalReason.TERMINATED,
            episode_steps=5,
            metrics={"success_rate": 1.0},
        )

    def test_runtime_evaluate_episode_can_start_env_from_state(self) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        env = _StatefulDummyEnv(success_step_count=3)
        evaluator.env = env  # type: ignore[assignment]
        state = State(
            state={
                ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value,
                "step_count": 1,
            }
        )

        result = evaluate_policy_episode(
            evaluator,
            EvaluationRequest(max_steps=5, env_reset_input=state),
        )

        assert env.reset_calls == 0
        assert env.reset_from_state_calls == [state]
        assert result.episode_steps == 2
        assert result.metrics == {"success_rate": 1.0}

    def test_runtime_evaluate_episode_can_use_prepared_env_start(
        self,
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        env = _StatefulDummyEnv(success_step_count=2)
        evaluator.env = env  # type: ignore[assignment]
        prepared = PreparedEnvStart(
            observations={"obs": 7},
            info={"seed": 3},
        )

        result = evaluate_policy_episode(
            evaluator,
            EvaluationRequest(
                max_steps=5,
                env_reset_input=prepared,
                policy_reset_input={"episode_id": 123},
            ),
        )

        assert env.reset_calls == 0
        assert env.reset_from_state_calls == []
        assert env.actions[0]["obs"] == 7
        assert cast(DummyPolicy, evaluator.policy).last_reset_kwargs == {
            "episode_id": 123
        }
        assert result.episode_steps == 2
        assert result.metrics == {"success_rate": 1.0}

    def test_runtime_stream_can_start_env_from_state(self) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        env = _StatefulDummyEnv(success_step_count=3)
        evaluator.env = env  # type: ignore[assignment]
        state = State(
            state={
                ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value,
                "step_count": 1,
            }
        )

        steps = list(
            run_policy_episode_loop(
                evaluator,
                EvaluationRequest(
                    max_steps=5,
                    rollout_steps=1,
                    env_reset_input=state,
                ),
            )
        )

        assert env.reset_calls == 0
        assert env.reset_from_state_calls == [state]
        assert steps == [1, 1]

    def test_runtime_stream_can_use_prepared_env_start(self) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        env = _StatefulDummyEnv(success_step_count=2)
        evaluator.env = env  # type: ignore[assignment]

        steps = list(
            run_policy_episode_loop(
                evaluator,
                EvaluationRequest(
                    max_steps=5,
                    rollout_steps=1,
                    env_reset_input=PreparedEnvStart(
                        observations={"obs": 11},
                        info={"seed": 5},
                    ),
                ),
            )
        )

        assert env.reset_calls == 0
        assert env.actions[0]["obs"] == 11
        assert steps == [1, 1]

    def test_reset_env_uses_env_reset_input_contract(self) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        env = _StatefulDummyEnv(success_step_count=3)
        evaluator.env = env  # type: ignore[assignment]
        state = State(
            state={
                ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value,
                "step_count": 1,
            }
        )

        assert evaluator.reset_env(env_reset_input={"seed": 1}) == (
            {"obs": 0},
            {"step": 0},
        )
        assert env.reset_calls == 1

        assert evaluator.reset_env(env_reset_input=state) == (
            {"obs": 1},
            {"step": 1},
        )
        assert env.reset_from_state_calls == [state]

        with pytest.raises(TypeError, match="env_reset_input"):
            evaluator.reset_env(env_reset_input=PreparedEnvStart({}))

        with pytest.raises(TypeError, match="either env_reset_input"):
            evaluator.reset_env(env_reset_input=state, seed=1)

    def test_setup_accepts_evaluator_metrics_and_preserves_surface(
        self,
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        metric = _RecoverableMetric()
        evaluator_metrics = EvaluatorMetrics.from_metric(
            metric,
            name="reward",
        )

        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=evaluator_metrics,
        )

        assert evaluator.metrics is evaluator_metrics
        assert evaluator.get_metrics() is evaluator_metrics
        assert evaluator.get_metrics() is not None
        assert evaluator.get_metrics().get_metric("reward") is metric
        assert evaluator.compute_metrics() == {"reward": -1.0}
        assert evaluator.evaluate_episode(max_steps=20) == {"reward": 1.0}

    def test_runtime_uses_terminated_reason_for_custom_early_stop(
        self, monkeypatch
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            ),
        )
        monkeypatch.setattr(
            evaluator.env,
            "step",
            lambda action: EnvStepReturn(
                observations={"obs": 1, "act": action},
                rewards=1.0,
                terminated=None,
                truncated=None,
                info={"step": 1},
            ),
        )
        result = evaluate_policy_episode(
            evaluator,
            EvaluationRequest(
                max_steps=5,
                rollout_stop_condition=lambda _: True,
            ),
        )

        assert result == EpisodeResult(
            status=EvaluationStatus.SUCCEEDED,
            terminal_reason=TerminalReason.TERMINATED,
            episode_steps=1,
            metrics={"success_rate": 1.0},
        )

    def test_evaluator_exports_and_restores_one_recovery_snapshot(
        self,
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        metric = _RecoverableMetric()
        evaluator_metrics = EvaluatorMetrics.from_metric(
            metric,
            name="reward",
        )
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=evaluator_metrics,
        )
        policy = cast(DummyPolicy, evaluator.policy)
        policy.data = torch.tensor([1.0, 2.0, 3.0])
        evaluator.evaluate_episode(max_steps=20)

        snapshot = evaluator._export_recovery_snapshot()

        assert isinstance(snapshot, PolicyEvaluatorRecoverySnapshot)
        assert isinstance(snapshot.policy_runtime_state, State)
        assert isinstance(snapshot.metric_runtime_state, State)
        assert snapshot.metric_runtime_state.class_type is MetricDict
        assert not hasattr(evaluator, "get_policy_runtime_state")
        assert not hasattr(evaluator, "load_policy_runtime_state")

        policy.data = None
        metric.load_state(State(state={"last_reward": -5.0}))

        evaluator._restore_recovery_snapshot(snapshot)

        restored_policy = cast(DummyPolicy, evaluator.policy)
        restored_data = restored_policy.data
        assert restored_data is not None
        assert torch.equal(restored_data, torch.tensor([1.0, 2.0, 3.0]))
        assert evaluator.compute_metrics() == {"reward": 1.0}

    def test_recovery_snapshot_freezes_env_and_policy_inputs_on_export(
        self,
    ) -> None:
        evaluator = PolicyEvaluatorConfig()()
        env_cfg = DummyEnvConfig(success_step_count=5)
        policy_cfg = _TaggedDummyPolicyConfig(tag=1)
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric(
                _RecoverableMetric(),
                name="reward",
            ),
        )

        env_cfg.success_step_count = 7
        policy_cfg.tag = 2
        snapshot = evaluator._export_recovery_snapshot()
        env_cfg.success_step_count = 9
        policy_cfg.tag = 3

        assert isinstance(snapshot.env_cfg, DummyEnvConfig)
        assert snapshot.env_cfg.success_step_count == 7
        restored_policy_cfg = snapshot.policy_recovery_input
        assert isinstance(restored_policy_cfg, _TaggedDummyPolicyConfig)
        assert restored_policy_cfg.tag == 2
