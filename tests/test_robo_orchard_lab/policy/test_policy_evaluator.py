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
from typing import Any, cast

import pytest
import torch
from ut_help import (
    DummyEnvConfig,
    DummyPolicy,
    DummyPolicyConfig,
    DummyPolicyDataMetric,
    DummySuccessRateMetricConfig,
)

from robo_orchard_lab.envs.base import EnvStepReturn
from robo_orchard_lab.metrics import (
    MetricDict,
    MetricDictConfig,
)
from robo_orchard_lab.policy import (
    PolicyEvaluationError,
    PolicyEvaluationExecutionError,
    PolicyEvaluatorConfig,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)


class _FailingMetric:
    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret
        raise ValueError("metric update failed")

    def compute(self) -> dict[str, float]:
        return {}

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _ComputeFailingMetric:
    def reset(self, **kwargs) -> None:
        del kwargs

    def update(self, action, step_ret) -> None:
        del action, step_ret

    def compute(self) -> dict[str, float]:
        raise ValueError("metric compute failed")

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


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


class _CountingStepMetric:
    def __init__(self) -> None:
        self.count = 0

    def reset(self, **kwargs) -> None:
        del kwargs
        self.count = 0

    def update(self, action, step_ret) -> None:
        del action, step_ret
        self.count += 1

    def compute(self) -> int:
        return self.count

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _ClosableMetric(_ConstantMetric):
    def __init__(self, value: Any) -> None:
        super().__init__(value)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _LegacyTerminalDispatchMetric:
    def __init__(self) -> None:
        self.update_rewards: list[float] = []

    def reset(self, **kwargs) -> None:
        del kwargs
        self.update_rewards = []

    def update(self, action, step_ret) -> None:
        del action
        self.update_rewards.append(step_ret[1])

    def compute(self) -> list[float]:
        return list(self.update_rewards)

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _TerminalDispatchMetric(_LegacyTerminalDispatchMetric):
    pass


class _StepDispatchMetric:
    def __init__(self) -> None:
        self.step_rewards: list[float] = []

    def reset(self, **kwargs) -> None:
        del kwargs
        self.step_rewards = []

    def update(self, action, step_ret) -> None:
        del action
        self.step_rewards.append(step_ret[1])

    def compute(self) -> list[float]:
        return list(self.step_rewards)

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


class _StepSummaryMetric:
    def __init__(self) -> None:
        self.episode_summary: dict[str, object] | None = None
        self.action_count = 0
        self.step_count = 0

    def reset(self, **kwargs) -> None:
        del kwargs
        self.episode_summary = None
        self.action_count = 0
        self.step_count = 0

    def update(self, action, step_ret) -> None:
        del action
        self.action_count += 1
        self.step_count += 1
        terminated = bool(step_ret[2])
        truncated = bool(step_ret[3])
        if truncated:
            terminal_reason = "TRUNCATED"
        elif terminated:
            terminal_reason = "TERMINATED"
        else:
            terminal_reason = "IN_PROGRESS"
        self.episode_summary = {
            "action_count": self.action_count,
            "step_count": self.step_count,
            "last_reward": step_ret[1],
            "terminal_reason": terminal_reason,
        }

    def compute(self) -> dict[str, object] | None:
        return self.episode_summary

    def to(self, *args, **kwargs) -> None:
        del args, kwargs


def _make_local_evaluator(
    metrics: EvaluatorMetrics | None = None,
):
    env_cfg = DummyEnvConfig()
    policy_cfg = DummyPolicyConfig()
    if metrics is None:
        metrics = EvaluatorMetrics.from_metric_dict(
            MetricDictConfig(
                {
                    "success_rate": DummySuccessRateMetricConfig(),
                }
            )()
        )

    evaluator = PolicyEvaluatorConfig()()
    evaluator.setup(
        env_cfg=env_cfg,
        policy_or_cfg=policy_cfg,
        metrics=metrics,
    )
    return evaluator


class TestPolicyEvaluator:
    @pytest.mark.parametrize(
        ("metrics_factory", "match"),
        [
            (
                lambda: EvaluatorMetrics.from_channels(
                    terminal={"score": _TerminalDispatchMetric()},
                    step={"score": _StepDispatchMetric()},
                ),
                "unique",
            ),
            (
                lambda: cast(
                    Any,
                    _TerminalDispatchMetric(),
                ),
                "EvaluatorMetrics",
            ),
        ],
    )
    def test_reconfigure_metrics_validates_capabilities_at_config_boundary(
        self,
        metrics_factory,
        match: str,
    ) -> None:
        evaluator = _make_local_evaluator()
        old_metrics = evaluator.metrics

        with pytest.raises((TypeError, ValueError), match=match):
            evaluator.reconfigure_metrics(metrics_factory())

        assert evaluator.metrics is old_metrics

    def test_reconfigure_metrics_accepts_evaluator_metrics(
        self,
    ) -> None:
        evaluator = _make_local_evaluator()
        metric = _TerminalDispatchMetric()
        evaluator_metrics = EvaluatorMetrics.from_metric(
            metric,
            name="terminal",
        )

        evaluator.reconfigure_metrics(evaluator_metrics)

        assert evaluator.metrics is evaluator_metrics
        assert evaluator.get_metrics() is evaluator_metrics
        assert evaluator.get_metrics() is not None
        assert evaluator.get_metrics().get_metric("terminal") is metric
        assert evaluator.compute_metrics() == {"terminal": []}
        assert evaluator.evaluate_episode(max_steps=20) == {"terminal": [1.0]}

    def test_reconfigure_metrics_uses_configured_terminal_timing(self) -> None:
        metric = _TerminalDispatchMetric()
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(metric, name="terminal")
        )

        assert evaluator.compute_metrics() == {"terminal": []}

        evaluator.evaluate_episode(max_steps=20)
        evaluator.evaluate_episode(max_steps=20)

        assert metric.update_rewards == [1.0, 1.0]

    def test_metric_dict_accepts_supported_nonterminal_metrics(self) -> None:
        step_metric = _StepDispatchMetric()
        summary_metric = _StepSummaryMetric()

        incremental_metrics = MetricDict()
        incremental_metrics["step"] = step_metric
        incremental_metrics["summary"] = summary_metric

        constructed_metrics = MetricDict(
            {
                "step": _StepDispatchMetric(),
                "summary": _StepSummaryMetric(),
            }
        )

        assert incremental_metrics["step"] is step_metric
        assert incremental_metrics["summary"] is summary_metric
        assert set(constructed_metrics.keys()) == {"step", "summary"}

    def test_metric_dict_update_fans_out_to_metric_protocol(self) -> None:
        metrics = MetricDict({"step": _StepDispatchMetric()})

        metrics.update(
            None,
            (None, 0.0, False, False, {}),
        )

        assert cast(_StepDispatchMetric, metrics["step"]).step_rewards == [0.0]

    def test_evaluator_metrics_reject_duplicate_live_metric_instances(
        self,
    ) -> None:
        metric = _TerminalDispatchMetric()
        with pytest.raises(
            ValueError,
            match="cannot be registered under multiple evaluator metric names",
        ):
            EvaluatorMetrics.from_channels(
                terminal={"left": metric, "right": metric}
            )

    def test_evaluate_episode_dispatches_metrics_by_primary_update_timing(
        self,
    ) -> None:
        legacy_metric = _LegacyTerminalDispatchMetric()
        terminal_metric = _TerminalDispatchMetric()
        step_metric = _StepDispatchMetric()
        summary_metric = _StepSummaryMetric()
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_channels(
                terminal={
                    "legacy_terminal": legacy_metric,
                    "terminal": terminal_metric,
                },
                step={
                    "step": step_metric,
                    "summary": summary_metric,
                },
            )
        )

        metrics = evaluator.evaluate_episode(max_steps=20)

        assert metrics["legacy_terminal"] == [1.0]
        assert metrics["terminal"] == [1.0]
        assert metrics["step"] == [0.0, 0.0, 0.0, 0.0, 1.0]
        assert metrics["summary"] == {
            "action_count": 5,
            "step_count": 5,
            "last_reward": 1.0,
            "terminal_reason": "TERMINATED",
        }

    def test_evaluate_episode_returns_metrics_dict(self) -> None:
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        eval_cfg = PolicyEvaluatorConfig()

        evaluator = eval_cfg()
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric_dict(metric_cfg()),
        )

        assert evaluator.evaluate_episode(max_steps=20) == {
            "success_rate": 1.0
        }

    def test_make_episode_evaluation_requires_setup(self) -> None:
        evaluator = PolicyEvaluatorConfig()()

        with pytest.raises(PolicyEvaluationError, match="not ready"):
            next(evaluator.make_episode_evaluation(max_steps=1))

    def test_accessors_require_setup(self) -> None:
        evaluator = PolicyEvaluatorConfig()()

        assert evaluator.get_metrics() is None

        with pytest.raises(
            RuntimeError,
            match="Environment is not configured",
        ):
            _ = evaluator.env

        with pytest.raises(
            RuntimeError,
            match="Policy is not configured",
        ):
            _ = evaluator.policy

        with pytest.raises(
            RuntimeError,
            match="Metrics are not configured",
        ):
            _ = evaluator.metrics

    def test_reset_env_accepts_legacy_kwargs(self) -> None:
        evaluator = _make_local_evaluator()
        reset_calls: list[dict[str, Any]] = []

        def _reset(**kwargs):
            reset_calls.append(dict(kwargs))
            return {"obs": 0}, {"step": 0}

        evaluator.env.reset = _reset  # type: ignore[method-assign]

        assert evaluator.reset_env(seed=3) == ({"obs": 0}, {"step": 0})
        assert reset_calls == [{"seed": 3}]

    def test_close_releases_env_once_and_detaches_policy_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        metric = _ClosableMetric(1.0)
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(metric, name="metric")
        )
        close_calls = {"env": 0, "policy": 0}

        def _close_env() -> None:
            close_calls["env"] += 1

        def _close_policy() -> None:
            close_calls["policy"] += 1

        monkeypatch.setattr(evaluator.env, "close", _close_env)
        monkeypatch.setattr(evaluator.policy, "close", _close_policy)

        evaluator.close()
        evaluator.close()

        assert close_calls == {"env": 1, "policy": 0}
        assert metric.close_calls == 0
        assert evaluator.get_metrics() is None

    def test_close_detaches_borrowed_policy_and_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        policy = DummyPolicyConfig()()
        metric = _ClosableMetric(1.0)
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=policy,
            metrics=EvaluatorMetrics.from_metric(metric, name="metric"),
        )
        close_calls = {"env": 0, "policy": 0}

        def _close_env() -> None:
            close_calls["env"] += 1

        def _close_policy() -> None:
            close_calls["policy"] += 1

        monkeypatch.setattr(evaluator.env, "close", _close_env)
        monkeypatch.setattr(policy, "close", _close_policy)

        evaluator.close()

        assert close_calls == {"env": 1, "policy": 0}
        assert metric.close_calls == 0
        assert evaluator.get_metrics() is None

    def test_context_manager_closes_evaluator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator()
        close_calls = 0

        def _close_env() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(evaluator.env, "close", _close_env)

        with evaluator as active:
            assert active is evaluator

        assert close_calls == 1
        assert evaluator.get_metrics() is None

    def test_reconfigure_closes_replaced_env_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        old_metric = _ClosableMetric(0.0)
        new_metric = _ClosableMetric(1.0)
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(old_metric, name="metric")
        )
        close_calls = {"env": 0, "policy": 0}

        def _close_env() -> None:
            close_calls["env"] += 1

        def _close_policy() -> None:
            close_calls["policy"] += 1

        monkeypatch.setattr(evaluator.env, "close", _close_env)
        monkeypatch.setattr(evaluator.policy, "close", _close_policy)
        evaluator.reconfigure_metrics(
            EvaluatorMetrics.from_metric(new_metric, name="metric")
        )
        evaluator.reconfigure_env(DummyEnvConfig())
        evaluator.reconfigure_policy(DummyPolicyConfig())

        assert close_calls == {"env": 1, "policy": 0}
        assert old_metric.close_calls == 0
        assert new_metric.close_calls == 0

    def test_reconfigure_env_can_reuse_same_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator()
        original_env = evaluator.env
        close_calls = 0

        def _close_env() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(original_env, "close", _close_env)

        evaluator.reconfigure_env(
            DummyEnvConfig(),
            force_recreate=False,
        )

        assert evaluator.env is original_env
        assert close_calls == 0

        evaluator.reconfigure_env(
            DummyEnvConfig(success_step_count=6),
            force_recreate=False,
        )

        assert evaluator.env is not original_env
        assert close_calls == 1

    def test_reconfigure_env_reuse_checks_runtime_env_cfg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator()
        original_env = evaluator.env
        original_env.cfg = DummyEnvConfig(success_step_count=6)
        close_calls = 0

        def _close_env() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(original_env, "close", _close_env)

        evaluator.reconfigure_env(
            DummyEnvConfig(),
            force_recreate=False,
        )

        assert evaluator.env is not original_env
        assert close_calls == 1

    def test_reconfigure_env_none_follows_config_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = PolicyEvaluatorConfig(
            reconfigure_env_force_recreate=False,
        )()
        evaluator.setup(
            env_cfg=DummyEnvConfig(),
            policy_or_cfg=DummyPolicyConfig(),
            metrics=EvaluatorMetrics.from_metric(
                DummySuccessRateMetricConfig()(),
                name="success_rate",
            ),
        )
        original_env = evaluator.env
        close_calls = 0

        def _close_env() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(original_env, "close", _close_env)

        evaluator.reconfigure_env(DummyEnvConfig())

        assert evaluator.env is original_env
        assert close_calls == 0

    def test_env_finalizer_tracks_env_changes_only(self) -> None:
        evaluator = _make_local_evaluator()
        env_finalizer = evaluator._env_finalizer

        assert env_finalizer is not None
        assert env_finalizer.alive

        evaluator.reconfigure_metrics(
            EvaluatorMetrics.from_metric(_ClosableMetric(1.0), name="metric")
        )
        evaluator.reconfigure_policy(DummyPolicyConfig())

        assert evaluator._env_finalizer is env_finalizer

        evaluator.reconfigure_env(DummyEnvConfig())

        assert evaluator._env_finalizer is not env_finalizer
        assert evaluator._env_finalizer is not None
        assert evaluator._env_finalizer.alive

        evaluator.close()

        assert evaluator._env_finalizer is None

    def test_evaluate_passes_reset_input_to_policy(self) -> None:
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        evaluator = PolicyEvaluatorConfig()()
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric_dict(metric_cfg()),
        )

        for _ in evaluator.make_episode_evaluation(
            max_steps=1,
            rollout_steps=1,
            policy_reset_input={"episode_id": 123},
        ):
            pass

        assert cast(DummyPolicy, evaluator.policy).last_reset_kwargs == {
            "episode_id": 123
        }

    def test_evaluate_init(self) -> None:
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        eval_cfg = PolicyEvaluatorConfig()

        evaluator = eval_cfg.__call__()
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric_dict(metric_cfg()),
        )
        # print(evaluator.compute_metrics())
        metric_res = evaluator.compute_metrics()
        assert metric_res == {"success_rate": 0.0}

    @pytest.mark.parametrize("rollout_steps", [1, 3])
    def test_evaluate_eval(self, rollout_steps: int) -> None:
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
            }
        )
        eval_cfg = PolicyEvaluatorConfig(
            # env_cfg=env_cfg,
            # policy_cfg=policy_cfg,
            # metrics=metric_cfg,
        )

        evaluator = eval_cfg.__call__()
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric_dict(metric_cfg()),
        )
        total_step = 0
        for step in evaluator.make_episode_evaluation(
            max_steps=20,
            rollout_steps=rollout_steps,
        ):
            print(step)
            # assert step == 1
            total_step += step

        assert total_step == 5
        assert evaluator.compute_metrics() == {"success_rate": 1.0}

        evaluator.reset_metrics()
        assert evaluator.compute_metrics() == {"success_rate": 0.0}

    def test_evaluate_episode_raises_execution_error_for_empty_rollout(
        self,
    ) -> None:
        evaluator = _make_local_evaluator()

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=0)

        assert exc_info.value.result.episode_steps == 0
        assert exc_info.value.result.metrics == {}
        assert exc_info.value.result.status.name == "FAILED"
        assert exc_info.value.result.terminal_reason.name == "EMPTY_ROLLOUT"

    def test_evaluate_episode_normalizes_metrics_compute_failure(
        self,
    ) -> None:
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(
                _ComputeFailingMetric(),
                name="metric",
            )
        )

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert exc_info.value.result.status.name == "FAILED"
        assert exc_info.value.result.terminal_reason.name == "ERROR"
        assert exc_info.value.result.episode_steps == 5
        assert exc_info.value.result.metrics == {}
        assert exc_info.value.__cause__ is not None
        assert exc_info.value.cause_type == "ValueError"
        assert exc_info.value.cause_message == "metric compute failed"
        assert "ValueError: metric compute failed" in str(exc_info.value)

    def test_evaluate_episode_restores_metric_state_after_failed_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_channels(
                step={"counter": _CountingStepMetric()},
            )
        )
        original_step = evaluator.env.step
        step_count = 0

        def _fail_on_second_step(action):
            nonlocal step_count
            step_count += 1
            if step_count == 1:
                return original_step(action)
            raise ValueError("env step failed")

        monkeypatch.setattr(evaluator.env, "step", _fail_on_second_step)

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            evaluator.evaluate_episode(max_steps=5)

        assert evaluator.compute_metrics() == {"counter": 0}
        assert exc_info.value.cause_type == "ValueError"
        assert exc_info.value.cause_message == "env step failed"

    def test_evaluate_episode_finalizes_env_once_on_success(self) -> None:
        evaluator = _make_local_evaluator()

        assert evaluator.env.finalize_episode_calls == 0

        evaluator.evaluate_episode(max_steps=20)

        assert evaluator.env.finalize_episode_calls == 1

    def test_evaluate_episode_finalizes_env_when_step_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator()

        monkeypatch.setattr(
            evaluator.env,
            "step",
            lambda action: (_ for _ in ()).throw(
                ValueError("env step failed")
            ),
        )

        with pytest.raises(PolicyEvaluationExecutionError):
            evaluator.evaluate_episode(max_steps=5)

        assert evaluator.env.finalize_episode_calls == 1

    def test_evaluate_episode_finalizes_env_when_reset_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator = _make_local_evaluator()

        monkeypatch.setattr(
            evaluator.env,
            "reset",
            lambda **kwargs: (_ for _ in ()).throw(
                ValueError("env reset failed")
            ),
        )

        with pytest.raises(PolicyEvaluationExecutionError):
            evaluator.evaluate_episode(max_steps=5)

        assert evaluator.env.finalize_episode_calls == 1

    def test_make_episode_evaluation_finalizes_env_on_complete_stream(
        self,
    ) -> None:
        evaluator = _make_local_evaluator()

        assert list(
            evaluator.make_episode_evaluation(max_steps=20, rollout_steps=2)
        ) == [2, 2, 1]

        assert evaluator.env.finalize_episode_calls == 1

    def test_make_episode_evaluation_finalizes_env_on_generator_close(
        self,
    ) -> None:
        evaluator = _make_local_evaluator()
        episode_iter = evaluator.make_episode_evaluation(
            max_steps=20,
            rollout_steps=1,
        )

        assert next(episode_iter) == 1
        episode_iter.close()

        assert evaluator.env.finalize_episode_calls == 1

    def test_evaluate_episode_succeeds_with_tensor_terminal_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(
                _ConstantMetric(1.0),
                name="success_rate",
            )
        )
        monkeypatch.setattr(
            evaluator.env,
            "step",
            lambda action: EnvStepReturn(
                observations={"obs": 1, "act": action},
                rewards=1.0,
                terminated=torch.tensor([True, False]),
                truncated=torch.tensor([False, False]),
                info={"step": 1},
            ),
        )

        assert evaluator.evaluate_episode(max_steps=5) == {"success_rate": 1.0}

    @pytest.mark.parametrize(
        "failure_site",
        [
            "env_reset",
            "policy_reset",
            "policy_act",
            "env_step",
            "metrics_update",
        ],
    )
    def test_make_episode_evaluation_normalizes_sync_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_site: str,
    ) -> None:
        evaluator = _make_local_evaluator(
            metrics=EvaluatorMetrics.from_metric(
                _FailingMetric(),
                name="metric",
            )
        )

        if failure_site == "env_reset":
            monkeypatch.setattr(
                evaluator.env,
                "reset",
                lambda **kwargs: (_ for _ in ()).throw(
                    ValueError("env reset failed")
                ),
            )
        elif failure_site == "policy_reset":
            monkeypatch.setattr(
                evaluator.policy,
                "reset",
                lambda **kwargs: (_ for _ in ()).throw(
                    ValueError("policy reset failed")
                ),
            )
        elif failure_site == "policy_act":
            monkeypatch.setattr(
                evaluator.policy,
                "act",
                lambda obs: (_ for _ in ()).throw(
                    ValueError("policy act failed")
                ),
            )
        elif failure_site == "env_step":
            monkeypatch.setattr(
                evaluator.env,
                "step",
                lambda action: (_ for _ in ()).throw(
                    ValueError("env step failed")
                ),
            )

        with pytest.raises(PolicyEvaluationExecutionError) as exc_info:
            list(
                evaluator.make_episode_evaluation(
                    max_steps=1,
                    rollout_steps=1,
                )
            )

        assert exc_info.value.result.status.name == "FAILED"
        assert exc_info.value.result.terminal_reason.name == "ERROR"
        assert exc_info.value.__cause__ is not None
        assert exc_info.value.cause_type == "ValueError"
        assert "failed" in (exc_info.value.cause_message or "")

    def test_evaluate_reconfig_policy(self) -> None:
        env_cfg = DummyEnvConfig()
        policy_cfg = DummyPolicyConfig()
        metric_cfg = MetricDictConfig(
            {
                "success_rate": DummySuccessRateMetricConfig(),
                "policy_data": DummySuccessRateMetricConfig(
                    class_type=DummyPolicyDataMetric
                ),
            }
        )
        eval_cfg = PolicyEvaluatorConfig()

        evaluator = eval_cfg.__call__()
        evaluator.setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_cfg,
            metrics=EvaluatorMetrics.from_metric_dict(metric_cfg()),
        )
        total_step = 0
        for step in evaluator.make_episode_evaluation(
            max_steps=20,
            rollout_steps=3,
        ):
            total_step += step

        assert total_step == 5
        # print("metrics:", evaluator.compute_metrics())
        metric_res = evaluator.compute_metrics()
        assert metric_res["policy_data"]["action"]["data"] is None

        new_policy = policy_cfg()
        new_policy.data = torch.tensor([1.0, 2.0, 3.0])
        evaluator.reconfigure_policy(new_policy)
        total_step = 0
        for step in evaluator.make_episode_evaluation(
            max_steps=20,
            rollout_steps=3,
        ):
            total_step += step

        assert total_step == 5
        metric_res = evaluator.compute_metrics()
        assert torch.equal(
            metric_res["policy_data"]["action"]["data"], new_policy.data
        )
