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
import inspect
import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields
from typing import Any, ClassVar

import pytest
from robo_orchard_core.envs.rollout import EnvRolloutReturn
from robo_orchard_core.utils.config import ClassType
from ut_help import DummyPolicyConfig, DummySuccessRateMetricConfig

import robo_orchard_lab.policy.evaluator.benchmark.backend as benchmark_backend
from robo_orchard_lab.envs.base import EnvBase, EnvBaseCfg
from robo_orchard_lab.policy.evaluator.base import (
    PolicyEvaluationRemoteTimeoutError,
    PolicyEvaluationWorkerLostError,
    PolicyEvaluatorConfig,
)
from robo_orchard_lab.policy.evaluator.benchmark import (
    BenchmarkAttemptError,
    BenchmarkAttemptRequest,
    BenchmarkDriver,
    BenchmarkEpisode,
    BenchmarkEpisodeRecord,
    BenchmarkEvaluateFailedEvent,
    BenchmarkEvaluateSucceededEvent,
    BenchmarkPrepareFailedEvent,
    BenchmarkPrepareJob,
    BenchmarkPrepareSucceededEvent,
    BenchmarkResult,
    BenchmarkTerminalEvent,
    LocalBenchmarkBackend,
    LocalBenchmarkBackendConfig,
    RemoteBenchmarkBackend,
    RemoteBenchmarkBackendConfig,
)
from robo_orchard_lab.policy.evaluator.contracts import PreparedEnvStart
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)


class _NoopMetric:
    def reset(self, **kwargs: Any) -> None:
        del kwargs

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def compute(self) -> dict[str, Any]:
        return {}

    def to(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _ResetError(RuntimeError):
    def __init__(self, message: str, reset_info: dict[str, Any]) -> None:
        super().__init__(message)
        self.reset_info = reset_info


class _RecordingLogger:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.exceptions: list[str] = []

    def debug(self, message: str, *args: object) -> None:
        self.debugs.append(message % args if args else message)

    def info(self, message: str, *args: object) -> None:
        self.infos.append(message % args if args else message)

    def exception(self, message: str, *args: object) -> None:
        self.exceptions.append(message % args if args else message)


class _LocalRecordingEnv(EnvBase):
    instances: ClassVar[list["_LocalRecordingEnv"]] = []

    def __init__(self, cfg: "_LocalRecordingEnvConfig") -> None:
        self.cfg = cfg
        self.reset_calls: list[dict[str, Any]] = []
        self.rollout_init_obs: list[Any] = []
        self.closed = False
        self.instances.append(self)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_calls.append(dict(kwargs))
        if self.cfg.reset_failure_message is not None:
            raise _ResetError(
                self.cfg.reset_failure_message,
                {"offset_seed": self.cfg.offset_seed},
            )
        return (
            {"obs": kwargs.get("job")},
            {
                "job": kwargs.get("job"),
                "offset_seed": self.cfg.offset_seed,
            },
        )

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return {"obs": action}, 1.0, True, False, {}

    def rollout(
        self,
        max_steps: int,
        init_obs: Any,
        policy: Any = None,
        env_step_callback: Callable[[Any, Any], None] | None = None,
        terminal_condition: Callable[[Any], bool] | None = None,
        keep_last_results: int = -1,
    ) -> EnvRolloutReturn[Any, Any]:
        del max_steps, policy, keep_last_results
        self.rollout_init_obs.append(init_obs)
        if self.cfg.rollout_failure_message is not None:
            raise RuntimeError(self.cfg.rollout_failure_message)
        action = {"action": init_obs}
        step_ret = (
            {"obs": init_obs["obs"], "done": True},
            1.0,
            True,
            False,
            {"job": init_obs["obs"]},
        )
        if env_step_callback is not None:
            env_step_callback(action, step_ret)
        return EnvRolloutReturn(
            init_obs=init_obs,
            actions=[action],
            step_results=[step_ret],
            rollout_actual_steps=1,
            terminal_condition_triggered=(
                terminal_condition(step_ret)
                if terminal_condition is not None
                else None
            ),
            env_step_callback=env_step_callback,
        )

    def close(self) -> None:
        self.closed = True

    def num_envs(self) -> int:
        return 1


class _LocalRecordingEnvConfig(EnvBaseCfg[_LocalRecordingEnv]):
    class_type: ClassType[_LocalRecordingEnv] = _LocalRecordingEnv

    offset_seed: int = 7
    reset_failure_message: str | None = None
    rollout_failure_message: str | None = None


class _FakeRemoteEvaluator:
    def __init__(
        self,
        worker_id: int,
        *,
        on_reset: Callable[[dict[str, Any] | None], None] | None = None,
        on_evaluate: Callable[[Any], None] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.on_reset = on_reset
        self.on_evaluate = on_evaluate
        self.closed = False
        self.setup_calls: list[dict[str, Any]] = []
        self.reconfigure_env_calls: list[Any] = []
        self.reset_metric_calls: list[dict[str, Any]] = []
        self.reset_env_calls: list[dict[str, Any] | None] = []
        self.evaluate_calls: list[Any] = []
        self.metrics: EvaluatorMetrics | None = None

    def setup(
        self,
        *,
        env_cfg: Any,
        policy_or_cfg: Any,
        metrics: Any,
        device: Any = None,
        timeout_s: float | None = None,
    ) -> None:
        self.metrics = metrics
        self.setup_calls.append(
            {
                "env_cfg": env_cfg,
                "policy_or_cfg": policy_or_cfg,
                "metrics": metrics,
                "device": device,
                "timeout_s": timeout_s,
            }
        )

    def reconfigure_env(
        self,
        env_cfg: Any,
        *,
        force_recreate: bool | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.reconfigure_env_calls.append(
            {
                "env_cfg": env_cfg,
                "force_recreate": force_recreate,
                "timeout_s": timeout_s,
            }
        )

    def reset_metrics(
        self,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.reset_metric_calls.append({"timeout_s": timeout_s, **kwargs})

    def reset_env(
        self,
        *,
        env_reset_input: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del timeout_s
        self.reset_env_calls.append(env_reset_input)
        if self.on_reset is not None:
            self.on_reset(env_reset_input)
        job_key = None if env_reset_input is None else env_reset_input["job"]
        reset_info = {
            "job": job_key,
            "offset_seed": 10 + self.worker_id,
        }
        return {"obs": job_key}, reset_info

    def evaluate_episode(
        self,
        *,
        max_steps: int,
        env_reset_input: Any,
        policy_reset_input: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        del max_steps, policy_reset_input, timeout_s
        self.evaluate_calls.append(env_reset_input)
        if self.on_evaluate is not None:
            self.on_evaluate(env_reset_input)
        return {"job": env_reset_input.info["job"]}

    def get_metrics(
        self,
        *,
        timeout_s: float | None = None,
    ) -> EvaluatorMetrics | None:
        del timeout_s
        return self.metrics

    def close(self) -> None:
        self.closed = True


class _FakeRemoteConfig:
    def __init__(
        self,
        evaluator_factory: Callable[[int], _FakeRemoteEvaluator] | None = None,
    ) -> None:
        self.evaluator_factory = evaluator_factory
        self.created: list[_FakeRemoteEvaluator] = []

    def __call__(self) -> _FakeRemoteEvaluator:
        worker_id = len(self.created)
        if self.evaluator_factory is None:
            evaluator = _FakeRemoteEvaluator(worker_id)
        else:
            evaluator = self.evaluator_factory(worker_id)
        self.created.append(evaluator)
        return evaluator


def _episode(key: str, group: str = "task") -> BenchmarkEpisode:
    return BenchmarkEpisode(
        episode_key=key,
        group_key=group,
        episode_id=int(key.split("-")[-1]),
    )


def _job(key: str) -> BenchmarkPrepareJob:
    return BenchmarkPrepareJob(
        episode=_episode(key),
        attempt_index=0,
        max_steps=5,
        policy_reset_input={"policy": key},
        metric_reset_input={"metric": key},
        metadata={"job": key},
    )


def _request(job: BenchmarkPrepareJob) -> BenchmarkAttemptRequest:
    return BenchmarkAttemptRequest(
        episode=job.episode,
        attempt_index=job.attempt_index,
        env_cfg={"env": job.metadata["job"]},  # type: ignore[arg-type]
        env_reset_input={"job": job.metadata["job"]},
        max_steps=job.max_steps,
        policy_reset_input=job.policy_reset_input,
        metric_reset_input=job.metric_reset_input,
        metadata={"attempted_offset_seed": job.episode.episode_id},
    )


def _local_request(
    job: BenchmarkPrepareJob,
    env_cfg: _LocalRecordingEnvConfig | None = None,
) -> BenchmarkAttemptRequest:
    return BenchmarkAttemptRequest(
        episode=job.episode,
        attempt_index=job.attempt_index,
        env_cfg=_LocalRecordingEnvConfig() if env_cfg is None else env_cfg,
        env_reset_input={"job": job.metadata["job"]},
        max_steps=job.max_steps,
        policy_reset_input=job.policy_reset_input,
        metric_reset_input=job.metric_reset_input,
        metadata={"attempted_offset_seed": job.episode.episode_id},
    )


def _backend(
    remote_cfg: _FakeRemoteConfig,
    *,
    num_workers: int = 1,
) -> RemoteBenchmarkBackend:
    return RemoteBenchmarkBackend(
        RemoteBenchmarkBackendConfig(
            remote_cfg=remote_cfg,  # type: ignore[arg-type]
            num_workers=num_workers,
            policy_or_cfg="policy",  # type: ignore[arg-type]
            worker_metrics_factory=lambda: EvaluatorMetrics.from_metric(
                _NoopMetric(),
            ),
            device="cpu",
        )
    )


def _local_backend() -> LocalBenchmarkBackend:
    return LocalBenchmarkBackend(
        LocalBenchmarkBackendConfig(
            evaluator_cfg=PolicyEvaluatorConfig(
                reconfigure_env_force_recreate=False,
            ),
            policy_or_cfg=DummyPolicyConfig(),
            worker_metrics_factory=lambda: EvaluatorMetrics.from_metric(
                DummySuccessRateMetricConfig()(),
                name="success_rate",
            ),
            device="cpu",
        )
    )


class _SingleEventDriver(BenchmarkDriver):
    def __init__(self, job: BenchmarkPrepareJob) -> None:
        self.job = job
        self.submitted = False
        self.prepared: list[BenchmarkPrepareSucceededEvent] = []
        self.terminals: list[BenchmarkTerminalEvent] = []
        self.max_jobs_seen: list[int | None] = []

    def has_unfinished_work(self) -> bool:
        return not self.terminals

    def get_ready_jobs(
        self,
        max_jobs: int | None = None,
    ) -> Sequence[BenchmarkPrepareJob]:
        self.max_jobs_seen.append(max_jobs)
        if self.submitted:
            return []
        self.submitted = True
        return [self.job]

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        return _request(job)

    def on_attempt_prepared(
        self,
        event: BenchmarkPrepareSucceededEvent,
    ) -> None:
        self.prepared.append(event)

    def on_terminal_event(
        self,
        event: BenchmarkTerminalEvent,
    ) -> None:
        self.terminals.append(event)

    def result(self) -> BenchmarkResult:
        records = [
            BenchmarkEpisodeRecord(
                episode=self.job.episode,
                succeeded=isinstance(
                    self.terminals[0],
                    BenchmarkEvaluateSucceededEvent,
                ),
                attempts=1,
            )
        ]
        return BenchmarkResult(metrics={}, episodes=records, metadata={})


class _LocalSingleEventDriver(_SingleEventDriver):
    def __init__(
        self,
        job: BenchmarkPrepareJob,
        *,
        env_cfg: _LocalRecordingEnvConfig | None = None,
    ) -> None:
        super().__init__(job)
        self.env_cfg = env_cfg

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        return _local_request(job, self.env_cfg)


class _LocalTwoEventDriver(BenchmarkDriver):
    def __init__(self) -> None:
        self.jobs = [_job("job-0"), _job("job-1")]
        self.submitted = 0
        self.prepared: list[BenchmarkPrepareSucceededEvent] = []
        self.terminals: list[BenchmarkTerminalEvent] = []

    def has_unfinished_work(self) -> bool:
        return len(self.terminals) < len(self.jobs)

    def get_ready_jobs(
        self,
        max_jobs: int | None = None,
    ) -> Sequence[BenchmarkPrepareJob]:
        del max_jobs
        if self.submitted >= len(self.jobs):
            return []
        job = self.jobs[self.submitted]
        self.submitted += 1
        return [job]

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        return _local_request(job)

    def on_attempt_prepared(
        self,
        event: BenchmarkPrepareSucceededEvent,
    ) -> None:
        self.prepared.append(event)

    def on_terminal_event(
        self,
        event: BenchmarkTerminalEvent,
    ) -> None:
        self.terminals.append(event)

    def result(self) -> BenchmarkResult:
        return BenchmarkResult(metrics={}, episodes=[], metadata={})


class _TwoStageReadyDriver(BenchmarkDriver):
    def __init__(self) -> None:
        self.first_job = _job("job-0")
        self.second_job = _job("job-1")
        self.first_submitted = False
        self.second_submitted = False
        self.prepared: list[BenchmarkPrepareSucceededEvent] = []
        self.terminals: list[BenchmarkTerminalEvent] = []
        self.max_jobs_seen: list[int | None] = []

    def has_unfinished_work(self) -> bool:
        return len(self.terminals) < 2

    def get_ready_jobs(
        self,
        max_jobs: int | None = None,
    ) -> Sequence[BenchmarkPrepareJob]:
        self.max_jobs_seen.append(max_jobs)
        if not self.first_submitted:
            self.first_submitted = True
            return [self.first_job]
        if self.prepared and not self.second_submitted:
            self.second_submitted = True
            return [self.second_job]
        return []

    def make_attempt_request(
        self,
        job: BenchmarkPrepareJob,
    ) -> BenchmarkAttemptRequest:
        return _request(job)

    def on_attempt_prepared(
        self,
        event: BenchmarkPrepareSucceededEvent,
    ) -> None:
        self.prepared.append(event)

    def on_terminal_event(
        self,
        event: BenchmarkTerminalEvent,
    ) -> None:
        self.terminals.append(event)

    def result(self) -> BenchmarkResult:
        return BenchmarkResult(metrics={}, episodes=[], metadata={})


def test_benchmark_attempt_error_keeps_terminal_event() -> None:
    event = BenchmarkPrepareFailedEvent(
        request=_request(_job("job-0")),
        reset_info={"offset_seed": 3},
        error_type="RuntimeError",
        error_message="reset failed",
        worker_id=0,
    )

    error = BenchmarkAttemptError(event)

    assert error.terminal_event is event
    assert "job-0" in str(error)
    assert "RuntimeError: reset failed" in str(error)


def test_local_backend_runs_one_job_with_prepared_start() -> None:
    _LocalRecordingEnv.instances.clear()
    driver = _LocalSingleEventDriver(_job("job-0"))

    result = _local_backend().run(driver)

    assert len(result.episodes) == 1
    assert result.episodes[0].succeeded is True
    assert driver.max_jobs_seen == [1]
    assert len(driver.prepared) == 1
    assert len(driver.terminals) == 1
    terminal = driver.terminals[0]
    assert isinstance(terminal, BenchmarkEvaluateSucceededEvent)
    assert terminal.episode_metrics == {"success_rate": 1.0}
    assert terminal.reset_info == {"job": "job-0", "offset_seed": 7}
    assert len(_LocalRecordingEnv.instances) == 1
    env = _LocalRecordingEnv.instances[0]
    assert env.reset_calls == [{"job": "job-0"}]
    assert env.rollout_init_obs == [{"obs": "job-0"}]
    assert env.closed is True


def test_local_backend_reuses_same_env_config_between_jobs() -> None:
    _LocalRecordingEnv.instances.clear()
    driver = _LocalTwoEventDriver()

    _local_backend().run(driver)

    assert len(driver.prepared) == 2
    assert len(driver.terminals) == 2
    assert len(_LocalRecordingEnv.instances) == 1
    env = _LocalRecordingEnv.instances[0]
    assert env.reset_calls == [{"job": "job-0"}, {"job": "job-1"}]
    assert env.closed is True


def test_local_backend_logs_first_time_evaluator_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(benchmark_backend, "logger", logger)
    _LocalRecordingEnv.instances.clear()
    driver = _LocalTwoEventDriver()

    _local_backend().run(driver)

    assert logger.infos == [
        "Initializing local benchmark evaluator: worker=0 task=task episode=0",
        "Initialized local benchmark evaluator: worker=0 task=task episode=0",
    ]


def test_local_backend_prepare_failure_closes_evaluator() -> None:
    _LocalRecordingEnv.instances.clear()
    driver = _LocalSingleEventDriver(
        _job("job-0"),
        env_cfg=_LocalRecordingEnvConfig(
            reset_failure_message="reset failed",
        ),
    )

    _local_backend().run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkPrepareFailedEvent)
    assert event.reset_info == {"offset_seed": 7}
    assert event.error_type == "_ResetError"
    assert event.error_message == "reset failed"
    assert _LocalRecordingEnv.instances[0].closed is True


def test_local_backend_evaluate_failure_keeps_reset_info_and_closes() -> None:
    _LocalRecordingEnv.instances.clear()
    driver = _LocalSingleEventDriver(
        _job("job-0"),
        env_cfg=_LocalRecordingEnvConfig(
            rollout_failure_message="rollout failed",
        ),
    )

    _local_backend().run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkEvaluateFailedEvent)
    assert event.reset_info == {"job": "job-0", "offset_seed": 7}
    assert event.error_type == "PolicyEvaluationExecutionError"
    assert "episode execution" in event.error_message
    assert _LocalRecordingEnv.instances[0].closed is True


def test_local_backend_keyboard_interrupt_closes_evaluator() -> None:
    class _InterruptAfterPrepareDriver(_LocalSingleEventDriver):
        def on_attempt_prepared(
            self,
            event: BenchmarkPrepareSucceededEvent,
        ) -> None:
            super().on_attempt_prepared(event)
            raise KeyboardInterrupt

    _LocalRecordingEnv.instances.clear()
    driver = _InterruptAfterPrepareDriver(_job("job-0"))

    with pytest.raises(KeyboardInterrupt):
        _local_backend().run(driver)

    assert len(_LocalRecordingEnv.instances) == 1
    assert _LocalRecordingEnv.instances[0].closed is True


def test_prepare_can_continue_while_previous_evaluate_is_blocked() -> None:
    evaluate_started = threading.Event()
    release_evaluate = threading.Event()
    second_prepare_started = threading.Event()

    def _on_reset(env_reset_input: dict[str, Any] | None) -> None:
        if env_reset_input is not None and env_reset_input["job"] == "job-1":
            second_prepare_started.set()

    def _on_evaluate(prepared_start: Any) -> None:
        if prepared_start.info["job"] == "job-0":
            evaluate_started.set()
            if not release_evaluate.wait(timeout=2.0):
                raise RuntimeError("test did not release blocked evaluate")

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(
            worker_id,
            on_reset=_on_reset,
            on_evaluate=_on_evaluate,
        )
    )
    backend = _backend(remote_cfg, num_workers=2)
    driver = _TwoStageReadyDriver()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.run, driver)
        assert evaluate_started.wait(timeout=2.0)
        assert second_prepare_started.wait(timeout=2.0)
        release_evaluate.set()
        future.result(timeout=2.0)

    prepared_keys = [
        event.request.episode.episode_key for event in driver.prepared
    ]
    assert prepared_keys == ["job-0", "job-1"]
    assert len(driver.terminals) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_remote_backend_logs_worker_actor_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(benchmark_backend, "logger", logger)
    remote_cfg = _FakeRemoteConfig()
    backend = _backend(remote_cfg, num_workers=2)

    try:
        assert logger.infos == [
            "Starting remote benchmark evaluator workers: workers=2",
            "Started remote benchmark evaluator workers: workers=2",
        ]
    finally:
        backend.close()


def test_remote_backend_logs_first_time_worker_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(benchmark_backend, "logger", logger)
    remote_cfg = _FakeRemoteConfig()
    backend = _backend(remote_cfg, num_workers=1)
    logger.infos.clear()
    logger.debugs.clear()
    driver = _SingleEventDriver(_job("job-0"))

    backend.run(driver)

    assert logger.debugs == [
        "Initializing remote benchmark evaluator: "
        "worker=0 task=task episode=0",
        "Initialized remote benchmark evaluator: worker=0 task=task episode=0",
    ]
    assert logger.infos == []


def test_remote_backend_reconfigures_reused_worker_with_config_default() -> (
    None
):
    remote_cfg = _FakeRemoteConfig()
    driver = _TwoStageReadyDriver()

    _backend(remote_cfg, num_workers=1).run(driver)

    evaluator = remote_cfg.created[0]
    assert len(evaluator.setup_calls) == 1
    assert evaluator.reconfigure_env_calls == [
        {
            "env_cfg": {"env": "job-1"},
            "force_recreate": None,
            "timeout_s": None,
        }
    ]
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_ready_jobs_must_honor_backend_capacity() -> None:
    class _TooManyJobsDriver(_SingleEventDriver):
        def get_ready_jobs(
            self,
            max_jobs: int | None = None,
        ) -> Sequence[BenchmarkPrepareJob]:
            self.max_jobs_seen.append(max_jobs)
            return [_job("job-0"), _job("job-1")]

    remote_cfg = _FakeRemoteConfig()
    driver = _TooManyJobsDriver(_job("job-0"))

    with pytest.raises(RuntimeError, match="too many ready jobs"):
        _backend(remote_cfg, num_workers=1).run(driver)

    assert driver.max_jobs_seen == [1]
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_backend_config_has_no_polling_or_total_timeout_parameters() -> None:
    field_names = {
        field.name for field in fields(RemoteBenchmarkBackendConfig)
    }
    backend_parameters = set(
        inspect.signature(RemoteBenchmarkBackend).parameters
    )

    assert "worker_poll_interval_s" not in field_names
    assert "episode_timeout_s" not in field_names
    assert "rollout_timeout_s" not in field_names
    assert "reset_timeout_s" not in field_names
    assert "worker_poll_interval_s" not in backend_parameters
    assert "episode_timeout_s" not in backend_parameters
    assert backend_parameters == {"cfg"}


def test_driver_no_progress_raises_instead_of_deadlocking() -> None:
    class _NoProgressDriver(_SingleEventDriver):
        def has_unfinished_work(self) -> bool:
            return True

        def get_ready_jobs(
            self,
            max_jobs: int | None = None,
        ) -> Sequence[BenchmarkPrepareJob]:
            self.max_jobs_seen.append(max_jobs)
            return []

    remote_cfg = _FakeRemoteConfig()
    driver = _NoProgressDriver(_job("job-0"))

    with pytest.raises(RuntimeError, match="made no progress"):
        _backend(remote_cfg, num_workers=1).run(driver)

    assert driver.max_jobs_seen == [1]
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_prepare_failure_event_keeps_reset_info_and_request_metadata() -> None:
    def _on_reset(env_reset_input: dict[str, Any] | None) -> None:
        assert env_reset_input is not None
        raise _ResetError("reset failed", {"offset_seed": 7})

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(worker_id, on_reset=_on_reset)
    )
    driver = _SingleEventDriver(_job("job-0"))

    _backend(remote_cfg, num_workers=1).run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkPrepareFailedEvent)
    assert event.reset_info == {"offset_seed": 7}
    assert event.request.metadata == {"attempted_offset_seed": 0}
    assert event.worker_id == 0
    assert len(remote_cfg.created) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_prepare_timeout_replaces_worker_and_reports_terminal_event() -> None:
    def _on_reset(env_reset_input: dict[str, Any] | None) -> None:
        del env_reset_input
        raise PolicyEvaluationRemoteTimeoutError("reset timed out")

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(worker_id, on_reset=_on_reset)
    )
    driver = _SingleEventDriver(_job("job-0"))

    _backend(remote_cfg, num_workers=1).run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkPrepareFailedEvent)
    assert event.error_type == "PolicyEvaluationRemoteTimeoutError"
    assert event.error_message == "reset timed out"
    assert len(remote_cfg.created) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_evaluate_failure_event_keeps_prepare_reset_info() -> None:
    def _on_evaluate(prepared_start: Any) -> None:
        assert prepared_start.info["job"] == "job-0"
        raise RuntimeError("rollout failed")

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(
            worker_id,
            on_evaluate=_on_evaluate,
        )
    )
    driver = _SingleEventDriver(_job("job-0"))

    _backend(remote_cfg, num_workers=1).run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkEvaluateFailedEvent)
    assert event.reset_info == {"job": "job-0", "offset_seed": 10}
    assert event.request.metadata == {"attempted_offset_seed": 0}
    assert event.worker_id == 0
    assert len(remote_cfg.created) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_evaluate_timeout_replaces_worker_and_keeps_reset_info() -> None:
    def _on_evaluate(prepared_start: Any) -> None:
        assert prepared_start.info["job"] == "job-0"
        raise PolicyEvaluationRemoteTimeoutError("rollout timed out")

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(
            worker_id,
            on_evaluate=_on_evaluate,
        )
    )
    driver = _SingleEventDriver(_job("job-0"))

    _backend(remote_cfg, num_workers=1).run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkEvaluateFailedEvent)
    assert event.reset_info == {"job": "job-0", "offset_seed": 10}
    assert event.error_type == "PolicyEvaluationRemoteTimeoutError"
    assert event.error_message == "rollout timed out"
    assert len(remote_cfg.created) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_evaluate_worker_lost_replaces_worker_and_keeps_reset_info() -> None:
    def _on_evaluate(prepared_start: Any) -> None:
        assert prepared_start.info["job"] == "job-0"
        raise PolicyEvaluationWorkerLostError("worker lost")

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(
            worker_id,
            on_evaluate=_on_evaluate,
        )
    )
    driver = _SingleEventDriver(_job("job-0"))

    _backend(remote_cfg, num_workers=1).run(driver)

    assert len(driver.terminals) == 1
    event = driver.terminals[0]
    assert isinstance(event, BenchmarkEvaluateFailedEvent)
    assert event.reset_info == {"job": "job-0", "offset_seed": 10}
    assert event.error_type == "PolicyEvaluationWorkerLostError"
    assert event.error_message == "worker lost"
    assert len(remote_cfg.created) == 2
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_late_completion_after_close_does_not_dispatch_driver_event() -> None:
    prepare_started = threading.Event()
    release_prepare = threading.Event()

    def _on_reset(env_reset_input: dict[str, Any] | None) -> None:
        del env_reset_input
        prepare_started.set()
        assert release_prepare.wait(timeout=2.0)

    remote_cfg = _FakeRemoteConfig(
        lambda worker_id: _FakeRemoteEvaluator(worker_id, on_reset=_on_reset)
    )
    backend = _backend(remote_cfg, num_workers=1)
    driver = _SingleEventDriver(_job("job-0"))
    worker = backend._workers[0]
    backend._start_prepare_future(worker, _request(_job("job-0")))
    future = worker.future
    assert future is not None

    assert prepare_started.wait(timeout=2.0)
    backend.close()
    release_prepare.set()
    future.result(timeout=2.0)
    assert worker.future is None

    backend._drain_worker_events(driver)

    assert driver.prepared == []
    assert driver.terminals == []
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_stale_generation_event_does_not_dispatch_driver_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(benchmark_backend, "logger", logger)
    remote_cfg = _FakeRemoteConfig()
    backend = _backend(remote_cfg, num_workers=1)
    driver = _SingleEventDriver(_job("job-0"))
    request = _request(_job("job-0"))

    try:
        backend._workers[0].generation = 1
        backend._handle_worker_event(
            driver,
            benchmark_backend._BenchmarkWorkerEvent(
                worker_id=0,
                generation=0,
                payload=BenchmarkPrepareFailedEvent(
                    request=request,
                    error_type="ResetError",
                    error_message="stale",
                ),
            ),
        )
    finally:
        backend.close()

    assert driver.prepared == []
    assert driver.terminals == []
    assert logger.debugs == [
        "Ignoring stale benchmark worker event: worker=0 "
        "event_generation=0 current_generation=1"
    ]


def test_same_generation_prepared_evaluator_mismatch_raises() -> None:
    remote_cfg = _FakeRemoteConfig()
    backend = _backend(remote_cfg, num_workers=1)
    driver = _SingleEventDriver(_job("job-0"))
    worker = backend._workers[0]
    request = _request(_job("job-0"))
    wrong_evaluator = _FakeRemoteEvaluator(worker_id=99)

    try:
        with pytest.raises(RuntimeError, match="does not match worker slot"):
            backend._handle_worker_event(
                driver,
                benchmark_backend._BenchmarkWorkerEvent(
                    worker_id=worker.worker_id,
                    generation=worker.generation,
                    payload=benchmark_backend._BenchmarkPreparedWork(
                        lease=benchmark_backend._BenchmarkWorkerLease(
                            worker_id=worker.worker_id,
                            generation=worker.generation,
                            evaluator=wrong_evaluator,
                        ),
                        request=request,
                        prepared_start=PreparedEnvStart(
                            observations={"obs": "job-0"},
                            info={"job": "job-0", "offset_seed": 10},
                        ),
                    ),
                ),
            )
    finally:
        backend.close()

    assert driver.prepared == []
    assert driver.terminals == []


def test_remote_backend_pending_event_wait_is_bounded_for_interrupts() -> None:
    class _InterruptingQueue:
        def __init__(self) -> None:
            self.get_kwargs: list[dict[str, Any]] = []

        def get_nowait(self) -> Any:
            raise queue.Empty

        def get(self, *args: Any, **kwargs: Any) -> Any:
            del args
            self.get_kwargs.append(dict(kwargs))
            raise KeyboardInterrupt

        def put(self, event: Any) -> None:
            del event

    remote_cfg = _FakeRemoteConfig()
    backend = _backend(remote_cfg, num_workers=1)
    interrupting_queue = _InterruptingQueue()
    backend._worker_event_queue = interrupting_queue  # type: ignore[assignment]
    worker = backend._workers[0]
    worker.state = "evaluating"
    worker.future = Future()
    driver = _SingleEventDriver(_job("job-0"))

    with pytest.raises(KeyboardInterrupt):
        backend.run(driver)

    assert interrupting_queue.get_kwargs
    assert interrupting_queue.get_kwargs[0]["timeout"] > 0.0
    assert all(evaluator.closed for evaluator in remote_cfg.created)


def test_keyboard_interrupt_closes_remote_workers() -> None:
    class _InterruptDriver(_SingleEventDriver):
        def get_ready_jobs(
            self,
            max_jobs: int | None = None,
        ) -> Sequence[BenchmarkPrepareJob]:
            self.max_jobs_seen.append(max_jobs)
            raise KeyboardInterrupt

    remote_cfg = _FakeRemoteConfig()
    driver = _InterruptDriver(_job("job-0"))

    with pytest.raises(KeyboardInterrupt):
        _backend(remote_cfg, num_workers=2).run(driver)

    assert driver.max_jobs_seen == [2]
    assert all(evaluator.closed for evaluator in remote_cfg.created)
