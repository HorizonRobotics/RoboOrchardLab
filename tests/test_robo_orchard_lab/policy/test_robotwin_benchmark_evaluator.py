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
from typing import Any, cast

import pytest

import robo_orchard_lab.policy.evaluator.benchmark.backend as benchmark_backend
from robo_orchard_lab.policy.evaluator.benchmark import (
    BenchmarkAttemptError,
    BenchmarkEvaluateFailedEvent,
    BenchmarkEvaluateSucceededEvent,
    BenchmarkPrepareFailedEvent,
    BenchmarkPrepareSucceededEvent,
    BenchmarkResult,
    robotwin as rb,
)
from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
    SuccessRateInfo,
    SuccessRateMetric,
)
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)


@pytest.fixture(autouse=True)
def _fake_robotwin_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "config_robotwin_path", lambda: "/tmp/robotwin")
    monkeypatch.setattr(rb.os.path, "exists", lambda path: True)


def _cfg(
    *,
    task_names: list[str] | None = None,
    episode_num: int = 1,
    max_retries: int = 3,
    start_seed: int = 0,
    backend: (
        rb.RoboTwinLocalBenchmarkBackendCfg
        | rb.RoboTwinRemoteBenchmarkBackendCfg
        | None
    ) = None,
    artifact_root_dir: str | None = None,
    fail_fast: bool = False,
    log_progress: bool = True,
    progress_log_every_n_episodes: int | None = None,
) -> rb.RoboTwinBenchmarkEvaluatorCfg:
    kwargs: dict[str, Any] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if progress_log_every_n_episodes is not None:
        kwargs["progress_log_every_n_episodes"] = progress_log_every_n_episodes
    return rb.RoboTwinBenchmarkEvaluatorCfg(
        task_names=["task_a"] if task_names is None else task_names,
        episode_num=episode_num,
        max_retries=max_retries,
        start_seed=start_seed,
        artifact_root_dir=artifact_root_dir,
        fail_fast=fail_fast,
        log_progress=log_progress,
        **kwargs,
    )


def _worker_metrics(
    *,
    task_name: str = "task_a",
    success: bool = True,
    offset_seed: int = 0,
    start_seed: int = 0,
) -> EvaluatorMetrics:
    last_update = {
        "task_name": task_name,
        "seed": 1000 + offset_seed,
        "start_seed": start_seed,
        "resolved_start_seed": 1000 + start_seed,
        "offset_seed": offset_seed,
        "success": success,
    }
    metric = SuccessRateMetric()
    metric.info = {
        task_name: SuccessRateInfo(
            task_name=task_name,
            success_count=1 if success else 0,
            total_count=1,
            info_list=[dict(last_update)],
        )
    }
    metric.last_update_info = dict(last_update)
    return EvaluatorMetrics.from_metric(
        metric,
        name="success_rate",
    )


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.infos.append(message % args if args else message)

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append(message % args if args else message)


def test_config_has_no_episode_timeout_or_polling_fields() -> None:
    field_names = set(rb.RoboTwinBenchmarkEvaluatorCfg.model_fields)
    local_field_names = set(rb.RoboTwinLocalBenchmarkBackendCfg.model_fields)
    remote_field_names = set(rb.RoboTwinRemoteBenchmarkBackendCfg.model_fields)

    assert "episode_timeout_s" not in field_names
    assert "worker_poll_interval_s" not in field_names
    assert "num_parallel_envs" not in field_names
    assert "rollout_timeout_s" not in field_names
    assert "reset_timeout_s" not in field_names
    assert "timeout_grace_retries" not in field_names
    assert "remote_class_config" not in field_names
    assert "ray_init_config" not in field_names
    assert "evaluator_cfg" not in field_names
    assert "evaluator_cfg" not in local_field_names
    assert "evaluator_cfg" not in remote_field_names
    assert "num_parallel_envs" not in local_field_names
    assert "rollout_timeout_s" not in local_field_names
    assert "reset_timeout_s" not in local_field_names
    assert "timeout_grace_retries" not in local_field_names
    assert "remote_class_config" not in local_field_names
    assert "ray_init_config" not in local_field_names
    assert {
        "num_parallel_envs",
        "rollout_timeout_s",
        "reset_timeout_s",
        "timeout_grace_retries",
        "remote_class_config",
        "ray_init_config",
    }.issubset(remote_field_names)


def test_config_validation_does_not_load_robotwin_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_if_loaded(task_name: str) -> object:
        raise AssertionError(f"loaded RoboTwin task {task_name}")

    monkeypatch.setattr(
        rb,
        "create_task_from_name",
        _raise_if_loaded,
        raising=False,
    )

    cfg = _cfg(task_names=["possibly_missing_robotwin_task"])

    assert cfg.task_names == ["possibly_missing_robotwin_task"]


def test_backend_config_round_trip_preserves_concrete_type() -> None:
    local_cfg = _cfg()
    loaded_local_cfg = rb.RoboTwinBenchmarkEvaluatorCfg.model_validate(
        local_cfg.model_dump(mode="json")
    )

    remote_cfg = _cfg(
        backend=rb.RoboTwinRemoteBenchmarkBackendCfg(
            num_parallel_envs=4,
            rollout_timeout_s=12.0,
            reset_timeout_s=34.0,
            timeout_grace_retries=2,
            ray_init_config={"ignore_reinit_error": True},
        )
    )
    loaded_remote_cfg = rb.RoboTwinBenchmarkEvaluatorCfg.model_validate(
        remote_cfg.model_dump(mode="json")
    )

    assert isinstance(
        loaded_local_cfg.backend,
        rb.RoboTwinLocalBenchmarkBackendCfg,
    )
    assert isinstance(
        loaded_remote_cfg.backend,
        rb.RoboTwinRemoteBenchmarkBackendCfg,
    )
    assert loaded_remote_cfg.backend.num_parallel_envs == 4
    assert loaded_remote_cfg.backend.rollout_timeout_s == 12.0
    assert loaded_remote_cfg.backend.reset_timeout_s == 34.0
    assert loaded_remote_cfg.backend.timeout_grace_retries == 2
    assert loaded_remote_cfg.backend.ray_init_config == {
        "ignore_reinit_error": True
    }


def test_evaluator_config_call_constructs_evaluator_from_config() -> None:
    cfg = _cfg(episode_num=3, max_retries=2)

    evaluator = cfg()
    overridden_evaluator = cfg(episode_num=4)

    assert isinstance(evaluator, rb.RoboTwinBenchmarkEvaluator)
    assert evaluator.cfg == cfg
    assert overridden_evaluator.cfg.episode_num == 4
    assert overridden_evaluator.cfg.max_retries == 2


def test_make_attempt_request_allocates_offset_and_env_cfg() -> None:
    cfg = _cfg(
        start_seed=42,
        artifact_root_dir="/artifacts",
    )
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)

    assert request.metadata["attempted_offset_seed"] == 0
    assert request.metadata["reset_context_pinned"] is False
    assert request.env_reset_input == {
        "offset_seed": 0,
        "task_name": "task_a",
        "episode_id": 0,
        "clear_cache": True,
        "return_obs": True,
        "video_dir": "/artifacts/task_a/demo_clean",
    }
    assert request.env_cfg.task_name == "task_a"
    assert request.env_cfg.seed == 42
    assert request.env_cfg.task_config_path == (
        "/tmp/robotwin/task_config/demo_clean.yml"
    )
    assert request.env_cfg.format_datatypes is True
    assert request.env_cfg.action_type == "qpos"


def test_prepare_submission_parallelism_respects_task_prepare_slots() -> None:
    cfg = _cfg(task_names=["task_a", "task_b"], episode_num=2)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    jobs = list(driver.get_ready_jobs(max_jobs=4))

    assert {job.episode.group_key for job in jobs} == {"task_a", "task_b"}
    assert driver.get_ready_jobs(max_jobs=4) == []

    task_a_job = next(job for job in jobs if job.episode.group_key == "task_a")
    task_a_request = driver.make_attempt_request(task_a_job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=task_a_request,
            reset_info={"offset_seed": 3},
        )
    )

    [next_job] = driver.get_ready_jobs(max_jobs=4)
    assert next_job.episode.group_key == "task_a"
    assert next_job.episode.episode_id == 1


def test_ready_jobs_prioritize_task_names_order() -> None:
    cfg = _cfg(task_names=["task_a", "task_b"], episode_num=2)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [first_job] = driver.get_ready_jobs(max_jobs=1)
    first_request = driver.make_attempt_request(first_job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=first_request,
            reset_info={"offset_seed": 0},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateSucceededEvent(
            request=first_request,
            episode_metrics={},
            worker_metrics=_worker_metrics(
                task_name="task_a",
                success=True,
                offset_seed=0,
            ),
            reset_info={"offset_seed": 0},
        )
    )

    [second_job] = driver.get_ready_jobs(max_jobs=1)

    assert second_job.episode.group_key == "task_a"
    assert second_job.episode.episode_id == 1


def test_success_merges_worker_metrics_without_reversing_offset() -> None:
    cfg = _cfg(episode_num=2, start_seed=7)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 7},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateSucceededEvent(
            request=request,
            episode_metrics={},
            worker_metrics=_worker_metrics(
                success=True,
                offset_seed=99,
                start_seed=7,
            ),
            reset_info={"offset_seed": 7},
        )
    )

    [next_job] = driver.get_ready_jobs(max_jobs=1)
    next_request = driver.make_attempt_request(next_job)
    result = driver.result()

    assert next_request.env_reset_input["offset_seed"] == 8
    assert result.metrics["average_success_rate"] == 1.0
    assert len(result.episodes) == 1
    assert result.episodes[0].succeeded is True
    assert result.episodes[0].attempts == 1


def test_episode_completion_log_reports_task_and_total_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(rb, "logger", logger, raising=False)
    cfg = _cfg(task_names=["task_a", "task_b"], episode_num=2)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 0},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateSucceededEvent(
            request=request,
            episode_metrics={},
            worker_metrics=_worker_metrics(
                task_name="task_a",
                success=True,
                offset_seed=0,
            ),
            reset_info={"offset_seed": 0},
        )
    )

    assert logger.infos == [
        "RoboTwin benchmark episode completed: task=task_a "
        "episode=0 success=True attempts=1 offset_seed=0 "
        "task_success_rate=1.000 task_progress=1/2 total_progress=1/4"
    ]


def test_episode_completion_log_throttles_success_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(rb, "logger", logger, raising=False)
    cfg = _cfg(episode_num=3, progress_log_every_n_episodes=2)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    for episode_id in range(3):
        [job] = driver.get_ready_jobs(max_jobs=1)
        request = driver.make_attempt_request(job)
        driver.on_attempt_prepared(
            BenchmarkPrepareSucceededEvent(
                request=request,
                reset_info={"offset_seed": episode_id},
            )
        )
        driver.on_terminal_event(
            BenchmarkEvaluateSucceededEvent(
                request=request,
                episode_metrics={},
                worker_metrics=_worker_metrics(
                    success=True,
                    offset_seed=episode_id,
                ),
                reset_info={"offset_seed": episode_id},
            )
        )

    assert logger.infos == [
        "RoboTwin benchmark episode completed: task=task_a "
        "episode=1 success=True attempts=1 offset_seed=1 "
        "task_success_rate=1.000 task_progress=2/3 total_progress=2/3",
        "RoboTwin benchmark episode completed: task=task_a "
        "episode=2 success=True attempts=1 offset_seed=2 "
        "task_success_rate=1.000 task_progress=3/3 total_progress=3/3",
    ]


def test_retry_attempt_log_reports_failure_before_resubmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(rb, "logger", logger, raising=False)
    cfg = _cfg(episode_num=1, max_retries=1)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 4},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateFailedEvent(
            request=request,
            reset_info={"offset_seed": 4},
            error_type="RolloutError",
            error_message="rollout failed",
        )
    )

    assert logger.warnings == [
        "RoboTwin benchmark attempt failed, retrying: task=task_a "
        "episode=0 attempt=1/2 phase=evaluate offset_seed=4 "
        "error=RolloutError: rollout failed"
    ]
    assert logger.infos == []


def test_result_logs_benchmark_summary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(rb, "logger", logger, raising=False)
    cfg = _cfg(episode_num=1)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 0},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateSucceededEvent(
            request=request,
            episode_metrics={},
            worker_metrics=_worker_metrics(success=True, offset_seed=0),
            reset_info={"offset_seed": 0},
        )
    )

    driver.result()
    driver.result()

    assert logger.infos == [
        "RoboTwin benchmark episode completed: task=task_a "
        "episode=0 success=True attempts=1 offset_seed=0 "
        "task_success_rate=1.000 task_progress=1/1 total_progress=1/1",
        "RoboTwin benchmark completed: tasks=1 episodes=1 "
        "average_success_rate=1.000",
    ]


def test_progress_logging_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(rb, "logger", logger, raising=False)
    cfg = _cfg(episode_num=1, log_progress=False)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 0},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateSucceededEvent(
            request=request,
            episode_metrics={},
            worker_metrics=_worker_metrics(success=True, offset_seed=0),
            reset_info={"offset_seed": 0},
        )
    )
    driver.result()

    assert logger.infos == []
    assert logger.warnings == []


def test_prepare_failure_retries_then_records_failure_and_continues() -> None:
    cfg = _cfg(
        episode_num=2,
        max_retries=1,
        artifact_root_dir="/artifacts",
    )
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_terminal_event(
        BenchmarkPrepareFailedEvent(
            request=request,
            error_type="ResetError",
            error_message="reset before info",
        )
    )

    [retry_job] = driver.get_ready_jobs(max_jobs=1)
    retry_request = driver.make_attempt_request(retry_job)

    assert retry_job.episode.episode_id == 0
    assert retry_job.attempt_index == 1
    assert retry_request.env_reset_input["offset_seed"] == 0
    assert retry_request.metadata["reset_context_pinned"] is False
    assert (
        retry_request.env_reset_input["video_dir"]
        == "/artifacts/task_a/demo_clean"
    )

    driver.on_terminal_event(
        BenchmarkPrepareFailedEvent(
            request=retry_request,
            reset_info={"offset_seed": 5},
            error_type="ResetError",
            error_message="reset after info",
        )
    )

    [next_job] = driver.get_ready_jobs(max_jobs=1)
    next_request = driver.make_attempt_request(next_job)
    result = driver.result()

    assert len(result.episodes) == 1
    assert result.episodes[0].succeeded is False
    assert result.episodes[0].attempts == 2
    assert len(result.episodes[0].attempt_errors) == 2
    assert result.metrics["average_success_rate"] == 0.0
    assert result.metrics["tasks"] == [
        {
            "task_name": "task_a",
            "success_count": 0,
            "total_count": 1,
            "success_rate": 0.0,
        }
    ]
    assert next_job.episode.episode_id == 1
    assert next_request.env_reset_input["offset_seed"] == 0


def test_evaluate_failure_retries_from_pinned_reset_context() -> None:
    cfg = _cfg(episode_num=1, max_retries=1)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 4},
        )
    )
    driver.on_terminal_event(
        BenchmarkEvaluateFailedEvent(
            request=request,
            reset_info={"offset_seed": 4},
            error_type="RolloutError",
            error_message="rollout failed",
        )
    )

    [retry_job] = driver.get_ready_jobs(max_jobs=1)
    retry_request = driver.make_attempt_request(retry_job)

    assert retry_job.episode.episode_id == 0
    assert retry_job.attempt_index == 1
    assert retry_request.env_reset_input["offset_seed"] == 4
    assert retry_request.metadata["attempted_offset_seed"] == 4
    assert retry_request.metadata["reset_context_pinned"] is True


def test_pinned_retry_ignores_parallel_frontier_advance() -> None:
    cfg = _cfg(episode_num=2, max_retries=1)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [first_job] = driver.get_ready_jobs(max_jobs=1)
    first_request = driver.make_attempt_request(first_job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=first_request,
            reset_info={"offset_seed": 4},
        )
    )

    [second_job] = driver.get_ready_jobs(max_jobs=1)
    second_request = driver.make_attempt_request(second_job)
    assert second_request.env_reset_input["offset_seed"] == 5
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=second_request,
            reset_info={"offset_seed": 9},
        )
    )

    driver.on_terminal_event(
        BenchmarkEvaluateFailedEvent(
            request=first_request,
            reset_info={"offset_seed": 4},
            error_type="RolloutError",
            error_message="rollout failed",
        )
    )

    [retry_job] = driver.get_ready_jobs(max_jobs=1)
    retry_request = driver.make_attempt_request(retry_job)

    assert retry_job.episode.episode_id == 0
    assert retry_job.attempt_index == 1
    assert retry_request.env_reset_input["offset_seed"] == 4
    assert retry_request.metadata["attempted_offset_seed"] == 4
    assert retry_request.metadata["reset_context_pinned"] is True


def test_attempt_prepared_requires_reset_context_pinned_metadata() -> None:
    cfg = _cfg()
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    request.metadata.pop("reset_context_pinned")
    with pytest.raises(RuntimeError, match="reset_context_pinned"):
        driver.on_attempt_prepared(
            BenchmarkPrepareSucceededEvent(
                request=request,
                reset_info={"offset_seed": 0},
            )
        )

    request.metadata["reset_context_pinned"] = "false"
    with pytest.raises(RuntimeError, match="must be a bool"):
        driver.on_attempt_prepared(
            BenchmarkPrepareSucceededEvent(
                request=request,
                reset_info={"offset_seed": 0},
            )
        )


def test_fail_fast_prepare_failure_raises_without_recording() -> None:
    cfg = _cfg(fail_fast=True)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    event = BenchmarkPrepareFailedEvent(
        request=request,
        error_type="ResetError",
        error_message="reset failed",
    )

    with pytest.raises(BenchmarkAttemptError) as exc_info:
        driver.on_terminal_event(event)

    assert exc_info.value.terminal_event is event
    assert driver.result().episodes == []
    episode_state = driver._episode_states[job.episode.episode_key]
    task_state = driver._task_states["task_a"]
    assert episode_state.attempt_errors == []
    assert episode_state.retry_reset_context is None
    assert list(task_state.retry_queue) == []


def test_fail_fast_evaluate_failure_raises_without_recording() -> None:
    cfg = _cfg(fail_fast=True)
    driver = rb.RoboTwinBenchmarkDriver(cfg)

    [job] = driver.get_ready_jobs(max_jobs=1)
    request = driver.make_attempt_request(job)
    driver.on_attempt_prepared(
        BenchmarkPrepareSucceededEvent(
            request=request,
            reset_info={"offset_seed": 4},
        )
    )
    event = BenchmarkEvaluateFailedEvent(
        request=request,
        reset_info={"offset_seed": 4},
        error_type="RolloutError",
        error_message="rollout failed",
    )

    with pytest.raises(BenchmarkAttemptError) as exc_info:
        driver.on_terminal_event(event)

    assert exc_info.value.terminal_event is event
    assert driver.result().episodes == []
    episode_state = driver._episode_states[job.episode.episode_key]
    task_state = driver._task_states["task_a"]
    assert episode_state.attempt_errors == []
    assert episode_state.retry_reset_context is None
    assert list(task_state.retry_queue) == []


def test_evaluator_evaluate_constructs_remote_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[_FakeBackend] = []

    class _FakeBackend:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg
            self.worker_metrics: EvaluatorMetrics | None = None
            instances.append(self)

        def __enter__(self) -> _FakeBackend:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: Any,
        ) -> None:
            del exc_type, exc, tb

        def run(
            self,
            driver: rb.RoboTwinBenchmarkDriver,
        ) -> BenchmarkResult:
            self.worker_metrics = self.cfg.worker_metrics_factory()
            return BenchmarkResult(
                metrics={"ran": True},
                episodes=[],
                metadata={"driver": type(driver).__name__},
            )

    monkeypatch.setattr(rb, "RemoteBenchmarkBackend", _FakeBackend)
    cfg = _cfg(
        backend=rb.RoboTwinRemoteBenchmarkBackendCfg(
            num_parallel_envs=3,
            rollout_timeout_s=2.0,
            reset_timeout_s=5.0,
            timeout_grace_retries=4,
        )
    )
    result = rb.RoboTwinBenchmarkEvaluator(cfg).evaluate(
        cast(Any, "policy"),
        device="cpu",
    )

    assert result.metrics == {"ran": True}
    assert len(instances) == 1
    backend = instances[0]
    assert backend.cfg.num_workers == 3
    assert backend.cfg.policy_or_cfg == "policy"
    assert backend.cfg.device == "cpu"
    assert backend.cfg.remote_cfg.rollout_timeout_s == 2.0
    assert backend.cfg.remote_cfg.reset_timeout_s == 5.0
    assert backend.cfg.remote_cfg.timeout_grace_retries == 4
    assert (
        backend.cfg.remote_cfg.instance_config.reconfigure_env_force_recreate
        is False
    )
    assert backend.worker_metrics is not None
    assert isinstance(
        backend.worker_metrics.get_metric("success_rate"),
        SuccessRateMetric,
    )


def test_evaluator_evaluate_constructs_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_instances: list[_FakeLocalBackend] = []
    remote_instances: list[Any] = []

    class _FakeLocalBackend:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg
            self.worker_metrics: EvaluatorMetrics | None = None
            local_instances.append(self)

        def __enter__(self) -> _FakeLocalBackend:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: Any,
        ) -> None:
            del exc_type, exc, tb

        def run(
            self,
            driver: rb.RoboTwinBenchmarkDriver,
        ) -> BenchmarkResult:
            self.worker_metrics = self.cfg.worker_metrics_factory()
            return BenchmarkResult(
                metrics={"local": True},
                episodes=[],
                metadata={"driver": type(driver).__name__},
            )

    class _UnexpectedRemoteBackend:
        def __init__(self, cfg: Any) -> None:
            remote_instances.append(cfg)
            raise AssertionError("remote backend should not be constructed")

    monkeypatch.setattr(rb, "LocalBenchmarkBackend", _FakeLocalBackend)
    monkeypatch.setattr(rb, "RemoteBenchmarkBackend", _UnexpectedRemoteBackend)

    cfg = _cfg()
    result = rb.RoboTwinBenchmarkEvaluator(cfg).evaluate(
        cast(Any, "policy"),
        device="cpu",
    )

    assert result.metrics == {"local": True}
    assert len(local_instances) == 1
    assert remote_instances == []
    backend = local_instances[0]
    assert backend.cfg.evaluator_cfg.reconfigure_env_force_recreate is False
    assert backend.cfg.policy_or_cfg == "policy"
    assert backend.cfg.device == "cpu"
    assert backend.worker_metrics is not None
    assert isinstance(
        backend.worker_metrics.get_metric("success_rate"),
        SuccessRateMetric,
    )


def test_local_backend_runs_ready_jobs_in_task_priority_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class _RecordingPolicyEvaluator:
        def __init__(self) -> None:
            self.metrics = _worker_metrics()
            self.env_cfg_snapshot: str | None = None

        def setup(
            self,
            *,
            env_cfg: Any,
            policy_or_cfg: Any,
            metrics: EvaluatorMetrics,
            device: Any = None,
        ) -> None:
            del policy_or_cfg, metrics, device
            self.env_cfg_snapshot = (
                f"{env_cfg.task_name}:{env_cfg.task_config_path}"
            )
            calls.append(("setup", env_cfg.task_name))

        def reconfigure_env(
            self,
            env_cfg: Any,
            *,
            force_recreate: bool | None = None,
        ) -> None:
            assert force_recreate is None
            env_cfg_snapshot = (
                f"{env_cfg.task_name}:{env_cfg.task_config_path}"
            )
            if self.env_cfg_snapshot == env_cfg_snapshot:
                return
            self.env_cfg_snapshot = env_cfg_snapshot
            calls.append(("reconfigure", env_cfg.task_name))

        def reset_metrics(self, **kwargs: Any) -> None:
            del kwargs

        def reset_env(
            self,
            *,
            env_reset_input: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if env_reset_input is None:
                raise RuntimeError("env_reset_input is required")
            task_name = env_reset_input["task_name"]
            episode_id = env_reset_input["episode_id"]
            offset_seed = env_reset_input["offset_seed"]
            calls.append(("prepare", task_name, episode_id, offset_seed))
            return (
                {"obs": dict(env_reset_input)},
                {
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "offset_seed": offset_seed,
                },
            )

        def evaluate_episode(
            self,
            *,
            max_steps: int,
            env_reset_input: Any,
            policy_reset_input: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del max_steps, policy_reset_input
            info = env_reset_input.info
            task_name = info["task_name"]
            episode_id = info["episode_id"]
            offset_seed = info["offset_seed"]
            calls.append(("evaluate", task_name, episode_id, offset_seed))
            self.metrics = _worker_metrics(
                task_name=task_name,
                success=True,
                offset_seed=offset_seed,
            )
            return {"task_name": task_name, "episode_id": episode_id}

        def get_metrics(self) -> EvaluatorMetrics:
            return self.metrics

        def close(self) -> None:
            pass

    class _FakePolicyEvaluatorConfig:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"reconfigure_env_force_recreate": False}

        def __call__(self) -> _RecordingPolicyEvaluator:
            return _RecordingPolicyEvaluator()

    monkeypatch.setattr(
        benchmark_backend,
        "PolicyEvaluatorConfig",
        _FakePolicyEvaluatorConfig,
    )
    cfg = _cfg(
        task_names=["task_a", "task_b"],
        episode_num=2,
        backend=rb.RoboTwinLocalBenchmarkBackendCfg(),
    )

    driver = rb.RoboTwinBenchmarkDriver(cfg)
    result = benchmark_backend.LocalBenchmarkBackend(
        benchmark_backend.LocalBenchmarkBackendConfig(
            evaluator_cfg=benchmark_backend.PolicyEvaluatorConfig(
                reconfigure_env_force_recreate=False,
            ),
            policy_or_cfg=cast(Any, "policy"),
            worker_metrics_factory=driver.make_worker_metrics,
            device="cpu",
        )
    ).run(driver)

    assert calls == [
        ("setup", "task_a"),
        ("prepare", "task_a", 0, 0),
        ("evaluate", "task_a", 0, 0),
        ("prepare", "task_a", 1, 1),
        ("evaluate", "task_a", 1, 1),
        ("reconfigure", "task_b"),
        ("prepare", "task_b", 0, 0),
        ("evaluate", "task_b", 0, 0),
        ("prepare", "task_b", 1, 1),
        ("evaluate", "task_b", 1, 1),
    ]
    assert [
        (record.episode.group_key, record.episode.episode_id)
        for record in result.episodes
    ] == [
        ("task_a", 0),
        ("task_a", 1),
        ("task_b", 0),
        ("task_b", 1),
    ]
