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
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.horizon_manipulation.tools.check_config import (
    InspectConfig,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.check_models import (
    EpisodeData,
    EpisodeInspection,
    RuleResult,
)


def _camera_topic(topic: str) -> bool:
    """Return whether the topic belongs to a camera stream.

    Args:
        topic: Topic name to classify.

    Returns:
        bool: True when the topic is a camera stream.
    """

    return "/observation/cameras/" in topic


def _robot_state_topic(topic: str) -> bool:
    """Return whether the topic belongs to a robot-state stream.

    Args:
        topic: Topic name to classify.

    Returns:
        bool: True when the topic is a robot-state stream.
    """

    return "/observation/robot_state/" in topic


def _to_scalar_frequency(value) -> float:
    """Collapse scalar or windowed frequency values into one comparable mean.

    Args:
        value: Scalar frequency or per-step frequency array.

    Returns:
        float: Comparable mean frequency value.
    """

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        finite_value = value[np.isfinite(value) & (value > 0)]
        if finite_value.size == 0:
            return 0.0
        return float(np.mean(finite_value))
    return float(value) if value else 0.0


def _fps_rule_metrics(
    freq, mean_fps_limit: float, min_fps_limit: float
) -> dict[str, float | int]:
    """Build FPS-related metrics for threshold and dropped-frame checks.

    Args:
        freq: Scalar or per-step frequency payload.
        mean_fps_limit: Configured mean FPS lower bound for the topic.
        min_fps_limit: Configured minimum FPS lower bound for the topic.

    Returns:
        dict[str, float | int]: Metrics used by FPS-related rules and logs.
    """

    interval_limit = 1.0 / min_fps_limit if min_fps_limit > 0 else float("inf")
    if isinstance(freq, np.ndarray):
        if freq.size == 0:
            mean_fps = 0.0
            min_fps = 0.0
            max_interval = 0.0
            exceed_count = 0
        else:
            finite = np.isfinite(freq) & (freq > 0)
            mean_fps = float(np.mean(freq[finite])) if finite.any() else 0.0
            min_fps = float(np.min(freq[finite])) if finite.any() else 0.0
            intervals = np.where(finite, 1.0 / freq, np.inf)
            if np.isfinite(intervals).any():
                max_interval = float(np.max(intervals))
            else:
                max_interval = float("inf")
            exceed_count = int(np.count_nonzero(intervals > interval_limit))
    else:
        mean_fps = (
            float(freq) if isinstance(freq, (int, float, np.number)) else 0.0
        )
        min_fps = mean_fps
        if mean_fps > 0:
            max_interval = 1.0 / mean_fps
            exceed_count = 1 if max_interval > interval_limit else 0
        else:
            max_interval = 0.0
            exceed_count = 0
    return {
        "mean_fps": round(mean_fps, 6),
        "min_fps": round(min_fps, 6),
        "mean_fps_limit": round(mean_fps_limit, 6),
        "min_fps_limit": round(min_fps_limit, 6),
        "max_interval": round(max_interval, 6),
        "max_interval_limit": round(interval_limit, 6)
        if np.isfinite(interval_limit)
        else float("inf"),
        "max_interval_exceed_count": exceed_count,
    }


def _timestamp_metric(actual: float, expected: float) -> dict[str, float]:
    """Build a timestamp comparison payload for logging and reporting.

    Args:
        actual: Observed timestamp-derived value in seconds.
        expected: Reference value in seconds.

    Returns:
        dict[str, float]: Timestamp comparison payload.
    """

    return {
        "actual": round(actual, 6),
        "expected": round(expected, 6),
        "delta": round(actual - expected, 6),
    }


def _joint_limit_violation_metrics(
    joint_positions: np.ndarray,
    joint_names: list[str],
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    tolerance: float,
) -> dict[str, float | int | str] | None:
    """Build URDF-based joint-limit violation metrics.

    Args:
        joint_positions: Recorded joint positions with shape ``[T, J]``.
        joint_names: Recorded joint names aligned with ``joint_positions``.
        joint_lower_limits: Lower URDF joint limits.
        joint_upper_limits: Upper URDF joint limits.
        tolerance: Symmetric tolerance applied around the URDF limits.

    Returns:
        dict[str, float | int | str] | None: Metrics for the largest
        violation, or ``None`` when all joints are within limits.
    """

    if (
        joint_lower_limits.shape != joint_upper_limits.shape
        or joint_lower_limits.ndim != 1
        or joint_lower_limits.shape[0] != joint_positions.shape[1]
    ):
        raise ValueError(
            "joint limit arrays must be 1D and aligned with joint_positions"
        )

    lower_with_tolerance = joint_lower_limits - tolerance
    upper_with_tolerance = joint_upper_limits + tolerance
    above_upper = joint_positions - upper_with_tolerance[None, :]
    below_lower = lower_with_tolerance[None, :] - joint_positions
    violation = np.maximum(above_upper, below_lower)
    max_violation = float(np.max(violation))
    if max_violation <= 0:
        return None

    step_index, joint_index = np.unravel_index(
        int(np.argmax(violation)), violation.shape
    )
    return {
        "joint_name": (
            joint_names[joint_index]
            if joint_index < len(joint_names)
            else f"joint_{joint_index}"
        ),
        "joint_index": int(joint_index),
        "step_index": int(step_index),
        "actual": round(float(joint_positions[step_index, joint_index]), 6),
        "lower_limit": round(float(lower_with_tolerance[joint_index]), 6),
        "upper_limit": round(float(upper_with_tolerance[joint_index]), 6),
        "max_violation": round(max_violation, 6),
    }


def _max_orientation_gap(
    recorded_ee_poses: np.ndarray, fk_ee_poses: np.ndarray
) -> float:
    """Return the largest rotation-angle gap between recorded and FK poses.

    Args:
        recorded_ee_poses:
            Recorded EE poses in ``[..., x, y, z, qx, qy, qz, qw]``.
        fk_ee_poses: FK EE poses in the same layout.

    Returns:
        float: Maximum angle difference in radians.
    """

    recorded_rot = Rotation.from_quat(
        recorded_ee_poses[..., 3:].reshape(-1, 4), scalar_first=False
    )
    fk_rot = Rotation.from_quat(
        fk_ee_poses[..., 3:].reshape(-1, 4), scalar_first=False
    )
    return float(np.max((fk_rot.inv() * recorded_rot).magnitude()))


def _aggregate_rule_results(
    episode_data: EpisodeData, rule_results: list[RuleResult]
) -> EpisodeInspection:
    """Collapse rule hits into an episode-level status and summary metrics.

    Args:
        episode_data: Normalized episode payload.
        rule_results: Rule results collected for the episode.

    Returns:
        EpisodeInspection: Aggregate inspection output.
    """

    blocking_failed = any(
        item.status == "fail" and item.severity == "blocking"
        for item in rule_results
    )
    has_signal = any(
        item.status in {"warning", "fail"} for item in rule_results
    )
    if blocking_failed:
        episode_status = "fail"
    elif has_signal:
        episode_status = "warning"
    else:
        episode_status = "pass"

    render_hints = []
    for item in rule_results:
        if item.status == "pass":
            continue
        metric_text = ", ".join(
            f"{key}={value}" for key, value in item.metrics.items()
        )
        hint = f"{item.status}: {item.rule_id}"
        if metric_text:
            hint = f"{hint} ({metric_text})"
        render_hints.append(hint)
    render_hints = render_hints[:3]
    status_counts = Counter(item.status for item in rule_results)
    episode_metrics = {
        "rule_hit_count": sum(item.status != "pass" for item in rule_results),
        "parse_warning_count": len(episode_data.parse_warnings),
        "warning_rule_count": status_counts.get("warning", 0),
        "failed_rule_count": status_counts.get("fail", 0),
    }
    return EpisodeInspection(
        episode_status=episode_status,
        rule_results=rule_results,
        render_hints=render_hints,
        episode_metrics=episode_metrics,
    )


def evaluate_episode_rules(
    episode_data: EpisodeData, config: InspectConfig
) -> EpisodeInspection:
    """Run the configured rule set on one normalized episode.

    Args:
        episode_data: Normalized episode payload.
        config: Inspection thresholds and tolerances.

    Returns:
        EpisodeInspection: Aggregate inspection result for the episode.
    """

    rule_results: list[RuleResult] = []

    observed_topics = episode_data.observed_topics or set(
        episode_data.topic_counts
    )
    missing_topics = [
        topic
        for topic in episode_data.required_topics
        if topic not in observed_topics
    ]
    if missing_topics:
        rule_results.append(
            RuleResult(
                rule_id="missing_topic",
                status="fail",
                severity="blocking",
                message="required topics are missing",
                metrics={"missing_topics": ",".join(sorted(missing_topics))},
            )
        )
    empty_topics = [
        topic
        for topic in episode_data.required_topics
        if topic in observed_topics
        and episode_data.topic_counts.get(topic, 0) <= 0
    ]
    if empty_topics:
        rule_results.append(
            RuleResult(
                rule_id="empty_stream",
                status="fail",
                severity="blocking",
                message="required topics have no usable messages",
                metrics={"empty_topics": ",".join(sorted(empty_topics))},
            )
        )

    if episode_data.base_time.size > 1:
        deltas = np.diff(episode_data.base_time)
        if np.any(deltas < -config.timestamp_limit):
            rule_results.append(
                RuleResult(
                    rule_id="timestamp_non_monotonic",
                    status="fail",
                    severity="blocking",
                    message="base timestamps are not strictly increasing",
                    metrics={"min_delta": float(np.min(deltas))},
                )
            )

    fps_failures: dict[str, dict[str, float | int]] = {}
    interval_failures: dict[str, dict[str, float | int]] = {}
    for topic, fps in episode_data.topic_frequencies.items():
        if _camera_topic(topic):
            mean_fps_limit = config.camera_topics_mean_fps_limit
            min_fps_limit = config.camera_topics_min_fps_limit
        elif _robot_state_topic(topic):
            mean_fps_limit = config.robot_state_topics_mean_fps_limit
            min_fps_limit = config.robot_state_topics_min_fps_limit
        else:
            continue
        metrics = _fps_rule_metrics(fps, mean_fps_limit, min_fps_limit)
        if float(metrics.get("mean_fps", 0.0)) < mean_fps_limit:
            fps_failures[topic] = metrics
        elif float(metrics.get("min_fps", 0.0)) < min_fps_limit:
            interval_failures[topic] = {
                "mean_fps": float(metrics["mean_fps"]),
                "mean_fps_limit": float(metrics["mean_fps_limit"]),
                "min_fps": float(metrics["min_fps"]),
                "min_fps_limit": float(metrics["min_fps_limit"]),
            }
    if fps_failures:
        rule_results.append(
            RuleResult(
                rule_id="fps_out_of_range",
                status="fail",
                severity="blocking",
                message="stream fps is below configured threshold",
                metrics=fps_failures,
            )
        )
    if interval_failures:
        rule_results.append(
            RuleResult(
                rule_id="interval_spike_or_drop_frame",
                status="fail",
                severity="blocking",
                message="stream min fps is below configured threshold",
                metrics=interval_failures,
            )
        )
    if (
        not episode_data.static_filter_applied
        and episode_data.base_time.size
        and episode_data.topic_summaries
    ):
        base_start = float(episode_data.base_time[0])
        base_end = float(episode_data.base_time[-1])
        base_duration = base_end - base_start
        start_mismatches: dict[str, dict[str, float]] = {}
        end_mismatches: dict[str, dict[str, float]] = {}
        duration_mismatches: dict[str, dict[str, float]] = {}

        for topic, count in episode_data.topic_counts.items():
            if count <= 0:
                continue
            summary = episode_data.topic_summaries.get(topic)
            if not summary:
                continue
            start_time_ns = summary.get("start_time_ns")
            end_time_ns = summary.get("end_time_ns")
            if start_time_ns is None or end_time_ns is None:
                continue
            actual_start = float(start_time_ns) / 1e9
            actual_end = float(end_time_ns) / 1e9
            actual_duration = actual_end - actual_start

            if abs(actual_start - base_start) > config.timestamp_limit:
                start_mismatches[topic] = _timestamp_metric(
                    actual_start, base_start
                )
            if abs(actual_end - base_end) > config.timestamp_limit:
                end_mismatches[topic] = _timestamp_metric(actual_end, base_end)
            if abs(actual_duration - base_duration) > config.timestamp_limit:
                duration_mismatches[topic] = _timestamp_metric(
                    actual_duration, base_duration
                )

        if start_mismatches:
            rule_results.append(
                RuleResult(
                    rule_id="start_ts_mismatch",
                    status="warning",
                    severity="non_blocking",
                    message=(
                        "stream start timestamps differ from base timeline"
                    ),
                    metrics=start_mismatches,
                )
            )
        if end_mismatches:
            rule_results.append(
                RuleResult(
                    rule_id="end_ts_mismatch",
                    status="warning",
                    severity="non_blocking",
                    message="stream end timestamps differ from base timeline",
                    metrics=end_mismatches,
                )
            )
        if duration_mismatches:
            rule_results.append(
                RuleResult(
                    rule_id="duration_mismatch",
                    status="warning",
                    severity="non_blocking",
                    message="stream durations differ from base timeline",
                    metrics=duration_mismatches,
                )
            )

    if not episode_data.static_filter_applied:
        alignment_mismatches: dict[str, dict[str, float]] = {}
        for topic, stats in episode_data.alignment_time_diff_stats.items():
            max_time_diff = float(stats.get("max_time_diff", 0.0))
            if max_time_diff <= config.alignment_time_diff_limit:
                continue
            alignment_mismatches[topic] = {
                "max_time_diff": round(max_time_diff, 6),
                "mean_time_diff": round(
                    float(stats.get("mean_time_diff", 0.0)), 6
                ),
                "limit": round(config.alignment_time_diff_limit, 6),
            }
        if alignment_mismatches:
            rule_results.append(
                RuleResult(
                    rule_id="alignment_time_diff_out_of_range",
                    status="warning",
                    severity="non_blocking",
                    message=(
                        "aligned timestamps deviate too much from base "
                        "timeline"
                    ),
                    metrics=alignment_mismatches,
                )
            )

    if episode_data.joint_positions.size:
        if (
            episode_data.joint_lower_limits is not None
            and episode_data.joint_upper_limits is not None
        ):
            joint_limit_metrics = _joint_limit_violation_metrics(
                joint_positions=episode_data.joint_positions,
                joint_names=episode_data.joint_limit_names,
                joint_lower_limits=np.asarray(
                    episode_data.joint_lower_limits, dtype=np.float64
                ),
                joint_upper_limits=np.asarray(
                    episode_data.joint_upper_limits, dtype=np.float64
                ),
                tolerance=config.joint_limit_tolerance,
            )
        else:
            joint_limit_metrics = None
        if joint_limit_metrics is not None:
            rule_results.append(
                RuleResult(
                    rule_id="joint_limit_violation",
                    status="fail",
                    severity="blocking",
                    message="joint position exceeds URDF joint limit",
                    metrics=joint_limit_metrics,
                )
            )
        if episode_data.joint_positions.shape[0] > 1:
            max_joint_delta = float(
                np.max(np.abs(np.diff(episode_data.joint_positions, axis=0)))
            )
            if max_joint_delta > config.joint_change_tolerance:
                rule_results.append(
                    RuleResult(
                        rule_id="joint_jump_violation",
                        status="warning",
                        severity="non_blocking",
                        message="adjacent joint delta exceeds tolerance",
                        metrics={"max_joint_delta": round(max_joint_delta, 6)},
                    )
                )

    if (
        episode_data.master_joint_positions.size
        and episode_data.joint_positions.size
        and episode_data.master_joint_positions.shape
        == episode_data.joint_positions.shape
    ):
        max_gap = float(
            np.max(
                np.abs(
                    episode_data.master_joint_positions
                    - episode_data.joint_positions
                )
            )
        )
        if max_gap > config.master_slave_joint_tolerance:
            rule_results.append(
                RuleResult(
                    rule_id="master_slave_joint_gap",
                    status="warning",
                    severity="non_blocking",
                    message="master and follower joints diverge",
                    metrics={"max_gap": max_gap},
                )
            )

    if (
        episode_data.recorded_ee_poses is not None
        and episode_data.recorded_ee_poses.size
        and episode_data.fk_ee_poses.size
        and episode_data.recorded_ee_poses.shape
        == episode_data.fk_ee_poses.shape
    ):
        position_gap = float(
            np.max(
                np.abs(
                    episode_data.recorded_ee_poses[..., :3]
                    - episode_data.fk_ee_poses[..., :3]
                )
            )
        )
        orientation_gap = _max_orientation_gap(
            episode_data.recorded_ee_poses,
            episode_data.fk_ee_poses,
        )
        if (
            position_gap > config.ee_pose_position_tolerance
            or orientation_gap > config.ee_pose_orientation_tolerance
        ):
            rule_results.append(
                RuleResult(
                    rule_id="fk_ee_pose_mismatch",
                    status="warning",
                    severity="non_blocking",
                    message="recorded EE pose differs from FK result",
                    metrics={
                        "position_gap": round(position_gap, 6),
                        "orientation_gap": round(orientation_gap, 6),
                    },
                )
            )

    return _aggregate_rule_results(episode_data, rule_results)
