# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import csv
import hashlib
import html
import json
import logging
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)
FFMPEG_SILENT_LOGLEVEL = "error"
MANUAL_REVIEW_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "manual_review.html"
)
INSPECT_ERROR_LOG_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "inspect_error_log.html"
)

REPORT_COLUMNS = [
    "mcap_path",
    "topic_integrity: missing_topics",
    "data_compression: average_size_error_count",
    "topic_fps: camera_frame_rate_error_count",
    "topic_fps: robot_state_frame_rate_error_count",
    "message_timestamps: start_ts_error_count",
    "message_timestamps: end_ts_error_count",
    "message_timestamps: duration_error_count",
    "robot_state: joint_out_of_bounds_count",
    "robot_state: joint_sudden_change_count",
    "robot_state: ee_pose_position_diff_count",
    "robot_state: ee_pose_angle_diff_count",
    "parse_exception",
    "total_error_count",
]

FULL_LOG_TOPIC_ORDER = [
    "/observation/cameras/left/color_image/image_raw",
    "/observation/cameras/middle/color_image/image_raw",
    "/observation/cameras/right/color_image/image_raw",
    "/observation/cameras/left/depth_image/image_raw",
    "/observation/cameras/middle/depth_image/image_raw",
    "/observation/cameras/right/depth_image/image_raw",
    "/observation/robot_state/left/joint",
    "/observation/robot_state/right/joint",
    "/observation/robot_state/left_master/joint",
    "/observation/robot_state/right_master/joint",
    "/observation/cameras/left/color_image/camera_info",
    "/observation/cameras/middle/color_image/camera_info",
    "/observation/cameras/right/color_image/camera_info",
    "/observation/cameras/left/depth_image/camera_info",
    "/observation/cameras/middle/depth_image/camera_info",
    "/observation/cameras/right/depth_image/camera_info",
]

TOPIC_SUMMARY_TOPIC_WIDTH = 60
TOPIC_SUMMARY_FPS_WIDTH = 13
TOPIC_SUMMARY_TS_WIDTH = 35
TOPIC_SUMMARY_DURATION_WIDTH = 12
TOPIC_SUMMARY_COUNT_WIDTH = 10
TOPIC_SUMMARY_SIZE_WIDTH = 13

MANUAL_REVIEW_RULE_SUMMARIES = {
    "missing_topic": "missing topic",
    "empty_stream": "empty stream",
    "fps_out_of_range": "fps low",
    "interval_spike_or_drop_frame": "fps spike/drop",
    "start_ts_mismatch": "start ts",
    "end_ts_mismatch": "end ts",
    "duration_mismatch": "duration mismatch",
    "alignment_time_diff_out_of_range": "time diff",
    "timestamp_non_monotonic": "timestamp order",
    "joint_limit_violation": "joint limit",
    "joint_jump_violation": "joint jump",
    "fk_ee_pose_mismatch": "ee mismatch",
    "master_slave_joint_gap": "master/follower gap",
}


def get_h264_encoder() -> str:
    """Return an available ffmpeg H.264 encoder."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    encoders = result.stdout
    for encoder in ("libx264", "libopenh264", "h264_v4l2m2m"):
        if encoder in encoders:
            return encoder
    raise RuntimeError("No supported H.264 encoder found in ffmpeg")


class FfmpegVideoWriter:
    """Stream raw BGR frames to ffmpeg and encode them as mp4."""

    def __init__(
        self,
        video_file: str,
        frame_size,
        fps: int,
        enable_ffmpeg_log: bool = False,
    ):
        width, height = frame_size
        cmd = [
            "ffmpeg",
        ]
        if not enable_ffmpeg_log:
            cmd.extend(["-hide_banner", "-loglevel", FFMPEG_SILENT_LOGLEVEL])
        cmd.extend(
            [
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                get_h264_encoder(),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                video_file,
            ]
        )
        popen_kwargs = {"stdin": subprocess.PIPE}
        if not enable_ffmpeg_log:
            popen_kwargs["stdout"] = subprocess.DEVNULL
            popen_kwargs["stderr"] = subprocess.DEVNULL
        self.process = subprocess.Popen(cmd, **popen_kwargs)
        self.stdin = self.process.stdin

    def write(self, frame: np.ndarray) -> None:
        if self.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.stdin is not None and not self.stdin.closed:
            self.stdin.close()
        returncode = self.process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with non-zero status: {returncode}"
            )


def concat_videos(
    video_files, output_path: str, enable_ffmpeg_log: bool = False
) -> None:
    """Concatenate rendered episode videos into one output mp4.

    Args:
        video_files: Individual episode video paths.
        output_path: Output directory for the concat artifact.
    """

    if not video_files:
        return
    list_file = os.path.join(output_path, "video_files.txt")
    with open(list_file, "w") as f:
        for file in video_files:
            f.write(f"file '{os.path.abspath(file)}'\n")
    cmd = [
        "ffmpeg",
    ]
    if not enable_ffmpeg_log:
        cmd.extend(["-hide_banner", "-loglevel", FFMPEG_SILENT_LOGLEVEL])
    cmd.extend(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            os.path.join(output_path, "concat_videos.mp4"),
        ]
    )
    run_kwargs = {"check": True}
    if not enable_ffmpeg_log:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(cmd, **run_kwargs)


def build_job_summary(results, output_path: str | None = None):
    """Aggregate batch-level counts and artifact lists from episode results.

    Args:
        results: Episode report payloads for the batch.
        output_path: Optional output directory used to resolve artifact paths.

    Returns:
        dict: Aggregate batch summary.
    """

    if output_path is None:
        output_path = ""
    rule_hit_counts = {}
    runtime_failure_count = 0
    video_files = []
    report_files = []
    problematic_data_files = []
    clean_data_files = []
    for item in results:
        data_file = item.get("source_path") or item.get("uuid")
        if item.get("runtime_error"):
            runtime_failure_count += 1
        if item.get("video_file"):
            video_files.append(item["video_file"])
        if item.get("report_file"):
            report_files.append(item["report_file"])
        if item.get("episode_status") == "pass":
            clean_data_files.append(data_file)
        else:
            problematic_data_files.append(data_file)
        for rule in item.get("rule_results", []):
            rule_id = (
                rule["rule_id"] if isinstance(rule, dict) else rule.rule_id
            )
            rule_hit_counts[rule_id] = rule_hit_counts.get(rule_id, 0) + 1
    return {
        "total_episodes": len(results),
        "pass_count": sum(
            item.get("episode_status") == "pass" for item in results
        ),
        "warning_count": sum(
            item.get("episode_status") == "warning" for item in results
        ),
        "fail_count": sum(
            item.get("episode_status") == "fail" for item in results
        ),
        "runtime_failure_count": runtime_failure_count,
        "parse_error_count": runtime_failure_count,
        "rule_hit_counts": rule_hit_counts,
        "report_dir": os.path.join(output_path, "reports"),
        "video_dir": output_path,
        "report_files": report_files,
        "video_files": video_files,
        "problematic_data_files": problematic_data_files,
        "clean_data_files": clean_data_files,
    }


def iter_rule_results(item: dict) -> Iterator[dict]:
    for rule in item.get("rule_results", []):
        if isinstance(rule, dict):
            yield rule
        else:
            yield rule.to_dict()


def count_topics(
    metric_value: str | list[str] | tuple[str, ...] | None,
) -> int:
    if not metric_value:
        return 0
    if isinstance(metric_value, (list, tuple, set)):
        return len([topic for topic in metric_value if str(topic).strip()])
    return len([topic for topic in metric_value.split(",") if topic])


def build_report_row(item: dict) -> dict[str, int | float | str]:
    row = {column: 0 for column in REPORT_COLUMNS}
    row["mcap_path"] = (
        item.get("mcap_path")
        or item.get("source_path")
        or item.get("uuid")
        or ""
    )
    row["parse_exception"] = bool(item.get("runtime_error"))

    for rule in iter_rule_results(item):
        rule_id = rule.get("rule_id")
        metrics = rule.get("metrics") or {}
        if rule_id == "missing_topic":
            row["topic_integrity: missing_topics"] += count_topics(
                metrics.get("missing_topics")
            )
        elif rule_id == "fps_out_of_range":
            for topic in metrics:
                if "/observation/cameras/" in topic:
                    row["topic_fps: camera_frame_rate_error_count"] += 1
                elif "/observation/robot_state/" in topic:
                    row["topic_fps: robot_state_frame_rate_error_count"] += 1
        elif rule_id == "interval_spike_or_drop_frame":
            for topic in metrics:
                if "/observation/cameras/" in topic:
                    row["topic_fps: camera_frame_rate_error_count"] += 1
                elif "/observation/robot_state/" in topic:
                    row["topic_fps: robot_state_frame_rate_error_count"] += 1
        elif rule_id == "start_ts_mismatch":
            row["message_timestamps: start_ts_error_count"] += len(metrics)
        elif rule_id == "end_ts_mismatch":
            row["message_timestamps: end_ts_error_count"] += len(metrics)
        elif rule_id in {
            "duration_mismatch",
            "alignment_time_diff_out_of_range",
        }:
            row["message_timestamps: duration_error_count"] += len(metrics)
        elif rule_id == "timestamp_non_monotonic":
            row["message_timestamps: duration_error_count"] += 1
        elif rule_id == "joint_limit_violation":
            row["robot_state: joint_out_of_bounds_count"] += 1
        elif rule_id == "joint_jump_violation":
            row["robot_state: joint_sudden_change_count"] += 1
        elif rule_id == "fk_ee_pose_mismatch":
            if metrics.get("position_gap", 0) > 0:
                row["robot_state: ee_pose_position_diff_count"] += 1
            if metrics.get("orientation_gap", 0) > 0:
                row["robot_state: ee_pose_angle_diff_count"] += 1

    row["total_error_count"] = sum(
        int(row[column])
        for column in REPORT_COLUMNS
        if column not in {"mcap_path", "total_error_count"}
    )
    return row


def write_csv_report(output_path: str, rows: list[dict[str, Any]]) -> str:
    csv_path = os.path.join(output_path, "inspection_report.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_html_report(output_path: str, rows: list[dict[str, Any]]) -> str:
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive CSV Table</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/colreorder/1.7.0/css/colReorder.dataTables.min.css">
        <script type="text/javascript" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/colreorder/1.7.0/js/dataTables.colReorder.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Interactive CSV Table</h1>
        <table id="csvTable" class="display">
            <thead>
                <tr>
                    <th>序号</th>
                    {thead}
                </tr>
            </thead>
            <tbody>
                {tbody}
            </tbody>
        </table>
        <script>
            $(document).ready(function() {{
                $('#csvTable').DataTable({{
                    pageLength: 500,
                    lengthMenu: [
                        [10, 25, 50, 100, 500, -1],
                        [10, 25, 50, 100, 500, "All"]
                    ],
                    colReorder: true,
                    columnDefs: [
                        {{
                            targets: 0,
                            render: function(data, type, row, meta) {{
                                return meta.row + 1;
                            }}
                        }}
                    ]
                }});
            }});
        </script>
    </body>
    </html>
    """
    thead = "".join(f"<th>{column}</th>" for column in REPORT_COLUMNS)
    tbody_parts = []
    for row in rows:
        cells = "".join(f"<td>{row[column]}</td>" for column in REPORT_COLUMNS)
        tbody_parts.append(f"<tr><td></td>{cells}</tr>")
    html_path = os.path.join(output_path, "inspection_report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(
            html_template.format(
                thead=thead,
                tbody="".join(tbody_parts),
            )
        )
    return html_path


def is_quick_screen_pass_row(row: dict[str, Any]) -> bool:
    """Return whether a row passes the simplified PASS/ERROR screening."""

    return (
        row["topic_integrity: missing_topics"] == 0
        and row["topic_fps: camera_frame_rate_error_count"] == 0
        and row["topic_fps: robot_state_frame_rate_error_count"] == 0
        and row["parse_exception"] == 0
    )


def write_mcap_lists(output_path: str, rows: list[dict[str, Any]]) -> None:
    """Write one merged PASS/ERROR list using the simplified screening rules.

    Args:
        output_path: Output directory for the merged list.
        rows: Report rows derived from episode results.
    """

    merged_path = os.path.join(output_path, "inspect_mcap_result.log")
    no_error_rows = [row for row in rows if is_quick_screen_pass_row(row)]
    error_rows = [row for row in rows if row not in no_error_rows]
    with open(merged_path, "w", encoding="utf-8") as fh:
        for row in error_rows:
            fh.write(f"ERROR {row['mcap_path']}\n")
        for row in no_error_rows:
            fh.write(f"PASS {row['mcap_path']}\n")


def _count_rendered_frames(num_steps: int, vis_interval: int) -> int:
    """Return the number of frames rendered for one episode video."""

    step_interval = max(1, int(vis_interval))
    if num_steps <= 0:
        return 0
    return (num_steps + step_interval - 1) // step_interval


def build_manual_review_timeline(
    results: list[dict[str, Any]], vis_interval: int, vis_fps: int
) -> dict[str, Any]:
    """Build a concat timeline for manual review playback.

    Args:
        results: Episode report payloads with rendered video paths.
        vis_interval: Frame sampling interval used during rendering.
        vis_fps: Rendering FPS configured for the checker.

    Returns:
        dict[str, Any]: Timeline payload written next to the concat video.
    """

    render_fps = max(1, round(vis_fps / max(1, vis_interval)))
    segments = []
    current_start = 0.0
    for index, item in enumerate(
        item for item in results if item.get("video_file")
    ):
        video_file = item["video_file"]
        num_steps = int(item.get("num_steps", 0))
        frame_count = _count_rendered_frames(num_steps, vis_interval)
        duration_sec = frame_count / render_fps if frame_count > 0 else 0.0
        if duration_sec <= 0:
            continue
        mcap_id = item.get("uuid") or item.get("mcap_path") or ""
        segment = {
            "index": index,
            "mcap_id": mcap_id,
            "mcap_path": item.get("mcap_path")
            or item.get("source_path")
            or "",
            "video_file": os.path.basename(video_file),
            "start_sec": round(current_start, 6),
            "end_sec": round(current_start + duration_sec, 6),
            "duration_sec": round(duration_sec, 6),
        }
        segments.append(segment)
        current_start += duration_sec
    return {
        "concat_video": "concat_videos.mp4",
        "generated_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "segments": segments,
    }


def build_manual_review_error_log_anchor(source_id: str) -> str:
    """Build a stable DOM anchor for one error-log block."""

    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
    return f"error-{digest}"


def build_manual_review_note_summary(item: dict[str, Any]) -> str:
    """Build a short system-generated summary for manual review."""

    labels: list[str] = []
    if item.get("runtime_error"):
        labels.append("runtime error")
    for rule in iter_rule_results(item):
        rule_id = str(rule.get("rule_id"))
        label = MANUAL_REVIEW_RULE_SUMMARIES.get(rule_id)
        if not label or label in labels:
            continue
        labels.append(label)
        if len(labels) >= 3:
            break
    return " / ".join(labels)


def build_manual_review_failures(
    results: list[dict[str, Any]], timeline: dict[str, Any]
) -> dict[str, Any]:
    """Build the initial manual review failure payload.

    Args:
        results: Episode report payloads for the batch.
        timeline: Concat timeline used by the review page.

    Returns:
        dict[str, Any]: Failure payload initialized for human review.
    """

    segment_by_mcap_id = {
        segment["mcap_id"]: segment for segment in timeline.get("segments", [])
    }
    items = []
    for item in results:
        report_row = build_report_row(item)
        if is_quick_screen_pass_row(report_row):
            continue
        mcap_id = item.get("uuid") or report_row["mcap_path"] or ""
        segment = segment_by_mcap_id.get(mcap_id)
        items.append(
            {
                "mcap_id": mcap_id,
                "mcap_path": report_row["mcap_path"],
                "video_file": segment["video_file"] if segment else "",
                "mark_time_sec": segment["start_sec"] if segment else None,
                "note_summary": build_manual_review_note_summary(item),
                "error_log_anchor": (
                    build_manual_review_error_log_anchor(
                        report_row["mcap_path"] or mcap_id
                    )
                    if segment
                    else ""
                ),
                "note": "",
            }
        )
    items.sort(
        key=lambda item: (
            float("inf")
            if item["mark_time_sec"] is None
            else item["mark_time_sec"],
            item["mcap_id"],
        )
    )

    return {
        "concat_video": "concat_videos.mp4",
        "updated_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "items": items,
    }


def render_manual_review_page(
    timeline: dict[str, Any], failures: dict[str, Any], storage_key: str
) -> str:
    """Render the standalone manual review page with embedded data."""

    template_html = MANUAL_REVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    bootstrap = {
        "timeline": timeline,
        "failures": failures,
        "storage_key": storage_key,
    }
    return template_html.replace(
        "__MANUAL_REVIEW_BOOTSTRAP__",
        json.dumps(bootstrap, ensure_ascii=False),
    )


def render_inspect_error_log_page(
    blocks: list[dict[str, str]], generated_at: str
) -> str:
    """Render a standalone HTML view for inspect_error_log blocks."""

    template_html = INSPECT_ERROR_LOG_TEMPLATE_PATH.read_text(encoding="utf-8")
    if blocks:
        blocks_html = "\n".join(
            "".join(
                [
                    '<section class="error-block" '
                    f'id="{html.escape(block["anchor"])}">',
                    '<div class="error-head">',
                    '<a class="error-anchor" '
                    f'href="#{html.escape(block["anchor"])}">#</a>',
                    f"<code>{html.escape(block['mcap_path'])}</code>",
                    "</div>",
                    (
                        '<pre class="error-body">'
                        f"{html.escape(block['body'])}"
                        "</pre>"
                    ),
                    "</section>",
                ]
            )
            for block in blocks
        )
    else:
        blocks_html = (
            '<div class="empty-state">'
            "当前没有可展示的 inspect error block。"
            "</div>"
        )
    return template_html.replace(
        "__INSPECT_ERROR_LOG_GENERATED_AT__", generated_at
    ).replace("__INSPECT_ERROR_LOG_BLOCKS__", blocks_html)


def write_inspect_error_log_page(
    output_path: str, blocks: list[dict[str, str]]
) -> str:
    """Write the HTML companion page for inspect_error_log.log."""

    html_path = os.path.join(output_path, "inspect_error_log.html")
    generated_at = (
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )
    page_html = render_inspect_error_log_page(blocks, generated_at)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(page_html)
    return html_path


def write_manual_review_page(
    output_path: str,
    timeline: dict[str, Any],
    failures: dict[str, Any],
) -> str:
    """Write a standalone manual review HTML page into one output directory."""

    html_path = os.path.join(output_path, "manual_review.html")
    failure_seed = "|".join(
        (
            f"{item['mcap_id']}::{item.get('note_summary', '')}"
            f"::{item.get('error_log_anchor', '')}"
        )
        for item in failures.get("items", [])
    )
    failure_seed_digest = hashlib.sha1(
        failure_seed.encode("utf-8")
    ).hexdigest()[:12]
    concat_video = timeline.get("concat_video", "concat_videos.mp4")
    storage_key = (
        f"manualReviewFailures::{concat_video}"
        f"::{len(timeline.get('segments', []))}"
        f"::{len(failures.get('items', []))}"
        f"::{failure_seed_digest}"
    )
    template_html = render_manual_review_page(
        timeline,
        failures,
        storage_key,
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(template_html)
    return html_path


def write_manual_review_artifacts(
    output_path: str,
    results: list[dict[str, Any]],
    vis_interval: int,
    vis_fps: int,
) -> tuple[str, str, str] | tuple[None, None, None]:
    """Write manual-review timeline and failure placeholders.

    Args:
        output_path: Output directory for the manual review files.
        results: Episode report payloads for the batch.
        vis_interval: Frame sampling interval used during rendering.
        vis_fps: Rendering FPS configured for the checker.

    Returns:
        tuple[str, str, str] | tuple[None, None, None]: Paths to the written
        HTML page and JSON files.
    """

    timeline = build_manual_review_timeline(results, vis_interval, vis_fps)
    if not timeline["segments"]:
        return None, None, None
    failures = build_manual_review_failures(results, timeline)
    os.makedirs(output_path, exist_ok=True)
    page_path = write_manual_review_page(output_path, timeline, failures)
    timeline_path = os.path.join(output_path, "manual_review_timeline.json")
    failures_path = os.path.join(output_path, "manual_review_failures.json")
    with open(timeline_path, "w", encoding="utf-8") as fh:
        json.dump(timeline, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(failures_path, "w", encoding="utf-8") as fh:
        json.dump(failures, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return page_path, timeline_path, failures_path


def format_numeric_detail_table(
    label_header: str, rows: list[tuple[str, float, float, float]]
) -> list[str]:
    """Render aligned numeric rule details as a plain-text table.

    Args:
        label_header: Header for the left-most label column.
        rows: Numeric rows in ``(label, actual, limit, gap)`` format.

    Returns:
        list[str]: Rendered table lines.
    """

    if not rows:
        return []
    label_width = max(
        len(label_header),
        max(len(label) for label, *_ in rows),
    )
    numeric_headers = ("actual", "limit", "gap")
    numeric_width = max(
        10,
        *(
            len(f"{value:.6f}")
            for _, actual, limit, gap in rows
            for value in (actual, limit, gap)
        ),
    )
    header = (
        f"  {label_header:<{label_width}}  "
        f"{numeric_headers[0]:>{numeric_width}}  "
        f"{numeric_headers[1]:>{numeric_width}}  "
        f"{numeric_headers[2]:>{numeric_width}}"
    )
    separator = (
        f"  {'-' * label_width}  "
        f"{'-' * numeric_width}  "
        f"{'-' * numeric_width}  "
        f"{'-' * numeric_width}"
    )
    lines = [header, separator]
    for label, actual, limit, gap in rows:
        lines.append(
            f"  {label:<{label_width}}  "
            f"{actual:>{numeric_width}.6f}  "
            f"{limit:>{numeric_width}.6f}  "
            f"{gap:>{numeric_width}.6f}"
        )
    return lines


def format_text_detail_table(
    headers: tuple[str, ...], rows: list[tuple[str, ...]]
) -> list[str]:
    """Render aligned text rule details as a plain-text table.

    Args:
        headers: Table headers.
        rows: Text rows matching the header layout.

    Returns:
        list[str]: Rendered table lines.
    """

    if not rows:
        return []
    widths = [
        max(len(headers[idx]), max(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header = "  " + "  ".join(
        f"{headers[idx]:<{widths[idx]}}" for idx in range(len(headers))
    )
    separator = "  " + "  ".join(
        "-" * widths[idx] for idx in range(len(headers))
    )
    lines = [header, separator]
    for row in rows:
        lines.append(
            "  "
            + "  ".join(
                f"{row[idx]:<{widths[idx]}}" for idx in range(len(headers))
            )
        )
    return lines


def format_rule_detail_lines(rule: dict, inspect_config) -> list[str]:
    """Expand rule metrics into human-readable full-log detail lines.

    Args:
        rule: Serialized rule result.
        inspect_config: Effective inspection thresholds.

    Returns:
        list[str]: Detail lines for the full log.
    """

    if inspect_config is None:
        return []

    rule_id = rule.get("rule_id")
    metrics = rule.get("metrics") or {}
    if rule_id == "fps_out_of_range":
        lines = []
        numeric_rows = []
        for topic in sorted(metrics):
            actual = metrics[topic]
            if "/observation/cameras/" in topic:
                mean_limit = inspect_config.camera_topics_mean_fps_limit
                min_limit = inspect_config.camera_topics_min_fps_limit
            elif "/observation/robot_state/" in topic:
                mean_limit = inspect_config.robot_state_topics_mean_fps_limit
                min_limit = inspect_config.robot_state_topics_min_fps_limit
            else:
                continue
            if isinstance(actual, dict):
                mean_fps = float(actual.get("mean_fps", 0.0))
                min_fps = float(actual.get("min_fps", 0.0))
                max_interval = float(actual.get("max_interval", 0.0))
                max_interval_limit = float(
                    actual.get("max_interval_limit", 0.0)
                )
                exceed_count = int(actual.get("max_interval_exceed_count", 0))
                lines.append(
                    f"  Topic '{topic}': mean_fps={mean_fps:.6f}, "
                    f"mean_fps_limit={mean_limit:.6f}, "
                    f"min_fps={min_fps:.6f}, min_fps_limit={min_limit:.6f}, "
                    f"max_interval={max_interval:.6f}, "
                    f"max_interval_limit={max_interval_limit:.6f}, "
                    f"max_interval_exceed_count={exceed_count}"
                )
                continue
            actual_value = float(actual)
            numeric_rows.append(
                (topic, actual_value, mean_limit, actual_value - mean_limit)
            )
        return lines + format_numeric_detail_table("topic", numeric_rows)
    if rule_id == "interval_spike_or_drop_frame":
        lines = []
        for topic in sorted(metrics):
            actual = metrics[topic]
            if "/observation/cameras/" in topic:
                min_limit = inspect_config.camera_topics_min_fps_limit
            elif "/observation/robot_state/" in topic:
                min_limit = inspect_config.robot_state_topics_min_fps_limit
            else:
                min_limit = float(actual.get("min_fps_limit", 0.0))
            lines.append(
                f"  Topic '{topic}': "
                f"mean_fps={float(actual.get('mean_fps', 0.0)):.6f}, "
                "mean_fps_limit="
                f"{float(actual.get('mean_fps_limit', 0.0)):.6f}, "
                f"min_fps={float(actual.get('min_fps', 0.0)):.6f}, "
                f"min_fps_limit={min_limit:.6f}"
            )
        return lines
    if rule_id in {
        "start_ts_mismatch",
        "end_ts_mismatch",
        "duration_mismatch",
    }:
        label_map = {
            "start_ts_mismatch": ("actual_start", "expected_start"),
            "end_ts_mismatch": ("actual_end", "expected_end"),
            "duration_mismatch": ("actual_duration", "expected_duration"),
        }
        actual_label, expected_label = label_map[rule_id]
        lines = []
        for topic in sorted(metrics):
            actual = metrics[topic]
            lines.append(
                f"  Topic '{topic}': "
                f"{actual_label}={float(actual.get('actual', 0.0)):.6f}, "
                f"{expected_label}={float(actual.get('expected', 0.0)):.6f}, "
                f"delta={float(actual.get('delta', 0.0)):.6f}"
            )
        return lines
    if rule_id == "alignment_time_diff_out_of_range":
        lines = []
        for topic in sorted(metrics):
            actual = metrics[topic]
            lines.append(
                f"  Topic '{topic}': "
                "max_time_diff="
                f"{float(actual.get('max_time_diff', 0.0)):.6f}, "
                "mean_time_diff="
                f"{float(actual.get('mean_time_diff', 0.0)):.6f}, "
                f"limit={float(actual.get('limit', 0.0)):.6f}"
            )
        return lines

    if rule_id == "joint_jump_violation" and "max_joint_delta" in metrics:
        actual = float(metrics["max_joint_delta"])
        limit = inspect_config.joint_change_tolerance
        return format_numeric_detail_table(
            "metric",
            [("max_joint_delta", actual, limit, actual - limit)],
        )
    if rule_id == "master_slave_joint_gap" and "max_gap" in metrics:
        actual = float(metrics["max_gap"])
        limit = inspect_config.master_slave_joint_tolerance
        return format_numeric_detail_table(
            "metric",
            [("max_gap", actual, limit, actual - limit)],
        )
    if rule_id == "fk_ee_pose_mismatch":
        rows = []
        if "position_gap" in metrics:
            actual = float(metrics["position_gap"])
            limit = inspect_config.ee_pose_position_tolerance
            rows.append(("position_gap", actual, limit, actual - limit))
        if "orientation_gap" in metrics:
            actual = float(metrics["orientation_gap"])
            limit = inspect_config.ee_pose_orientation_tolerance
            rows.append(("orientation_gap", actual, limit, actual - limit))
        return format_numeric_detail_table("metric", rows)
    if rule_id == "joint_limit_violation":
        if {
            "joint_name",
            "step_index",
            "joint_index",
            "actual",
            "lower_limit",
            "upper_limit",
            "max_violation",
        }.issubset(metrics):
            return format_text_detail_table(
                (
                    "joint",
                    "step",
                    "index",
                    "actual",
                    "lower",
                    "upper",
                    "violation",
                ),
                [
                    (
                        str(metrics["joint_name"]),
                        str(metrics["step_index"]),
                        str(metrics["joint_index"]),
                        f"{float(metrics['actual']):.6f}",
                        f"{float(metrics['lower_limit']):.6f}",
                        f"{float(metrics['upper_limit']):.6f}",
                        f"{float(metrics['max_violation']):.6f}",
                    )
                ],
            )
        if "max_joint_value" in metrics:
            actual = float(metrics["max_joint_value"])
            limit = np.pi * 2 + inspect_config.joint_limit_tolerance
            return format_numeric_detail_table(
                "metric",
                [("max_joint_value", actual, limit, actual - limit)],
            )
        return []
    if rule_id == "timestamp_non_monotonic" and "min_delta" in metrics:
        actual = float(metrics["min_delta"])
        limit = -inspect_config.timestamp_limit
        return format_numeric_detail_table(
            "metric",
            [("min_delta", actual, limit, actual - limit)],
        )
    if rule_id == "missing_topic" and "missing_topics" in metrics:
        topics = [
            topic.strip()
            for topic in str(metrics["missing_topics"]).split(",")
            if topic.strip()
        ]
        return [
            f"Error: Topic '{topic}' is missing in the mcap file."
            for topic in topics
        ]
    if rule_id == "empty_stream" and "empty_topics" in metrics:
        topics = [
            topic.strip()
            for topic in str(metrics["empty_topics"]).split(",")
            if topic.strip()
        ]
        return [
            f"Error: Topic '{topic}' has no usable messages in the mcap file."
            for topic in topics
        ]
    return []


def write_rule_block(fh, rule: dict, inspect_config) -> None:
    """Write one rule block with an Error/Warning prefix and detail lines.

    Args:
        fh: Writable file handle.
        rule: Serialized rule result.
        inspect_config: Effective inspection thresholds.
    """

    level = "Error" if rule.get("status") == "fail" else "Warning"
    fh.write(f"{level}: {rule.get('rule_id')}: {rule.get('message', '')}\n")
    for detail_line in format_rule_detail_lines(rule, inspect_config):
        fh.write(f"{detail_line}\n")
    fh.write("\n")


def format_timestamp(timestamp_ns: int | None) -> str:
    """Format a nanosecond timestamp as an RFC3339 UTC string.

    Args:
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        str: RFC3339-formatted UTC timestamp, or an empty string.
    """

    if timestamp_ns is None:
        return ""
    sec = timestamp_ns // int(1e9)
    nsec = timestamp_ns % int(1e9)
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{nsec:09d}Z"


def topic_sort_key(topic: str) -> tuple[int, str]:
    """Sort known topics in a stable, inspection-friendly order.

    Args:
        topic: Topic name.

    Returns:
        tuple[int, str]: Sort key for inspection log ordering.
    """

    try:
        return (FULL_LOG_TOPIC_ORDER.index(topic), topic)
    except ValueError:
        return (len(FULL_LOG_TOPIC_ORDER), topic)


def iter_logged_topics(
    topic_summaries: dict[str, dict],
) -> list[tuple[str, dict]]:
    """Filter and sort topics that should appear in the full log table.

    Args:
        topic_summaries: Per-topic summary payloads.

    Returns:
        list[tuple[str, dict]]: Sorted topic-summary pairs.
    """

    return sorted(
        (
            (topic, summary)
            for topic, summary in topic_summaries.items()
            if topic != "/tf_static" and not topic.endswith("/end_pose")
        ),
        key=lambda item: topic_sort_key(item[0]),
    )


def _topic_summary_has_timestamps(summary: dict) -> bool:
    return (
        summary.get("start_time_ns") is not None
        and summary.get("end_time_ns") is not None
    )


def format_topic_summary_line(topic: str, summary: dict) -> str:
    """Format one topic summary row for the full log FPS table.

    Args:
        topic: Topic name.
        summary: Topic summary payload.

    Returns:
        str: Rendered topic summary row.
    """

    duration = (
        (summary["end_time_ns"] - summary["start_time_ns"]) / 1e9
        if summary["count"] > 1 and _topic_summary_has_timestamps(summary)
        else 0.0
    )
    mean_fps = summary.get("mean_frequency", 0.0)
    min_fps = summary.get("min_frequency", 0.0)
    size_mb = summary["size_bytes"] / 1e6
    return (
        f"{topic:^{TOPIC_SUMMARY_TOPIC_WIDTH}} | "
        f"{f'{mean_fps:.3f} hz':^{TOPIC_SUMMARY_FPS_WIDTH}} | "
        f"{f'{min_fps:.3f} hz':^{TOPIC_SUMMARY_FPS_WIDTH}} | "
        f"{format_timestamp(summary['start_time_ns']):^{TOPIC_SUMMARY_TS_WIDTH}}"
        " | "
        f"{format_timestamp(summary['end_time_ns']):^{TOPIC_SUMMARY_TS_WIDTH}}"
        " | "
        f"{f'{duration:.3f} s':^{TOPIC_SUMMARY_DURATION_WIDTH}} | "
        f"{str(summary['count']):^{TOPIC_SUMMARY_COUNT_WIDTH}} | "
        f"{f'{size_mb:.3f} MB':^{TOPIC_SUMMARY_SIZE_WIDTH}}"
    )


def format_topic_summary_header() -> str:
    """Format the topic summary table header using the same column widths."""

    return (
        f"{'Topic':^{TOPIC_SUMMARY_TOPIC_WIDTH}} | "
        f"{'Mean FPS':^{TOPIC_SUMMARY_FPS_WIDTH}} | "
        f"{'Min FPS':^{TOPIC_SUMMARY_FPS_WIDTH}} | "
        f"{'Start TS':^{TOPIC_SUMMARY_TS_WIDTH}} | "
        f"{'End TS':^{TOPIC_SUMMARY_TS_WIDTH}} | "
        f"{'Duration':^{TOPIC_SUMMARY_DURATION_WIDTH}} | "
        f"{'Count':^{TOPIC_SUMMARY_COUNT_WIDTH}} | "
        f"{'Size':^{TOPIC_SUMMARY_SIZE_WIDTH}}"
    )


def write_full_log_episode(fh, item: dict, inspect_config) -> None:
    """Write the full inspection log block for one episode.

    Args:
        fh: Writable file handle.
        item: Serialized episode report with logging payloads.
        inspect_config: Effective inspection thresholds.
    """

    mcap_path = (
        item.get("mcap_path")
        or item.get("source_path")
        or item.get("uuid")
        or ""
    )
    topic_summaries = item.get("topic_summaries") or {}
    separator = "=" * 24
    fh.write(f"{separator} BEGIN MCAP: {mcap_path} {separator}\n")
    if not topic_summaries:
        fh.write(f"Starting data checks: mcap_path is {mcap_path}\n")
        for rule in iter_rule_results(item):
            if rule.get("status") in {"warning", "fail"}:
                write_rule_block(fh, rule, inspect_config)
        if item.get("runtime_error"):
            fh.write(f"ERROR parse_exception: {item['runtime_error']}\n")
        fh.write(f"{separator} END MCAP: {mcap_path} {separator}\n")
        return

    rule_results = list(iter_rule_results(item))
    rule_map = {rule.get("rule_id"): rule for rule in rule_results}
    logged_topics = iter_logged_topics(topic_summaries)
    all_steps = item.get("all_steps")
    non_static_steps = item.get("non_static_steps")

    fh.write(f"Starting data checks: mcap_path is {mcap_path}\n")
    if all_steps is not None and non_static_steps is not None:
        fh.write(
            f"all steps: {int(all_steps)}, non static steps: "
            f"{int(non_static_steps)}\n"
        )
    fh.write("Checking topic integrity...\n")
    missing_rule = rule_map.get("missing_topic")
    if missing_rule:
        missing_topics = [
            topic.strip()
            for topic in str(
                (missing_rule.get("metrics") or {}).get("missing_topics", "")
            ).split(",")
            if topic.strip()
        ]
        for topic in missing_topics:
            fh.write(f"Error: Topic '{topic}' is missing in the mcap file.\n")
        fh.write(f"Error: Missing {len(missing_topics)} topic(s).\n")
    else:
        fh.write("All topics are present in the mcap file.\n")
    empty_rule = rule_map.get("empty_stream")
    if empty_rule:
        empty_topics = [
            topic.strip()
            for topic in str(
                (empty_rule.get("metrics") or {}).get("empty_topics", "")
            ).split(",")
            if topic.strip()
        ]
        for topic in empty_topics:
            fh.write(
                "Error: Topic "
                f"'{topic}' has no usable messages in the mcap file.\n"
            )

    fh.write("Checking camera FPS...\n")
    fh.write(f"{format_topic_summary_header()}\n")
    for topic, summary in logged_topics:
        fh.write(f"{format_topic_summary_line(topic, summary)}\n")
    fps_rule = rule_map.get("fps_out_of_range")
    if fps_rule:
        for topic, actual in sorted((fps_rule.get("metrics") or {}).items()):
            if "/observation/cameras/" in topic:
                mean_limit = inspect_config.camera_topics_mean_fps_limit
            elif "/observation/robot_state/" in topic:
                mean_limit = inspect_config.robot_state_topics_mean_fps_limit
            else:
                continue
            if isinstance(actual, dict):
                mean_fps = float(actual.get("mean_fps", 0.0))
                min_fps = float(actual.get("min_fps", 0.0))
                max_interval = float(actual.get("max_interval", 0.0))
                max_interval_exceed_count = int(
                    actual.get("max_interval_exceed_count", 0)
                )
                fh.write(
                    "Error: Topic "
                    f"'{topic}' has a frame rate below the limit. "
                    f"mean_fps={mean_fps}, mean_fps_limit={mean_limit}, "
                    f"min_fps={min_fps}, "
                    f"max_interval={max_interval}, "
                    f"max_interval_exceed_count={max_interval_exceed_count}\n"
                )
                continue
            fh.write(
                f"Error: Topic '{topic}' has a frame rate below the limit. "
                f"mean_fps is {actual}, mean_fps_limit is {mean_limit}\n"
            )
    interval_rule = rule_map.get("interval_spike_or_drop_frame")
    if interval_rule:
        for topic, actual in sorted(
            (interval_rule.get("metrics") or {}).items()
        ):
            if "/observation/cameras/" in topic:
                min_limit = inspect_config.camera_topics_min_fps_limit
            elif "/observation/robot_state/" in topic:
                min_limit = inspect_config.robot_state_topics_min_fps_limit
            else:
                min_limit = float(actual.get("min_fps_limit", 0.0))
            fh.write(
                "Error: Topic "
                f"'{topic}' has a min frame rate below the limit. "
                f"mean_fps={actual.get('mean_fps')}, "
                f"min_fps={actual.get('min_fps')}, "
                f"min_fps_limit={min_limit}\n"
            )

    fh.write("Checking data compression...\n")
    total_size = sum(summary["size_bytes"] for _, summary in logged_topics)
    fh.write(f"Total size of the mcap file is {total_size / 1e6:.3f} MB.\n")
    timed_topics = [
        (topic, summary)
        for topic, summary in logged_topics
        if _topic_summary_has_timestamps(summary)
    ]
    average_duration = (
        sum(
            (summary["end_time_ns"] - summary["start_time_ns"]) / 1e9
            for _, summary in timed_topics
        )
        / len(timed_topics)
        if timed_topics
        else 0.0
    )
    average_size = (
        total_size / average_duration if average_duration > 0 else 0.0
    )
    fh.write(
        f"Average size of the mcap file is {average_size / 1e6:.3f} MB/s.\n"
    )
    if average_size > 100 * 1e6:
        fh.write("Warning: Data Average Size is too large.\n")

    fh.write("Checking message timestamps...\n")
    start_ns = (
        min(summary["start_time_ns"] for _, summary in timed_topics)
        if timed_topics
        else None
    )
    end_ns = (
        max(summary["end_time_ns"] for _, summary in timed_topics)
        if timed_topics
        else None
    )
    duration = (
        (end_ns - start_ns) / 1e9
        if start_ns is not None and end_ns is not None
        else 0.0
    )
    fh.write(f"Minimum start time is {format_timestamp(start_ns)}.\n")
    fh.write(f"Maximum end time is {format_timestamp(end_ns)}.\n")
    fh.write(f"Total duration is {duration} s.\n")
    timestamp_rule = rule_map.get("timestamp_non_monotonic")
    if timestamp_rule:
        min_delta = (timestamp_rule.get("metrics") or {}).get("min_delta")
        fh.write(
            "Error: base timestamps are not strictly increasing. "
            f"min_delta is {min_delta} s.\n"
        )
    start_rule = rule_map.get("start_ts_mismatch")
    if start_rule:
        for topic, metric in sorted((start_rule.get("metrics") or {}).items()):
            fh.write(
                "Warning: start timestamp of topic "
                f"'{topic}' differs from base timeline. "
                f"expected_start={metric.get('expected')}, "
                f"actual_start={metric.get('actual')}, "
                f"delta={metric.get('delta')} s.\n"
            )
    end_rule = rule_map.get("end_ts_mismatch")
    if end_rule:
        for topic, metric in sorted((end_rule.get("metrics") or {}).items()):
            fh.write(
                "Warning: end timestamp of topic "
                f"'{topic}' differs from base timeline. "
                f"expected_end={metric.get('expected')}, "
                f"actual_end={metric.get('actual')}, "
                f"delta={metric.get('delta')} s.\n"
            )
    duration_rule = rule_map.get("duration_mismatch")
    if duration_rule:
        for topic, metric in sorted(
            (duration_rule.get("metrics") or {}).items()
        ):
            fh.write(
                "Warning: duration of topic "
                f"'{topic}' differs from base timeline. "
                f"expected_duration={metric.get('expected')}, "
                f"actual_duration={metric.get('actual')}, "
                f"delta={metric.get('delta')} s.\n"
            )
    alignment_rule = rule_map.get("alignment_time_diff_out_of_range")
    if alignment_rule:
        for topic, metric in sorted(
            (alignment_rule.get("metrics") or {}).items()
        ):
            fh.write(
                "Warning: aligned timestamps of topic "
                f"'{topic}' exceed tolerance. "
                f"max_time_diff={metric.get('max_time_diff')}, "
                f"mean_time_diff={metric.get('mean_time_diff')}, "
                f"limit={metric.get('limit')} s.\n"
            )

    fh.write("Checking robot state with URDF joint limit...\n")
    fh.write("Checking joint state smoothness...\n")
    fh.write("Checking end-effector pose and joint state matchness...\n")
    if "joint_limit_violation" in rule_map:
        joint_limit_metrics = (
            rule_map["joint_limit_violation"].get("metrics") or {}
        )
        if {
            "joint_name",
            "step_index",
            "joint_index",
            "actual",
            "lower_limit",
            "upper_limit",
            "max_violation",
        }.issubset(joint_limit_metrics):
            fh.write(
                "Error: Joint position exceeds URDF joint limit, "
                f"joint={joint_limit_metrics['joint_name']}, "
                f"step_index={joint_limit_metrics['step_index']}, "
                f"joint_index={joint_limit_metrics['joint_index']}, "
                f"actual={joint_limit_metrics['actual']}, "
                f"lower_limit={joint_limit_metrics['lower_limit']}, "
                f"upper_limit={joint_limit_metrics['upper_limit']}, "
                "max_violation="
                f"{joint_limit_metrics['max_violation']}\n"
            )
        else:
            max_joint_value = joint_limit_metrics.get("max_joint_value")
            fh.write(
                "Error: Joint position exceeds configured limit, "
                f"max_joint_value: {max_joint_value}\n"
            )
    else:
        fh.write("All joint positions are within the limits.\n")
    if "joint_jump_violation" in rule_map:
        max_joint_delta = (
            rule_map["joint_jump_violation"].get("metrics") or {}
        ).get("max_joint_delta")
        fh.write(
            "Warning: Adjacent joint delta exceeds tolerance, "
            f"max_joint_delta: {max_joint_delta}\n"
        )
    else:
        fh.write("All joint positions are smooth.\n")
    fk_rule = rule_map.get("fk_ee_pose_mismatch")
    if fk_rule:
        fk_metrics = fk_rule.get("metrics") or {}
        if "position_gap" in fk_metrics:
            fh.write(
                "Warning: End-effector pose position is not consistent "
                "with joint state, "
                f"position difference: {fk_metrics['position_gap']} meters.\n"
            )
        if "orientation_gap" in fk_metrics:
            fh.write(
                "Warning: End-effector pose orientation is not "
                "consistent with joint state, "
                f"angle difference: {fk_metrics['orientation_gap']} radians.\n"
            )
    else:
        fh.write("All end-effector pose positions are consistent.\n")
        fh.write("All end-effector pose orientations are consistent.\n")
    if "master_slave_joint_gap" in rule_map:
        max_gap = (
            rule_map["master_slave_joint_gap"].get("metrics") or {}
        ).get("max_gap")
        fh.write(
            "Warning: master and follower joints diverge, "
            f"max_gap: {max_gap}\n"
        )
    if item.get("runtime_error"):
        runtime_error = item["runtime_error"]
        fh.write(
            f"When processing {mcap_path}, error occurred: {runtime_error}\n"
        )
    else:
        fh.write(f"{mcap_path} checks completed successfully.\n")
    fh.write(f"{separator} END MCAP: {mcap_path} {separator}\n")


def write_error_logs(
    output_path: str,
    results,
    inspect_config,
    runtime_config: dict[str, Any] | None = None,
) -> None:
    """Write the full log and derive the compact signal log from it.

    Args:
        output_path: Output directory for the log files.
        results: Episode report payloads for the batch.
        inspect_config: Effective inspection thresholds.
        runtime_config: Optional runtime configuration payload for log headers.
    """

    full_log_path = os.path.join(output_path, "inspect_full_log.log")
    with open(full_log_path, "w", encoding="utf-8") as fh:
        if runtime_config is not None:
            fh.write("Running with the following configurations:\n")
            fh.write(json.dumps(runtime_config, indent=2) + "\n")
        if inspect_config is not None:
            fh.write("Inspecting with the following configurations:\n")
            fh.write(json.dumps(asdict(inspect_config), indent=2) + "\n")
        for item in results:
            write_full_log_episode(fh, item, inspect_config)
            fh.write("\n")

    with open(full_log_path, "r", encoding="utf-8") as full_fh:
        full_lines = full_fh.readlines()

    block_separator = "=" * 24

    def _extract_signal_lines(lines: list[str]) -> list[str]:
        selected_indices: list[int] = []
        seen_indices: set[int] = set()

        def _append_index(index: int) -> None:
            if index in seen_indices:
                return
            selected_indices.append(index)
            seen_indices.add(index)

        has_camera_min_fps_error = any(
            line.startswith("Error: Topic '/observation/cameras/")
            and "has a min frame rate below the limit" in line
            for line in lines
        )
        if has_camera_min_fps_error:
            camera_section_start = next(
                (
                    idx
                    for idx, line in enumerate(lines)
                    if line.startswith("Checking camera FPS...")
                ),
                None,
            )
            if camera_section_start is not None:
                camera_section_end = next(
                    (
                        idx
                        for idx in range(camera_section_start + 1, len(lines))
                        if lines[idx].startswith(
                            "Checking data compression..."
                        )
                    ),
                    len(lines),
                )
                for idx in range(camera_section_start, camera_section_end):
                    _append_index(idx)

        for idx, line in enumerate(lines):
            if (
                line.startswith("Warning:")
                or line.startswith("Error:")
                or line.startswith("ERROR ")
            ):
                _append_index(idx)
                detail_idx = idx + 1
                while detail_idx < len(lines):
                    detail_line = lines[detail_idx]
                    if not detail_line.strip():
                        break
                    if (
                        detail_line.startswith("Warning:")
                        or detail_line.startswith("Error:")
                        or detail_line.startswith("ERROR ")
                        or detail_line.startswith("Checking ")
                    ):
                        break
                    _append_index(detail_idx)
                    detail_idx += 1

        return [lines[idx] for idx in selected_indices]

    def _iter_episodes(lines: list[str]):
        begin_prefix = block_separator + " BEGIN MCAP:"
        end_prefix = block_separator + " END MCAP:"
        in_episode = False
        current = []
        for line in lines:
            if line.startswith(begin_prefix):
                in_episode = True
                current = [line]
                continue
            if in_episode:
                current.append(line)
                if line.startswith(end_prefix):
                    yield current
                    in_episode = False
                    current = []

    def _extract_mcap_path(header: str) -> str:
        prefix = block_separator + " BEGIN MCAP: "
        suffix = " " + block_separator
        header = header.strip()
        if header.startswith(prefix) and header.endswith(suffix):
            return header[len(prefix) : -len(suffix)]
        return header

    error_blocks: list[dict[str, str]] = []
    for episode in _iter_episodes(full_lines):
        signal_lines = _extract_signal_lines(episode)
        if not signal_lines:
            continue
        header = next(
            (
                line.strip()
                for line in episode
                if line.startswith(block_separator + " BEGIN MCAP:")
            ),
            None,
        )
        if header is None:
            continue
        mcap_path = _extract_mcap_path(header)
        error_blocks.append(
            {
                "header": header,
                "mcap_path": mcap_path,
                "anchor": build_manual_review_error_log_anchor(mcap_path),
                "body": "".join(signal_lines).rstrip(),
            }
        )

    error_log_path = os.path.join(output_path, "inspect_error_log.log")
    with open(error_log_path, "w", encoding="utf-8") as error_fh:
        for block in error_blocks:
            error_fh.write(f"{block['header']}\n")
            if block["body"]:
                error_fh.write(f"{block['body']}\n")
            error_fh.write("\n")

    write_inspect_error_log_page(output_path, error_blocks)


def sample(tgt_time, src_time, src_data=None, prefix=""):
    """Align a source timeline to the target timeline by nearest timestamp.

    Args:
        tgt_time: Target timeline.
        src_time: Source timeline to be sampled.
        src_data: Optional source payloads aligned with ``src_time``.
        prefix: Prefix used in logging.

    Returns:
        np.ndarray | tuple[np.ndarray, list]:
            Aligned timestamps, and aligned data when ``src_data`` is
            provided.
    """

    insert_pos = np.searchsorted(src_time, tgt_time, side="left")
    left_idx = np.clip(insert_pos - 1, 0, src_time.shape[0] - 1)
    right_idx = np.clip(insert_pos, 0, src_time.shape[0] - 1)
    left_diff = np.abs(tgt_time - src_time[left_idx])
    right_diff = np.abs(src_time[right_idx] - tgt_time)
    index = np.where(left_diff <= right_diff, left_idx, right_idx)
    output_time = src_time[index]
    aligned_diff = np.abs(tgt_time - output_time)
    logger.info(
        f"{prefix:<50} - "
        + f"max time diff: {aligned_diff.max():.4f}, "
        + f"mean time diff: {aligned_diff.mean():.4f}"
    )
    if src_data is not None:
        output = []
        for src in src_data:
            _output = []
            for i in index:
                _output.append(src[i])
            output.append(_output)
        return output_time, output
    return output_time


def format_time(timestamp):
    """Convert ROS-style sec/nsec timestamps to floating-point seconds.

    Args:
        timestamp: Timestamp array in ``[sec, nsec]`` format.

    Returns:
        np.ndarray: Timestamps expressed in seconds.
    """

    timestamp = np.asarray(timestamp, dtype="float64")
    if timestamp.size == 0:
        return np.array([], dtype=np.float64)
    if timestamp.ndim == 1:
        timestamp = timestamp.reshape(-1, 2)
    timestamp = timestamp[:, 0] + timestamp[:, 1] / 1e9
    return timestamp


def pose_to_mat(pose):
    """Convert a pose-like object or dict into a 4x4 transform matrix.

    Args:
        pose: Pose payload expressed as a dict or ROS-like object.

    Returns:
        np.ndarray: Homogeneous 4x4 transform matrix.
    """

    if isinstance(pose, dict):
        x, y, z = pose["position"]
        qx, qy, qz, w = pose["orientation"]
    elif hasattr(pose, "position"):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, w = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    else:
        x, y, z = pose.translation.x, pose.translation.y, pose.translation.z
        qx, qy, qz, w = (
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
        )

    trans = np.array([x, y, z])
    rot = Rotation.from_quat([qx, qy, qz, w], scalar_first=False).as_matrix()
    ret = np.eye(4)
    ret[:3, 3] = trans
    ret[:3, :3] = rot
    return ret


def get_frequency(timestamp, prefix="", window_size=3):
    """Estimate smoothed per-step frequency values from a timestamp series.

    Args:
        timestamp: Timestamp series in seconds.
        prefix: Prefix used in logging.
        window_size: Moving-average window used before inverting intervals.

    Returns:
        np.ndarray: Per-step frequency estimates.
    """

    if not isinstance(timestamp, np.ndarray):
        timestamp = np.array(timestamp)
    time_diff = np.diff(timestamp)
    if time_diff.size == 0:
        return np.array([], dtype=np.float64)
    if time_diff.size < window_size:
        freq = 1 / time_diff
        logger.info(
            f"{prefix:<50} - "
            f"duration: {timestamp[-1] - timestamp[0]:.2f}s, "
            + f"min frequency: {freq.min():.1f}Hz, "
            + f"mean frequency: {freq.mean():.1f}Hz"
        )
        return freq
    time_diff = torch.from_numpy(time_diff)[None, None]
    time_diff = torch.nn.functional.avg_pool1d(
        time_diff, window_size, 1
    ).numpy()[0, 0]
    freq = 1 / time_diff
    logger.info(
        f"{prefix:<50} - "
        f"duration: {timestamp[-1] - timestamp[0]:.2f}s, "
        + f"min frequency: {freq.min():.1f}Hz, "
        + f"mean frequency: {freq.mean():.1f}Hz"
    )
    return freq
