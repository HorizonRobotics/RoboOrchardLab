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

import gc
import io
import shutil
import subprocess

import numpy as np
import pytest

from robo_orchard_lab.utils.video import (
    VideoBackendUnavailableError,
    VideoEncodeError,
    VideoFrameError,
    VideoPixelFormat,
    VideoWriter,
    VideoWriterError,
)


def _get_ffmpeg_binary(*, require_libx264: bool = False) -> str:
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        pytest.skip("ffmpeg is required for real video encode/decode tests.")

    if not require_libx264:
        return ffmpeg_binary

    encoders = subprocess.run(
        [ffmpeg_binary, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    encoder_listing = f"{encoders.stdout}\n{encoders.stderr}"
    if encoders.returncode != 0 or "libx264" not in encoder_listing:
        pytest.skip(
            "ffmpeg with libx264 support is required for real VideoWriter "
            "encode tests."
        )

    return ffmpeg_binary


def _decode_first_frame_rgb(
    video_path,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    ffmpeg_binary = _get_ffmpeg_binary()
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        height, width, 3
    )


class _FakeStdin:
    def __init__(
        self,
        *,
        write_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._write_error = write_error
        self._close_error = close_error
        self.write_calls = 0
        self.close_calls = 0

    def write(self, data: bytes) -> int:
        self.write_calls += 1
        if self._write_error is not None:
            raise self._write_error
        return len(data)

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class _FakeProc:
    def __init__(
        self,
        *,
        stdin: _FakeStdin,
        stderr_data: bytes = b"",
        return_code: int = 0,
    ) -> None:
        self.stdin = stdin
        self.stderr = io.BytesIO(stderr_data)
        self._return_code = return_code
        self._killed = False
        self.wait_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        if self._killed:
            return -9
        return None

    def wait(self) -> int:
        self.wait_calls += 1
        if self._killed:
            return -9
        return self._return_code

    def kill(self) -> None:
        self.kill_calls += 1
        self._killed = True


def _write_staging_output(
    tmp_path,
    name: str,
    *,
    data: bytes = b"partial",
):
    staging_output_path = tmp_path / name
    staging_output_path.write_bytes(data)
    return staging_output_path


def test_video_writer_writes_rgb24_video_with_real_ffmpeg(tmp_path):
    _get_ffmpeg_binary(require_libx264=True)
    frame = np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8)
    output_path = tmp_path / "episode_rgb.mp4"

    writer = VideoWriter(output_path, pixel_format=VideoPixelFormat.RGB24)
    assert writer.output_path == output_path
    assert writer.is_open
    assert not writer.is_closed
    writer.write_frame(frame)
    assert not output_path.exists()
    writer.write_frame(frame)
    writer.close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert list(tmp_path.glob(".episode_rgb.*.mp4")) == []
    assert writer.frame_count == 2
    assert writer.is_closed
    assert not writer.is_open

    decoded = _decode_first_frame_rgb(output_path, width=16, height=16)

    assert decoded[..., 0].mean() > 200
    assert decoded[..., 1].mean() < 40
    assert decoded[..., 2].mean() < 40


def test_video_writer_writes_bgr24_video_with_real_ffmpeg(tmp_path):
    _get_ffmpeg_binary(require_libx264=True)
    frame = np.full((16, 16, 3), [0, 0, 255], dtype=np.uint8)
    output_path = tmp_path / "episode_bgr.mp4"

    writer = VideoWriter(output_path, pixel_format=VideoPixelFormat.BGR24)
    writer.write_frame(frame)
    writer.close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    decoded = _decode_first_frame_rgb(output_path, width=16, height=16)

    assert decoded[..., 0].mean() > 200
    assert decoded[..., 1].mean() < 40
    assert decoded[..., 2].mean() < 40


def test_video_writer_can_open_and_reuse_new_output_path(tmp_path):
    _get_ffmpeg_binary(require_libx264=True)
    first_path = tmp_path / "episode_first.mp4"
    second_path = tmp_path / "episode_second.mp4"
    frame = np.full((16, 16, 3), [255, 255, 0], dtype=np.uint8)

    writer = VideoWriter(pixel_format=VideoPixelFormat.RGB24)

    assert writer.output_path is None
    assert writer.is_closed
    assert not writer.is_open

    writer.open(first_path)
    assert writer.output_path == first_path
    assert writer.is_open
    writer.write_frame(frame)
    writer.close()

    assert first_path.exists()
    assert writer.is_closed

    writer.open(second_path)
    assert writer.output_path == second_path
    assert writer.frame_count == 0
    assert writer.is_open
    writer.write_frame(frame)
    writer.close()

    assert second_path.exists()
    decoded = _decode_first_frame_rgb(second_path, width=16, height=16)
    assert decoded[..., 0].mean() > 200
    assert decoded[..., 1].mean() > 200
    assert decoded[..., 2].mean() < 40


def test_video_writer_open_supports_context_manager(tmp_path):
    _get_ffmpeg_binary(require_libx264=True)
    output_path = tmp_path / "episode_context.mp4"
    frame = np.full((16, 16, 3), [0, 255, 255], dtype=np.uint8)
    writer = VideoWriter(pixel_format=VideoPixelFormat.RGB24)

    with writer.open(output_path) as opened_writer:
        assert opened_writer is writer
        assert writer.is_open
        writer.write_frame(frame)

    assert writer.is_closed
    decoded = _decode_first_frame_rgb(output_path, width=16, height=16)
    assert decoded[..., 0].mean() < 40
    assert decoded[..., 1].mean() > 200
    assert decoded[..., 2].mean() > 200


def test_video_writer_first_failed_write_does_not_poison_open_session(
    tmp_path,
):
    _get_ffmpeg_binary(require_libx264=True)
    output_path = tmp_path / "episode_recover.mp4"
    writer = VideoWriter(output_path, pixel_format=VideoPixelFormat.RGB24)

    with pytest.raises(
        VideoFrameError,
        match=(
            r"output_pixel_format='yuv420p' requires even frame width "
            r"and height"
        ),
    ):
        writer.write_frame(np.zeros((5, 5, 3), dtype=np.uint8))

    assert writer.is_open
    assert writer.frame_count == 0

    writer.write_frame(np.full((6, 6, 3), [255, 0, 255], dtype=np.uint8))
    writer.close()

    assert writer.frame_count == 1
    decoded = _decode_first_frame_rgb(output_path, width=6, height=6)
    assert decoded[..., 0].mean() > 200
    assert decoded[..., 1].mean() < 40
    assert decoded[..., 2].mean() > 200


def test_video_writer_context_manager_preserves_body_exception(
    tmp_path,
    monkeypatch,
):
    writer = VideoWriter(tmp_path / "episode_error.mp4")
    close_calls = {"count": 0}

    def _raise_close_error() -> None:
        close_calls["count"] += 1
        raise VideoEncodeError("close failed")

    monkeypatch.setattr(writer, "close", _raise_close_error)

    with pytest.raises(ValueError, match="body failed") as exc_info:
        with writer:
            raise ValueError("body failed")

    assert close_calls["count"] == 1
    assert str(exc_info.value) == "body failed"
    assert isinstance(exc_info.value.__cause__, VideoEncodeError)


def test_video_writer_context_manager_rejects_closed_writer():
    writer = VideoWriter(pixel_format=VideoPixelFormat.RGB24)

    with pytest.raises(VideoWriterError, match="because it is not open"):
        with writer:
            pass


def test_video_writer_reuses_after_post_start_write_failure(tmp_path):
    output_path = tmp_path / "post_start_failure.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "post_start_failure.staging.mp4",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(
        stdin=_FakeStdin(write_error=BrokenPipeError("broken pipe"))
    )
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )
    writer._frame_size = (6, 4)

    with pytest.raises(
        VideoEncodeError,
        match="Failed to write a video frame",
    ):
        writer.write_frame(np.zeros((4, 6, 3), dtype=np.uint8))

    assert writer.is_closed
    assert writer._proc is None
    assert writer._staging_output_path is None
    assert not output_path.exists()
    assert not staging_output_path.exists()
    assert proc.kill_calls == 1
    assert proc.wait_calls == 1

    reopened_path = tmp_path / "post_start_reopened.mp4"
    writer.open(reopened_path)
    assert writer.is_open
    assert writer.output_path == reopened_path


def test_video_writer_reaps_process_when_finalize_fails(tmp_path):
    output_path = tmp_path / "finalize_failure.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "finalize_failure.staging.mp4",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(
        stdin=_FakeStdin(close_error=BrokenPipeError("close failed"))
    )
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    with pytest.raises(
        VideoEncodeError,
        match="Failed to finalize video output",
    ):
        writer.close()

    assert writer.is_closed
    assert writer._proc is None
    assert writer._staging_output_path is None
    assert not output_path.exists()
    assert not staging_output_path.exists()
    assert proc.kill_calls == 1
    assert proc.wait_calls == 1


def test_video_writer_commits_staged_output_on_close(tmp_path):
    output_path = tmp_path / "committed.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "committed.staging.mp4",
        data=b"encoded",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(stdin=_FakeStdin())
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    writer.close()

    assert writer.is_closed
    assert writer._proc is None
    assert writer._staging_output_path is None
    assert output_path.read_bytes() == b"encoded"
    assert not staging_output_path.exists()
    assert proc.kill_calls == 0
    assert proc.wait_calls == 1


def test_video_writer_replaces_existing_output_on_success_when_overwrite_true(
    tmp_path,
):
    output_path = tmp_path / "finalize_nonzero.mp4"
    output_path.write_bytes(b"original")
    staging_output_path = _write_staging_output(
        tmp_path,
        "finalize_nonzero.staging.mp4",
        data=b"encoded",
    )
    writer = VideoWriter(output_path, overwrite=True)
    proc = _FakeProc(stdin=_FakeStdin())
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    writer.close()

    assert output_path.read_bytes() == b"encoded"
    assert not staging_output_path.exists()


def test_video_writer_deletes_staged_output_on_nonzero_finalize(tmp_path):
    output_path = tmp_path / "finalize_nonzero.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "finalize_nonzero.staging.mp4",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(
        stdin=_FakeStdin(),
        stderr_data=b"encode failed",
        return_code=1,
    )
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    with pytest.raises(
        VideoEncodeError,
        match="ffmpeg exited with non-zero status 1",
    ):
        writer.close()

    assert writer.is_closed
    assert writer._proc is None
    assert writer._staging_output_path is None
    assert not output_path.exists()
    assert not staging_output_path.exists()
    assert proc.kill_calls == 0
    assert proc.wait_calls == 1


def test_video_writer_gc_cleanup_reaps_unclosed_process(tmp_path, recwarn):
    output_path = tmp_path / "gc_cleanup.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "gc_cleanup.staging.mp4",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(stdin=_FakeStdin())
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    del writer
    gc.collect()

    resource_warnings = [
        warning
        for warning in recwarn
        if issubclass(warning.category, ResourceWarning)
    ]
    assert any(
        "garbage-collected without close" in str(warning.message)
        for warning in resource_warnings
    )
    assert not output_path.exists()
    assert not staging_output_path.exists()
    assert proc.kill_calls == 1
    assert proc.wait_calls == 1


def test_video_writer_close_clears_gc_cleanup_finalizer(tmp_path, recwarn):
    output_path = tmp_path / "gc_cleanup_cleared.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "gc_cleanup_cleared.staging.mp4",
        data=b"encoded",
    )
    writer = VideoWriter(output_path)
    proc = _FakeProc(stdin=_FakeStdin())
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )

    writer.close()
    del writer
    gc.collect()

    resource_warnings = [
        warning
        for warning in recwarn
        if issubclass(warning.category, ResourceWarning)
    ]
    assert resource_warnings == []
    assert output_path.read_bytes() == b"encoded"
    assert not staging_output_path.exists()


def test_video_writer_rejects_existing_output_on_start_when_overwrite_false(
    tmp_path,
):
    output_path = tmp_path / "existing_output.mp4"
    output_path.write_bytes(b"original")
    writer = VideoWriter(output_path, overwrite=False)

    with pytest.raises(
        VideoEncodeError,
        match="already exists and overwrite=False",
    ):
        writer.write_frame(np.zeros((4, 6, 3), dtype=np.uint8))

    assert writer.is_open
    assert writer.frame_count == 0
    assert output_path.read_bytes() == b"original"


def test_video_writer_preserves_output_file_created_after_start_when_overwrite_false(  # noqa: E501
    tmp_path,
):
    output_path = tmp_path / "created_during_session.mp4"
    staging_output_path = _write_staging_output(
        tmp_path,
        "created_during_session.staging.mp4",
    )
    writer = VideoWriter(output_path, overwrite=False)
    proc = _FakeProc(stdin=_FakeStdin())
    writer._set_proc(
        proc,
        path=writer.output_path,
        staging_output_path=staging_output_path,
    )
    output_path.write_bytes(b"original")

    with pytest.raises(
        VideoEncodeError,
        match="Failed to publish finalized video output",
    ):
        writer.close()

    assert output_path.read_bytes() == b"original"
    assert not staging_output_path.exists()


def test_video_writer_rejects_non_color_frame(tmp_path):
    writer = VideoWriter(tmp_path / "invalid.mp4")

    with pytest.raises(VideoFrameError, match=r"Expected frame shape"):
        writer.write_frame(np.zeros((4, 5), dtype=np.uint8))


def test_video_writer_rejects_frame_size_change(tmp_path):
    _get_ffmpeg_binary(require_libx264=True)
    writer = VideoWriter(tmp_path / "size_change.mp4")
    writer.write_frame(np.zeros((4, 6, 3), dtype=np.uint8))

    with pytest.raises(VideoFrameError, match=r"Expected frame size"):
        writer.write_frame(np.zeros((6, 6, 3), dtype=np.uint8))

    writer.close()


def test_video_writer_rejects_odd_dimensions_for_default_yuv420p(tmp_path):
    writer = VideoWriter(tmp_path / "odd_size.mp4")

    with pytest.raises(
        VideoFrameError,
        match=(
            r"output_pixel_format='yuv420p' requires even frame width "
            r"and height"
        ),
    ):
        writer.write_frame(np.zeros((4, 5, 3), dtype=np.uint8))


def test_video_writer_raises_when_ffmpeg_missing(tmp_path, monkeypatch):
    writer = VideoWriter(tmp_path / "missing_ffmpeg.mp4")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(
        VideoBackendUnavailableError,
        match="ffmpeg is not available in PATH",
    ):
        writer.write_frame(np.zeros((4, 6, 3), dtype=np.uint8))
