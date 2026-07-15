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

"""Internal exceptions for RODataset repacking."""

from __future__ import annotations

__all__ = ["RepackFrameTransformError"]


class RepackFrameTransformError(RuntimeError):
    """Frame-stream failure with repack source-frame context.

    ``repack_dataset`` raises this exception when draining a transformed frame
    stream fails after a source episode and selected frame offset are known.
    The original exception is preserved as both ``__cause__`` and
    ``original_error`` so callers can inspect the low-level failure without
    parsing the diagnostic message.

    This type is intentionally defined in an internal module and is not
    exported from ``robo_orchard_lab.dataset.robot.re_packing`` in the first
    version.

    Args:
        source_episode_index (int): Source dataset episode index being
            repacked.
        source_frame_index (int | None): Source dataset frame index selected
            at ``frame_offset``. ``None`` means the failing offset could not
            be mapped to a selected source frame.
        frame_offset (int): Zero-based offset within the selected source
            episode frame stream.
        original_error (BaseException): Original frame-stream exception.
    """

    def __init__(
        self,
        *,
        source_episode_index: int,
        source_frame_index: int | None,
        frame_offset: int,
        original_error: BaseException,
    ) -> None:
        self.source_episode_index = source_episode_index
        self.source_frame_index = source_frame_index
        self.frame_offset = frame_offset
        self.original_error = original_error
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        context = (
            f"source_episode_index={self.source_episode_index} "
            f"frame_offset={self.frame_offset}"
        )
        if self.source_frame_index is not None:
            context += f" source_frame_index={self.source_frame_index}"
        return (
            f"{type(self.original_error).__name__}: "
            f"{self.original_error} ({context})"
        )
