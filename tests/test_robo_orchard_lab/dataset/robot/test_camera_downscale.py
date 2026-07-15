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

from typing import Any, Mapping

import cv2
import datasets as hg_datasets
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from robo_orchard_core.datatypes.camera_data import BatchCameraDataEncoded

from robo_orchard_lab.dataset.datatypes import BatchCameraDataEncodedFeature
from robo_orchard_lab.dataset.robot import (
    camera_downscale as camera_downscale_module,
)
from robo_orchard_lab.dataset.robot.camera_downscale import (
    CameraDownscaleConfig,
    downscale_camera_data,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.metadata_schema import (
    EpisodeTimingInfo,
    ROEpisodeInfo,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
)
from robo_orchard_lab.dataset.robot.re_packing._errors import (
    RepackFrameTransformError,
)
from robo_orchard_lab.dataset.robot.re_packing.camera_downscale import (
    downscale_ro_dataset,
)


def _encoded_rgb_camera(format: str = "jpeg") -> BatchCameraDataEncoded:
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return BatchCameraDataEncoded(
        sensor_data=[encoded.tobytes()],
        format=format,  # type: ignore[arg-type]
        image_shape=(5, 6),
        intrinsic_matrices=torch.tensor(
            [[[100.0, 0.0, 3.0], [0.0, 80.0, 2.5], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        timestamps=[123],
        frame_id="camera",
    )


def _encoded_depth_camera() -> BatchCameraDataEncoded:
    depth = np.array(
        [
            [0, 100, 200, 300],
            [400, 500, 600, 700],
            [800, 900, 1000, 1100],
            [1200, 1300, 1400, 1500],
            [1600, 1700, 1800, 1900],
        ],
        dtype=np.uint16,
    )
    ok, encoded = cv2.imencode(".png", depth)
    assert ok
    return BatchCameraDataEncoded(
        sensor_data=[encoded.tobytes()],
        format="png",
        image_shape=(5, 4),
        intrinsic_matrices=torch.tensor(
            [[[80.0, 0.0, 2.0], [0.0, 90.0, 2.5], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        timestamps=[456],
        frame_id="depth_camera",
    )


class _CameraDownscaleEpisode(EpisodePackaging):
    features = hg_datasets.Features(
        {
            "front_camera_image": BatchCameraDataEncodedFeature(
                dtype="float32"
            ),
            "front_camera_depth": BatchCameraDataEncodedFeature(
                dtype="float32"
            ),
            "label": hg_datasets.Value("string"),
        }
    )

    def __init__(
        self,
        *,
        info: Mapping[str, object] | None,
        corrupt_frame_index: int | None = None,
    ):
        self._info = dict(info) if info is not None else None
        self._corrupt_frame_index = corrupt_frame_index

    def generate_episode_meta(self) -> EpisodeMeta:
        return EpisodeMeta(
            episode=EpisodeData(
                truncated=True,
                success=False,
                info=self._info,
            )
        )

    def generate_frames(self):
        for frame_index in range(2):
            timestamp_ns = frame_index * 100
            image = _encoded_rgb_camera()
            depth = _encoded_depth_camera()
            if frame_index == self._corrupt_frame_index:
                image = image.model_copy(
                    update={"sensor_data": [b"bad image"]}
                )
            image = image.model_copy(update={"timestamps": [timestamp_ns]})
            depth = depth.model_copy(update={"timestamps": [timestamp_ns]})
            yield DataFrame(
                features={
                    "front_camera_image": image,
                    "front_camera_depth": depth,
                    "label": f"frame-{frame_index}",
                },
                timestamp_ns_min=timestamp_ns,
                timestamp_ns_max=timestamp_ns,
            )


def _make_camera_downscale_source_dataset(
    tmp_path,
    *,
    info: Mapping[str, object] | None,
    corrupt_frame_index: int | None = None,
) -> str:
    dataset_path = tmp_path / "source_ro_dataset"
    episode = _CameraDownscaleEpisode(
        info=info,
        corrupt_frame_index=corrupt_frame_index,
    )
    DatasetPackaging(
        features=episode.features,
        database_driver="sqlite",
    ).packaging(
        episodes=[episode],
        dataset_path=str(dataset_path),
        writer_batch_size=1,
        force_overwrite=True,
    )
    return str(dataset_path)


def _assert_repack_frame_transform_error(
    exc: BaseException,
) -> None:
    assert isinstance(exc, RepackFrameTransformError)


class TestCameraDownscaleConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"downscale": 0.0},
            {"downscale": -0.5},
            {"downscale": 1.1},
            {"downscale": 0.5, "jpeg_quality": 0},
            {"downscale": 0.5, "jpeg_quality": 101},
            {"downscale": 0.5, "png_compression": -1},
            {"downscale": 0.5, "png_compression": 10},
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict[str, Any]):
        with pytest.raises(ValidationError):
            CameraDownscaleConfig(**kwargs)

    def test_accepts_none_png_compression(self):
        config = CameraDownscaleConfig(
            downscale=0.5,
            png_compression=None,
        )

        assert config.png_compression is None


class TestDownscaleCameraData:
    @pytest.mark.parametrize("format", ["jpeg", "jpg"])
    def test_rgb_downscale_preserves_format_and_updates_effective_intrinsic(
        self, format: str
    ):
        camera = _encoded_rgb_camera(format=format)
        config = CameraDownscaleConfig(downscale=0.5)

        downscaled = downscale_camera_data(
            camera,
            config=config,
            is_depth=False,
            context="episode=7 column=front_camera_image",
        )

        assert downscaled.format == format
        assert downscaled.image_shape == (2, 3)
        assert downscaled.timestamps == camera.timestamps
        decoded = downscaled.decode()
        assert decoded.sensor_data.shape == (1, 2, 3, 3)
        effective_intrinsic = downscaled.get_intrinsic_with_transform()
        assert effective_intrinsic is not None
        assert camera.intrinsic_matrices is not None
        expected_scale = torch.tensor(
            [[[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        assert torch.allclose(
            effective_intrinsic,
            expected_scale @ camera.intrinsic_matrices,
            atol=1e-6,
        )

    def test_png_depth_downscale_uses_nearest_and_preserves_uint16(self):
        camera = _encoded_depth_camera()
        config = CameraDownscaleConfig(downscale=0.5)

        downscaled = downscale_camera_data(
            camera,
            config=config,
            is_depth=True,
            context="episode=7 column=front_camera_depth",
        )

        assert downscaled.format == "png"
        assert downscaled.image_shape == (2, 2)
        decoded = downscaled.decode()
        assert decoded.sensor_data.dtype == torch.uint16
        expected = torch.tensor(
            [[0, 200], [800, 1000]],
            dtype=torch.uint16,
        ).view(1, 2, 2, 1)
        assert torch.equal(decoded.sensor_data, expected)

    def test_missing_image_shape_decodes_once(self, monkeypatch):
        camera = _encoded_rgb_camera(format="jpeg").model_copy(
            update={"image_shape": None}
        )
        config = CameraDownscaleConfig(downscale=0.5)
        decode_calls = 0
        original_decode = BatchCameraDataEncoded.decode

        def counting_decode(self, *args, **kwargs):
            nonlocal decode_calls
            decode_calls += 1
            return original_decode(self, *args, **kwargs)

        monkeypatch.setattr(
            BatchCameraDataEncoded,
            "decode",
            counting_decode,
        )

        downscaled = downscale_camera_data(
            camera,
            config=config,
            is_depth=False,
            context="episode=7 column=front_camera_image",
        )

        assert decode_calls == 1
        assert downscaled.image_shape == (2, 3)

    def test_rejects_non_png_depth_with_context(self):
        camera = _encoded_rgb_camera(format="jpeg")
        config = CameraDownscaleConfig(downscale=0.5)

        with pytest.raises(ValueError) as exc_info:
            downscale_camera_data(
                camera,
                config=config,
                is_depth=True,
                context="episode=3 column=front_camera_depth",
            )

        message = str(exc_info.value)
        assert "episode=3 column=front_camera_depth" in message
        assert "format=jpeg" in message

    def test_reports_decode_errors_with_context_and_frame_index(self):
        camera = BatchCameraDataEncoded(
            sensor_data=[b"not an image"],
            format="jpeg",
            image_shape=(5, 6),
        )
        config = CameraDownscaleConfig(downscale=0.5)

        with pytest.raises(ValueError) as exc_info:
            downscale_camera_data(
                camera,
                config=config,
                is_depth=False,
                context="episode=9 column=front_camera_image",
            )

        message = str(exc_info.value)
        assert "episode=9 column=front_camera_image" in message
        assert "frame=0" in message
        assert "format=jpeg" in message


class TestCameraDownscaleMetadata:
    def test_camera_downscale_module_does_not_expose_record_builder(self):
        assert "build_camera_downscale_processing_record" not in (
            camera_downscale_module.__all__
        )
        assert not hasattr(
            camera_downscale_module,
            "build_camera_downscale_processing_record",
        )


class TestDownscaleRODataset:
    def test_downscale_ro_dataset_success_preserves_metadata(
        self,
        tmp_path,
    ):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=ROEpisodeInfo(
                episode_id="episode-1",
                timing=EpisodeTimingInfo(duration_s=0.1, average_fps=10.0),
            ).to_json_dict(),
        )
        target_path = tmp_path / "target_ro_dataset"

        downscale_ro_dataset(
            source_path,
            str(target_path),
            downscale=0.5,
            writer_batch_size=1,
            force_overwrite=True,
        )

        dataset = RODataset(str(target_path))
        assert len(dataset) == 2
        frame0 = dataset[0]
        assert frame0["label"] == "frame-0"
        assert frame0["timestamp_min"] == 0
        image = frame0["front_camera_image"]
        depth = frame0["front_camera_depth"]
        assert image.image_shape == (2, 3)
        assert depth.image_shape == (2, 2)
        assert image.timestamps == [0]
        assert depth.timestamps == [0]
        assert image.get_intrinsic_with_transform() is not None
        assert depth.get_intrinsic_with_transform() is not None

        episode = dataset.get_meta(Episode, int(frame0["episode_index"]))
        assert episode is not None
        assert episode.truncated is True
        assert episode.success is False
        parsed = ROEpisodeInfo.model_validate(episode.info)
        assert parsed.episode_id == "episode-1"
        assert parsed.timing is not None
        assert parsed.timing.duration_s == 0.1
        assert parsed.extras is None

    def test_downscale_ro_dataset_noop_does_not_write_provenance(
        self,
        tmp_path,
    ):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=ROEpisodeInfo(episode_id="episode-1").to_json_dict(),
        )
        target_path = tmp_path / "target_ro_dataset"

        downscale_ro_dataset(
            source_path,
            str(target_path),
            downscale=1.0,
            writer_batch_size=1,
            force_overwrite=True,
        )

        dataset = RODataset(str(target_path))
        frame0 = dataset[0]
        image = frame0["front_camera_image"]
        depth = frame0["front_camera_depth"]
        assert image.image_shape == (5, 6)
        assert depth.image_shape == (5, 4)
        episode = dataset.get_meta(Episode, int(frame0["episode_index"]))
        assert episode is not None
        parsed = ROEpisodeInfo.model_validate(episode.info)
        assert parsed.extras is None

    @pytest.mark.parametrize("info", [None, {"legacy": True}])
    def test_downscale_ro_dataset_noop_preserves_non_canonical_info(
        self,
        tmp_path,
        info: dict[str, object] | None,
    ):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=info,
        )
        target_path = tmp_path / "target_ro_dataset"

        downscale_ro_dataset(
            source_path,
            str(target_path),
            downscale=1.0,
            writer_batch_size=1,
            force_overwrite=True,
        )

        dataset = RODataset(str(target_path))
        frame0 = dataset[0]
        episode = dataset.get_meta(Episode, int(frame0["episode_index"]))
        assert episode is not None
        assert episode.info == info

    @pytest.mark.parametrize("info", [None, {"legacy": True}])
    def test_downscale_ro_dataset_preserves_non_canonical_info(
        self,
        tmp_path,
        info: dict[str, object] | None,
    ):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=info,
        )
        target_path = tmp_path / "target_ro_dataset"

        downscale_ro_dataset(
            source_path,
            str(target_path),
            downscale=0.5,
            writer_batch_size=1,
            force_overwrite=True,
        )

        dataset = RODataset(str(target_path))
        frame0 = dataset[0]
        assert frame0["front_camera_image"].image_shape == (2, 3)
        episode = dataset.get_meta(Episode, int(frame0["episode_index"]))
        assert episode is not None
        assert episode.info == info

    def test_downscale_ro_dataset_fail_fast_on_bad_camera_frame(
        self,
        tmp_path,
    ):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=ROEpisodeInfo(episode_id="episode-1").to_json_dict(),
            corrupt_frame_index=1,
        )

        with pytest.raises(Exception) as exc_info:
            downscale_ro_dataset(
                source_path,
                str(tmp_path / "target_ro_dataset"),
                downscale=0.5,
                writer_batch_size=1,
                force_overwrite=True,
            )

        exc = exc_info.value
        _assert_repack_frame_transform_error(exc)
        assert isinstance(exc.__cause__, ValueError)
        assert exc.original_error is exc.__cause__
        assert exc.source_episode_index == 0
        assert exc.frame_offset == 1
        assert exc.source_frame_index == 1
        message = str(exc)
        assert "column=front_camera_image" in message
        assert "source_episode_index=0" in message
        assert "frame_offset=1" in message
        assert "source_frame_index=1" in message

    def test_downscale_ro_dataset_rejects_bad_explicit_columns(self, tmp_path):
        source_path = _make_camera_downscale_source_dataset(
            tmp_path,
            info=ROEpisodeInfo(episode_id="episode-1").to_json_dict(),
        )

        with pytest.raises(ValueError, match="encoded camera"):
            downscale_ro_dataset(
                source_path,
                str(tmp_path / "target_bad_column"),
                downscale=0.5,
                columns=["label"],
                force_overwrite=True,
            )
        with pytest.raises(ValueError, match="depth_columns"):
            downscale_ro_dataset(
                source_path,
                str(tmp_path / "target_bad_depth"),
                downscale=0.5,
                columns=["front_camera_image"],
                depth_columns=["front_camera_depth"],
                force_overwrite=True,
            )
