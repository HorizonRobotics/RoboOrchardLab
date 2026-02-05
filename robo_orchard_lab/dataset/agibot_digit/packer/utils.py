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

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import ffmpeg
import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path: str, data: dict | list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json_threaded(
    json_paths: list[str],
    desc: str = "Loading json files",
    max_workers: int = 16,
) -> list[dict]:

    results = [None] * len(json_paths)

    def load_one_json(i, path):
        try:
            with open(path, "r") as f:
                return i, json.load(f)
        except Exception as e:
            return i, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(load_one_json, i, p) for i, p in enumerate(json_paths)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i, data = f.result()
            results[i] = data  # keep original order

    return results  # type: ignore


def load_txt(file_path: str):
    with open(file_path, "r") as f:
        data = f.read()
    return data


def load_h5_to_dict(file_path: str) -> dict:
    data_dict = {}
    with h5py.File(file_path, "r") as f:

        def recursive_load(h5obj, current_dict):
            for key, item in h5obj.items():
                if isinstance(item, h5py.Dataset):
                    current_dict[key] = item[:]
                elif isinstance(item, h5py.Group):
                    current_dict[key] = {}
                    recursive_load(item, current_dict[key])

        recursive_load(f, data_dict)
    return data_dict


def load_one_image(img_path):
    try:
        if img_path.endswith(".png"):
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image = cv2.imencode(".png", image)[1].tobytes()  # type: ignore
        elif img_path.endswith(".jpg") or img_path.endswith(".jpeg"):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.imencode(".jpg", image)[1].tobytes()  # type: ignore
        else:
            raise ValueError(f"Unsupported image format: {img_path}")
    except Exception as e:
        logger.error(f"Failed to load image {img_path}: {e}")
        raise e
    return image


def load_images_threaded(
    img_paths, max_workers=32, desc="Loading images"
) -> list[bytes]:
    """Load multiple images in parallel using threads."""
    images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img in tqdm(
            executor.map(load_one_image, img_paths),
            total=len(img_paths),
            desc=desc,
            ncols=100,
        ):
            images.append(img)
    return images


def decode_video_to_frames_ffmpeg(
    video_path: str,
) -> tuple[list[bytes], int, int]:
    """Robust decoder that handles AV1 / H.264 / HEVC."""
    probe = ffmpeg.probe(video_path)
    video_info = next(
        s for s in probe['streams'] if s['codec_type'] == 'video'
    )
    width = int(video_info['width'])
    height = int(video_info['height'])

    process = (
        ffmpeg.input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
        .run_async(pipe_stdout=True)
    )

    frames = []
    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            frames.append(buf.tobytes())
    process.wait()
    return frames, width, height


def generate_static_mask(positions, threshold=5e-4):
    """Generate a binary mask indicating static positions."""
    positions = np.array(positions)  # (N, D)
    diffs = np.abs(np.diff(positions, axis=0))
    static_mask = np.all(diffs < threshold, axis=1)
    # First frame is static
    static_mask = np.concatenate(([True], static_mask))
    return static_mask


def resize_image(
    image: np.ndarray,
    intrinsic: np.ndarray,
    max_image_hw: tuple,
):
    """Resize image to fit within max_image_hw while adjusting intrinsic matrix."""
    h, w = image.shape[:2]
    target_h, target_w = max_image_hw

    if h <= target_h and w <= target_w:
        return image, intrinsic

    scale_w = target_w / w
    scale_h = target_h / h

    resized_image = cv2.resize(
        image, (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    new_intrinsic = intrinsic.copy()
    new_intrinsic[0, 0] *= scale_w  # fx' = fx * scale_w
    new_intrinsic[0, 2] *= scale_w  # cx' = cx * scale_w
    new_intrinsic[1, 1] *= scale_h  # fy' = fy * scale_h
    new_intrinsic[1, 2] *= scale_h  # cy' = cy * scale_h

    return resized_image, new_intrinsic
