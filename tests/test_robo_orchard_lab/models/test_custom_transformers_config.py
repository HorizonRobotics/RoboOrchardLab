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
import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]

    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_root / relative_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


LlavaConfig = _load_module(
    "robo_orchard_lab.models.monodream.configuration_llava",
    "robo_orchard_lab/models/monodream/configuration_llava.py",
).LlavaConfig
MultimodalProjectorConfig = _load_module(
    "robo_orchard_lab.models.monodream.multimodal_projector.base_projector",
    "robo_orchard_lab/models/monodream/multimodal_projector/base_projector.py",
).MultimodalProjectorConfig
VLMImageProcessorConfig = _load_module(
    "robo_orchard_lab.models.mapdream.janus.inference.models.image_processing_vlm",
    "robo_orchard_lab/models/mapdream/janus/inference/models/"
    "image_processing_vlm.py",
).VLMImageProcessorConfig


def test_llava_config_preserves_unknown_hf_kwargs():
    cfg = LlavaConfig(output_hidden_states=True, custom_runtime_field="x")

    dumped = cfg.to_dict()
    restored = LlavaConfig(**dumped)

    assert restored.output_hidden_states is True
    assert restored.custom_runtime_field == "x"


def test_multimodal_projector_config_preserves_unknown_hf_kwargs():
    cfg = MultimodalProjectorConfig(
        mm_projector_type="mlp2x_gelu",
        output_attentions=True,
        custom_runtime_field="projector",
    )

    dumped = cfg.to_dict()
    restored = MultimodalProjectorConfig(**dumped)

    assert restored.mm_projector_type == "mlp2x_gelu"
    assert restored.output_attentions is True
    assert restored.custom_runtime_field == "projector"


def test_vlm_image_processor_config_round_trips_sequence_defaults():
    cfg = VLMImageProcessorConfig(
        image_size=384,
        image_mean=(0.1, 0.2, 0.3),
        image_std=(0.4, 0.5, 0.6),
        output_hidden_states=True,
    )

    dumped = cfg.to_dict()
    restored = VLMImageProcessorConfig(**dumped)

    assert restored.image_size == 384
    assert list(restored.image_mean) == [0.1, 0.2, 0.3]
    assert list(restored.image_std) == [0.4, 0.5, 0.6]
    assert restored.output_hidden_states is True
