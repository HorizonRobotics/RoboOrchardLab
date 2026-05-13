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
import types
from pathlib import Path

import torch.nn as nn
from easydict import EasyDict
from transformers import LlamaConfig


def _load_modeling_vlm_module():
    module_name = (
        "robo_orchard_lab.models.mapdream.janus.inference.models.modeling_vlm"
    )
    if module_name in sys.modules:
        return sys.modules[module_name]

    repo_root = Path(__file__).resolve().parents[3]
    models_dir = (
        repo_root
        / "robo_orchard_lab"
        / "models"
        / "mapdream"
        / "janus"
        / "inference"
        / "models"
    )
    package_name = "robo_orchard_lab.models.mapdream.janus.inference.models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir)]
    sys.modules[package_name] = package

    clip_encoder = types.ModuleType(f"{package_name}.clip_encoder")
    clip_encoder.CLIPVisionTower = type("CLIPVisionTower", (nn.Module,), {})
    sys.modules[clip_encoder.__name__] = clip_encoder

    class FakeVQModel(nn.Module):
        pass

    vq_model = types.ModuleType(f"{package_name}.vq_model")
    vq_model.VQ_models = {"VQ-16": FakeVQModel}
    sys.modules[vq_model.__name__] = vq_model

    spec = importlib.util.spec_from_file_location(
        module_name,
        models_dir / "modeling_vlm.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


modeling_vlm = _load_modeling_vlm_module()
VisionConfig = modeling_vlm.VisionConfig
AlignerConfig = modeling_vlm.AlignerConfig
GenVisionConfig = modeling_vlm.GenVisionConfig
GenAlignerConfig = modeling_vlm.GenAlignerConfig
GenHeadConfig = modeling_vlm.GenHeadConfig
MultiModalityConfig = modeling_vlm.MultiModalityConfig


def test_named_params_config_round_trips_plain_nested_values():
    cfg = VisionConfig(
        cls="CLIPVisionTower",
        params=EasyDict(
            {
                "outer": EasyDict({"value": 3}),
                "entries": [EasyDict({"name": "token"})],
            }
        ),
    )

    dumped = cfg.to_dict()
    restored = VisionConfig(**dumped)

    assert isinstance(cfg.params, EasyDict)
    assert cfg.params.outer.value == 3
    assert type(dumped["params"]) is dict
    assert type(dumped["params"]["outer"]) is dict
    assert type(dumped["params"]["entries"][0]) is dict
    assert restored.cls == "CLIPVisionTower"
    assert isinstance(restored.params, EasyDict)
    assert restored.params.outer.value == 3
    assert restored.params.entries[0].name == "token"


def test_named_params_config_accepts_class_object_as_cls():
    cfg = GenHeadConfig(cls=list, params={"n_embed": 4})

    assert cfg.cls == "list"
    assert cfg.params == {"n_embed": 4}


def test_named_params_config_rejects_non_mapping_params():
    try:
        VisionConfig(params=["not", "a", "mapping"])
    except TypeError as exc:
        assert "params must be a mapping" in str(exc)
    else:
        raise AssertionError("VisionConfig accepted non-mapping params.")


def test_multi_modality_config_accepts_dict_instance_and_none_inputs():
    vision = VisionConfig(cls="CLIPVisionTower", params={"image_size": 224})
    gen_vision = {
        "cls": "VQ-16",
        "params": {"image_token_size": 8192, "n_embed": 1024},
    }

    cfg = MultiModalityConfig(
        vision_config=vision,
        aligner_config=None,
        gen_vision_config=gen_vision,
        gen_aligner_config=GenAlignerConfig(cls="MlpProjector"),
        gen_head_config=None,
        language_config=None,
    )

    assert cfg.vision_config is vision
    assert isinstance(cfg.aligner_config, AlignerConfig)
    assert isinstance(cfg.gen_vision_config, GenVisionConfig)
    assert isinstance(cfg.gen_aligner_config, GenAlignerConfig)
    assert isinstance(cfg.gen_head_config, GenHeadConfig)
    assert isinstance(cfg.language_config, LlamaConfig)
    assert cfg.gen_vision_config.params == {
        "image_token_size": 8192,
        "n_embed": 1024,
    }


def test_multi_modality_config_round_trips_sub_configs():
    cfg = MultiModalityConfig(
        vision_config=VisionConfig(cls="CLIPVisionTower", params={"a": 1}),
        aligner_config=AlignerConfig(cls="MlpProjector", params={"b": 2}),
        gen_vision_config=GenVisionConfig(cls="VQ-16", params={"c": 3}),
        gen_aligner_config=GenAlignerConfig(
            cls="MlpProjector",
            params={"d": 4},
        ),
        gen_head_config=GenHeadConfig(cls="vision_head", params={"e": 5}),
        language_config=LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
        ),
    )

    dumped = cfg.to_dict()
    restored = MultiModalityConfig(**dumped)

    assert restored.vision_config.params == {"a": 1}
    assert restored.aligner_config.params == {"b": 2}
    assert restored.gen_vision_config.params == {"c": 3}
    assert restored.gen_aligner_config.params == {"d": 4}
    assert restored.gen_head_config.params == {"e": 5}
    assert isinstance(restored.language_config, LlamaConfig)
