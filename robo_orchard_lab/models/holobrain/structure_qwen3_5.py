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


import os

import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
except ImportError as exc:
    Qwen3_5ForConditionalGeneration = Qwen3_5Config = None
    _QWEN3_5_IMPORT_ERROR = exc
else:
    _QWEN3_5_IMPORT_ERROR = None

from robo_orchard_lab.models.holobrain.structure import (
    HoloBrain_Qwen2_5_VL,
    HoloBrain_Qwen2_5_VLConfig,
)
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
    build,
)

__all__ = [
    "HoloBrain_Qwen3_5_VL",
    "HoloBrain_Qwen3_5_VLConfig",
]


def _get_patch_size(vision_config) -> int:
    patch_size = vision_config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[-1]
    return int(patch_size) * int(vision_config.spatial_merge_size)


class HoloBrain_Qwen3_5_VL(HoloBrain_Qwen2_5_VL):  # noqa: N801
    cfg: "HoloBrain_Qwen3_5_VLConfig"

    def __init__(self, cfg: "HoloBrain_Qwen3_5_VLConfig"):
        if Qwen3_5ForConditionalGeneration is None or Qwen3_5Config is None:
            raise ImportError(
                "Building `HoloBrain_Qwen3_5_VL` requires a transformers "
                "installation with Qwen3.5 support and a compatible PyTorch "
                "version."
            ) from _QWEN3_5_IMPORT_ERROR

        super(HoloBrain_Qwen2_5_VL, self).__init__(cfg)
        self.decoder = build(self.cfg.decoder)
        self.spatial_enhancer = build(self.cfg.spatial_enhancer)
        self.data_preprocessor = build(self.cfg.data_preprocessor)
        self.backbone_3d = build(self.cfg.backbone_3d)
        self.neck_3d = build(self.cfg.neck_3d)
        self.input_2d = self.cfg.input_2d
        self.input_3d = self.cfg.input_3d
        self.use_state_dict_with_vlm = self.cfg.use_state_dict_with_vlm
        if not self.use_state_dict_with_vlm:
            assert self.cfg.freeze_vlm, (
                "The VLM's state_dict must be saved when it is not frozen."
            )
        self.with_cot = self.cfg.with_cot

        vlm_pretrain = os.path.expanduser(self.cfg.vlm_pretrain)
        if self.cfg.load_vlm_checkpoint:
            self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
                vlm_pretrain,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            config = Qwen3_5Config.from_pretrained(vlm_pretrain)
            self.vlm = Qwen3_5ForConditionalGeneration._from_config(
                config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )

        language_model = self.vlm.model.language_model
        vision_model = self.vlm.model.visual

        if self.cfg.freeze_vlm:
            self.vlm.eval()
            self.vlm.requires_grad_(False)
        else:
            self.vlm.model.language_model.gradient_checkpointing_enable()
            if self.cfg.freeze_vision:
                vision_model.eval()
                vision_model.requires_grad_(False)

        origin_num_layers = len(language_model.layers)
        if (
            self.cfg.num_vlm_layers is not None
            and self.cfg.num_vlm_layers >= 0
        ):
            language_model.layers = language_model.layers[
                : self.cfg.num_vlm_layers
            ]
        num_layers = len(language_model.layers)

        if not self.cfg.freeze_vlm:
            language_model.norm.requires_grad_(False)
            if num_layers > 0:
                language_model.layers[-1].requires_grad_(False)

        self.vlm_processor = AutoProcessor.from_pretrained(
            vlm_pretrain, use_fast=True
        )
        self.vlm_processor.tokenizer.padding_side = "left"

        self.feat_mapping = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    language_model.config.hidden_size,
                    self.decoder.embed_dims,
                    bias=True,
                    dtype=torch.bfloat16,
                )
                for _ in range(num_layers + 1)
            ]
        )
        temperature = 3
        highlighted_layer = origin_num_layers // 2
        weight = torch.cat(
            [
                torch.linspace(0.1, 1, highlighted_layer + 1),
                torch.linspace(1, 0.1, origin_num_layers - highlighted_layer),
            ]
        )[: num_layers + 1]
        weight = weight.to(dtype=torch.bfloat16) * temperature
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.qwen_patch_size = _get_patch_size(self.vlm.config.vision_config)

    @torch.no_grad()
    def _generate_vlm(self, inputs):
        self.vlm.model.language_model.gradient_checkpointing_disable()
        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=256,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        # (ar_times, num_layers + 1, batch_size, seq, hidden_size)
        hidden_states = outputs.hidden_states
        # + 1 for the input embeddings
        num_hidden_states = len(self.vlm.model.language_model.layers) + 1
        cated_hidden_states = []
        for i in range(num_hidden_states):
            hs_i = torch.cat([h[i] for h in hidden_states], dim=1)
            cated_hidden_states.append(hs_i)
        cated_hidden_states = tuple(cated_hidden_states)
        outputs.hidden_states = cated_hidden_states
        return outputs


MODULE_TYPE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class HoloBrain_Qwen3_5_VLConfig(  # noqa: N801
    HoloBrain_Qwen2_5_VLConfig
):
    class_type: ClassType_co[HoloBrain_Qwen3_5_VL] = HoloBrain_Qwen3_5_VL
