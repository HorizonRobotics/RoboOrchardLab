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


import torch
from torch import nn
from transformers import (
    AutoProcessor,
)

try:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = Qwen3VLConfig = None

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
    "HoloBrain_Qwen3VLTextTemplate",
    "HoloBrain_Qwen3VL",
    "HoloBrain_Qwen3VLConfig",
]


class HoloBrain_Qwen3VLTextTemplate(nn.Module):  # noqa: N801
    def __init__(self, with_subtask=True):
        super().__init__()
        self.with_subtask = with_subtask
        self.template = (
            "<|im_start|>user\n{image_token}"
            "You are a robot. {instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def forward(self, data):
        batch_size, num_cams = data["imgs"].shape[:2]
        image_token = [
            "<|vision_start|><|image_pad|><|vision_end|>"
        ] * num_cams
        image_token = "".join(image_token)
        instructions = data.get("text", [""] * batch_size)
        if self.with_subtask and "subtask" in data:
            for i, subtask in enumerate(data["subtask"]):
                if subtask is not None and len(subtask) > 0:
                    instructions[i] += f"Current task: {subtask}"
        text = [
            self.template.format(
                image_token=image_token, instruction=instruction
            )
            for instruction in instructions
        ]
        data["text"] = text
        return data


class HoloBrain_Qwen3VL(HoloBrain_Qwen2_5_VL):  # noqa: N801
    cfg: "HoloBrain_Qwen3VLConfig"

    def __init__(self, cfg: "HoloBrain_Qwen3VLConfig"):
        if Qwen3VLForConditionalGeneration is None or Qwen3VLConfig is None:
            raise ImportError(
                "Building `HoloBrain_Qwen3VL` requires `transformers>=4.57.1`."
            )
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

        vlm_pretrain = self.cfg.vlm_pretrain
        if self.cfg.load_vlm_checkpoint:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                vlm_pretrain,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            config = Qwen3VLConfig.from_pretrained(vlm_pretrain)
            self.vlm = Qwen3VLForConditionalGeneration._from_config(
                config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )

        if not hasattr(self.vlm, "language_model"):
            self.vlm.language_model = self.vlm.model.language_model
        if not hasattr(self.vlm, "visual"):
            self.vlm.visual = self.vlm.model.visual

        if self.cfg.freeze_vlm:
            self.vlm.eval()
            self.vlm.requires_grad_(False)
        else:
            if self.cfg.freeze_vision:
                self.vlm.visual.eval()
                self.vlm.visual.requires_grad_(False)

        origin_num_layers = len(self.vlm.language_model.layers)
        if self.cfg.num_vlm_layers is not None:
            assert self.cfg.num_vlm_layers > 0
            self.vlm.language_model.layers = self.vlm.language_model.layers[
                : self.cfg.num_vlm_layers
            ]
        num_layers = len(self.vlm.language_model.layers)

        if not self.cfg.freeze_vlm:
            self.vlm.language_model.norm.requires_grad_(False)
            self.vlm.language_model.layers[-1].requires_grad_(False)

        self.vlm_processor = AutoProcessor.from_pretrained(
            vlm_pretrain, use_fast=True
        )
        self.vlm_processor.tokenizer.padding_side = "left"

        hidden_size = (
            self.vlm.language_model.config.head_dim
            * self.vlm.language_model.config.num_key_value_heads
        )
        self.feat_mapping = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_size,
                    self.decoder.embed_dims,
                    bias=True,
                    dtype=torch.bfloat16,
                )
                for _ in range(num_layers)
            ]
        )
        temperature = 3
        highlighted_layer = origin_num_layers // 2  # the highlighted layer
        weight = torch.cat(
            [
                torch.linspace(0.1, 1, highlighted_layer),
                torch.linspace(1, 0.1, origin_num_layers - highlighted_layer),
            ]
        )[:num_layers]
        weight = weight.to(dtype=torch.bfloat16) * temperature
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.qwen_patch_size = 32
        self.with_cot = False

    def _forward_vlm(self, **vlm_inputs):
        vlm_outputs = self.vlm.model(**vlm_inputs)
        outputs = dict(
            hidden_states=[
                x.values.permute(0, 2, 1, 3).flatten(2)
                for x in vlm_outputs.past_key_values.layers
                if x.values is not None
            ]
        )
        return outputs

    def _generate_vlm(self, **vlm_inputs):
        raise NotImplementedError


MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class HoloBrain_Qwen3VLConfig(HoloBrain_Qwen2_5_VLConfig):  # noqa: N801
    class_type: ClassType_co[HoloBrain_Qwen3VL] = HoloBrain_Qwen3VL
